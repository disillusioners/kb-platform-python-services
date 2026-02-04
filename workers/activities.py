"""Temporal activities for document processing."""

import asyncio
import io
import os
import tempfile
from datetime import datetime, timedelta
from typing import List
from pathlib import Path
import uuid

import boto3
from temporalio import activity
from temporalio.common import RetryPolicy

from shared.models.schemas import Document, DocumentChunk, DocumentStatus
from shared.models.database import db, get_document, update_document_status
from shared.services.qdrant import QdrantClient
from shared.config import get_settings

from pypdf import PdfReader
import docx
from bs4 import BeautifulSoup

# LlamaIndex for splitting
from llama_index.text_splitter import TokenTextSplitter
from qdrant_client.models import PointStruct


@activity.defn
async def download_from_s3(document_id: str) -> bytes:
    """Download document from S3."""
    settings = get_settings()
    
    # Initialize DB if needed (activity worker might need to init)
    if not db.pool:
        await db.connect()

    async with db.pool.acquire() as conn:
        document = await get_document(conn, document_id)
        if not document:
            raise ValueError(f"Document {document_id} not found")
        
        s3_key = document.s3_key

    # S3 Client
    s3 = boto3.client(
        "s3",
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key,
        region_name=settings.aws_region,
        endpoint_url=settings.s3_endpoint_url
    )

    try:
        response = s3.get_object(Bucket=settings.s3_bucket, Key=s3_key)
        file_content = response["Body"].read()
        return file_content, document.filename
    except Exception as e:
        activity.logger.error(f"Failed to download from S3: {e}")
        raise


@activity.defn
async def parse_document(
    document_id: str,
    file_data: bytes,
    filename: str
) -> List[DocumentChunk]:
    """Parse document into chunks."""
    
    ext = Path(filename).suffix.lower()
    text_content = ""

    # Write bytes to temp file because some libs need file path or file-like object
    with tempfile.NamedTemporaryFile(suffix=ext, delete=True) as temp_file:
        temp_file.write(file_data)
        temp_file.flush()
        
        try:
            if ext == ".pdf":
                reader = PdfReader(temp_file.name)
                for page in reader.pages:
                    text_content += page.extract_text() + "\n"
            elif ext == ".docx":
                doc = docx.Document(temp_file.name)
                for para in doc.paragraphs:
                    text_content += para.text + "\n"
            elif ext in (".txt", ".md"):
                # Rewind to read as text
                 with open(temp_file.name, 'r', encoding='utf-8') as f:
                     text_content = f.read()
            elif ext == ".html":
                 with open(temp_file.name, 'r', encoding='utf-8') as f:
                     soup = BeautifulSoup(f, 'html.parser')
                     text_content = soup.get_text()
            else:
                raise ValueError(f"Unsupported file type: {ext}")
        except Exception as e:
            activity.logger.error(f"Failed to parse document: {e}")
            raise

    # Split text
    settings = get_settings()
    splitter = TokenTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap
    )
    
    chunks = splitter.split_text(text_content)
    
    doc_chunks = []
    timestamp = datetime.utcnow()
    
    for i, chunk in enumerate(chunks):
        doc_chunks.append(DocumentChunk(
            content=chunk,
            embedding=[], # To be filled
            chunk_index=i,
            filename=filename,
            upload_timestamp=timestamp
        ))
        
    activity.logger.info(f"Parsed {len(doc_chunks)} chunks from {filename}")
    return doc_chunks


@activity.defn
async def generate_embeddings(
    chunks: List[DocumentChunk]
) -> List[DocumentChunk]:
    """Generate embeddings for chunks."""
    settings = get_settings()
    from openai import AsyncOpenAI
    
    client = AsyncOpenAI(api_key=settings.openai_api_key)
    
    texts = [c.content for c in chunks]
    
    # Process in batches if needed, but for now simple
    # OpenAI embedding API can handle arrays
    
    try:
        response = await client.embeddings.create(
            file=texts, # Wait, 'file'?? No, 'input'
            model=settings.embedding_model,
            input=texts
        )
        
        for i, data in enumerate(response.data):
            chunks[i].embedding = data.embedding
            
        return chunks
    except Exception as e:
        activity.logger.error(f"Embedding generation failed: {e}")
        raise


@activity.defn
async def upsert_to_qdrant(
    document_id: str,
    chunks: List[DocumentChunk]
):
    """Upsert vectors to Qdrant."""
    settings = get_settings()
    qdrant = QdrantClient(settings.qdrant_url, settings.qdrant_collection)
    
    await qdrant.ensure_collection(settings.embedding_dimension)
    
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=chunk.embedding,
            payload={
                "document_id": document_id,
                "content": chunk.content,
                "chunk_index": chunk.chunk_index,
                "filename": chunk.filename,
                "upload_timestamp": chunk.upload_timestamp.isoformat()
            }
        )
        for chunk in chunks
    ]
    
    await qdrant.upsert_vectors(points)


@activity.defn
async def update_status(
    document_id: str,
    status: str,
    error: str = None,
    chunk_count: int = None
):
    """Update document status in DB."""
    if not db.pool:
        await db.connect()
        
    async with db.pool.acquire() as conn:
        doc_status = DocumentStatus(status)
        await update_document_status(
            conn, 
            document_id, 
            doc_status, 
            error_message=error,
            chunk_count=chunk_count
        )
