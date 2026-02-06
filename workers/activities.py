"""Temporal activities for document processing."""

import asyncio
import io
import os
import tempfile
from datetime import datetime, timedelta
from typing import List, Optional, Any, Tuple
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
try:
    from llama_index.text_splitter import TokenTextSplitter
except ImportError:
    from llama_index.core.node_parser import TextSplitter as TokenTextSplitter
from qdrant_client.models import PointStruct


def _create_s3_client(settings) -> Any:
    """Create S3 client. Extracted for testability."""
    return boto3.client(
        "s3",
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key,
        region_name=settings.aws_region,
        endpoint_url=settings.s3_endpoint_url
    )


@activity.defn
async def download_from_s3(
    document_id: str,
    s3_client: Optional[Any] = None,
    db_pool: Optional[Any] = None
) -> Tuple[bytes, str]:
    """Download document from S3."""
    settings = get_settings()

    # Use injected dependencies or create default
    if s3_client is None:
        s3_client = _create_s3_client(settings)
    pool = db_pool if db_pool is not None else db.pool
    if pool is None:
        await db.connect()
        pool = db.pool

    async with pool.acquire() as conn:
        document = await get_document(conn, document_id)
        if not document:
            raise ValueError(f"Document {document_id} not found")

        s3_key = document.s3_key

    try:
        response = s3_client.get_object(Bucket=settings.s3_bucket, Key=s3_key)
        file_content = response["Body"].read()
        return file_content, document.filename
    except Exception as e:
        activity.logger.error(f"Failed to download from S3: {e}")
        raise


@activity.defn
async def delete_from_s3(
    s3_key: str,
    s3_client: Optional[Any] = None
) -> str:
    """Delete document from S3 for cleanup."""
    settings = get_settings()

    # Use injected S3 client or create default
    if s3_client is None:
        s3_client = _create_s3_client(settings)

    try:
        s3_client.delete_object(Bucket=settings.s3_bucket, Key=s3_key)
        activity.logger.info(f"Deleted file from S3: {s3_key}")
        return f"Deleted {s3_key}"
    except Exception as e:
        activity.logger.error(f"Failed to delete from S3: {e}")
        # Don't fail the workflow if cleanup fails, just log it
        return f"Failed to delete {s3_key}: {e}"


def _parse_pdf(temp_file_path: str) -> str:
    """Parse PDF file. Extracted for testability."""
    reader = PdfReader(temp_file_path)
    text_content = ""
    for page in reader.pages:
        text_content += page.extract_text() + "\n"
    return text_content


def _parse_docx(temp_file_path: str) -> str:
    """Parse DOCX file. Extracted for testability."""
    doc = docx.Document(temp_file_path)
    text_content = ""
    for para in doc.paragraphs:
        text_content += para.text + "\n"
    return text_content


def _parse_html(temp_file_path: str) -> str:
    """Parse HTML file. Extracted for testability."""
    with open(temp_file_path, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f, 'html.parser')
        return soup.get_text()


def _parse_text(temp_file_path: str) -> str:
    """Parse plain text file. Extracted for testability."""
    with open(temp_file_path, 'r', encoding='utf-8') as f:
        return f.read()


@activity.defn
async def parse_document(
    document_id: str,
    file_data: bytes,
    filename: str,
    file_parser: Optional[Any] = None,
    text_splitter: Optional[Any] = None
) -> List[DocumentChunk]:
    """Parse document into chunks."""
    
    ext = Path(filename).suffix.lower()
    text_content = ""

    # Use injected file parser or use default logic
    if file_parser is None:
        # Write bytes to temp file because some libs need file path or file-like object
        with tempfile.NamedTemporaryFile(suffix=ext, delete=True) as temp_file:
            temp_file.write(file_data)
            temp_file.flush()
            
            try:
                if ext == ".pdf":
                    text_content = _parse_pdf(temp_file.name)
                elif ext == ".docx":
                    text_content = _parse_docx(temp_file.name)
                elif ext in (".txt", ".md"):
                    text_content = _parse_text(temp_file.name)
                elif ext == ".html":
                    text_content = _parse_html(temp_file.name)
                else:
                    raise ValueError(f"Unsupported file type: {ext}")
            except Exception as e:
                activity.logger.error(f"Failed to parse document: {e}")
                raise
    else:
        # Use injected parser (for testing)
        text_content = file_parser(file_data, ext)

    # Split text - use injected splitter or create default
    if text_splitter is None:
        settings = get_settings()
        text_splitter = TokenTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap
        )
    
    chunks = text_splitter.split_text(text_content)
    
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
    chunks: List[DocumentChunk],
    openai_client: Optional[Any] = None
) -> List[DocumentChunk]:
    """Generate embeddings for chunks."""
    settings = get_settings()
    
    # Use injected client or create default
    if openai_client is None:
        from openai import AsyncOpenAI
        openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
    
    texts = [c.content for c in chunks]
    
    # Process in batches if needed, but for now simple
    # OpenAI embedding API can handle arrays
    
    try:
        response = await openai_client.embeddings.create(
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
    chunks: List[DocumentChunk],
    qdrant_client: Optional[QdrantClient] = None
):
    """Upsert vectors to Qdrant."""
    settings = get_settings()
    
    # Use injected Qdrant client or create default
    if qdrant_client is None:
        qdrant_client = QdrantClient(settings.qdrant_url, settings.qdrant_collection)
    
    await qdrant_client.ensure_collection(settings.embedding_dimension)
    
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
    
    await qdrant_client.upsert_vectors(points)


@activity.defn
async def update_status(
    document_id: str,
    status: str,
    error: Optional[str] = None,
    chunk_count: Optional[int] = None,
    db_pool: Optional[Any] = None
):
    """Update document status in DB."""
    # Use injected pool or get from db module
    if db_pool is None:
        if not db.pool:
            await db.connect()
        db_pool = db.pool
        
    async with db_pool.acquire() as conn:
        doc_status = DocumentStatus(status)
        await update_document_status(
            conn, 
            document_id, 
            doc_status, 
            error_message=error,
            chunk_count=chunk_count
        )
