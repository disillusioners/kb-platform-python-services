"""FastAPI application for KB platform core service."""

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
from typing import AsyncIterator
import json

from .config import get_settings
from .models.database import init_db, get_document, create_conversation, save_message
from .models.schemas import (
    QueryRequest, Document, Conversation, Message,
    SSEEvent, HealthResponse, ReadinessResponse
)
from .services.rag import RAGEngine
from .services.qdrant import QdrantClient

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    await init_db()
    yield


app = FastAPI(
    title="KB Platform Core API",
    description="Python LlamaIndex Core Service",
    version="1.0.0",
    lifespan=lifespan,
)


qdrant_client = QdrantClient(settings.qdrant_url, settings.qdrant_collection)
rag_engine = RAGEngine(
    qdrant_client,
    settings.openai_api_key,
    settings.llm_model,
    settings.llm_temperature,
)


@app.get("/healthz", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=__import__("datetime").datetime.utcnow().isoformat()
    )


@app.get("/readyz", response_model=ReadinessResponse)
async def readiness_check():
    """Readiness check endpoint."""
    dependencies = {
        "database": "ok",
        "qdrant": "ok",
        "openai": "ok",
    }
    return ReadinessResponse(
        status="ready",
        dependencies=dependencies
    )


@app.post("/api/v1/query")
async def query(request: QueryRequest):
    """Query endpoint with SSE streaming."""
    async def generate_response() -> AsyncIterator[str]:
        query_id = str(__import__("uuid").uuid4())

        yield f"data: {json.dumps(SSEEvent(type='start', id=query_id).model_dump())}\n\n"

        try:
            async for chunk in rag_engine.query(
                request.query,
                request.conversation_id,
                request.top_k
            ):
                yield f"data: {json.dumps(SSEEvent(type='chunk', content=chunk).model_dump())}\n\n"

            yield f"data: {json.dumps(SSEEvent(type='end', id=query_id).model_dump())}\n\n"

        except Exception as e:
            yield f"data: {json.dumps(SSEEvent(type='error', code='INTERNAL_ERROR', message=str(e)).model_dump())}\n\n"

    return StreamingResponse(
        generate_response(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.get("/api/v1/documents/{document_id}", response_model=Document)
async def get_document_endpoint(document_id: str):
    """Get document by ID."""
    async for conn in get_db():
        doc = await get_document(conn, document_id)
        if doc is None:
            raise HTTPException(status_code=404, detail="Document not found")
        return doc


@app.delete("/api/v1/documents/{document_id}/vectors")
async def delete_document_vectors(document_id: str):
    """Delete document vectors from Qdrant."""
    await qdrant_client.delete_document(document_id)
    return None


@app.get("/api/v1/conversations/{conversation_id}", response_model=Conversation)
async def get_conversation_endpoint(conversation_id: str):
    """Get conversation by ID."""
    async for conn in get_db():
        conv = await get_document(conn, conversation_id)
        if conv is None:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return conv


def main():
    """Main entry point."""
    import uvicorn
    uvicorn.run(
        "core.api.main:app",
        host=settings.server_host,
        port=settings.server_port,
        reload=True,
    )


if __name__ == "__main__":
    main()
