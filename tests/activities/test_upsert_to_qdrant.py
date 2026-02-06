"""Tests for upsert_to_qdrant activity."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from workers.activities import upsert_to_qdrant
from shared.models.schemas import DocumentChunk
from datetime import datetime


@pytest.fixture
def mock_chunks():
    """Create mock document chunks with embeddings."""
    return [
        DocumentChunk(
            content="First chunk content",
            embedding=[0.1, 0.2, 0.3],
            chunk_index=0,
            filename="test.pdf",
            upload_timestamp=datetime.utcnow()
        ),
        DocumentChunk(
            content="Second chunk content",
            embedding=[0.4, 0.5, 0.6],
            chunk_index=1,
            filename="test.pdf",
            upload_timestamp=datetime.utcnow()
        )
    ]


@pytest.fixture
def mock_qdrant_client():
    """Create a mock Qdrant client."""
    mock_client = MagicMock()
    mock_client.ensure_collection = AsyncMock()
    mock_client.upsert_vectors = AsyncMock()
    return mock_client


@pytest.mark.asyncio
async def test_upsert_to_qdrant_success(mock_chunks, mock_qdrant_client):
    """Test successful vector upsert to Qdrant."""
    document_id = "doc-123"

    await upsert_to_qdrant(
        document_id=document_id,
        chunks=mock_chunks,
        qdrant_client=mock_qdrant_client
    )

    mock_qdrant_client.ensure_collection.assert_called_once()
    mock_qdrant_client.upsert_vectors.assert_called_once()


@pytest.mark.asyncio
async def test_upsert_to_qdrant_empty_chunks(mock_qdrant_client):
    """Test upsert with empty chunks - should return early."""
    document_id = "doc-empty"

    await upsert_to_qdrant(
        document_id=document_id,
        chunks=[],
        qdrant_client=mock_qdrant_client
    )

    # Should not call Qdrant methods for empty chunks
    mock_qdrant_client.ensure_collection.assert_not_called()
    mock_qdrant_client.upsert_vectors.assert_not_called()


@pytest.mark.asyncio
async def test_upsert_to_qdrant_ensure_collection_error(mock_chunks):
    """Test upsert fails when collection creation fails."""
    document_id = "doc-123"

    mock_client = MagicMock()
    mock_client.ensure_collection = AsyncMock(
        side_effect=Exception("Qdrant collection creation failed")
    )

    with pytest.raises(Exception, match="Qdrant collection creation failed"):
        await upsert_to_qdrant(
            document_id=document_id,
            chunks=mock_chunks,
            qdrant_client=mock_client
        )


@pytest.mark.asyncio
async def test_upsert_to_qdrant_upsert_error(mock_chunks):
    """Test upsert fails when vector upsert fails."""
    document_id = "doc-123"

    mock_client = MagicMock()
    mock_client.ensure_collection = AsyncMock()
    mock_client.upsert_vectors = AsyncMock(
        side_effect=Exception("Qdrant upsert failed")
    )

    with pytest.raises(Exception, match="Qdrant upsert failed"):
        await upsert_to_qdrant(
            document_id=document_id,
            chunks=mock_chunks,
            qdrant_client=mock_client
        )


@pytest.mark.asyncio
async def test_upsert_to_qdrant_single_chunk():
    """Test upsert with single chunk."""
    document_id = "doc-single"

    chunks = [
        DocumentChunk(
            content="Only chunk",
            embedding=[0.1, 0.2, 0.3],
            chunk_index=0,
            filename="test.txt",
            upload_timestamp=datetime.utcnow()
        )
    ]

    mock_client = MagicMock()
    mock_client.ensure_collection = AsyncMock()
    mock_client.upsert_vectors = AsyncMock()

    await upsert_to_qdrant(
        document_id=document_id,
        chunks=chunks,
        qdrant_client=mock_client
    )

    mock_client.ensure_collection.assert_called_once()
    mock_client.upsert_vectors.assert_called_once()
