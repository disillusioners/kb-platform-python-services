"""Unit tests for upsert_to_qdrant activity."""

import pytest
from unittest.mock import MagicMock, AsyncMock
from datetime import datetime
from workers.activities import upsert_to_qdrant
from shared.models.schemas import DocumentChunk


@pytest.fixture
def mock_qdrant_client(mocker):
    """Create a mock Qdrant client."""
    mock = mocker.MagicMock()
    mock.ensure_collection = AsyncMock()
    mock.upsert_vectors = AsyncMock()
    return mock


@pytest.fixture
def sample_chunks():
    """Create sample document chunks with embeddings."""
    return [
        DocumentChunk(
            content="First chunk of text",
            embedding=[0.1, 0.2, 0.3],
            chunk_index=0,
            filename="test.pdf",
            upload_timestamp=datetime.utcnow()
        ),
        DocumentChunk(
            content="Second chunk of text",
            embedding=[0.4, 0.5, 0.6],
            chunk_index=1,
            filename="test.pdf",
            upload_timestamp=datetime.utcnow()
        )
    ]


@pytest.mark.asyncio
async def test_upsert_success(mock_qdrant_client, sample_chunks):
    """Test successful vector upsert to Qdrant."""
    result = await upsert_to_qdrant(
        "test-doc-id",
        sample_chunks,
        qdrant_client=mock_qdrant_client
    )
    
    # Verify collection was ensured
    mock_qdrant_client.ensure_collection.assert_called_once()
    
    # Verify vectors were upserted
    mock_qdrant_client.upsert_vectors.assert_called_once()
    
    # Get the points that were upserted
    call_args = mock_qdrant_client.upsert_vectors.call_args
    points = call_args[0][0]  # First positional argument
    
    # Verify point structure
    assert len(points) == 2
    assert all(hasattr(point, 'vector') for point in points)
    assert all(hasattr(point, 'payload') for point in points)
    assert all(hasattr(point, 'id') for point in points)


@pytest.mark.asyncio
async def test_upsert_empty_chunks(mock_qdrant_client):
    """Test handling of empty chunk list."""
    result = await upsert_to_qdrant(
        "test-doc-id",
        [],
        qdrant_client=mock_qdrant_client
    )
    
    # Should not call upsert_vectors with empty list
    mock_qdrant_client.upsert_vectors.assert_not_called()


@pytest.mark.asyncio
async def test_upsert_single_chunk(mock_qdrant_client):
    """Test upsert with single chunk."""
    single_chunk = [
        DocumentChunk(
            content="Single chunk",
            embedding=[0.1, 0.2, 0.3],
            chunk_index=0,
            filename="test.pdf",
            upload_timestamp=datetime.utcnow()
        )
    ]
    
    result = await upsert_to_qdrant(
        "test-doc-id",
        single_chunk,
        qdrant_client=mock_qdrant_client
    )
    
    # Verify single point was upserted
    mock_qdrant_client.upsert_vectors.assert_called_once()
    points = mock_qdrant_client.upsert_vectors.call_args[0][0]
    assert len(points) == 1


@pytest.mark.asyncio
async def test_upsert_payload_structure(mock_qdrant_client, sample_chunks):
    """Test that payload contains correct document metadata."""
    result = await upsert_to_qdrant(
        "test-doc-id",
        sample_chunks,
        qdrant_client=mock_qdrant_client
    )
    
    # Verify payload structure
    points = mock_qdrant_client.upsert_vectors.call_args[0][0]
    first_point = points[0]
    
    # Verify payload fields
    assert first_point.payload["document_id"] == "test-doc-id"
    assert first_point.payload["content"] == "First chunk of text"
    assert first_point.payload["chunk_index"] == 0
    assert first_point.payload["filename"] == "test.pdf"
    assert "upload_timestamp" in first_point.payload


@pytest.mark.asyncio
async def test_qdrant_connection_error(mock_qdrant_client, sample_chunks):
    """Test exception propagation on Qdrant connection error."""
    # Setup mock to raise exception
    mock_qdrant_client.upsert_vectors.side_effect = Exception(
        "Connection refused"
    )
    
    # Call should propagate exception
    with pytest.raises(Exception) as exc_info:
        await upsert_to_qdrant(
            "test-doc-id",
            sample_chunks,
            qdrant_client=mock_qdrant_client
        )
    
    assert "Connection refused" in str(exc_info.value)
