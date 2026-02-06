"""Tests for generate_embeddings activity."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from workers.activities import generate_embeddings
from shared.models.schemas import DocumentChunk
from datetime import datetime


class MockEmbeddingData:
    """Mock embedding API response data."""
    def __init__(self, embedding):
        self.embedding = embedding


class MockEmbeddingResponse:
    """Mock OpenAI embeddings response."""
    def __init__(self, embeddings):
        self.data = [MockEmbeddingData(e) for e in embeddings]


@pytest.fixture
def mock_chunks():
    """Create mock document chunks."""
    return [
        DocumentChunk(
            content="First chunk content",
            embedding=[],
            chunk_index=0,
            filename="test.pdf",
            upload_timestamp=datetime.utcnow()
        ),
        DocumentChunk(
            content="Second chunk content",
            embedding=[],
            chunk_index=1,
            filename="test.pdf",
            upload_timestamp=datetime.utcnow()
        )
    ]


@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client."""
    mock_client = MagicMock()
    embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    mock_response = MockEmbeddingResponse(embeddings)
    mock_client.embeddings.create = AsyncMock(return_value=mock_response)
    return mock_client


@pytest.mark.asyncio
async def test_generate_embeddings_success(mock_chunks, mock_openai_client):
    """Test successful embedding generation."""
    result = await generate_embeddings(
        chunks=mock_chunks,
        openai_client=mock_openai_client
    )

    assert len(result) == 2
    assert result[0].embedding == [0.1, 0.2, 0.3]
    assert result[1].embedding == [0.4, 0.5, 0.6]
    mock_openai_client.embeddings.create.assert_called_once()


@pytest.mark.asyncio
async def test_generate_embeddings_empty_chunks():
    """Test embedding generation with empty chunks list."""
    mock_client = MagicMock()

    result = await generate_embeddings(
        chunks=[],
        openai_client=mock_client
    )

    assert result == []
    mock_client.embeddings.create.assert_not_called()


@pytest.mark.asyncio
async def test_generate_embeddings_api_error(mock_chunks):
    """Test embedding generation fails on API error."""
    mock_client = MagicMock()
    mock_client.embeddings.create = AsyncMock(
        side_effect=Exception("OpenAI API rate limit")
    )

    with pytest.raises(Exception, match="OpenAI API rate limit"):
        await generate_embeddings(
            chunks=mock_chunks,
            openai_client=mock_client
        )


@pytest.mark.asyncio
async def test_generate_embeddings_single_chunk():
    """Test embedding generation with single chunk."""
    chunks = [
        DocumentChunk(
            content="Single chunk",
            embedding=[],
            chunk_index=0,
            filename="test.txt",
            upload_timestamp=datetime.utcnow()
        )
    ]

    mock_client = MagicMock()
    mock_response = MockEmbeddingResponse([[0.1, 0.2, 0.3]])
    mock_client.embeddings.create = AsyncMock(return_value=mock_response)

    result = await generate_embeddings(
        chunks=chunks,
        openai_client=mock_client
    )

    assert len(result) == 1
    assert result[0].embedding == [0.1, 0.2, 0.3]
