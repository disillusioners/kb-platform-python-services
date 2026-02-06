"""Unit tests for generate_embeddings activity."""

import pytest
from unittest.mock import MagicMock, AsyncMock
from datetime import datetime
from workers.activities import generate_embeddings
from shared.models.schemas import DocumentChunk


@pytest.fixture
def mock_openai_client(mocker):
    """Create a mock OpenAI client."""
    mock = mocker.MagicMock()
    
    # Setup mock embeddings response
    mock_response = MagicMock()
    mock_response.data = [
        MagicMock(embedding=[0.1, 0.2, 0.3]),
        MagicMock(embedding=[0.4, 0.5, 0.6]),
        MagicMock(embedding=[0.7, 0.8, 0.9])
    ]
    mock.embeddings.create = AsyncMock(return_value=mock_response)
    
    return mock


@pytest.fixture
def sample_chunks():
    """Create sample document chunks."""
    return [
        DocumentChunk(
            content="First chunk of text content",
            embedding=[],
            chunk_index=0,
            filename="test.pdf",
            upload_timestamp=datetime.utcnow()
        ),
        DocumentChunk(
            content="Second chunk of text content",
            embedding=[],
            chunk_index=1,
            filename="test.pdf",
            upload_timestamp=datetime.utcnow()
        ),
        DocumentChunk(
            content="Third chunk of text content",
            embedding=[],
            chunk_index=2,
            filename="test.pdf",
            upload_timestamp=datetime.utcnow()
        )
    ]


@pytest.mark.asyncio
async def test_generate_embeddings_success(mock_openai_client, sample_chunks):
    """Test successful embedding generation."""
    result = await generate_embeddings(
        sample_chunks,
        openai_client=mock_openai_client
    )
    
    # Verify embeddings were generated
    assert len(result) == 3
    assert result[0].embedding == [0.1, 0.2, 0.3]
    assert result[1].embedding == [0.4, 0.5, 0.6]
    assert result[2].embedding == [0.7, 0.8, 0.9]
    
    # Verify original content preserved
    assert result[0].content == "First chunk of text content"
    assert result[0].chunk_index == 0
    assert result[0].filename == "test.pdf"


@pytest.mark.asyncio
async def test_empty_chunks(mock_openai_client):
    """Test handling of empty chunk list."""
    result = await generate_embeddings(
        [],
        openai_client=mock_openai_client
    )
    
    # Verify empty result
    assert result == []
    # Verify OpenAI API was not called
    mock_openai_client.embeddings.create.assert_not_called()


@pytest.mark.asyncio
async def test_single_chunk(mock_openai_client):
    """Test embedding generation for single chunk."""
    single_chunk = [
        DocumentChunk(
            content="Single chunk",
            embedding=[],
            chunk_index=0,
            filename="test.pdf",
            upload_timestamp=datetime.utcnow()
        )
    ]
    
    # Setup mock response for single chunk
    mock_response = MagicMock()
    mock_response.data = [
        MagicMock(embedding=[0.1, 0.2, 0.3])
    ]
    mock_openai_client.embeddings.create = AsyncMock(return_value=mock_response)
    
    result = await generate_embeddings(
        single_chunk,
        openai_client=mock_openai_client
    )
    
    # Verify single embedding
    assert len(result) == 1
    assert result[0].embedding == [0.1, 0.2, 0.3]


@pytest.mark.asyncio
async def test_openai_api_error(mock_openai_client, sample_chunks):
    """Test exception propagation on OpenAI API error."""
    # Setup mock to raise exception
    mock_openai_client.embeddings.create.side_effect = Exception(
        "OpenAI API rate limit exceeded"
    )
    
    # Call should propagate exception
    with pytest.raises(Exception) as exc_info:
        await generate_embeddings(
            sample_chunks,
            openai_client=mock_openai_client
        )
    
    assert "OpenAI API rate limit exceeded" in str(exc_info.value)


@pytest.mark.asyncio
async def test_embedding_api_called_correctly(mock_openai_client, sample_chunks):
    """Test that OpenAI API is called with correct parameters."""
    await generate_embeddings(
        sample_chunks,
        openai_client=mock_openai_client
    )
    
    # Verify API was called
    mock_openai_client.embeddings.create.assert_called_once()
    
    # Verify call parameters
    call_kwargs = mock_openai_client.embeddings.create.call_args.kwargs
    assert "model" in call_kwargs
    assert "input" in call_kwargs
    assert len(call_kwargs["input"]) == 3
