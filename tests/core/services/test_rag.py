"""Unit tests for RAG Engine."""

import pytest
from unittest.mock import MagicMock, AsyncMock
from core.services.rag import RAGEngine

@pytest.fixture
def mock_qdrant(mocker):
    """Mock Qdrant wrapper."""
    return mocker.AsyncMock()

@pytest.fixture
def mock_openai(mocker):
    """Mock OpenAI client."""
    mock = mocker.AsyncMock()
    # Mock embeddings response
    mock_embedding_resp = MagicMock()
    mock_embedding_resp.data = [MagicMock(embedding=[0.1, 0.2])]
    mock.embeddings.create.return_value = mock_embedding_resp
    return mock

@pytest.fixture
def rag_engine(mock_qdrant, mock_openai):
    """Create RAG Engine instance."""
    engine = RAGEngine(
        qdrant_client=mock_qdrant,
        openai_api_key="test-key",
        embedding_model="test-embed",
        llm_model="test-model"
    )
    # Patch the internal client created in __init__
    engine.client = mock_openai
    return engine

@pytest.mark.asyncio
async def test_retrieve_relevant_chunks(rag_engine, mock_qdrant, mock_openai):
    """Test chunk retrieval logic."""
    # Setup Qdrant search results
    mock_qdrant.search.return_value = [
        {"payload": {"content": "Chunk 1"}},
        {"payload": {"content": "Chunk 2"}},
        {"payload": {}} # Missing content key (should be ignored)
    ]

    chunks = await rag_engine._retrieve_relevant_chunks("test query", top_k=3)

    # Verify embeddings call
    mock_openai.embeddings.create.assert_called_once()
    assert mock_openai.embeddings.create.call_args.kwargs["input"] == "test query"

    # Verify Qdrant search
    mock_qdrant.search.assert_called_once()
    assert mock_qdrant.search.call_args.kwargs["limit"] == 3
    assert mock_qdrant.search.call_args.kwargs["query_vector"] == [0.1, 0.2]

    # Verify result filtering
    assert len(chunks) == 2
    assert chunks == ["Chunk 1", "Chunk 2"]

def test_build_rag_prompt_basic(rag_engine):
    """Test prompt construction with context."""
    chunks = ["Info A", "Info B"]
    history = []
    
    prompt = rag_engine._build_rag_prompt("My Query", chunks, history)

    assert "You are a helpful assistant" in prompt
    assert "Context from documents:" in prompt
    assert "Info A" in prompt
    assert "Info B" in prompt
    assert "User: My Query" in prompt

def test_build_rag_prompt_empty_context(rag_engine):
    """Test prompt construction with empty context."""
    prompt = rag_engine._build_rag_prompt("My Query", [], [])
    
    assert "No relevant context found" in prompt

def test_build_rag_prompt_with_history(rag_engine):
    """Test prompt construction with history."""
    history = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"}
    ]
    
    prompt = rag_engine._build_rag_prompt("Query", [], history)
    
    assert "Conversation history:" in prompt
    assert "user: Hi" in prompt
    assert "assistant: Hello" in prompt

@pytest.mark.asyncio
async def test_stream_response_success(rag_engine, mock_openai):
    """Test successful streaming response."""
    # Setup mock stream chunks
    chunk1 = MagicMock()
    chunk1.choices = [MagicMock()]
    chunk1.choices[0].delta.content = "Hello "

    chunk2 = MagicMock()
    chunk2.choices = [MagicMock()]
    chunk2.choices[0].delta.content = "World"

    # Setup async generator mock
    async def mock_stream(*args, **kwargs):
        yield chunk1
        yield chunk2

    mock_openai.chat.completions.create.side_effect = mock_stream

    # Collect yielded chunks
    result_chunks = []
    async for chunk in rag_engine._stream_response(
        query="test",
        context_chunks=["context"],
        history=[]
    ):
        result_chunks.append(chunk)

    assert result_chunks == ["Hello ", "World"]
    
    # Verify LLM call
    mock_openai.chat.completions.create.assert_called_once()
    kwargs = mock_openai.chat.completions.create.call_args.kwargs
    assert kwargs["stream"] is True
    assert kwargs["model"] == "test-model"
    assert len(kwargs["messages"]) == 1
    assert kwargs["messages"][0]["role"] == "user"

@pytest.mark.asyncio
async def test_stream_response_error(rag_engine, mock_openai):
    """Test handling of LLM API error."""
    mock_openai.chat.completions.create.side_effect = Exception("API Error")

    with pytest.raises(Exception) as exc:
        async for _ in rag_engine._stream_response("q", [], []):
            pass
    
    assert "LLM generation failed" in str(exc.value)

@pytest.mark.asyncio
async def test_full_query_flow(rag_engine, mock_qdrant, mock_openai):
    """Test the public query method (integration of retrieval + generation)."""
    # Mock retrieval results
    mock_qdrant.search.return_value = [{"payload": {"content": "ctx"}}]
    
    # Mock streaming response
    chunk = MagicMock()
    chunk.choices = [MagicMock()]
    chunk.choices[0].delta.content = "Answer"
    
    async def mock_stream(*args, **kwargs):
        yield chunk
    
    mock_openai.chat.completions.create.side_effect = mock_stream

    # Execute
    result = []
    async for item in rag_engine.query("User ask", "conv-id"):
        result.append(item)
    
    # Verify flow
    assert result == ["Answer"]
    mock_openai.embeddings.create.assert_called_once()
    mock_qdrant.search.assert_called_once()
    mock_openai.chat.completions.create.assert_called_once()
