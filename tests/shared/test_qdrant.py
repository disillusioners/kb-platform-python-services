"""Unit tests for Qdrant client wrapper."""

import pytest
from unittest.mock import MagicMock, AsyncMock
from shared.services.qdrant import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

@pytest.fixture
def mock_qdrant_sdk(mocker):
    """Mock the internal QdrantSDK."""
    return mocker.MagicMock()

@pytest.fixture
def qdrant_client(mock_qdrant_sdk):
    """Create a QdrantClient instance with mocked SDK."""
    client = QdrantClient("http://localhost:6333", "test_collection")
    client.client = mock_qdrant_sdk
    return client

@pytest.mark.asyncio
async def test_ensure_collection_exists(qdrant_client, mock_qdrant_sdk):
    """Test ensure_collection when collection already exists."""
    # Setup mock to return success (no exception)
    mock_qdrant_sdk.get_collection.return_value = MagicMock()

    await qdrant_client.ensure_collection()

    # Verify get_collection called
    mock_qdrant_sdk.get_collection.assert_called_once_with("test_collection")
    # Verify create_collection NOT called
    mock_qdrant_sdk.create_collection.assert_not_called()

@pytest.mark.asyncio
async def test_ensure_collection_creates_new(qdrant_client, mock_qdrant_sdk):
    """Test ensure_collection creates collection if missing."""
    # Setup mock to raise exception (collection missing)
    mock_qdrant_sdk.get_collection.side_effect = Exception("Not found")

    await qdrant_client.ensure_collection(vector_size=128)

    # Verify create_collection called with correct params
    mock_qdrant_sdk.create_collection.assert_called_once()
    call_kwargs = mock_qdrant_sdk.create_collection.call_args.kwargs
    assert call_kwargs["collection_name"] == "test_collection"
    assert isinstance(call_kwargs["vectors_config"], VectorParams)
    assert call_kwargs["vectors_config"].size == 128
    assert call_kwargs["vectors_config"].distance == Distance.COSINE

@pytest.mark.asyncio
async def test_upsert_vectors(qdrant_client, mock_qdrant_sdk):
    """Test upsert_vectors calls client correctly."""
    points = [
        PointStruct(id="1", vector=[0.1]*128, payload={"key": "val"})
    ]

    await qdrant_client.upsert_vectors(points)

    mock_qdrant_sdk.upsert.assert_called_once_with(
        collection_name="test_collection",
        points=points
    )

@pytest.mark.asyncio
async def test_search_basic(qdrant_client, mock_qdrant_sdk):
    """Test search without filter."""
    query_vector = [0.1] * 128
    
    # Mock search result
    mock_result = MagicMock()
    mock_result.id = "doc1"
    mock_result.score = 0.95
    mock_result.payload = {"content": "text"}
    mock_qdrant_sdk.search.return_value = [mock_result]

    results = await qdrant_client.search(query_vector, limit=3)

    mock_qdrant_sdk.search.assert_called_once()
    call_kwargs = mock_qdrant_sdk.search.call_args.kwargs
    assert call_kwargs["collection_name"] == "test_collection"
    assert call_kwargs["query_vector"] == query_vector
    assert call_kwargs["limit"] == 3
    assert call_kwargs["query_filter"] is None

    assert len(results) == 1
    assert results[0]["id"] == "doc1"
    assert results[0]["score"] == 0.95

@pytest.mark.asyncio
async def test_search_with_filter(qdrant_client, mock_qdrant_sdk):
    """Test search with metadata filter."""
    query_vector = [0.1] * 128
    filter_dict = {"document_id": "123"}
    
    await qdrant_client.search(query_vector, filter_dict=filter_dict)

    mock_qdrant_sdk.search.assert_called_once()
    call_kwargs = mock_qdrant_sdk.search.call_args.kwargs
    query_filter = call_kwargs["query_filter"]
    
    assert isinstance(query_filter, Filter)
    assert len(query_filter.must) == 1
    condition = query_filter.must[0]
    assert isinstance(condition, FieldCondition)
    assert condition.key == "document_id"
    assert condition.match.value == "123"

@pytest.mark.asyncio
async def test_delete_document(qdrant_client, mock_qdrant_sdk):
    """Test delete_document constructs correct filter."""
    await qdrant_client.delete_document("doc-123")

    mock_qdrant_sdk.delete.assert_called_once()
    call_kwargs = mock_qdrant_sdk.delete.call_args.kwargs
    assert call_kwargs["collection_name"] == "test_collection"
    
    selector = call_kwargs["points_selector"]
    assert isinstance(selector, Filter)
    assert len(selector.must) == 1
    condition = selector.must[0]
    assert condition.key == "document_id"
    assert condition.match.value == "doc-123"
