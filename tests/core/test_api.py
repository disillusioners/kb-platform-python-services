"""Integration tests for Core API."""

import pytest
import json
import uuid
from unittest.mock import MagicMock, AsyncMock, patch
try:
    from fastapi.testclient import TestClient
    from core.api.main import app, get_db
    deps_installed = True
except ImportError:
    TestClient = None
    app = None
    get_db = None
    deps_installed = False

from shared.models.schemas import SSEEvent

@pytest.fixture
def mock_db_conn():
    """Mock database connection."""
    conn = AsyncMock()
    conn.mock_add_all = MagicMock()
    return conn

@pytest.fixture
def client(mock_db_conn):
    """FastAPI test client with overwritten DB dependency."""
    if not deps_installed:
        yield None
        return
        
    async def override_get_db():
        yield mock_db_conn
    
    app.dependency_overrides[get_db] = override_get_db
    # Also patch the global rag_engine to avoid real API calls
    with patch("core.api.main.rag_engine") as mock_engine:
        # Default mock: empty query result
        async def mock_query(*args, **kwargs):
            yield "Mock Answer"
        mock_engine.query.side_effect = mock_query
        
        with TestClient(app) as test_client:
            yield test_client
    
    app.dependency_overrides.clear()

@pytest.mark.skipif(not deps_installed, reason="FastAPI dependencies not installed")
def test_health_check(client):
    """Test healthz endpoint."""
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

@pytest.mark.skipif(not deps_installed, reason="FastAPI dependencies not installed")
def test_readiness_check(client):
    """Test readyz endpoint."""
    response = client.get("/readyz")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"
    assert "openai" in response.json()["dependencies"]

@pytest.mark.skipif(not deps_installed, reason="FastAPI dependencies not installed")
@patch("core.api.main.create_conversation")
@patch("core.api.main.get_messages")
@patch("core.api.main.save_message")
def test_query_new_conversation(mock_save, mock_get_msgs, mock_create, client, mock_db_conn):
    """Test query endpoint generates conversation ID if missing."""
    # Setup mocks
    new_conv_id = uuid.uuid4()
    mock_create.return_value = MagicMock(id=new_conv_id)
    mock_get_msgs.return_value = [] # No history
    
    # Request without conversation_id
    payload = {"query": "Hello", "top_k": 3}
    response = client.post("/api/v1/query", json=payload)
    
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream"
    
    # Verify DB interactions
    mock_create.assert_called_once() # Should create conversation
    mock_get_msgs.assert_called_once()
    assert mock_save.call_count == 2 # 1 user msg + 1 assistant msg

    # Verify stream content
    content = response.content.decode("utf-8")
    assert "data: " in content
    
    # Check for start/chunk/end events
    events = [line for line in content.split("\n\n") if line.startswith("data: ")]
    assert len(events) >= 3 # start, chunk, end
    
    # Verify 'start' event has ID
    start_event = json.loads(events[0].replace("data: ", ""))
    assert start_event["type"] == "start"
    assert start_event["id"] is not None

@pytest.mark.skipif(not deps_installed, reason="FastAPI dependencies not installed")
@patch("core.api.main.create_conversation")
@patch("core.api.main.get_messages")
@patch("core.api.main.save_message")
def test_query_existing_conversation(mock_save, mock_get_msgs, mock_create, client):
    """Test query endpoint uses existing conversation ID."""
    # Setup mocks
    existing_id = str(uuid.uuid4())
    mock_get_msgs.return_value = [
        MagicMock(role=MagicMock(value="user"), content="Hi"),
        MagicMock(role=MagicMock(value="assistant"), content="Hello")
    ]
    
    # Request WITH conversation_id
    payload = {
        "query": "Follow up", 
        "conversation_id": existing_id
    }
    response = client.post("/api/v1/query", json=payload)
    
    assert response.status_code == 200
    
    # Verify create_conversation NOT called
    mock_create.assert_not_called()
    
    # Verify history fetched for correct ID
    mock_get_msgs.assert_called_once()
    assert mock_get_msgs.call_args[0][1] == existing_id

@pytest.mark.skipif(not deps_installed, reason="FastAPI dependencies not installed")
@patch("core.api.main.create_conversation")
@patch("core.api.main.get_messages")
@patch("core.api.main.save_message")
def test_query_rag_error(mock_save, mock_get_msgs, mock_create, client):
    """Test handling of RAG engine error."""
    # Patch rag_engine locally to raise error
    with patch("core.api.main.rag_engine.query") as mock_query:
        mock_query.side_effect = Exception("RAG Failed")
        
        # Mocks setup
        new_conv_id = uuid.uuid4()
        mock_create.return_value = MagicMock(id=new_conv_id)
        mock_get_msgs.return_value = []

        response = client.post("/api/v1/query", json={"query": "fail"})
        
        assert response.status_code == 200
        
        # Check stream for error event
        content = response.content.decode("utf-8")
        assert "INTERNAL_ERROR" in content
        assert "RAG Failed" in content
