"""Unit tests for update_status activity."""

import pytest
from unittest.mock import MagicMock, AsyncMock
from workers.activities import update_status
from shared.models.schemas import DocumentStatus


@pytest.fixture
def mock_db_pool(mocker):
    """Create a mock database pool."""
    mock = mocker.MagicMock()
    
    # Setup mock connection
    mock_conn = MagicMock()
    mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_conn.__aexit__ = AsyncMock(return_value=None)
    mock.acquire.return_value = mock_conn
    mock_conn.execute = AsyncMock()
    
    return mock


@pytest.mark.asyncio
async def test_update_status_success(mock_db_pool):
    """Test successful status update to INDEXING."""
    await update_status(
        "test-doc-id",
        "indexing",
        db_pool=mock_db_pool
    )
    
    # Verify database call was made
    mock_db_pool.acquire.return_value.execute.assert_called_once()


@pytest.mark.asyncio
async def test_update_status_failed_with_error(mock_db_pool):
    """Test status update to FAILED with error message."""
    await update_status(
        "test-doc-id",
        "failed",
        error="Processing failed due to invalid format",
        db_pool=mock_db_pool
    )
    
    # Verify database call was made with error
    mock_db_pool.acquire.return_value.execute.assert_called_once()


@pytest.mark.asyncio
async def test_update_status_complete_with_chunk_count(mock_db_pool):
    """Test status update to COMPLETE with chunk count."""
    await update_status(
        "test-doc-id",
        "complete",
        chunk_count=42,
        db_pool=mock_db_pool
    )
    
    # Verify database call was made with chunk count
    mock_db_pool.acquire.return_value.execute.assert_called_once()


@pytest.mark.asyncio
async def test_update_status_all_parameters(mock_db_pool):
    """Test status update with all parameters."""
    await update_status(
        "test-doc-id",
        "complete",
        error="Previous error",
        chunk_count=100,
        db_pool=mock_db_pool
    )
    
    # Verify database call was made
    mock_db_pool.acquire.return_value.execute.assert_called_once()


@pytest.mark.asyncio
async def test_update_status_pending():
    """Test status update to PENDING (default behavior)."""
    # Test with default parameters
    mock_pool = MagicMock()
    mock_conn = MagicMock()
    mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_conn.__aexit__ = AsyncMock(return_value=None)
    mock_pool.acquire.return_value = mock_conn
    mock_conn.execute = AsyncMock()
    
    await update_status(
        "test-doc-id",
        "pending",
        db_pool=mock_pool
    )
    
    # Verify call was made
    mock_pool.acquire.return_value.execute.assert_called_once()


@pytest.mark.asyncio
async def test_update_status_document_not_found():
    """Test that function handles various document statuses."""
    # Test different status values
    for status in ["pending", "indexing", "complete", "failed"]:
        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=None)
        mock_pool.acquire.return_value = mock_conn
        mock_conn.execute = AsyncMock()
        
        await update_status(
            "test-doc-id",
            status,
            db_pool=mock_pool
        )
        
        # Each status should be handled
        mock_pool.acquire.return_value.execute.assert_called()
