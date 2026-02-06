"""Tests for update_status activity."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from workers.activities import update_status
from shared.models.schemas import DocumentStatus


@pytest.fixture
def mock_db_pool():
    """Create a mock database pool."""
    mock_pool = MagicMock()
    mock_conn = MagicMock()
    mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
    mock_pool.acquire.return_value.__aexit__.return_value = None
    return mock_pool


@pytest.mark.asyncio
async def test_update_status_success(mock_db_pool):
    """Test successful status update."""
    document_id = "doc-123"
    status = "complete"

    with patch('workers.activities.update_document_status', new_callable=AsyncMock) as mock_update:
        await update_status(
            document_id=document_id,
            status=status,
            db_pool=mock_db_pool
        )

        mock_update.assert_called_once()
        call_args = mock_update.call_args
        assert call_args[0][1] == document_id


@pytest.mark.asyncio
async def test_update_status_with_error(mock_db_pool):
    """Test status update with error message."""
    document_id = "doc-failed"
    status = "failed"
    error_message = "Failed to process document"

    with patch('workers.activities.update_document_status', new_callable=AsyncMock) as mock_update:
        await update_status(
            document_id=document_id,
            status=status,
            error=error_message,
            db_pool=mock_db_pool
        )

        mock_update.assert_called_once()
        call_args = mock_update.call_args
        assert call_args[1]["error_message"] == error_message


@pytest.mark.asyncio
async def test_update_status_with_chunk_count(mock_db_pool):
    """Test status update with chunk count."""
    document_id = "doc-123"
    status = "complete"
    chunk_count = 42

    with patch('workers.activities.update_document_status', new_callable=AsyncMock) as mock_update:
        await update_status(
            document_id=document_id,
            status=status,
            chunk_count=chunk_count,
            db_pool=mock_db_pool
        )

        mock_update.assert_called_once()
        call_args = mock_update.call_args
        assert call_args[1]["chunk_count"] == chunk_count


@pytest.mark.asyncio
async def test_update_status_indexing():
    """Test status update to indexing."""
    document_id = "doc-123"

    mock_pool = MagicMock()
    mock_conn = MagicMock()
    mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
    mock_pool.acquire.return_value.__aexit__.return_value = None

    with patch('workers.activities.update_document_status', new_callable=AsyncMock) as mock_update:
        await update_status(
            document_id=document_id,
            status="indexing",
            db_pool=mock_pool
        )

        mock_update.assert_called_once()
