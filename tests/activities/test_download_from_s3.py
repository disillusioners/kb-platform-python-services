"""Unit tests for download_from_s3 activity."""

import pytest
from unittest.mock import MagicMock, AsyncMock
from workers.activities import download_from_s3


@pytest.fixture
def mock_s3_client(mocker):
    """Create a mock S3 client."""
    return mocker.MagicMock()


@pytest.fixture
def mock_db_pool(mocker):
    """Create a mock database pool."""
    return mocker.MagicMock()


@pytest.mark.asyncio
async def test_download_success(mock_s3_client, mock_db_pool):
    """Test successful document download from S3."""
    # Setup mock document
    mock_doc = MagicMock()
    mock_doc.s3_key = "documents/test-id/file.pdf"
    mock_doc.filename = "test-file.pdf"
    
    # Setup mock connection
    mock_conn = MagicMock()
    mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_conn.__aexit__ = AsyncMock(return_value=None)
    mock_db_pool.acquire.return_value = mock_conn
    mock_conn.fetchrow = AsyncMock(return_value={
        "id": "12345678-1234-5678-1234-567812345678",
        "s3_key": "documents/test-id/file.pdf",
        "filename": "test-file.pdf",
        "file_size": 1024,
        "status": "pending",
        "created_at": "2023-01-01T00:00:00"
    })
    
    # Setup mock S3 response
    mock_s3_client.get_object.return_value = {
        "Body": MagicMock(read=MagicMock(return_value=b"file content"))
    }
    
    # Call the activity with mocked dependencies
    result = await download_from_s3(
        "test-doc-id",
        s3_client=mock_s3_client,
        db_pool=mock_db_pool
    )
    
    # Assertions
    assert result == (b"file content", "test-file.pdf")
    mock_s3_client.get_object.assert_called_once_with(
        Bucket="kb-documents",
        Key="documents/test-id/file.pdf"
    )


@pytest.mark.asyncio
async def test_document_not_found(mock_s3_client, mock_db_pool):
    """Test ValueError when document doesn't exist in database."""
    # Setup mock connection to return None (document not found)
    mock_conn = MagicMock()
    mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_conn.__aexit__ = AsyncMock(return_value=None)
    mock_db_pool.acquire.return_value = mock_conn
    mock_conn.fetchrow = AsyncMock(return_value=None)
    
    # Call the activity - should raise ValueError
    with pytest.raises(ValueError) as exc_info:
        await download_from_s3(
            "nonexistent-doc-id",
            s3_client=mock_s3_client,
            db_pool=mock_db_pool
        )
    
    # Assert error message
    assert "not found" in str(exc_info.value).lower()
    assert "nonexistent-doc-id" in str(exc_info.value)


@pytest.mark.asyncio
async def test_s3_download_error(mock_s3_client, mock_db_pool):
    """Test exception propagation when S3 download fails."""
    # Setup mock document
    mock_doc = MagicMock()
    mock_doc.s3_key = "documents/test-id/file.pdf"
    mock_doc.filename = "test-file.pdf"
    
    # Setup mock connection
    mock_conn = MagicMock()
    mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_conn.__aexit__ = AsyncMock(return_value=None)
    mock_db_pool.acquire.return_value = mock_conn
    mock_conn.fetchrow = AsyncMock(return_value={
        "id": "12345678-1234-5678-1234-567812345678",
        "s3_key": "documents/test-id/file.pdf",
        "filename": "test-file.pdf",
        "file_size": 1024,
        "status": "pending",
        "created_at": "2023-01-01T00:00:00"
    })
    
    # Setup mock S3 to raise exception
    mock_s3_client.get_object.side_effect = Exception("S3 access denied")
    
    # Call the activity - should propagate exception
    with pytest.raises(Exception) as exc_info:
        await download_from_s3(
            "test-doc-id",
            s3_client=mock_s3_client,
            db_pool=mock_db_pool
        )
    
    # Assert error message
    assert "S3 access denied" in str(exc_info.value)
