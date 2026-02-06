"""Unit tests for delete_from_s3 activity."""

import pytest
from unittest.mock import MagicMock, AsyncMock
from workers.activities import delete_from_s3


@pytest.fixture
def mock_s3_client(mocker):
    """Create a mock S3 client."""
    return mocker.MagicMock()


@pytest.fixture
def mock_settings(mocker):
    """Mock settings."""
    settings = mocker.Mock()
    settings.s3_bucket = "test-bucket"
    mocker.patch("workers.activities.get_settings", return_value=settings)
    return settings


@pytest.mark.asyncio
async def test_delete_success(mock_s3_client, mock_settings):
    """Test successful file deletion from S3."""
    # Call the activity with mocked S3 client
    result = await delete_from_s3(
        "documents/test-id/file.pdf",
        s3_client=mock_s3_client
    )
    
    # Assert successful deletion
    assert result == "Deleted documents/test-id/file.pdf"
    mock_s3_client.delete_object.assert_called_once_with(
        Bucket="test-bucket",
        Key="documents/test-id/file.pdf"
    )


@pytest.mark.asyncio
async def test_delete_file_not_found(mock_s3_client):
    """Test handling of 404 Not Found error."""
    # Simulate S3 404 error
    error_response = {
        "Error": {
            "Code": "404",
            "Message": "Not Found"
        }
    }
    mock_s3_client.delete_object.side_effect = Exception(
        "An error occurred (404) when calling the DeleteObject operation: Not Found"
    )
    
    # Call the activity - should handle gracefully and return error message
    result = await delete_from_s3(
        "documents/test-id/nonexistent.pdf",
        s3_client=mock_s3_client
    )
    
    # Assert that the error is handled gracefully (no exception raised)
    assert "Failed to delete" in result
    assert "documents/test-id/nonexistent.pdf" in result


@pytest.mark.asyncio
async def test_delete_s3_error(mock_s3_client):
    """Test handling of general S3 errors."""
    # Simulate a general S3 error
    mock_s3_client.delete_object.side_effect = Exception(
        "Access Denied"
    )
    
    # Call the activity - should handle gracefully
    result = await delete_from_s3(
        "documents/test-id/protected-file.pdf",
        s3_client=mock_s3_client
    )
    
    # Assert that the error is handled gracefully
    assert "Failed to delete" in result
    assert "documents/test-id/protected-file.pdf" in result
