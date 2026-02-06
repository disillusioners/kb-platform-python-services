"""Tests for delete_from_s3 activity."""

import pytest
from unittest.mock import MagicMock, ANY, patch
from workers.activities import delete_from_s3


@pytest.fixture
def mock_s3_client():
    """Create a mock S3 client."""
    mock_client = MagicMock()
    return mock_client


@pytest.mark.asyncio
async def test_delete_from_s3_success(mock_s3_client):
    """Test successful deletion from S3."""
    s3_key = "documents/doc-123/test.pdf"

    result = await delete_from_s3(
        s3_key=s3_key,
        s3_client=mock_s3_client
    )

    assert result == f"Deleted {s3_key}"
    mock_s3_client.delete_object.assert_called_once_with(
        Bucket=ANY,  # Settings will provide bucket
        Key=s3_key
    )


@pytest.mark.asyncio
async def test_delete_from_s3_failure_returns_message(mock_s3_client):
    """Test deletion failure returns error message (doesn't raise)."""
    s3_key = "documents/doc-123/missing.pdf"

    mock_s3_client.delete_object.side_effect = Exception("S3 access denied")

    result = await delete_from_s3(
        s3_key=s3_key,
        s3_client=mock_s3_client
    )

    assert "Failed to delete" in result
    assert s3_key in result


@pytest.mark.asyncio
async def test_delete_from_s3_with_different_keys():
    """Test deletion with various S3 key formats."""
    test_cases = [
        "documents/doc-123/file.pdf",
        "uploads/2024/01/15/large-file.docx",
        "archive/old-document.txt"
    ]

    for s3_key in test_cases:
        mock_client = MagicMock()
        result = await delete_from_s3(s3_key=s3_key, s3_client=mock_client)
        assert result == f"Deleted {s3_key}"
        mock_client.delete_object.assert_called_once()


@pytest.mark.asyncio
async def test_delete_from_s3_cleanup_failure_still_returns():
    """Test that cleanup failure doesn't raise exception."""
    s3_key = "documents/doc-to-delete.pdf"

    mock_client = MagicMock()
    mock_client.delete_object.side_effect = Exception("Network error")

    result = await delete_from_s3(s3_key=s3_key, s3_client=mock_client)

    assert "Failed to delete" in result
    assert s3_key in result
