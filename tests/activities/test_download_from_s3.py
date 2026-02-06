"""Tests for download_from_s3 activity."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch, ANY
from workers.activities import download_from_s3


class MockDocument:
    """Mock Document for testing."""
    def __init__(self, doc_id: str, s3_key: str, filename: str):
        self.id = doc_id
        self.s3_key = s3_key
        self.filename = filename


@pytest.fixture
def mock_s3_client():
    """Create a mock S3 client."""
    mock_client = MagicMock()
    mock_response = {
        "Body": MagicMock(read=MagicMock(return_value=b"test document content"))
    }
    mock_client.get_object.return_value = mock_response
    return mock_client


@pytest.fixture
def mock_db_pool():
    """Create a mock database pool."""
    mock_pool = MagicMock()
    mock_conn = MagicMock()
    mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
    mock_pool.acquire.return_value.__aexit__.return_value = None
    return mock_pool


@pytest.mark.asyncio
async def test_download_from_s3_success(mock_s3_client, mock_db_pool):
    """Test successful document download from S3."""
    document_id = "doc-123"
    s3_key = "documents/doc-123/test.pdf"
    filename = "test.pdf"
    file_content = b"test document content"

    mock_doc = MockDocument(document_id, s3_key, filename)

    with patch('workers.activities.get_document', new_callable=AsyncMock) as mock_get_doc:
        mock_get_doc.return_value = mock_doc

        result_content, result_filename = await download_from_s3(
            document_id=document_id,
            s3_client=mock_s3_client,
            db_pool=mock_db_pool
        )

        assert result_content == file_content
        assert result_filename == filename
        mock_s3_client.get_object.assert_called_once_with(
            Bucket=ANY,  # Settings will provide bucket
            Key=s3_key
        )


@pytest.mark.asyncio
async def test_download_from_s3_document_not_found(mock_s3_client, mock_db_pool):
    """Test download fails when document not found in DB."""
    document_id = "nonexistent-doc"

    with patch('workers.activities.get_document', new_callable=AsyncMock) as mock_get_doc:
        mock_get_doc.return_value = None

        with pytest.raises(ValueError, match=f"Document {document_id} not found"):
            await download_from_s3(
                document_id=document_id,
                s3_client=mock_s3_client,
                db_pool=mock_db_pool
            )


@pytest.mark.asyncio
async def test_download_from_s3_s3_error(mock_s3_client, mock_db_pool):
    """Test download fails when S3 returns error."""
    document_id = "doc-123"
    s3_key = "documents/doc-123/test.pdf"

    mock_doc = MockDocument(document_id, s3_key, "test.pdf")

    mock_s3_client.get_object.side_effect = Exception("S3 access denied")

    with patch('workers.activities.get_document', new_callable=AsyncMock) as mock_get_doc:
        mock_get_doc.return_value = mock_doc

        with pytest.raises(Exception, match="S3 access denied"):
            await download_from_s3(
                document_id=document_id,
                s3_client=mock_s3_client,
                db_pool=mock_db_pool
            )
