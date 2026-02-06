"""Unit tests for parse_document activity."""

import pytest
from unittest.mock import MagicMock
from datetime import datetime
from workers.activities import parse_document
from shared.models.schemas import DocumentChunk


@pytest.fixture
def mock_text_splitter(mocker):
    """Create a mock text splitter."""
    mock = mocker.MagicMock()
    mock.split_text.return_value = ["chunk1", "chunk2", "chunk3"]
    return mock


@pytest.fixture
def mock_file_parser(mocker):
    """Create a mock file parser."""
    return mocker.MagicMock()


@pytest.mark.asyncio
async def test_parse_pdf_success(mock_text_splitter):
    """Test successful PDF parsing."""
    # Setup mock file parser to return sample text
    def mock_parser(file_data, ext):
        return "Sample PDF text content with multiple paragraphs."
    
    result = await parse_document(
        "test-doc-id",
        b"fake pdf bytes",
        "document.pdf",
        file_parser=mock_parser,
        text_splitter=mock_text_splitter
    )
    
    # Verify result
    assert len(result) == 3
    assert all(isinstance(chunk, DocumentChunk) for chunk in result)
    assert result[0].filename == "document.pdf"
    assert result[0].chunk_index == 0
    assert result[0].embedding == []  # Empty until embeddings generated
    assert isinstance(result[0].upload_timestamp, datetime)


@pytest.mark.asyncio
async def test_parse_txt_success(mock_text_splitter):
    """Test successful TXT parsing."""
    def mock_parser(file_data, ext):
        return "Plain text content for testing."
    
    result = await parse_document(
        "test-doc-id",
        b"text content",
        "notes.txt",
        file_parser=mock_parser,
        text_splitter=mock_text_splitter
    )
    
    # Verify result
    assert len(result) == 3
    assert all(isinstance(chunk, DocumentChunk) for chunk in result)


@pytest.mark.asyncio
async def test_unsupported_file_type():
    """Test ValueError for unsupported file type."""
    def mock_parser(file_data, ext):
        return "Some content"
    
    # Call with unsupported file type
    with pytest.raises(ValueError) as exc_info:
        await parse_document(
            "test-doc-id",
            b"some data",
            "file.xyz",
            file_parser=mock_parser
        )
    
    # Verify error message
    assert "Unsupported file type" in str(exc_info.value)
    assert ".xyz" in str(exc_info.value)


@pytest.mark.asyncio
async def test_empty_document(mock_file_parser):
    """Test handling of empty document content."""
    def mock_parser(file_data, ext):
        return ""  # Empty content
    
    # Mock splitter to return empty list
    mock_splitter = MagicMock()
    mock_splitter.split_text.return_value = []
    
    result = await parse_document(
        "test-doc-id",
        b"",
        "empty.txt",
        file_parser=mock_parser,
        text_splitter=mock_splitter
    )
    
    # Verify empty result
    assert len(result) == 0


@pytest.mark.asyncio
async def test_single_chunk_document(mock_file_parser):
    """Test document that produces single chunk."""
    def mock_parser(file_data, ext):
        return "Short text"
    
    # Mock splitter to return single chunk
    mock_splitter = MagicMock()
    mock_splitter.split_text.return_value = ["Short text"]
    
    result = await parse_document(
        "test-doc-id",
        b"short content",
        "short.md",
        file_parser=mock_parser,
        text_splitter=mock_splitter
    )
    
    # Verify single chunk
    assert len(result) == 1
    assert result[0].content == "Short text"
    assert result[0].chunk_index == 0
