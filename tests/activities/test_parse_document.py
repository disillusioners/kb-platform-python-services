"""Tests for parse_document activity."""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import MagicMock, patch
from workers.activities import parse_document


@pytest.mark.asyncio
async def test_parse_document_pdf_success():
    """Test successful PDF document parsing."""
    document_id = "doc-123"
    filename = "test.pdf"
    # Minimal PDF content
    file_data = b"%PDF-1.4 test content"

    with patch('workers.activities._parse_pdf') as mock_parse_pdf:
        mock_parse_pdf.return_value = "Parsed PDF text content"

        with patch('workers.activities.TokenTextSplitter') as MockSplitter:
            mock_splitter = MagicMock()
            mock_splitter.split_text.return_value = ["chunk1", "chunk2"]
            MockSplitter.return_value = mock_splitter

            chunks = await parse_document(
                document_id=document_id,
                file_data=file_data,
                filename=filename
            )

            assert len(chunks) == 2
            assert chunks[0].content == "chunk1"
            assert chunks[0].filename == filename


@pytest.mark.asyncio
async def test_parse_document_txt_success():
    """Test successful TXT document parsing."""
    document_id = "doc-456"
    filename = "notes.txt"
    file_data = b"This is a test document.\nWith multiple lines."

    with patch('workers.activities._parse_text') as mock_parse_text:
        mock_parse_text.return_value = "Parsed text content"

        with patch('workers.activities.TokenTextSplitter') as MockSplitter:
            mock_splitter = MagicMock()
            mock_splitter.split_text.return_value = ["Single chunk"]
            MockSplitter.return_value = mock_splitter

            chunks = await parse_document(
                document_id=document_id,
                file_data=file_data,
                filename=filename
            )

            assert len(chunks) == 1
            assert chunks[0].content == "Single chunk"


@pytest.mark.asyncio
async def test_parse_document_unsupported_type():
    """Test parsing fails with unsupported file type."""
    document_id = "doc-789"
    filename = "document.xyz"
    file_data = b"unknown file format"

    with pytest.raises(ValueError, match="Unsupported file type: .xyz"):
        await parse_document(
            document_id=document_id,
            file_data=file_data,
            filename=filename
        )


@pytest.mark.asyncio
async def test_parse_document_empty_content():
    """Test parsing document with empty content."""
    document_id = "doc-empty"
    filename = "empty.txt"
    file_data = b""

    with patch('workers.activities._parse_text') as mock_parse_text:
        mock_parse_text.return_value = ""

        with patch('workers.activities.TokenTextSplitter') as MockSplitter:
            mock_splitter = MagicMock()
            mock_splitter.split_text.return_value = []
            MockSplitter.return_value = mock_splitter

            chunks = await parse_document(
                document_id=document_id,
                file_data=file_data,
                filename=filename
            )

            assert len(chunks) == 0


@pytest.mark.asyncio
async def test_parse_document_with_custom_parser():
    """Test parsing with custom injected parser."""
    document_id = "doc-custom"
    filename = "test.dat"
    file_data = b"custom data"

    def custom_parser(data: bytes, ext: str) -> str:
        return f"Custom parsed: {data.decode()}"

    mock_splitter = MagicMock()
    mock_splitter.split_text.return_value = ["custom parsed chunk"]

    chunks = await parse_document(
        document_id=document_id,
        file_data=file_data,
        filename=filename,
        file_parser=custom_parser,
        text_splitter=mock_splitter
    )

    assert len(chunks) == 1
    assert chunks[0].content == "custom parsed chunk"
