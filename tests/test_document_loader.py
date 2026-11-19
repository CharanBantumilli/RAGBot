"""Tests for ragbot.document_loader — DocumentLoader class."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ragbot.document_loader import DocumentLoader
from ragbot.exceptions import DocumentLoadError, UnsupportedFileTypeError


class TestDocumentLoader:
    """Verify that DocumentLoader can load various file types and reject bad inputs."""

    def test_load_text_file(self, sample_text_file: Path):
        """Loading a .txt file should return a non-empty list of documents."""
        loader = DocumentLoader()
        documents = loader.load(str(sample_text_file))

        assert isinstance(documents, list)
        assert len(documents) > 0
        # Each document should contain some text content
        for doc in documents:
            assert hasattr(doc, "page_content")
            assert len(doc.page_content) > 0

    def test_load_csv_file(self, sample_csv_file: Path):
        """Loading a .csv file should return documents, one per row."""
        loader = DocumentLoader()
        documents = loader.load(str(sample_csv_file))

        assert isinstance(documents, list)
        assert len(documents) > 0
        for doc in documents:
            assert hasattr(doc, "page_content")

    def test_unsupported_extension(self, tmp_path: Path):
        """Loading a file with an unrecognised extension should raise UnsupportedFileTypeError."""
        bad_file = tmp_path / "data.xyz"
        bad_file.write_text("some content", encoding="utf-8")

        loader = DocumentLoader()

        with pytest.raises(UnsupportedFileTypeError):
            loader.load(str(bad_file))

    def test_nonexistent_file(self):
        """Loading a file that does not exist should raise DocumentLoadError."""
        loader = DocumentLoader()

        with pytest.raises(DocumentLoadError):
            loader.load("/nonexistent/path/to/file.txt")

    @pytest.mark.parametrize(
        "url",
        [
            "http://example.com/page",
            "https://example.com/page",
            "https://docs.python.org/3/library/os.html",
        ],
        ids=["http", "https", "https-nested"],
    )
    def test_url_detection(self, url: str):
        """URLs starting with http:// or https:// should be identified as web sources."""
        loader = DocumentLoader()

        # DocumentLoader should expose a way to detect URLs
        assert loader._is_url(url) is True

    @pytest.mark.parametrize(
        "not_url",
        [
            "ftp://example.com",
            "/local/path.txt",
            "relative/path.pdf",
        ],
        ids=["ftp", "absolute-path", "relative-path"],
    )
    def test_non_url_detection(self, not_url: str):
        """Non-HTTP paths should NOT be detected as web URLs."""
        loader = DocumentLoader()

        assert loader._is_url(not_url) is False

    def test_validate_file(self, sample_text_file: Path):
        """The validation method should accept an existing file with a supported extension."""
        loader = DocumentLoader()

        # Should not raise for a valid file
        loader._validate_file(str(sample_text_file))

    def test_validate_file_missing(self):
        """The validation method should raise for a missing file."""
        loader = DocumentLoader()

        with pytest.raises(DocumentLoadError):
            loader._validate_file("/no/such/file.txt")

    def test_validate_file_unsupported_type(self, tmp_path: Path):
        """The validation method should raise for an unsupported file type."""
        bad_file = tmp_path / "notes.xyz"
        bad_file.write_text("content", encoding="utf-8")

        loader = DocumentLoader()

        with pytest.raises(UnsupportedFileTypeError):
            loader._validate_file(str(bad_file))
