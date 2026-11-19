"""Tests for ragbot.text_processor — TextProcessor class."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ragbot.text_processor import TextProcessor


def _make_document(text: str) -> MagicMock:
    """Create a mock LangChain Document with the given page_content."""
    doc = MagicMock()
    doc.page_content = text
    doc.metadata = {}
    return doc


class TestTextProcessor:
    """Verify that TextProcessor correctly splits documents into chunks."""

    def test_split_documents(self):
        """Splitting a long document should produce multiple smaller chunks."""
        long_text = "Artificial intelligence is transforming the world. " * 100
        docs = [_make_document(long_text)]

        processor = TextProcessor(chunk_size=500, chunk_overlap=50)
        chunks = processor.split_documents(docs)

        assert isinstance(chunks, list)
        assert len(chunks) > 1, "A long document should be split into multiple chunks"

    def test_chunk_size_respected(self):
        """Each chunk's length should not greatly exceed the configured chunk_size.

        A small tolerance is allowed because RecursiveCharacterTextSplitter
        may occasionally produce chunks slightly over the target when it
        cannot find a good split boundary.
        """
        long_text = "Word " * 2000  # ~10 000 characters
        docs = [_make_document(long_text)]

        chunk_size = 300
        processor = TextProcessor(chunk_size=chunk_size, chunk_overlap=30)
        chunks = processor.split_documents(docs)

        tolerance = chunk_size * 1.15  # 15 % tolerance
        for chunk in chunks:
            assert len(chunk.page_content) <= tolerance, (
                f"Chunk length {len(chunk.page_content)} exceeds tolerance {tolerance}"
            )

    def test_empty_documents(self):
        """Splitting an empty list should return an empty list, not raise."""
        processor = TextProcessor(chunk_size=500, chunk_overlap=50)
        result = processor.split_documents([])

        assert result == []

    def test_custom_chunk_params(self):
        """Custom chunk_size and chunk_overlap should be respected."""
        text = "Hello world. " * 500
        docs = [_make_document(text)]

        small_processor = TextProcessor(chunk_size=100, chunk_overlap=10)
        large_processor = TextProcessor(chunk_size=1000, chunk_overlap=100)

        small_chunks = small_processor.split_documents(docs)
        large_chunks = large_processor.split_documents(docs)

        assert len(small_chunks) > len(large_chunks), (
            "Smaller chunk_size should produce more chunks than larger chunk_size"
        )

    def test_single_short_document(self):
        """A document shorter than chunk_size should come back as one chunk."""
        docs = [_make_document("Short text.")]

        processor = TextProcessor(chunk_size=500, chunk_overlap=50)
        chunks = processor.split_documents(docs)

        assert len(chunks) == 1
        assert chunks[0].page_content.strip() == "Short text."

    @pytest.mark.parametrize(
        ("chunk_size", "overlap"),
        [
            (200, 20),
            (500, 50),
            (1000, 100),
        ],
        ids=["small", "medium", "large"],
    )
    def test_various_chunk_configurations(self, chunk_size: int, overlap: int):
        """TextProcessor should work correctly across a range of configurations."""
        text = "Data science involves statistics, programming, and domain expertise. " * 200
        docs = [_make_document(text)]

        processor = TextProcessor(chunk_size=chunk_size, chunk_overlap=overlap)
        chunks = processor.split_documents(docs)

        assert len(chunks) >= 1
        for chunk in chunks:
            assert len(chunk.page_content) > 0
