"""Text chunking utilities for RAGBot.

Splits loaded :class:`Document` objects into smaller, overlapping chunks
suitable for embedding and retrieval.
"""

from __future__ import annotations

import logging
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


class TextProcessor:
    """Split documents into chunks using a recursive character strategy.

    Args:
        chunk_size: Maximum number of characters per chunk.
        chunk_overlap: Number of overlapping characters between consecutive
            chunks.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50) -> None:
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
            length_function=len,
        )
        logger.info(
            "TextProcessor initialised (chunk_size=%d, chunk_overlap=%d).",
            self._chunk_size,
            self._chunk_overlap,
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split *documents* into smaller chunks.

        Args:
            documents: The documents to split.

        Returns:
            A list of chunked :class:`Document` objects.
        """
        chunks = self._splitter.split_documents(documents)
        logger.info(
            "Split %d document(s) into %d chunk(s).",
            len(documents),
            len(chunks),
        )
        return chunks
