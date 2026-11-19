"""Vector store management for RAGBot.

Wraps Chroma and HuggingFace embeddings behind a manager that lazily
loads the embedding model so import-time cost stays minimal.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from langchain_core.documents import Document

from ragbot.exceptions import VectorStoreError

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Create and query a Chroma vector store.

    The embedding model is loaded on first use, not at import time, to keep
    startup fast and avoid GPU allocation until it is actually needed.

    Args:
        embedding_model_name: HuggingFace model identifier for sentence
            embeddings.
        persist_directory: Optional directory to persist the Chroma index.
    """

    def __init__(
        self,
        embedding_model_name: str = "intfloat/multilingual-e5-small",
        persist_directory: Optional[str] = None,
    ) -> None:
        self._embedding_model_name = embedding_model_name
        self._persist_directory = persist_directory
        self._embeddings = None  # lazy
        self._store = None

    # ------------------------------------------------------------------
    # Lazy embedding property
    # ------------------------------------------------------------------

    @property
    def embeddings(self):
        """Return the HuggingFace embedding model, loading it on first access.

        Returns:
            A :class:`HuggingFaceEmbeddings` instance.
        """
        if self._embeddings is None:
            from langchain_huggingface import HuggingFaceEmbeddings

            logger.info(
                "Loading embedding model '%s'…", self._embedding_model_name
            )
            self._embeddings = HuggingFaceEmbeddings(
                model_name=self._embedding_model_name
            )
        return self._embeddings

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_store(self, documents: List[Document]) -> None:
        """Build a Chroma vector store from *documents*.

        Args:
            documents: Chunked documents to embed and index.

        Raises:
            VectorStoreError: If the store cannot be created.
        """
        from langchain_chroma import Chroma

        try:
            logger.info("Creating Chroma vector store from %d chunk(s)…", len(documents))
            kwargs: Dict = {
                "documents": documents,
                "embedding": self.embeddings,
            }
            if self._persist_directory:
                kwargs["persist_directory"] = self._persist_directory
            self._store = Chroma.from_documents(**kwargs)
            logger.info("Vector store created successfully.")
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to create vector store: {exc}"
            ) from exc

    def get_retriever(self, search_kwargs: Optional[Dict] = None):
        """Return a retriever backed by the current vector store.

        Args:
            search_kwargs: Optional dictionary of search parameters
                (e.g. ``{"k": 4}``).

        Returns:
            A LangChain retriever instance.

        Raises:
            VectorStoreError: If no vector store has been created yet.
        """
        if self._store is None:
            raise VectorStoreError(
                "No vector store available. Call create_store() first."
            )
        search_kwargs = search_kwargs or {"k": 4}
        return self._store.as_retriever(search_kwargs=search_kwargs)
