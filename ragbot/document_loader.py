"""Document loading utilities for RAGBot.

Supports PDF, plain-text, CSV, DOCX files, and web URLs.  Unknown
extensions fall back to :class:`UnstructuredFileLoader`.
"""

from __future__ import annotations

import logging
import os
from typing import List

from langchain_community.document_loaders import (
    CSVLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredFileLoader,
    WebBaseLoader,
)
from langchain_core.documents import Document

from ragbot.exceptions import DocumentLoadError, UnsupportedFileTypeError

logger = logging.getLogger(__name__)


class DocumentLoader:
    """Load documents from local files or URLs.

    Attributes:
        SUPPORTED_EXTENSIONS: Mapping of file extensions to human-readable
            format names used for validation messages and UI hints.
    """

    SUPPORTED_EXTENSIONS: dict[str, str] = {
        ".pdf": "PDF",
        ".txt": "Plain Text",
        ".csv": "CSV",
        ".docx": "Microsoft Word",
    }

    @staticmethod
    def _is_url(path: str) -> bool:
        """Check if *path* is an HTTP or HTTPS URL."""
        return path.startswith(("http://", "https://"))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, file_path: str) -> List[Document]:
        """Load and return documents from *file_path*.

        Args:
            file_path: A local filesystem path or an HTTP(S) URL.

        Returns:
            A list of :class:`Document` objects.

        Raises:
            DocumentLoadError: If the file cannot be read or parsed.
            UnsupportedFileTypeError: If the extension is not supported.
        """
        # --- URL handling (bug-fix: original used broken `"http" or "https"`) ---
        if self._is_url(file_path):
            return self._load_url(file_path)

        self._validate_file(file_path)
        return self._load_local(file_path)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def validate_file(file_path: str) -> None:
        """Ensure *file_path* exists and is readable.

        Args:
            file_path: Path to validate.

        Raises:
            DocumentLoadError: If the file does not exist or is not a file.
        """
        if not os.path.exists(file_path):
            raise DocumentLoadError(f"File not found: {file_path}")
        if not os.path.isfile(file_path):
            raise DocumentLoadError(f"Path is not a file: {file_path}")

    def _validate_file(self, file_path: str) -> None:
        """Validate file existence and extension compatibility.

        Args:
            file_path: Path to validate.

        Raises:
            DocumentLoadError: If file not found or not regular file.
            UnsupportedFileTypeError: If file extension is unsupported.
        """
        self.validate_file(file_path)
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise UnsupportedFileTypeError(
                f"Unsupported file extension '{ext}'. Supported formats: {', '.join(self.SUPPORTED_EXTENSIONS.keys())}"
            )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_local(self, file_path: str) -> List[Document]:
        """Dispatch to the correct loader based on extension."""
        ext = os.path.splitext(file_path)[1].lower()
        logger.info("Loading local file '%s' (type: %s).", file_path, ext)

        loader_map: dict[str, type] = {
            ".pdf": PyPDFLoader,
            ".txt": TextLoader,
            ".csv": CSVLoader,
        }

        try:
            if ext in loader_map:
                loader = loader_map[ext](file_path)
            elif ext == ".docx":
                loader = UnstructuredFileLoader(file_path)
            else:
                logger.warning(
                    "Unrecognised extension '%s'; falling back to "
                    "UnstructuredFileLoader.",
                    ext,
                )
                loader = UnstructuredFileLoader(file_path)

            documents = loader.load()
            logger.info("Loaded %d document(s) from '%s'.", len(documents), file_path)
            return documents
        except Exception as exc:
            raise DocumentLoadError(
                f"Failed to load '{file_path}': {exc}"
            ) from exc

    @staticmethod
    def _load_url(url: str) -> List[Document]:
        """Load content from a web URL."""
        logger.info("Loading web content from '%s'.", url)
        try:
            loader = WebBaseLoader(url)
            documents = loader.load()
            logger.info("Loaded %d document(s) from URL.", len(documents))
            return documents
        except Exception as exc:
            raise DocumentLoadError(
                f"Failed to load URL '{url}': {exc}"
            ) from exc
