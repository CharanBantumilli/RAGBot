"""Custom exception hierarchy for RAGBot.

Every public exception inherits from :class:`RAGBotError` so callers can
catch the base class for a broad handler or a specific subclass for
fine-grained control.
"""


class RAGBotError(Exception):
    """Base exception for all RAGBot errors."""


class DocumentLoadError(RAGBotError):
    """Raised when a document cannot be loaded or parsed."""


class UnsupportedFileTypeError(RAGBotError):
    """Raised when the provided file type is not supported."""


class VectorStoreError(RAGBotError):
    """Raised when vector-store creation or retrieval fails."""


class LLMError(RAGBotError):
    """Raised when the language model fails to generate a response."""


class ConfigurationError(RAGBotError):
    """Raised when required configuration is missing or invalid."""
