"""Tests for ragbot.exceptions — custom exception hierarchy."""

from __future__ import annotations

import pytest

from ragbot.exceptions import (
    ConfigurationError,
    DocumentLoadError,
    LLMError,
    RAGBotError,
    UnsupportedFileTypeError,
    VectorStoreError,
)


class TestRAGBotErrorBase:
    """Verify that RAGBotError is a proper Exception subclass."""

    def test_ragbot_error_is_exception(self):
        """RAGBotError should be a subclass of the built-in Exception."""
        assert issubclass(RAGBotError, Exception)

    def test_ragbot_error_instantiation(self):
        """RAGBotError can be instantiated and raised."""
        error = RAGBotError("something went wrong")

        with pytest.raises(RAGBotError):
            raise error


class TestExceptionHierarchy:
    """Ensure all custom exceptions inherit from RAGBotError."""

    @pytest.mark.parametrize(
        "exc_class",
        [
            DocumentLoadError,
            UnsupportedFileTypeError,
            VectorStoreError,
            LLMError,
            ConfigurationError,
        ],
        ids=[
            "DocumentLoadError",
            "UnsupportedFileTypeError",
            "VectorStoreError",
            "LLMError",
            "ConfigurationError",
        ],
    )
    def test_exception_inherits_from_ragbot_error(self, exc_class: type):
        """Each custom exception should be a subclass of RAGBotError."""
        assert issubclass(exc_class, RAGBotError), (
            f"{exc_class.__name__} does not inherit from RAGBotError"
        )

    @pytest.mark.parametrize(
        "exc_class",
        [
            DocumentLoadError,
            UnsupportedFileTypeError,
            VectorStoreError,
            LLMError,
            ConfigurationError,
        ],
        ids=[
            "DocumentLoadError",
            "UnsupportedFileTypeError",
            "VectorStoreError",
            "LLMError",
            "ConfigurationError",
        ],
    )
    def test_exception_is_catchable_as_ragbot_error(self, exc_class: type):
        """Raising a specific exception should be catchable via RAGBotError."""
        with pytest.raises(RAGBotError):
            raise exc_class("test message")


class TestExceptionMessages:
    """Verify that custom exceptions correctly carry their message string."""

    @pytest.mark.parametrize(
        ("exc_class", "message"),
        [
            (RAGBotError, "General RAGBot failure"),
            (DocumentLoadError, "Failed to load document: file.pdf"),
            (UnsupportedFileTypeError, "Unsupported file type: .xyz"),
            (VectorStoreError, "Vector store initialisation failed"),
            (LLMError, "LLM inference error: timeout"),
            (ConfigurationError, "Missing required configuration: API key"),
        ],
        ids=[
            "RAGBotError",
            "DocumentLoadError",
            "UnsupportedFileTypeError",
            "VectorStoreError",
            "LLMError",
            "ConfigurationError",
        ],
    )
    def test_exception_message(self, exc_class: type, message: str):
        """The exception's string representation should contain the original message."""
        error = exc_class(message)

        assert str(error) == message
        assert message in repr(error)

    def test_exception_args(self):
        """Exception.args should contain the message as the first element."""
        msg = "Something broke"
        error = RAGBotError(msg)

        assert error.args == (msg,)
