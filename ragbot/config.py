"""Application configuration for RAGBot.

Settings are loaded once from environment variables (with an optional
``.env`` file) and cached as a module-level singleton via
:func:`get_settings`.  Every setting can be overridden by exporting an
environment variable with the ``RAGBOT_`` prefix — for example
``RAGBOT_CHUNK_SIZE=1000``.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

try:
    from dotenv import load_dotenv

    # Load .env file if present (does not overwrite existing env vars).
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)


def _env(name: str, default: str | None = None) -> str | None:
    """Return the ``RAGBOT_<name>`` env var, falling back to *default*."""
    return os.environ.get(f"RAGBOT_{name}", default)


@dataclass
class Settings:
    """Central configuration dataclass.

    Attributes:
        hf_token: HuggingFace Hub API token.
        embedding_model_name: Sentence-transformer model for embeddings.
        llm_model_name: Causal-LM model used for text generation.
        chunk_size: Maximum number of characters per text chunk.
        chunk_overlap: Overlap between consecutive chunks.
        max_new_tokens: Maximum tokens the LLM may generate per query.
        temperature: Sampling temperature for the LLM.
        device_map: Device placement strategy (``"auto"`` by default).
        chroma_persist_directory: Optional path to persist the Chroma DB.
    """

    hf_token: str = ""
    embedding_model_name: str = "intfloat/multilingual-e5-small"
    llm_model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    chunk_size: int = 500
    chunk_overlap: int = 50
    max_new_tokens: int = 200
    temperature: float = 0.5
    device_map: str = "auto"
    chroma_persist_directory: Optional[str] = field(default=None)

    @property
    def embedding_model(self) -> str:
        """Alias for embedding_model_name."""
        return self.embedding_model_name

    @property
    def llm_model(self) -> str:
        """Alias for llm_model_name."""
        return self.llm_model_name

    def validate(self) -> None:
        """Raise :class:`ConfigurationError` if critical values are missing."""
        if not self.hf_token:
            raise ConfigurationError(
                "HuggingFace API token is required. Set the "
                "HUGGINGFACEHUB_API_TOKEN or RAGBOT_HF_TOKEN environment variable."
            )
        if self.chunk_size <= 0:
            raise ConfigurationError("chunk_size must be a positive integer.")
        if self.chunk_overlap < 0:
            raise ConfigurationError("chunk_overlap must be non-negative.")


def _build_settings() -> Settings:
    """Construct a :class:`Settings` instance from environment variables."""
    settings = Settings(
        hf_token=(
            _env("HF_TOKEN")
            or os.environ.get("HUGGINGFACEHUB_API_TOKEN", "")
        ),
        embedding_model_name=_env(
            "EMBEDDING_MODEL_NAME", "intfloat/multilingual-e5-small"
        ),  # type: ignore[arg-type]
        llm_model_name=_env(
            "LLM_MODEL_NAME", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
        ),  # type: ignore[arg-type]
        chunk_size=int(_env("CHUNK_SIZE", "500")),  # type: ignore[arg-type]
        chunk_overlap=int(_env("CHUNK_OVERLAP", "50")),  # type: ignore[arg-type]
        max_new_tokens=int(_env("MAX_NEW_TOKENS", "200")),  # type: ignore[arg-type]
        temperature=float(_env("TEMPERATURE", "0.5")),  # type: ignore[arg-type]
        device_map=_env("DEVICE_MAP", "auto"),  # type: ignore[arg-type]
        chroma_persist_directory=_env("CHROMA_PERSIST_DIRECTORY"),
    )
    logger.info("Settings loaded (embedding=%s, llm=%s).",
                settings.embedding_model_name, settings.llm_model_name)
    return settings


_settings: Settings | None = None


def get_settings() -> Settings:
    """Return the cached application settings singleton.

    On the first call the settings are built from environment variables and
    validated.  Subsequent calls return the same instance.

    Returns:
        The validated :class:`Settings` singleton.

    Raises:
        ConfigurationError: If required configuration is missing.
    """
    global _settings  # noqa: PLW0603
    if _settings is None:
        _settings = _build_settings()
        _settings.validate()
    return _settings
