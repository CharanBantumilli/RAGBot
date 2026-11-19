"""LLM management for RAGBot.

Provides a manager that lazily loads the HuggingFace text-generation
pipeline **once** and caches it for subsequent queries, fixing the
original bug where the entire model was reloaded on every call.
"""

from __future__ import annotations

import logging
from typing import Optional

from ragbot.exceptions import LLMError

logger = logging.getLogger(__name__)


class LLMManager:
    """Manage the lifecycle of the HuggingFace text-generation LLM.

    The underlying transformer pipeline is constructed on the first call to
    :meth:`get_llm` and reused thereafter.

    Args:
        model_name: HuggingFace model identifier.
        max_new_tokens: Maximum number of new tokens to generate.
        temperature: Sampling temperature.
        device_map: Device placement strategy.
    """

    def __init__(
        self,
        model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        max_new_tokens: int = 200,
        temperature: float = 0.5,
        device_map: str = "auto",
    ) -> None:
        self._model_name = model_name
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature
        self._device_map = device_map
        self._llm = None  # cached HuggingFacePipeline

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _load_pipeline(self) -> None:
        """Load the transformer pipeline and wrap it in a LangChain LLM.

        Raises:
            LLMError: If the pipeline cannot be created.
        """
        from langchain_huggingface import HuggingFacePipeline
        from transformers import pipeline as hf_pipeline

        try:
            logger.info("Loading LLM '%s'…", self._model_name)
            pipe = hf_pipeline(
                task="text-generation",
                model=self._model_name,
                temperature=self._temperature,
                do_sample=False,
                device_map=self._device_map,
                max_new_tokens=self._max_new_tokens,
                return_full_text=False,
            )
            self._llm = HuggingFacePipeline(pipeline=pipe)
            logger.info("LLM loaded successfully.")
        except Exception as exc:
            raise LLMError(f"Failed to load LLM '{self._model_name}': {exc}") from exc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_llm(self):
        """Return the cached :class:`HuggingFacePipeline` LLM.

        The model is loaded lazily on the first invocation and reused for
        all subsequent calls.

        Returns:
            A :class:`HuggingFacePipeline` instance ready for inference.

        Raises:
            LLMError: If the model fails to load.
        """
        if self._llm is None:
            self._load_pipeline()
        return self._llm
