"""RAG chain orchestration for RAGBot.

Composes the document loader, text processor, vector store, and LLM
into a single :class:`RAGChain` that supports document ingestion and
question-answering.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from ragbot.config import Settings
from ragbot.document_loader import DocumentLoader
from ragbot.exceptions import RAGBotError
from ragbot.llm import LLMManager
from ragbot.text_processor import TextProcessor
from ragbot.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)

_PROMPT_TEMPLATE = """\
You are a helpful assistant. Use ONLY the context below to answer the
question.  If the context does not contain enough information, say
"I don't have enough information to answer that."

Context:
{context}

Question: {question}

Answer:"""

_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=_PROMPT_TEMPLATE,
)


class RAGChain:
    """End-to-end Retrieval-Augmented Generation chain.

    Args:
        settings: Application :class:`Settings` instance.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._loader = DocumentLoader()
        self._processor = TextProcessor(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        self._vector_store = VectorStoreManager(
            embedding_model_name=settings.embedding_model_name,
            persist_directory=settings.chroma_persist_directory,
        )
        self._llm_manager = LLMManager(
            model_name=settings.llm_model_name,
            max_new_tokens=settings.max_new_tokens,
            temperature=settings.temperature,
            device_map=settings.device_map,
        )
        self._current_file: Optional[str] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest(self, file_path: str) -> None:
        """Load, chunk, and index a document.

        If the same *file_path* has already been ingested, this is a
        no-op.  Call with a different path to replace the index.

        Args:
            file_path: Local path or URL of the document to ingest.

        Raises:
            DocumentLoadError: If loading fails.
            VectorStoreError: If indexing fails.
        """
        if file_path == self._current_file:
            logger.info("File '%s' already ingested — skipping.", file_path)
            return

        logger.info("Ingesting '%s'…", file_path)
        documents = self._loader.load(file_path)
        chunks = self._processor.split_documents(documents)
        self._vector_store.create_store(chunks)
        self._current_file = file_path
        logger.info("Ingestion complete (%d chunks indexed).", len(chunks))

    def query(self, question: str) -> str:
        """Answer *question* using the ingested document.

        Args:
            question: Natural-language question.

        Returns:
            A cleaned answer string.

        Raises:
            RAGBotError: If no document has been ingested or the LLM fails.
        """
        if self._current_file is None:
            raise RAGBotError("No document has been ingested yet.")

        logger.info("Querying: %s", question)
        retriever = self._vector_store.get_retriever()
        llm = self._llm_manager.get_llm()

        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": _PROMPT},
        )

        response: str = qa.run(question)
        cleaned = self._clean_response(response)
        logger.info("Answer generated (%d chars).", len(cleaned))
        return cleaned

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_response(text: str) -> str:
        """Strip non-printable characters from *text*.

        Args:
            text: Raw LLM output.

        Returns:
            Cleaned text containing only printable ASCII characters.
        """
        return re.sub(r"[^\x20-\x7E]+", " ", text).strip()
