"""Shared pytest fixtures for the RAGBot test suite."""

from __future__ import annotations

import os
import textwrap
from pathlib import Path

import pytest


@pytest.fixture()
def sample_text_file(tmp_path: Path) -> Path:
    """Create a temporary .txt file with content about machine learning and AI.

    Returns the path to the created file.
    """
    content = textwrap.dedent("""\
        Machine Learning and Artificial Intelligence

        Machine learning is a subset of artificial intelligence that enables
        systems to learn and improve from experience without being explicitly
        programmed. Deep learning, a further subset, uses neural networks with
        many layers to model complex patterns in data.

        Common applications include natural language processing, computer vision,
        recommendation systems, and autonomous vehicles. Transfer learning allows
        models pre-trained on large datasets to be fine-tuned for specific tasks,
        dramatically reducing the data and compute requirements.

        Retrieval-Augmented Generation (RAG) combines the strengths of retrieval
        systems with generative language models, grounding responses in real
        documents to reduce hallucination and improve factual accuracy.
    """)
    file_path = tmp_path / "ml_overview.txt"
    file_path.write_text(content, encoding="utf-8")
    return file_path


@pytest.fixture()
def sample_csv_file(tmp_path: Path) -> Path:
    """Create a temporary .csv file with question-and-answer data.

    Returns the path to the created file.
    """
    content = textwrap.dedent("""\
        question,answer
        What is machine learning?,A subset of AI that learns from data.
        What is deep learning?,A subset of ML using multi-layer neural networks.
        What is RAG?,Retrieval-Augmented Generation combines retrieval with LLMs.
        What is a vector store?,A database optimised for similarity search on embeddings.
    """)
    file_path = tmp_path / "qa_data.csv"
    file_path.write_text(content, encoding="utf-8")
    return file_path


@pytest.fixture(autouse=True)
def set_hf_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """Automatically set a dummy HuggingFace Hub API token for every test.

    This prevents any test from accidentally hitting the real API and ensures
    code paths that read the token find a value.
    """
    monkeypatch.setenv("HUGGINGFACEHUB_API_TOKEN", "test_token")


@pytest.fixture()
def mock_settings():
    """Return a ``Settings`` instance populated with safe test defaults.

    Heavy model names are replaced with lightweight placeholders so that no
    test accidentally triggers a download.
    """
    from ragbot.config import Settings

    return Settings(
        embedding_model="intfloat/multilingual-e5-small",
        llm_model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        chunk_size=500,
        chunk_overlap=50,
        max_new_tokens=200,
        temperature=0.5,
    )
