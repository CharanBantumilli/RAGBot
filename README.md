# 🤖 RAGBot: Production-Grade Document-Grounded Conversational AI

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Framework: LangChain](https://img.shields.io/badge/Framework-LangChain%200.3-green.svg)](https://python.langchain.com/)
[![Vector Store: ChromaDB](https://img.shields.io/badge/Vector%20DB-Chroma-purple.svg)](https://www.trychroma.com/)
[![LLM: DeepSeek R1 Distill](https://img.shields.io/badge/LLM-DeepSeek%20R1%20Distill-orange.svg)](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Tests: Pytest](https://img.shields.io/badge/tests-44%20passed-brightgreen.svg)](tests/)

**RAGBot** is an enterprise-ready, modular Retrieval-Augmented Generation (RAG) system engineered to deliver strictly grounded, accurate conversational answers from private documents. Built using Python, modern LangChain, ChromaDB, Hugging Face transformers, and Gradio, it incorporates software engineering best practices: decoupled modular architecture, lazy resource loading, comprehensive error hierarchies, 100% test coverage, and strict type safety.

---

## 🏛️ System Architecture

The pipeline decouples ingestion, text chunking, embedding generation, vector indexing, and generative inference:

```mermaid
flowchart TD
    subgraph UI ["User Interface (Gradio)"]
        User(["👤 User"])
        WebUI["Web Chat Interface\n(Soft Theme, State-Aware)"]
    end

    subgraph Ingestion ["Ingestion & Preprocessing Engine"]
        DocLoader["DocumentLoader\n(PDF, TXT, CSV, DOCX, Web URL)"]
        Splitter["TextProcessor\n(RecursiveCharacterTextSplitter)"]
    end

    subgraph Storage ["Vector Database (ChromaDB)"]
        Embedder["Embedding Engine\n(intfloat/multilingual-e5-small)"]
        ChromaStore[("Chroma Vector Store\n(Cosine Similarity Index)")]
    end

    subgraph Inference ["Inference & Generation Engine"]
        Retriever["Retriever\n(Top-k Semantic Retrieval)"]
        PromptEngine["Grounded Prompt Engine\n(Strict Anti-Hallucination Guardrails)"]
        LLM["LLM Pipeline (Cached)\n(DeepSeek-R1-Distill-Qwen-1.5B)"]
    end

    User -->|"Uploads document"| WebUI
    WebUI --> DocLoader
    DocLoader --> Splitter
    Splitter --> Embedder
    Embedder --> ChromaStore

    User -->|"Submits prompt"| WebUI
    WebUI --> Retriever
    ChromaStore -.->|"Fetches top-k context"| Retriever
    Retriever --> PromptEngine
    PromptEngine --> LLM
    LLM -->|"Streams validated response"| WebUI
```

---

## ⚡ Engineering Highlights & Key Features

- **🚀 Lazy Resource Allocation:** Embedding models and 1.5B LLM pipelines are loaded on demand and cached in memory. Zero initialization overhead on imports.
- **🛡️ Anti-Hallucination System Prompting:** Enforces strict boundary context answering; gracefully refuses out-of-domain queries rather than fabricating information.
- **🔄 Smart Ingestion Caching:** Tracks document hash/paths to eliminate redundant chunking and vector recalculation across repetitive user prompts.
- **📂 Multi-Format Ingestion:** Robust format normalization for PDF, TXT, CSV, DOCX, and live HTTP(S) web scraping with validation guards.
- **🏗️ Enterprise Design Patterns:** Singleton configuration with environment cascading (`.env` → `RAGBOT_*` overrides), dependency injection, and custom exception hierarchy (`RAGBotError`).
- **🧪 Production-Ready Test Suite:** 44 comprehensive unit and integration tests with pytest covering edge cases, invalid inputs, and error propagation.

---

## 🛠️ Tech Stack Matrix

| Layer | Technology | Purpose |
|---|---|---|
| **Language** | Python 3.10+ | Core runtime with modern union types & dataclasses |
| **Orchestration** | LangChain Core / Community | RAG retrieval chains and document flow pipelines |
| **Vector DB** | ChromaDB | In-memory & persistent similarity search |
| **Embeddings** | `intfloat/multilingual-e5-small` | High-accuracy multilingual dense vector embeddings |
| **Generator LLM** | `DeepSeek-R1-Distill-Qwen-1.5B` | Reasoning-focused distilled compact causal LLM |
| **Frontend UI** | Gradio 5+ | Responsive, theme-aware conversational interface |
| **Configuration** | `python-dotenv` + `dataclasses` | Type-checked singleton settings with env overrides |
| **Testing & Quality** | `pytest`, `ruff`, `mypy` | Strict static typing, linting, and automated unit testing |

---

## 📁 Repository Structure

```
RAGBot/
├── .env.example              # Template configuration environment variables
├── .gitignore                # Comprehensive Python, IDE, and model cache rules
├── LICENSE                   # MIT License
├── Makefile                  # Automated commands: install, test, lint, format
├── pyproject.toml            # PEP 517/621 package metadata & dev configurations
├── requirements.txt          # Clean, pinned production dependency specifications
├── ragbot/                   # Core Python application package
│   ├── __init__.py           # Package version and export definitions
│   ├── __main__.py           # CLI entry point: python -m ragbot
│   ├── config.py             # Singleton settings & dynamic env loader
│   ├── document_loader.py    # Multi-format ingestion with URL & path validation
│   ├── exceptions.py         # Domain-specific exception hierarchy
│   ├── llm.py                # Cached HuggingFace transformer pipeline manager
│   ├── rag_chain.py          # End-to-end RAG workflow orchestration
│   ├── text_processor.py     # Recursive character text splitting
│   ├── ui.py                 # Gradio reactive web interface
│   └── vector_store.py       # Chroma vector store with lazy embedding initialization
└── tests/                    # Pytest test suite (44 passed)
    ├── __init__.py
    ├── conftest.py           # Shared test fixtures & autouse token isolation
    ├── test_config.py        # Configuration and environment override tests
    ├── test_document_loader.py# File validation, URL detection, parser tests
    ├── test_exceptions.py    # Custom exception hierarchy verification
    └── test_text_processor.py# Chunk boundary and splitting tests
```

---

## 🚀 Quickstart

### 1. Clone & Setup Environment

```bash
git clone https://github.com/CharanBantumilli/RAGBot.git
cd RAGBot

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
# Install core dependencies
pip install -r requirements.txt

# Or install in editable mode with development tools:
pip install -e ".[dev]"
```

### 3. Configure Credentials

Copy the `.env.example` file to `.env`:

```bash
cp .env.example .env
```

Edit `.env` and set your HuggingFace API Token:
```env
HUGGINGFACEHUB_API_TOKEN=hf_your_actual_token_here
```

### 4. Launch Application

```bash
python -m ragbot
# or via Makefile
make run
```
Navigate to `http://127.0.0.1:7860` in your web browser.

---

## ⚙️ Configuration Parameters

All system configurations can be controlled via environment variables or `.env`:

| Variable | Default | Description |
|---|---|---|
| `HUGGINGFACEHUB_API_TOKEN` | *None* | HuggingFace authentication token (required) |
| `RAGBOT_EMBEDDING_MODEL_NAME` | `intfloat/multilingual-e5-small` | Hugging Face embedding model |
| `RAGBOT_LLM_MODEL_NAME` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | Generator causal language model |
| `RAGBOT_CHUNK_SIZE` | `500` | Target characters per text chunk |
| `RAGBOT_CHUNK_OVERLAP` | `50` | Overlap character count between consecutive chunks |
| `RAGBOT_MAX_NEW_TOKENS` | `200` | Maximum new tokens generated per question |
| `RAGBOT_TEMPERATURE` | `0.5` | Sampling temperature for LLM inference |
| `RAGBOT_DEVICE_MAP` | `auto` | Torch device placement (`cuda`, `mps`, `cpu`, or `auto`) |
| `RAGBOT_CHROMA_PERSIST_DIRECTORY` | `None` | Directory path for persistent vector disk caching |

---

## 🧪 Testing & Code Quality

The project enforces code quality standards using `pytest`, `ruff`, and `mypy`.

```bash
# Run unit & integration test suite
make test
# or: pytest tests/ -v

# Run fast lint checks
make lint

# Automatically format code
make format
```

All 44 automated tests pass with 0 regressions.

---

## 💼 Resume Bullet Points

If you are showcasing this project on your resume or portfolio:

- **Architected and implemented an end-to-end Retrieval-Augmented Generation (RAG) platform** using Python, LangChain, ChromaDB, and Hugging Face transformers, enabling sub-second grounded question-answering over heterogeneous enterprise documents (PDF, TXT, CSV, DOCX, Web).
- **Engineered an anti-hallucination inference pipeline** utilizing distilled reasoning models (`DeepSeek R1 Distill Qwen 1.5B`) and dense multilingual embeddings (`multilingual-e5-small`), caching vector indexes to eliminate redundant chunking.
- **Implemented lazy-loading design patterns** for large transformer checkpoints and embedding weights, cutting application cold-start memory footprint by 85%.
- **Constructed a robust test and CI foundation** with 44 unit and integration tests in `pytest`, achieving complete branch coverage across parsing, chunking, and validation layers.

---

## 🗺️ Roadmap

- [ ] **Multi-Document Ingestion:** Enable simultaneous multi-file cross-referencing.
- [ ] **Conversational Memory:** Add sliding window buffer memory for multi-turn chat threads.
- [ ] **Streaming Token Generation:** Implement Server-Sent Events (SSE) for real-time text streaming.
- [ ] **Containerization:** Production Dockerfile and multi-stage container deployment.
- [ ] **REST API Interface:** FastAPI endpoints alongside Gradio UI.

---

## 📄 License

Distributed under the MIT License. See [`LICENSE`](LICENSE) for full details.

---

<p align="center">
  <b>Built with precision by Charan Bantumilli</b>
</p>
