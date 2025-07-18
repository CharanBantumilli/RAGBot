# RAGBot

**RAGBot** is a sophisticated, Python-powered chatbot leveraging Retrieval-Augmented Generation (RAG) to deliver context-rich, accurate answers from your own documents. Powered by LangChain, HuggingFace, and seamless file handling, it represents a modern approach to intelligent, document-aware conversational AI.

---

## ✨ Key Features

- **Plug & Play Document Ingestion**  
  Effortlessly upload PDFs, TXT, CSV, or web URLs; RAGBot handles them all via modular loaders.
- **Robust Chunking & Embedding**  
  Documents are intelligently split and embedded using state-of-the-art multilingual models (`intfloat/multilingual-e5-small`).
- **Efficient Vector Search**  
  Uses Chroma for blazing fast document retrieval and similarity search.
- **Powerful LLM Integration**  
  Answers are generated with DeepSeek R1 Distill Qwen 1.5B, wrapped via HuggingFace and LangChain for flexibility.
- **Elegant Chat UI**  
  Built with Gradio for a beautiful, intuitive user experience—just upload and start chatting!

---

## 🏗️ Architecture Overview

RAGBot combines modern NLP and retrieval in a streamlined pipeline:

1. **User uploads a file or URL via chat UI.**
2. **Content is loaded and split into chunks.**
3. **Chunks are embedded into vectors (multilingual-e5-small).**
4. **Vectors are stored in Chroma DB for semantic search.**
5. **When a question is asked, relevant chunks are retrieved.**
6. **A prompt is built with context and query, sent to the LLM (DeepSeek R1 Distill Qwen).**
7. **Answer is returned in the Gradio chat interface.**

Simple. Powerful. Document-grounded conversations.

---

## 🚀 Quickstart

### **1. Installation**

```bash
git clone https://github.com/CharanBantumilli/RAGBot.git
cd RAGBot
pip install -r requirements.txt
```

### **2. API Token Setup**
Set your `HUGGINGFACEHUB_API_TOKEN` as an environment variable for access to models.

### **3. Run the App**

```bash
python RAGBot/Main.py
```
Open the Gradio interface and start your document-grounded chat!

---

## 🧩 Usage

- **Upload your data**: Accepts `.pdf`, `.txt`, `.csv`, or URLs.
- **Ask questions**: Type queries—RAGBot retrieves context and generates concise, relevant answers.
- **Explore & Integrate**: Extend or adapt the pipeline for your own RAG workflows.

---

## 🛠️ Core Modules

- `data_loader(file)`: Loads and parses multiple file types.
- `data_splitter(data)`: Chunks documents for optimal embedding.
- `vector_database(docs)`: Creates a vector store for semantic search.
- `retriver(files)`: Assembles the retrieval pipeline.
- `chatbot(file, query)`: Orchestrates retrieval and generation for user queries.
- `chat_interface(...)`: Manages the chat flow in the Gradio UI.

---

## 🖥️ UI Example

Gradio powers a simple, elegant chat experience:
- Upload a file.
- Ask a question.
- Get instant, context-aware answers!

---

## 📂 File Structure

```
RAGBot/
├── Main.py
└── requirements.txt
```

<p align="center">
  <b>RAGBot &mdash; Where your documents meet intelligent conversation.</b>
</p>
