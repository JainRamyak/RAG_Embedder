
# RAG Embedder

A swappable embedding layer for RAG pipelines.
Switch between local and cloud backends by changing one line in `.env` — no code changes needed.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)
![Tests](https://img.shields.io/badge/tests-12%20passing-brightgreen?style=flat-square)

---

## Overview

Most RAG projects hardcode their embedding logic. This one doesn't.

- Run fully **offline** with a local HuggingFace model — no API key, no cost
- Switch to **OpenAI or Gemini** by editing `.env`
- Built-in **vector store** — add, search, delete, and persist to disk
- Clean abstract base class — adding a new backend takes ~30 lines

---

## Quick Start

\`\`\`bash
git clone https://github.com/JainRamyak/RAG_Embedder.git
cd RAG_Embedder
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python main.py
\`\`\`

First run downloads the embedding model (~90MB). After that it's instant.

---

## Project Structure

\`\`\`
RAG_Embedder/
├── embedder/
│   ├── __init__.py        # get_embedder() factory — your only import
│   ├── base.py            # Abstract base with shared similarity()
│   ├── local_embedder.py  # HuggingFace, offline, free
│   ├── api_embedder.py    # OpenAI & Gemini backends
│   └── vector_store.py    # add · search · delete · save · load
├── tests/
│   └── test_embedder.py   # 12 tests: shape, quality, edge cases
├── notebooks/
│   └── embedder_demo.ipynb
├── main.py
├── requirements.txt
└── .env.example
\`\`\`

---

## Configuration

Copy `.env.example` to `.env` and set one variable:

\`\`\`bash
EMBEDDER_TYPE=local       # free, offline
# EMBEDDER_TYPE=openai    # requires OPENAI_API_KEY
# EMBEDDER_TYPE=gemini    # requires GEMINI_API_KEY
\`\`\`

| EMBEDDER_TYPE | Key required | Model                  | Dims |
|---------------|-------------|------------------------|------|
| `local`       | No          | all-MiniLM-L6-v2       | 384  |
| `openai`      | Yes         | text-embedding-3-small | 1536 |
| `gemini`      | Yes         | text-embedding-004     | 768  |

---

## Usage

\`\`\`python
from embedder import get_embedder
from embedder.vector_store import SimpleVectorStore

embedder = get_embedder()
store = SimpleVectorStore()

store.add("doc1", "The cat sat on the mat", embedder.embed_text("The cat sat on the mat"))
store.add("doc2", "Stock markets fell today", embedder.embed_text("Stock markets fell today"))

results = store.search(embedder.embed_text("a kitten on a rug"), top_k=3)
# [('doc1', 'The cat sat on the mat', 0.87), ...]

store.save("store.json")
store.load("store.json")
\`\`\`

---

## Tests

\`\`\`bash
pytest tests/ -v
# 12 passed in ~8s
\`\`\`

---

## Roadmap

- [x] Local embedder (HuggingFace)
- [x] API embedder (OpenAI + Gemini)
- [x] Vector store with persistence
- [ ] Chunking utilities for long documents
- [ ] FAISS backend for large-scale search
- [ ] FastAPI wrapper
