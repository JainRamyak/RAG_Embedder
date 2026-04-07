

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

-git clone https://github.com/JainRamyak/RAG_Embedder.git
-cd RAG_Embedder

-python3 -m venv venv
-source venv/bin/activate

-pip install -r requirements.txt
-python main.py

First run downloads the embedding model (~90MB). After that it's instant.


---

## Configuration

Copy `.env.example` to `.env` and set one variable:


| EMBEDDER_TYPE | Key required | Model                  | Dims |
|---------------|-------------|------------------------|------|
| `local`       | No          | all-MiniLM-L6-v2       | 384  |
| `openai`      | Yes         | text-embedding-3-small | 1536 |
| `gemini`      | Yes         | text-embedding-004     | 768  |

---