# RAG_Embedder — Ask My Docs

A production-style **Retrieval-Augmented Generation (RAG)** system.
Point it at any folder of documents and ask questions in plain English.
It answers **only from your documents** and cites every claim by source number.

```
Q: Who created Python?
A: Python was created by Guido van Rossum [1].
Sources: python.txt, fastapi.txt
```

---

## Architecture

```
HTTP Request
    │
    ▼
[api/main.py] FastAPI
    GET  /health   — liveness check
    POST /ingest   — load documents into the index
    POST /query    — ask a question, get a cited answer
    GET  /docs     — interactive Swagger UI
    │
    ▼
[src/pipeline.py] AskMyDocsPipeline
    │
    ├── INGEST
    │     loader.py       → reads .txt / .md / .pdf
    │     chunker.py      → splits into 512-char overlapping chunks
    │     embedder/       → converts chunks to 384-dim vectors
    │     vec_store       → FAISS index   (semantic search)
    │     bm25_store      → BM25 index    (keyword search)
    │
    └── ASK
          hybrid_retriever.py  → fuses FAISS + BM25 results via RRF
          reranker.py          → cross-encoder re-scores top candidates
          answer_chain.py      → calls LLM with citation-enforcing prompt
          → {"answer": "...[1]...", "sources": [...]}
```

---

## Quickstart

```bash
# 1. Clone
git clone https://github.com/JainRamyak/RAG_Embedder.git
cd RAG_Embedder

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Open .env and set GEMINI_API_KEY (free at aistudio.google.com)
# Leave LLM_PROVIDER=gemini

# 5. Start the API server
uvicorn api.main:app --reload
```

The API is now running at `http://localhost:8000`.
Open `http://localhost:8000/docs` for the interactive Swagger UI.

---

## Usage

### Via curl

```bash
# 1. Ingest your documents
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"directory": "docs/"}'

# Response:
# {"message":"Ingested documents from 'docs/'","chunks_ingested":23}

# 2. Ask a question
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Who created Python?"}'

# Response:
# {"answer":"Guido van Rossum [1].","sources":[{"source":"python.txt","score":0},...]}
```

### Via Python

```python
from src.pipeline import AskMyDocsPipeline

p = AskMyDocsPipeline()
p.ingest("docs/")
result = p.ask("Who created Python?")

print(result["answer"])
# → Guido van Rossum [1].

print([s["source"] for s in result["sources"]])
# → ['python.txt', 'fastapi.txt', ...]
```

---

## Supported LLM Providers

Switch provider by changing **one line** in `.env`:

```
LLM_PROVIDER=gemini      # default — free
LLM_PROVIDER=mistral     # free tier available
LLM_PROVIDER=openai      # requires paid credits
LLM_PROVIDER=anthropic   # requires paid credits
```

| Provider  | Model                    | Cost         | Get API Key                  |
|-----------|--------------------------|--------------|------------------------------|
| `gemini`  | gemini-2.5-flash         | Free         | aistudio.google.com          |
| `mistral` | mistral-small-latest     | Free tier    | console.mistral.ai           |
| `openai`  | gpt-4o-mini              | Paid         | platform.openai.com          |
| `anthropic` | claude-sonnet-4-...    | Paid         | console.anthropic.com        |

---

## Environment Variables

Copy `.env.example` to `.env` and fill in the values you need.

| Variable            | Required          | Description                              |
|---------------------|-------------------|------------------------------------------|
| `EMBEDDER_TYPE`     | No (default: local) | `local`, `openai`, or `gemini`         |
| `LOCAL_MODEL_NAME`  | No                | HuggingFace model for local embeddings   |
| `LLM_PROVIDER`      | No (default: gemini) | Which LLM to use for answers          |
| `GEMINI_API_KEY`    | If using gemini   | Google AI Studio API key                 |
| `MISTRAL_API_KEY`   | If using mistral  | Mistral AI API key                       |
| `OPENAI_API_KEY`    | If using openai   | OpenAI API key                           |
| `ANTHROPIC_API_KEY` | If using anthropic | Anthropic API key                       |
| `CHUNK_SIZE`        | No (default: 512) | Characters per document chunk            |
| `CHUNK_OVERLAP`     | No (default: 50)  | Overlap between chunks                   |
| `TOP_K_RETRIEVAL`   | No (default: 5)   | Number of chunks to retrieve per query   |
| `LOG_LEVEL`         | No (default: INFO) | `DEBUG`, `INFO`, `WARNING`, `ERROR`    |

---

## Supported Document Formats

| Format | Extension | Notes                        |
|--------|-----------|------------------------------|
| Plain text | `.txt` | UTF-8 encoded              |
| Markdown   | `.md`  | Treated as plain text       |
| PDF        | `.pdf` | Extracted page by page via PyMuPDF |

Place documents in any directory and pass the path to `/ingest`.

---

## Project Structure

```
RAG_Embedder/
├── .env.example              # environment variable template
├── config.py                 # central settings (reads .env)
├── requirements.txt
│
├── embedder/                 # text → vector conversion
│   ├── local_embedder.py     # HuggingFace all-MiniLM-L6-v2 (default)
│   └── api_embedder.py       # OpenAI / Gemini cloud embeddings
│
├── src/
│   ├── pipeline.py           # main orchestrator: ingest() + ask()
│   ├── ingestion/
│   │   ├── loader.py         # file loading (.txt, .md, .pdf)
│   │   └── chunker.py        # text splitting
│   ├── storage/
│   │   ├── vector_store.py   # FAISS vector index
│   │   └── bm25_store.py     # BM25 keyword index
│   ├── retrieval/
│   │   ├── hybrid_retriever.py  # RRF fusion of both indexes
│   │   └── reranker.py          # cross-encoder reranking
│   └── generation/
│       └── answer_chain.py      # LLM call with citation prompt
│
├── api/
│   └── main.py               # FastAPI: /health /ingest /query
│
├── tests/
│   ├── test_embedder.py      # 12 unit tests
│   └── test_pipeline.py      # 11 integration tests
│
└── docs/                     # sample documents for testing
```

---

## Running Tests

```bash
# All tests
pytest tests/ -v

# Expected output:
# tests/test_embedder.py  — 12 passed
# tests/test_pipeline.py  — 11 passed
# Total: 23 passed, 0 failed
```

> **Note:** Pipeline tests call the LLM. Make sure `LLM_PROVIDER` is set
> to a working provider with a valid key before running.
> If Gemini returns a 503, wait a moment and re-run — it's a temporary overload.

---

## Key Design Decisions

- **Hybrid search**: Combines FAISS (semantic) and BM25 (keyword) via
  Reciprocal Rank Fusion — catches both conceptually similar and
  exact-match results that either index alone would miss.

- **Cross-encoder reranking**: After retrieval, a `cross-encoder/ms-marco-MiniLM-L-6-v2`
  model re-scores candidates by reading the question and each chunk
  together, significantly improving answer quality.

- **Citation enforcement**: The LLM system prompt requires every factual
  claim to cite a source number in brackets `[1]`. The pipeline returns
  both the answer and the source list so the citation can be verified.

- **Provider-agnostic**: All LLM calls go through `answer_chain.py`.
  Adding a new provider means adding one function — no other files change.

---

## License

MIT