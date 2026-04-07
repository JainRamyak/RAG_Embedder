# RAG Embedder

A clean, swappable embedding layer for RAG pipelines.  
Switch between local HuggingFace models and cloud APIs (OpenAI, Gemini) by changing **one line** in `.env`. No code changes needed.

## Quick Start

\`\`\`bash
git clone <your-repo-url>
cd rag-embedder
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python main.py
\`\`\`

## Project Structure

\`\`\`
rag-embedder/
├── embedder/
│   ├── __init__.py        # Factory: get_embedder()
│   ├── base.py            # Abstract base class
│   ├── local_embedder.py  # HuggingFace (offline, free)
│   └── api_embedder.py    # OpenAI / Gemini
├── tests/
│   └── test_embedder.py   # 12 passing tests
├── notebooks/
│   └── embedder_demo.ipynb
├── main.py
├── requirements.txt
└── .env.example
\`\`\`

## Switching Backends

Edit `.env`:

| EMBEDDER_TYPE | Needs key? | Model                  | Dims |
|---------------|------------|------------------------|------|
| `local`       | No         | all-MiniLM-L6-v2       | 384  |
| `openai`      | Yes        | text-embedding-3-small | 1536 |
| `gemini`      | Yes        | text-embedding-004     | 768  |

## Run Tests

\`\`\`bash
pytest tests/ -v
\`\`\`

## Notebook

\`\`\`bash
cd notebooks
jupyter notebook embedder_demo.ipynb
\`\`\`
