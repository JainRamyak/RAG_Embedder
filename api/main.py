"""
api/main.py

FastAPI REST layer for the RAG pipeline.

Endpoints:
    GET  /health        — liveness check
    POST /ingest        — ingest a directory of documents
    POST /query         — ask a question, get cited answer

Run with:
    uvicorn api.main:app --reload
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.pipeline import AskMyDocsPipeline

logger = logging.getLogger(__name__)

# ── Pipeline singleton ────────────────────────────────────────────
pipeline = AskMyDocsPipeline()


# ── Lifespan (startup/shutdown) ───────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("RAG API starting up")
    yield
    logger.info("RAG API shutting down")


# ── App ───────────────────────────────────────────────────────────
app = FastAPI(
    title="RAG Embedder API",
    description="Ask questions against your ingested documents.",
    version="0.1.0",
    lifespan=lifespan,
)


# ── Request / Response models ─────────────────────────────────────
class IngestRequest(BaseModel):
    directory: str = "docs/"


class IngestResponse(BaseModel):
    message: str
    chunks_ingested: int


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    sources: list


# ── Endpoints ─────────────────────────────────────────────────────
@app.get("/health")
def health():
    """Liveness check — returns ok if the API is running."""
    return {"status": "ok"}


@app.post("/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest):
    """
    Ingest all documents from a directory into the vector store.
    Call this once before querying.
    """
    try:
        count = pipeline.ingest(request.directory)
        return IngestResponse(
            message=f"Ingested documents from '{request.directory}'",
            chunks_ingested=count,
        )
    except Exception as e:
        logger.error("Ingest failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """
    Ask a question. Returns a cited answer and source list.
    You must call /ingest before calling /query.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    try:
        result = pipeline.ask(request.question)
        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"],
        )
    except Exception as e:
        logger.error("Query failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))