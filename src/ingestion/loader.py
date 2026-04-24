"""
src/ingestion/loader.py

Loads documents from a directory.
Supported formats: .txt, .md, .pdf
"""
import logging
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


def load_documents(directory: str) -> List[dict]:
    """
    Load all supported files from a directory.
    Returns list of {"text": str, "source": str}
    """
    docs = []
    path = Path(directory)

    for filepath in sorted(path.rglob("*")):
        if filepath.suffix == ".pdf":
            docs.extend(_load_pdf(filepath))
        elif filepath.suffix in (".txt", ".md"):
            docs.extend(_load_text(filepath))

    logger.info("Loaded %d pages from %s", len(docs), directory)
    return docs


def _load_text(filepath: Path) -> List[dict]:
    """Load a plain text or markdown file."""
    try:
        text = filepath.read_text(encoding="utf-8").strip()
        if not text:
            return []
        return [{"text": text, "source": filepath.name}]
    except Exception as e:
        logger.warning("Failed to load %s: %s", filepath, e)
        return []


def _load_pdf(filepath: Path) -> List[dict]:
    """Load a PDF file, one dict per page."""
    try:
        import pymupdf  # PyMuPDF
    except ImportError:
        raise ImportError("Run: pip install pymupdf")

    docs = []
    try:
        pdf = pymupdf.open(str(filepath))
        for page_num, page in enumerate(pdf, 1):
            text = page.get_text().strip()
            if text:
                docs.append({
                    "text": text,
                    "source": f"{filepath.name}#page{page_num}"
                })
        pdf.close()
        logger.info("PDF loaded: %s | %d pages", filepath.name, len(docs))
    except Exception as e:
        logger.warning("Failed to load PDF %s: %s", filepath, e)

    return docs