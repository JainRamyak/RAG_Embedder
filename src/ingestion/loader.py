import logging
from pathlib import Path
import pymupdf  # pip install pymupdf

logger = logging.getLogger(__name__)

def load_file(path: str) -> list[dict]:
    p = Path(path)
    if p.suffix == ".pdf":
        doc = pymupdf.open(str(p))
        return [{"text": page.get_text(), "source": p.name, "page": i+1}
                for i, page in enumerate(doc) if page.get_text().strip()]
    else:
        text = p.read_text(encoding="utf-8")
        return [{"text": text, "source": p.name, "page": 1}]

def load_directory(dir_path: str, exts=None) -> list[dict]:
    exts = exts or [".txt", ".md", ".pdf"]
    docs = []
    for path in Path(dir_path).rglob("*"):
        if path.suffix.lower() in exts:
            docs.extend(load_file(str(path)))
    logger.info("Loaded %d pages from %s", len(docs), dir_path)
    return docs