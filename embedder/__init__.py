"""
embedder package public API.

Usage:
    from embedder import get_embedder
    emb = get_embedder()
    vec = emb.embed_text("hello world")
"""
import logging
from .base import BaseEmbedder
from .local_embedder import LocalEmbedder
from .api_embedder import APIEmbedder
from config import settings

# NO load_dotenv() here — that lives in config.py only

logger = logging.getLogger(__name__)


def get_embedder() -> BaseEmbedder:
    etype = settings.embedder_type
    logger.info("Initialising embedder | type=%s", etype)

    if etype == "local":
        return LocalEmbedder(model_name=settings.local_model_name)
    elif etype in ("openai", "gemini"):
        return APIEmbedder(provider=etype)
    else:
        raise ValueError(
            f"Unknown EMBEDDER_TYPE='{etype}' in .env. "
            f"Valid: local, openai, gemini"
        )


__all__ = ["get_embedder", "BaseEmbedder", "LocalEmbedder", "APIEmbedder"]