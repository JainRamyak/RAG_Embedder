import os
from dotenv import load_dotenv

from .base import BaseEmbedder
from .local_embedder import LocalEmbedder
from .api_embedder import APIEmbedder

load_dotenv()


def get_embedder() -> BaseEmbedder:
    """
    Factory function — the only import your other projects need.

    Reads EMBEDDER_TYPE from .env and returns the right embedder.
    Switching backends = changing one line in .env. No code changes.

    Usage:
        from embedder import get_embedder
        embedder = get_embedder()
        vector = embedder.embed_text("hello world")
    """
    embedder_type = os.getenv("EMBEDDER_TYPE", "local").lower().strip()

    if embedder_type == "local":
        model_name = os.getenv("LOCAL_MODEL_NAME", "all-MiniLM-L6-v2")
        return LocalEmbedder(model_name=model_name)

    elif embedder_type in ("openai", "gemini"):
        return APIEmbedder(provider=embedder_type)

    else:
        raise ValueError(
            f"Unknown EMBEDDER_TYPE='{embedder_type}' in .env.\n"
            f"Valid options: local, openai, gemini"
        )


__all__ = [
    "get_embedder",
    "BaseEmbedder",
    "LocalEmbedder",
    "APIEmbedder",
]