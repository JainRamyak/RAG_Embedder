"""
config.py — single source of truth for all configuration.

Import pattern everywhere in your project:
    from config import settings
    print(settings.embedder_type)
"""
import os
import logging
from dotenv import load_dotenv

load_dotenv()  # This is the ONLY place load_dotenv() is called


def _optional(key: str, default: str = "") -> str:
    return os.getenv(key, default).strip()


class _Settings:
    # Embedder
    embedder_type: str       = _optional("EMBEDDER_TYPE", "local").lower()
    local_model_name: str    = _optional("LOCAL_MODEL_NAME", "all-MiniLM-L6-v2")

    # API keys (empty string if not set — validated lazily when provider is used)
    openai_api_key: str      = _optional("OPENAI_API_KEY")
    gemini_api_key: str      = _optional("GEMINI_API_KEY")

    # Chunking
    chunk_size: int          = int(_optional("CHUNK_SIZE", "512"))
    chunk_overlap: int       = int(_optional("CHUNK_OVERLAP", "50"))

    # Retrieval
    top_k: int               = int(_optional("TOP_K_RETRIEVAL", "5"))

    # Logging
    log_level: str           = _optional("LOG_LEVEL", "INFO").upper()

    def validate_provider(self, provider: str) -> None:
        """Call this before using an API provider."""
        if provider == "openai" and not self.openai_api_key:
            raise EnvironmentError(
                "EMBEDDER_TYPE=openai requires OPENAI_API_KEY in .env"
            )
        if provider == "gemini" and not self.gemini_api_key:
            raise EnvironmentError(
                "EMBEDDER_TYPE=gemini requires GEMINI_API_KEY in .env"
            )


settings = _Settings()