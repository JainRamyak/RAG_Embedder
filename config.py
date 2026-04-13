import os
from dotenv import load_dotenv
load_dotenv()

def _opt(key, default=""):
    return os.getenv(key, default).strip()

class _Settings:
    embedder_type    = _opt("EMBEDDER_TYPE", "local").lower()
    local_model_name = _opt("LOCAL_MODEL_NAME", "all-MiniLM-L6-v2")
    openai_api_key   = _opt("OPENAI_API_KEY")
    gemini_api_key   = _opt("GEMINI_API_KEY")
    chunk_size       = int(_opt("CHUNK_SIZE", "512"))
    chunk_overlap    = int(_opt("CHUNK_OVERLAP", "50"))
    top_k            = int(_opt("TOP_K_RETRIEVAL", "5"))
    log_level        = _opt("LOG_LEVEL", "INFO").upper()
    llm_model        = _opt("LLM_MODEL", "gpt-4o-mini")

    def validate_provider(self, provider):
        if provider == "openai" and not self.openai_api_key:
            raise EnvironmentError("OPENAI_API_KEY not set in .env")
        if provider == "gemini" and not self.gemini_api_key:
            raise EnvironmentError("GEMINI_API_KEY not set in .env")

settings = _Settings()