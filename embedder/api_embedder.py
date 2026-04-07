import os
from typing import List
from .base import BaseEmbedder


class APIEmbedder(BaseEmbedder):
    """
    Calls a cloud embedding API (OpenAI or Gemini).
    Requires the relevant API key set in .env.

    Providers:
      openai  → text-embedding-3-small  (1536 dims)
      gemini  → text-embedding-004      (768 dims)
    """

    PROVIDER_CONFIG = {
        "openai": {
            "model": "text-embedding-3-small",
            "dimension": 1536,
        },
        "gemini": {
            "model": "models/text-embedding-004",
            "dimension": 768,
        },
    }

    def __init__(self, provider: str = "openai"):
        if provider not in self.PROVIDER_CONFIG:
            raise ValueError(
                f"Unknown provider '{provider}'. "
                f"Choose from: {list(self.PROVIDER_CONFIG.keys())}"
            )
        self.provider = provider
        self.config = self.PROVIDER_CONFIG[provider]
        self._init_client()
        print(
            f"[APIEmbedder] Provider={provider} | "
            f"Model={self.config['model']} | "
            f"Dimension={self.dimension}"
        )

    def _init_client(self):
        if self.provider == "openai":
            from openai import OpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise EnvironmentError(
                    "OPENAI_API_KEY is not set in your .env file."
                )
            self.client = OpenAI(api_key=api_key)

        elif self.provider == "gemini":
            import google.generativeai as genai
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise EnvironmentError(
                    "GEMINI_API_KEY is not set in your .env file."
                )
            genai.configure(api_key=api_key)
            self.client = genai

    def embed_text(self, text: str) -> List[float]:
        if not text or not text.strip():
            raise ValueError("embed_text() received an empty string.")

        if self.provider == "openai":
            response = self.client.embeddings.create(
                input=text,
                model=self.config["model"]
            )
            return response.data[0].embedding

        elif self.provider == "gemini":
            result = self.client.embed_content(
                model=self.config["model"],
                content=text
            )
            return result["embedding"]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            raise ValueError("embed_batch() received an empty list.")
        return [self.embed_text(t) for t in texts]

    @property
    def dimension(self) -> int:
        return self.config["dimension"]