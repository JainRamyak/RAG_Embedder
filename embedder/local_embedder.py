from typing import List
from sentence_transformers import SentenceTransformer
from .base import BaseEmbedder


class LocalEmbedder(BaseEmbedder):
    """
    Runs a HuggingFace sentence-transformer model locally.
    No API key needed. Works fully offline after first download.

    Recommended models:
      all-MiniLM-L6-v2    → fast, 384 dims, good for dev/testing
      all-mpnet-base-v2   → slower, 768 dims, better quality
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        print(f"[LocalEmbedder] Loading '{model_name}' ...")
        self.model = SentenceTransformer(model_name)
        self._model_name = model_name
        print(f"[LocalEmbedder] Ready. Dimension = {self.dimension}")

    def embed_text(self, text: str) -> List[float]:
        if not text or not text.strip():
            raise ValueError("embed_text() received an empty string.")
        vector = self.model.encode(text, convert_to_numpy=True)
        return vector.tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            raise ValueError("embed_batch() received an empty list.")
        vectors = self.model.encode(texts, convert_to_numpy=True)
        return vectors.tolist()

    @property
    def dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()