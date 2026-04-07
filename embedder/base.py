from abc import ABC, abstractmethod
from typing import List
import numpy as np


class BaseEmbedder(ABC):
    """
    Abstract base class for all embedders.

    Every embedder (local or API-based) must implement:
      - embed_text()   → single string to vector
      - embed_batch()  → list of strings to list of vectors
      - dimension      → how many numbers per vector

    The similarity() method is implemented here once and
    inherited by all subclasses for free.
    """

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Convert a single string into a float vector."""
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Convert a list of strings into a list of float vectors."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Number of dimensions in each output vector."""
        pass

    def similarity(self, text_a: str, text_b: str) -> float:
        """
        Compute cosine similarity between two texts.
        Returns float between 0.0 (unrelated) and 1.0 (identical).
        Defined once here — all subclasses inherit it automatically.
        """
        vec_a = np.array(self.embed_text(text_a))
        vec_b = np.array(self.embed_text(text_b))
        return float(
            np.dot(vec_a, vec_b) /
            (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
        )