import numpy as np
from typing import List, Tuple


class SimpleVectorStore:
    """
    Stores text + embeddings and allows similarity search.
    """

    def __init__(self):
        self.texts = []
        self.vectors = []

    def add(self, text: str, vector: List[float]):
        self.texts.append(text)
        self.vectors.append(vector)

    def search(self, query_vector: List[float], top_k: int = 3) -> List[Tuple[str, float]]:
        scores = []

        query = np.array(query_vector)

        for text, vec in zip(self.texts, self.vectors):
            vec = np.array(vec)
            score = float(np.dot(query, vec) / (np.linalg.norm(query) * np.linalg.norm(vec)))
            scores.append((text, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]