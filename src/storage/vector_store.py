import json
import os
import numpy as np
from typing import List, Tuple, Optional


class SimpleVectorStore:
    """
    In-memory vector store with similarity search and disk persistence.

    Usage:
        store = SimpleVectorStore()
        store.add("doc1", "The cat sat on the mat", embedder.embed_text("The cat sat on the mat"))
        results = store.search(embedder.embed_text("kitten on rug"), top_k=3)
    """

    def __init__(self):
        self.ids: List[str] = []
        self.texts: List[str] = []
        self.vectors: List[List[float]] = []

    def add(self, doc_id: str, text: str, vector: List[float]):
        if doc_id in self.ids:
            raise ValueError(f"ID '{doc_id}' already exists. Use delete() first.")
        self.ids.append(doc_id)
        self.texts.append(text)
        self.vectors.append(vector)

    def delete(self, doc_id: str):
        if doc_id not in self.ids:
            raise KeyError(f"ID '{doc_id}' not found.")
        idx = self.ids.index(doc_id)
        self.ids.pop(idx)
        self.texts.pop(idx)
        self.vectors.pop(idx)

    def search(self, query_vector: List[float], top_k: int = 3) -> List[Tuple[str, str, float]]:
        if not self.vectors:
            return []

        query = np.array(query_vector)
        scores = []

        for doc_id, text, vec in zip(self.ids, self.texts, self.vectors):
            vec = np.array(vec)
            score = float(
                np.dot(query, vec) /
                (np.linalg.norm(query) * np.linalg.norm(vec))
            )
            scores.append((doc_id, text, score))

        scores.sort(key=lambda x: x[2], reverse=True)
        return scores[:top_k]

    def save(self, path: str):
        data = {
            "ids": self.ids,
            "texts": self.texts,
            "vectors": self.vectors,
        }
        with open(path, "w") as f:
            json.dump(data, f)
        print(f"[VectorStore] Saved {len(self.ids)} documents → {path}")

    def load(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"No store found at '{path}'")
        with open(path, "r") as f:
            data = json.load(f)
        self.ids = data["ids"]
        self.texts = data["texts"]
        self.vectors = data["vectors"]
        print(f"[VectorStore] Loaded {len(self.ids)} documents ← {path}")

    def __len__(self):
        return len(self.ids)

    def __repr__(self):
        return f"SimpleVectorStore({len(self.ids)} documents)"