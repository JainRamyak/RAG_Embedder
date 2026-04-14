import logging, pickle
from pathlib import Path
import numpy as np
import faiss

logger = logging.getLogger(__name__)

class FAISSVectorStore:
    def __init__(self, dimension: int):
        self.dim = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.chunks: list[dict] = []

    def add(self, chunks, vectors):
        arr = np.array(vectors, dtype="float32")
        faiss.normalize_L2(arr)
        self.index.add(arr)
        self.chunks.extend(chunks)

    def search(self, query_vec, k=5):
        arr = np.array([query_vec], dtype="float32")
        faiss.normalize_L2(arr)
        scores, idxs = self.index.search(arr, min(k, self.index.ntotal))
        return [(self.chunks[i], float(s))
                for s, i in zip(scores[0], idxs[0]) if i != -1]

    def save(self, path: str):
        p = Path(path); p.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(p/"index.faiss"))
        with open(p/"chunks.pkl","wb") as f: pickle.dump(self.chunks, f)

    def load(self, path: str):
        p = Path(path)
        self.index = faiss.read_index(str(p/"index.faiss"))
        with open(p/"chunks.pkl","rb") as f: self.chunks = pickle.load(f)

    @property
    def size(self): return self.index.ntotal