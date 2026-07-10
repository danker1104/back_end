import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from config import TOP_K
from embeddings.embed_chunks import ChunkEmbedder
from vector_db.faiss_store import FaissVectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Retriever:
    def __init__(self, top_k: int = TOP_K, store_path: Optional[str] = None, metadata_path: Optional[str] = None):
        self.top_k = top_k
        self.store = FaissVectorStore(index_path=store_path, metadata_path=metadata_path)
        self.embedder = ChunkEmbedder()
        self._load_store()

    def _load_store(self) -> None:
        self.store.load()

    def _embed_query(self, query: str) -> np.ndarray:
        embedding = self.embedder.encode([query])[0]
        return np.asarray(embedding, dtype=np.float32)

    def search(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        if self.store.index is None:
            raise ValueError("Vector store is not initialized")

        if top_k is None:
            top_k = self.top_k

        query_vector = self._embed_query(query)
        query_vector = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)

        distances, indices = self.store.index.search(query_vector, min(top_k, self.store.index.ntotal))
        results: List[Dict[str, Any]] = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue
            metadata = self.store.metadata[int(idx)] if int(idx) < len(self.store.metadata) else {}
            results.append(
                {
                    "score": float(distance),
                    "metadata": metadata,
                    "text": metadata.get("text", ""),
                }
            )

        return results


def load_retriever(top_k: int = TOP_K) -> Retriever:
    return Retriever(top_k=top_k)


if __name__ == "__main__":
    retriever = load_retriever(top_k=3)
    results = retriever.search("소유권이전등기")
    print(results[0])
