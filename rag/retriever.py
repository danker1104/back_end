import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from config import TOP_K
from embeddings.embed_chunks import ChunkEmbedder
from vector_db.faiss_store import FaissVectorStore

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class Retriever:
    def __init__(self, top_k: int = TOP_K, store_path: Optional[str] = None, metadata_path: Optional[str] = None):
        self.top_k = top_k
        self.store = FaissVectorStore(index_path=store_path, metadata_path=metadata_path)
        self.embedder = None
        try:
            self.embedder = ChunkEmbedder()
        except Exception as exc:  # pragma: no cover - environment dependent
            logger.warning("Embedding model unavailable, using keyword fallback search: %s", exc)
        self._load_store()

    def _load_store(self) -> None:
        try:
            self.store.load()
        except Exception as exc:  # pragma: no cover - environment dependent
            logger.warning("FAISS index unavailable, using metadata-only search: %s", exc)
            self.store.index = None
            self.store.metadata = self._load_metadata()

    def _load_metadata(self) -> List[Dict[str, Any]]:
        if not self.store.metadata_path.exists():
            return []

        try:
            return json.loads(self.store.metadata_path.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - environment dependent
            logger.warning("Could not read metadata file: %s", exc)
            return []

    def _embed_query(self, query: str) -> np.ndarray:
        embedding = self.embedder.encode([query])[0]
        return np.asarray(embedding, dtype=np.float32)

    def _rewrite_query(self, query: str) -> str:
        text = query.strip()
        if not text:
            return ""

        normalized = re.sub(r"[^가-힣a-z0-9\s]", " ", text.lower()).strip()
        normalized = re.sub(r"\s+", " ", normalized)

        keyword_map = {
            "보험사기 를 당했어": ["보험사기", "보험사기방지 특별법", "보험금 편취", "사기죄", "기망", "보험"],
            "보험사기를 당했어": ["보험사기", "보험사기방지 특별법", "보험금 편취", "사기죄", "기망", "보험"],
            "폭행당했어": ["폭행", "상해", "형법 폭행죄", "형법 상해죄"],
            "임금 을 못 받았어": ["임금체불", "근로기준법", "체불임금"],
            "임금을 못 받았어": ["임금체불", "근로기준법", "체불임금"],
        }

        if normalized in keyword_map:
            return " ".join(keyword_map[normalized])

        keywords: List[str] = []
        for token in re.findall(r"[가-힣a-z0-9]+", normalized):
            if len(token) <= 1:
                continue
            if token in {"당했어", "당했냐", "못", "받았어", "했어", "했냐", "해", "해줘", "해줘요", "좀", "내", "나"}:
                continue
            keywords.append(token)

        if not keywords:
            return text

        return " ".join(keywords)

    def _keyword_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        tokens = [token for token in re.findall(r"[가-힣a-z0-9]+", query.lower()) if len(token) > 1]
        if not tokens:
            return []

        scored_results: List[Tuple[float, Dict[str, Any]]] = []
        for metadata in self.store.metadata:
            text = metadata.get("text", "")
            if not text:
                continue

            text_lower = text.lower()
            score = 0.0
            for token in tokens:
                if token in text_lower:
                    score += 2.0
                if token in str(metadata.get("casenames", "")).lower():
                    score += 3.0
                if token in str(metadata.get("statute_name", "")).lower():
                    score += 3.0

            if score > 0:
                scored_results.append((score, metadata))

        scored_results.sort(key=lambda item: (item[0], str(item[1].get("doc_id", ""))), reverse=True)
        return [
            {
                "score": float(score),
                "metadata": metadata,
                "text": metadata.get("text", ""),
            }
            for score, metadata in scored_results[:top_k]
        ]

    def search(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        if top_k is None:
            top_k = self.top_k

        rewritten_query = self._rewrite_query(query)

        if self.store.index is None or self.embedder is None:
            bm25_results = self._keyword_search(rewritten_query, min(top_k, len(self.store.metadata)))
            return bm25_results

        bm25_results = self._keyword_search(rewritten_query, min(top_k, len(self.store.metadata)))
        query_vector = self._embed_query(rewritten_query)
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
