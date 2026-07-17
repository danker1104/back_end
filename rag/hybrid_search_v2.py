import json
import math
import pickle
import re
import zlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

import faiss
from config import TOP_K
from rag.query_rewrite_v2 import rewrite_query_v2

ROOT = Path(__file__).resolve().parent.parent
VECTOR_DB_V2_DIR = ROOT / "vector_db_v2"
BM25_V2_PATH = VECTOR_DB_V2_DIR / "bm25_v2.pkl"
VECTOR_INDEX_PATH = VECTOR_DB_V2_DIR / "faiss.index"
VECTOR_METADATA_PATH = VECTOR_DB_V2_DIR / "metadata.json"
PREPROCESS_DIR = ROOT / "preprocess_v2"
GLOSSARY_PATH = ROOT / "output" / "legal_glossary.json"
EMBEDDING_DIM = 256

TOKEN_PATTERN = re.compile(r"[가-힣]+|[A-Za-z0-9]+")


def _tokenize(text: str) -> List[str]:
    if not text:
        return []
    return [token.lower() for token in TOKEN_PATTERN.findall(text) if len(token) > 1]


def _hash_text_to_vector(text: str) -> np.ndarray:
    normalized = re.sub(r"\s+", " ", str(text or "").lower()).strip().encode("utf-8")
    digest = zlib.adler32(normalized) % (2**31 - 1)
    vector = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    for index in range(EMBEDDING_DIM):
        vector[index] = float((digest >> (index % 31)) & 1)
    return vector


def _unique_preserve_order(items: Iterable[str]) -> List[str]:
    seen: Set[str] = set()
    results: List[str] = []
    for item in items:
        if not item:
            continue
        if item in seen:
            continue
        seen.add(item)
        results.append(item)
    return results


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _load_pickle(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        with path.open("rb") as handle:
            return pickle.load(handle)
    except Exception:
        return None


def _load_document_summaries(preprocess_dir: Path) -> Dict[str, Dict[str, Any]]:
    documents: Dict[str, Dict[str, Any]] = {}
    for filename in ["law_summary.json", "precedent_summary.json", "legal_terms_summary.json"]:
        path = preprocess_dir / filename
        payload = _load_json(path)
        if not isinstance(payload, list):
            continue
        for item in payload:
            doc_id = str(item.get("doc_id", ""))
            if not doc_id:
                continue
            documents[doc_id] = {
                "doc_id": doc_id,
                "doc_type": str(item.get("doc_type", "")),
                "title": str(item.get("title", "")),
                "summary": str(item.get("summary", "")),
                "keywords": item.get("keywords", []) if isinstance(item.get("keywords"), list) else [],
                "related_terms": item.get("related_terms", []) if isinstance(item.get("related_terms"), list) else [],
                "citations": item.get("citations", []) if isinstance(item.get("citations"), list) else [],
            }
    return documents


def _load_vector_db(index_path: Path, metadata_path: Path) -> Tuple[Optional[faiss.Index], List[Dict[str, Any]]]:
    if not index_path.exists() or not metadata_path.exists():
        return None, []

    try:
        index = faiss.read_index(str(index_path))
        metadata = _load_json(metadata_path)
        if not isinstance(metadata, list):
            metadata = []
        return index, metadata
    except Exception:
        return None, []


def _normalize_search_tokens(tokens: List[str]) -> List[str]:
    normalized: List[str] = []
    for token in tokens:
        value = str(token or "").strip()
        if not value:
            continue
        if len(value) <= 1:
            continue
        normalized.append(value)
    return normalized


def _score_bm25(query_tokens: List[str], bm25_index: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not bm25_index or not query_tokens:
        return []

    tokenized_docs = bm25_index.get("tokenized_docs", [])
    idf = bm25_index.get("idf", {})
    doc_len = bm25_index.get("doc_len", [])
    avgdl = float(bm25_index.get("avgdl", 0.0))
    k1 = float(bm25_index.get("k1", 1.5))
    b = float(bm25_index.get("b", 0.75))
    titles = bm25_index.get("titles", [])
    doc_types = bm25_index.get("doc_types", [])
    doc_ids = bm25_index.get("doc_ids", [])

    query_tokens = [token.lower() for token in query_tokens if len(token) > 1]
    if not query_tokens:
        return []

    results: List[Dict[str, Any]] = []
    for idx, doc_tokens in enumerate(tokenized_docs):
        if idx >= len(doc_ids):
            continue
        frequencies = {}
        for token in doc_tokens:
            frequencies[token] = frequencies.get(token, 0) + 1

        score = 0.0
        for token in query_tokens:
            token_idf = float(idf.get(token, 0.0))
            f = frequencies.get(token, 0)
            if f <= 0:
                continue
            denom = f + k1 * (1.0 - b + b * (doc_len[idx] / avgdl if avgdl > 0 else 1.0))
            score += token_idf * ((f * (k1 + 1.0)) / denom)

        if score > 0:
            results.append(
                {
                    "doc_id": str(doc_ids[idx]),
                    "doc_type": str(doc_types[idx]) if idx < len(doc_types) else "",
                    "title": str(titles[idx]) if idx < len(titles) else "",
                    "score": float(score),
                }
            )

    results.sort(key=lambda item: item["score"], reverse=True)
    return results


def _vector_search(query_tokens: List[str], index: Optional[faiss.Index], metadata: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    if index is None or not metadata or not query_tokens:
        return []

    query_text = " ".join(query_tokens).strip()
    if not query_text:
        return []

    query_vector = _hash_text_to_vector(query_text).reshape(1, -1)
    distances, indices = index.search(query_vector.astype(np.float32), min(top_k, index.ntotal))

    results: List[Dict[str, Any]] = []
    for distance, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(metadata):
            continue
        item = metadata[int(idx)]
        results.append(
            {
                "doc_id": str(item.get("doc_id", "")),
                "doc_type": str(item.get("doc_type", "")),
                "title": str(item.get("title", "")),
                "summary": str(item.get("summary", "")),
                "distance": float(distance),
                "vector_similarity": 1.0 / (1.0 + float(distance)),
            }
        )

    return results


def _merge_and_rank(
    bm25_results: List[Dict[str, Any]],
    vector_results: List[Dict[str, Any]],
    doc_summaries: Dict[str, Dict[str, Any]],
    top_k: int,
) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}

    max_bm25 = max((item["score"] for item in bm25_results), default=0.0)
    max_vector = max((item["vector_similarity"] for item in vector_results), default=0.0)

    def _normalize(value: float, maximum: float) -> float:
        return float(value / maximum) if maximum > 0 else 0.0

    for item in bm25_results:
        doc_id = item["doc_id"]
        merged[doc_id] = {
            "doc_id": doc_id,
            "doc_type": item.get("doc_type", ""),
            "title": item.get("title", ""),
            "bm25_score": item["score"],
            "vector_similarity": 0.0,
            "summary": doc_summaries.get(doc_id, {}).get("summary", ""),
            "combined_score": 0.0,
        }

    for item in vector_results:
        doc_id = item["doc_id"]
        entry = merged.get(doc_id)
        if entry is None:
            merged[doc_id] = {
                "doc_id": doc_id,
                "doc_type": item.get("doc_type", ""),
                "title": item.get("title", ""),
                "bm25_score": 0.0,
                "vector_similarity": item["vector_similarity"],
                "summary": item.get("summary", ""),
                "combined_score": 0.0,
            }
        else:
            entry["vector_similarity"] = max(entry.get("vector_similarity", 0.0), item["vector_similarity"])
            if not entry.get("summary"):
                entry["summary"] = item.get("summary", "")

    for entry in merged.values():
        bm25_norm = _normalize(entry["bm25_score"], max_bm25)
        vector_norm = _normalize(entry["vector_similarity"], max_vector)
        if bm25_norm and vector_norm:
            entry["combined_score"] = (bm25_norm + vector_norm) / 2.0
        else:
            entry["combined_score"] = bm25_norm or vector_norm
        if not entry.get("summary"):
            entry["summary"] = doc_summaries.get(entry["doc_id"], {}).get("summary", "")

    ranked = sorted(merged.values(), key=lambda item: item["combined_score"], reverse=True)
    return ranked[:top_k]


class HybridSearchV2:
    def __init__(
        self,
        top_k: int = TOP_K,
        bm25_path: Optional[Path] = None,
        vector_dir: Optional[Path] = None,
        preprocess_dir: Optional[Path] = None,
        glossary_path: Optional[Path] = None,
    ):
        self.top_k = top_k
        self.bm25_path = Path(bm25_path or BM25_V2_PATH)
        self.vector_dir = Path(vector_dir or VECTOR_DB_V2_DIR)
        self.vector_index_path = self.vector_dir / "faiss.index"
        self.vector_metadata_path = self.vector_dir / "metadata.json"
        self.preprocess_dir = Path(preprocess_dir or PREPROCESS_DIR)
        self.glossary_path = Path(glossary_path or GLOSSARY_PATH)

        self.bm25_index = _load_pickle(self.bm25_path)
        self.vector_index, self.vector_metadata = _load_vector_db(self.vector_index_path, self.vector_metadata_path)
        self.doc_summaries = _load_document_summaries(self.preprocess_dir)

    def search(self, question: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        if top_k is None:
            top_k = self.top_k

        query_data = rewrite_query_v2(
            question,
            glossary_path=self.glossary_path,
            preprocess_dir=self.preprocess_dir,
        )
        query_tokens = _normalize_search_tokens(query_data.get("expanded_keywords", []) or query_data.get("keywords", []))

        bm25_results = _score_bm25(query_tokens, self.bm25_index) if self.bm25_index else []
        vector_results = _vector_search(query_tokens, self.vector_index, self.vector_metadata, top_k)

        merged_results = _merge_and_rank(bm25_results, vector_results, self.doc_summaries, top_k)

        return [
            {
                "doc_id": item["doc_id"],
                "doc_type": item["doc_type"],
                "title": item["title"],
                "context": item.get("summary", ""),
                "bm25_score": item["bm25_score"],
                "vector_similarity": item["vector_similarity"],
                "combined_score": item["combined_score"],
            }
            for item in merged_results
        ]


def search_hybrid_v2(
    question: str,
    top_k: Optional[int] = None,
    bm25_path: Optional[Path] = None,
    vector_dir: Optional[Path] = None,
    preprocess_dir: Optional[Path] = None,
    glossary_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    return HybridSearchV2(
        top_k=top_k or TOP_K,
        bm25_path=bm25_path,
        vector_dir=vector_dir,
        preprocess_dir=preprocess_dir,
        glossary_path=glossary_path,
    ).search(question, top_k=top_k or TOP_K)
