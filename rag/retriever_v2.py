import json
import pickle
import re
import zlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import faiss
import numpy as np

from rag.query_rewrite_v2 import rewrite_query_v2

ROOT = Path(__file__).resolve().parent.parent
VECTOR_DB_V2_DIR = ROOT / "vector_db_v2"
PREPROCESS_DIR = ROOT / "preprocess_v2"
GLOSSARY_PATH = ROOT / "output" / "legal_glossary.json"
BM25_V2_PATH = VECTOR_DB_V2_DIR / "bm25_v2.pkl"
VECTOR_INDEX_PATH = VECTOR_DB_V2_DIR / "faiss.index"
VECTOR_METADATA_PATH = VECTOR_DB_V2_DIR / "metadata.json"
EMBEDDING_DIM = 256
TOKEN_PATTERN = re.compile(r"[가-힣]+|[A-Za-z0-9]+")


def _tokenize(text: str) -> List[str]:
    if not text:
        return []
    return [token.lower() for token in TOKEN_PATTERN.findall(text) if len(token) > 1]


def _normalize_keywords(tokens: Iterable[str]) -> List[str]:
    seen: Set[str] = set()
    normalized: List[str] = []
    for token in tokens:
        if not token:
            continue
        word = str(token).strip().lower()
        if len(word) <= 1:
            continue
        if word in {"있어", "없어", "합니다", "해요", "입니다", "인가요", "어때요", "어떻게"}:
            continue
        if word in {"제", "조", "항", "호"}:
            continue
        if word.isdigit():
            continue
        if word in seen:
            continue
        seen.add(word)
        normalized.append(word)
    return normalized


def _hash_text_to_vector(text: str) -> np.ndarray:
    normalized = re.sub(r"\s+", " ", str(text or "").lower()).strip().encode("utf-8")
    digest = zlib.adler32(normalized) % (2**31 - 1)
    vector = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    for index in range(EMBEDDING_DIM):
        vector[index] = float((digest >> (index % 31)) & 1)
    return vector


def _load_json(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _load_pickle(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    try:
        with path.open("rb") as handle:
            return pickle.load(handle)
    except Exception:
        return None


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


def _load_document_summaries(preprocess_dir: Path) -> Dict[str, Dict[str, Any]]:
    summaries: Dict[str, Dict[str, Any]] = {}
    for filename in ["law_summary.json", "precedent_summary.json", "legal_terms_summary.json"]:
        path = preprocess_dir / filename
        payload = _load_json(path)
        if not isinstance(payload, list):
            continue
        for item in payload:
            doc_id = str(item.get("doc_id", ""))
            if not doc_id:
                continue
            summaries[doc_id] = {
                "doc_type": str(item.get("doc_type", "")),
                "title": str(item.get("title", "")),
                "summary": str(item.get("summary", "")),
                "keywords": item.get("keywords", []) if isinstance(item.get("keywords", []), list) else [],
                "related_laws": item.get("citations", []) if isinstance(item.get("citations", []), list) else [],
            }
    return summaries


def _bm25_score(query_tokens: List[str], bm25_index: Dict[str, Any], top_k: int) -> List[Dict[str, Any]]:
    if not query_tokens or not bm25_index:
        return []

    tokenized_docs = bm25_index.get("tokenized_docs", [])
    idf = bm25_index.get("idf", {})
    doc_len = bm25_index.get("doc_len", [])
    avgdl = float(bm25_index.get("avgdl", 0.0))
    k1 = float(bm25_index.get("k1", 1.5))
    b = float(bm25_index.get("b", 0.75))
    doc_ids = bm25_index.get("doc_ids", [])
    titles = bm25_index.get("titles", [])
    doc_types = bm25_index.get("doc_types", [])

    results: List[Dict[str, Any]] = []
    query_terms = [token.lower() for token in query_tokens]

    for idx, doc_tokens in enumerate(tokenized_docs):
        if idx >= len(doc_ids):
            continue
        frequencies: Dict[str, int] = {}
        for token in doc_tokens:
            frequencies[token] = frequencies.get(token, 0) + 1

        score = 0.0
        for term in query_terms:
            f = frequencies.get(term, 0)
            if f <= 0:
                continue
            term_idf = float(idf.get(term, 0.0))
            score += term_idf * ((f * (k1 + 1.0)) / (f + k1 * (1.0 - b + b * (doc_len[idx] / avgdl if avgdl > 0 else 1.0))))

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
    return results[:top_k]


def _vector_search(query_tokens: List[str], index: Optional[faiss.Index], metadata: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    if index is None or not metadata or not query_tokens:
        return []

    query_text = " ".join(query_tokens)
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
                "vector_score": float(1.0 / (1.0 + float(distance))),
            }
        )
    return results


def _merge_results(
    bm25_results: List[Dict[str, Any]],
    vector_results: List[Dict[str, Any]],
    doc_summaries: Dict[str, Dict[str, Any]],
    top_k: int,
) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    max_bm25 = max((item["score"] for item in bm25_results), default=0.0)
    max_vector = max((item["vector_score"] for item in vector_results), default=0.0)

    def norm(value: float, maximum: float) -> float:
        return float(value / maximum) if maximum > 0 else 0.0

    for item in bm25_results:
        merged[item["doc_id"]] = {
            "doc_id": item["doc_id"],
            "doc_type": item.get("doc_type", ""),
            "title": item.get("title", ""),
            "bm25_score": item["score"],
            "vector_score": 0.0,
        }

    for item in vector_results:
        entry = merged.setdefault(item["doc_id"], {
            "doc_id": item["doc_id"],
            "doc_type": item.get("doc_type", ""),
            "title": item.get("title", ""),
            "bm25_score": 0.0,
            "vector_score": item["vector_score"],
        })
        entry["vector_score"] = max(entry.get("vector_score", 0.0), item["vector_score"])

    for entry in merged.values():
        bm25_norm = norm(entry["bm25_score"], max_bm25)
        vector_norm = norm(entry["vector_score"], max_vector)
        entry["combined_score"] = bm25_norm * 0.6 + vector_norm * 0.4 if bm25_norm and vector_norm else bm25_norm or vector_norm

    ranked = sorted(merged.values(), key=lambda item: item["combined_score"], reverse=True)
    return ranked[:top_k]


def _format_result(entry: Dict[str, Any], doc_summaries: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    doc = doc_summaries.get(entry["doc_id"], {})
    summary = str(doc.get("summary", ""))[:300].strip()
    if not summary:
        summary = "(요약 정보 없음)"

    return {
        "doc_id": entry["doc_id"],
        "doc_type": doc.get("doc_type", entry.get("doc_type", "")),
        "title": doc.get("title", entry.get("title", "")),
        "summary": summary,
        "keywords": doc.get("keywords", []),
        "related_laws": doc.get("related_laws", []),
        "bm25_score": entry.get("bm25_score", 0.0),
        "vector_score": entry.get("vector_score", 0.0),
        "combined_score": entry.get("combined_score", 0.0),
    }


class RetrieverV2:
    def __init__(
        self,
        top_k: int = 5,
        bm25_path: Optional[Path] = None,
        vector_dir: Optional[Path] = None,
        preprocess_dir: Optional[Path] = None,
        glossary_path: Optional[Path] = None,
    ):
        self.top_k = min(top_k, 5)
        self.bm25_path = Path(bm25_path or BM25_V2_PATH)
        self.vector_dir = Path(vector_dir or VECTOR_DB_V2_DIR)
        self.vector_index_path = self.vector_dir / "faiss.index"
        self.vector_metadata_path = self.vector_dir / "metadata.json"
        self.preprocess_dir = Path(preprocess_dir or PREPROCESS_DIR)
        self.glossary_path = Path(glossary_path or GLOSSARY_PATH)

        self.bm25_index = _load_pickle(self.bm25_path) or {}
        self.vector_index, self.vector_metadata = _load_vector_db(self.vector_index_path, self.vector_metadata_path)
        self.doc_summaries = _load_document_summaries(self.preprocess_dir)

    def search(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        if top_k is None:
            top_k = self.top_k
        top_k = min(top_k, 5)

        query_data = rewrite_query_v2(
            query,
            glossary_path=self.glossary_path,
            preprocess_dir=self.preprocess_dir,
        )
        query_tokens = query_data.get("expanded_keywords") or query_data.get("keywords")
        query_tokens = _normalize_keywords(query_tokens)

        if not query_tokens:
            return []

        bm25_results = _bm25_score(query_tokens, self.bm25_index, top_k)
        vector_results = _vector_search(query_tokens, self.vector_index, self.vector_metadata, top_k)
        merged_results = _merge_results(bm25_results, vector_results, self.doc_summaries, top_k)

        return [_format_result(entry, self.doc_summaries) for entry in merged_results]


def load_retriever_v2(top_k: int = 5) -> RetrieverV2:
    return RetrieverV2(top_k=top_k)
