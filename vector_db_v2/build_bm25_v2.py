import json
import math
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
PREPROCESS_DIR = ROOT / "preprocess_v2"
OUTPUT_DIR = Path(__file__).resolve().parent

BM25_PATH = OUTPUT_DIR / "bm25_v2.pkl"

TOKEN_PATTERN = re.compile(r"[가-힣]+|[A-Za-z0-9]+")


def load_json(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def tokenize(text: str) -> List[str]:
    if not text:
        return []
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


def build_document_text(item: Dict[str, Any]) -> str:
    parts: List[str] = []
    summary = str(item.get("summary", "") or "").strip()
    keywords = item.get("keywords", []) or []
    related_terms = item.get("related_terms", []) or []
    law_name = ""
    case_name = ""

    if item.get("doc_type") == "law":
        law_name = str(item.get("title", "") or "").strip()
    elif item.get("doc_type") == "precedent":
        case_name = str(item.get("title", "") or "").strip()
        law_name = " ".join(item.get("citations", []) or [])
    elif item.get("doc_type") == "term":
        law_name = str(item.get("title", "") or "").strip()
    
    if summary:
        parts.append(summary)
    if keywords:
        parts.append(" ".join([str(x) for x in keywords if str(x).strip()]))
    if related_terms:
        parts.append(" ".join([str(x) for x in related_terms if str(x).strip()]))
    if law_name:
        parts.append(law_name)
    if case_name:
        parts.append(case_name)

    return " ".join([part for part in parts if part])


def build_corpus() -> List[Dict[str, Any]]:
    documents: List[Dict[str, Any]] = []
    for filename in ["law_summary.json", "precedent_summary.json", "legal_terms_summary.json"]:
        path = PREPROCESS_DIR / filename
        items = load_json(path)
        for item in items:
            text = build_document_text(item)
            if not text:
                continue
            documents.append({
                "doc_id": item.get("doc_id", ""),
                "doc_type": item.get("doc_type", ""),
                "title": item.get("title", ""),
                "text": text,
                "tokens": tokenize(text),
                "law_name": item.get("title") if item.get("doc_type") == "law" else "",
                "case_name": item.get("title") if item.get("doc_type") == "precedent" else "",
            })
    return documents


def compute_bm25(corpus: List[Dict[str, Any]], k1: float = 1.5, b: float = 0.75) -> Dict[str, Any]:
    N = len(corpus)
    doc_freq: Dict[str, int] = defaultdict(int)
    doc_len: List[int] = []
    tokenized_docs: List[List[str]] = []
    postings: Dict[str, List[int]] = defaultdict(list)

    for idx, item in enumerate(corpus):
        tokens = item["tokens"]
        tokenized_docs.append(tokens)
        doc_len.append(len(tokens))
        unique_tokens = set(tokens)
        for token in unique_tokens:
            doc_freq[token] += 1
            postings[token].append(idx)

    avgdl = sum(doc_len) / N if N else 0.0
    idf: Dict[str, float] = {}
    for token, freq in doc_freq.items():
        idf[token] = math.log((N - freq + 0.5) / (freq + 0.5) + 1)

    bm25_index = {
        "doc_ids": [item["doc_id"] for item in corpus],
        "doc_types": [item["doc_type"] for item in corpus],
        "titles": [item["title"] for item in corpus],
        "texts": [item["text"] for item in corpus],
        "law_names": [item["law_name"] for item in corpus],
        "case_names": [item["case_name"] for item in corpus],
        "tokenized_docs": tokenized_docs,
        "doc_freq": dict(doc_freq),
        "idf": idf,
        "doc_len": doc_len,
        "avgdl": avgdl,
        "k1": k1,
        "b": b,
        "N": N,
        "postings": {token: ids for token, ids in postings.items()},
    }
    return bm25_index


def save_bm25(index: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(index, handle)


def main() -> None:
    corpus = build_corpus()
    if not corpus:
        raise RuntimeError("No documents found for BM25 corpus")

    print(f"Building BM25 index for {len(corpus)} documents")
    bm25_index = compute_bm25(corpus)
    save_bm25(bm25_index, BM25_PATH)
    print(f"Saved BM25 index to {BM25_PATH}")


if __name__ == "__main__":
    main()
