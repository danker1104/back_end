import json
import re
import zlib
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

try:
    import faiss
except ImportError as exc:  # pragma: no cover
    raise SystemExit("faiss-cpu가 설치되어 있지 않습니다. 먼저 pip install faiss-cpu 를 실행하세요.") from exc

ROOT = Path(__file__).resolve().parent.parent
PREPROCESS_DIR = ROOT / "preprocess_v2"
OUTPUT_DIR = Path(__file__).resolve().parent

EMBEDDING_DIM = 256


def load_json(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def build_text(record: Dict[str, Any]) -> str:
    parts: List[str] = []
    summary = str(record.get("summary", "") or "").strip()
    keywords = record.get("keywords", []) or []
    related_terms = record.get("related_terms", []) or []

    if summary:
        parts.append(summary)
    if keywords:
        parts.append(" ".join([str(item) for item in keywords if str(item).strip()]))
    if related_terms:
        parts.append(" ".join([str(item) for item in related_terms if str(item).strip()]))

    return " \n".join([part for part in parts if part]).strip()


def build_records() -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for filename in ["law_summary.json", "precedent_summary.json", "legal_terms_summary.json"]:
        path = PREPROCESS_DIR / filename
        data = load_json(path)
        for item in data:
            text = build_text(item)
            if not text:
                continue
            records.append({
                "doc_id": item.get("doc_id", ""),
                "doc_type": item.get("doc_type", ""),
                "title": item.get("title", ""),
                "summary": item.get("summary", ""),
                "text": text,
            })
    return records


def hash_text_to_vector(text: str, dimension: int = EMBEDDING_DIM) -> np.ndarray:
    normalized = re.sub(r"\s+", " ", text.lower()).strip().encode("utf-8")
    digest = zlib.adler32(normalized) % (2**31 - 1)
    vector = np.zeros(dimension, dtype=np.float32)
    for index in range(dimension):
        vector[index] = ((digest >> (index % 31)) & 1) * 1.0
    return vector


def embed_texts(records: List[Dict[str, Any]], batch_size: int = 1024) -> Tuple[faiss.Index, List[Dict[str, Any]]]:
    if not records:
        raise ValueError("No records to embed")

    index = faiss.IndexFlatL2(EMBEDDING_DIM)
    metadata: List[Dict[str, Any]] = []

    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        embeddings = np.vstack([hash_text_to_vector(record.get("text", "")) for record in batch]).astype(np.float32)
        index.add(embeddings)
        metadata.extend([
            {
                "doc_id": record.get("doc_id", ""),
                "doc_type": record.get("doc_type", ""),
                "title": record.get("title", ""),
                "summary": record.get("summary", ""),
                "text": record.get("text", ""),
            }
            for record in batch
        ])

    return index, metadata


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records = build_records()
    if not records:
        raise RuntimeError("No records were found to embed")

    print(f"Embedding {len(records)} records")
    index, metadata = embed_texts(records)

    index_path = OUTPUT_DIR / "faiss.index"
    metadata_path = OUTPUT_DIR / "metadata.json"

    faiss.write_index(index, str(index_path))
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved index to {index_path}")
    print(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    main()
