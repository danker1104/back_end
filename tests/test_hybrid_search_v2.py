import json
from pathlib import Path

import faiss
import numpy as np

from rag.hybrid_search_v2 import HybridSearchV2


def create_faiss_index_file(index_path: Path, metadata_path: Path, text: str, doc_id: str, title: str, summary: str, doc_type: str = "law"):
    dimension = 256
    index = faiss.IndexFlatL2(dimension)
    vector = np.zeros(dimension, dtype=np.float32)
    normalized = text.lower().encode("utf-8")
    digest = int.from_bytes(normalized[:4].ljust(4, b"\0"), "little")
    for i in range(dimension):
        vector[i] = float((digest >> (i % 31)) & 1)
    index.add(vector.reshape(1, -1))
    faiss.write_index(index, str(index_path))

    metadata = [
        {
            "doc_id": doc_id,
            "doc_type": doc_type,
            "title": title,
            "summary": summary,
            "text": text,
        }
    ]
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")


def test_hybrid_search_v2_returns_summary_context(tmp_path: Path):
    preprocess_dir = tmp_path / "preprocess_v2"
    preprocess_dir.mkdir(parents=True)

    law_summary = [
        {
            "doc_id": "law-0001",
            "doc_type": "law",
            "title": "민법",
            "summary": "민법은 채무와 계약을 다룹니다.",
            "keywords": ["민법", "채무"],
            "related_terms": ["계약"],
            "citations": ["민법"],
        }
    ]
    (preprocess_dir / "law_summary.json").write_text(json.dumps(law_summary, ensure_ascii=False), encoding="utf-8")
    (preprocess_dir / "precedent_summary.json").write_text(json.dumps([]), encoding="utf-8")
    (preprocess_dir / "legal_terms_summary.json").write_text(json.dumps([]), encoding="utf-8")

    glossary = {
        "민법": {
            "topics": ["채무"],
            "links": {"민법": ["계약"]},
        }
    }
    glossary_path = tmp_path / "legal_glossary.json"
    glossary_path.write_text(json.dumps(glossary, ensure_ascii=False), encoding="utf-8")

    bm25_path = tmp_path / "bm25_v2.pkl"
    bm25_object = {
        "doc_ids": ["law-0001"],
        "doc_types": ["law"],
        "titles": ["민법"],
        "texts": ["민법은 채무와 계약을 다룹니다."],
        "tokenized_docs": [["민법", "채무", "계약"]],
        "doc_freq": {"민법": 1, "채무": 1, "계약": 1},
        "idf": {"민법": 1.0, "채무": 1.0, "계약": 1.0},
        "doc_len": [3],
        "avgdl": 3.0,
        "k1": 1.5,
        "b": 0.75,
        "N": 1,
    }
    with bm25_path.open("wb") as handle:
        import pickle

        pickle.dump(bm25_object, handle)

    vector_dir = tmp_path / "vector_db_v2"
    vector_dir.mkdir(parents=True)
    create_faiss_index_file(
        vector_dir / "faiss.index",
        vector_dir / "metadata.json",
        text="민법 채무 계약",
        doc_id="law-0001",
        title="민법",
        summary="민법은 채무와 계약을 다룹니다.",
    )

    search = HybridSearchV2(
        top_k=3,
        bm25_path=bm25_path,
        vector_dir=vector_dir,
        preprocess_dir=preprocess_dir,
        glossary_path=glossary_path,
    )

    results = search.search("민법 채무")
    assert len(results) == 1
    assert results[0]["doc_id"] == "law-0001"
    assert "민법은 채무와 계약을 다룹니다." in results[0]["context"]
    assert results[0]["bm25_score"] >= 0
    assert results[0]["vector_similarity"] >= 0
    assert results[0]["combined_score"] >= 0
