import json
import tempfile
from pathlib import Path

import numpy as np

from vector_db.faiss_store import build_vector_db_from_embeddings


def test_build_vector_db_from_embeddings_preserves_text_in_metadata(tmp_path):
    embeddings_path = tmp_path / "chunk_embeddings.npy"
    payload = np.array(
        [
            {
                "text": "계약 해제는 당사자 일방이 계약을 종료할 수 있는 사유가 있을 때 가능합니다.",
                "metadata": {"doc_id": "test-001", "casenames": "계약해제"},
                "embedding": np.array([0.1, 0.2], dtype=np.float32),
            }
        ],
        dtype=object,
    )
    np.save(embeddings_path, payload, allow_pickle=True)

    index_path, metadata_path = build_vector_db_from_embeddings(str(embeddings_path))

    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    assert metadata[0]["text"].startswith("계약 해제")
    assert metadata[0]["doc_id"] == "test-001"
    assert metadata[0]["casenames"] == "계약해제"
