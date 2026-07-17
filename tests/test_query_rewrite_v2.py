import json
from pathlib import Path

from rag.query_rewrite_v2 import rewrite_query_v2


def test_rewrite_query_v2_insurance_fraud(tmp_path: Path):
    glossary = {
        "보험사기": {
            "topics": ["보험금 편취"],
            "links": {"보험사기방지 특별법": ["사기죄"]},
        }
    }
    glossary_path = tmp_path / "legal_glossary.json"
    glossary_path.write_text(json.dumps(glossary, ensure_ascii=False), encoding="utf-8")

    result = rewrite_query_v2(
        "보험사기를 당했어",
        glossary_path=glossary_path,
        preprocess_dir=tmp_path,
    )

    assert result["keywords"] == ["보험사기"]
    assert "보험사기방지 특별법" in result["expanded_keywords"]
    assert "보험금 편취" in result["expanded_keywords"]
    assert "사기죄" in result["expanded_keywords"]
