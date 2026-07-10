import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

from config import PROJECT_ROOT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


TARGET_DIRECTORIES = [
    PROJECT_ROOT / "data" / "TS_01. 민사법_001. 판결문",
    PROJECT_ROOT / "data" / "TS_01. 민사법_002. 법령",
]


def discover_json_files() -> List[Path]:
    """지정된 두 데이터 폴더에서 JSON 파일 목록을 자동 탐색한다."""
    json_files: List[Path] = []
    for directory in TARGET_DIRECTORIES:
        if not directory.exists():
            logger.warning("Data directory not found: %s", directory)
            continue
        json_files.extend(sorted(directory.glob("*.json")))
    return json_files


def load_json_documents() -> List[Dict[str, Any]]:
    """JSON 파일을 읽어 파이썬 객체로 로드한다."""
    documents: List[Dict[str, Any]] = []
    for path in discover_json_files():
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, dict):
                documents.append({"source_path": str(path), **data})
            else:
                logger.warning("Unexpected JSON structure in %s", path)
        except json.JSONDecodeError as exc:
            logger.warning("Invalid JSON file %s: %s", path, exc)
        except OSError as exc:
            logger.warning("Failed to read %s: %s", path, exc)
    return documents


def extract_text_from_document(document: Dict[str, Any]) -> str:
    """문서에서 텍스트를 추출한다. 판결문과 법령 구조를 모두 지원한다."""
    sentences = document.get("sentences")
    if isinstance(sentences, list):
        text_parts = [str(item).strip() for item in sentences if isinstance(item, (str, int, float))]
        return "\n".join(part for part in text_parts if part)

    if isinstance(document.get("text"), str):
        return document["text"].strip()

    return ""


def extract_metadata(document: Dict[str, Any]) -> Dict[str, Any]:
    """가능한 메타데이터를 추출한다."""
    metadata: Dict[str, Any] = {}

    for key in [
        "doc_id",
        "doc_class",
        "casenames",
        "normalized_court",
        "casetype",
        "statute_name",
        "statute_abbrv",
        "statute_type",
        "statute_category",
        "effective_date",
        "proclamation_date",
        "announce_date",
    ]:
        value = document.get(key)
        if value not in (None, "", []):
            metadata[key] = value

    metadata["source_path"] = document.get("source_path", "")
    return metadata


def normalize_text(text: str) -> str:
    """공백과 줄바꿈을 정리한다."""
    if not isinstance(text, str):
        return ""
    cleaned = " ".join(text.split())
    return cleaned.strip()


def clean_document(document: Dict[str, Any]) -> Dict[str, Any]:
    """텍스트 정제와 메타데이터 정리를 수행한다."""
    text = extract_text_from_document(document)
    cleaned_text = normalize_text(text)

    if not cleaned_text:
        return {}

    metadata = extract_metadata(document)
    metadata["text"] = cleaned_text
    return metadata


def prepare_documents() -> List[Dict[str, Any]]:
    """모든 문서를 읽고 정제된 형태로 반환한다."""
    raw_documents = load_json_documents()
    cleaned_documents: List[Dict[str, Any]] = []

    seen_texts = set()
    for document in raw_documents:
        cleaned = clean_document(document)
        if not cleaned:
            continue

        text = cleaned.get("text", "")
        if not text:
            continue
        if text in seen_texts:
            continue
        seen_texts.add(text)
        cleaned_documents.append(cleaned)

    logger.info("Prepared %d documents", len(cleaned_documents))
    return cleaned_documents


if __name__ == "__main__":
    docs = prepare_documents()
    print(f"Prepared {len(docs)} documents")
    if docs:
        print(docs[0])
