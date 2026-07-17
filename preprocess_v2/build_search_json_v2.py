import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
SOURCE_ROOT = ROOT / "114.법률 지식기반 관계 데이터"
RAW_DATA_DIR = SOURCE_ROOT / "01.원천데이터"
LABEL_DATA_DIR = SOURCE_ROOT / "02.라벨링데이터"
OUTPUT_DIR = Path(__file__).resolve().parent

STOPWORDS = {
    "이", "그", "저", "것", "들", "때", "및", "등", "또는", "그리고", "하여", "하였", "한", "한데",
    "위", "같", "같은", "각", "의", "를", "을", "가", "나", "다", "로", "에", "에서", "에게", "에게서",
    "대한", "관련", "이상", "이하", "또", "있다", "있어", "있었", "하여서", "하여야", "사실", "관하여",
    "따라", "보면", "볼", "대하여", "대해", "한다", "하고", "하였다", "하여", "로서", "있는", "없다"
}


def normalize_text(value: str) -> str:
    if not value:
        return ""
    text = str(value).strip()
    text = text.replace("\u3000", " ")
    text = re.sub(r"\s+", " ", text)
    return text


def extract_candidate_text(values: List[str]) -> str:
    cleaned_values = []
    for value in values:
        text = normalize_text(value)
        if not text:
            continue
        if len(text) < 8:
            continue
        if re.fullmatch(r"[\d.\-_/]+", text):
            continue
        if text.startswith("http"):
            continue
        if text in {"국패", "국세기본법", "국패"}:
            continue
        cleaned_values.append(text)

    if not cleaned_values:
        return ""

    preferred = []
    for text in cleaned_values:
        if re.search(r"[가-힣]{2,}", text):
            preferred.append(text)

    if preferred:
        cleaned_values = preferred

    best = max(cleaned_values, key=lambda item: (len(item), item))
    return best[:400]


def extract_law_references(values: List[str]) -> List[str]:
    refs: List[str] = []
    for value in values:
        text = normalize_text(value)
        if not text:
            continue
        for match in re.finditer(r"([가-힣A-Za-z0-9]+(?:법|법률|법령))(?:\s*제\s*\d+(?:조|항|호)?)?", text):
            ref = match.group(1)
            if ref in {"법원", "법률", "법령"}:
                continue
            refs.append(ref)
    return sorted(set(refs))


def extract_keywords(title: str, summary: str, citations: List[str]) -> List[str]:
    combined = " ".join([title, summary, " ".join(citations)])
    tokens = re.findall(r"[가-힣A-Za-z0-9]+", combined)
    keywords = []
    for token in tokens:
        token = token.strip()
        if not token:
            continue
        if len(token) <= 1:
            continue
        if token.lower() in STOPWORDS:
            continue
        if token in {"제", "조", "항", "호"}:
            continue
        keywords.append(token)
    seen = set()
    unique_keywords: List[str] = []
    for token in keywords:
        key = token.lower()
        if key in seen:
            continue
        seen.add(key)
        unique_keywords.append(token)
    return unique_keywords[:10]


def read_csv_records(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        values: List[str] = []
        for row in reader:
            for item in row:
                text = normalize_text(item)
                if text:
                    values.append(text)
        cleaned_values = []
        seen = set()
        for value in values:
            if value in seen:
                continue
            seen.add(value)
            cleaned_values.append(value)

    if not cleaned_values:
        return []

    title = path.stem
    summary = extract_candidate_text(cleaned_values)
    citations = extract_law_references(cleaned_values)
    keywords = extract_keywords(title, summary, citations)

    return [{
        "doc_id": f"source-{path.stem}",
        "doc_type": "precedent",
        "title": title,
        "summary": summary or f"{title} 관련 법률 사례 요약",
        "search_text": " ".join([title, summary, *citations]),
        "keywords": keywords,
        "citations": citations,
        "source": {
            "source_id": path.stem,
            "source_name": "AI Hub 원천데이터",
            "source_type": "ai_hub_raw"
        },
        "metadata": {
            "category": "precedent",
            "source_file": str(path.relative_to(ROOT)).replace("\\", "/"),
            "source_dir": str(path.parent.relative_to(ROOT)).replace("\\", "/")
        }
    }]


def build_precedent_summaries() -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for csv_path in sorted(RAW_DATA_DIR.rglob("*.csv")):
        if "판결문" not in csv_path.as_posix() and "결정문" not in csv_path.as_posix():
            continue
        records.extend(read_csv_records(csv_path))
    return records


def build_law_summaries(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, Dict[str, Any]] = {}
    for record in records:
        citations = record.get("citations", [])
        for citation in citations:
            base_name = citation
            if " 제" in citation:
                base_name = citation.split(" 제", 1)[0]
            entry = grouped.setdefault(base_name, {
                "doc_id": f"law-{len(grouped) + 1:04d}",
                "doc_type": "law",
                "title": base_name,
                "summary": record.get("summary", ""),
                "search_text": " ".join([base_name, record.get("summary", "")]),
                "keywords": [],
                "citations": [],
                "source": {
                    "source_id": base_name,
                    "source_name": "AI Hub 원천데이터",
                    "source_type": "ai_hub_raw"
                },
                "metadata": {
                    "category": "law",
                    "source_count": 0
                }
            })
            entry["citations"] = sorted(set(entry.get("citations", []) + [citation]))
            entry["search_text"] = " ".join(sorted(set([entry["search_text"], record.get("search_text", ""), base_name])) )
            entry["keywords"] = extract_keywords(entry["title"], entry["summary"], entry["citations"])
            entry["metadata"]["source_count"] = int(entry["metadata"].get("source_count", 0)) + 1
    return list(grouped.values())


def build_legal_term_summaries() -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for json_path in sorted(LABEL_DATA_DIR.rglob("*.json")):
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        entities = payload.get("Entities", [])
        if not isinstance(entities, list):
            continue
        for index, entity in enumerate(entities):
            thing = normalize_text(entity.get("Thing"))
            topic = normalize_text(entity.get("Topic"))
            link = normalize_text(entity.get("Link"))
            if not thing:
                continue
            summary = topic or f"{thing} 관련 법률 용어"
            keywords = extract_keywords(thing, summary, [link] if link else [])
            records.append({
                "doc_id": f"term-{index + 1:04d}",
                "doc_type": "term",
                "title": thing,
                "definition": topic or "법률 용어",
                "summary": summary,
                "search_text": " ".join([thing, summary, link]),
                "keywords": keywords,
                "citations": [link] if link else [],
                "related_terms": [topic] if topic else [],
                "source": {
                    "source_id": json_path.stem,
                    "source_name": "AI Hub 라벨링데이터",
                    "source_type": "ai_hub_label"
                },
                "metadata": {
                    "category": "legal_term",
                    "source_file": str(json_path.relative_to(ROOT)).replace("\\", "/")
                }
            })
    return records


def save_json(path: Path, payload: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    precedent_records = build_precedent_summaries()
    law_records = build_law_summaries(precedent_records)
    term_records = build_legal_term_summaries()

    save_json(OUTPUT_DIR / "law_summary.json", law_records)
    save_json(OUTPUT_DIR / "precedent_summary.json", precedent_records)
    save_json(OUTPUT_DIR / "legal_terms_summary.json", term_records)

    print(f"Generated {len(law_records)} law records")
    print(f"Generated {len(precedent_records)} precedent records")
    print(f"Generated {len(term_records)} legal term records")


if __name__ == "__main__":
    main()
