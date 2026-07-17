import json
import logging
from pathlib import Path
from typing import Dict, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
LABEL_DIR = ROOT / "114.법률 지식기반 관계 데이터" / "02.라벨링데이터"
OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def build_legal_glossary(label_dir: Path) -> Dict[str, Dict[str, object]]:
    glossary: Dict[str, Dict[str, object]] = {}
    json_files = sorted(label_dir.rglob("*.json"))

    logger.info("Found %d labeling JSON files", len(json_files))

    for path in json_files:
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except json.JSONDecodeError as exc:
            logger.warning("Skipping invalid JSON %s: %s", path, exc)
            continue

        entities = data.get("Entities")
        if not isinstance(entities, list):
            continue

        for entity in entities:
            thing = normalize_text(entity.get("Thing"))
            if not thing:
                continue

            glossary.setdefault(thing, {
                "thing": thing,
                "topics": [],
                "links": {},
                "source_files": set(),
                "example_contexts": [],
            })

            item = glossary[thing]
            topic = normalize_text(entity.get("Topic"))
            link = normalize_text(entity.get("Link"))
            if topic:
                if topic not in item["topics"]:
                    item["topics"].append(topic)
            if link:
                item["links"].setdefault(link, [])
                if topic and topic not in item["links"][link]:
                    item["links"][link].append(topic)
            item["source_files"].add(str(path.relative_to(ROOT)))

    for thing, item in glossary.items():
        item["source_files"] = sorted(item["source_files"])
        item["topics"] = sorted(item["topics"])
        item["links"] = {k: sorted(v) for k, v in sorted(item["links"].items())}

    return glossary


def save_json(glossary: Dict[str, Dict[str, object]], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(glossary, handle, ensure_ascii=False, indent=2)
    logger.info("Saved glossary JSON to %s", output_path)


def save_markdown(glossary: Dict[str, Dict[str, object]], output_path: Path, top_n: int = 100) -> None:
    sorted_items = sorted(glossary.values(), key=lambda item: (-len(item["source_files"]), item["thing"]))
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# 법률 라벨링 데이터 기반 법률 사전\n\n")
        handle.write("이 사전은 라벨링 데이터의 `Entities` 항목에서 추출한 `Thing` 값과 관련 메타정보를 정리합니다.\n\n")
        handle.write(f"총 항목 수: {len(sorted_items)}\n\n")
        handle.write("## 상위 항목\n\n")

        for item in sorted_items[:top_n]:
            handle.write(f"### {item['thing']}\n")
            handle.write(f"- source_files: {len(item['source_files'])}개\n")
            if item["topics"]:
                handle.write(f"- topics: {', '.join(item['topics'])}\n")
            if item["links"]:
                handle.write("- links:\n")
                for link, topics in item["links"].items():
                    handle.write(f"  - {link}: {', '.join(topics)}\n")
            handle.write("\n")

    logger.info("Saved glossary Markdown to %s", output_path)


def main() -> None:
    glossary = build_legal_glossary(LABEL_DIR)
    save_json(glossary, OUTPUT_DIR / "legal_glossary.json")
    save_markdown(glossary, OUTPUT_DIR / "legal_glossary.md")


if __name__ == "__main__":
    main()
