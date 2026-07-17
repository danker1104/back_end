import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

ROOT = Path(__file__).resolve().parent.parent
PREPROCESS_DIR = ROOT / "preprocess_v2"
GLOSSARY_PATH = ROOT / "output" / "legal_glossary.json"

STOPWORDS = {
    "이", "그", "저", "것", "들", "때", "및", "등", "또는", "그리고", "하여", "하였", "한", "한데",
    "위", "같", "같은", "각", "의", "를", "을", "가", "나", "다", "로", "에", "에서", "에게", "에게서",
    "대한", "관련", "이상", "이하", "또", "있다", "있어", "있었", "하여서", "하여야", "사실", "관하여",
    "따라", "보면", "볼", "대하여", "대해", "한다", "하고", "하였다", "하여", "로서", "있는", "없다",
    "좀", "좀요", "요", "제", "조", "항", "호"
}

QUERY_STOPWORDS = {
    "당했어", "당했냐", "못", "받았어", "했어", "했냐", "해", "해줘", "해줘요", "좀", "내", "나", "저",
    "있어", "없어", "합니다", "해요", "입니다", "인가요", "어때요", "어떻게", "있나요"
}

LEGAL_SYNONYMS: Dict[str, List[str]] = {
    "보험사기": ["보험사기", "보험사기방지 특별법", "보험금 편취", "사기죄", "기망", "보험"],
    "폭행": ["폭행", "상해", "형법 폭행죄", "형법 상해죄"],
    "임금": ["임금", "임금체불", "근로기준법", "체불임금"],
    "해고": ["해고", "근로기준법", "부당해고", "해고무효"],
    "임대차": ["임대차", "주택임대차보호법", "전월세", "임대차계약"],
    "상속": ["상속", "상속세및증여세법", "유류분", "민법 상속"],
    "증여": ["증여", "증여세법", "상속세및증여세법"],
    "국세": ["국세", "국세기본법", "국세징수법", "세무조사"],
    "계약": ["계약", "계약서", "채무", "손해배상"],
}

CACHE: Dict[str, object] = {}


def _normalize_text(value: str) -> str:
    text = str(value or "").strip()
    text = text.replace("\u3000", " ")
    text = re.sub(r"[^가-힣A-Za-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()


def _normalize_keyword(token: str) -> str:
    token = _normalize_text(token)
    if not token:
        return token
    token = re.sub(r"(을|를|이|가|은|는|으로|로|에게|께|에서|부터|까지|과|와|로서|보다|한테|께서)$", "", token)
    return token.strip()


def _unique_preserve_order(items: Iterable[str]) -> List[str]:
    seen: Set[str] = set()
    result: List[str] = []
    for item in items:
        if not item:
            continue
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _load_json(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []


def _load_legal_glossary(glossary_path: Optional[Path] = None) -> Dict[str, Dict[str, object]]:
    glossary_path = glossary_path or GLOSSARY_PATH
    if str(glossary_path) in CACHE:
        return CACHE[str(glossary_path)]  # type: ignore[return-value]

    data: Dict[str, Dict[str, object]] = {}
    if not glossary_path.exists():
        CACHE[str(glossary_path)] = data
        return data

    try:
        raw = json.loads(glossary_path.read_text(encoding="utf-8"))
        for thing, item in raw.items():
            if not isinstance(item, dict):
                continue
            data[str(thing)] = {
                "topics": item.get("topics", []) if isinstance(item.get("topics"), list) else [],
                "links": item.get("links", {}) if isinstance(item.get("links"), dict) else {},
            }
    except Exception:
        data = {}

    CACHE[str(glossary_path)] = data
    return data


def _load_search_summaries(preprocess_dir: Optional[Path] = None) -> Dict[str, Dict[str, Dict[str, object]]]:
    preprocess_dir = preprocess_dir or PREPROCESS_DIR
    cache_key = f"search_summaries:{preprocess_dir}"
    if cache_key in CACHE:
        return CACHE[cache_key]  # type: ignore[return-value]

    summary_files = {
        "law": preprocess_dir / "law_summary.json",
        "precedent": preprocess_dir / "precedent_summary.json",
        "term": preprocess_dir / "legal_terms_summary.json",
    }
    summaries: Dict[str, Dict[str, Dict[str, object]]] = {
        "law": {},
        "precedent": {},
        "term": {},
    }

    for doc_type, path in summary_files.items():
        items = _load_json(path)
        for item in items:
            title = str(item.get("title", ""))
            if not title:
                continue
            summaries[doc_type][title] = {
                "search_text": str(item.get("search_text", "")),
                "keywords": item.get("keywords", []) if isinstance(item.get("keywords"), list) else [],
                "citations": item.get("citations", []) if isinstance(item.get("citations"), list) else [],
                "related_terms": item.get("related_terms", []) if isinstance(item.get("related_terms"), list) else [],
            }

    CACHE[cache_key] = summaries
    return summaries


def _extract_keywords(question: str) -> List[str]:
    normalized = _normalize_text(question)
    if not normalized:
        return []

    tokens = re.findall(r"[가-힣A-Za-z0-9]+", normalized)
    raw_keywords: List[str] = []
    for token in tokens:
        keyword = _normalize_keyword(token)
        if not keyword:
            continue
        if len(keyword) <= 1:
            continue
        if keyword in STOPWORDS or keyword in QUERY_STOPWORDS:
            continue
        if keyword.isdigit():
            continue
        raw_keywords.append(keyword)

    return _unique_preserve_order(raw_keywords)


def _expand_with_glossary(keywords: List[str], glossary: Dict[str, Dict[str, object]]) -> List[str]:
    expanded: List[str] = []
    glossary_items = list(glossary.items())
    for keyword in keywords:
        for thing, data in glossary_items:
            thing_normalized = _normalize_text(thing)
            lower_kw = keyword.lower()
            if lower_kw == thing_normalized or lower_kw in thing_normalized or thing_normalized in lower_kw:
                expanded.append(thing)
                for topic in data.get("topics", []):
                    if isinstance(topic, str):
                        expanded.append(topic)
                for link, topics in data.get("links", {}).items():
                    expanded.append(link)
                    if isinstance(topics, list):
                        for topic in topics:
                            if isinstance(topic, str):
                                expanded.append(topic)
                break
    return _unique_preserve_order(expanded)


def _expand_with_legal_synonyms(keywords: List[str]) -> List[str]:
    expanded: List[str] = []
    for keyword in keywords:
        if keyword in LEGAL_SYNONYMS:
            expanded.extend(LEGAL_SYNONYMS[keyword])
        elif keyword in {syn for synonyms in LEGAL_SYNONYMS.values() for syn in synonyms}:
            expanded.append(keyword)
    return _unique_preserve_order(expanded)


def _expand_with_preprocess_related_laws(keywords: List[str], summaries: Dict[str, Dict[str, Dict[str, object]]]) -> List[str]:
    expanded: List[str] = []
    for keyword in keywords:
        lower_keyword = keyword.lower()
        for doc_type, docs in summaries.items():
            for title, metadata in docs.items():
                title_norm = _normalize_text(title)
                search_text = _normalize_text(metadata.get("search_text", ""))
                keyword_list = [str(item).lower() for item in metadata.get("keywords", [])]
                citation_list = [str(item).lower() for item in metadata.get("citations", [])]
                related_terms = [str(item).lower() for item in metadata.get("related_terms", [])]

                is_match = (
                    lower_keyword == title_norm
                    or lower_keyword in title_norm
                    or title_norm in lower_keyword
                    or lower_keyword in search_text
                    or lower_keyword in " ".join(keyword_list)
                    or lower_keyword in " ".join(citation_list)
                    or lower_keyword in " ".join(related_terms)
                )
                if not is_match:
                    continue

                expanded.append(title)
                expanded.extend(metadata.get("keywords", []))
                expanded.extend(metadata.get("citations", []))
                expanded.extend(metadata.get("related_terms", []))
    return _unique_preserve_order(expanded)


def rewrite_query_v2(
    question: str,
    glossary_path: Optional[Path] = None,
    preprocess_dir: Optional[Path] = None,
) -> Dict[str, List[str]]:
    keywords = _extract_keywords(question)
    glossary = _load_legal_glossary(glossary_path)
    summaries = _load_search_summaries(preprocess_dir)

    expanded_keywords: List[str] = []
    expanded_keywords.extend(_expand_with_legal_synonyms(keywords))
    expanded_keywords.extend(_expand_with_glossary(keywords, glossary))
    expanded_keywords.extend(_expand_with_preprocess_related_laws(keywords, summaries))

    expanded_keywords = _unique_preserve_order(keywords + expanded_keywords)

    return {
        "keywords": keywords,
        "expanded_keywords": expanded_keywords,
    }
