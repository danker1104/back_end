import argparse
import logging
import os
import re
import warnings

from inference.prompt_builder import build_answer_prompt, build_answer_prompt_v2, build_gemini_system_prompt_v2
from model.gemini_client import load_gemini_client

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)
logging.getLogger().setLevel(logging.WARNING)
logging.getLogger("faiss").setLevel(logging.ERROR)
logging.getLogger("embeddings").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("tokenizers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)


def sanitize_model_output(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I)
    cleaned = re.sub(r"</?think>", "", cleaned, flags=re.I)
    cleaned = re.sub(r"^\s*답변\s*[:\-]?\s*", "", cleaned, flags=re.M)
    cleaned = re.sub(r"^\s*(핵심 답변|주요 포인트|참고할 점)\s*[:\-]?\s*", "", cleaned, flags=re.M)
    cleaned = re.sub(r"^\s*[-•]\s*", "", cleaned, flags=re.M)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = cleaned.strip()
    return cleaned


def is_civil_law_question(query: str) -> bool:
    text = query.lower()
    civil_keywords = [
        "민사", "계약", "해지", "채무", "손해배상", "소유권", "상속", "임대", "임차", "법률", "판례"
    ]
    return any(keyword in text for keyword in civil_keywords)


def build_source_footer(query: str, search_results: list[dict]) -> str:
    if not is_civil_law_question(query):
        return ""

    if not search_results:
        return "\n\n## 출처\n- 검색된 관련 판례 및 법령이 없습니다."

    sources: list[str] = []
    for result in search_results[:3]:
        metadata = result.get("metadata", {}) if isinstance(result.get("metadata"), dict) else {}
        case_name = metadata.get("casenames") or metadata.get("title") or "관련 판결"
        court = metadata.get("normalized_court") or metadata.get("court") or "관련 기관"
        statute_name = metadata.get("statute_name") or metadata.get("statute_abbrv") or "관련 법령"
        sources.append(f"- {case_name} ({court}) / {statute_name}")

    if not sources:
        return "\n\n## 출처\n- 검색된 관련 판례 및 법령이 없습니다."

    return "\n\n## 출처\n" + "\n".join(sources)


def build_fallback_answer(query: str, search_results: list[dict]) -> str:
    if not search_results:
        return (
            "## 답변\n"
            "잘 모르겠습니다. 관련 데이터가 없어서 정확한 판단을 내리기 어렵습니다. "
            "그 대신 일반 법률 지식으로 답변하겠습니다.\n\n"
            "## 관련 법률\n"
            "- 검색된 관련 법령이 없습니다.\n\n"
            "## 참고 판례\n"
            "- 검색된 관련 판례가 없습니다.\n\n"
            "## 안내\n"
            "구체적인 사건은 전문가 상담이 필요할 수 있습니다."
        )

    first_result = search_results[0]
    metadata = first_result.get("metadata", {})
    case_name = metadata.get("casenames") or "관련 판결"
    court = metadata.get("normalized_court") or "관련 기관"
    statute_name = metadata.get("statute_name") or metadata.get("statute_abbrv") or "관련 법령"
    excerpt = (first_result.get("text") or "").replace("\n", " ")[:220]
    base_answer = (
        "## 답변\n"
        f"{query}와 관련해 검색된 자료에서는 {case_name}와 같은 사례가 확인됩니다. "
        f"{court}의 판결 사례와 {statute_name}을 참고하면 일반적인 방향을 파악할 수 있습니다. "
        f"핵심 문구는 '{excerpt}'와 같이 정리되어 있습니다.\n\n"
        "## 관련 법률\n"
        f"- {statute_name}\n\n"
        "## 참고 판례\n"
        f"- {case_name} ({court})\n\n"
        "## 안내\n"
        "검색 결과가 부족한 부분은 일반적인 법률 지식을 바탕으로 설명하였으며, 구체적인 사건은 전문가의 상담이 필요할 수 있습니다."
    )
    return base_answer + build_source_footer(query, search_results)


def build_fallback_answer_v2(query: str, search_results: list[dict]) -> str:
    if search_results:
        return build_fallback_answer(query, search_results)

    base_answer = (
        "## 답변\n"
        "잘 모르겠습니다. 관련 데이터가 없어서 정확한 판단을 내리기 어렵습니다. "
        "그 대신 일반 법률 지식으로 답변하겠습니다.\n\n"
        "## 관련 법률\n"
        "- 검색된 관련 법령이 없습니다.\n\n"
        "## 참고 판례\n"
        "- 검색된 관련 판례가 없습니다.\n\n"
        "## 안내\n"
        "법률과 직접 관련된 질문을 다시 보내주세요."
    )
    return base_answer + build_source_footer(query, search_results)


def run_chat(query: str, top_k: int = 3) -> str:
    from rag.retriever import load_retriever

    retriever = load_retriever(top_k=top_k)
    search_results = retriever.search(query, top_k=top_k)

    if not search_results:
        return build_fallback_answer(query, search_results)

    prompt = build_answer_prompt(query, search_results)

    try:
        gemini_client = load_gemini_client()
        response_text = gemini_client.generate_answer(prompt)
        cleaned = sanitize_model_output(response_text)
        if cleaned:
            return cleaned + build_source_footer(query, search_results)
    except Exception as exc:
        logger.warning("Gemini API 호출 실패, fallback 답변을 사용합니다: %s", exc)

    return build_fallback_answer(query, search_results)


def run_chat_v2(query: str, top_k: int = 3) -> str:
    from rag.retriever_v2 import load_retriever_v2

    retriever = load_retriever_v2(top_k=top_k)
    search_results = retriever.search(query, top_k=top_k)

    if not search_results:
        return build_fallback_answer_v2(query, search_results)

    prompt = build_answer_prompt_v2(query, search_results)

    try:
        gemini_client = load_gemini_client()
        response_text = gemini_client.generate_answer(prompt, system_prompt=build_gemini_system_prompt_v2())
        cleaned = sanitize_model_output(response_text)
        if cleaned:
            return cleaned + build_source_footer(query, search_results)
    except Exception as exc:
        logger.warning("Gemini API 호출 실패, fallback 답변을 사용합니다: %s", exc)

    return build_fallback_answer_v2(query, search_results)


def main() -> None:
    parser = argparse.ArgumentParser(description="Legal RAG CLI")
    parser.add_argument("query", nargs="?", default=None, help="질문을 입력하세요")
    parser.add_argument("--top-k", type=int, default=3, help="검색할 문서 수")
    args = parser.parse_args()

    query = args.query or input("질문을 입력하세요: ").strip()
    answer = run_chat(query, top_k=args.top_k)
    print(answer)


if __name__ == "__main__":
    main()
