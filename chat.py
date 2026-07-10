import argparse
import logging
import re

import torch

from inference.prompt_builder import build_answer_prompt
from model.load_model import load_exaone_model
from rag.retriever import load_retriever

logging.basicConfig(level=logging.INFO)
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


def build_fallback_answer(query: str, search_results: list[dict]) -> str:
    if not search_results:
        return (
            f"{query}와 관련해 현재 검색된 자료가 충분하지 않아, 구체적인 판단을 내리기 어렵습니다. "
            "사건의 사실관계와 관련 법령을 함께 확인해 보세요."
        )

    first_result = search_results[0]
    metadata = first_result.get("metadata", {})
    case_name = metadata.get("casenames") or "관련 판결"
    court = metadata.get("normalized_court") or "관련 기관"
    return (
        f"{query}와 관련해 검색된 자료에서는 {case_name}와 같은 사례가 확인됩니다. "
        f"{court}의 판결 사례를 참고하면 일반적인 방향을 파악할 수 있지만, "
        "구체적인 상황에 따라 결과는 달라질 수 있습니다."
    )


def run_chat(query: str, top_k: int = 3) -> str:
    retriever = load_retriever(top_k=top_k)
    search_results = retriever.search(query, top_k=top_k)
    prompt = build_answer_prompt(query, search_results)

    tokenizer, model = load_exaone_model()
    inputs = tokenizer(prompt, return_tensors="pt")
    model.to(inputs["input_ids"].device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=False,
        )

    decoded = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    cleaned = sanitize_model_output(decoded.strip())
    if not cleaned:
        return build_fallback_answer(query, search_results)
    return cleaned


def main() -> None:
    parser = argparse.ArgumentParser(description="Legal RAG CLI")
    parser.add_argument("query", nargs="?", default=None, help="질문을 입력하세요")
    parser.add_argument("--top-k", type=int, default=3, help="검색할 문서 수")
    args = parser.parse_args()

    query = args.query or input("질문을 입력하세요: ").strip()
    answer = run_chat(query, top_k=args.top_k)
    print("\n답변:\n")
    print(answer)


if __name__ == "__main__":
    main()
