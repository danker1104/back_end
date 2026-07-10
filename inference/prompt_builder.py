from typing import Any, Dict, List


def build_answer_prompt(query: str, search_results: List[Dict[str, Any]]) -> str:
    """Retriever 결과를 바탕으로 EXAONE에 전달할 프롬프트를 생성한다."""
    context_parts: List[str] = []
    for index, result in enumerate(search_results, start=1):
        text = result.get("text", "")
        metadata = result.get("metadata", {})
        source_path = metadata.get("source_path", "")
        doc_id = metadata.get("doc_id", "")
        doc_class = metadata.get("doc_class", "")
        casenames = metadata.get("casenames", "")
        normalized_court = metadata.get("normalized_court", "")
        statute_name = metadata.get("statute_name", "")
        statute_abbrv = metadata.get("statute_abbrv", "")

        context_parts.append(
            f"[근거 {index}]\n"
            f"- 문서ID: {doc_id}\n"
            f"- 문서유형: {doc_class}\n"
            f"- 사건명: {casenames}\n"
            f"- 법원/기관: {normalized_court}\n"
            f"- 법령명: {statute_name or statute_abbrv or '-'}\n"
            f"- 원문: {text or '(검색 결과에 텍스트가 없습니다)'}\n"
            f"- 출처: {source_path}\n"
        )

    context_block = "\n".join(context_parts) if context_parts else "- 검색된 근거가 없습니다."

    return f"""당신은 한국어 법률 정보를 바탕으로 실질적인 답변을 제공하는 도우미입니다.
다음 지침을 반드시 따르세요.

1. 아래 검색된 근거 자료의 내용을 직접 바탕으로 답변하세요.
2. 근거 자료에 포함된 사실과 판단 내용을 요약해서, 질문에 대한 실제 답을 알려주세요.
3. 검색된 자료에 없는 내용은 절대 추측하지 마세요.
4. 이 서비스는 전문 법률 자문이 아니라 일반적인 법률 정보 안내입니다.
5. 답변은 쉬운 한국어로 작성하되, 구체적인 사실관계와 판단 기준이 드러나게 설명하세요.
6. 가능하면 근거 자료에서 확인된 핵심 내용을 2~3문장 안에 포함해 설명하세요.
7. 내부 사고 과정이나 메타 설명은 쓰지 말고, 바로 최종 답변만 제공하세요.

질문:
{query}

검색된 근거 자료:
{context_block}

답변 형식:
- 먼저 질문에 대한 직접적인 답을 한 문단으로 말씀하세요.
- 그다음, 근거 자료에서 확인된 핵심 포인트를 짧게 설명하세요.
- 마지막에 참고할 점을 한두 문장으로 정리하세요.
"""


def build_simple_prompt(query: str, search_results: List[Dict[str, Any]]) -> str:
    """간단한 프롬프트 문자열을 반환한다."""
    return build_answer_prompt(query, search_results)
