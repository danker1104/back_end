from typing import Any, Dict, List


def build_gemini_system_prompt() -> str:
    """Gemini에 전달할 시스템 프롬프트를 생성한다."""
    return (
        "당신은 대한민국 법률 AI 비서 역할을 수행합니다. "
        "당신은 대한민국 법률 AI 비서입니다. "
        "검색된 법률 정보만 이용하되, 검색 결과가 부족하면 일반 지식으로 보완할 수 있습니다. "
        "검색되지 않은 내용은 추측 금지하며, 추측하지 말고 일반적인 설명으로 한정하세요. "
        "일반인이 이해하기 쉽게 설명합니다. "
        "사용자의 질문에 대해 먼저 제공된 RAG 검색 결과를 가장 중요한 근거로 사용하여 답변하세요. "
        "검색 결과가 충분한 경우에는 검색 결과를 기반으로 답변하고, 관련 법령과 판례를 함께 설명하세요. "
        "검색 결과가 부족하거나 사용자의 질문이 일반적인 개념 설명인 경우에는 일반 지식을 활용하여 부족한 부분을 보완할 수 있습니다. "
        "필요한 경우 전문가 상담을 권장합니다. "
        "단, 존재하지 않는 법률, 판례, 조항을 생성하거나 추측해서는 안 됩니다. "
        "검색 결과가 없는 법률 내용은 반드시 일반적인 설명이라는 점을 명확히 알려주세요."
    )


def build_answer_prompt(query: str, search_results: List[Dict[str, Any]]) -> str:
    """Retriever 결과를 바탕으로 Gemini에 전달할 프롬프트를 생성한다."""
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

    if context_parts:
        context_block = "\n".join(context_parts)
        empty_search_guidance = ""
    else:
        context_block = (
            "검색된 법률 자료가 없습니다.\n"
            "법률과 직접 관련되지 않은 질문이면 '관련 질문이 아닙니다.'라고 답변하세요.\n"
            "단, 존재하지 않는 법률, 판례, 조항은 생성하지 마세요."
        )
        empty_search_guidance = (
            "검색 결과가 없으면 아래 형식으로 답변하세요.\n"
            "- 검색된 관련 법령이 없습니다.\n"
            "- 검색된 관련 판례가 없습니다.\n"
            "- 잘 모르겠습니다. 관련 데이터가 없어서 정확한 판단을 내리기 어렵습니다. 그 대신 일반 법률 지식으로 답변하겠습니다."
        )

    return f"""당신은 대한민국 법률 AI 비서입니다.

답변은 짧고 핵심적으로 작성하세요.
- 불필요한 설명은 생략하고, 바로 핵심만 말하세요.
- 청소년 사용자를 위해 너무 겁주거나 과하게 단정하지 말고, 부드럽고 예방 중심으로 답하세요.
- 검색 결과가 있으면 그 내용을 바탕으로 간단히 답하세요.
- 검색 결과가 없으면 '잘 모르겠습니다. 관련 데이터가 없어서 정확한 판단을 내리기 어렵습니다. 그 대신 일반 법률 지식으로 답변하겠습니다.'라고 답하세요.
- 법률과 직접 관련되지 않은 질문이면 '관련 질문이 아닙니다.'라고 답변하세요.
- 존재하지 않는 법률, 조항, 판례, 사건번호는 만들지 마세요.
- 필요하면 안전하게 행동하는 방법이나 주변 도움을 받는 방법을 짧게 안내하세요.

# 사용자 질문
{query}

# RAG 검색 결과
{context_block}

{empty_search_guidance}
"""


def build_simple_prompt(query: str, search_results: List[Dict[str, Any]]) -> str:
    """간단한 프롬프트 문자열을 반환한다."""
    return build_answer_prompt(query, search_results)


def build_gemini_system_prompt_v2() -> str:
    return (
        "당신은 대한민국 법률 AI 비서입니다. "
        "검색 결과가 있으면 검색 결과를 최우선 근거로 사용하고, "
        "검색 결과가 부족할 때만 Gemini 일반 지식을 보완합니다. "
        "법률 조항, 판례, 사건번호는 검색 결과에 있을 때만 인용하며 추측 금지합니다. "
        "검색 결과와 일반 설명을 명확히 구분하십시오. "
        "일반인도 이해하기 쉬운 문장으로 답변합니다."
    )


def build_answer_prompt_v2(query: str, search_results: List[Dict[str, Any]]) -> str:
    context_parts: List[str] = []
    for index, result in enumerate(search_results, start=1):
        metadata = result.get("metadata", {}) if isinstance(result.get("metadata"), dict) else {}
        title = str(
            result.get("title")
            or metadata.get("title")
            or metadata.get("doc_id")
            or result.get("doc_id", "")
        ).strip()
        doc_class = str(
            result.get("doc_type")
            or metadata.get("doc_class")
            or metadata.get("doc_type")
            or ""
        ).strip()
        summary = str(
            result.get("summary")
            or metadata.get("summary")
            or result.get("text")
            or ""
        ).strip()
        if not summary:
            summary = "(검색 결과 요약이 없습니다.)"

        context_parts.append(
            f"[근거 {index}] {title} | {doc_class}\n{summary}"
        )

    if context_parts:
        context_block = "\n\n".join(context_parts)
        empty_guidance = ""
    else:
        context_block = (
            "검색된 법률 자료가 없습니다.\n"
            "법률과 직접 관련되지 않은 질문이면 '관련 질문이 아닙니다.'라고 답변하세요.\n"
            "일반적인 법률 지식을 활용하되, 존재하지 않는 법률, 판례, 조항, 사건번호는 추측하지 마세요."
        )
        empty_guidance = (
            "검색 결과가 없으면 아래 형식으로 작성하세요.\n"
            "- 검색된 관련 법령이 없습니다.\n"
            "- 검색된 관련 판례가 없습니다.\n"
            "- 잘 모르겠습니다. 관련 데이터가 없어서 정확한 판단을 내리기 어렵습니다. 그 대신 일반 법률 지식으로 답변하겠습니다."
        )

    return f"""당신은 대한민국 법률 AI 비서입니다.

짧고 핵심적인 답변만 작성하세요.
- 법률 용어 설명은 생략하고, 바로 핵심 판단만 말하세요.
- 청소년 사용자를 위해 너무 겁주거나 과하게 단정하지 말고, 부드럽고 예방 중심으로 답하세요.
- 검색 결과가 있으면 그 내용을 우선으로 사용하세요.
- 검색 결과가 없으면 '잘 모르겠습니다. 관련 데이터가 없어서 정확한 판단을 내리기 어렵습니다. 그 대신 일반 법률 지식으로 답변하겠습니다.'라고 답변하세요.
- 법률과 직접 관련되지 않은 질문이면 '관련 질문이 아닙니다.'라고 답변하세요.
- 존재하지 않는 법률, 조항, 판례, 사건번호는 만들지 마세요.
- 필요하면 안전하게 행동하는 방법이나 주변 도움을 받는 방법을 짧게 안내하세요.

# 사용자 질문
{query}

# 검색 결과 요약
{context_block}

{empty_guidance}
"""


def build_simple_prompt_v2(query: str, search_results: List[Dict[str, Any]]) -> str:
    return build_answer_prompt_v2(query, search_results)
