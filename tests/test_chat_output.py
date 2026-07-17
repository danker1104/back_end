from chat import sanitize_model_output
from inference.prompt_builder import build_answer_prompt, build_gemini_system_prompt


def test_sanitize_model_output_removes_think_blocks_and_labels():
    raw = """<think>이 부분은 내부 사고 과정입니다.</think>

답변:
계약 해제는 일반적으로 당사자 간 계약 관계를 끝내는 방법입니다.
"""

    cleaned = sanitize_model_output(raw)

    assert "<think>" not in cleaned
    assert "답변:" not in cleaned
    assert "계약 해제는 일반적으로" in cleaned


def test_build_answer_prompt_requires_evidence_based_rules():
    prompt = build_answer_prompt(
        "중도해지 가능한가요?",
        [
            {
                "text": "계약 해지는 당사자 합의가 필요합니다.",
                "metadata": {
                    "source_path": "test.json",
                    "doc_id": "doc-1",
                    "doc_class": "판례",
                    "casenames": "사례",
                    "normalized_court": "대법원",
                    "statute_name": "민법",
                    "statute_abbrv": "",
                },
            }
        ],
    )

    assert "검색 결과가 있으면 그 내용을 바탕으로 간단히 답하세요." in prompt
    assert "검색 결과가 없으면 '잘 모르겠습니다. 관련 데이터가 없어서 정확한 판단을 내리기 어렵습니다. 그 대신 일반 법률 지식으로 답변하겠습니다.'라고 답하세요." in prompt
    assert "법률과 직접 관련되지 않은 질문이면 '관련 질문이 아닙니다.'라고 답변하세요." in prompt


def test_build_answer_prompt_includes_empty_search_guidance():
    prompt = build_answer_prompt("보험사기 관련 정보 알려줘", [])

    assert "검색된 법률 자료가 없습니다." in prompt
    assert "법률과 직접 관련되지 않은 질문이면 '관련 질문이 아닙니다.'라고 답변하세요." in prompt
    assert "존재하지 않는 법률, 판례, 조항은 생성하지 마세요." in prompt


def test_build_gemini_system_prompt_contains_required_rules():
    system_prompt = build_gemini_system_prompt()

    assert "대한민국 법률 AI 비서 역할" in system_prompt
    assert "검색된 법률 정보만 이용" in system_prompt
    assert "검색되지 않은 내용은 추측 금지" in system_prompt
    assert "일반인이 이해하기 쉽게 설명" in system_prompt
    assert "전문가 상담을 권장" in system_prompt
