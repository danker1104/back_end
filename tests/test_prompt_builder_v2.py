from inference.prompt_builder import build_answer_prompt_v2, build_gemini_system_prompt_v2


def test_build_gemini_system_prompt_v2_contains_rules():
    prompt = build_gemini_system_prompt_v2()

    assert "검색 결과가 있으면 검색 결과를 최우선 근거로 사용" in prompt
    assert "법률 조항, 판례, 사건번호는 검색 결과에 있을 때만 인용" in prompt
    assert "추측 금지" in prompt
    assert "검색 결과와 일반 설명을 명확히 구분" in prompt


def test_build_answer_prompt_v2_includes_summary_context_and_rules():
    prompt = build_answer_prompt_v2(
        "임금 체불 관련 질문",
        [
            {
                "summary": "임금 체불은 근로기준법 위반입니다.",
                "metadata": {
                    "title": "임금체불 판례",
                    "doc_class": "판례",
                },
            }
        ],
    )

    assert "검색 결과가 있으면 그 내용을 우선으로 사용하세요." in prompt
    assert "존재하지 않는 법률, 조항, 판례, 사건번호는 만들지 마세요." in prompt
    assert "일반 법률 지식으로 답변하겠습니다" in prompt
    assert "관련 질문이 아닙니다." in prompt
    assert "임금 체불은 근로기준법 위반입니다." in prompt
