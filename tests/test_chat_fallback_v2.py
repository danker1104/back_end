from chat import build_fallback_answer_v2


def test_build_fallback_answer_v2_with_no_results():
    prompt = build_fallback_answer_v2("보험사기 관련 정보 알려줘", [])

    assert "잘 모르겠습니다" in prompt
    assert "일반 법률 지식으로 답변하겠습니다" in prompt
    assert "검색된 관련 법령이 없습니다." in prompt
    assert "검색된 관련 판례가 없습니다." in prompt


def test_build_fallback_answer_v2_with_results_uses_existing():
    prompt = build_fallback_answer_v2(
        "보험사기 관련 정보 알려줘",
        [
            {
                "text": "보험사기는 형사범죄입니다.",
                "metadata": {
                    "casenames": "보험사기 사건",
                    "normalized_court": "서울중앙지방법원",
                    "statute_name": "보험사기방지 특별법",
                },
            }
        ],
    )

    assert "보험사기 사건" in prompt
    assert "보험사기방지 특별법" in prompt
