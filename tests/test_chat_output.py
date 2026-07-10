from chat import sanitize_model_output


def test_sanitize_model_output_removes_think_blocks_and_labels():
    raw = """<think>이 부분은 내부 사고 과정입니다.</think>

답변:
계약 해제는 일반적으로 당사자 간 계약 관계를 끝내는 방법입니다.
"""

    cleaned = sanitize_model_output(raw)

    assert "<think>" not in cleaned
    assert "답변:" not in cleaned
    assert "계약 해제는 일반적으로" in cleaned
