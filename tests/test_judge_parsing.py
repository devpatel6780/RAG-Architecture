import pytest

from eval.judge import parse_judge_response


def test_parse_plain_json():
    raw = '{"faithfulness": 5, "relevance": 4, "completeness": 3, "reasoning": "ok"}'
    result = parse_judge_response(raw)
    assert result == {"faithfulness": 5, "relevance": 4, "completeness": 3, "reasoning": "ok"}


def test_parse_json_wrapped_in_think_tags():
    raw = '<think>reasoning about the answer here</think>\n{"faithfulness": 1, "relevance": 2, "completeness": 3, "reasoning": "bad"}'
    result = parse_judge_response(raw)
    assert result["faithfulness"] == 1
    assert result["relevance"] == 2
    assert result["completeness"] == 3


def test_parse_json_wrapped_in_markdown_fence():
    raw = '```json\n{"faithfulness": 4, "relevance": 4, "completeness": 4, "reasoning": "x"}\n```'
    result = parse_judge_response(raw)
    assert result["faithfulness"] == 4


def test_parse_clamps_out_of_range_scores():
    raw = '{"faithfulness": 0, "relevance": 9, "completeness": -3, "reasoning": "x"}'
    result = parse_judge_response(raw)
    assert result["faithfulness"] == 1
    assert result["relevance"] == 5
    assert result["completeness"] == 1


def test_parse_malformed_input_raises_value_error():
    with pytest.raises(ValueError):
        parse_judge_response("this is not json at all")


def test_parse_missing_axis_raises_key_error():
    raw = '{"faithfulness": 5, "relevance": 4}'
    with pytest.raises(KeyError):
        parse_judge_response(raw)
