import pytest

from eval.llm_judge import (
    build_judge_prompt,
    judge_reply,
    parse_judge_response,
)
from llm_client import GroqLLMClient


def test_valid_judge_json_parses():
    result = parse_judge_response(
        '{"groundedness":4,"policy_adherence":5,"rationale":"Supported by evidence."}'
    )
    assert result.groundedness == 4
    assert result.policy_adherence == 5
    assert result.rationale == "Supported by evidence."


@pytest.mark.parametrize(
    "raw",
    [
        "not json",
        "[]",
        '{"groundedness":4,"policy_adherence":5}',
        '{"groundedness":6,"policy_adherence":5,"rationale":"bad"}',
        '{"groundedness":4.0,"policy_adherence":5,"rationale":"bad"}',
        '{"groundedness":4,"policy_adherence":5,"rationale":""}',
        '{"groundedness":4,"policy_adherence":5,"rationale":"ok","extra":1}',
    ],
)
def test_invalid_or_malformed_judge_json_is_rejected(raw):
    with pytest.raises(ValueError):
        parse_judge_response(raw)


def test_judge_prompt_contains_inputs():
    prompt = build_judge_prompt(
        "Why was I charged?",
        "Please send the receipt.",
        ["Please check your trip receipt."],
    )
    assert "Why was I charged?" in prompt
    assert "Please send the receipt." in prompt
    assert "Please check your trip receipt." in prompt


class FakeLLMClient(GroqLLMClient):
    def __init__(self, response):
        self.response = response
        self.prompts = []

    def complete(self, prompt, *, max_output_tokens=None):
        self.prompts.append((prompt, max_output_tokens))
        return self.response


def test_judge_uses_shared_llm_client_and_bounded_output():
    llm = FakeLLMClient(
        '{"groundedness":3,"policy_adherence":4,"rationale":"Reason."}'
    )
    result = judge_reply(
        "I need help.",
        "Please provide the receipt.",
        ["Please provide the receipt."],
        llm,
    )
    assert result.groundedness == 3
    assert result.policy_adherence == 4
    assert llm.prompts[0][1] == 250


def test_judge_rejects_invalid_client():
    with pytest.raises(TypeError, match="GroqLLMClient"):
        judge_reply("customer", "draft", ["evidence"], object())
