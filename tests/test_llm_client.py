from types import SimpleNamespace

import pytest

from llm_client import (
    GroqLLMClient,
    LLMCallBudgetExceeded,
    LLMConfig,
)


class FakeCompletions:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)

        response = next(self.responses)

        if isinstance(response, Exception):
            raise response

        return response


class FakeClient:
    def __init__(self, responses):
        self.chat = SimpleNamespace(
            completions=FakeCompletions(responses)
        )


def make_response(text, completion_tokens=12):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=text)
            )
        ],
        usage=SimpleNamespace(
            completion_tokens=completion_tokens
        ),
    )


def test_complete_returns_model_text():
    fake = FakeClient(
        [make_response("hello from the mocked model")]
    )

    client = GroqLLMClient(
        api_key="test-key",
        client=fake,
    )

    result = client.complete("Say hello.")

    assert result == "hello from the mocked model"
    assert client.calls_used == 1
    assert client.output_tokens_used == 12


def test_complete_passes_expected_model_and_parameters():
    fake = FakeClient(
        [make_response("mocked response")]
    )

    config = LLMConfig(
        model="gpt-oss-120b",
        max_output_tokens=200,
    )

    client = GroqLLMClient(
        api_key="test-key",
        config=config,
        client=fake,
    )

    client.complete("Classify this message.")

    request = fake.chat.completions.calls[0]

    assert request["model"] == "gpt-oss-120b"
    assert request["temperature"] == 0
    assert request["max_tokens"] == 200
    assert request["messages"][0]["role"] == "user"


def test_retry_is_bounded_and_eventually_succeeds():
    fake = FakeClient(
        [
            RuntimeError("temporary failure"),
            make_response("success"),
        ]
    )

    sleeps = []

    client = GroqLLMClient(
        api_key="test-key",
        config=LLMConfig(max_retries=1),
        client=fake,
        sleep_fn=sleeps.append,
    )

    result = client.complete("Retry this.")

    assert result == "success"
    assert len(fake.chat.completions.calls) == 2
    assert client.calls_used == 2
    assert sleeps == [0.5]


def test_retry_failure_stops_after_configured_attempts():
    fake = FakeClient(
        [
            RuntimeError("failure 1"),
            RuntimeError("failure 2"),
            RuntimeError("failure 3"),
        ]
    )

    sleeps = []

    client = GroqLLMClient(
        api_key="test-key",
        config=LLMConfig(max_retries=2),
        client=fake,
        sleep_fn=sleeps.append,
    )

    with pytest.raises(RuntimeError, match="failed after 3 attempt"):
        client.complete("This will fail.")

    assert len(fake.chat.completions.calls) == 3
    assert client.calls_used == 3
    assert sleeps == [0.5, 1.0]


def test_call_budget_is_enforced():
    fake = FakeClient(
        [
            make_response("first"),
            make_response("second"),
        ]
    )

    client = GroqLLMClient(
        api_key="test-key",
        config=LLMConfig(
            max_calls=1,
            max_output_tokens=100,
        ),
        client=fake,
    )

    assert client.complete("First") == "first"

    with pytest.raises(LLMCallBudgetExceeded):
        client.complete("Second")

    assert client.calls_used == 1


def test_call_budget_limits_retries():
    fake = FakeClient(
        [
            RuntimeError("failure 1"),
            RuntimeError("failure 2"),
            RuntimeError("failure 3"),
        ]
    )

    client = GroqLLMClient(
        api_key="test-key",
        config=LLMConfig(
            max_calls=1,
            max_retries=2,
        ),
        client=fake,
        sleep_fn=lambda _: None,
    )

    with pytest.raises(LLMCallBudgetExceeded):
        client.complete("This must not retry past the budget.")

    assert len(fake.chat.completions.calls) == 1
    assert client.calls_used == 1


def test_empty_prompt_is_rejected():
    fake = FakeClient([])

    client = GroqLLMClient(
        api_key="test-key",
        client=fake,
    )

    with pytest.raises(ValueError, match="non-empty"):
        client.complete("   ")


def test_missing_api_key_is_rejected(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        GroqLLMClient()