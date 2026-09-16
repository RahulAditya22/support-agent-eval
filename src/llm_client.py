"""Groq LLM client with bounded retries, timeout, and call/token budgets."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

from openai import OpenAI


@dataclass(frozen=True)
class LLMConfig:
    """Configuration for the Groq OpenAI-compatible client."""

    model: str = "gpt-oss-120b"
    base_url: str = "https://api.groq.com/openai/v1"
    timeout_seconds: float = 30.0
    max_retries: int = 2
    max_calls: int = 20
    max_output_tokens: int = 500


class LLMCallBudgetExceeded(RuntimeError):
    """Raised when the configured request budget has been exhausted."""


class GroqLLMClient:
    """Small provider wrapper used by classification and generation modules."""

    def __init__(
        self,
        api_key: str | None = None,
        config: LLMConfig | None = None,
        client=None,
        sleep_fn=time.sleep,
    ) -> None:
        self.config = config or LLMConfig()

        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GROQ_API_KEY is required for the real LLM client."
            )

        self._client = client or OpenAI(
            api_key=self.api_key,
            base_url=self.config.base_url,
            timeout=self.config.timeout_seconds,
            max_retries=0,
        )

        self._sleep = sleep_fn
        self.calls_used = 0
        self.output_tokens_used = 0

    @property
    def remaining_calls(self) -> int:
        """Return the number of LLM calls still available."""
        return max(0, self.config.max_calls - self.calls_used)

    @property
    def remaining_output_tokens(self) -> int:
        """Return the configured output-token budget still available."""
        return max(
            0,
            self.config.max_calls * self.config.max_output_tokens
            - self.output_tokens_used,
        )

    def complete(
        self,
        prompt: str,
        *,
        max_output_tokens: int | None = None,
    ) -> str:
        """Send one chat completion request with bounded retries."""

        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")

        requested_tokens = (
            max_output_tokens
            if max_output_tokens is not None
            else self.config.max_output_tokens
        )

        if requested_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")

        if (
            self.output_tokens_used + requested_tokens
            > self.config.max_calls * self.config.max_output_tokens
        ):
            raise LLMCallBudgetExceeded("LLM output-token budget exhausted.")

        last_error: Exception | None = None

        for attempt in range(self.config.max_retries + 1):
            # IMPORTANT:
            # Check the call budget before EVERY individual attempt.
            # A retry is also a real API call and must consume a budget slot.
            if self.calls_used >= self.config.max_calls:
                raise LLMCallBudgetExceeded(
                    f"LLM call budget exhausted: "
                    f"{self.config.max_calls} calls."
                )

            try:
                self.calls_used += 1

                response = self._client.chat.completions.create(
                    model=self.config.model,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                    temperature=0,
                    max_tokens=requested_tokens,
                )

                content = response.choices[0].message.content

                if not isinstance(content, str) or not content.strip():
                    raise ValueError("LLM returned an empty response")

                usage = getattr(response, "usage", None)
                completion_tokens = getattr(
                    usage,
                    "completion_tokens",
                    None,
                )

                if isinstance(completion_tokens, int):
                    self.output_tokens_used += completion_tokens
                else:
                    self.output_tokens_used += requested_tokens

                return content.strip()

            except Exception as exc:
                last_error = exc

                if attempt >= self.config.max_retries:
                    break

                self._sleep(0.5 * (2**attempt))

        raise RuntimeError(
            f"LLM request failed after "
            f"{self.config.max_retries + 1} attempt(s)."
        ) from last_error