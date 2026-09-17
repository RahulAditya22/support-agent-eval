"""Groq LLM client with bounded retries, timeout, and call/token budgets."""

from __future__ import annotations

import os
import random
import re
import time
from dataclasses import dataclass

from openai import OpenAI


@dataclass(frozen=True)
class LLMConfig:
    """Configuration for the Groq OpenAI-compatible client."""

    model: str = "openai/gpt-oss-120b"
    base_url: str = "https://api.groq.com/openai/v1"

    timeout_seconds: float = 30.0

    # Maximum number of retries after the initial request.
    max_retries: int = 2

    # Application-level safety budget.
    max_calls: int = 100

    # Maximum completion tokens requested by default.
    max_output_tokens: int = 500

    # Minimum delay used when a retryable error does not provide
    # a server-suggested retry interval.
    retry_base_seconds: float = 1.0

    # Maximum delay imposed by our client for a single retry.
    retry_max_seconds: float = 10.0

    # Small random jitter prevents repeated requests from lining up
    # exactly with the same rate-limit boundary.
    retry_jitter_seconds: float = 0.25


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
        random_fn=random.random,
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
        self._random = random_fn

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
            (
                self.config.max_calls * self.config.max_output_tokens
                - self.output_tokens_used
            ),
        )

    @staticmethod
    def _retry_after_seconds(error: Exception) -> float | None:
        """Extract a server-provided retry delay when available.

        Groq rate-limit errors can contain text such as:
            "Please try again in 922.5ms."
        or:
            "Please try again in 2s."
        """

        message = str(error)

        patterns = (
            (
                r"try again in\s+([0-9]+(?:\.[0-9]+)?)\s*ms",
                0.001,
            ),
            (
                r"try again in\s+([0-9]+(?:\.[0-9]+)?)\s*s",
                1.0,
            ),
        )

        for pattern, multiplier in patterns:
            match = re.search(
                pattern,
                message,
                flags=re.IGNORECASE,
            )

            if match:
                return float(match.group(1)) * multiplier

        return None

    @staticmethod
    def _is_rate_limit_error(error: Exception) -> bool:
        """Return True when the exception appears to be an HTTP 429."""

        status_code = getattr(error, "status_code", None)

        if status_code == 429:
            return True

        message = str(error).lower()

        return (
            "rate limit" in message
            or "rate_limit_exceeded" in message
            or "too many requests" in message
            or "error code: 429" in message
        )

    @staticmethod
    def _is_retryable_error(error: Exception) -> bool:
        """Return whether an exception is safe to retry."""

        if GroqLLMClient._is_rate_limit_error(error):
            return True

        status_code = getattr(error, "status_code", None)

        # Server-side failures are normally transient.
        if isinstance(status_code, int):
            if status_code >= 500:
                return True

            # Other 4xx errors generally indicate a bad request,
            # invalid credentials, or another non-transient issue.
            return False

        error_name = type(error).__name__.lower()

        retryable_names = (
            "timeout",
            "connection",
            "apierror",
            "serviceunavailable",
            "internalserver",
        )

        return any(
            name in error_name
            for name in retryable_names
        )

    def _retry_delay(
        self,
        error: Exception,
        attempt: int,
    ) -> float:
        """Calculate a bounded retry delay.

        Priority:
        1. Server-provided retry interval for rate limits.
        2. Existing deterministic exponential backoff for other errors.
        3. Small jitter only when using a server-provided delay.

        The deterministic fallback preserves the original client
        retry contract of 0.5s, 1.0s, 2.0s, ...
        """

        server_delay = self._retry_after_seconds(error)

        if server_delay is not None:
            jitter = (
                self._random()
                * self.config.retry_jitter_seconds
            )

            return min(
                server_delay + jitter,
                self.config.retry_max_seconds,
            )

        # Preserve the original retry timing:
        # attempt 0 -> 0.5s
        # attempt 1 -> 1.0s
        # attempt 2 -> 2.0s
        return min(
            0.5 * (2**attempt),
            self.config.retry_max_seconds,
        )

    def complete(
        self,
        prompt: str,
        *,
        max_output_tokens: int | None = None,
    ) -> str:
        """Send one chat completion request with bounded retries.

        Every API attempt consumes one call from the application
        call budget, including retries.

        Rate-limit/server/network errors use either the
        server-provided retry interval or bounded exponential
        backoff.

        Generic RuntimeError is also retried to preserve the
        client's existing retry contract and test behavior.
        """

        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(
                "prompt must be a non-empty string"
            )

        requested_tokens = (
            max_output_tokens
            if max_output_tokens is not None
            else self.config.max_output_tokens
        )

        if requested_tokens <= 0:
            raise ValueError(
                "max_output_tokens must be positive"
            )

        if (
            self.output_tokens_used + requested_tokens
            > (
                self.config.max_calls
                * self.config.max_output_tokens
            )
        ):
            raise LLMCallBudgetExceeded(
                "LLM output-token budget exhausted."
            )

        last_error: Exception | None = None

        for attempt in range(self.config.max_retries + 1):

            # A retry is also a real API call.
            if self.calls_used >= self.config.max_calls:
                raise LLMCallBudgetExceeded(
                    "LLM call budget exhausted: "
                    f"{self.config.max_calls} calls."
                )

            try:
                self.calls_used += 1

                response = (
                    self._client.chat.completions.create(
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
                )

                content = (
                    response.choices[0]
                    .message
                    .content
                )

                if (
                    not isinstance(content, str)
                    or not content.strip()
                ):
                    raise ValueError(
                        "LLM returned an empty response"
                    )

                usage = getattr(
                    response,
                    "usage",
                    None,
                )

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

                # Generic RuntimeError is kept retryable because
                # existing client tests and callers rely on that
                # bounded retry behavior.
                retryable = (
                    isinstance(exc, RuntimeError)
                    or self._is_retryable_error(exc)
                )

                if not retryable:
                    raise RuntimeError(
                        "LLM request failed with "
                        f"{type(exc).__name__}: {exc}"
                    ) from exc

                # No attempts remaining.
                if attempt >= self.config.max_retries:
                    break

                delay = self._retry_delay(
                    exc,
                    attempt,
                )

                if self._is_rate_limit_error(exc):
                    print(
                        "LLM rate limit encountered; "
                        f"waiting {delay:.2f}s before retry "
                        f"(attempt {attempt + 2}/"
                        f"{self.config.max_retries + 1}).",
                        flush=True,
                    )

                self._sleep(delay)

        if last_error is None:
            raise RuntimeError(
                "LLM request failed without "
                "an underlying exception."
            )

        raise RuntimeError(
            "LLM request failed after "
            f"{self.config.max_retries + 1} attempt(s): "
            f"{type(last_error).__name__}: "
            f"{last_error}"
        ) from last_error