
"""Budget-bounded LLM judge for draft support replies."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Sequence

from llm_client import GroqLLMClient


@dataclass(frozen=True)
class JudgeResult:
    groundedness: int
    policy_adherence: int
    rationale: str


def build_judge_prompt(
    customer_text: str,
    draft_reply: str,
    evidence: Sequence[str],
) -> str:
    if not isinstance(customer_text, str) or not customer_text.strip():
        raise ValueError("customer_text must be non-empty")
    if not isinstance(draft_reply, str) or not draft_reply.strip():
        raise ValueError("draft_reply must be non-empty")

    evidence_text = "\n".join(
        f"{index + 1}. {text}"
        for index, text in enumerate(evidence)
        if isinstance(text, str) and text.strip()
    )
    if not evidence_text:
        raise ValueError("evidence must contain at least one non-empty item")

    return f"""Evaluate this Uber Support draft reply.

Score each dimension from 1 to 5:
- groundedness: whether the reply is supported by the supplied historical evidence
  and does not invent customer/account-specific facts.
- policy_adherence: whether the reply is safe, appropriate, non-deceptive,
  avoids unsupported promises, and follows the supplied support context.

Evidence:
{evidence_text}

Customer tweet:
{customer_text.strip()}

Draft reply:
{draft_reply.strip()}

Return ONLY one JSON object.
Do not use Markdown fences.
Do not add commentary before or after the JSON.
Use exactly these fields:
- groundedness: integer 1-5
- policy_adherence: integer 1-5
- rationale: brief string

Example:
{{"groundedness": 3, "policy_adherence": 4, "rationale": "Brief explanation."}}
"""


def _extract_json_object(raw: str) -> Any:
    """Extract a JSON object without interpreting arbitrary prose as JSON."""
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("judge response was empty")

    text = raw.strip()

    # First try the complete response.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Accept a single JSON object wrapped in Markdown fences or surrounded
    # by harmless whitespace/commentary, but still require valid JSON.
    fenced = re.search(
        r"```(?:json)?\s*(\{.*\})\s*```",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass

    # Find the first balanced JSON object. This avoids blindly slicing text
    # between the first and last braces when the rationale contains braces.
    start = text.find("{")
    if start >= 0:
        depth = 0
        in_string = False
        escaped = False

        for index in range(start, len(text)):
            char = text[index]

            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue

            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : index + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break

    raise ValueError("judge response was not valid JSON")


def parse_judge_response(raw: str) -> JudgeResult:
    """Parse judge JSON and validate its schema/ranges."""
    payload = _extract_json_object(raw)

    if not isinstance(payload, dict):
        raise ValueError("judge response must be a JSON object")

    required = {"groundedness", "policy_adherence", "rationale"}
    if set(payload) != required:
        raise ValueError(
            "judge response must contain exactly groundedness, "
            "policy_adherence, and rationale"
        )

    groundedness = payload["groundedness"]
    policy = payload["policy_adherence"]
    rationale = payload["rationale"]

    for name, score in (
        ("groundedness", groundedness),
        ("policy_adherence", policy),
    ):
        if isinstance(score, bool) or not isinstance(score, int):
            raise ValueError(f"{name} must be an integer from 1 to 5")
        if not 1 <= score <= 5:
            raise ValueError(f"{name} must be an integer from 1 to 5")

    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError("rationale must be a non-empty string")

    return JudgeResult(
        groundedness=groundedness,
        policy_adherence=policy,
        rationale=rationale.strip(),
    )


def judge_reply(
    customer_text: str,
    draft_reply: str,
    evidence: Sequence[str],
    llm_client: GroqLLMClient,
) -> JudgeResult:
    """Evaluate one draft through the shared budgeted Groq client."""
    if not isinstance(llm_client, GroqLLMClient):
        raise TypeError("llm_client must be a GroqLLMClient")

    prompt = build_judge_prompt(customer_text, draft_reply, evidence)
    raw = llm_client.complete(prompt, max_output_tokens=250)

    if not isinstance(raw, str):
        raise TypeError("llm_client.complete must return a string")

    return parse_judge_response(raw)
