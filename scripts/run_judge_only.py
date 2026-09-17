"""Run the LLM judge over existing agent evaluation results.

This script does not rerun the support agent. It only judges existing
draft replies and their retrieved evidence.

Successful judgments are accumulated across runs. Already-judged tweet IDs
are skipped so API calls are not wasted on duplicate cases.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eval.llm_judge import judge_reply
from llm_client import GroqLLMClient, LLMConfig


def select_evenly_spaced(rows: list[dict], limit: int) -> list[dict]:
    if limit <= 0:
        return []

    if limit >= len(rows):
        return rows

    step = len(rows) / limit
    selected = []
    seen = set()

    for i in range(limit):
        index = min(int(i * step), len(rows) - 1)

        if index not in seen:
            seen.add(index)
            selected.append(rows[index])

    return selected


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default="data/eval/agent_eval_batch1.json",
    )

    parser.add_argument(
        "--output",
        default="data/eval/phase6_llm_judge.json",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=60,
        help="Maximum number of new rows to attempt.",
    )

    parser.add_argument(
        "--target",
        type=int,
        default=50,
        help="Target total number of successful judgments.",
    )

    parser.add_argument(
        "--max-calls",
        type=int,
        default=60,
        help="Maximum LLM calls for this run.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    if args.limit <= 0:
        raise ValueError("--limit must be positive.")

    if args.target <= 0:
        raise ValueError("--target must be positive.")

    if args.max_calls <= 0:
        raise ValueError("--max-calls must be positive.")

    input_path = ROOT / args.input
    output_path = ROOT / args.output

    data = json.loads(input_path.read_text(encoding="utf-8"))

    results = [
        row
        for row in data["results"]
        if row.get("draft_reply")
        and isinstance(row.get("evidence"), list)
    ]

    if not results:
        raise RuntimeError(
            "No completed agent results with drafts/evidence found."
        )

    # Preserve previous successful judgments.
    existing_judged = []
    existing_failures = []

    if output_path.exists():
        previous = json.loads(
            output_path.read_text(encoding="utf-8")
        )

        existing_judged = previous.get("judged", [])
        existing_failures = previous.get("failed", [])

    judged_ids = {
        str(row["tweet_id"])
        for row in existing_judged
        if row.get("tweet_id") is not None
    }

    # Only consider cases that have not already received a valid judgment.
    remaining = [
        row
        for row in results
        if str(row["tweet_id"]) not in judged_ids
    ]

    needed = max(
        args.target - len(existing_judged),
        0,
    )

    if needed == 0:
        print(
            f"Target already reached: "
            f"{len(existing_judged)} successful judgments."
        )
        return

    attempt_limit = min(
        args.limit,
        needed,
        args.max_calls,
        len(remaining),
    )

    selected = select_evenly_spaced(
        remaining,
        attempt_limit,
    )

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set."
        )

    client = GroqLLMClient(
        api_key=api_key,
        config=LLMConfig(
            max_calls=args.max_calls,
        ),
    )

    new_judged = []
    new_failures = []

    print(
        f"Available completed results: {len(results)}"
    )

    print(
        f"Existing successful judgments: "
        f"{len(existing_judged)}"
    )

    print(
        f"Remaining unjudged results: "
        f"{len(remaining)}"
    )

    print(
        f"Successful judgments needed: "
        f"{needed}"
    )

    print(
        f"New rows selected: "
        f"{len(selected)}"
    )

    print(
        f"Judge call budget: "
        f"{args.max_calls}"
    )

    for number, row in enumerate(selected, 1):
        tweet_id = str(row["tweet_id"])

        try:
            judge = judge_reply(
                customer_text=row["text"],
                draft_reply=row["draft_reply"],
                evidence=row["evidence"],
                llm_client=client,
            )

            new_judged.append(
                {
                    "tweet_id": tweet_id,
                    "human_intent_label": row.get(
                        "human_intent_label"
                    ),
                    "predicted_intent": row.get(
                        "predicted_intent"
                    ),
                    "draft_reply": row["draft_reply"],
                    "evidence": row["evidence"],
                    "groundedness": judge.groundedness,
                    "policy_adherence": judge.policy_adherence,
                    "rationale": judge.rationale,
                }
            )

            print(
                f"[{number}/{len(selected)}] "
                f"{tweet_id}: "
                f"groundedness={judge.groundedness}, "
                f"policy_adherence={judge.policy_adherence}"
            )

        except Exception as exc:
            new_failures.append(
                {
                    "tweet_id": tweet_id,
                    "error": (
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    ),
                }
            )

            print(
                f"[{number}/{len(selected)}] "
                f"{tweet_id}: FAILED - "
                f"{type(exc).__name__}: {exc}"
            )

    all_judged = existing_judged + new_judged
    all_failures = existing_failures + new_failures

    output = {
        "source": args.input,
        "selection": {
            "method": (
                "deterministic_even_spacing_over_unjudged_rows"
            ),
            "seed": args.seed,
            "requested_new_rows": args.limit,
            "selected_new_rows": len(selected),
        },
        "judged": all_judged,
        "failed": all_failures,
        "summary": {
            "successful_total": len(all_judged),
            "failed_total": len(all_failures),
            "successful_this_run": len(new_judged),
            "failed_this_run": len(new_failures),
            "llm_calls_used_this_run": client.calls_used,
            "output_tokens_used_this_run": (
                client.output_tokens_used
            ),
        },
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            output,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n=== Judge Summary ===")
    print(
        f"Successful total: "
        f"{len(all_judged)}"
    )
    print(
        f"Failed total: "
        f"{len(all_failures)}"
    )
    print(
        f"New successful: "
        f"{len(new_judged)}"
    )
    print(
        f"New failed: "
        f"{len(new_failures)}"
    )
    print(
        f"LLM calls used: "
        f"{client.calls_used}"
    )
    print(
        f"Output tokens: "
        f"{client.output_tokens_used}"
    )
    print(
        f"Wrote: "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()