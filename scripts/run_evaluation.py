"""Safe evaluation runner for the human-owned golden set.

Real mode:
- Loads the human-reviewed golden set.
- Runs the production pipeline without using human labels as predictions.
- Uses a bounded Groq client.
- Checkpoints after every processed row.
- Records ordinary row failures and continues.
- Stops cleanly when the LLM call budget is exhausted.
- Supports resume from a checkpoint.
- Supports retrying previously failed rows.
- Optionally runs the LLM judge.

Mock mode:
- Exercises the exact evaluation/checkpoint control flow without API calls.
- Must never be interpreted as model performance.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from sentence_transformers import SentenceTransformer

from eval.llm_judge import judge_reply
from src.corpus import build_uber_reply_retriever
from src.escalate import decide_escalation
from llm_client import GroqLLMClient, LLMCallBudgetExceeded, LLMConfig
from pipeline import run_pipeline
from retrieve import ReplyRetriever


TAXONOMY = {
    "cancellation_no_show_charge",
    "unauthorized_transaction",
    "fare_price_dispute",
    "missing_incomplete_ride",
    "uber_eats_order_delivery_issue",
    "ride_pass_promotion_coupon",
    "account_login_app_access",
    "payment_refund_billing",
    "driver_partner_safety_issue",
    "other_out_of_scope",
}


REQUIRED_COLUMNS = {
    "tweet_id",
    "text",
    "predicted_intent",
    "human_intent_label",
    "human_auto_handle_label",
    "notes",
}


def load_golden_set(path: Path) -> list[dict[str, str]]:
    """Load and structurally validate the human-owned golden set."""
    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise ValueError(f"Golden set is empty: {path}")

    missing = REQUIRED_COLUMNS - set(rows[0].keys())

    if missing:
        raise ValueError(
            f"Golden set missing required columns: {sorted(missing)}"
        )

    seen_ids: set[str] = set()

    for index, row in enumerate(rows, start=2):
        tweet_id = str(row["tweet_id"]).strip()
        text = str(row["text"]).strip()
        human_intent = str(
            row["human_intent_label"]
        ).strip()
        human_auto = str(
            row["human_auto_handle_label"]
        ).strip().upper()

        if not tweet_id:
            raise ValueError(
                f"Row {index}: missing tweet_id"
            )

        if tweet_id in seen_ids:
            raise ValueError(
                f"Row {index}: duplicate tweet_id={tweet_id}"
            )

        seen_ids.add(tweet_id)

        if not text:
            raise ValueError(
                f"Row {index}: missing text"
            )

        if human_intent not in TAXONOMY:
            raise ValueError(
                f"Row {index}: invalid "
                f"human_intent_label={human_intent!r}"
            )

        if human_auto not in {"TRUE", "FALSE"}:
            raise ValueError(
                f"Row {index}: "
                "human_auto_handle_label must be TRUE/FALSE"
            )

    return rows


def load_checkpoint(
    path: Path,
) -> list[dict[str, Any]]:
    """Load checkpoint results from disk."""
    if not path.exists():
        return []

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        payload = json.load(handle)

    if isinstance(payload, list):
        return payload

    if not isinstance(payload, dict) or "results" not in payload:
        raise ValueError(
            f"Invalid checkpoint format: {path}"
        )

    results = payload["results"]

    if not isinstance(results, list):
        raise ValueError(
            f"Invalid checkpoint results: {path}"
        )

    return results


def save_checkpoint(
    path: Path,
    results: list[dict[str, Any]],
    *,
    mode: str,
    completed: int,
    total: int,
) -> None:
    """Atomically save the current evaluation checkpoint."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "mode": mode,
        "completed": completed,
        "total": total,
        "results": results,
    }

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    with temporary.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            ensure_ascii=False,
        )

    temporary.replace(path)


def save_results(
    path: Path,
    results: list[dict[str, Any]],
    *,
    mode: str,
) -> None:
    """Save the final evaluation output."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    successful = sum(
        1
        for result in results
        if result.get(
            "status",
            "completed",
        )
        == "completed"
    )

    failed = sum(
        1
        for result in results
        if result.get("status") == "error"
    )

    payload = {
        "mode": mode,
        "processed": len(results),
        "successful": successful,
        "failed": failed,
        "results": results,
    }

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    with temporary.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            ensure_ascii=False,
        )

    temporary.replace(path)


def mock_prediction(
    row: dict[str, str],
    top_k: int,
) -> dict[str, Any]:
    """Control-flow-only prediction.

    This intentionally does not copy human labels into predictions.
    """
    return {
        "predicted_intent": row["predicted_intent"],
        "confidence": None,
        "reason": "mock_control_flow_only",
        "predicted_escalate": False,
        "escalation_reason": "mock_control_flow_only",
        "draft_reply": "",
        "evidence": [],
        "retrieval_similarities": [],
        "judge": None,
        "top_k": top_k,
    }


def build_retriever(
    raw_data: Path,
) -> ReplyRetriever:
    """Build the Uber historical-reply retriever."""
    print("Loading embedding model...")

    embedder = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    print("Building historical reply corpus...")

    retriever, historical_replies = (
        build_uber_reply_retriever(
            embedder,
            raw_data,
        )
    )

    print(
        f"Historical replies: "
        f"{len(historical_replies)}"
    )

    return retriever


def real_prediction(
    row: dict[str, str],
    retriever: ReplyRetriever,
    llm_client: GroqLLMClient,
    *,
    top_k: int,
    use_judge: bool,
) -> dict[str, Any]:
    """Run one production evaluation example."""
    pipeline_result = run_pipeline(
        row["text"],
        retriever=retriever,
        llm_client=llm_client,
        top_k=top_k,
    )

    escalation = decide_escalation(
        pipeline_result.classification,
    )

    retrieved = retriever.search(
        row["text"],
        top_k=top_k,
    )

    similarities = [
        float(item.similarity)
        for item in retrieved
        if getattr(
            item,
            "similarity",
            None,
        )
        is not None
    ]

    judge = None

    if use_judge:
        judge_attempts = 0
        max_judge_attempts = 2

        while judge_attempts < max_judge_attempts:
            judge_attempts += 1

            try:
                judge_result = judge_reply(
                    customer_text=row["text"],
                    draft_reply=pipeline_result.draft.reply,
                    evidence=pipeline_result.draft.evidence,
                    llm_client=llm_client,
                )

                judge = {
                    "status": "ok",
                    "attempts": judge_attempts,
                    "groundedness": (
                        judge_result.groundedness
                    ),
                    "policy_adherence": (
                        judge_result.policy_adherence
                    ),
                    "rationale": (
                        judge_result.rationale
                    ),
                }

                break

            except ValueError as exc:
                # The judge request succeeded but returned
                # malformed JSON/schema. Retry the judge.
                if judge_attempts >= max_judge_attempts:
                    judge = {
                        "status": "error",
                        "attempts": judge_attempts,
                        "groundedness": None,
                        "policy_adherence": None,
                        "rationale": None,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }

            except Exception as exc:
                # A judge failure must not discard a valid
                # agent result.
                judge = {
                    "status": "error",
                    "attempts": judge_attempts,
                    "groundedness": None,
                    "policy_adherence": None,
                    "rationale": None,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }

                break

    return {
        "predicted_intent": (
            pipeline_result.classification.intent
        ),
        "confidence": (
            pipeline_result.classification.confidence
        ),
        "reason": (
            pipeline_result.classification.reason
        ),
        "predicted_escalate": (
            escalation.escalate
        ),
        "escalation_reason": (
            escalation.reason
        ),
        "draft_reply": (
            pipeline_result.draft.reply
        ),
        "evidence": (
            pipeline_result.draft.evidence
        ),
        "retrieval_similarities": similarities,
        "judge": judge,
        "top_k": top_k,
    }


def process_rows(
    rows: list[dict[str, str]],
    results: list[dict[str, Any]],
    *,
    mode: str,
    retriever: ReplyRetriever | None,
    llm_client: GroqLLMClient | None,
    top_k: int,
    use_judge: bool,
    checkpoint_path: Path,
    retry_failed: bool = False,
) -> list[dict[str, Any]]:
    """Process rows with durable checkpointing.

    Successful rows are skipped during normal resume.

    Failed rows are also skipped during normal resume.

    When retry_failed=True, previously failed rows are
    eligible for processing.

    Ordinary row failures are recorded and processing
    continues.

    LLMCallBudgetExceeded remains fatal for the invocation
    because the configured call budget is a safety boundary.
    """
    completed_ids = {
        str(result["tweet_id"])
        for result in results
        if (
            "tweet_id" in result
            and result.get(
                "status",
                "completed",
            )
            == "completed"
        )
    }

    total = len(rows)

    for position, row in enumerate(
        rows,
        start=1,
    ):
        tweet_id = str(
            row["tweet_id"]
        )

        # Normal resume skips completed rows.
        #
        # When retry_failed=True, failed rows are intentionally
        # allowed through.
        if tweet_id in completed_ids:
            continue

        print(
            f"[{position}/{total}] "
            f"tweet_id={tweet_id}",
            flush=True,
        )

        try:
            if mode == "mock":
                prediction = mock_prediction(
                    row,
                    top_k,
                )

            else:
                if (
                    retriever is None
                    or llm_client is None
                ):
                    raise RuntimeError(
                        "Real mode requires retriever "
                        "and LLM client"
                    )

                prediction = real_prediction(
                    row,
                    retriever,
                    llm_client,
                    top_k=top_k,
                    use_judge=use_judge,
                )

            # If this tweet had a previous failure record,
            # remove it before inserting the successful result.
            results[:] = [
                existing
                for existing in results
                if str(
                    existing.get("tweet_id")
                )
                != tweet_id
            ]

            result = {
                "tweet_id": tweet_id,
                "text": row["text"],
                "human_intent_label": row[
                    "human_intent_label"
                ],
                "human_auto_handle_label": row[
                    "human_auto_handle_label"
                ],
                "status": "completed",
                **prediction,
            }

            results.append(result)
            completed_ids.add(tweet_id)

            print(
                f"Completed tweet_id={tweet_id}",
                flush=True,
            )

        except LLMCallBudgetExceeded as exc:
            # Global invocation safety boundary.
            print(
                f"LLM budget exhausted: {exc}",
                flush=True,
            )

            print(
                "Saving checkpoint and "
                "stopping cleanly.",
                flush=True,
            )

            save_checkpoint(
                checkpoint_path,
                results,
                mode=mode,
                completed=len(results),
                total=total,
            )

            break

        except Exception as exc:
            # Ordinary row failure.
            #
            # Remove any previous failure record first so that
            # retrying a row replaces the old failure instead
            # of creating duplicate records.
            results[:] = [
                existing
                for existing in results
                if str(
                    existing.get("tweet_id")
                )
                != tweet_id
            ]

            print(
                f"Row failed: tweet_id={tweet_id}; "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )

            print(
                "Recording failure and "
                "continuing to the next row.",
                flush=True,
            )

            results.append(
                {
                    "tweet_id": tweet_id,
                    "text": row["text"],
                    "human_intent_label": row[
                        "human_intent_label"
                    ],
                    "human_auto_handle_label": row[
                        "human_auto_handle_label"
                    ],
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

        # Checkpoint after every processed row,
        # whether successful or failed.
        save_checkpoint(
            checkpoint_path,
            results,
            mode=mode,
            completed=len(results),
            total=total,
        )

    return results


def print_summary(
    results: list[dict[str, Any]],
    mode: str,
) -> None:
    """Print an honest evaluation summary."""
    print()
    print("=== Evaluation Run Summary ===")
    print(f"Mode: {mode}")

    successful = sum(
        1
        for result in results
        if result.get(
            "status",
            "completed",
        )
        == "completed"
    )

    failed = sum(
        1
        for result in results
        if result.get("status") == "error"
    )

    processed = successful + failed

    print(
        f"Processed rows: {processed}"
    )

    print(
        f"Successful rows: {successful}"
    )

    print(
        f"Failed rows: {failed}"
    )

    if failed:
        print("Failed rows:")

        for result in results:
            if result.get("status") != "error":
                continue

            print(
                f"  tweet_id={result.get('tweet_id')}; "
                f"{result.get('error_type')}: "
                f"{result.get('error')}"
            )

    if successful == 0:
        return

    intents: dict[str, int] = {}

    for result in results:
        if (
            result.get(
                "status",
                "completed",
            )
            != "completed"
        ):
            continue

        intent = str(
            result.get(
                "predicted_intent",
                "",
            )
        )

        intents[intent] = (
            intents.get(intent, 0) + 1
        )

    print(
        "Predicted intent distribution:"
    )

    for intent, count in sorted(
        intents.items()
    ):
        print(
            f"  {intent}: {count}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the support-agent "
            "evaluation harness safely."
        )
    )

    parser.add_argument(
        "--golden-set",
        type=Path,
        default=Path(
            "data/eval/"
            "golden_set_candidates.csv"
        ),
    )

    parser.add_argument(
        "--raw-data",
        type=Path,
        default=Path(
            "data/raw/twcs/twcs.csv"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/eval/"
            "real_evaluation_results.json"
        ),
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "data/eval/"
            "real_evaluation_results_"
            "checkpoint.json"
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Process the first N rows from "
            "the golden set; --resume skips "
            "already processed rows."
        ),
    )

    parser.add_argument(
        "--mock",
        action="store_true",
        help=(
            "Run without LLM calls. "
            "Results are control-flow only."
        ),
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume from the specified checkpoint."
        ),
    )

    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help=(
            "Retry rows previously recorded "
            "with status=error in the checkpoint."
        ),
    )

    parser.add_argument(
        "--judge",
        action="store_true",
        help=(
            "Run the LLM judge after "
            "generating each draft."
        ),
    )

    parser.add_argument(
        "--max-calls",
        type=int,
        default=100,
        help=(
            "Maximum LLM calls allowed "
            "in this evaluation invocation."
        ),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if (
        args.limit is not None
        and args.limit <= 0
    ):
        raise SystemExit(
            "--limit must be greater than zero"
        )

    if args.max_calls <= 0:
        raise SystemExit(
            "--max-calls must be greater than zero"
        )

    if args.top_k <= 0:
        raise SystemExit(
            "--top-k must be greater than zero"
        )

    if args.retry_failed and not args.resume:
        raise SystemExit(
            "--retry-failed requires --resume"
        )

    rows = load_golden_set(
        args.golden_set
    )

    if args.limit is not None:
        rows = rows[:args.limit]

    mode = (
        "mock"
        if args.mock
        else "real"
    )

    existing_results = (
        load_checkpoint(
            args.checkpoint
        )
        if args.resume
        else []
    )

    completed_ids = {
        str(result["tweet_id"])
        for result in existing_results
        if (
            "tweet_id" in result
            and result.get(
                "status",
                "completed",
            )
            == "completed"
        )
    }

    failed_ids = {
        str(result["tweet_id"])
        for result in existing_results
        if (
            "tweet_id" in result
            and result.get("status")
            == "error"
        )
    }

    if args.retry_failed:
        rows_to_process = [
            row
            for row in rows
            if str(
                row["tweet_id"]
            )
            in failed_ids
        ]
    else:
        rows_to_process = [
            row
            for row in rows
            if str(
                row["tweet_id"]
            ) not in completed_ids
            and str(
                row["tweet_id"]
            ) not in failed_ids
        ]

    print(
        f"Golden rows selected: {len(rows)}"
    )

    print(
        f"Already completed: "
        f"{len(completed_ids)}"
    )

    print(
        f"Previously failed: "
        f"{len(failed_ids)}"
    )

    print(
        f"Remaining this invocation: "
        f"{len(rows_to_process)}"
    )

    print(
        f"Mode: {mode}"
    )

    if args.retry_failed:
        print(
            "Retry mode: previously failed "
            "rows will be retried."
        )

    if not rows_to_process:
        print(
            "Nothing new to process."
        )

        save_results(
            args.output,
            existing_results,
            mode=mode,
        )

        print_summary(
            existing_results,
            mode,
        )

        return 0

    retriever = None
    llm_client = None

    if not args.mock:
        # Each invocation has an explicit LLM call budget.
        # Default: 100 calls.
        #
        # Checkpointing protects completed work if the
        # budget is exhausted or the process is interrupted.
        retriever = build_retriever(
            args.raw_data
        )

        llm_client = GroqLLMClient(
            config=LLMConfig(
                model="openai/gpt-oss-120b",
                max_calls=args.max_calls,
                max_output_tokens=500,
            )
        )

        calls_per_row = (
            3
            if args.judge
            else 2
        )

        expected_calls = (
            len(rows_to_process)
            * calls_per_row
        )

        if expected_calls > llm_client.config.max_calls:
            print(
                f"Requested batch contains "
                f"{len(rows_to_process)} rows "
                f"and normally needs approximately "
                f"{expected_calls} calls."
            )

            print(
                f"Invocation budget: "
                f"{llm_client.config.max_calls}."
            )

            print(
                "The runner will process rows "
                "until the call budget is exhausted."
            )

    results = process_rows(
        rows_to_process,
        existing_results,
        mode=mode,
        retriever=retriever,
        llm_client=llm_client,
        top_k=args.top_k,
        use_judge=args.judge,
        checkpoint_path=args.checkpoint,
        retry_failed=args.retry_failed,
    )

    save_results(
        args.output,
        results,
        mode=mode,
    )

    print_summary(
        results,
        mode,
    )

    if (
        not args.mock
        and llm_client is not None
    ):
        print(
            f"LLM calls used: "
            f"{llm_client.calls_used}"
        )

        print(
            f"Output tokens used: "
            f"{llm_client.output_tokens_used}"
        )

        print(
            f"LLM calls remaining: "
            f"{llm_client.remaining_calls}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())