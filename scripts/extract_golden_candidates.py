from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from classify import INTENTS


OUTPUT_COLUMNS = [
    "tweet_id",
    "text",
    "predicted_intent",
    "human_intent_label",
    "human_auto_handle_label",
    "notes",
]


# These patterns are ONLY used to create a review candidate pool.
# They are not ground-truth labels.
INTENT_PATTERNS = {
    "cancellation_no_show_charge": [
        r"\bcancel(?:led|ling)?\b.*\bcharg",
        r"\bcharg.*\bcancel",
        r"\bcancellation fee\b",
        r"\bcancel fee\b",
        r"\bno[- ]?show\b",
        r"\bdriver\b.*\bcancel",
    ],
    "unauthorized_transaction": [
        r"\bunauthori[sz]",
        r"\bfraud\b",
        r"\bnot my (?:charge|transaction|payment)\b",
        r"\bdidn'?t make (?:this )?(?:charge|transaction|payment)\b",
        r"\bsomeone (?:used|charged)\b.*\b(?:card|account)\b",
        r"\bstolen\b.*\b(?:card|account|money)\b",
    ],
    "fare_price_dispute": [
        r"\bovercharg",
        r"\btoo (?:much|expensive)\b",
        r"\b(?:fare|price|pricing)\b.*\b(?:wrong|high|expensive|incorrect)\b",
        r"\b(?:wrong|high|incorrect)\b.*\b(?:fare|price|pricing)\b",
        r"\bsurge\b",
        r"\bcharged\b.*\bmore\b",
        r"\bprice\b.*\btrip\b",
    ],
    "missing_incomplete_ride": [
        r"\bdriver\b.*\bnever (?:arrived|came|showed)\b",
        r"\bdriver\b.*\bwon'?t (?:come|arrive|show)\b",
        r"\bdriver\b.*\bnot (?:here|arriv|come)\b",
        r"\b(?:ride|trip)\b.*\b(?:missing|disappear|gone)\b",
        r"\b(?:ride|trip)\b.*\bincomplete\b",
        r"\bwaiting\b.*\bdriver\b",
        r"\bdriver\b.*\bwaiting\b",
        r"\bdriver\b.*\bminutes away\b",
    ],
    "uber_eats_order_delivery_issue": [
        r"\buber eats\b",
        r"\bubereats\b",
        r"\bfood\b.*\b(?:order|deliver)\b",
        r"\b(?:order|delivery)\b.*\b(?:food|restaurant)\b",
        r"\b(?:order|food)\b.*\b(?:late|missing|never arrived|not delivered)\b",
        r"\bdelivery\b.*\b(?:late|missing|stolen|never)\b",
    ],
    "ride_pass_promotion_coupon": [
        r"\bpromo(?:tion)?\b",
        r"\bcoupon\b",
        r"\bdiscount\b",
        r"\bvoucher\b",
        r"\bpromo code\b",
        r"\bpass\b.*\buber\b",
        r"\buber\b.*\bpass\b",
    ],
    "account_login_app_access": [
        r"\blog ?in\b",
        r"\bsign ?in\b",
        r"\bpassword\b",
        r"\baccount\b.*\b(?:blocked|banned|locked)\b",
        r"\b(?:blocked|banned|locked)\b.*\baccount\b",
        r"\bapp\b.*\b(?:not work|doesn'?t work|won'?t work|broken)\b",
        r"\bapp\b.*\b(?:access|login)\b",
        r"\bcustomer service\b.*\bcontact\b",
        r"\bsupport phone\b",
    ],
    "payment_refund_billing": [
        r"\brefund(?:ed)?\b",
        r"\brefund\b.*\b(?:money|payment|charge)\b",
        r"\bcharg(?:e|ed)\b.*\b(?:twice|double)\b",
        r"\bcharged twice\b",
        r"\bduplicate\b.*\bcharge\b",
        r"\bbilling\b",
        r"\bpayment\b.*\b(?:failed|wrong|issue|problem)\b",
    ],
    "driver_partner_safety_issue": [
        r"\bunsafe\b",
        r"\bdangerous\b.*\bdriver\b",
        r"\bdriver\b.*\b(?:harass|threat|assault|hit)\b",
        r"\b(?:harass|threat|assault)\b.*\bdriver\b",
        r"\b(?:police|police complaint)\b.*\bdriver\b",
        r"\bdriver\b.*\bpolice\b",
        r"\bhit by an uber driver\b",
        r"\bsafety\b.*\bdriver\b",
    ],
}


COMPILED_PATTERNS = {
    intent: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    for intent, patterns in INTENT_PATTERNS.items()
}


def load_candidates(input_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path)

    required = {"tweet_id", "inbound", "text"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    candidates = df[
        (df["inbound"] == True)  # noqa: E712
        & df["text"].fillna("").astype(str).str.strip().ne("")
    ].copy()

    candidates["text"] = candidates["text"].astype(str).str.strip()
    candidates = candidates.drop_duplicates(subset=["tweet_id"])

    return candidates


def predict_intent(text: str) -> str:
    """
    Assign a provisional intent for candidate sampling only.

    This is intentionally deterministic and must not be treated as
    human ground truth.
    """
    matches: list[str] = []

    for intent in INTENTS:
        if intent == "other_out_of_scope":
            continue

        patterns = COMPILED_PATTERNS.get(intent, [])

        if any(pattern.search(text) for pattern in patterns):
            matches.append(intent)

    if len(matches) == 1:
        return matches[0]

    # Ambiguous or unmatched examples are retained for human review.
    return "other_out_of_scope"


def build_balanced_sample(
    candidates: pd.DataFrame,
    target_size: int,
    random_state: int,
) -> pd.DataFrame:
    candidates = candidates.copy()
    candidates["predicted_intent"] = candidates["text"].map(predict_intent)

    selected_parts: list[pd.DataFrame] = []

    # Aim for an even candidate pool. If an intent has fewer candidates,
    # take everything available and fill the remaining slots later.
    per_intent = target_size // len(INTENTS)

    for intent in INTENTS:
        group = candidates[candidates["predicted_intent"] == intent]

        if group.empty:
            continue

        selected_parts.append(
            group.sample(
                n=min(per_intent, len(group)),
                random_state=random_state,
            )
        )

    if selected_parts:
        result = pd.concat(selected_parts, ignore_index=True)
    else:
        result = pd.DataFrame(columns=candidates.columns)

    result = result.drop_duplicates(subset=["tweet_id"])

    # Fill remaining positions from the unused pool, prioritising examples
    # with a provisional intent before falling back to ambiguous examples.
    remaining = candidates[
        ~candidates["tweet_id"].isin(result["tweet_id"])
    ].copy()

    if len(result) < target_size and not remaining.empty:
        extra_needed = target_size - len(result)

        # Prefer non-out-of-scope provisional matches.
        informative = remaining[
            remaining["predicted_intent"] != "other_out_of_scope"
        ]

        extra_parts = []

        if not informative.empty:
            n = min(extra_needed, len(informative))
            extra_parts.append(
                informative.sample(
                    n=n,
                    random_state=random_state,
                )
            )
            extra_needed -= n

        if extra_needed > 0:
            remaining = remaining[
                ~remaining["tweet_id"].isin(
                    pd.concat(extra_parts)["tweet_id"]
                    if extra_parts
                    else []
                )
            ]

            if not remaining.empty:
                extra_parts.append(
                    remaining.sample(
                        n=min(extra_needed, len(remaining)),
                        random_state=random_state,
                    )
                )

        if extra_parts:
            result = pd.concat(
                [result, *extra_parts],
                ignore_index=True,
            )

    result = result.drop_duplicates(subset=["tweet_id"])

    if len(result) > target_size:
        result = result.sample(
            n=target_size,
            random_state=random_state,
        )

    return result


def extract(
    input_path: Path,
    output_path: Path,
    random_state: int = 42,
) -> pd.DataFrame:
    candidates = load_candidates(input_path)

    result = build_balanced_sample(
        candidates,
        target_size=200,
        random_state=random_state,
    )

    result["human_intent_label"] = ""
    result["human_auto_handle_label"] = ""
    result["notes"] = ""

    result = result[OUTPUT_COLUMNS]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    return result


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/uber_support_sample.csv"),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/eval/golden_set_candidates.csv"),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    source = load_candidates(args.input)

    result = extract(
        args.input,
        args.output,
        random_state=args.seed,
    )

    print(f"Input candidates: {len(source)}")
    print(f"Golden candidates written: {len(result)}")
    print(f"Output: {args.output}")

    print("\nProvisional intent distribution:")
    print(result["predicted_intent"].value_counts().sort_index())


if __name__ == "__main__":
    main()