# Phase 3 Decision Record — D-Phase3-02

## Choice
**Option A: adopt a narrow deterministic heuristic from raw TWCS structure.**

The deadline makes a small hand-reviewed retrieval corpus insufficient for a useful retrieval baseline, so the project now derives retrieval eligibility from the raw Kaggle/TWCS rows.

## Author derivation
A row is treated as an Uber agent reply only when:

- `inbound == False`
- `author_id == "Uber_Support"`

The second condition narrows the generic TWCS outbound/company indicator to the target brand.

## Resolution derivation
A root thread is considered heuristically `resolved` only when:

1. the root is an inbound customer tweet that mentions `@Uber_Support`;
2. the thread can be reconstructed through `tweet_id` and `in_response_to_tweet_id` parent-child edges; and
3. every terminal message in that reconstructed thread is an Uber outbound message.

If a customer message is terminal, parent context is missing, or the structure is otherwise incomplete/ambiguous, the thread contributes no replies to retrieval.

This is a structural corpus-eligibility heuristic. It does **not** claim that the customer was satisfied, that the issue was actually solved, or that the agent response was correct.

## Relationship to D-Phase3-01
This explicitly **reverses the no-heuristic stance in D-Phase3-01**. D-Phase3-01 remains the validation boundary for records that already carry explicit provenance; D-Phase3-02 adds the raw-data derivation layer that creates those provenance fields.

## Headline-number caveat
Any retrieval or evaluation headline number produced from this corpus must state that the historical-reply corpus is **heuristically resolved, not human-verified**. Such metrics must not be described as performance against a human-labeled resolution corpus.

## Implementation
- `src/uber_history_derivation.py` derives the provenance from raw TWCS rows.
- `src/corpus.py` exposes `build_uber_reply_retriever(...)` so raw CSV rows pass through the derivation before FAISS indexing.
- `tests/test_uber_history_derivation.py` covers the graph rule and contains an integration test against the existing real `data/processed/uber_support_sample.csv` when that local dataset is present.
- `tests/test_corpus.py` contains a corresponding real-sample retrieval-corpus integration test when that local dataset is present.
