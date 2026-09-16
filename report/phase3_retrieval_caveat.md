# Phase 3 Retrieval Corpus Caveat

The retrieval corpus is now constructed from raw Customer Support on Twitter rows using D-Phase3-02.

`author_type == "agent"` is derived from `inbound == False` plus `author_id == "Uber_Support"`.

`thread_status == "resolved"` is a structural heuristic: the reconstructed thread must have an inbound Uber root and every terminal message must be an Uber outbound message. Customer-terminal, incomplete, and ambiguous threads are excluded.

This is not human resolution labeling. It does not establish customer satisfaction, actual issue resolution, response correctness, or policy correctness.

**Headline-number caveat:** any retrieval/evaluation metric based on this corpus must explicitly state that the historical replies were heuristically resolved rather than human-verified. These metrics should not be described as performance against a human-labeled resolution corpus.
