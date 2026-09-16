# Decision Log

## Phase 2 — Intent Discovery

### D-Phase2-09 — Hierarchical stability across k=8→9→10
- The three largest clusters were checked for membership retention across adjacent k values.
- k=8 cluster 4 (n=752) mapped to k=9 cluster 1 with 752/752 retained (Jaccard 1.000), then to k=10 cluster 1 with 752/752 retained.
- k=8 cluster 5 (n=385) mapped to k=9 cluster 5 with 385/385 retained (Jaccard 1.000), then to k=10 cluster 2 with 385/385 retained.
- k=8 cluster 6 (n=134) mapped to k=9 cluster 6 with 134/134 retained (Jaccard 1.000), then to k=10 cluster 6 with 134/134 retained.
- Conclusion: the largest clusters were hierarchically stable rather than scrambling as k increased. Stability is treated as evidence of reproducible partitioning, not evidence that a cluster is a valid semantic intent.

### D-Phase2-10 — Large-cluster heterogeneity
- Random inspection of the n=752 cluster showed materially heterogeneous support themes, including cancellation/no-show charges, account/access issues, missing rides, unauthorized transactions, fare questions, driver/partner issues, and app/trip problems.
- Because the cluster remains heterogeneous despite its hierarchical stability, it will not be further sub-clustered for this assignment.
- The taxonomy will instead name the most frequent defensible sub-themes already observed in this cluster (cancellation/no-show charges, unauthorized transactions, fare disputes, and missing rides). Messages that do not fit a named intent will be routed to `other_out_of_scope` rather than forcing full separation of the heterogeneous cluster.
- This decision prioritizes explainability and defensible labels over producing a superficially cleaner clustering score.

## Phase 3 — Retrieval Provenance

### D-Phase3-01 — Do not infer resolved/completed status from raw tweet structure
- The Kaggle Twitter support export contains structural fields such as `inbound`, `response_tweet_id`, and `in_response_to_tweet_id`, but the current retrieval corpus boundary does not treat those fields alone as proof that a thread is resolved or completed.
- `src/uber_history_adapter.py` therefore requires explicit `author_type` and `thread_status` metadata on reconstructed records. It does not derive either field, call an LLM, or use a fixed heuristic to label a thread as resolved/completed.
- Records are admitted only when `author_type == "agent"` and `thread_status` is `resolved` or `completed`; customer-authored records and open/other-status records are excluded. Records missing the required provenance fields raise `ValueError` rather than being guessed or silently accepted.
- Operationally, this means the repository currently has a provenance validation boundary, not a raw-Kaggle resolution-status derivation algorithm. A future dataset-specific reconstruction must establish these fields from documented evidence before the resulting replies are used for retrieval.
- Threads that do not cleanly establish both authorship and resolution/completion remain outside the retrieval corpus rather than being classified by inference.

### D-Phase3-02 — Adopt narrow raw-TWCS heuristic for retrieval corpus construction
- **Choice: Option A.** Given the deadline, a deterministic raw-data heuristic is used instead of waiting for a small hand-reviewed retrieval corpus.
- `author_type == "agent"` is derived only when `inbound == False` and `author_id == "Uber_Support"`.
- A thread is heuristically `resolved` only when it is reconstructable from `tweet_id` / `in_response_to_tweet_id` and has an inbound Uber root where every terminal message is an Uber outbound message.
- Forward `response_tweet_id` links are also checked: if a row references a child tweet that is absent from the local reconstruction, that root thread is treated as incomplete and excluded.
- This is a structural eligibility heuristic, not evidence that the customer was satisfied or that the response actually solved the issue.
- **This explicitly reverses D-Phase3-01's no-heuristic stance.** D-Phase3-01 remains the validation boundary for explicitly annotated records; D-Phase3-02 adds the raw-data derivation layer that produces those annotations.
- **Headline-number caveat:** retrieval/evaluation metrics using this corpus must state that the corpus is heuristically resolved, not human-verified. Metrics therefore should not be presented as performance against a human-labeled resolution corpus.
- Threads that do not satisfy the heuristic are excluded rather than forced into `resolved`.
