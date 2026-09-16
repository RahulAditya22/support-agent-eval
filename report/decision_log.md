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
