# Evaluation Layer

This directory contains reproducible evaluation code only. It does not contain fabricated labels or benchmark results.

## Human-owned golden set

The assignment requires 150–250 human-labelled examples. Each golden row should contain, at minimum:

- stable example ID
- initiating customer tweet text
- human intent label from the documented Uber taxonomy
- human escalation decision
- reference-good-reply criteria or reference response
- optional second annotator labels for agreement measurement

Human labels are authoritative; generated suggestions are not used as ground truth.

## Baselines

`baselines.py` provides two intent baselines:

1. majority-class baseline
2. TF-IDF + logistic regression baseline

Both require labelled training data supplied by the evaluator.

## Metrics

`metrics.py` provides:

- accuracy
- macro F1
- exact-match retrieval recall@k
- Cohen's kappa for human annotator agreement
- Pearson correlation for numeric judge/human scores

No benchmark number should be reported until the human-owned golden set has been created and executed locally.
