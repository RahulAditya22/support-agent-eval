"""Compatibility import for evaluation metrics.

The canonical implementation lives in ``eval.metrics``.  This module keeps
existing test and script imports working when ``src`` is on PYTHONPATH.
"""

from eval.metrics import (  # noqa: F401
    ClassificationMetrics,
    classification_metrics,
    cohen_kappa,
    pearson_correlation,
    top_k_recall,
)
