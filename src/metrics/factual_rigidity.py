"""Metrics for assessing stability of factual predictions under interventions."""

from typing import Any, Dict


def compute_factual_rigidity(baseline_logits: Any, counterfactual_logits: Any, factual_id: int, counterfactual_id: int) -> Dict[str, float]:
    """Quantify how resistant the model is to counterfactual steering.

    Suggested outputs include logit margin shifts, probability mass redistribution,
    and normalized geometric cost where available.
    """
    raise NotImplementedError


def summarize_batch_metrics(batch_results: Any) -> Dict[str, float]:
    """Aggregate per-example metrics across a batch of interventions."""
    raise NotImplementedError
