"""Metrics for measuring and normalizing CIS geometric costs.

This module provides functions to compute normalized measures of factual rigidity
that are comparable across layers, models, and experiments.
"""

import math
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.hooks.residual_hooks import (  # noqa: E402
    get_hidden_size,
    register_residual_capture,
)


def measure_residual_norm(
    model: Any,
    tokenizer: Any,
    prompt: str,
    layer_idx: int,
    token_position: int = -1,
    device: str = "cuda",
) -> Dict[str, float]:
    """Measure the baseline residual stream norm at a specific layer and token.

    This provides a normalization baseline for delta perturbations, allowing
    us to compute normalized geometric cost as delta_norm / residual_norm.

    Args:
        model: The transformer model
        tokenizer: Tokenizer
        prompt: Input text
        layer_idx: Which layer to measure
        token_position: Which token position (-1 = last)
        device: Device for computation

    Returns:
        Dictionary with:
            - residual_norm: ||h||_2 of the residual stream vector
            - residual_mean: Mean activation value
            - residual_std: Standard deviation of activations
            - hidden_dim: Dimensionality of residual stream

    Scientific intent:
        The residual norm provides a natural scale for the perturbation.
        A delta_norm that is 10% of residual_norm means we're perturbing
        by 10% of the typical activation magnitude at that position.
    """
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Capture residual activations
    capture = register_residual_capture(
        model=model,
        layer_idx=layer_idx,
        token_position=token_position,
    )

    try:
        # Run forward pass
        with torch.no_grad():
            _ = model(**inputs)

        # Get captured activation
        if len(capture["activations"]) == 0:
            raise ValueError("No activations captured. Check layer_idx and token_position.")

        residual_vector = capture["activations"][0]  # [batch, hidden_dim] -> [1, hidden_dim]
        residual_vector = residual_vector.squeeze(0)  # [hidden_dim]

        # Compute statistics
        residual_norm = residual_vector.norm(p=2).item()
        residual_mean = residual_vector.mean().item()
        residual_std = residual_vector.std().item()
        hidden_dim = residual_vector.shape[0]

    finally:
        # Clean up hook
        capture["handle"].remove()

    return {
        "residual_norm": residual_norm,
        "residual_mean": residual_mean,
        "residual_std": residual_std,
        "hidden_dim": hidden_dim,
    }


def compute_normalized_metrics(
    delta_norm: float,
    residual_norm: float,
    hidden_dim: int,
) -> Dict[str, float]:
    """Compute normalized geometric cost metrics.

    Args:
        delta_norm: L2 norm of perturbation ||δ||_2
        residual_norm: L2 norm of baseline residual ||h||_2
        hidden_dim: Dimensionality of vectors

    Returns:
        Dictionary with:
            - delta_norm: Raw L2 norm of delta
            - residual_norm: Baseline residual norm
            - normalized_cost_ratio: delta_norm / residual_norm (most important)
            - per_dim_rms: delta_norm / sqrt(hidden_dim)
            - relative_perturbation_pct: (delta_norm / residual_norm) * 100

    Scientific interpretation:
        - normalized_cost_ratio: Perturbation as fraction of typical activation
        - per_dim_rms: Average per-dimension perturbation magnitude
        - relative_perturbation_pct: Percentage change in activation magnitude

    Example:
        If delta_norm = 5.0 and residual_norm = 50.0:
        - normalized_cost_ratio = 0.1 (10% of baseline)
        - This is a relatively small perturbation

        If delta_norm = 25.0 and residual_norm = 50.0:
        - normalized_cost_ratio = 0.5 (50% of baseline)
        - This is a large perturbation, suggesting weak factual encoding
    """
    normalized_cost_ratio = delta_norm / residual_norm if residual_norm > 0 else float("inf")
    per_dim_rms = delta_norm / math.sqrt(hidden_dim) if hidden_dim > 0 else 0.0
    relative_perturbation_pct = normalized_cost_ratio * 100.0

    return {
        "delta_norm": delta_norm,
        "residual_norm": residual_norm,
        "normalized_cost_ratio": normalized_cost_ratio,
        "per_dim_rms": per_dim_rms,
        "relative_perturbation_pct": relative_perturbation_pct,
    }


def get_comprehensive_metrics(
    model: Any,
    tokenizer: Any,
    prompt: str,
    layer_idx: int,
    token_position: int,
    delta_norm: float,
    device: str = "cuda",
) -> Dict[str, float]:
    """Compute all metrics in one call: residual baseline + normalized costs.

    This is a convenience function that combines measure_residual_norm()
    and compute_normalized_metrics().

    Args:
        model: Transformer model
        tokenizer: Tokenizer
        prompt: Input text
        layer_idx: Layer where perturbation was applied
        token_position: Token position where perturbation was applied
        delta_norm: L2 norm of the optimized perturbation
        device: Device

    Returns:
        Dictionary with all metrics combined
    """
    # Measure baseline
    baseline_metrics = measure_residual_norm(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        layer_idx=layer_idx,
        token_position=token_position,
        device=device,
    )

    # Compute normalized metrics
    normalized_metrics = compute_normalized_metrics(
        delta_norm=delta_norm,
        residual_norm=baseline_metrics["residual_norm"],
        hidden_dim=baseline_metrics["hidden_dim"],
    )

    # Combine and return
    return {**baseline_metrics, **normalized_metrics}
