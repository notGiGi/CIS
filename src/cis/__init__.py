"""Counterfactual Internal State (CIS) optimization module.

This module implements gradient-based optimization to find minimal perturbations
to internal activations that flip factual predictions to counterfactual targets.
"""

from src.cis.delta import LearnableDelta
from src.cis.losses import combined_loss, counterfactual_loss, margin_loss, regularization_loss
from src.cis.optimizer import CISOptimizer

__all__ = [
    "LearnableDelta",
    "counterfactual_loss",
    "margin_loss",
    "combined_loss",
    "regularization_loss",
    "CISOptimizer",
]
