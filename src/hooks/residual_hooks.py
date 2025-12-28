"""Utilities for attaching and managing residual stream hooks."""

from typing import Any, Callable, Dict, Optional


def register_residual_capture(model: Any, layer: int, token_position: int) -> Dict[str, Any]:
    """Register forward hooks to capture residual activations at a target layer and token.

    The returned dictionary should include handles and captured tensors to support
    subsequent optimization of counterfactual interventions.
    """
    raise NotImplementedError


def apply_residual_injection(model: Any, layer: int, token_position: int, delta: Any) -> Callable:
    """Attach a hook that adds a perturbation `delta` to the residual stream.

    The callable returned by this function should be removable to restore the
    unperturbed model state after evaluation.
    """
    raise NotImplementedError


def clear_hooks(handles: Optional[Dict[str, Any]]) -> None:
    """Remove all provided hooks to avoid persistent graph modifications."""
    raise NotImplementedError
