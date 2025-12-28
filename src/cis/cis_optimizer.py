"""Optimization routines for Counterfactual Internal States (CIS)."""

from typing import Any, Dict, Optional


class CISOptimizer:
    """Search for minimal-norm activation interventions that flip factual predictions."""

    def __init__(self, model: Any, tokenizer: Any, config: Dict[str, Any], logger: Optional[Any] = None) -> None:
        """Initialize optimizer with model, tokenizer, and experiment configuration."""
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.logger = logger

    def compute_geometric_cost(self, delta: Any) -> float:
        """Return the L2 norm of the activation perturbation representing geometric cost."""
        raise NotImplementedError

    def objective(self, logits: Any, target_id: int, baseline_id: int) -> float:
        """Compute an objective that favors the counterfactual token over the factual token."""
        raise NotImplementedError

    def search(self, prompt: str, target_completion: str) -> Dict[str, Any]:
        """Run the CIS optimization loop for a single prompt-target pair."""
        raise NotImplementedError

    def apply_intervention(self, prompt: str, delta: Any) -> Dict[str, Any]:
        """Evaluate the model with a fixed perturbation and return logits and text output."""
        raise NotImplementedError
