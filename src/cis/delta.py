"""Learnable perturbation parameter for CIS optimization.

This module defines a learnable tensor δ that is optimized to minimally
perturb internal activations to achieve counterfactual predictions.
"""

from typing import Optional

import torch
import torch.nn as nn


class LearnableDelta(nn.Module):
    """Learnable perturbation vector for residual stream intervention.

    This is the core parameter we optimize in CIS experiments.
    It represents an additive perturbation to hidden states at a specific
    layer and token position.

    Attributes:
        delta: The learnable perturbation tensor [hidden_dim]
        hidden_dim: Dimensionality of the residual stream
    """

    def __init__(
        self,
        hidden_dim: int,
        init_method: str = "zeros",
        init_scale: float = 0.01,
        device: Optional[str] = None,
    ):
        """Initialize learnable delta parameter.

        Args:
            hidden_dim: Size of the hidden state (e.g., 4096 for Mistral-7B)
            init_method: Initialization method ('zeros', 'randn', 'uniform')
            init_scale: Scale for random initialization (if init_method != 'zeros')
            device: Device to create tensor on (None = default)

        Scientific intent:
            We typically start with delta=0 (no intervention) and optimize from there.
            This lets us measure the minimal perturbation needed.
        """
        super().__init__()

        self.hidden_dim = hidden_dim

        # Initialize delta based on method
        if init_method == "zeros":
            delta_init = torch.zeros(hidden_dim, device=device)
        elif init_method == "randn":
            delta_init = torch.randn(hidden_dim, device=device) * init_scale
        elif init_method == "uniform":
            delta_init = (torch.rand(hidden_dim, device=device) - 0.5) * 2 * init_scale
        else:
            raise ValueError(f"Unknown init_method: {init_method}")

        # Register as parameter (requires_grad=True by default)
        self.delta = nn.Parameter(delta_init)

    def forward(self) -> torch.Tensor:
        """Return the current delta value.

        Returns:
            delta: The learnable perturbation [hidden_dim]
        """
        return self.delta

    def get_norm(self, p: int = 2) -> float:
        """Compute Lp norm of delta.

        Args:
            p: Norm type (1, 2, or float('inf'))

        Returns:
            norm: ||δ||_p

        Scientific intent:
            The L2 norm is our primary measure of "geometric cost".
            Smaller norm = less perturbation needed = weaker factual rigidity.
        """
        return self.delta.norm(p=p).item()

    def reset(self) -> None:
        """Reset delta to zeros.

        Useful for running multiple optimization experiments.
        """
        with torch.no_grad():
            self.delta.zero_()

    def project_to_norm(self, max_norm: float) -> None:
        """Project delta to have at most max_norm.

        Args:
            max_norm: Maximum allowed L2 norm

        This can be used as a constraint during optimization to prevent
        delta from growing unboundedly.
        """
        with torch.no_grad():
            current_norm = self.delta.norm(p=2)
            if current_norm > max_norm:
                self.delta.mul_(max_norm / current_norm)

    def get_stats(self) -> dict:
        """Get statistics about current delta.

        Returns:
            stats: Dictionary with norm, mean, std, min, max

        Useful for monitoring during optimization.
        """
        delta_np = self.delta.detach().cpu()
        return {
            "norm_l2": self.get_norm(p=2),
            "norm_l1": self.get_norm(p=1),
            "norm_linf": self.get_norm(p=float("inf")),
            "mean": delta_np.mean().item(),
            "std": delta_np.std().item(),
            "min": delta_np.min().item(),
            "max": delta_np.max().item(),
        }

    def __repr__(self) -> str:
        return f"LearnableDelta(hidden_dim={self.hidden_dim}, norm={self.get_norm():.6f})"
