"""Loss functions for CIS optimization.

These losses are designed to be differentiable with respect to the perturbation δ,
allowing gradient-based optimization to find minimal interventions that flip
factual predictions to counterfactual targets.
"""

from typing import Optional

import torch
import torch.nn.functional as F


def counterfactual_loss(
    logits: torch.Tensor,
    target_token_id: int,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute negative log probability of target token.

    This loss encourages the model to assign high probability to the target token.
    Minimizing this loss = maximizing P(target).

    Args:
        logits: Model output logits [vocab_size] or [batch, vocab_size]
        target_token_id: ID of the counterfactual target token
        reduction: 'mean', 'sum', or 'none'

    Returns:
        loss: Scalar loss value (if reduction != 'none')

    Scientific intent:
        The gradient ∇δ L tells us which direction to adjust δ to increase
        the probability of the target token.
    """
    # Convert logits to probabilities
    log_probs = F.log_softmax(logits, dim=-1)

    # Extract log probability of target token
    # Handle both [vocab_size] and [batch, vocab_size]
    if logits.dim() == 1:
        target_log_prob = log_probs[target_token_id]
    else:
        target_log_prob = log_probs[:, target_token_id]

    # Negative log likelihood (minimizing = maximizing probability)
    nll = -target_log_prob

    # Apply reduction
    if reduction == "mean":
        return nll.mean()
    elif reduction == "sum":
        return nll.sum()
    else:
        return nll


def margin_loss(
    logits: torch.Tensor,
    target_token_id: int,
    original_token_id: int,
    margin: float = 0.0,
) -> torch.Tensor:
    """Compute margin-based loss between target and original tokens.

    This loss encourages the target token to have higher logit than the original
    factual token by at least the specified margin.

    Args:
        logits: Model output logits [vocab_size] or [batch, vocab_size]
        target_token_id: ID of counterfactual target
        original_token_id: ID of original factual token
        margin: Desired margin (default: 0.0 means just need target > original)

    Returns:
        loss: Scalar loss value

    Loss form:
        L = max(0, original_logit - target_logit + margin)

    When target_logit > original_logit + margin, loss = 0 (satisfied).
    Otherwise, loss > 0 (need to optimize more).
    """
    if logits.dim() == 1:
        target_logit = logits[target_token_id]
        original_logit = logits[original_token_id]
    else:
        target_logit = logits[:, target_token_id]
        original_logit = logits[:, original_token_id]

    # Hinge loss: encourage target > original + margin
    loss = torch.clamp(original_logit - target_logit + margin, min=0.0)

    return loss.mean()


def margin_flip_loss(
    logits: torch.Tensor,
    target_token_id: int,
    original_token_id: int,
    margin: float = 1.0,
) -> torch.Tensor:
    """Compute margin-based flip loss for CIS optimization.

    This is the recommended loss for finding minimal-norm perturbations that flip
    a factual prediction to a counterfactual target.

    Args:
        logits: Model output logits [vocab_size] or [batch, vocab_size]
        target_token_id: ID of counterfactual target
        original_token_id: ID of original factual token
        margin: Desired margin (default: 1.0 for stable optimization)

    Returns:
        loss: Scalar loss value

    Loss form:
        L = relu(margin - (logit_target - logit_orig))
        L = max(0, margin - logit_target + logit_orig)

    When logit_target > logit_orig + margin, loss = 0 (flip achieved).
    Otherwise, loss > 0 (need to optimize more).

    Scientific intent:
        Combined with L2 regularization on delta, this objective encourages
        the minimal-norm perturbation that achieves the flip by at least the margin.
    """
    if logits.dim() == 1:
        target_logit = logits[target_token_id]
        original_logit = logits[original_token_id]
    else:
        target_logit = logits[:, target_token_id]
        original_logit = logits[:, original_token_id]

    # Hinge loss: relu(margin - (target - original))
    loss = torch.clamp(margin - (target_logit - original_logit), min=0.0)

    return loss.mean()


def combined_loss(
    logits: torch.Tensor,
    target_token_id: int,
    original_token_id: Optional[int] = None,
    alpha: float = 1.0,
    beta: float = 0.0,
    margin: float = 0.0,
) -> torch.Tensor:
    """Combined loss for CIS optimization.

    Combines negative log likelihood of target with margin-based separation.

    Args:
        logits: Model output logits
        target_token_id: ID of counterfactual target
        original_token_id: ID of original factual token (optional)
        alpha: Weight for NLL term (default: 1.0)
        beta: Weight for margin term (default: 0.0)
        margin: Margin for margin_loss (default: 0.0)

    Returns:
        loss: Combined loss value

    Usage:
        # Pure NLL (maximize target probability):
        loss = combined_loss(logits, target_id, alpha=1.0, beta=0.0)

        # NLL + margin (maximize target AND beat original):
        loss = combined_loss(logits, target_id, original_id, alpha=1.0, beta=0.5)
    """
    loss = alpha * counterfactual_loss(logits, target_token_id)

    if beta > 0.0 and original_token_id is not None:
        loss = loss + beta * margin_loss(logits, target_token_id, original_token_id, margin)

    return loss


def regularization_loss(delta: torch.Tensor, reg_type: str = "l2", weight: float = 0.01) -> torch.Tensor:
    """Regularization term to keep perturbation small.

    Args:
        delta: Perturbation vector
        reg_type: Type of regularization ('l2', 'l1')
        weight: Regularization weight

    Returns:
        reg_loss: Regularization term to add to main loss

    Scientific intent:
        We want to find the MINIMAL perturbation that achieves the flip.
        Regularization encourages small ||δ||, which is what we measure
        as "geometric cost" of the factual change.
    """
    if reg_type == "l2":
        return weight * (delta**2).sum()
    elif reg_type == "l1":
        return weight * delta.abs().sum()
    else:
        raise ValueError(f"Unknown regularization type: {reg_type}")
