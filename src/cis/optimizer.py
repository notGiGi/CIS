"""CIS Optimizer: Find minimal perturbations to flip factual predictions.

This module implements the core optimization loop for Counterfactual Internal State
experiments, using gradient descent to find the smallest δ that achieves a target prediction.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.optim as optim

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.cis.delta import LearnableDelta  # noqa: E402
from src.cis.losses import combined_loss, margin_flip_loss, regularization_loss  # noqa: E402
from src.hooks.residual_hooks import (  # noqa: E402
    add_residual_perturbation_hook,
    get_hidden_size,
)
from src.utils.token_utils import decode_topk_predictions  # noqa: E402


class CISOptimizer:
    """Optimizer for finding minimal counterfactual internal states.

    This class orchestrates the optimization process:
    1. Initialize learnable delta
    2. Attach hook to inject delta
    3. Optimize delta via gradient descent
    4. Monitor convergence
    5. Report results
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        layer_idx: int,
        token_position: int = -1,
        device: str = "cuda",
    ):
        """Initialize CIS optimizer.

        Args:
            model: Frozen transformer model
            tokenizer: Tokenizer
            layer_idx: Which layer to intervene on
            token_position: Which token position to perturb (-1 = last)
            device: Device for computations
        """
        self.model = model
        self.tokenizer = tokenizer
        self.layer_idx = layer_idx
        self.token_position = token_position
        self.device = device

        # Get model dimensions
        self.hidden_size = get_hidden_size(model)

        # Will be initialized in optimize()
        self.delta: Optional[LearnableDelta] = None
        self.optimizer: Optional[optim.Optimizer] = None

    def optimize(
        self,
        prompt: str,
        target_completion: str,
        original_completion: Optional[str] = None,
        max_steps: int = 200,
        learning_rate: float = 0.05,
        reg_weight: float = 0.01,
        reg_type: str = "l2",
        loss_type: str = "nll",
        margin: float = 0.0,
        alpha: float = 1.0,
        beta: float = 0.0,
        tolerance: float = 1e-4,
        early_stop_margin: float = 0.5,
        verbose: bool = True,
        log_every: int = 10,
    ) -> Dict[str, Any]:
        """Run CIS optimization to find minimal perturbation.

        Args:
            prompt: Input text (e.g., "The Eiffel Tower is located in")
            target_completion: Counterfactual target (e.g., " London")
            original_completion: Original factual completion (e.g., " Paris", optional)
            max_steps: Maximum optimization steps
            learning_rate: Step size for gradient descent
            reg_weight: Weight for L2 regularization
            reg_type: Regularization type ('l2', 'l1')
            loss_type: Loss function ('nll', 'margin', 'combined')
            margin: Margin for margin loss
            alpha: Weight for NLL term (if loss_type='combined')
            beta: Weight for margin term (if loss_type='combined')
            tolerance: Convergence threshold for loss
            early_stop_margin: Stop if P(target) > this threshold
            verbose: Print progress
            log_every: Log every N steps

        Returns:
            results: Dictionary with:
                - delta: Final perturbation tensor
                - geometric_cost: ||δ||_2
                - success: Whether target became top-1
                - target_prob: Final P(target)
                - original_prob: Final P(original) if provided
                - num_steps: Steps taken
                - final_loss: Final loss value
                - top_predictions: Top-5 final predictions
                - optimization_history: List of stats per step
        """
        # Tokenize target and original
        target_token_id = self.tokenizer.encode(target_completion, add_special_tokens=False)[0]

        original_token_id = None
        if original_completion is not None:
            original_token_id = self.tokenizer.encode(original_completion, add_special_tokens=False)[0]

        # Initialize learnable delta
        self.delta = LearnableDelta(
            hidden_dim=self.hidden_size,
            init_method="zeros",
            device=self.device,
        )

        # Verify delta starts at zero
        initial_norm = self.delta.get_norm(p=2)
        assert initial_norm < 1e-6, f"Delta should start at zero, but ||δ|| = {initial_norm}"

        # Initialize optimizer (only optimize delta)
        self.optimizer = optim.Adam(self.delta.parameters(), lr=learning_rate)

        # Tokenize prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Track history
        history = []

        if verbose:
            print(f"\nOptimizing delta at layer {self.layer_idx}, token position {self.token_position}")
            print(f"Target: {target_completion!r} (token_id={target_token_id})")
            if original_completion:
                print(f"Original: {original_completion!r} (token_id={original_token_id})")
            print(f"\nLearning rate: {learning_rate}, Reg weight: {reg_weight}")
            print(f"Loss type: {loss_type}, Max steps: {max_steps}")
            print(f"Initial ||δ|| = {initial_norm:.8f} (should be ~0.0)\n")

        # Optimization loop
        for step in range(max_steps):
            # Log delta norm BEFORE optimizer step (for first 3 iterations)
            if step < 3 and verbose:
                delta_norm_before = self.delta.get_norm(p=2)
                print(f"[Step {step}] BEFORE step: ||δ|| = {delta_norm_before:.8f}")

            self.optimizer.zero_grad()

            # Get current delta value
            delta_value = self.delta()

            # Forward pass with intervention
            handle, _ = add_residual_perturbation_hook(
                model=self.model,
                layer_idx=self.layer_idx,
                delta_vector=delta_value,
                token_position=self.token_position,
            )

            try:
                # Run model
                with torch.set_grad_enabled(True):
                    outputs = self.model(**inputs)
                    logits = outputs.logits[0, -1, :]  # [vocab_size]

                    # Compute task loss
                    if loss_type == "nll":
                        task_loss = combined_loss(
                            logits,
                            target_token_id,
                            original_token_id=None,
                            alpha=1.0,
                            beta=0.0,
                        )
                    elif loss_type == "margin":
                        if original_token_id is None:
                            raise ValueError("margin loss requires original_completion")
                        # Use margin_flip_loss for scientifically valid minimal-norm optimization
                        task_loss = margin_flip_loss(
                            logits,
                            target_token_id,
                            original_token_id,
                            margin=margin,
                        )
                    elif loss_type == "combined":
                        task_loss = combined_loss(
                            logits,
                            target_token_id,
                            original_token_id,
                            alpha=alpha,
                            beta=beta,
                            margin=margin,
                        )
                    else:
                        raise ValueError(f"Unknown loss_type: {loss_type}")

                    # Add regularization
                    reg_loss = regularization_loss(delta_value, reg_type=reg_type, weight=reg_weight)

                    # Total loss
                    total_loss = task_loss + reg_loss

                # Backward pass
                total_loss.backward()

                # Gradient step
                self.optimizer.step()

            finally:
                # Always remove hook
                handle.remove()

            # Compute metrics (no grad)
            with torch.no_grad():
                probs = torch.softmax(logits, dim=-1)
                target_prob = probs[target_token_id].item()
                original_prob = probs[original_token_id].item() if original_token_id is not None else None

                delta_norm = self.delta.get_norm(p=2)

                # Check if target is top-1
                top1_id = logits.argmax().item()
                success = top1_id == target_token_id

            # Log delta norm AFTER optimizer step (for first 3 iterations)
            if step < 3 and verbose:
                print(f"[Step {step}] AFTER step:  ||δ|| = {delta_norm:.8f}")
                print()

            # Log
            if verbose and (step % log_every == 0 or success):
                log_str = f"Step {step:3d}: Loss={total_loss.item():.4f}, "
                log_str += f"P(target)={target_prob:.4f}, "
                if original_prob is not None:
                    log_str += f"P(orig)={original_prob:.4f}, "
                log_str += f"||δ||={delta_norm:.4f}"
                if success:
                    log_str += "  ← TARGET IS TOP-1!"
                print(log_str)

            # Record history
            history.append(
                {
                    "step": step,
                    "loss": total_loss.item(),
                    "task_loss": task_loss.item(),
                    "reg_loss": reg_loss.item(),
                    "target_prob": target_prob,
                    "original_prob": original_prob,
                    "delta_norm": delta_norm,
                    "success": success,
                }
            )

            # Early stopping
            if success and target_prob > early_stop_margin:
                if verbose:
                    print(f"\n✓ Converged at step {step}!")
                break

            if total_loss.item() < tolerance:
                if verbose:
                    print(f"\n✓ Loss below tolerance at step {step}!")
                break

        # Final evaluation
        with torch.no_grad():
            delta_value = self.delta()
            handle, _ = add_residual_perturbation_hook(
                model=self.model,
                layer_idx=self.layer_idx,
                delta_vector=delta_value,
                token_position=self.token_position,
            )

            try:
                outputs = self.model(**inputs)
                logits = outputs.logits[0, -1, :]
                probs = torch.softmax(logits, dim=-1)

                top_preds = decode_topk_predictions(self.tokenizer, logits, k=5)
                target_prob = probs[target_token_id].item()
                original_prob = probs[original_token_id].item() if original_token_id is not None else None
                success = logits.argmax().item() == target_token_id

            finally:
                handle.remove()

        # Prepare results
        results = {
            "delta": self.delta.delta.detach().cpu(),
            "geometric_cost": self.delta.get_norm(p=2),
            "success": success,
            "target_prob": target_prob,
            "original_prob": original_prob,
            "num_steps": step + 1,
            "final_loss": history[-1]["loss"],
            "top_predictions": top_preds,
            "optimization_history": history,
            "delta_stats": self.delta.get_stats(),
        }

        return results
