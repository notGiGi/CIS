"""Learning rate sweep with post-hoc bisection for minimal delta estimation.

This script addresses the overshoot problem in CIS optimization:
- Problem: Optimizer overshoots margin (gap=6.58 when margin=1.0)
- Solution 1: Test multiple learning rates to reduce overshoot
- Solution 2: Post-hoc bisection to find minimal alpha such that gap(alpha*delta) >= margin

This produces defensible near-minimal geometric cost estimates.
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.optim as optim
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.cis.delta import LearnableDelta  # noqa: E402
from src.cis.losses import margin_flip_loss, regularization_loss  # noqa: E402
from src.hooks.residual_hooks import (  # noqa: E402
    add_residual_perturbation_hook,
    get_hidden_size,
)
from src.models.load_model import load_model_and_tokenizer  # noqa: E402
from src.prompts.factual_prompts import make_factual_prompt  # noqa: E402


def load_config(path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_single_optimization(
    model: Any,
    tokenizer: Any,
    prompt: str,
    target_completion: str,
    original_completion: str,
    layer_idx: int,
    token_position: int,
    margin: float,
    lambda_l2: float,
    max_steps: int,
    learning_rate: float,
    device: str,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run CIS optimization with specified learning rate.

    Args:
        model: Transformer model
        tokenizer: Tokenizer
        prompt: Input text
        target_completion: Counterfactual target
        original_completion: Factual completion
        layer_idx: Layer to intervene on
        token_position: Token position to intervene on
        margin: Margin for margin_flip_loss
        lambda_l2: L2 regularization weight
        max_steps: Maximum optimization steps
        learning_rate: Learning rate
        device: Device
        verbose: If True, print iteration details

    Returns:
        Dictionary with optimization results
    """
    # Tokenize
    target_token_id = tokenizer.encode(target_completion, add_special_tokens=False)[0]
    original_token_id = tokenizer.encode(original_completion, add_special_tokens=False)[0]

    # Initialize delta
    hidden_size = get_hidden_size(model)
    delta = LearnableDelta(
        hidden_dim=hidden_size,
        init_method="zeros",
        device=device,
    )

    # Verify zero initialization
    initial_norm = delta.get_norm(p=2)
    assert initial_norm < 1e-6, f"Delta not initialized to zero! ||δ|| = {initial_norm}"

    # Initialize optimizer
    optimizer = optim.Adam(delta.parameters(), lr=learning_rate)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Optimization loop
    if verbose:
        print(f"\nOptimizing with LR={learning_rate}...")
        print(f"{'Iter':<6} {'||δ||':<12} {'Gap':<10} {'Loss':<10}")
        print("-" * 50)

    success = False
    final_gap = None
    final_delta_norm = None
    final_target_prob = None
    final_original_prob = None

    for step in range(max_steps):
        delta_value = delta()
        delta_norm = delta.get_norm(p=2)

        optimizer.zero_grad()

        # Forward pass with hook
        handle = None
        try:
            handle, _ = add_residual_perturbation_hook(
                model=model,
                layer_idx=layer_idx,
                delta_vector=delta_value,
                token_position=token_position,
            )

            with torch.set_grad_enabled(True):
                outputs = model(**inputs)
                logits = outputs.logits[0, -1, :]

                # Extract logits
                logit_target = logits[target_token_id]
                logit_original = logits[original_token_id]

                # Compute gap
                gap = (logit_target - logit_original).item()

                # Compute loss
                task_loss = margin_flip_loss(
                    logits,
                    target_token_id,
                    original_token_id,
                    margin=margin,
                )
                reg_loss = regularization_loss(delta_value, reg_type="l2", weight=lambda_l2)
                total_loss = task_loss + reg_loss

            # Backward pass
            total_loss.backward()

        finally:
            if handle is not None:
                handle.remove()

        # Log
        if verbose and step % 10 == 0:
            print(f"{step:<6} {delta_norm:<12.6f} {gap:<10.4f} {total_loss.item():<10.4f}")

        # Apply gradient step
        optimizer.step()

        # Check stopping condition
        if gap >= margin:
            # Get final state AFTER this step
            final_delta_norm = delta.get_norm(p=2)
            final_gap = gap

            # Get probabilities
            with torch.no_grad():
                probs = torch.softmax(logits, dim=-1)
                final_target_prob = probs[target_token_id].item()
                final_original_prob = probs[original_token_id].item()

            success = True
            if verbose:
                print(f"\n✓ Converged at step {step}")
                print(f"  Gap: {final_gap:.4f} >= margin {margin:.4f}")
                print(f"  Final ||δ||: {final_delta_norm:.6f}")
            break

    if not success:
        # Failed to converge
        final_delta_norm = delta.get_norm(p=2)
        final_gap = gap
        with torch.no_grad():
            probs = torch.softmax(logits, dim=-1)
            final_target_prob = probs[target_token_id].item()
            final_original_prob = probs[original_token_id].item()

    # Return final delta vector for bisection
    final_delta_vector = delta().detach().clone()

    return {
        "success": success,
        "num_steps": step + 1 if success else max_steps,
        "final_gap": final_gap,
        "final_delta_norm": final_delta_norm,
        "final_delta_vector": final_delta_vector,
        "target_prob": final_target_prob,
        "original_prob": final_original_prob,
    }


def evaluate_gap_at_scale(
    model: Any,
    tokenizer: Any,
    prompt: str,
    target_completion: str,
    original_completion: str,
    layer_idx: int,
    token_position: int,
    delta_vector: torch.Tensor,
    alpha: float,
    device: str,
) -> float:
    """Evaluate gap when applying alpha * delta_vector.

    Args:
        model: Transformer model
        tokenizer: Tokenizer
        prompt: Input text
        target_completion: Counterfactual target
        original_completion: Factual completion
        layer_idx: Layer to intervene on
        token_position: Token position
        delta_vector: Delta vector to scale
        alpha: Scaling factor
        device: Device

    Returns:
        gap: logit_target - logit_original
    """
    # Tokenize
    target_token_id = tokenizer.encode(target_completion, add_special_tokens=False)[0]
    original_token_id = tokenizer.encode(original_completion, add_special_tokens=False)[0]

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Scale delta
    scaled_delta = alpha * delta_vector

    # Forward pass with scaled delta
    handle = None
    try:
        handle, _ = add_residual_perturbation_hook(
            model=model,
            layer_idx=layer_idx,
            delta_vector=scaled_delta,
            token_position=token_position,
        )

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0, -1, :]

            # Extract logits
            logit_target = logits[target_token_id]
            logit_original = logits[original_token_id]

            # Compute gap
            gap = (logit_target - logit_original).item()

    finally:
        if handle is not None:
            handle.remove()

    return gap


def bisection_find_minimal_alpha(
    model: Any,
    tokenizer: Any,
    prompt: str,
    target_completion: str,
    original_completion: str,
    layer_idx: int,
    token_position: int,
    delta_vector: torch.Tensor,
    margin: float,
    device: str,
    num_iterations: int = 20,
    verbose: bool = False,
) -> Tuple[float, float]:
    """Find minimal alpha such that gap(alpha * delta) >= margin using bisection.

    Args:
        model: Transformer model
        tokenizer: Tokenizer
        prompt: Input text
        target_completion: Counterfactual target
        original_completion: Factual completion
        layer_idx: Layer to intervene on
        token_position: Token position
        delta_vector: Learned delta vector
        margin: Required margin
        device: Device
        num_iterations: Number of bisection iterations
        verbose: If True, print iteration details

    Returns:
        Tuple of (alpha_star, gap_at_alpha_star)
    """
    # Binary search in [0, 1]
    alpha_low = 0.0
    alpha_high = 1.0

    # Verify that alpha=1.0 satisfies margin (should be true if optimization succeeded)
    gap_at_one = evaluate_gap_at_scale(
        model, tokenizer, prompt, target_completion, original_completion,
        layer_idx, token_position, delta_vector, 1.0, device
    )

    if gap_at_one < margin:
        # Optimization didn't actually succeed - return alpha=1.0
        if verbose:
            print(f"  Warning: gap at alpha=1.0 is {gap_at_one:.4f} < margin {margin:.4f}")
        return 1.0, gap_at_one

    if verbose:
        print(f"\nBisection search (margin={margin:.4f}):")
        print(f"  Initial: alpha=1.0, gap={gap_at_one:.4f}")

    # Bisection
    for i in range(num_iterations):
        alpha_mid = (alpha_low + alpha_high) / 2.0

        gap_mid = evaluate_gap_at_scale(
            model, tokenizer, prompt, target_completion, original_completion,
            layer_idx, token_position, delta_vector, alpha_mid, device
        )

        if verbose and i % 5 == 0:
            print(f"  Iter {i}: alpha={alpha_mid:.6f}, gap={gap_mid:.4f}")

        if gap_mid >= margin:
            # Can reduce alpha further
            alpha_high = alpha_mid
        else:
            # Need higher alpha
            alpha_low = alpha_mid

    # Final alpha is alpha_high (smallest alpha that satisfies margin)
    alpha_star = alpha_high
    gap_star = evaluate_gap_at_scale(
        model, tokenizer, prompt, target_completion, original_completion,
        layer_idx, token_position, delta_vector, alpha_star, device
    )

    if verbose:
        print(f"  Final: alpha*={alpha_star:.6f}, gap={gap_star:.4f}")

    return alpha_star, gap_star


def run_lr_sweep(config_path: str, verbose: bool = False) -> None:
    """Run learning rate sweep with post-hoc bisection.

    Args:
        config_path: Path to experiment config YAML
        verbose: If True, print detailed logs
    """
    import time

    # Load configuration
    exp_config = load_config(config_path)
    model_config_path = exp_config.get("model_config", "config/model.yaml")
    model_config = load_config(model_config_path)

    # Set seed
    seed = exp_config.get("seed", 0)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print("=" * 80)
    print("LEARNING RATE SWEEP WITH POST-HOC BISECTION")
    print("=" * 80)
    print(f"\nSeed: {seed}")
    print(f"Model: {model_config.get('model_name', 'mistralai/Mistral-7B-v0.1')}")

    # Load model
    print("\nLoading model and tokenizer...")
    start_time = time.time()
    model, tokenizer = load_model_and_tokenizer(model_config)
    load_time = time.time() - start_time
    print(f"✓ Model loaded in {load_time:.2f}s")

    device = model_config.get("device", "cuda")

    # Get experiment parameters
    subject = exp_config.get("subject", "Eiffel Tower")
    relation = exp_config.get("relation", "located in")
    expected_completion = exp_config.get("expected_completion", " Paris")
    cis_config = exp_config.get("cis_optimization", {})
    target_completion = cis_config.get("target_completion", " London")

    # Build prompt
    prompt = make_factual_prompt(subject, relation)

    print(f"\nFact Configuration:")
    print(f"  Prompt: {prompt!r}")
    print(f"  Expected: {expected_completion!r}")
    print(f"  Target: {target_completion!r}")

    # Fixed parameters
    layer_idx = 16
    token_position = -1
    margin = 1.0
    lambda_l2 = 0.01
    max_steps = 200

    # Learning rates to sweep
    learning_rates = [0.05, 0.01, 0.005, 0.001]

    print(f"\nFixed Parameters:")
    print(f"  Layer: {layer_idx}")
    print(f"  Token position: {token_position}")
    print(f"  Margin: {margin}")
    print(f"  Lambda: {lambda_l2:.1e}")
    print(f"  Max steps: {max_steps}")

    print(f"\nLearning Rates: {learning_rates}")

    # Results storage
    results = []

    # Run sweep
    print("\n" + "=" * 80)
    print("OPTIMIZATION SWEEP")
    print("=" * 80)

    for lr in learning_rates:
        print(f"\n{'='*80}")
        print(f"Learning Rate: {lr}")
        print(f"{'='*80}")

        # Run optimization
        opt_result = run_single_optimization(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            target_completion=target_completion,
            original_completion=expected_completion,
            layer_idx=layer_idx,
            token_position=token_position,
            margin=margin,
            lambda_l2=lambda_l2,
            max_steps=max_steps,
            learning_rate=lr,
            device=device,
            verbose=verbose,
        )

        if not opt_result["success"]:
            print(f"✗ FAILED to converge with LR={lr}")
            results.append({
                "lr": lr,
                "success": False,
                "num_steps": opt_result["num_steps"],
                "delta_found_norm": opt_result["final_delta_norm"],
                "gap_found": opt_result["final_gap"],
                "alpha_star": None,
                "delta_scaled_norm": None,
                "gap_scaled": None,
            })
            continue

        # Run bisection
        print(f"\nRunning bisection to find minimal alpha...")
        alpha_star, gap_star = bisection_find_minimal_alpha(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            target_completion=target_completion,
            original_completion=expected_completion,
            layer_idx=layer_idx,
            token_position=token_position,
            delta_vector=opt_result["final_delta_vector"],
            margin=margin,
            device=device,
            num_iterations=20,
            verbose=verbose,
        )

        # Compute scaled norm
        delta_scaled_norm = alpha_star * opt_result["final_delta_norm"]

        print(f"\n✓ Results for LR={lr}:")
        print(f"  Optimization:")
        print(f"    Steps: {opt_result['num_steps']}")
        print(f"    ||δ_found||: {opt_result['final_delta_norm']:.6f}")
        print(f"    Gap: {opt_result['final_gap']:.4f}")
        print(f"  Bisection:")
        print(f"    alpha*: {alpha_star:.6f}")
        print(f"    ||δ_scaled||: {delta_scaled_norm:.6f}")
        print(f"    Gap: {gap_star:.4f}")
        print(f"  Reduction: {(1 - alpha_star) * 100:.2f}%")

        results.append({
            "lr": lr,
            "success": True,
            "num_steps": opt_result["num_steps"],
            "delta_found_norm": opt_result["final_delta_norm"],
            "gap_found": opt_result["final_gap"],
            "alpha_star": alpha_star,
            "delta_scaled_norm": delta_scaled_norm,
            "gap_scaled": gap_star,
        })

    # Print summary table
    print("\n" + "=" * 100)
    print("SUMMARY TABLE: Learning Rate Sweep with Bisection")
    print("=" * 100)
    print(f"\n{'LR':<10} {'Success':<10} {'Steps':<8} {'||δ_found||':<14} {'Gap_found':<12} "
          f"{'alpha*':<10} {'||δ_scaled||':<14} {'Gap_scaled':<12} {'Reduction':<12}")
    print("-" * 100)

    for res in results:
        if not res["success"]:
            print(f"{res['lr']:<10.4f} {'✗ NO':<10} {res['num_steps']:<8} "
                  f"{res['delta_found_norm']:<14.6f} {res['gap_found']:<12.4f} "
                  f"{'N/A':<10} {'N/A':<14} {'N/A':<12} {'N/A':<12}")
        else:
            reduction_pct = (1 - res["alpha_star"]) * 100
            print(f"{res['lr']:<10.4f} {'✓ YES':<10} {res['num_steps']:<8} "
                  f"{res['delta_found_norm']:<14.6f} {res['gap_found']:<12.4f} "
                  f"{res['alpha_star']:<10.6f} {res['delta_scaled_norm']:<14.6f} "
                  f"{res['gap_scaled']:<12.4f} {reduction_pct:<12.2f}%")

    # Find best result (minimal scaled norm among successful runs)
    successful = [r for r in results if r["success"]]
    if successful:
        best = min(successful, key=lambda x: x["delta_scaled_norm"])
        print("\n" + "-" * 100)
        print("BEST RESULT (minimal ||δ_scaled||):")
        print(f"  Learning rate: {best['lr']}")
        print(f"  ||δ_scaled||: {best['delta_scaled_norm']:.6f}")
        print(f"  Gap at boundary: {best['gap_scaled']:.4f} (margin: {margin:.4f})")
        print(f"  Alpha*: {best['alpha_star']:.6f}")
        print(f"  Reduction from found delta: {(1 - best['alpha_star']) * 100:.2f}%")

    print("\n" + "=" * 100)
    print("INTERPRETATION")
    print("=" * 100)
    print("\nThis table shows:")
    print("1. ||δ_found||: Norm of delta from optimization (may overshoot margin)")
    print("2. Gap_found: Actual gap achieved (typically > margin)")
    print("3. alpha*: Minimal scaling factor to hit margin boundary")
    print("4. ||δ_scaled||: Near-minimal norm = alpha* × ||δ_found||")
    print("5. Reduction: How much overshoot was eliminated")
    print("\nLower learning rates typically reduce overshoot.")
    print("Bisection further refines to find the margin boundary.")
    print(f"\nThe best estimate of minimal geometric cost is: {best['delta_scaled_norm']:.6f}")

    print("\n" + "=" * 100)
    print("SWEEP COMPLETE")
    print("=" * 100)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run learning rate sweep with post-hoc bisection for minimal delta estimation"
    )
    parser.add_argument("--config", type=str, required=True, help="Path to experiment YAML config")
    parser.add_argument("--verbose", action="store_true", help="Print detailed optimization logs")
    return parser.parse_args()


def main() -> None:
    """CLI wrapper for LR sweep with bisection."""
    args = parse_args()
    run_lr_sweep(args.config, verbose=args.verbose)


if __name__ == "__main__":
    main()
