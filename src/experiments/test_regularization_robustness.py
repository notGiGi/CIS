"""Test regularization robustness of minimal delta estimation.

This script tests whether the minimal geometric cost estimate is robust to
the L2 regularization hyperparameter (lambda).

We run LR=0.001 twice:
- lambda_l2 = 0.0 (no regularization)
- lambda_l2 = 0.01 (original setting)

If both produce similar ||delta_scaled|| after bisection, this validates
that the minimal cost is a property of the problem, not an artifact of
regularization.
"""

import argparse
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.experiments.run_lr_sweep_with_bisection import (  # noqa: E402
    bisection_find_minimal_alpha,
    load_config,
    run_single_optimization,
)
from src.models.load_model import load_model_and_tokenizer  # noqa: E402
from src.prompts.factual_prompts import make_factual_prompt  # noqa: E402


def test_regularization_robustness(config_path: str, verbose: bool = False) -> None:
    """Test robustness to regularization by comparing lambda=0.0 vs lambda=0.01.

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
    print("REGULARIZATION ROBUSTNESS TEST")
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
    max_steps = 200
    learning_rate = 0.001  # Best LR from sweep

    # Test two lambda values
    lambda_values = [0.0, 0.01]

    print(f"\nFixed Parameters:")
    print(f"  Layer: {layer_idx}")
    print(f"  Token position: {token_position}")
    print(f"  Margin: {margin}")
    print(f"  Learning rate: {learning_rate} (best from sweep)")
    print(f"  Max steps: {max_steps}")

    print(f"\nTesting Lambda values: {lambda_values}")
    print(f"  Lambda=0.0: No regularization")
    print(f"  Lambda=0.01: Original setting")

    # Results storage
    results = []

    # Run for each lambda
    print("\n" + "=" * 80)
    print("OPTIMIZATION + BISECTION")
    print("=" * 80)

    for lambda_l2 in lambda_values:
        print(f"\n{'='*80}")
        print(f"Lambda: {lambda_l2}")
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
            learning_rate=learning_rate,
            device=device,
            verbose=verbose,
        )

        if not opt_result["success"]:
            print(f"✗ FAILED to converge with lambda={lambda_l2}")
            results.append({
                "lambda": lambda_l2,
                "success": False,
                "num_steps": opt_result["num_steps"],
                "delta_found_norm": opt_result["final_delta_norm"],
                "gap_found": opt_result["final_gap"],
                "gap_recomputed": opt_result["gap_recomputed"],
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

        print(f"\n✓ Results for lambda={lambda_l2}:")
        print(f"  Optimization:")
        print(f"    Steps: {opt_result['num_steps']}")
        print(f"    ||δ_found||: {opt_result['final_delta_norm']:.6f}")
        print(f"    Gap (optimization): {opt_result['final_gap']:.6f}")
        print(f"    Gap (recomputed): {opt_result['gap_recomputed']:.6f}")
        print(f"  Bisection:")
        print(f"    alpha*: {alpha_star:.6f}")
        print(f"    ||δ_scaled||: {delta_scaled_norm:.6f}")
        print(f"    Gap: {gap_star:.4f}")
        print(f"  Reduction: {(1 - alpha_star) * 100:.2f}%")

        results.append({
            "lambda": lambda_l2,
            "success": True,
            "num_steps": opt_result["num_steps"],
            "delta_found_norm": opt_result["final_delta_norm"],
            "gap_found": opt_result["final_gap"],
            "gap_recomputed": opt_result["gap_recomputed"],
            "alpha_star": alpha_star,
            "delta_scaled_norm": delta_scaled_norm,
            "gap_scaled": gap_star,
        })

    # Print comparison table
    print("\n" + "=" * 100)
    print("REGULARIZATION ROBUSTNESS: Comparison Table")
    print("=" * 100)
    print(f"\n{'Lambda':<10} {'Success':<10} {'Steps':<8} {'||δ_found||':<14} {'Gap_found':<12} "
          f"{'Gap_recomp':<12} {'alpha*':<10} {'||δ_scaled||':<14} {'Gap_scaled':<12}")
    print("-" * 100)

    for res in results:
        if not res["success"]:
            print(f"{res['lambda']:<10.4f} {'✗ NO':<10} {res['num_steps']:<8} "
                  f"{res['delta_found_norm']:<14.6f} {res['gap_found']:<12.4f} "
                  f"{res['gap_recomputed']:<12.4f} {'N/A':<10} {'N/A':<14} {'N/A':<12}")
        else:
            print(f"{res['lambda']:<10.4f} {'✓ YES':<10} {res['num_steps']:<8} "
                  f"{res['delta_found_norm']:<14.6f} {res['gap_found']:<12.4f} "
                  f"{res['gap_recomputed']:<12.4f} {res['alpha_star']:<10.6f} "
                  f"{res['delta_scaled_norm']:<14.6f} {res['gap_scaled']:<12.4f}")

    # Robustness analysis
    successful = [r for r in results if r["success"]]
    if len(successful) == 2:
        lambda_0 = successful[0]
        lambda_001 = successful[1]

        delta_scaled_0 = lambda_0["delta_scaled_norm"]
        delta_scaled_001 = lambda_001["delta_scaled_norm"]

        diff_abs = abs(delta_scaled_0 - delta_scaled_001)
        diff_rel = diff_abs / min(delta_scaled_0, delta_scaled_001) * 100

        print("\n" + "-" * 100)
        print("ROBUSTNESS ANALYSIS:")
        print(f"  ||δ_scaled|| with lambda=0.0: {delta_scaled_0:.6f}")
        print(f"  ||δ_scaled|| with lambda=0.01: {delta_scaled_001:.6f}")
        print(f"  Absolute difference: {diff_abs:.6f}")
        print(f"  Relative difference: {diff_rel:.2f}%")

        if diff_rel < 10.0:
            print(f"\n  ✓ ROBUST: Minimal cost is consistent (<10% variation)")
            print(f"    The estimate is NOT an artifact of regularization.")
        elif diff_rel < 25.0:
            print(f"\n  ~ MODERATE: Some variation (10-25%)")
            print(f"    Regularization has moderate effect on minimal cost.")
        else:
            print(f"\n  ✗ SENSITIVE: High variation (>25%)")
            print(f"    Regularization significantly affects minimal cost estimate.")
            print(f"    This suggests the problem may be ill-conditioned.")

    print("\n" + "=" * 100)
    print("INTERPRETATION")
    print("=" * 100)
    print("\nThis test validates whether the minimal geometric cost is:")
    print("1. An intrinsic property of the factual encoding (robust to lambda)")
    print("2. Or an artifact of the regularization hyperparameter (sensitive to lambda)")
    print("\nIf ||δ_scaled|| is similar for both lambda values, the estimate is robust.")
    print("If they differ significantly, the cost depends on optimization details.")

    if len(successful) == 2:
        print(f"\nResult: {diff_rel:.2f}% variation → ", end="")
        if diff_rel < 10.0:
            print("ROBUST ✓")
        elif diff_rel < 25.0:
            print("MODERATE")
        else:
            print("SENSITIVE ✗")

    print("\n" + "=" * 100)
    print("TEST COMPLETE")
    print("=" * 100)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Test regularization robustness of minimal delta estimation"
    )
    parser.add_argument("--config", type=str, required=True, help="Path to experiment YAML config")
    parser.add_argument("--verbose", action="store_true", help="Print detailed optimization logs")
    return parser.parse_args()


def main() -> None:
    """CLI wrapper for regularization robustness test."""
    args = parse_args()
    test_regularization_robustness(args.config, verbose=args.verbose)


if __name__ == "__main__":
    main()
