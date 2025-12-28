"""Run comprehensive CIS sweeps: margin sensitivity and layer-wise rigidity.

This script produces paper-grade measurements:
1. Margin sweep: How does geometric cost vary with margin?
2. Layer sweep: Which layers encode factual knowledge most rigidly?

All metrics are normalized for cross-experiment comparability.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.cis.metrics import get_comprehensive_metrics  # noqa: E402
from src.cis.optimizer import CISOptimizer  # noqa: E402
from src.hooks.residual_hooks import get_model_num_layers  # noqa: E402
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
) -> Dict[str, Any]:
    """Run a single CIS optimization and return comprehensive metrics.

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

    Returns:
        Dictionary with optimization results + normalized metrics
    """
    # Initialize optimizer
    optimizer = CISOptimizer(
        model=model,
        tokenizer=tokenizer,
        layer_idx=layer_idx,
        token_position=token_position,
        device=device,
    )

    # Run optimization
    results = optimizer.optimize(
        prompt=prompt,
        target_completion=target_completion,
        original_completion=original_completion,
        max_steps=max_steps,
        learning_rate=learning_rate,
        reg_weight=lambda_l2,
        reg_type="l2",
        loss_type="margin",
        margin=margin,
        tolerance=1e-6,
        early_stop_margin=0.5,
        verbose=False,
        log_every=10,
    )

    # Get normalized metrics
    normalized_metrics = get_comprehensive_metrics(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        layer_idx=layer_idx,
        token_position=token_position,
        delta_norm=results["geometric_cost"],
        device=device,
    )

    # Combine results
    return {
        "success": results["success"],
        "num_steps": results["num_steps"],
        "delta_norm": results["geometric_cost"],
        "target_prob": results["target_prob"],
        "original_prob": results["original_prob"],
        **normalized_metrics,
    }


def run_margin_sweep(
    model: Any,
    tokenizer: Any,
    prompt: str,
    target_completion: str,
    original_completion: str,
    layer_idx: int,
    token_position: int,
    lambda_l2: float,
    max_steps: int,
    learning_rate: float,
    device: str,
) -> List[Dict[str, Any]]:
    """Sweep over different margin values to measure cost sensitivity.

    Fixed: lambda_l2
    Varied: margin in [0.5, 1.0, 2.0, 4.0]

    Args:
        See run_single_optimization()

    Returns:
        List of result dictionaries, one per margin value
    """
    margin_values = [0.5, 1.0, 2.0, 4.0]
    results = []

    print("\n" + "=" * 80)
    print("MARGIN SWEEP: Measuring cost sensitivity to margin")
    print("=" * 80)
    print(f"Fixed: lambda_l2 = {lambda_l2:.1e}, layer = {layer_idx}")
    print(f"Varied: margin = {margin_values}")
    print()

    for margin in margin_values:
        print(f"Running margin = {margin}...")
        result = run_single_optimization(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            target_completion=target_completion,
            original_completion=original_completion,
            layer_idx=layer_idx,
            token_position=token_position,
            margin=margin,
            lambda_l2=lambda_l2,
            max_steps=max_steps,
            learning_rate=learning_rate,
            device=device,
        )
        result["margin"] = margin
        result["lambda_l2"] = lambda_l2
        result["layer_idx"] = layer_idx
        results.append(result)

        # Print quick summary
        status = "✓ SUCCESS" if result["success"] else "✗ FAILED"
        print(f"  {status}: ||δ|| = {result['delta_norm']:.4f}, "
              f"normalized = {result['normalized_cost_ratio']:.4f}, "
              f"steps = {result['num_steps']}")

    return results


def run_layer_sweep(
    model: Any,
    tokenizer: Any,
    prompt: str,
    target_completion: str,
    original_completion: str,
    token_position: int,
    margin: float,
    lambda_l2: float,
    max_steps: int,
    learning_rate: float,
    device: str,
) -> List[Dict[str, Any]]:
    """Sweep over different layers to measure layer-wise factual rigidity.

    Fixed: margin, lambda_l2
    Varied: layer in [8, 12, 16, 20, 24]

    Args:
        See run_single_optimization()

    Returns:
        List of result dictionaries, one per layer
    """
    layer_values = [8, 12, 16, 20, 24]
    results = []

    print("\n" + "=" * 80)
    print("LAYER SWEEP: Measuring layer-wise factual rigidity")
    print("=" * 80)
    print(f"Fixed: margin = {margin}, lambda_l2 = {lambda_l2:.1e}")
    print(f"Varied: layers = {layer_values}")
    print()

    for layer_idx in layer_values:
        print(f"Running layer = {layer_idx}...")
        result = run_single_optimization(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            target_completion=target_completion,
            original_completion=original_completion,
            layer_idx=layer_idx,
            token_position=token_position,
            margin=margin,
            lambda_l2=lambda_l2,
            max_steps=max_steps,
            learning_rate=learning_rate,
            device=device,
        )
        result["margin"] = margin
        result["lambda_l2"] = lambda_l2
        result["layer_idx"] = layer_idx
        results.append(result)

        # Print quick summary
        status = "✓ SUCCESS" if result["success"] else "✗ FAILED"
        print(f"  {status}: ||δ|| = {result['delta_norm']:.4f}, "
              f"normalized = {result['normalized_cost_ratio']:.4f}, "
              f"steps = {result['num_steps']}")

    return results


def print_margin_sweep_table(results: List[Dict[str, Any]]) -> None:
    """Print formatted table for margin sweep results."""
    print("\n" + "=" * 100)
    print("MARGIN SWEEP RESULTS")
    print("=" * 100)
    print(f"\n{'Margin':<10} {'Success':<10} {'Steps':<8} {'||δ||':<12} "
          f"{'Normalized':<12} {'RMS/dim':<12} {'P(target)':<12} {'P(orig)':<12}")
    print("-" * 100)

    for res in results:
        success_str = "✓ YES" if res["success"] else "✗ NO"
        print(
            f"{res['margin']:<10.1f} {success_str:<10} {res['num_steps']:<8} "
            f"{res['delta_norm']:<12.4f} {res['normalized_cost_ratio']:<12.4f} "
            f"{res['per_dim_rms']:<12.6f} {res['target_prob']:<12.4f} "
            f"{res['original_prob']:<12.4f}"
        )

    # Highlight minimal-norm successful solution
    successful = [r for r in results if r["success"]]
    if successful:
        minimal = min(successful, key=lambda x: x["normalized_cost_ratio"])
        print("\n" + "-" * 100)
        print("MINIMAL NORMALIZED COST:")
        print(f"  Margin: {minimal['margin']:.1f}")
        print(f"  Normalized cost ratio: {minimal['normalized_cost_ratio']:.4f}")
        print(f"  Raw ||δ||: {minimal['delta_norm']:.4f}")
        print(f"  Steps: {minimal['num_steps']}")


def print_layer_sweep_table(results: List[Dict[str, Any]]) -> None:
    """Print formatted table for layer sweep results."""
    print("\n" + "=" * 100)
    print("LAYER SWEEP RESULTS")
    print("=" * 100)
    print(f"\n{'Layer':<10} {'Success':<10} {'Steps':<8} {'||δ||':<12} "
          f"{'Normalized':<12} {'RMS/dim':<12} {'P(target)':<12} {'P(orig)':<12}")
    print("-" * 100)

    for res in results:
        success_str = "✓ YES" if res["success"] else "✗ NO"
        print(
            f"{res['layer_idx']:<10} {success_str:<10} {res['num_steps']:<8} "
            f"{res['delta_norm']:<12.4f} {res['normalized_cost_ratio']:<12.4f} "
            f"{res['per_dim_rms']:<12.6f} {res['target_prob']:<12.4f} "
            f"{res['original_prob']:<12.4f}"
        )

    # Highlight minimal-norm successful solution
    successful = [r for r in results if r["success"]]
    if successful:
        minimal = min(successful, key=lambda x: x["normalized_cost_ratio"])
        print("\n" + "-" * 100)
        print("MOST RIGID LAYER (highest normalized cost):")
        most_rigid = max(successful, key=lambda x: x["normalized_cost_ratio"])
        print(f"  Layer: {most_rigid['layer_idx']}")
        print(f"  Normalized cost ratio: {most_rigid['normalized_cost_ratio']:.4f}")
        print(f"  Raw ||δ||: {most_rigid['delta_norm']:.4f}")

        print("\nLEAST RIGID LAYER (lowest normalized cost):")
        print(f"  Layer: {minimal['layer_idx']}")
        print(f"  Normalized cost ratio: {minimal['normalized_cost_ratio']:.4f}")
        print(f"  Raw ||δ||: {minimal['delta_norm']:.4f}")


def save_results_jsonl(
    margin_results: List[Dict[str, Any]],
    layer_results: List[Dict[str, Any]],
    output_path: str,
) -> None:
    """Save results as JSON lines.

    Args:
        margin_results: Results from margin sweep
        layer_results: Results from layer sweep
        output_path: Path to save JSONL file
    """
    # Create output directory if needed
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        # Write margin sweep results
        for res in margin_results:
            record = {"sweep_type": "margin", **res}
            f.write(json.dumps(record) + "\n")

        # Write layer sweep results
        for res in layer_results:
            record = {"sweep_type": "layer", **res}
            f.write(json.dumps(record) + "\n")

    print(f"\n✓ Results saved to: {output_path}")


def run_sweeps(config_path: str) -> None:
    """Run both margin and layer sweeps.

    Args:
        config_path: Path to experiment config YAML
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
    print("CIS COMPREHENSIVE SWEEPS: Margin & Layer Analysis")
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
    num_layers = get_model_num_layers(model)

    # Get experiment parameters
    cis_config = exp_config.get("cis_optimization", {})

    # Fact details
    subject = exp_config.get("subject", "Eiffel Tower")
    relation = exp_config.get("relation", "located in")
    expected_completion = exp_config.get("expected_completion", " Paris")
    target_completion = cis_config.get("target_completion", " London")

    # Build prompt
    prompt = make_factual_prompt(subject, relation)

    print(f"\nFact Configuration:")
    print(f"  Prompt: {prompt!r}")
    print(f"  Expected: {expected_completion!r}")
    print(f"  Target: {target_completion!r}")
    print(f"  Model layers: {num_layers}")

    # Sweep parameters
    token_position = cis_config.get("token_position", -1)
    max_steps = cis_config.get("max_steps", 200)
    learning_rate = cis_config.get("learning_rate", 0.05)

    # Fixed parameters for sweeps
    lambda_l2_fixed = 0.01  # Fixed for margin sweep
    margin_fixed = 1.0  # Fixed for layer sweep
    layer_fixed = 16  # Fixed for margin sweep

    # Run margin sweep
    margin_results = run_margin_sweep(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        target_completion=target_completion,
        original_completion=expected_completion,
        layer_idx=layer_fixed,
        token_position=token_position,
        lambda_l2=lambda_l2_fixed,
        max_steps=max_steps,
        learning_rate=learning_rate,
        device=device,
    )

    # Run layer sweep
    layer_results = run_layer_sweep(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        target_completion=target_completion,
        original_completion=expected_completion,
        token_position=token_position,
        margin=margin_fixed,
        lambda_l2=lambda_l2_fixed,
        max_steps=max_steps,
        learning_rate=learning_rate,
        device=device,
    )

    # Print summary tables
    print_margin_sweep_table(margin_results)
    print_layer_sweep_table(layer_results)

    # Save results
    output_path = "artifacts/cis_sweeps.jsonl"
    save_results_jsonl(margin_results, layer_results, output_path)

    print("\n" + "=" * 80)
    print("SWEEPS COMPLETE")
    print("=" * 80)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run CIS margin and layer sweeps")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment YAML config")
    return parser.parse_args()


def main() -> None:
    """CLI wrapper for CIS sweeps."""
    args = parse_args()
    run_sweeps(args.config)


if __name__ == "__main__":
    main()
