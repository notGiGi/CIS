"""Run CIS optimization to find minimal perturbation that flips a factual prediction.

This experiment implements the core CIS methodology:
1. Load frozen model
2. Select a fact and counterfactual target
3. Optimize δ via gradient descent to flip the prediction
4. Measure geometric cost (||δ||) as factual rigidity

Scientific intent:
This quantifies how "rigid" a factual representation is by measuring
the minimal internal perturbation needed to change the prediction.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.cis.optimizer import CISOptimizer  # noqa: E402
from src.hooks.residual_hooks import get_model_num_layers  # noqa: E402
from src.models.load_model import load_model_and_tokenizer  # noqa: E402
from src.prompts.factual_prompts import make_factual_prompt  # noqa: E402


def load_config(path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_cis_experiment(config_path: str) -> None:
    """Execute CIS optimization experiment.

    Args:
        config_path: Path to experiment YAML config
    """
    import time

    # Load configuration
    exp_config = load_config(config_path)
    model_config_path = exp_config.get("model_config", "config/model.yaml")
    model_config = load_config(model_config_path)

    # Set seed for reproducibility
    seed = exp_config.get("seed", 0)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print("=" * 80)
    print("CIS OPTIMIZATION: Counterfactual Internal State")
    print("=" * 80)
    print(f"\nSeed: {seed}")
    print(f"Model: {model_config.get('model_name', 'mistralai/Mistral-7B-v0.1')}")
    print(f"Dtype: {model_config.get('dtype', 'float16')}")
    print(f"4-bit: {model_config.get('use_4bit', False)}")

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

    # Intervention configuration
    layer_idx = cis_config.get("layer", num_layers // 2)
    token_position = cis_config.get("token_position", -1)

    # Optimization hyperparameters
    max_steps = cis_config.get("max_steps", 200)
    learning_rate = cis_config.get("learning_rate", 0.05)
    reg_weight = cis_config.get("reg_weight", 0.01)
    reg_type = cis_config.get("reg_type", "l2")
    loss_type = cis_config.get("loss_type", "nll")
    tolerance = cis_config.get("tolerance", 1e-4)
    early_stop_margin = cis_config.get("early_stop_margin", 0.5)

    # Build prompt
    prompt = make_factual_prompt(subject, relation)

    print(f"\nExperiment Configuration:")
    print(f"  Prompt: {prompt!r}")
    print(f"  Expected (factual): {expected_completion!r}")
    print(f"  Target (counterfactual): {target_completion!r}")
    print(f"\nIntervention:")
    print(f"  Layer: {layer_idx} (out of {num_layers})")
    print(f"  Token position: {token_position}")
    print(f"\nOptimization:")
    print(f"  Loss type: {loss_type}")
    print(f"  Max steps: {max_steps}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Regularization: {reg_type} (weight={reg_weight})")

    # Get baseline prediction
    print("\n" + "=" * 80)
    print("BASELINE (No Intervention)")
    print("=" * 80)

    with torch.no_grad():
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model(**inputs)
        baseline_logits = outputs.logits[0, -1, :]
        baseline_probs = torch.softmax(baseline_logits, dim=-1)

        # Get baseline top-5
        top5_ids = baseline_logits.topk(5).indices
        print("\nTop-5 predictions (baseline):")
        for rank, token_id in enumerate(top5_ids, 1):
            token_str = tokenizer.decode([token_id])
            prob = baseline_probs[token_id].item()
            marker = " ✓" if token_str.strip() == expected_completion.strip() else ""
            print(f"  {rank}. {token_str!r:<20} prob={prob:.4f}{marker}")

    # Initialize CIS optimizer
    optimizer = CISOptimizer(
        model=model,
        tokenizer=tokenizer,
        layer_idx=layer_idx,
        token_position=token_position,
        device=device,
    )

    # Run optimization
    print("\n" + "=" * 80)
    print("OPTIMIZATION")
    print("=" * 80)

    results = optimizer.optimize(
        prompt=prompt,
        target_completion=target_completion,
        original_completion=expected_completion,
        max_steps=max_steps,
        learning_rate=learning_rate,
        reg_weight=reg_weight,
        reg_type=reg_type,
        loss_type=loss_type,
        tolerance=tolerance,
        early_stop_margin=early_stop_margin,
        verbose=True,
        log_every=10,
    )

    # Print results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    print(f"\nSuccess: {'✓ YES' if results['success'] else '✗ NO'}")
    print(f"Geometric Cost: {results['geometric_cost']:.6f}")
    print(f"Target Probability: {results['target_prob']:.4f}")
    if results['original_prob'] is not None:
        print(f"Original Probability: {results['original_prob']:.4f}")
    print(f"Optimization Steps: {results['num_steps']}")
    print(f"Final Loss: {results['final_loss']:.6f}")

    print("\nDelta Statistics:")
    stats = results['delta_stats']
    print(f"  L2 norm: {stats['norm_l2']:.6f}")
    print(f"  L1 norm: {stats['norm_l1']:.6f}")
    print(f"  L∞ norm: {stats['norm_linf']:.6f}")
    print(f"  Mean: {stats['mean']:.6f}")
    print(f"  Std: {stats['std']:.6f}")

    print("\nTop-5 predictions (with intervention):")
    for rank, pred in enumerate(results['top_predictions'], 1):
        token_str = pred["token_str"]
        prob = pred["prob"]
        marker = ""
        if token_str.strip() == target_completion.strip():
            marker = " ← TARGET ✓"
        elif token_str.strip() == expected_completion.strip():
            marker = " (original)"
        print(f"  {rank}. {token_str!r:<20} prob={prob:.4f}{marker}")

    # Interpretation
    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)

    if results['success']:
        print(f"\n✓ Successfully flipped {expected_completion!r} → {target_completion!r}")
        cost = results['geometric_cost']
        if cost < 0.5:
            rigidity = "LOW (easy to change)"
        elif cost < 2.0:
            rigidity = "MODERATE"
        else:
            rigidity = "HIGH (hard to change)"
        print(f"✓ Geometric cost = {cost:.6f} ({rigidity} factual rigidity)")
        print(f"✓ Converged in {results['num_steps']} steps")
    else:
        print(f"\n✗ Failed to flip to {target_completion!r}")
        print(f"  Target probability: {results['target_prob']:.4f} (< 0.5)")
        print(f"  Geometric cost: {results['geometric_cost']:.6f}")
        print("\nPossible reasons:")
        print("  - Target may be semantically incompatible")
        print("  - Layer chosen may not encode this fact")
        print("  - Learning rate too low or max_steps too small")
        print("  - Regularization weight too high")

    # Save results if requested
    save_path = exp_config.get("save_results")
    if save_path:
        # Convert tensors to lists for JSON serialization
        save_results = {
            "config": {
                "prompt": prompt,
                "target_completion": target_completion,
                "expected_completion": expected_completion,
                "layer_idx": layer_idx,
                "token_position": token_position,
                "max_steps": max_steps,
                "learning_rate": learning_rate,
                "reg_weight": reg_weight,
            },
            "results": {
                "success": results["success"],
                "geometric_cost": results["geometric_cost"],
                "target_prob": results["target_prob"],
                "original_prob": results["original_prob"],
                "num_steps": results["num_steps"],
                "final_loss": results["final_loss"],
                "delta_stats": results["delta_stats"],
                "top_predictions": results["top_predictions"],
            },
            "optimization_history": results["optimization_history"],
        }

        with open(save_path, "w") as f:
            json.dump(save_results, f, indent=2)
        print(f"\n✓ Results saved to: {save_path}")


def run_lambda_sweep(config_path: str) -> None:
    """Run CIS optimization with regularization sweep to find minimal-norm solution.

    This sweeps over different L2 regularization weights to find the minimal
    perturbation that achieves a successful flip.

    Args:
        config_path: Path to experiment YAML config
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
    print("CIS OPTIMIZATION: Regularization Sweep")
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

    # Intervention configuration
    layer_idx = cis_config.get("layer", num_layers // 2)
    token_position = cis_config.get("token_position", -1)

    # Optimization hyperparameters
    max_steps = cis_config.get("max_steps", 200)
    learning_rate = cis_config.get("learning_rate", 0.05)
    margin = cis_config.get("margin", 1.0)

    # Build prompt
    prompt = make_factual_prompt(subject, relation)

    print(f"\nExperiment Configuration:")
    print(f"  Prompt: {prompt!r}")
    print(f"  Expected (factual): {expected_completion!r}")
    print(f"  Target (counterfactual): {target_completion!r}")
    print(f"\nIntervention:")
    print(f"  Layer: {layer_idx} (out of {num_layers})")
    print(f"  Token position: {token_position}")
    print(f"\nOptimization:")
    print(f"  Max steps: {max_steps}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Margin: {margin}")
    print(f"  Loss type: margin_flip")

    # Lambda sweep values
    lambda_values = [1e-4, 1e-3, 1e-2, 1e-1]

    print(f"\n✓ Sweeping over lambda_l2 = {lambda_values}")
    print("\n" + "=" * 80)

    # Store results for each lambda
    sweep_results = []

    for lambda_l2 in lambda_values:
        print(f"\n{'='*80}")
        print(f"LAMBDA = {lambda_l2:.1e}")
        print(f"{'='*80}")

        # Initialize optimizer
        optimizer = CISOptimizer(
            model=model,
            tokenizer=tokenizer,
            layer_idx=layer_idx,
            token_position=token_position,
            device=device,
        )

        # Run optimization with margin_flip_loss
        # We need to import margin_flip_loss
        from src.cis.losses import margin_flip_loss  # noqa: E402

        results = optimizer.optimize(
            prompt=prompt,
            target_completion=target_completion,
            original_completion=expected_completion,
            max_steps=max_steps,
            learning_rate=learning_rate,
            reg_weight=lambda_l2,
            reg_type="l2",
            loss_type="margin",  # Use margin loss
            margin=margin,
            tolerance=1e-6,
            early_stop_margin=0.5,
            verbose=False,  # Suppress per-step logging for sweep
            log_every=10,
        )

        # Record results
        sweep_results.append({
            "lambda": lambda_l2,
            "success": results["success"],
            "steps": results["num_steps"],
            "geometric_cost": results["geometric_cost"],
            "target_prob": results["target_prob"],
            "original_prob": results["original_prob"],
        })

        # Print summary for this lambda
        if results["success"]:
            print(f"✓ SUCCESS: Flipped in {results['num_steps']} steps")
        else:
            print(f"✗ FAILED: Did not flip")
        print(f"  ||δ|| = {results['geometric_cost']:.6f}")
        print(f"  P(target) = {results['target_prob']:.4f}")
        print(f"  P(orig) = {results['original_prob']:.4f}")

    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(f"\n{'Lambda':<12} {'Success':<10} {'Steps':<8} {'||δ||':<12} {'P(target)':<12} {'P(orig)':<12}")
    print("-" * 80)

    for res in sweep_results:
        success_str = "✓ YES" if res["success"] else "✗ NO"
        print(
            f"{res['lambda']:<12.1e} {success_str:<10} {res['steps']:<8} "
            f"{res['geometric_cost']:<12.6f} {res['target_prob']:<12.4f} "
            f"{res['original_prob']:<12.4f}"
        )

    # Find minimal-norm successful solution
    successful = [r for r in sweep_results if r["success"]]
    if successful:
        minimal = min(successful, key=lambda x: x["geometric_cost"])
        print("\n" + "=" * 80)
        print("MINIMAL-NORM SOLUTION")
        print("=" * 80)
        print(f"\n✓ Best lambda: {minimal['lambda']:.1e}")
        print(f"✓ Geometric cost (factual rigidity): {minimal['geometric_cost']:.6f}")
        print(f"✓ Steps to flip: {minimal['steps']}")
        print(f"✓ Final P(target): {minimal['target_prob']:.4f}")
        print(f"✓ Final P(orig): {minimal['original_prob']:.4f}")
    else:
        print("\n✗ No successful flips found. Try:")
        print("  - Increasing max_steps")
        print("  - Decreasing lambda values")
        print("  - Trying a different layer")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run CIS optimization experiment")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment YAML config")
    parser.add_argument("--sweep", action="store_true", help="Run lambda sweep instead of single optimization")
    return parser.parse_args()


def main() -> None:
    """CLI wrapper for CIS optimization."""
    args = parse_args()
    if args.sweep:
        run_lambda_sweep(args.config)
    else:
        run_cis_experiment(args.config)


if __name__ == "__main__":
    main()
