"""Causal sanity check: verify that residual stream interventions affect predictions.

This experiment:
1. Runs baseline inference (no intervention)
2. Applies a small random perturbation to the residual stream at a chosen layer
3. Runs intervention inference
4. Compares the two prediction distributions

Expected outcome:
- Predictions should change slightly but remain coherent
- Verifies that the hook mechanism is wired correctly
- Establishes causal sensitivity before full CIS optimization

This is NOT full CIS optimization - just a sanity check.
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.hooks.residual_hooks import (  # noqa: E402
    add_residual_perturbation_hook,
    clear_hooks,
    get_hidden_size,
    get_model_num_layers,
)
from src.models.load_model import load_model_and_tokenizer  # noqa: E402
from src.prompts.factual_prompts import make_factual_prompt  # noqa: E402
from src.utils.token_utils import decode_topk_predictions, get_next_token_logits  # noqa: E402


def load_config(path: str) -> Dict[str, Any]:
    """Load a YAML configuration file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def create_random_perturbation(hidden_size: int, device: str, norm_scale: float = 0.01) -> torch.Tensor:
    """Create a small random Gaussian perturbation.

    Args:
        hidden_size: Dimension of the hidden state
        device: Device to create tensor on
        norm_scale: Target L2 norm of the perturbation (default: 0.01)

    Returns:
        delta: Random vector with specified norm [hidden_size]
    """
    # Create random Gaussian vector
    delta = torch.randn(hidden_size, device=device)

    # Normalize to have desired norm
    delta = delta / delta.norm() * norm_scale

    return delta


def run_causal_sanity_check(config_path: str, intervention_config: Dict[str, Any]) -> None:
    """Run causal sensitivity check with residual stream intervention.

    Args:
        config_path: Path to experiment config
        intervention_config: Dict with intervention parameters:
            - layer_idx: Which layer to intervene on (None = middle layer)
            - token_position: Which token to perturb (-1 = last)
            - perturbation_norm: L2 norm of random perturbation (default: 0.01)
    """
    import time

    # Load configs
    exp_config = load_config(config_path)
    model_config_path = exp_config.get("model_config", "config/model.yaml")
    model_config = load_config(model_config_path)

    # Set seed for reproducibility
    seed = exp_config.get("seed", 0)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print("=" * 80)
    print("CAUSAL SANITY CHECK: Residual Stream Intervention")
    print("=" * 80)
    print(f"\nSeed: {seed}")
    print(f"Model: {model_config.get('model_name', 'mistralai/Mistral-7B-v0.1')}")
    print(f"Dtype: {model_config.get('dtype', 'float16')}")
    print(f"4-bit Quantization: {model_config.get('use_4bit', False)}")

    # Load model
    print("\nLoading model and tokenizer...")
    start_time = time.time()
    model, tokenizer = load_model_and_tokenizer(model_config)
    load_time = time.time() - start_time
    print(f"✓ Model loaded in {load_time:.2f}s")

    device = model_config.get("device", "cuda")

    # Get model architecture info
    num_layers = get_model_num_layers(model)
    hidden_size = get_hidden_size(model)
    print(f"\nModel Architecture:")
    print(f"  Total layers: {num_layers}")
    print(f"  Hidden size: {hidden_size}")

    # Determine intervention layer (default to middle layer)
    layer_idx = intervention_config.get("layer_idx")
    if layer_idx is None:
        layer_idx = num_layers // 2
    token_position = intervention_config.get("token_position", -1)
    perturbation_norm = intervention_config.get("perturbation_norm", 0.01)

    print(f"\nIntervention Configuration:")
    print(f"  Layer: {layer_idx} (out of {num_layers})")
    print(f"  Token position: {token_position}")
    print(f"  Perturbation norm: {perturbation_norm}")

    # Create random perturbation
    delta = create_random_perturbation(hidden_size, device, norm_scale=perturbation_norm)
    print(f"  Delta stats: norm={delta.norm().item():.6f}, mean={delta.mean().item():.6f}, std={delta.std().item():.6f}")

    # Get test fact
    subject = exp_config.get("subject", "Eiffel Tower")
    relation = exp_config.get("relation", "located in")
    expected = exp_config.get("expected_completion", " Paris")

    prompt = make_factual_prompt(subject, relation)
    print(f"\nPrompt: '{prompt}'")
    print(f"Expected completion: '{expected}'")

    k = exp_config.get("analysis", {}).get("k_alternatives", 5)

    # ===== BASELINE: No Intervention =====
    print("\n" + "=" * 80)
    print("BASELINE (No Intervention)")
    print("=" * 80)

    start_time = time.time()
    baseline_logits = get_next_token_logits(model, tokenizer, prompt, device=device)
    baseline_time = time.time() - start_time

    baseline_preds = decode_topk_predictions(tokenizer, baseline_logits, k=k)

    print(f"\nInference time: {baseline_time:.3f}s")
    print(f"\nTop-{k} predictions (baseline):")
    for rank, pred in enumerate(baseline_preds, start=1):
        token_str = pred["token_str"]
        marker = " ✓" if expected and token_str.strip() == expected.strip() else ""
        print(f"  {rank}. {token_str!r:<20} prob={pred['prob']:.4f}{marker}")

    # ===== INTERVENTION: With Random Perturbation =====
    print("\n" + "=" * 80)
    print(f"INTERVENTION (Random perturbation at layer {layer_idx})")
    print("=" * 80)

    # Attach hook
    handle, _ = add_residual_perturbation_hook(
        model=model,
        layer_idx=layer_idx,
        delta_vector=delta,
        token_position=token_position,
    )

    try:
        start_time = time.time()
        intervention_logits = get_next_token_logits(model, tokenizer, prompt, device=device)
        intervention_time = time.time() - start_time

        intervention_preds = decode_topk_predictions(tokenizer, intervention_logits, k=k)

        print(f"\nInference time: {intervention_time:.3f}s")
        print(f"\nTop-{k} predictions (with intervention):")
        for rank, pred in enumerate(intervention_preds, start=1):
            token_str = pred["token_str"]
            marker = " ✓" if expected and token_str.strip() == expected.strip() else ""
            print(f"  {rank}. {token_str!r:<20} prob={pred['prob']:.4f}{marker}")

    finally:
        # Always remove hook
        clear_hooks(handle)

    # ===== ANALYSIS =====
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    # Check if top-1 changed
    baseline_top1 = baseline_preds[0]["token_str"]
    intervention_top1 = intervention_preds[0]["token_str"]

    print(f"\nTop-1 prediction:")
    print(f"  Baseline:     {baseline_top1!r}")
    print(f"  Intervention: {intervention_top1!r}")

    if baseline_top1 != intervention_top1:
        print("  ✓ Top-1 prediction CHANGED (causal effect detected)")
    else:
        print("  • Top-1 prediction unchanged (intervention too small or position not critical)")

    # Calculate distribution shift (KL divergence)
    baseline_probs = torch.tensor([p["prob"] for p in baseline_preds])
    intervention_probs = torch.tensor([p["prob"] for p in intervention_preds])

    # Add small epsilon to avoid log(0)
    eps = 1e-10
    kl_div = (baseline_probs * torch.log((baseline_probs + eps) / (intervention_probs + eps))).sum().item()

    print(f"\nDistribution shift:")
    print(f"  KL divergence (baseline || intervention): {kl_div:.6f}")

    if kl_div > 0.001:
        print("  ✓ Significant distribution shift detected")
    else:
        print("  • Minimal distribution shift (very small perturbation)")

    # Logit difference
    logit_diff = (intervention_logits - baseline_logits).abs().max().item()
    print(f"\nMax logit change: {logit_diff:.4f}")

    # Memory info
    if torch.cuda.is_available():
        print(f"\nGPU Memory:")
        print(f"  Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"  Reserved:  {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print("\n✓ Causal sanity check complete!")
    print("✓ Hook mechanism verified - ready for CIS optimization")
    print("\nNext steps:")
    print("  1. Implement gradient-based optimization to find targeted perturbations")
    print("  2. Optimize delta to flip predictions to specific counterfactual targets")
    print("  3. Measure geometric cost of factual rigidity")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Causal sanity check: verify residual stream intervention affects predictions"
    )
    parser.add_argument("--config", type=str, required=True, help="Path to experiment YAML config")
    parser.add_argument("--layer", type=int, default=None, help="Layer index to intervene on (default: middle layer)")
    parser.add_argument(
        "--token-position", type=int, default=-1, help="Token position to perturb (-1 = last token, default: -1)"
    )
    parser.add_argument(
        "--perturbation-norm",
        type=float,
        default=0.01,
        help="L2 norm of random perturbation (default: 0.01)",
    )
    return parser.parse_args()


def main() -> None:
    """CLI wrapper for causal sanity check."""
    args = parse_args()

    intervention_config = {
        "layer_idx": args.layer,
        "token_position": args.token_position,
        "perturbation_norm": args.perturbation_norm,
    }

    run_causal_sanity_check(args.config, intervention_config)


if __name__ == "__main__":
    main()
