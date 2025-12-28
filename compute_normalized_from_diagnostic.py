"""Compute normalized metrics from diagnostic output without loading model.

Usage:
    python compute_normalized_from_diagnostic.py --delta-norm 6.864809 --residual-norm <value>

If you don't know the residual norm, you can estimate it by running:
    python -c "import torch; from src.cis.metrics import measure_residual_norm; ..."

Or use typical values for Mistral-7B at layer 16 (usually ||h|| ~ 20-40)
"""
import argparse
import math


def compute_normalized_metrics(delta_norm: float, residual_norm: float, hidden_dim: int = 4096):
    """Compute all normalized metrics.

    Args:
        delta_norm: ||δ||₂ from optimization
        residual_norm: ||h||₂ baseline residual stream norm
        hidden_dim: Model hidden dimension (4096 for Mistral-7B)

    Returns:
        Dictionary with all metrics
    """
    normalized_cost_ratio = delta_norm / residual_norm
    per_dim_rms = delta_norm / math.sqrt(hidden_dim)
    relative_perturbation_pct = normalized_cost_ratio * 100.0

    return {
        "delta_norm": delta_norm,
        "residual_norm": residual_norm,
        "hidden_dim": hidden_dim,
        "normalized_cost_ratio": normalized_cost_ratio,
        "per_dim_rms": per_dim_rms,
        "relative_perturbation_pct": relative_perturbation_pct,
    }


def interpret_normalized_cost(normalized_cost_ratio: float) -> str:
    """Interpret the normalized cost ratio."""
    if normalized_cost_ratio < 0.05:
        return "Very high rigidity - fact strongly encoded"
    elif normalized_cost_ratio < 0.15:
        return "Moderate rigidity"
    elif normalized_cost_ratio < 0.30:
        return "Low rigidity"
    else:
        return "Very low rigidity - fact weakly encoded"


def main():
    parser = argparse.ArgumentParser(description="Compute normalized metrics from diagnostic output")
    parser.add_argument("--delta-norm", type=float, required=True, help="||δ||₂ from optimization output")
    parser.add_argument("--residual-norm", type=float, help="||h||₂ baseline residual norm (if known)")
    parser.add_argument("--estimate", action="store_true", help="Use estimated residual norm for Mistral-7B layer 16")
    parser.add_argument("--hidden-dim", type=int, default=4096, help="Hidden dimension (default: 4096 for Mistral-7B)")

    args = parser.parse_args()

    # Handle residual norm
    if args.residual_norm is not None:
        residual_norm = args.residual_norm
        print(f"Using provided residual norm: ||h||_2 = {residual_norm:.4f}")
    elif args.estimate:
        # Typical range for Mistral-7B at middle layers
        residual_norm_low = 20.0
        residual_norm_high = 40.0
        residual_norm_mid = 30.0

        print(f"\nUsing ESTIMATED residual norm for Mistral-7B layer 16:")
        print(f"  Low estimate: ||h||_2 = {residual_norm_low:.4f}")
        print(f"  Mid estimate: ||h||_2 = {residual_norm_mid:.4f}")
        print(f"  High estimate: ||h||_2 = {residual_norm_high:.4f}")
        print(f"\nNOTE: These are estimates. Run measure_baseline.py for exact value.\n")

        # Show range
        print("=" * 80)
        print("NORMALIZED METRICS (Range)")
        print("=" * 80)

        for label, res_norm in [("Low", residual_norm_low), ("Mid", residual_norm_mid), ("High", residual_norm_high)]:
            metrics = compute_normalized_metrics(args.delta_norm, res_norm, args.hidden_dim)
            interpretation = interpret_normalized_cost(metrics["normalized_cost_ratio"])

            print(f"\n{label} Estimate (||h||_2 = {res_norm:.1f}):")
            print(f"  Normalized cost ratio: {metrics['normalized_cost_ratio']:.4f} ({metrics['relative_perturbation_pct']:.2f}%)")
            print(f"  Per-dimension RMS: {metrics['per_dim_rms']:.6f}")
            print(f"  Interpretation: {interpretation}")

        return
    else:
        print("ERROR: Must provide either --residual-norm or --estimate")
        return

    # Compute metrics
    metrics = compute_normalized_metrics(args.delta_norm, residual_norm, args.hidden_dim)
    interpretation = interpret_normalized_cost(metrics["normalized_cost_ratio"])

    # Print results
    print("\n" + "=" * 80)
    print("NORMALIZED METRICS")
    print("=" * 80)
    print(f"\nInput:")
    print(f"  ||delta||_2 = {metrics['delta_norm']:.6f}")
    print(f"  ||h||_2 = {metrics['residual_norm']:.6f}")
    print(f"  hidden_dim = {metrics['hidden_dim']}")

    print(f"\nNormalized Metrics:")
    print(f"  Normalized cost ratio: {metrics['normalized_cost_ratio']:.4f}")
    print(f"  Relative perturbation: {metrics['relative_perturbation_pct']:.2f}%")
    print(f"  Per-dimension RMS: {metrics['per_dim_rms']:.6f}")

    print(f"\nInterpretation:")
    print(f"  {interpretation}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
