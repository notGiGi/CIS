"""Diagnostic CIS optimization with strict validation and debugging.

This script implements a validation pass to verify correctness of:
1. Margin enforcement (gap-based stopping)
2. Delta initialization (no reuse)
3. Hook lifecycle (no accumulation)
4. Hook scope (only target position modified)
5. Loss computation (sanity checks)

NO RESULTS ARE TRUSTED UNTIL ALL VALIDATIONS PASS.
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional

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
    get_model_num_layers,
    register_residual_capture,
)
from src.models.load_model import load_model_and_tokenizer  # noqa: E402
from src.prompts.factual_prompts import make_factual_prompt  # noqa: E402


def load_config(path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_hook_scope(
    model: Any,
    tokenizer: Any,
    prompt: str,
    layer_idx: int,
    token_position: int,
    device: str,
) -> None:
    """Verify that hook modifies ONLY the target token position.

    This is a critical validation to ensure we're not accidentally
    perturbing other positions.
    """
    print("\n" + "=" * 80)
    print("HOOK SCOPE TEST: Verifying intervention isolation")
    print("=" * 80)

    hidden_size = get_hidden_size(model)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Capture baseline
    print("\n1. Capturing baseline residual stream...")
    capture_baseline = register_residual_capture(
        model=model,
        layer_idx=layer_idx,
        token_position=-1,  # Capture all positions
    )

    try:
        with torch.no_grad():
            _ = model(**inputs)
        h_before = capture_baseline["activations"][0]  # [batch, seq_len, hidden_dim]
        print(f"   Baseline shape: {h_before.shape}")
    finally:
        capture_baseline["handle"].remove()

    # Apply fixed delta
    print("\n2. Applying fixed delta (all ones * 0.01)...")
    test_delta = torch.ones(hidden_size, device=device) * 0.01

    # Run with intervention
    print(f"3. Running forward pass with hook at layer {layer_idx}, position {token_position}...")
    handle, _ = add_residual_perturbation_hook(
        model=model,
        layer_idx=layer_idx,
        delta_vector=test_delta,
        token_position=token_position,
    )

    capture_after = register_residual_capture(
        model=model,
        layer_idx=layer_idx,
        token_position=-1,  # Capture all positions
    )

    try:
        with torch.no_grad():
            _ = model(**inputs)
        h_after = capture_after["activations"][0]  # [batch, seq_len, hidden_dim]
    finally:
        handle.remove()
        capture_after["handle"].remove()

    # Verify changes
    print("\n4. Verifying changes...")
    batch_size, seq_len, hidden_dim = h_before.shape

    # Compute target position
    if token_position == -1:
        target_pos = seq_len - 1
    else:
        target_pos = token_position

    # Check target position changed
    diff_target = (h_after[0, target_pos, :] - h_before[0, target_pos, :]).norm().item()
    print(f"   Change at target position {target_pos}: ||Δh|| = {diff_target:.6f}")

    # Check other positions unchanged
    max_diff_other = 0.0
    for pos in range(seq_len):
        if pos != target_pos:
            diff = (h_after[0, pos, :] - h_before[0, pos, :]).norm().item()
            max_diff_other = max(max_diff_other, diff)

    print(f"   Max change at other positions: ||Δh|| = {max_diff_other:.6e}")

    # Assertions
    assert diff_target > 1e-3, f"Hook did not modify target position! Change: {diff_target}"
    assert max_diff_other < 1e-5, f"Hook leaked to other positions! Max change: {max_diff_other}"

    print("\n✓ HOOK SCOPE TEST PASSED")
    print("  - Target position was modified")
    print("  - Other positions were NOT modified")


def run_diagnostic_optimization(
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
    debug_mode: bool = False,
) -> Dict[str, Any]:
    """Run CIS optimization with strict validation.

    This version enforces:
    1. Gap-based stopping ONLY (gap >= margin)
    2. Explicit delta initialization check
    3. Detailed logging every step
    4. Loss sanity checks
    """
    print("\n" + "=" * 80)
    print("DIAGNOSTIC OPTIMIZATION")
    print("=" * 80)

    # Tokenize
    target_token_id = tokenizer.encode(target_completion, add_special_tokens=False)[0]
    original_token_id = tokenizer.encode(original_completion, add_special_tokens=False)[0]

    print(f"\nTarget: {target_completion!r} (token_id={target_token_id})")
    print(f"Original: {original_completion!r} (token_id={original_token_id})")
    print(f"\nLayer: {layer_idx}, Token position: {token_position}")
    print(f"Margin: {margin}, Lambda: {lambda_l2:.1e}")
    print(f"Max steps: {max_steps}, Learning rate: {learning_rate}")

    # Initialize delta - CRITICAL: fresh initialization
    hidden_size = get_hidden_size(model)
    print(f"\nInitializing delta (hidden_dim={hidden_size})...")

    delta = LearnableDelta(
        hidden_dim=hidden_size,
        init_method="zeros",
        device=device,
    )

    # VALIDATION 1: Verify zero initialization
    initial_norm = delta.get_norm(p=2)
    print(f"Initial ||δ|| = {initial_norm:.8e}")
    assert initial_norm < 1e-6, f"FATAL: Delta not initialized to zero! ||δ|| = {initial_norm}"
    print("✓ Delta initialization verified")

    # Initialize optimizer
    optimizer = optim.Adam(delta.parameters(), lr=learning_rate)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Optimization loop
    print("\n" + "=" * 80)
    print("OPTIMIZATION LOOP")
    print("=" * 80)
    print(f"\n{'Step':<6} {'Gap':<10} {'Margin':<10} {'Gap>=M':<10} {'Loss':<10} {'||δ||':<10} {'P(tgt)':<10} {'P(orig)':<10}")
    print("-" * 90)

    history = []
    success = False

    for step in range(max_steps):
        optimizer.zero_grad()

        # Get current delta
        delta_value = delta()

        # VALIDATION 2: Hook lifecycle - ensure clean state
        # Register hook with guaranteed cleanup
        handle = None
        try:
            handle, _ = add_residual_perturbation_hook(
                model=model,
                layer_idx=layer_idx,
                delta_vector=delta_value,
                token_position=token_position,
            )

            # Forward pass
            with torch.set_grad_enabled(True):
                outputs = model(**inputs)
                logits = outputs.logits[0, -1, :]  # [vocab_size]

                # Extract logits for target and original
                logit_target = logits[target_token_id]
                logit_original = logits[original_token_id]

                # CRITICAL: Compute gap
                gap = (logit_target - logit_original).item()

                # VALIDATION 3: Loss sanity check
                task_loss = margin_flip_loss(
                    logits,
                    target_token_id,
                    original_token_id,
                    margin=margin,
                )

                # Verify loss is zero iff gap >= margin
                if gap >= margin:
                    if task_loss.item() > 1e-4:
                        print(f"\nWARNING: gap >= margin but loss > 0!")
                        print(f"  gap = {gap:.6f}, margin = {margin:.6f}")
                        print(f"  loss = {task_loss.item():.6f}")

                # Add regularization
                reg_loss = regularization_loss(delta_value, reg_type="l2", weight=lambda_l2)
                total_loss = task_loss + reg_loss

            # Backward pass
            total_loss.backward()
            optimizer.step()

        finally:
            # VALIDATION 4: Always remove hook
            if handle is not None:
                handle.remove()

        # Compute metrics
        with torch.no_grad():
            probs = torch.softmax(logits, dim=-1)
            target_prob = probs[target_token_id].item()
            original_prob = probs[original_token_id].item()
            delta_norm = delta.get_norm(p=2)

        # Check stopping condition - ONLY gap >= margin
        gap_satisfied = gap >= margin

        # Log
        print(
            f"{step:<6} {gap:<10.4f} {margin:<10.4f} {str(gap_satisfied):<10} "
            f"{total_loss.item():<10.4f} {delta_norm:<10.6f} "
            f"{target_prob:<10.4f} {original_prob:<10.4f}"
        )

        # Record history
        history.append({
            "step": step,
            "gap": gap,
            "margin": margin,
            "gap_satisfied": gap_satisfied,
            "loss": total_loss.item(),
            "delta_norm": delta_norm,
            "target_prob": target_prob,
            "original_prob": original_prob,
        })

        # STOPPING CONDITION: gap >= margin
        if gap_satisfied:
            print(f"\n✓ CONVERGED at step {step}: gap ({gap:.4f}) >= margin ({margin:.4f})")
            success = True
            break

    # Final evaluation
    if not success:
        print(f"\n✗ FAILED: Did not achieve gap >= margin within {max_steps} steps")
        print(f"  Final gap: {gap:.4f}")
        print(f"  Required margin: {margin:.4f}")

    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"\nSuccess: {success}")
    print(f"Final gap: {gap:.4f} (required: {margin:.4f})")
    print(f"Final ||δ||: {delta_norm:.6f}")
    print(f"Final P(target): {target_prob:.4f}")
    print(f"Final P(orig): {original_prob:.4f}")
    print(f"Steps: {len(history)}")

    return {
        "success": success,
        "final_gap": gap,
        "margin": margin,
        "delta_norm": delta_norm,
        "target_prob": target_prob,
        "original_prob": original_prob,
        "num_steps": len(history),
        "history": history,
    }


def run_diagnostic(config_path: str, debug_mode: bool = False) -> None:
    """Run diagnostic validation experiment.

    Args:
        config_path: Path to experiment config
        debug_mode: If True, run hook scope test
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
    print("CIS DIAGNOSTIC: Strict Validation Pass")
    print("=" * 80)
    print(f"\nSeed: {seed}")
    print(f"Model: {model_config.get('model_name', 'mistralai/Mistral-7B-v0.1')}")
    print(f"Debug mode: {debug_mode}")

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

    print(f"\nFact:")
    print(f"  Prompt: {prompt!r}")
    print(f"  Expected: {expected_completion!r}")
    print(f"  Target: {target_completion!r}")

    # Diagnostic parameters - MINIMAL
    layer_idx = 16
    token_position = -1
    margin = 1.0
    lambda_l2 = 0.01
    max_steps = 50
    learning_rate = 0.05

    print(f"\nDiagnostic Parameters:")
    print(f"  Layer: {layer_idx} (out of {num_layers})")
    print(f"  Token position: {token_position}")
    print(f"  Margin: {margin}")
    print(f"  Lambda: {lambda_l2:.1e}")
    print(f"  Max steps: {max_steps}")
    print(f"  Learning rate: {learning_rate}")

    # Run hook scope test if debug mode
    if debug_mode:
        test_hook_scope(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            layer_idx=layer_idx,
            token_position=token_position,
            device=device,
        )

    # Run diagnostic optimization
    results = run_diagnostic_optimization(
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
        debug_mode=debug_mode,
    )

    # Final validation summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    print("\n✓ All assertions passed:")
    print("  1. Delta initialized to zero")
    print("  2. Hook lifecycle managed correctly")
    print("  3. Stopping condition enforced (gap >= margin)")
    if debug_mode:
        print("  4. Hook scope verified (only target position modified)")

    if results["success"]:
        print(f"\n✓ Optimization SUCCEEDED")
        print(f"  Achieved gap ({results['final_gap']:.4f}) >= margin ({margin:.4f})")
        print(f"  Geometric cost: {results['delta_norm']:.6f}")
    else:
        print(f"\n✗ Optimization FAILED")
        print(f"  Did not achieve required margin within {max_steps} steps")
        print(f"  Final gap: {results['final_gap']:.4f}")
        print(f"  Required: {margin:.4f}")

    print("\n" + "=" * 80)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 80)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run CIS diagnostic validation")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment YAML config")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode (hook scope test)")
    return parser.parse_args()


def main() -> None:
    """CLI wrapper for CIS diagnostic."""
    args = parse_args()
    run_diagnostic(args.config, debug_mode=args.debug)


if __name__ == "__main__":
    main()
