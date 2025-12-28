"""Factual baseline: next-token recall for a single fact with a frozen Mistral-7B."""

import argparse
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    # Ensure local src/ modules are importable when running as a script.
    sys.path.append(str(ROOT))

from src.models.load_model import load_model_and_tokenizer  # noqa: E402
from src.prompts.factual_prompts import make_factual_prompt  # noqa: E402
from src.utils.token_utils import get_next_token_logits, print_topk_predictions  # noqa: E402


def load_model_config(path: Path) -> dict:
    """Load model configuration from YAML."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_factual_baseline(model_config_path: Path) -> None:
    """Run a single forward pass to assess factual recall without interventions."""
    model_config = load_model_config(model_config_path)

    # Scientific intent: use a fixed subject-relation pair to anchor later CIS comparisons.
    subject = "Eiffel Tower"
    relation = "located in"
    expected = "Paris"
    prompt = make_factual_prompt(subject, relation)

    # Fix RNG to keep any stochastic components reproducible across runs.
    torch.manual_seed(0)

    print(f"Loading model from config: {model_config_path}")
    model, tokenizer = load_model_and_tokenizer(model_config)
    device = next(model.parameters()).device
    print(f"Model loaded on device: {device}")
    print(f"Prompt: {prompt}")

    # Forward pass with frozen weights to measure the native factual distribution.
    logits = get_next_token_logits(model, tokenizer, prompt, device=device)

    print("\nTop-5 next-token predictions (no intervention):")
    print_topk_predictions(tokenizer, logits, k=5, expected=expected)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Factual recall baseline (no CIS)")
    parser.add_argument(
        "--model-config",
        type=Path,
        default=Path("config/model.yaml"),
        help="Path to model config YAML.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_factual_baseline(args.model_config)


if __name__ == "__main__":
    main()
