"""Entry point for running a single-fact factual recall sanity check (no CIS yet)."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    # Ensure local src/ modules are importable when running as a script.
    sys.path.append(str(ROOT))

from src.models.load_model import load_model_and_tokenizer  # noqa: E402
from src.prompts.factual_prompts import make_factual_prompt  # noqa: E402
from src.utils.token_utils import decode_topk_predictions, get_next_token_logits  # noqa: E402


def load_config(path: str) -> Dict[str, Any]:
    """Load a YAML configuration file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_fact_instance(config: Dict[str, Any]) -> Tuple[str, str, str]:
    """Return subject, relation text, and expected completion for the sanity check."""
    data_path = config.get("data_path")
    if data_path:
        with open(data_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        example = dataset[0]  # Single deterministic fact keeps the probe comparable over time.
        subject = example["subject"]
        expected_completion = example.get("gold_completion", "").strip("\n")
        # Prefer an explicit prompt string if provided; otherwise build from subject/relation.
        prompt_text = example.get("prompt")
        if prompt_text:
            relation_text = prompt_text.replace(f"The {subject} is ", "").strip()
            # Keep the trailing space for next-token prediction.
            relation_text = relation_text if relation_text.endswith(" ") else relation_text
        else:
            relation_text = example.get("relation", "located in")
    else:
        subject = config.get("subject", "Eiffel Tower")
        relation_text = config.get("relation", "located in")
        expected_completion = config.get("expected_completion", " Paris")
    return subject, relation_text, expected_completion


def run_experiment(config_path: str) -> None:
    """Load a frozen Mistral-7B and report its top next-token predictions for one fact."""
    import time

    exp_config = load_config(config_path)
    model_config_path = exp_config.get("model_config", "config/model.yaml")
    model_config = load_config(model_config_path)

    # Fixed seed keeps sampling-free probes numerically stable across reruns.
    seed = exp_config.get("seed", 0)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print(f"=== Factual Recall Experiment ===")
    print(f"Seed: {seed}")
    print(f"Model: {model_config.get('model_name', 'mistralai/Mistral-7B-v0.1')}")
    print(f"Device: {model_config.get('device', 'cuda')}")
    print(f"Dtype: {model_config.get('dtype', 'float16')}")
    print(f"4-bit Quantization: {model_config.get('use_4bit', False)}")
    print()

    subject, relation_text, expected_completion = load_fact_instance(exp_config)
    prompt = make_factual_prompt(subject, relation_text)

    # Loading the model/tokenizer is separated to keep inference identical across experiments.
    print("Loading model and tokenizer...")
    start_time = time.time()
    model, tokenizer = load_model_and_tokenizer(model_config)
    load_time = time.time() - start_time
    print(f"Model loaded in {load_time:.2f}s")
    print()

    device = model_config.get("device", "cuda")

    # Forward pass without any intervention to establish the factual baseline.
    print("Running inference...")
    start_time = time.time()
    logits = get_next_token_logits(model, tokenizer, prompt, device=device)
    inference_time = time.time() - start_time
    print(f"Inference completed in {inference_time:.3f}s")
    print()

    k = exp_config.get("analysis", {}).get("k_alternatives", 5)
    predictions = decode_topk_predictions(tokenizer, logits, k=k)

    print(f"Prompt: {prompt}")
    print(f"\nTop-{k} next-token predictions (no intervention):")
    for rank, pred in enumerate(predictions, start=1):
        token_str = pred["token_str"]
        marker = " <-- expected factual object" if expected_completion and token_str.strip() == expected_completion.strip() else ""
        print(f"{rank}. {token_str!r} \tprob={pred['prob']:.4f}{marker}")

    # Memory cleanup
    if torch.cuda.is_available():
        print(f"\nGPU Memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"GPU Memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for selecting configuration files."""
    parser = argparse.ArgumentParser(description="Run a factual recall sanity check (no CIS)")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment YAML config")
    return parser.parse_args()


def main() -> None:
    """CLI wrapper for executing a single factual recall experiment."""
    args = parse_args()
    run_experiment(args.config)


if __name__ == "__main__":
    main()
