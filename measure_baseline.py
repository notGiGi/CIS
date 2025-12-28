"""Quick script to measure baseline residual norm."""
import torch
import yaml
import sys
from src.models.load_model import load_model_and_tokenizer
from src.prompts.factual_prompts import make_factual_prompt
from src.cis.metrics import measure_residual_norm

# Load config
with open("config/model.yaml") as f:
    model_config = yaml.safe_load(f)

# Load model with error handling
print("Loading model...")
try:
    # Set torch to use less memory
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    model, tokenizer = load_model_and_tokenizer(model_config)
    device = model_config.get("device", "cuda")

    # Check GPU memory
    if torch.cuda.is_available():
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"GPU Memory Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
except Exception as e:
    print(f"ERROR: Failed to load model: {e}")
    print("\nThis might be due to insufficient GPU memory (RTX 3050 has 4GB).")
    print("The diagnostic script may have been run on a different machine or with different settings.")
    sys.exit(1)

# Measure baseline
prompt = make_factual_prompt("Eiffel Tower", "located in")
metrics = measure_residual_norm(
    model=model,
    tokenizer=tokenizer,
    prompt=prompt,
    layer_idx=16,
    token_position=-1,
    device=device,
)

print(f"\nBaseline Residual Metrics at layer 16:")
print(f"  ||h||_2 = {metrics['residual_norm']:.4f}")
print(f"  mean = {metrics['residual_mean']:.4f}")
print(f"  std = {metrics['residual_std']:.4f}")
print(f"  hidden_dim = {metrics['hidden_dim']}")

print(f"\nDiagnostic result: ||δ|| = 6.864809")
print(f"Normalized cost = ||δ|| / ||h|| = {6.864809 / metrics['residual_norm']:.4f}")
print(f"Per-dim RMS = ||δ|| / sqrt(d) = {6.864809 / (metrics['hidden_dim']**0.5):.6f}")
