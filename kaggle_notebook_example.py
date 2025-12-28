"""
Kaggle Notebook Example for CIS Factual LLM
============================================

Copy and paste this code into Kaggle notebook cells.
Make sure to enable GPU (T4 or P100) in Settings.
"""

# ===== CELL 1: Setup Repository =====
# Clone the repository and install dependencies
import subprocess
import sys

print("Cloning repository...")
subprocess.run(["git", "clone", "https://github.com/notGiGi/CIS.git"], check=True)
sys.path.insert(0, "/kaggle/working/CIS")

print("\nInstalling dependencies...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "/kaggle/working/CIS/requirements.txt"], check=True)

print("\n✓ Setup complete!")


# ===== CELL 2: Configure for Kaggle =====
# Update config to allow downloading from HuggingFace
import yaml
import os

os.chdir("/kaggle/working/CIS")

# Read and modify model config
with open("config/model.yaml", "r") as f:
    config = yaml.safe_load(f)

# Enable HuggingFace downloads (Kaggle has good HF cache)
config["local_files_only"] = False
config["cache_dir"] = "/kaggle/working/model_cache"

# Save updated config
with open("config/model.yaml", "w") as f:
    yaml.dump(config, f, default_flow_style=False)

print("✓ Configuration updated for Kaggle")
print(f"  - Model: {config['model_name']}")
print(f"  - Dtype: {config['dtype']}")
print(f"  - 4-bit: {config['use_4bit']}")
print(f"  - Flash Attention: {config['use_flash_attention']}")


# ===== CELL 3: Run Experiment =====
# Execute the factual recall experiment
import subprocess

print("Running factual recall experiment...\n")
print("=" * 60)

result = subprocess.run(
    [sys.executable, "src/experiments/run_single_fact.py", "--config", "config/experiment.yaml"],
    capture_output=False,
    text=True
)

if result.returncode == 0:
    print("\n" + "=" * 60)
    print("✓ Experiment completed successfully!")
else:
    print("\n" + "=" * 60)
    print("✗ Experiment failed. Check errors above.")


# ===== CELL 4 (Optional): Custom Experiment =====
# Run your own custom factual queries
import yaml
import torch
from src.models.load_model import load_model_and_tokenizer
from src.utils.token_utils import get_next_token_logits, decode_topk_predictions
from src.prompts.factual_prompts import make_factual_prompt

# Load model config
with open("config/model.yaml", "r") as f:
    model_config = yaml.safe_load(f)

# Load model (this will reuse cached model if already loaded)
print("Loading model...")
model, tokenizer = load_model_and_tokenizer(model_config)
print("✓ Model loaded\n")

# Define your own facts to test
test_facts = [
    ("Eiffel Tower", "located in", "Paris"),
    ("Mount Everest", "located in", "Nepal"),
    ("iPhone", "made by", "Apple"),
    ("Python", "created by", "Guido van Rossum"),
]

# Test each fact
for subject, relation, expected in test_facts:
    prompt = make_factual_prompt(subject, relation)
    logits = get_next_token_logits(model, tokenizer, prompt, device="cuda")
    predictions = decode_topk_predictions(tokenizer, logits, k=5)

    print(f"\n{'=' * 60}")
    print(f"Prompt: {prompt}")
    print(f"Expected: {expected}")
    print(f"\nTop-5 predictions:")
    for rank, pred in enumerate(predictions, 1):
        marker = " ✓" if expected.lower() in pred["token_str"].lower() else ""
        print(f"  {rank}. {pred['token_str']!r:<20} prob={pred['prob']:.4f}{marker}")

print("\n" + "=" * 60)
print("✓ All tests completed!")


# ===== CELL 5 (Optional): Memory Info =====
# Check GPU memory usage
import torch

if torch.cuda.is_available():
    print("GPU Memory Usage:")
    print(f"  Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"  Reserved:  {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    print(f"  Max:       {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
else:
    print("No GPU available")
