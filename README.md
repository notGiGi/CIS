# cis_factual_llm

Counterfactual Internal States (CIS) study the minimal activation perturbations that flip a model's factual prediction while keeping the underlying network weights frozen. This repository provides a research scaffold for probing factual knowledge representations in large language models using activation hooks rather than fine-tuning.

## What is a Counterfactual Internal State?
A Counterfactual Internal State is an activation pattern injected into a chosen layer and token position that steers the model from a ground-truth factual completion to a specified counterfactual alternative. The intervention is localized (layer, token position) and optimized to be as small as possible under an L2 metric.

## Geometric cost
The geometric cost of a CIS is defined as the minimal L2 norm of an additive activation perturbation that changes the model's predicted token from the factual answer to the counterfactual target. This scalar quantifies how resistant the model's internal representation is to counterfactual steering; lower cost indicates a more pliable factual representation.

## Factual pilot experiment
The pilot experiment targets a single fact such as "The Eiffel Tower is located in" using a frozen LLaMA-2-7B. We inject an activation vector into a mid-layer residual stream at the subject token position and optimize this vector to increase the logit of a counterfactual completion (e.g., " Rome") relative to the factual completion (e.g., " Paris"). Primary readouts include success rate of flipping the top-1 prediction, geometric cost of the perturbation, and collateral effects on nearby tokens.

## Repository layout
- `config/`: YAML configs for model loading and experiment hyperparameters.
- `data/`: Small CounterFact-style subset for quick smoke tests.
- `src/`: Modular code for loading models, constructing prompts, attaching hooks, searching for CIS, computing metrics, and orchestrating experiments.
- `notebooks/`: Lightweight notebooks for exploratory analyses and visualization.

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/notGiGi/CIS.git
cd CIS

# Install dependencies
pip install -r requirements.txt
```

### Running experiments

#### 1. Factual recall baseline (no intervention)

```bash
python src/experiments/run_single_fact.py --config config/experiment.yaml
```

This will:
1. Load Mistral-7B-v0.1 in FP16 precision
2. Run factual recall on a sample from CounterFact dataset
3. Display top-5 next-token predictions with probabilities
4. Show memory usage and timing information

#### 2. Causal sanity check (residual stream intervention)

```bash
python src/experiments/run_causal_check.py --config config/experiment.yaml
```

This experiment:
1. Runs baseline inference (no intervention)
2. Applies a small random perturbation to the residual stream at a chosen layer
3. Compares prediction distributions to verify causal sensitivity
4. Validates that the hook mechanism works correctly

Optional arguments:
- `--layer N`: Specify which transformer layer to intervene on (default: middle layer)
- `--token-position N`: Which token to perturb (-1 = last token)
- `--perturbation-norm X`: L2 norm of random perturbation (default: 0.01)

### Configuration

The model can be configured in `config/model.yaml`:

```yaml
model_name: mistralai/Mistral-7B-v0.1
dtype: float16              # Full precision (FP16)
use_4bit: false             # Set to true for 4-bit quantization
use_flash_attention: true   # Enable Flash Attention 2 for speed
device_map: auto            # Automatic GPU distribution
```

**For limited memory environments**, enable 4-bit quantization:

```yaml
use_4bit: true
bnb_4bit_quant_type: nf4
bnb_4bit_compute_dtype: float16
```

### Running on Kaggle

See [KAGGLE_SETUP.md](KAGGLE_SETUP.md) for detailed instructions on running experiments on Kaggle with GPU support.

## Features

- **Flexible model loading**: Supports both FP16 and 4-bit quantization
- **Flash Attention 2**: Faster inference when available
- **Residual stream hooks**: Causal intervention on internal activations
- **Activation capture**: Record hidden states for analysis
- **Detailed logging**: Track model loading, inference time, and memory usage
- **Memory optimized**: Configurable memory limits and automatic device mapping
- **Kaggle-ready**: Easy setup for cloud GPU environments

## Architecture

### Hook System

The residual stream hook system (`src/hooks/residual_hooks.py`) enables causal interventions:

```python
from src.hooks.residual_hooks import add_residual_perturbation_hook, get_hidden_size

# Create a small perturbation
hidden_size = get_hidden_size(model)
delta = torch.randn(hidden_size) * 0.01

# Apply intervention at layer 16, last token position
handle, _ = add_residual_perturbation_hook(
    model=model,
    layer_idx=16,
    delta_vector=delta,
    token_position=-1
)

# Run inference with intervention
logits = model(input_ids)

# Remove hook
handle.remove()
```

Key functions:
- `add_residual_perturbation_hook()`: Inject perturbations into residual stream
- `register_residual_capture()`: Capture activations for analysis
- `get_model_num_layers()`: Get total transformer layers
- `get_hidden_size()`: Get hidden dimension size
- `clear_hooks()`: Clean up all hooks

## Notes
- All experiments assume access to a GPU and do not perform any parameter updates.
- Activation interventions are implemented via hooks; no training or fine-tuning routines are included.
- Model weights are frozen and kept in eval mode throughout all experiments.
