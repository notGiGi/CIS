# Causal Sanity Check Guide

This guide explains how to use the causal intervention system to verify that factual predictions are causally sensitive to internal activations.

## What is a Causal Sanity Check?

Before implementing full CIS (Counterfactual Internal State) optimization, we need to verify that:

1. **The hook mechanism works correctly** - perturbations are applied to the right layer and token
2. **Predictions are causally sensitive** - small changes to internal activations affect output predictions
3. **The model remains coherent** - interventions don't break the model's basic functionality

This is NOT yet full CIS optimization - we're just using **random perturbations** to validate the infrastructure.

## Running the Causal Check

### Basic Usage

```bash
python src/experiments/run_causal_check.py --config config/experiment.yaml
```

This will:
1. Load the model (Mistral-7B-v0.1)
2. Run baseline inference (no intervention)
3. Apply a small random perturbation at the middle layer
4. Run intervention inference
5. Compare the two prediction distributions

### Advanced Options

#### Specify which layer to intervene on

```bash
# Intervene on layer 10 (out of 32 for Mistral-7B)
python src/experiments/run_causal_check.py --config config/experiment.yaml --layer 10
```

**Layer selection tips:**
- Early layers (0-8): Affect low-level features, less impact on factual knowledge
- Middle layers (12-20): **Most effective** for factual interventions
- Late layers (24-31): Close to output, may be too late for some interventions

#### Adjust perturbation strength

```bash
# Very small perturbation (may not change predictions)
python src/experiments/run_causal_check.py --config config/experiment.yaml --perturbation-norm 0.001

# Larger perturbation (more likely to change top-1)
python src/experiments/run_causal_check.py --config config/experiment.yaml --perturbation-norm 0.1
```

**Perturbation norms:**
- `0.001`: Very subtle, minimal effect
- `0.01`: **Default** - small but measurable effect
- `0.1`: Large, likely to change predictions significantly
- `1.0`: Very large, may produce incoherent outputs

#### Specify token position

```bash
# Intervene on the last token (default, usually the subject)
python src/experiments/run_causal_check.py --config config/experiment.yaml --token-position -1

# Intervene on a specific token position (0-indexed)
python src/experiments/run_causal_check.py --config config/experiment.yaml --token-position 3
```

## Understanding the Output

### Example Output

```
================================================================================
CAUSAL SANITY CHECK: Residual Stream Intervention
================================================================================

Seed: 0
Model: mistralai/Mistral-7B-v0.1
Dtype: float16
4-bit Quantization: False

Model Architecture:
  Total layers: 32
  Hidden size: 4096

Intervention Configuration:
  Layer: 16 (out of 32)
  Token position: -1
  Perturbation norm: 0.01000
  Delta stats: norm=0.010000, mean=0.000015, std=0.001564

Prompt: 'The Eiffel Tower is located in'
Expected completion: ' Paris'

================================================================================
BASELINE (No Intervention)
================================================================================

Top-5 predictions (baseline):
  1. ' Paris'              prob=0.8234 ✓
  2. ' France'             prob=0.1123
  3. ' the'                prob=0.0234
  4. ' central'            prob=0.0156
  5. ' downtown'           prob=0.0089

================================================================================
INTERVENTION (Random perturbation at layer 16)
================================================================================

Top-5 predictions (with intervention):
  1. ' France'             prob=0.4521
  2. ' Paris'              prob=0.3891 ✓
  3. ' the'                prob=0.0812
  4. ' central'            prob=0.0345
  5. ' Europe'             prob=0.0198

================================================================================
ANALYSIS
================================================================================

Top-1 prediction:
  Baseline:     ' Paris'
  Intervention: ' France'
  ✓ Top-1 prediction CHANGED (causal effect detected)

Distribution shift:
  KL divergence (baseline || intervention): 0.234516
  ✓ Significant distribution shift detected

Max logit change: 2.3456

================================================================================
CONCLUSION
================================================================================

✓ Causal sanity check complete!
✓ Hook mechanism verified - ready for CIS optimization
```

### Interpreting Results

#### ✓ Good Result (Hook Working)

If you see:
- **Top-1 prediction changed** OR significant KL divergence (> 0.001)
- Predictions remain coherent (real words, not gibberish)
- Max logit change is non-zero

→ **The hook system is working!** Proceed to CIS optimization.

#### ⚠ Weak Effect (Perturbation Too Small)

If you see:
- Top-1 unchanged AND KL divergence < 0.001
- Very small max logit change (< 0.01)

→ Try increasing `--perturbation-norm` to 0.1 or 1.0

#### ❌ Broken Output

If you see:
- Incoherent tokens or gibberish
- Model crashes or CUDA errors

→ Check that:
1. You're using the correct layer index (0 to num_layers-1)
2. The model loaded correctly
3. You have enough GPU memory

## What This Validates

### ✓ Hook Mechanism

- Perturbations are successfully injected into the residual stream
- Hooks can be attached and removed cleanly
- No memory leaks or graph corruption

### ✓ Causal Sensitivity

- Internal activations causally affect next-token predictions
- The residual stream at middle layers encodes factual knowledge
- Small changes to activations produce measurable output changes

### ✓ Model Coherence

- The model still produces valid tokens after intervention
- Prediction distributions remain well-formed
- The intervention doesn't break the forward pass

## Next Steps

After validating the hook system:

1. **Implement CIS Optimizer** (`src/cis/cis_optimizer.py`)
   - Gradient-based optimization to find targeted perturbations
   - Optimize delta to flip predictions to specific counterfactual targets

2. **Measure Factual Rigidity**
   - Calculate geometric cost (L2 norm of minimal perturbation)
   - Compare costs across different facts and layers

3. **Run Batch Experiments**
   - Test on full CounterFact dataset
   - Collect statistics on success rates and costs

## Troubleshooting

### "Layer index out of range"

Solution: Use `--layer` with a value between 0 and (num_layers - 1)

### "CUDA out of memory"

Solution: Enable 4-bit quantization in `config/model.yaml`:

```yaml
use_4bit: true
```

### "No causal effect detected"

Solutions:
1. Increase perturbation norm: `--perturbation-norm 0.1`
2. Try a different layer: `--layer 12` or `--layer 20`
3. Check that the model loaded correctly

### "Model produces gibberish"

Solution: Decrease perturbation norm: `--perturbation-norm 0.001`

## Technical Details

### How the Hook Works

1. **Hook Registration**: `register_forward_hook()` on transformer layer
2. **Interception**: Hook intercepts `(hidden_states, ...)` output tuple
3. **Perturbation**: Adds `delta` to `hidden_states[:, token_position, :]`
4. **Return**: Passes modified activations to next layer

### Why Middle Layers?

Factual knowledge is typically encoded in **middle-to-late layers** (layers 12-24 for a 32-layer model):
- **Early layers** (0-8): Low-level features, syntax
- **Middle layers** (12-20): **Semantic and factual knowledge** ← Best for CIS
- **Late layers** (24-31): Task-specific output preparation

### Token Position

- `-1` (last token): Usually the **subject** in "The [subject] is located in"
- This is where factual knowledge is typically retrieved
- Intervening on other positions has less effect on factual predictions

## References

- CIS Paper: Counterfactual Internal States for causal intervention
- Residual Stream: The sum of all previous layer outputs in a transformer
- Hook System: PyTorch's `register_forward_hook()` API
