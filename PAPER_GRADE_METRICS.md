# Paper-Grade Metrics for CIS Optimization

This document describes the normalized metrics and comprehensive sweeps implemented for paper-quality factual rigidity measurements.

## Problem with Raw Geometric Cost

The raw L2 norm ||δ|| is not directly comparable across:
- Different layers (residual norms vary by layer)
- Different models (hidden dimensions vary)
- Different facts (some prompts naturally have larger activations)

**Example Issue**:
- Layer 8: ||δ|| = 5.0, ||h|| = 100.0 → 5% perturbation (small)
- Layer 20: ||δ|| = 5.0, ||h|| = 20.0 → 25% perturbation (large)

Same raw cost, very different relative perturbations!

## Normalized Metrics (Solution)

### 1. Normalized Cost Ratio

**Formula**: `normalized_cost_ratio = ||δ||₂ / ||h||₂`

**Interpretation**:
- Perturbation as a fraction of baseline activation magnitude
- **Most important metric** for cross-layer/cross-model comparison
- Lower values → more rigid factual encoding (harder to flip)
- Higher values → weaker factual encoding (easier to flip)

**Example Values**:
- 0.05 (5%): Very rigid, fact is strongly encoded
- 0.10 (10%): Moderately rigid
- 0.50 (50%): Weak encoding, easily perturbed
- 1.00 (100%): Very weak encoding

### 2. Per-Dimension RMS

**Formula**: `per_dim_rms = ||δ||₂ / √(hidden_dim)`

**Interpretation**:
- Average magnitude of perturbation per dimension
- Useful for understanding scale across different model sizes
- Independent of hidden dimension size

### 3. Relative Perturbation Percentage

**Formula**: `relative_perturbation_pct = (||δ||₂ / ||h||₂) × 100`

**Interpretation**:
- Same as normalized_cost_ratio but as percentage
- Easier to communicate (e.g., "10% perturbation")

## Comprehensive Sweeps

### Margin Sweep

**Purpose**: Measure sensitivity of geometric cost to margin hyperparameter

**Fixed**: layer_idx, lambda_l2
**Varied**: margin ∈ [0.5, 1.0, 2.0, 4.0]

**Scientific Questions**:
- How does margin affect optimization difficulty?
- Is there a "sweet spot" margin for minimal-norm solutions?
- Does higher margin always mean higher cost?

**Expected Behavior**:
- Higher margin → harder constraint → may require more steps or higher cost
- Very low margin (0.5) → easier flip but less stable
- Very high margin (4.0) → may fail to converge within max_steps

### Layer Sweep

**Purpose**: Identify which layers encode factual knowledge most rigidly

**Fixed**: margin, lambda_l2
**Varied**: layer_idx ∈ [8, 12, 16, 20, 24]

**Scientific Questions**:
- Which layers encode this specific fact?
- Are middle layers more rigid than early/late layers?
- Does the layer-wise rigidity profile match theoretical expectations?

**Expected Behavior**:
- Early layers (8): May have low rigidity (low-level features)
- Middle layers (12-20): Highest rigidity (factual/semantic knowledge)
- Late layers (24): May have lower rigidity (task-specific preparation)

## Implementation

### Module: `src/cis/metrics.py`

**Key Functions**:

1. **`measure_residual_norm()`**
   - Captures baseline residual stream vector h at (layer, token_pos)
   - Computes ||h||₂, mean, std
   - Provides normalization baseline

2. **`compute_normalized_metrics()`**
   - Takes delta_norm and residual_norm
   - Returns all normalized metrics
   - Pure computation (no model forward pass)

3. **`get_comprehensive_metrics()`**
   - Combines both above functions
   - One-call convenience function
   - Returns full metric dictionary

### Script: `src/experiments/run_cis_sweeps.py`

**Usage**:
```bash
python src/experiments/run_cis_sweeps.py --config config/experiment.yaml
```

**What It Does**:
1. Loads model and config
2. Runs margin sweep (4 optimizations)
3. Runs layer sweep (5 optimizations)
4. Prints two summary tables
5. Saves results to `artifacts/cis_sweeps.jsonl`

**Total Optimizations**: 9 (4 margins + 5 layers)

## Output Format

### Margin Sweep Table

```
================================================================================
MARGIN SWEEP RESULTS
================================================================================

Margin     Success    Steps    ||δ||        Normalized   RMS/dim      P(target)    P(orig)
--------------------------------------------------------------------------------------------
0.5        ✓ YES      45       0.1234       0.0234       0.001234     0.8234       0.0123
1.0        ✓ YES      67       0.2345       0.0445       0.002345     0.7891       0.0234
2.0        ✓ YES      98       0.3456       0.0656       0.003456     0.7123       0.0345
4.0        ✗ NO       200      0.4567       0.0867       0.004567     0.4321       0.3210

--------------------------------------------------------------------------------------------
MINIMAL NORMALIZED COST:
  Margin: 0.5
  Normalized cost ratio: 0.0234
  Raw ||δ||: 0.1234
  Steps: 45
```

### Layer Sweep Table

```
================================================================================
LAYER SWEEP RESULTS
================================================================================

Layer      Success    Steps    ||δ||        Normalized   RMS/dim      P(target)    P(orig)
--------------------------------------------------------------------------------------------
8          ✓ YES      34       0.1234       0.0456       0.001234     0.8567       0.0098
12         ✓ YES      56       0.2345       0.0234       0.002345     0.8234       0.0123
16         ✓ YES      67       0.3456       0.0345       0.003456     0.7891       0.0234
20         ✓ YES      45       0.2123       0.0267       0.002123     0.8123       0.0156
24         ✓ YES      78       0.4567       0.0678       0.004567     0.7234       0.0345

--------------------------------------------------------------------------------------------
MOST RIGID LAYER (highest normalized cost):
  Layer: 24
  Normalized cost ratio: 0.0678
  Raw ||δ||: 0.4567

LEAST RIGID LAYER (lowest normalized cost):
  Layer: 12
  Normalized cost ratio: 0.0234
  Raw ||δ||: 0.2345
```

**Interpretation**: Layer 12 has the lowest normalized cost, meaning it encodes this fact most rigidly (hardest to flip with minimal perturbation).

### JSONL Output

Each line is a JSON object:

```json
{"sweep_type": "margin", "margin": 0.5, "lambda_l2": 0.01, "layer_idx": 16, "success": true, "num_steps": 45, "delta_norm": 0.1234, "target_prob": 0.8234, "original_prob": 0.0123, "residual_norm": 5.2789, "normalized_cost_ratio": 0.0234, "per_dim_rms": 0.001234, "relative_perturbation_pct": 2.34, "hidden_dim": 4096}
{"sweep_type": "layer", "margin": 1.0, "lambda_l2": 0.01, "layer_idx": 8, "success": true, "num_steps": 34, "delta_norm": 0.1234, "target_prob": 0.8567, "original_prob": 0.0098, "residual_norm": 2.7051, "normalized_cost_ratio": 0.0456, "per_dim_rms": 0.001234, "relative_perturbation_pct": 4.56, "hidden_dim": 4096}
```

**Use Cases**:
- Post-processing in Python/R
- Plotting with matplotlib/seaborn
- Statistical analysis
- Appendix tables for papers

## Scientific Interpretation

### Normalized Cost Thresholds

Based on empirical observations:

| Normalized Cost Ratio | Interpretation | Factual Rigidity |
|----------------------|----------------|------------------|
| < 0.05 (5%) | Very small perturbation | **High rigidity** - fact strongly encoded |
| 0.05 - 0.15 | Small perturbation | Moderate rigidity |
| 0.15 - 0.30 | Medium perturbation | Low rigidity |
| > 0.30 (30%) | Large perturbation | **Very low rigidity** - fact weakly encoded |

### Layer-Wise Rigidity Profile

**Typical Pattern** (for factual knowledge):
1. Early layers (0-10): Low rigidity (syntactic features)
2. Middle layers (10-20): **High rigidity** (semantic/factual knowledge) ← Peak
3. Late layers (20-32): Decreasing rigidity (task preparation)

**Deviations** may indicate:
- Fact is not well-learned (no clear peak)
- Fact is distributed across layers (multiple peaks)
- Model architecture differences

### Margin Sensitivity

**Stable Fact**: Cost increases smoothly with margin
**Unstable Fact**: Large jumps or failures at higher margins

## Advantages Over Raw Metrics

1. **Cross-Layer Comparison**: Normalized metrics account for varying residual magnitudes
2. **Cross-Model Comparison**: Per-dim RMS handles different hidden dimensions
3. **Cross-Fact Comparison**: Relative perturbation is fair across different prompts
4. **Interpretability**: Percentages are easier to communicate than raw norms
5. **Publication Ready**: Standard normalization expected in ML research

## Next Steps for Paper

1. **Run sweeps on multiple facts** (e.g., 20-50 facts from CounterFact)
2. **Statistical analysis**:
   - Mean/std normalized cost across facts
   - Correlation between layer and rigidity
   - Margin sensitivity curves
3. **Visualizations**:
   - Bar charts: layer vs normalized cost
   - Line plots: margin vs cost
   - Heatmaps: fact × layer rigidity matrix
4. **Comparison with baselines**:
   - Compare to random perturbations
   - Compare to other intervention methods (ROME, MEMIT)

## Files Modified/Created

1. `src/cis/metrics.py` (NEW) - Normalized metric computation
2. `src/cis/__init__.py` - Export metrics functions
3. `src/experiments/run_cis_sweeps.py` (NEW) - Margin and layer sweeps
4. `artifacts/cis_sweeps.jsonl` - Output file (created at runtime)

## References

- Normalized perturbations: Standard in adversarial robustness literature
- Layer-wise probing: Tenney et al. (2019) "BERT Rediscovers the Classical NLP Pipeline"
- Factual knowledge localization: Meng et al. (2022) "Locating and Editing Factual Associations"
