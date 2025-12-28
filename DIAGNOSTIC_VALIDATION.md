# CIS Diagnostic Validation

This document describes the strict validation pass implemented to verify correctness of the CIS optimization before trusting any research results.

## Philosophy

**DO NOT TRUST RESULTS UNTIL PROVEN CORRECT**

We assume there are bugs until we explicitly verify:
- ✅ Delta starts at zero (no reuse)
- ✅ Hook modifies only target position (no leakage)
- ✅ Margin is enforced correctly (gap >= margin)
- ✅ Loss computation is correct (zero iff gap >= margin)
- ✅ Hooks are cleaned up (no accumulation)

## Critical Issues Being Validated

### Issue 1: Suspicious Convergence Speed
**Symptom**: Optimization converges in 2-3 steps
**Possible causes**:
- Delta not initialized to zero (reused from previous run)
- Hook applying to multiple positions
- Stopping condition too permissive (top-1 instead of gap-based)

### Issue 2: Stopping Condition Ambiguity
**Symptom**: Unclear what triggers convergence
**Possible causes**:
- Multiple stopping conditions (top-1, probability, loss, gap)
- Conditions not clearly logged
- Success declared prematurely

### Issue 3: Hook Lifecycle
**Symptom**: Unknown if hooks are cleaned up properly
**Possible causes**:
- Hooks accumulate across runs
- Hooks not removed on error
- Multiple hooks active simultaneously

### Issue 4: Hook Scope
**Symptom**: Unclear if only target position is modified
**Possible causes**:
- Hook applies delta to all positions
- Token position indexing error
- Batch dimension confusion

## Validation Strategy

### 1. Enforce Margin Correctly (CRITICAL)

**Implementation**:
```python
# Compute gap at EVERY iteration
gap = logit_target - logit_original

# Print at EVERY step
print(f"Step {step}: gap={gap:.4f}, margin={margin:.4f}, satisfied={gap >= margin}")

# ONLY stopping condition
if gap >= margin:
    success = True
    break

# If max_steps reached without gap >= margin
if not success:
    print("FAILED: Did not achieve required margin")
```

**What This Fixes**:
- No ambiguity about convergence
- No premature stopping based on top-1 token
- Clear success/failure criteria

**Expected Behavior**:
- Gap starts negative (original > target)
- Gap increases gradually as delta grows
- Success when gap >= margin
- May fail if margin too high or max_steps too low

### 2. Verify Delta Initialization (CRITICAL)

**Implementation**:
```python
# Before EACH optimization run
delta = LearnableDelta(hidden_dim=hidden_size, init_method="zeros", device=device)

# Verify
initial_norm = delta.get_norm(p=2)
print(f"Initial ||δ|| = {initial_norm:.8e}")
assert initial_norm < 1e-6, f"Delta not zero! ||δ|| = {initial_norm}"
```

**What This Fixes**:
- No delta reuse across runs
- No hidden state from previous optimizations
- Explicit verification printed

**Expected Behavior**:
- ||δ|| = 0.00000000 at start
- Assertion never fails
- If assertion fails → ABORT, report bug

### 3. Guarantee Hook Lifecycle (CRITICAL)

**Implementation**:
```python
handle = None
try:
    handle, _ = add_residual_perturbation_hook(model, layer_idx, delta_value, token_position)

    # Run forward pass
    outputs = model(**inputs)

    # Compute loss, backward, step

finally:
    # ALWAYS remove hook
    if handle is not None:
        handle.remove()
```

**What This Fixes**:
- Hook removed even if optimization fails
- No hook accumulation
- No lingering hooks between runs

**Expected Behavior**:
- One hook active during forward pass
- Hook removed immediately after
- No hooks active between iterations

### 4. Hook Scope Test (CRITICAL)

**Debug Mode Test**:
```python
# 1. Capture baseline
h_before = capture_residual_at_all_positions()

# 2. Apply fixed delta (e.g., ones * 0.01)
test_delta = torch.ones(hidden_dim) * 0.01

# 3. Run with hook
h_after = capture_residual_at_all_positions()

# 4. Verify
diff_target = ||h_after[target_pos] - h_before[target_pos]||
diff_other = max(||h_after[other_pos] - h_before[other_pos]||)

assert diff_target > 1e-3  # Target changed
assert diff_other < 1e-5   # Others unchanged
```

**What This Fixes**:
- Verifies hook only touches target position
- Rules out batch dimension errors
- Rules out all-position perturbation

**Expected Behavior**:
- Target position changed significantly
- All other positions unchanged (< 1e-5)
- Test passes before running optimization

### 5. Loss Sanity Check (CRITICAL)

**Implementation**:
```python
gap = logit_target - logit_original
task_loss = margin_flip_loss(logits, target_id, original_id, margin)

# Verify
if gap >= margin:
    assert task_loss.item() < 1e-4, f"Loss should be zero! gap={gap}, loss={task_loss}"
```

**What This Fixes**:
- Verifies loss formula is correct
- Catches computational errors
- Ensures loss=0 iff gap>=margin

**Expected Behavior**:
- When gap >= margin, loss ≈ 0
- When gap < margin, loss > 0
- Assertion never fails

## Usage

### Basic Diagnostic Run

```bash
python src/experiments/run_cis_diagnostic.py --config config/experiment.yaml
```

**What It Does**:
1. Loads model (Mistral-7B)
2. Verifies delta initialization
3. Runs optimization with gap-based stopping
4. Logs gap, margin, loss, ||δ|| at EVERY step
5. Reports success/failure clearly

**Parameters** (FIXED for diagnostic):
- layer_idx = 16
- margin = 1.0
- lambda_l2 = 0.01
- max_steps = 50

### Debug Mode (with Hook Scope Test)

```bash
python src/experiments/run_cis_diagnostic.py --config config/experiment.yaml --debug
```

**Additional Verification**:
1. Runs hook scope test BEFORE optimization
2. Verifies only target position modified
3. Aborts if hook scope test fails

## Output Format

### Optimization Loop

```
Step   Gap        Margin     Gap>=M     Loss       ||δ||      P(tgt)     P(orig)
----------------------------------------------------------------------------------
0      -5.2340    1.0000     False      6.2340     0.000000   0.1234     0.7891
1      -4.1234    1.0000     False      5.1234     0.012345   0.1456     0.7234
2      -3.0123    1.0000     False      4.0123     0.034567   0.1789     0.6789
...
25     0.9876     1.0000     False      0.0124     0.234567   0.4987     0.3210
26     1.0123     1.0000     True       0.0000     0.245678   0.5123     0.3098

✓ CONVERGED at step 26: gap (1.0123) >= margin (1.0000)
```

### Final Summary

```
================================================================================
FINAL RESULTS
================================================================================

Success: True
Final gap: 1.0123 (required: 1.0000)
Final ||δ||: 0.245678
Final P(target): 0.5123
Final P(orig): 0.3098
Steps: 27

================================================================================
VALIDATION SUMMARY
================================================================================

✓ All assertions passed:
  1. Delta initialized to zero
  2. Hook lifecycle managed correctly
  3. Stopping condition enforced (gap >= margin)
  4. Hook scope verified (only target position modified)  [if --debug]

✓ Optimization SUCCEEDED
  Achieved gap (1.0123) >= margin (1.0000)
  Geometric cost: 0.245678
```

## Interpretation

### Success Case
- Gap starts negative (original > target)
- Gap increases gradually (NOT in 2-3 steps)
- Delta norm grows smoothly (NOT already large at step 0)
- Convergence when gap >= margin
- Final ||δ|| is defensible

**This indicates**: Optimization is working correctly

### Failure Case (Expected for Some Parameters)
- Gap increases but never reaches margin
- Max steps reached
- Failure is clearly reported

**This indicates**: Margin too high or max_steps too low (NOT a bug)

### Bug Detection

**If you see**:
- Step 0: ||δ|| > 0.001 → Delta not initialized to zero!
- Step 0-2: gap already >= margin → Hook bug or delta reuse!
- Gap satisfied but loss > 0 → Loss computation bug!
- Assertion failure → ABORT and debug!

## Differences from Production Code

| Aspect | Production | Diagnostic |
|--------|-----------|------------|
| Stopping condition | Multiple (top-1, prob, loss, gap) | ONLY gap >= margin |
| Delta init | Implicit | Explicit with assertion |
| Hook cleanup | Implicit | Explicit try/finally |
| Logging | Every N steps | EVERY step |
| Loss verification | None | Assertion on gap vs loss |
| Scope test | None | Optional --debug mode |
| Parameters | Configurable | Fixed for reproducibility |

## Next Steps

**After diagnostic passes**:
1. Review output for gradual convergence (NOT 2-3 steps)
2. Verify ||δ|| starts at zero
3. Check gap increases smoothly
4. Confirm success/failure matches gap >= margin

**If diagnostic fails**:
1. DO NOT proceed to sweeps
2. Debug the specific assertion that failed
3. Fix the bug
4. Re-run diagnostic
5. Only proceed when ALL validations pass

**If diagnostic passes**:
1. Trust that measurement pipeline is correct
2. Proceed to lambda sweep (find minimal-norm solution)
3. Proceed to margin/layer sweeps
4. Use normalized metrics for comparisons

## Scientific Standard

This validation pass embodies the principle:
**"A failed experiment with correct instrumentation is more valuable than a successful experiment with hidden bugs."**

We PREFER:
- Clear failure reports
- Assertion errors that catch bugs
- Conservative stopping conditions
- Verbose logging

Over:
- False successes
- Hidden assumptions
- Permissive conditions
- Silent failures

## Files

- `src/experiments/run_cis_diagnostic.py` - Main diagnostic script
- `DIAGNOSTIC_VALIDATION.md` - This document
- `config/experiment.yaml` - Configuration (unchanged)

## References

- Scientific debugging: Zeller, "Why Programs Fail" (2009)
- Validation in ML: Sculley et al., "Hidden Technical Debt in ML Systems" (2015)
- Test-driven research: Peng, "Reproducible Research in Computational Science" (2011)
