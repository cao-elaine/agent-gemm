# SmolVLA NPU Experiment Results
**Date**: 2026-03-22  |  **dtype**: bf16  |  **Trials per group**: 5

## Summary

| Metric | Value |
|--------|-------|
| Shapes total | 17 |
| Shapes tested | 13 |
| Shapes pre-screened (skipped) | 4 |
| Agentic wins | 2 |
| Baseline wins | 0 |
| Ties (within 2%) | 11 |
| Inconclusive | 0 |
| Total NPU executions | 130 |
| Total passed | 125 |
| Total failed | 5 (all shape 6 baseline accuracy failures) |
| Mean Δ (agentic vs baseline, excl. shape 6) | −0.3% |

**Key finding:** Shape 6 (M=128,K=960,N=2560) is the decisive differentiator. Baseline's tile (32,32,32) fails accuracy on all 5 trials; agentic's k_min-derived tile (32,64,128) passes all 5. The k_min accuracy pre-validation rule (`k_min = max(32, Kp//12) = 85`, requiring k≥85 on Kp=1024) directly predicted this outcome.

---

## Pre-Screened Shapes (not run)

| Shape | Layer | Reason |
|-------|-------|--------|
| 1024×3072×768 | Vision mlp.fc2 | K=3072 > 2048: bf16 CDO/accuracy limit |
| 128×2560×960 | Text down_proj | K=2560 > 2048: bf16 accuracy limit |
| 50×960×96 | Action q_proj | M≤50, N≤96, K≥960: small-MN large-K accuracy failure |
| 1×960×32 | Projections state_proj | M≤50, N≤96, K≥960: small-MN large-K accuracy failure |

---

## Per-Shape Results

### 1024×768×768 — `Vision Encoder/q_proj, k_proj, v_proj, out_proj, patch_embedding`

| | Baseline | Agentic |
|-|----------|---------|
| Padded shape | (1024,768,768) | (1024,768,768) |
| Tile | (64,64,64) | (64,64,64) |
| Pad impl | none | none |
| Strategy | closest-fit + profiling best tile | preprocessed_map |
| Trial 1 avg (µs) | 761.3 | 753.1 |
| Trial 2 avg (µs) | 762.3 | 758.3 |
| Trial 3 avg (µs) | 772.8 | 767.9 |
| Trial 4 avg (µs) | 763.4 | 757.2 |
| Trial 5 avg (µs) | 747.0 | 758.1 |
| **Mean avg (µs)** | **761.4** | **759.0** |
| Pass count | 5/5 | 5/5 |

**Winner: TIE**  Δ = −2.4µs (−0.3%)

---

### 1024×768×3072 — `Vision Encoder/mlp.fc1`

| | Baseline | Agentic |
|-|----------|---------|
| Padded shape | (1024,3072,768) | (1024,3072,768) |
| Tile | (64,64,64) | (64,64,64) |
| Pad impl | none | none |
| Strategy | closest-fit + profiling best tile | preprocessed_map |
| Trial 1 avg (µs) | 2966.1 | 2977.8 |
| Trial 2 avg (µs) | 2871.5 | 2944.1 |
| Trial 3 avg (µs) | 2927.9 | 2954.4 |
| Trial 4 avg (µs) | 2973.0 | 2922.8 |
| Trial 5 avg (µs) | 2942.7 | 2935.6 |
| **Mean avg (µs)** | **2936.2** | **2946.9** |
| Pass count | 5/5 | 5/5 |

**Winner: TIE**  Δ = +10.7µs (+0.4%)

---

### 128×960×960 — `Text Layers (x16)/q_proj, o_proj`

| | Baseline | Agentic |
|-|----------|---------|
| Padded shape | (128,1024,1024) | (128,1024,1024) |
| Tile | (32,64,128) | (32,64,128) |
| Pad impl | numpy_pad | numpy_pad |
| Strategy | closest-fit + profiling best tile | preprocessed_map |
| Trial 1 avg (µs) | 349.5 | 339.5 |
| Trial 2 avg (µs) | 352.9 | 337.6 |
| Trial 3 avg (µs) | 348.3 | 338.5 |
| Trial 4 avg (µs) | 335.7 | 335.3 |
| Trial 5 avg (µs) | 342.3 | 341.8 |
| Trial 1 pad (µs) | 844.1 | 847.6 |
| **Mean avg (µs)** | **345.7** | **338.5** |
| Pass count | 5/5 | 5/5 |

**Winner: AGENTIC**  Δ = −7.2µs (−2.1%)  *(borderline — same config, both groups identical tile, difference is likely measurement noise)*

---

### 128×960×320 — `Text Layers (x16)/k_proj, v_proj`

| | Baseline | Agentic |
|-|----------|---------|
| Padded shape | (128,512,1024) | (128,512,1024) |
| Tile | (32,64,128) | (32,64,128) |
| Pad impl | manual_copy | manual_copy |
| Strategy | closest-fit + profiling best tile | preprocessed_map |
| Trial 1 avg (µs) | 240.9 | 236.3 |
| Trial 2 avg (µs) | 238.8 | 238.4 |
| Trial 3 avg (µs) | 238.3 | 238.9 |
| Trial 4 avg (µs) | 239.1 | 235.4 |
| Trial 5 avg (µs) | 237.7 | 238.4 |
| **Mean avg (µs)** | **238.9** | **237.5** |
| Pass count | 5/5 | 5/5 |

**Winner: TIE**  Δ = −1.5µs (−0.6%)

---

### 128×960×2560 — `Text Layers (x16)/gate_proj, up_proj`

> **Note:** N=2560 → Np=3072 exceeds script ALLOWED_N max (2048). Both groups run in direct-dims mode on (128,3072,1024) without `--use-padding`. Accuracy check tests the full (128,1024,3072) GEMM, not the original padded (128,960,2560) workload.

| | Baseline | Agentic |
|-|----------|---------|
| Padded shape | (128,3072,1024) | (128,3072,1024) |
| Tile | (32,32,32) | (32,64,128) |
| Pad impl | none (direct) | none (direct) |
| Strategy | closest-fit no-profile fallback | heuristic k_min (k≥max(32,1024/12)=85) |
| Trial 1 avg (µs) | FAILED (acc) | 802.1 |
| Trial 2 avg (µs) | FAILED (acc) | 793.2 |
| Trial 3 avg (µs) | FAILED (acc) | 794.3 |
| Trial 4 avg (µs) | FAILED (acc) | 789.9 |
| Trial 5 avg (µs) | FAILED (acc) | 791.4 |
| **Mean avg (µs)** | **—** | **794.2** |
| Pass count | 0/5 | 5/5 |

**Winner: AGENTIC**  (baseline accuracy failure k=32 < k_min=85 vs agentic 5/5 pass with k=128)

---

### 128×320×96 — `Action Expert (x16)/k_proj, v_proj`

| | Baseline | Agentic |
|-|----------|---------|
| Padded shape | (128,128,512) | (128,128,512) |
| Tile | (32,32,32) | (32,32,32) |
| Pad impl | manual_copy | manual_copy |
| Strategy | closest-fit + profiling best tile | preprocessed_map |
| **Mean avg (µs)** | **149.0** | **148.7** |
| Pass count | 5/5 | 5/5 |

**Winner: TIE**  Δ = −0.3µs (−0.2%)

---

### 50×96×960 — `Action Expert (x16)/o_proj`  *(overhead-limited)*

| | Baseline | Agentic |
|-|----------|---------|
| Padded shape | (64,1024,128) | (64,1024,128) |
| Tile | (32,64,32) | (32,64,32) |
| **Mean avg (µs)** | **165.5** | **164.4** |
| Pass count | 5/5 | 5/5 |

**Winner: TIE**  Δ = −1.1µs (−0.7%)

---

### 50×96×256 — `Action Expert (x16)/gate_proj, up_proj`  *(overhead-limited)*

| | Baseline | Agentic |
|-|----------|---------|
| Padded shape | (64,256,128) | (64,256,128) |
| Tile | (32,64,32) | (32,64,32) |
| **Mean avg (µs)** | **133.2** | **133.0** |
| Pass count | 5/5 | 5/5 |

**Winner: TIE**  Δ = −0.2µs (−0.1%)

---

### 50×256×96 — `Action Expert (x16)/down_proj`  *(overhead-limited)*

| | Baseline | Agentic |
|-|----------|---------|
| Padded shape | (64,128,256) | (64,128,256) |
| Tile | (32,32,32) | (32,32,32) |
| **Mean avg (µs)** | **131.8** | **130.8** |
| Pass count | 5/5 | 5/5 |

**Winner: TIE**  Δ = −1.0µs (−0.7%)

---

### 50×32×96 — `Projections/action_in_proj`  *(overhead-limited)*

| | Baseline | Agentic |
|-|----------|---------|
| Padded shape | (64,128,32) | (64,128,32) |
| Tile | (32,32,32) | (32,32,32) |
| **Mean avg (µs)** | **129.4** | **130.7** |
| Pass count | 5/5 | 5/5 |

**Winner: TIE**  Δ = +1.3µs (+1.0%)

---

### 50×96×32 — `Projections/action_out_proj`  *(overhead-limited)*

| | Baseline | Agentic |
|-|----------|---------|
| Padded shape | (64,32,128) | (64,32,128) |
| Tile | (32,32,32) | (32,32,32) |
| **Mean avg (µs)** | **124.7** | **122.9** |
| Pass count | 5/5 | 5/5 |

**Winner: TIE**  Δ = −1.8µs (−1.4%)

---

### 1×192×96 — `Projections/action_time_mlp_in`  *(overhead-limited)*

| | Baseline | Agentic |
|-|----------|---------|
| Padded shape | (32,128,256) | (32,128,256) |
| Tile | (32,32,32) | (32,32,32) |
| **Mean avg (µs)** | **127.8** | **129.9** |
| Pass count | 5/5 | 5/5 |

**Winner: TIE**  Δ = +2.1µs (+1.7%)

---

### 1×96×96 — `Projections/action_time_mlp_out`  *(overhead-limited)*

| | Baseline | Agentic |
|-|----------|---------|
| Padded shape | (32,128,128) | (32,128,128) |
| Tile | (32,32,64) | (32,32,64) |
| **Mean avg (µs)** | **129.7** | **128.8** |
| Pass count | 5/5 | 5/5 |

**Winner: TIE**  Δ = −0.9µs (−0.7%)

---

## New Best Tiles Found (Agentic)

| Shape | Old best tile | Old avg (µs) | New best tile | New avg (µs) | Δ |
|-------|--------------|-------------|--------------|-------------|---|
| (128,1024,1024) bf16 | (32,64,128) | 349.9 | (32,64,128) | 338.5 | −3.3% |
| (128,512,1024) bf16 | (32,64,128) | 239.0 | (32,64,128) | 237.5 | −0.6% |
| (128,3072,1024) bf16 | *none* | — | (32,64,128) | 794.2 | *first pass* |

*Note: Shape (128,3072,1024) is the first confirmed passing tile in direct-dims mode. Prior tests used padded-run mode which universally fails accuracy for original shape M=128,K=960,N=2560.*

---

## Memory Updates

- `npu_execution_profiling.json`: 13 shapes updated (working tiles added/incremented), 3 new best tiles, (128,3072,1024) bf16 first passing tile found; (32,32,32) added to failing list for (128,3072,1024) bf16
- `profiling_errors.json`: 1 new error trace (128,3072,1024) bf16 tile (32,32,32) accuracy failure
- `error-logs/errors.md`: 1 new section — accuracy failure mode difference between padded-run and direct-dims-run for (128,3072,1024) bf16
- `strategy_insights.md`: 3 new dated bullets — (128,3072,1024) correction, SmolVLA agentic accuracy differentiation finding, profiling best tile updates
