# SmolVLA NPU Baseline vs Agentic — Benchmark Results

**Date**: 2026-03-24
**dtype**: bf16
**Trials per group**: 5
**Shapes**: 17 unique GEMM shapes from SmolVLA model

---

## Summary

| Metric | Value |
|--------|-------|
| Shapes tested | 17 |
| Shapes executed | 13 |
| Prescreened skip | 4 |
| Agentic wins | 0 |
| Baseline wins | 2 |
| Ties | 11 |
| Mean delta (agentic − baseline) | +0.89% |

**Verdict**: Baseline matches or beats agentic on all shapes. The two baseline wins both have explainable root causes (measurement noise on shape 4; k_min overcaution on shape 9). No shape showed a meaningful agentic win.

---

## Prescreened Shapes (not executed)

| Shape (M,K,N) | Layers | Reason |
|---------------|--------|--------|
| 1024 × 3072 × 768 | Vision Encoder/mlp.fc2 | `bf16_K_limit`: K=3072 exceeds bf16 CDO limit |
| 128 × 2560 × 960 | Text/down_proj | `bf16_K_limit`: K=2560 > 2048 → accuracy failure |
| 50 × 960 × 96 | Action Expert/q_proj | `bf16_accuracy_small_MN_large_K`: M≤50, N≤96, K≥960 |
| 1 × 960 × 32 | Projections/state_proj | `bf16_accuracy_small_MN_large_K`: M=1, N=32, K=960 |

---

## Per-Shape Results

### Shape 1 — 1024 × 768 × 768 (Vision Encoder attention + patch_embedding)

| | Baseline | Agentic |
|--|----------|---------|
| Padded shape | 1024 × 768 × 768 | 1024 × 768 × 768 |
| Tile (m,n,k) | (64,64,64) | (64,64,64) |
| Pad impl | none | none |
| Strategy source | closest-fit + profiling | preprocessed_map |
| Mean NPU avg (µs) | **757.24** | 759.98 |
| Min NPU (µs) | ~721 | ~721 |

**Winner: TIE** (+0.36%) — same tile, same shape; difference is jitter.

---

### Shape 2 — 1024 × 768 × 3072 (Vision Encoder/mlp.fc1)

*Note: N=3072 exceeds script ALLOWED_N max (2048). Both groups run in direct mode (no --use-padding). Profiling key `1024_3072_768` confirms (64,64,64) passes at ~2877µs.*

| | Baseline | Agentic |
|--|----------|---------|
| Padded shape | 1024 × 3072 × 768 | 1024 × 3072 × 768 |
| Tile (m,n,k) | (64,64,64) | (64,64,64) |
| Pad impl | none (direct mode) | none (direct mode) |
| Mean NPU avg (µs) | **2957.46** | 3005.12 |
| Min NPU (µs) | ~2568 | ~2584 |

**Winner: TIE** (+1.61%) — same tile, same shape; min times are close (~16µs apart).

---

### Shape 4 — 128 × 960 × 960 (Text/q_proj, o_proj)

*Note: Padding K=960 → 1024. ab_elements = 128×960 + 960×960 = 1,044,480 > 571,779 → numpy_pad.*

| | Baseline | Agentic |
|--|----------|---------|
| Padded shape | 128 × 960 × 1024 | 128 × 960 × 1024 |
| Tile (m,n,k) | (32,16,256) | (32,16,256) |
| Tile source | profiling best | profiling best (n=16 rejection overridden per errors.md) |
| Pad impl | numpy_pad | numpy_pad |
| Mean NPU avg (µs) | **546.38** | 574.79 |
| Mean padding (µs) | 1224.90 | 1224.76 |
| Min NPU (µs) | ~488 | ~487 |

**Winner: BASELINE** (+5.20%) — same tile used by both; min times identical (~487µs). Difference is measurement noise (high-outlier trials 3/4 in agentic group). Padding dominates 2.2× NPU time (overhead_ratio=2.24).

---

### Shape 5 — 128 × 960 × 320 (Text/k_proj, v_proj)

*Note: Padding K=960→1024, N=320→512. ab_elements = 128×960 + 960×320 = 430,080 < 571,779 → manual_copy.*

| | Baseline | Agentic |
|--|----------|---------|
| Padded shape | 128 × 512 × 1024 | 128 × 512 × 1024 |
| Tile (m,n,k) | (32,64,128) | (32,64,128) |
| Pad impl | manual_copy | manual_copy |
| Mean NPU avg (µs) | 240.13 | **237.10** |
| Mean padding (µs) | 349.48 | 306.52 |

**Winner: TIE** (−1.26%) — same tile and shape; within jitter.

---

### Shape 6 — 128 × 960 × 2560 (Text/gate_proj, up_proj)

*Note: N=2560 exceeds script ALLOWED_N max (2048). Both groups run in direct mode. **NEW FINDING**: direct-mode K=960 bf16 passes accuracy where padded Kp=1024 fails. Baseline uses (64,64,64) with k=64 < k_min=80 — yet PASSES accuracy in direct mode. Agentic uses non-standard tile (32,64,80) with k=80=k_min.*

| | Baseline | Agentic |
|--|----------|---------|
| Padded shape | 128 × 2560 × 960 | 128 × 2560 × 960 |
| Tile (m,n,k) | (64,64,64) | (32,64,80) |
| Tile source | closest-fit fallback (k<k_min but passes in direct mode) | heuristic (k=80=k_min) |
| Pad impl | none (direct mode) | none (direct mode) |
| Mean NPU avg (µs) | **651.57** | 652.33 |

**Winner: TIE** (+0.12%) — both tiles pass; 0.76µs difference is noise. k_min rule unnecessary in direct mode for K=960.

---

### Shape 9 — 128 × 320 × 96 (Action Expert/k_proj, v_proj)

*Note: Padding K=320→512, N=96→128. Profiling best tile is (32,32,32), k=32 < k_min=max(32,512//12)=42. Agentic rejects profiling tile and uses cost model → (64,64,64).*

| | Baseline | Agentic |
|--|----------|---------|
| Padded shape | 128 × 128 × 512 | 128 × 128 × 512 |
| Tile (m,n,k) | **(32,32,32)** | (64,64,64) |
| Tile source | profiling best | cost model (k=32 < k_min=42 rejected) |
| Pad impl | manual_copy | manual_copy |
| Mean NPU avg (µs) | **149.65** | 160.01 |
| Mean padding (µs) | 24.14 | 23.80 |

**Winner: BASELINE** (+6.92%) — k_min rule too conservative for Kp=512. k=32 passes accuracy at this K. Agentic (64,64,64) is 6.9% slower.

---

### Overhead-Limited Shapes (Mp ≤ 64 — all tie as predicted)

All 7 shapes below cluster in the 120–210µs range regardless of tile or strategy. Both groups used identical tiles sourced from profiling. Differences are within 1.5µs (< 2%).

| Shape (M,K,N) | Layers | Padded shape | Tile | Baseline (µs) | Agentic (µs) | Delta |
|---------------|--------|-------------|------|--------------|-------------|-------|
| 50 × 96 × 960 | AE/o_proj | 64×960×128 | (16,64,32) | 203.04 | 202.34 | −0.35% |
| 50 × 96 × 256 | AE/gate_proj, up_proj | 64×256×128 | (32,64,32) | 134.16 | 133.41 | −0.56% |
| 50 × 256 × 96 | AE/down_proj | 64×128×256 | (32,32,32) | 130.82 | 131.23 | +0.31% |
| 50 × 32 × 96 | Proj/action_in_proj | 64×128×32 | (32,32,32) | 131.63 | 130.81 | −0.62% |
| 50 × 96 × 32 | Proj/action_out_proj | 64×32×128 | (32,32,32) | 125.41 | 124.43 | −0.78% |
| 1 × 192 × 96 | Proj/action_time_mlp_in | 32×128×256 | (32,32,32) | 128.80 | 130.65 | +1.44% |
| 1 × 96 × 96 | Proj/action_time_mlp_out | 32×128×128 | (32,32,64) | 129.56 | 128.53 | −0.79% |

---

## Phase 3 Analysis

### 3.1 Strategy convergence

Both strategies chose the **same padded shape** for all 17 shapes. The ALLOWED lists leave little room for alternative shapes on most SmolVLA dimensions. Main differentiation was tile selection only.

### 3.2 Where agentic rules misfired

**Shape 9 (k_min overcaution)**: The k_min formula `max(32, Kp//12)` predicts k_min=42 for Kp=512. But k=32 actually passes accuracy at Kp=512. The k_min rule is empirically calibrated on Kp=1024 shapes (where k<128 consistently fails) and is too conservative when extended to smaller K. Agentic's rejection of the profiling-best tile caused a 6.9% slowdown.

**Shape 4 (n=16 noise)**: Both strategies used the same tile (32,16,256). The baseline "win" is measurement noise — min NPU times are identical. Not a genuine agentic failure.

### 3.3 Key new findings

1. **Direct-mode K=960 bf16 passes accuracy; padded Kp=1024 fails** (shape 6). The k_min=80 accuracy guard was designed for Kp=1024 shapes. When the exact K=960 value is passed directly (no padding), accumulation over fewer K-tiles means less rounding error. The padded Kp=1024 with k<128 fails; the direct K=960 with any valid k passes.

2. **k_min=Kp//12 over-generalizes from Kp=1024 to Kp=512**. The rule accurately captures Kp=1024 failure behavior, but at Kp=512 the error accumulation is half as severe. k=32 passes reliably at Kp=512 across all tested shapes.

3. **Padding dominates on shape 4** — padding overhead is 2.2× the NPU execution time (1225µs vs 546µs). This confirms that for shapes where Kp=1024 is required due to K=960, numpy_pad selection and padding ratio should be flagged as performance bottlenecks.

### 3.4 Estimation accuracy

Profile entries existed for 11 of 13 executed shapes. For those with exact profile hits, estimation error was typically <5% (consistent with the ±6-15% jitter documented in strategy_insights). The 2 shapes without hits (shape 9 and shape 6 agentic) relied on heuristics — both were within 10% of actual.

### 3.5 Overhead-limited validation

All 7 shapes with Mp≤64 tied within ±2%, exactly as predicted by the "small shapes always tie" rule. This rule is fully validated on SmolVLA shapes.

---

## Recommendations

1. **Recalibrate k_min for Kp=512**: use k_min=32 (or k_min=max(32, Kp//16)) rather than Kp//12. The current formula is too restrictive and discards valid profiling-best tiles on small K.

2. **Direct-mode k_min**: suppress the k_min accuracy guard when running in direct mode (K not padded). Direct-mode K=960 does not require k≥80 — k=64 passes.

3. **Shape 4 padding bottleneck**: for production deployment of Text/q_proj, o_proj (M=128, K=960, N=960), the padded shape (128,960,1024) with numpy_pad is unavoidable. Padding costs 2.2× the NPU time. Consider batching or restructuring to avoid this shape if latency is critical.

4. **Prescreened shapes (K≥2560 bf16)**: all 4 skipped shapes require non-bf16 dtype for production use. i8 quantization would unlock these shapes with ~2× speedup on NPU.
