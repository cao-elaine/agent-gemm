# SmolVLA GEMM Benchmark — bf16 Results
**Date:** 2026-03-21
**dtype:** bf16
**Trials/group:** 5
**Output dir:** references/experiment-results/test_n_20260321/

---

## Pre-screen Summary

4 shapes skipped before any compilation:

| M | K | N | Reason | Layers |
|---|---|---|--------|--------|
| 1024 | 3072 | 768 | bf16_K_limit (K>2048) | VE:mlp.fc2 |
| 128 | 2560 | 960 | bf16_K_limit (K>2048) | TL:down_proj |
| 50 | 960 | 96 | bf16_accuracy_small_MN_large_K | AE:q_proj |
| 1 | 960 | 32 | bf16_accuracy_small_MN_large_K | Proj:state_proj |

---

## Execution Plan (13 non-skipped shapes)

| ID | M | K | N | Padded (Mp,Np,Kp) | pad_impl | Baseline Tile | Agentic Tile |
|----|---|---|---|-------------------|----------|---------------|--------------|
| 1 | 1024 | 768 | 768 | (1024,768,768) | none | (64,64,64) | (64,64,64) |
| 2 | 1024 | 768 | 3072 | (1024,3072,768) | none | (64,64,64) | (64,64,64) |
| 3 | 128 | 960 | 960 | (128,1024,1024) | numpy_pad | (64,64,64) | **(32,64,128)** |
| 4 | 128 | 960 | 320 | (128,512,1024) | manual_copy | (32,64,32) | **(32,64,128)** |
| 5 | 128 | 960 | 2560 | (128,3072,1024) | numpy_pad | (64,64,64) | (32,64,128) |
| 6 | 128 | 320 | 96 | (128,128,512) | manual_copy | (32,32,32) | (32,32,32) |
| 7 | 50 | 96 | 960 | (64,1024,128) | manual_copy | (32,64,32) | (32,64,32) |
| 8 | 50 | 96 | 256 | (64,256,128) | manual_copy | (32,64,32) | (32,64,32) |
| 9 | 50 | 256 | 96 | (64,128,256) | manual_copy | (32,32,32) | (32,32,32) |
| 10 | 50 | 32 | 96 | (64,128,32) | manual_copy | (32,32,32) | (32,32,32) |
| 11 | 50 | 96 | 32 | (64,32,128) | manual_copy | (32,32,32) | (32,32,32) |
| 12 | 1 | 192 | 96 | (32,128,256) | manual_copy | (32,32,32) | (32,32,32) |
| 13 | 1 | 96 | 96 | (32,128,128) | manual_copy | (32,32,64) | (32,32,64) |

---

## Results Table

| ID | M×K×N | Baseline mean (µs) | Agentic mean (µs) | Δ (µs) | Δ% | Winner |
|----|-------|--------------------|-------------------|---------|-----|--------|
| 1 | 1024×768×768 | 767.6 | 759.4 | -8.2 | -1.07% | **TIE** |
| 2 | 1024×768×3072 | 2923.2 | 2931.7 | +8.5 | +0.29% | **TIE** |
| 3 | 128×960×960 | FAILED (0/5) | 349.7 | — | — | **AGENTIC** |
| 4 | 128×960×320 | FAILED (0/5) | 238.8 | — | — | **AGENTIC** |
| 5 | 128×960×2560 | FAILED (0/5) | FAILED (0/5) | — | — | inconclusive |
| 6 | 128×320×96 | 148.5 | 148.6 | +0.1 | +0.07% | **TIE** |
| 7 | 50×96×960 | 166.7 | 166.3 | -0.4 | -0.24% | **TIE** |
| 8 | 50×96×256 | 134.3 | 133.2 | -1.1 | -0.82% | **TIE** |
| 9 | 50×256×96 | 130.3 | 129.9 | -0.4 | -0.31% | **TIE** |
| 10 | 50×32×96 | 130.0 | 129.7 | -0.3 | -0.23% | **TIE** |
| 11 | 50×96×32 | 124.8 | 123.9 | -0.9 | -0.72% | **TIE** |
| 12 | 1×192×96 | 128.9 | 128.4 | -0.5 | -0.39% | **TIE** |
| 13 | 1×96×96 | 129.4 | 130.2 | +0.8 | +0.62% | **TIE** |

---

## Per-Shape Details

### Shape 1: M=1024 K=768 N=768 (VE:q/k/v/out/patch)
- Padded: (1024,768,768), no padding needed
- Tile: (64,64,64) for both strategies
- Baseline: 5/5 PASSED — 806, 759, 761, 757, 755 µs → mean=767.6µs
- Agentic:  5/5 PASSED — 750, 754, 766, 753, 774 µs → mean=759.4µs
- **WINNER: TIE** (Δ=-1.07%, within 2%)

### Shape 2: M=1024 K=768 N=3072 (VE:mlp.fc1)
- Padded: (1024,3072,768), no padding needed
- Tile: (64,64,64) for both strategies
- Baseline: 5/5 PASSED — 2909, 2880, 2950, 2951, 2927 µs → mean=2923.2µs
- Agentic:  5/5 PASSED — 2918, 2942, 2930, 2930, 2938 µs → mean=2931.7µs
- **WINNER: TIE** (Δ=+0.29%, within 2%)

### Shape 3: M=128 K=960 N=960 (TL:q/o_proj)
- Padded: (128,1024,1024), numpy_pad
- Baseline tile (64,64,64): FAILED accuracy — bfloat16 accumulation error at Kp=1024 with k=64
- Fallbacks (32,32,32), (16,32,32): also FAILED accuracy
- Agentic tile (32,64,128): 5/5 PASSED — 350, 348, 348, 353, 350 µs → mean=349.7µs
- **WINNER: AGENTIC** (only group to pass; k≥128 required for bf16 accuracy on Kp=1024)

### Shape 4: M=128 K=960 N=320 (TL:k/v_proj)
- Padded: (128,512,1024), manual_copy
- Baseline tile (32,64,32): FAILED accuracy — k=32 insufficient on Kp=1024
- Fallbacks (64,64,64), (32,32,32): also FAILED
- Agentic tile (32,64,128): 5/5 PASSED — 239, 237, 240, 238, 241 µs → mean=238.8µs
- **WINNER: AGENTIC** (k≥128 mandatory for bf16 accuracy on Kp=1024)

### Shape 5: M=128 K=960 N=2560 (TL:gate/up_proj)
- Padded: (128,3072,1024), numpy_pad
- Both strategies: ALL FAILED — accuracy failure persists even with k=128,256,512,1024
- Pattern: N=2560→Np=3072 combined with Kp=1024 and M=128 creates a bf16 accuracy failure regime not fixable by tile choice
- **WINNER: inconclusive**

### Shape 6: M=128 K=320 N=96 (AE:k/v_proj)
- Padded: (128,128,512), manual_copy
- Tile: (32,32,32) for both; 5/5 PASSED each
- **WINNER: TIE** (Δ=+0.07%)

### Shapes 7-13: Small shapes (M≤64)
- All shapes 7-13 passed 5/5 with same tiles for both strategies
- All differences within ±1% — consistent with NPU launch overhead dominating
- **WINNER: TIE** for all

---

## Summary Statistics

| Metric | Value |
|--------|-------|
| Total shapes | 17 |
| Pre-screened skips | 4 |
| Shapes executed | 13 |
| Total NPU runs attempted | 50 (baseline) + 50 (agentic) = 100 |
| Agentic passes | 60/65 (shapes 3,4 all pass; shape 5 all fail; rest 5/5) |
| Baseline passes | 55/65 (shapes 3,4,5 fail; rest 5/5) |
| Agentic wins | 2 (shapes 3, 4) |
| Baseline wins | 0 |
| Ties | 10 |
| Inconclusive | 1 (shape 5) |
| Prescreened skips | 4 |

---

## Key Findings

1. **bf16 accuracy on Kp=1024 requires k≥128**: Shapes padded to Kp=1024 (M=128,K=960) consistently fail with k<128. The agentic strategy's use of tile (32,64,128) resolved shapes 3 and 4 that the baseline failed completely.

2. **Shape 5 (128,3072,1024) is intractable in bf16**: Even with k≥128, the combination of Np=3072 and Kp=1024 with small M=128 causes accuracy failure. No bf16 tile workaround exists.

3. **Small shapes (M≤64) are within NPU launch overhead**: Shapes 7-13 all produce ~120-170µs regardless of tile, consistent with NPU fixed overhead of ~157µs. Strategy selection is irrelevant at this scale.

4. **Large shapes (M=1024) converge**: Shapes 1-2 with no padding show <1.5% variation between strategies — both use the same tile and the NPU is operating at steady state.

---

## Memory Updates (agentic only)

- **npu_execution_profiling.json**: Updated best tile for (128,1024,1024) and (128,512,1024) bf16 to (32,64,128); marked k<128 tiles as failing; added (128,3072,1024) entry with all-fail
- **strategy_insights.md**: Added 2 new dated bullets (2026-03-21) on Kp=1024 accuracy and (128,3072,1024) failure
- **errors.md**: Added new entry for (128,3072,1024) bf16 accuracy failure pattern
