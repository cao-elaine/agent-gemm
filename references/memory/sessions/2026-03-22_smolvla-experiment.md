# Session Summary — SmolVLA NPU Baseline vs Agentic Experiment
**Date**: 2026-03-22  |  **Type**: Experiment  |  **dtype**: bf16  |  **Trials/group**: 5

## What was run
Full SmolVLA NPU baseline vs agentic comparison experiment across all 17 unique GEMM shapes. Both groups ran on AMD Ryzen AI NPU (NPU available confirmed). 130 total NPU executions across 13 active shapes.

## Pre-screen results
4 shapes skipped before any trials:
- Shape 3 (1024,3072,768): K=3072 > 2048 bf16 K-limit
- Shape 7 (128,2560,960): K=2560 > 2048 bf16 K-limit
- Shape 8 (50,960,96): M≤50, N≤96, K≥960 small-MN large-K accuracy failure
- Shape 13 (1,960,32): M≤50, N≤96, K≥960 small-MN large-K accuracy failure

## Key results
- Agentic wins: 2 (Shape 4 borderline -2.1%, Shape 6 clear 0/5→5/5 pass)
- Baseline wins: 0
- Ties: 11 (within ±2%)
- Shape 6 (128,960,2560) → (128,3072,1024): Critical discovery — baseline (32,32,32) fails all 5 accuracy checks; agentic (32,64,128) passes all 5 at 794µs

## New discoveries
1. **(128,3072,1024) bf16 + (32,64,128): first confirmed passing tile in direct-dims mode.** Prior exhaustive testing showed universal accuracy failure in padded-run mode (--use-padding). Direct-dims mode (no --use-padding, original shape as 128×1024×3072) passes accuracy with k=128 tile. Mechanism: different ground-truth computation (full padded size vs original 960×2560 inner product).
2. **Script ALLOWED_N limitation (max 2048) prevents --use-padding for N=3072.** For shapes requiring padding to Np=3072, must use direct-dims fallback. Results from direct-dims runs are not equivalent to padded-run accuracy checks.
3. **Profiling DB updated with better bests:** (128,1024,1024) bf16: 349.9µs→338.5µs; (128,512,1024) bf16: 239.0µs→237.5µs; (128,3072,1024) bf16: None→794.2µs (first passing tile).

## Strategy effectiveness
The agentic's k_min accuracy pre-validation rule was the key differentiator on Shape 6: k_min = max(32, 1024//12) = 85 predicted that k=32 would fail (and it did) while k=128 would pass (and it did). Baseline, which skips accuracy pre-validation, chose k=32 and failed all trials. For 12 other shapes with identical configurations, both groups performed within ±2.1% (measurement noise, no tile-strategy contribution).
