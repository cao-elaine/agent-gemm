# Session Summary — 2026-03-24 (SmolVLA Baseline vs Agentic)

## Task
SmolVLA NPU Baseline vs Agentic benchmark experiment. All 17 unique GEMM shapes from the SmolVLA model, dtype=bf16, 5 trials per group.

## Strategies compared
- **Baseline**: closest-fit padded shape + profiling-best tile (no k_min accuracy guard)
- **Agentic**: profiling + error logs + k_min accuracy guard + cost model fallback

## Results
- Agentic wins: 0
- Baseline wins: 2
- Ties: 11
- Prescreened skip: 4
- Mean delta (agentic − baseline): +0.89%

## Prescreened shapes
- K=3072 (shape 3, Vision mlp.fc2): bf16_K_limit
- K=2560 (shape 7, Text down_proj): bf16_K_limit
- M=50, K=960, N=96 (shape 8, AE q_proj): small_MN_large_K
- M=1, K=960, N=32 (shape 13, Proj state_proj): small_MN_large_K

## Key findings
1. **k_min over-conservative at Kp=512** (shape 9): k=32 passes accuracy at Kp=512; agentic rejection of profiling-best (32,32,32) in favor of (64,64,64) caused 6.9% slowdown. Corrected rule: k_min = max(32, Kp//16).
2. **Direct-mode K=960 passes accuracy with k=64** (shape 6): k_min guard applies to padded Kp, not to direct-mode K. In direct mode, fewer K-tiles means less accumulation error.
3. **Shape 4 "baseline win" is noise**: identical tiles (32,16,256) used by both groups; min NPU times are ~487µs for both; difference is outlier trials 3/4 in agentic group.
4. **Padding dominates on shape 4**: padding overhead 2.2× the NPU execution time (1225µs vs 546µs) for Text/q_proj and o_proj layers.
5. **7 overhead-limited shapes all tie**: all shapes with Mp≤64 produced means within ±2% as predicted.

## Output files written
- `references/experiment-results/test_n_20260324/results.json`
- `references/experiment-results/test_n_20260324/results.md`

## Memory updates
- `strategy_insights.md`: added k_min correction for Kp=512; added direct-mode accuracy note
- `errors.md`: added entry for direct-mode K=960 accuracy behavior
