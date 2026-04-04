# Session Summary: SmolVLA Full 291-Shape Experiment (bf16)
Date: 2026-03-30
Experiment: test_n_20260330

## Experiment Setup
- dtype: bf16
- trials: 3 per (shape, group)
- shapes: 291 unique GEMM shapes from SmolVLA model
- groups: Baseline 0 (fixed tile), Baseline 2 (profiling best), Agentic (full reasoning)
- output: references/experiment-results/test_n_20260330/

## Results Summary
- shapes_run: 282 (9 prescreened: K>2048 or small-MN-large-K accuracy failure)
- B0 wins: 21 (7.4%)
- B2 wins: 40 (14.2%)
- Agentic wins: 53 (18.8%)
- Ties: 133 (47.2%)
- Inconclusive: 35 (12.4%)

## Mean Latency Deltas
- AG vs B0: -2.84% (agentic faster on average)
- AG vs B2: +0.5% (essentially tied)
- B2 vs B0: -3.28% (profiling lookup beats fixed tile)

## Key Findings

### 1. Np=320 n=16 heuristic failure for Mp=192
For M=179,K=960,N=320 (padded to 192,320,960), the agentic strategy applied the
n=16 tile rule (64,16,64) designed for Np=320 shapes. Result: 615µs vs B2's 298µs
(+105.4% slower). Mp=192 had no direct profiling data, making the n=16 rule
counterproductive. Fix: only apply Np=320 n=16 rule when profiling data confirms it.

### 2. 35 inconclusive shapes (B0 failures on large-Np shapes)
B0 uses (16,32,32) for small-tile fallback, which fails on large Np=960 padded shapes
because 960//32=30 is not divisible by any of {3,4,5}. B2 and Agentic pass these
shapes but the shape is still marked inconclusive (can't compute B0 delta).
31 of 35 inconclusive shapes have this pattern.

### 3. Agentic override for 256_320_960 succeeds
M=235,K=960,N=320 is the only shape where B0 and B2 both fail while Agentic succeeds.
Agentic overrides Np from 320 to 384 (avoids col_num=5 accuracy bug), running at
(256,384,960) with (64,64,64). This is the clearest demonstration of agentic value.

### 4. Ties dominate (47.2%)
Most shapes produce very similar latency across all three strategies because:
- The profiling DB covers all 106 unique padded shapes (100% hit rate for bf16)
- When B2 and Agentic both find the same best tile from profiling, they tie
- Agentic adds value mainly for pathological shapes (Np=320, Mp=192 no-profile cases)

### 5. mp=192 shapes with no working tile profiling
11 shapes pad to Mp=192 padded shapes that have profiling entries but no recorded
working tiles. Agentic used heuristic interpolated from 128_* and 256_* data.
These shapes are candidates for explicit profiling to improve future performance.

## Files Updated
- references/memory/gemm-data/npu_execution_profiling.json (agentic trial results)
- references/memory/knowledge-base/strategy_insights.md (Np=320 n=16 failure note)
- references/experiment-results/test_n_20260330/results.json
- references/experiment-results/test_n_20260330/results.md
- references/experiment-results/test_n_20260330/plots/ (4 plots)
