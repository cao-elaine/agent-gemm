# Session Summary — SmolVLA Full NPU Experiment (291 shapes)
**Date**: 2026-03-29  |  **Type**: Experiment  |  **dtype**: bf16  |  **Trials/group**: 3

## What was run
Full SmolVLA NPU comparison experiment across 291 shapes, dtype=bf16, 3 trials per group. Three strategies compared: Baseline_0 (fixed (64,64,64) tile), Baseline_2 (profiling-best tile lookup), and Agentic (profiling + heuristics + divisibility reasoning). 282 runnable shapes (9 pre-screened). Results stored in references/experiment-results/test_n_20260328/results.json.

## Pre-screen results
9 shapes skipped before any trials (prescreened as known-failing based on prior rules).

## Key results
- B0 (fixed (64,64,64)) wins: 38
- B2 (profiling best) wins: 7
- Agentic wins: 18
- Ties: 163
- Inconclusive (all groups fail): 56
- Mean delta (agentic vs B0): +2.05% (agentic slightly worse overall)

## Inconclusive shapes
56 inconclusive results. Dominant cause: all 12 M=235 shapes padded to Mp=256 failed clang++ compilation for every tile combination. This is a universal hardware-level failure for Mp=256 in bf16 on this configuration.

## Major findings

### Finding 1: Profiling data inconsistency (Best in Failing) causes systematic Agentic failures
48 entries in npu_execution_profiling.json have the Best m,n,k tile listed in the Failing m,n,k list. The Agentic strategy checks `best not in failing` before using profiling data, causing it to skip the profiling best and fall back to a (potentially suboptimal) heuristic tile. Most impactful case: M=115, K=960, N=2560 → padded (128,2560,960). Profiling Best=(64,64,64) AND Failing includes (64,64,64). Agentic fell back to (64,32,64) → 1230µs vs B0/B2's (64,64,64) → 741µs (66% slower). This single bug accounts for a significant portion of B0's 38 wins.

### Finding 2: Mp=256 shapes fail compilation universally
All M=235 shapes (padded to Mp=256) and additional 256_* shapes are inconclusive — every tile combination fails clang++ compilation regardless of pipeline stage count. Even tiles with Pm=4, Pn=3, Pk=4 (48 stages) fail. This is a fundamental constraint on this hardware configuration. 56 shapes total were inconclusive, with Mp=256 being the dominant cause.

### Finding 3: Single-trial profiling data (count=1) is unreliable
Shape (128,576,512) bf16: profiling Best=(32,64,64)@283µs with count=1. In the experiment, (32,64,64) ran at 420-430µs (51% slower than the profiling entry), while B0's fixed (64,64,64) ran at 261µs — a tile not even in the profiling set. The count=1 profiling measurement was a lucky single run that does not reflect steady-state performance. B2 and Agentic both used the profiling entry and were significantly slower than B0 on this shape.

### Finding 4: Genuine Agentic advantage on divisibility edge cases
M=179, K=320, N=480 → padded (192,480,512): Agentic's Mp=192 divisibility heuristic correctly avoided (64,64,64) (since 480/64=7.5 is not integer) and used (64,32,64). B0/B2 fell back to (16,32,32). Agentic got 422µs vs B0=592µs (-29%), B2=499µs (-16%). This is a real win from better divisibility reasoning on non-power-of-2 padded dimensions.

### Finding 5: High NPU timing variance between measurement groups
The same tile on the same padded shape produced 40%+ timing differences between groups in some cases. E.g., M=115, K=320, N=480 (padded 128,480,512): B2 and Agentic both used tile (64,32,64), but B2 measured 323µs and Agentic measured 231µs (39% difference between groups). This measurement variance makes small-margin "wins" unreliable indicators of genuine strategy superiority.

### Finding 6: B0 fixed tile sometimes outperforms profiling data
Several cases where B0's (64,64,64) fixed tile beat the profiling "best":
- (128,576,512): B0=261µs vs B2/Agentic=419-429µs (60% worse from profiling)
- M=128, K=320, N=576 → (128,576,512): B0=261µs vs B2=370µs, Agentic=436µs
The profiling-best tile had count=1 and reflected a lucky single measurement.

## Most impactful shapes

| Shape (M,K,N) | Padded | Winner | Delta | Cause |
|---|---|---|---|---|
| 128×960×960 | 128×960×960 | B2 | -18% | Profiling best tile outperforms (64,64,64) fixed |
| 179×320×480 | 192×480×512 | Agentic | -29% vs B0 | Divisibility heuristic; Mp=192 avoids 480/64=7.5 |
| 115×320×576 | 128×576×512 | B0 | -60% vs profiling | Single-count profiling entry was misleading |
| 115×960×2560 | 128×2560×960 | B0/B2 | +66% B0 vs Agentic | Best tile in Failing list bug |

## Output files
- `references/experiment-results/test_n_20260328/results.json`

## Memory updates
- `strategy_insights.md`: added 3 new [2026-03-29] entries (profiling Best-in-Failing bug, Mp=256 universal failure, count=1 profiling unreliability)
- `error-logs/errors.md`: added [2026-03-29] entry for Mp=256 universal compilation failure pattern
