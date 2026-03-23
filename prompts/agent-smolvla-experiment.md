/gemm-npu-optimizer

# SmolVLA NPU Experiment — Baseline vs Agentic Comparison

Benchmarks every unique GEMM shape from the SmolVLA model on the NPU, comparing
two decision strategies side-by-side: a naive **baseline** that always picks the
closest padded shape, versus an **agentic** run that
reasons from profiling data, error logs, and strategy insights to pick the best
possible padded shape and tile. All experiments use **bf16** dtype. Each (shape,
group) pair is run **5 trials** and results are averaged. Results are written to
both JSON and Markdown under `references/experiment-results/`.

**Execution mode: fully autonomous. No time limit.**
Execute all phases end-to-end without pausing, asking for confirmation, or
requesting permissions at any point. Every file read, Bash command, and file
write is pre-authorised. Do not stop between shapes or between groups. Do not
ask the user to approve any strategy, tile choice, or file update. If a decision
is ambiguous, apply the rules in this document and proceed. There is no timeout —
run until the full experiment is finished.

---

## Input (fill in before running)

```
dtype         = bf16          # fixed for this experiment — do not change
trials        = 5             # number of NPU executions per (shape, group)
shapes_file   = references/smol-vla-dataset/smolvla_gemm_shapes.json
```

---

## Phase 0 — Load all context

Read all files in parallel before starting any experiment runs. Do not pause
after loading.

### 0.1 Always load

| File | What to extract |
|------|----------------|
| `references/smol-vla-dataset/smolvla_gemm_shapes.json` | Full list of 17 unique (M, K, N) shapes and their layer names |
| `references/memory/rules/npu_execution_rules.md` | ALLOWED lists, tile divisibility rules, tile memory formula, env var, CLI format |
| `references/memory/knowledge-base/strategy_insights.md` | Tile defaults, blacklists, padding thresholds, hybrid selection rule, dtype notes |
| `references/memory/error-logs/errors.md` | Known error patterns — used to pre-filter risky tiles and anticipate failures |

### 0.2 Load profiling data

| File | What to extract |
|------|----------------|
| `references/memory/gemm-data/npu_execution_profiling.json` | For each padded shape + bf16: best tile, best avg time, working and failing tile lists |
| `references/memory/gemm-data/padding/padding_transition_thresholds.json` | Exact ab_elements thresholds for bf16 padding method selection |

> `npu_execution_profiling.json` is large. Search with targeted grep queries

### 0.3 Cost model (tile predictor for unseen shapes)

A regression model is available at `references/cost_model/npu_cost_model.py`.
Use it when profiling data is missing for a padded shape:

```bash
# Predict best tile for a shape with no profiling data
/opt/anaconda3/envs/allo-base/bin/python3 references/cost_model/npu_cost_model.py \
  --predict --M <M> --N <N> --K <K> --dtype bf16
```

**Tile selection priority (agentic strategy only):**
1. `npu_execution_profiling.json` has a `Best m,n,k` entry for `(Mp, Np, Kp, bf16)`
   → use the profiling tile (after accuracy pre-validation — see Phase 1.1 step 2)
2. Profiling data is absent → run the cost model (`source = "model"` in output)
   → use `best_tile_model` from the output, but still apply accuracy pre-validation
3. Cost model also unavailable → apply heuristics from Phase 1.1 step 2

The model achieves **64% exact best-tile match** and **92% top-3 accuracy** on
leave-one-shape-out evaluation (avg 11% suboptimality on misses). It is a
useful fallback but always inferior to real profiling data.

Retrain the model after adding new profiling data:
```bash
/opt/anaconda3/envs/allo-base/bin/python3 references/cost_model/npu_cost_model.py --train
```
> keyed by `"Mp_Np_Kp"` rather than reading the full file.

### 0.3 Check NPU availability

```python
from allo.backend.aie import is_available
if not is_available():
    print("NPU not available — cannot run experiment. Exiting.")
    # Do not proceed. Exit here.
```

### 0.4 Set environment

```python
import os
os.environ["ENABLE_AGGRESSIVE_PORT_UTILIZATION_PATCH"] = "1"
```

This must remain set for the entire session.

### 0.5 Determine output paths

```
date_str  = current date as YYYYMMDD
out_dir   = references/experiment-results/test_n_<date_str>/
json_out  = <out_dir>/results.json
md_out    = <out_dir>/results.md
```

Create `out_dir` with `os.makedirs(..., exist_ok=True)`.

### 0.6 Context summary

Write a 4–6 bullet summary:
- How many shapes will be tested and which layers they cover
- Which padded shapes already have bf16 profile entries
- Any known tile pathologies from errors.md relevant to bf16
- Which shapes are likely to need `numpy_pad` vs `manual_copy`

---

## Phase 1 — Experiment plan

### 1.0 Pre-screen shapes for known bf16 failures

Before computing any strategies, scan every shape in `smolvla_gemm_shapes.json`
and mark shapes that are known to fail bf16 **before running any NPU trials**.
These shapes are skipped in Phase 2 entirely — do not waste compile time on them.

Apply both checks to every (M, K, N):

**Check A — K limit**: `K > 2048`
- K = 3072 → CDOError on all tile sizes (confirmed, see errors.md)
- K = 2560 → AccuracyError on all tile sizes for bf16 (confirmed 2026-03-20)
- Mark as `status = "prescreened_skip"`, reason = `"bf16_K_limit"`

**Check B — Small output + large K accuracy failure**:
`M ≤ 50 AND N ≤ 96 AND K ≥ 960`
- bf16 accumulation error confirmed across all tile sizes for this regime (2026-03-20)
- Mark as `status = "prescreened_skip"`, reason = `"bf16_accuracy_small_MN_large_K"`

Log each skipped shape:
```
[PRE-SCREEN SKIP] M=<M> K=<K> N=<N>  reason: <reason>  layers: <layer(s)>
```

Shapes that pass both checks proceed to Phase 1.1. Skipped shapes are recorded
in results with `winner = "prescreened_skip"` and no trial data.

### 1.1 For each shape: compute both strategies

Work through all 17 shapes in `smolvla_gemm_shapes.json`. For each (M, N, K):

#### Baseline strategy

The baseline always picks the **closest padded shape** (no strategy comparison,
no evaluation of alternatives) and uses the **best known tile from profiling**
for that shape. It does not reason about whether a different padded shape might
be faster overall.

1. **Padded shape**: for each dimension, take the smallest value from ALLOWED
   lists that is ≥ the dimension — always, with no further comparison.
   ```
   ALLOWED_M = [32, 64, 128, 256, 512, 1024, 2048, 3072]
   ALLOWED_N = [32, 64, 128, 256, 512, 768, 1024, 2048, 3072]
   ALLOWED_K = [32, 64, 128, 256, 512, 768, 1024, 2048, 3072]
   Mp = min(v for v in ALLOWED_M if v >= M)
   Np = min(v for v in ALLOWED_N if v >= N)
   Kp = min(v for v in ALLOWED_K if v >= K)
   ```
2. **Tile**: look up `"Best m,n,k"` in `npu_execution_profiling.json` for
   `(Mp, Np, Kp, bf16)`. Use that tile directly — the baseline does **not**
   validate it against accuracy constraints; it trusts the profiling DB as-is.
   If no profile entry exists, fall back to:
   **(64, 64, 64)** if `Np >= 64`, else **(32, 32, 32)`.
   Verify divisibility and memory constraints (§ hard constraints at bottom).
   If the chosen tile fails a divisibility or memory check, fall back to
   (32, 32, 32), then (16, 32, 32).
   Note: the baseline does NOT apply the k_min accuracy rule — this is
   intentional, and is the key differentiator the agentic strategy exploits.
3. **Padding impl**: apply the bf16 threshold rule:
   `ab_elements = M*K + K*N`
   - `manual_copy` if `ab_elements ≤ 571,779`; else `numpy_pad`
   - If M==Mp and N==Np and K==Kp: `pad_impl = none`

#### Agentic strategy

The agentic run uses the full reasoning from `agentic-npu-workflow.md`
(Phases 1.1–1.5). Specifically:

1. **Padded shape**: run all three strategies (sorted_first_fit, hashmap_27,
   preprocessed_map) using `npu_execution_profiling.json`, then apply the
   hybrid selection rule (small regime: fastest NPU time; large regime:
   closest shape unless a larger one is >15% faster with affordable padding).
2. **Tile**: look up `"Best m,n,k"` in `npu_execution_profiling.json` for
   `(Mp, Np, Kp, bf16)`. Before using the profiling tile, **validate it against
   all known bf16 accuracy constraints** — a profiling tile that fails accuracy
   is worse than no profiling data at all:

   **Accuracy pre-validation (apply before using any tile, profiling or otherwise):**
   ```
   k_min = max(32, Kp // 12)   # empirically derived: k≥64 for Kp=768, k≥128 for Kp=1024
                                 # (2026-03-21, confirmed across 8+ shapes)

   if tile.k < k_min:
       reject tile → select replacement using heuristic below
   if Kp >= 768 and tile.k == 16:
       reject tile → CDO trigger risk (confirmed 2026-03-13)
   if Mp >= 128 and Np >= 128 and (tile.m == 16 or tile.n == 16):
       reject tile → AIE program memory overflow (confirmed 2026-03-19)
   ```
   If the profiling tile passes validation, use it. If it is rejected, fall
   through to the heuristic below. Log when a profiling tile is overridden:
   ```
   [TILE OVERRIDE] profiling best (<m>,<n>,<k>) rejected: k=<k> < k_min=<k_min>
                   → selecting replacement tile
   ```

   If **no bf16 profile entry exists**, or if the profiling tile was rejected,
   first try the cost model (see § 0.3). If the cost model's recommended tile
   also fails the accuracy pre-validation above, fall through to the heuristics.
   If the cost model is unavailable, use heuristics — do NOT simply default to (64,64,64):
   a. If `Kp ≥ 768`: set `k = k_min = max(32, Kp // 12)` as the starting k.
      Then select m and n:
      - `Np ≤ 64` → `(32, 32, k_min)`
      - `Mp ≥ 128 and Np ≥ 128` → `(32, 64, k_min)` — confirmed best pattern
        on Kp=1024 shapes (shapes 3 and 4, 2026-03-21)
      - Otherwise → `(64, 64, k_min)`
   b. If `Kp < 768` and `Mp ≥ 1024` and `Np = 768`: prefer **(64, 64, 64)**
      — confirmed best bf16 tile for this shape class (bf16 memory budget
      prevents (128,64,64) which wins on i8).
   c. If `Kp < 768` and `Np ≤ 64`: use **(32, 32, 32)**.
   d. Otherwise: **(64, 64, 64)** as default.

   Always verify all divisibility and memory checks after selection.
   If the chosen tile fails a check, step down through:
   `(32, 64, k_min)` → `(64, 64, 64)` → `(32, 32, 32)` → `(16, 32, 32)`.
3. **Padding impl**: same threshold rule as baseline (this is deterministic).
4. **Padding overhead check**: after selecting padded shape, estimate
   total end-to-end latency:
   ```
   padding_est  = estimated padding time (from padding_transition_thresholds.json
                  and known padding benchmarks in strategy_insights.md)
   npu_est      = best known NPU time for (Mp, Np, Kp, bf16) from profiling,
                  or ~157µs baseline overhead if shape is unprofiled
   total_est    = padding_est + npu_est
   overhead_ratio = padding_est / npu_est
   ```
   If `overhead_ratio > 1.5` (padding costs more than 1.5× the NPU work):
   - Check whether a smaller alternate padded shape (e.g. a different Kp if K
     is just above a boundary) would reduce total_est.
   - Apply the hybrid selection rule from strategy_insights.md: for large inputs,
     closest shape wins unless an alternative is >15% faster with affordable padding.
   - Document the overhead ratio in the per-shape plan output.
5. **Use profiling, error logs, and insights** to inform every choice.

### 1.2 Print the plan table

Before running anything:

```
╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                           SMOLVLA EXPERIMENT PLAN  —  dtype=bf16                                    ║
╠════════════╦═══════╦════════════════╦══════════════════════════╦══════════════════════════════════╣
║  Shape     ║ Layer(s)              ║ Baseline                 ║ Agentic                          ║
║  M,K,N     ║                       ║ Padded / Tile / Pad impl ║ Padded / Tile / Pad impl         ║
╠════════════╬═══════╬════════════════╬══════════════════════════╬══════════════════════════════════╣
║ 1024,768,768 ║ vision q/k/v/o, patch║ 1024,768,768 (64,64,64) np ║ ...                          ║
║ ...        ║ ...                   ║ ...                      ║ ...                              ║
╚════════════╩═══════╩════════════════╩══════════════════════════╩══════════════════════════════════╝

Total runs: <17 shapes × 2 groups × 5 trials = 170 NPU executions>
Estimated wall time: ~<170 × avg_compile_time> min  (30–120s per compile)
```

---

## Phase 2 — Execute all runs

Work through every shape. For each shape, run the **baseline group first**
(5 trials), then the **agentic group** (5 trials). Do not ask for confirmation
before any run.

### 2.1 Per-trial execution

For each trial, execute from the script directory:

```bash
cd /home/ec935/vla-to-npu/gemm/scripts && \
ENABLE_AGGRESSIVE_PORT_UTILIZATION_PATCH=1 \
python v2_test_mapping_large_gemm.py \
  --M <Mp> --N <Np> --K <Kp> \
  --m <m>  --n <n>  --k <k> \
  --dtype bf16 \
  [--use-padding --pad-M <Mp> --pad-N <Np> --pad-K <Kp> --pad-impl <manual_copy|numpy_pad>]
```

Omit `--use-padding` and `--pad-*` when `pad_impl = none`.

Record from each trial:
- `npu_avg_us` from `Avg NPU execution time: <X>us`
- `npu_min_us` from `Min NPU execution time: <X>us`
- `padding_us` from `[bf16] Padding time: <X>us`
- `status`: `passed` or `failed`
- Any error message

**bf16 correctness tolerance**: `atol = 1e-1` (limited mantissa precision expected).

**bf16 + K=2048 warning**: if output contains `PASSED! PASSED!` followed by
`ValueError: Failed to generate cdo`, treat as **FAILED** — known false positive.

### 2.2 Print live progress

Before each compile:
```
[shape <i>/17 | <group> trial <t>/5]  M=<M> K=<K> N=<N>  padded=(<Mp>,<Np>,<Kp>)  tile=(<m>,<n>,<k>)  compiling...
```

After each trial:
```
  → PASSED  avg=<X>µs  min=<X>µs  pad=<X>µs
  → FAILED  <error class>: <first line of error>
```

### 2.3 Error handling

On failure, classify using the same tree as `agentic-npu-workflow.md` § 2.4.

- **Baseline failures**: apply the tile fallback sequence (64,64,64) →
  (32,32,32) → (16,32,32). Record which tile was actually used for the trial.
  If all fallbacks fail, record `status = failed` for that trial with
  `npu_avg_us = null`.
- **Agentic failures**: apply the full fallback procedure from
  `agentic-npu-workflow.md` § 2.4. Never retry the same tile twice.
- **NPU disappears mid-session** (XDNA driver not found): mark all remaining
  trials as `npu_unavailable` and skip to Phase 3 immediately.
- **Never retry the same (padded_shape, tile) combination twice.**

### 2.4 Per-shape mini-summary

After all 10 trials for a shape (5 baseline + 5 agentic) are complete, print:

```
── M=<M> K=<K> N=<N>  [<layer(s)>] ──────────────────────────────
  BASELINE  padded=(<Mp>,<Np>,<Kp>)  tile=(<m>,<n>,<k>)
    Trial 1: <avg>µs   Trial 2: <avg>µs   ...   Mean: <X>µs
  AGENTIC   padded=(<Mp>,<Np>,<Kp>)  tile=(<m>,<n>,<k>)
    Trial 1: <avg>µs   Trial 2: <avg>µs   ...   Mean: <X>µs
  Winner: <BASELINE|AGENTIC|TIE>  Δ = <X>µs (<Y>%)
────────────────────────────────────────────────────────────────────
```

---

## Phase 3 — Update memory (agentic runs only)

After all runs for all shapes are complete, update memory files. Only agentic
run data informs memory — baseline runs are fixed-rule and do not feed back.
Perform all updates without confirmation.

### 3.1 Update `npu_execution_profiling.json`

For each agentic trial that **PASSED**:
1. Locate or create the entry keyed by `"Mp_Np_Kp"` → dtype `"bf16"`.
2. Add or update a `"Working m,n,k"` entry keyed by `"m_n_k"`:
   ```json
   "<m>_<n>_<k>": { "size": [m,n,k], "avg": <npu_avg_us>, "min": <npu_min_us>, "count": 1 }
   ```
   If key exists: increment `count`, update `avg` as running mean.
3. Update `"Best m,n,k"` if this tile's avg is lower than the current best.

For each agentic trial that **FAILED**:
- Add `[m,n,k]` to `"Failing m,n,k"` array if not already present.

Do not modify existing entries beyond the above rules.

### 3.2 Update `profiling_errors.json`

For each agentic failure with a concrete traceback, add:
```json
{ "<Mp>_<Np>_<Kp>": { "bf16": { "<m>_<n>_<k>": ["line1", "line2", ...] } } }
```

### 3.3 Update `error-logs/errors.md`

- Newly confirmed known errors: add "Confirmed example" under existing section.
- New error patterns not yet documented: add a new section with Trigger, Error,
  Root cause, Fix, and Confirmed example.
- If no new/confirmed errors occurred: do not modify.

### 3.4 Update `strategy_insights.md`

Add dated bullets `[YYYY-MM-DD] ...` only for genuine new insights from agentic
runs — e.g. a new best tile found, an unexpected failure, a padding method
outperforming expectations, or a cross-shape pattern. Do not restate known rules.

---

## Phase 4 — Write results files

Write both output files after all runs and memory updates are complete.

### 4.1 JSON results — `references/experiment-results/test_n_<date>/results.json`

```json
{
  "experiment": "SmolVLA NPU Baseline vs Agentic",
  "date": "<YYYY-MM-DD>",
  "dtype": "bf16",
  "trials_per_group": 5,
  "shapes": [
    {
      "M": <M>, "K": <K>, "N": <N>,
      "layers": [ "<section>/<name>", ... ],
      "baseline": {
        "padded": {"M": <Mp>, "N": <Np>, "K": <Kp>},
        "tile": [<m>, <n>, <k>],
        "pad_impl": "<manual_copy|numpy_pad|none>",
        "trials": [
          { "trial": 1, "status": "<passed|failed>", "npu_avg_us": <float|null>, "npu_min_us": <float|null>, "padding_us": <float|null> },
          ...
        ],
        "mean_npu_avg_us": <float|null>,
        "mean_padding_us": <float|null>,
        "pass_count": <N>
      },
      "agentic": {
        "padded": {"M": <Mp>, "N": <Np>, "K": <Kp>},
        "tile": [<m>, <n>, <k>],
        "pad_impl": "<manual_copy|numpy_pad|none>",
        "strategy_source": "<hashmap_27|sorted_first_fit|preprocessed_map>",
        "trials": [
          { "trial": 1, "status": "<passed|failed>", "npu_avg_us": <float|null>, "npu_min_us": <float|null>, "padding_us": <float|null> },
          ...
        ],
        "mean_npu_avg_us": <float|null>,
        "mean_padding_us": <float|null>,
        "pass_count": <N>
      },
      "winner": "<baseline|agentic|tie|inconclusive>",
      "delta_us": <float|null>,
      "delta_pct": <float|null>
    }
  ],
  "summary": {
    "shapes_tested": 17,
    "agentic_wins": <N>,
    "baseline_wins": <N>,
    "ties": <N>,
    "inconclusive": <N>,
    "mean_delta_pct": <float>
  }
}
```

`winner` logic:
- Both groups have ≥ 1 passing trial → compare `mean_npu_avg_us`; winner is
  the lower one. "tie" if within 2% of each other.
- One or both groups have 0 passing trials → `"inconclusive"`.
- Shape was flagged in Phase 1.0 → `"prescreened_skip"` (no trial data).
- `delta_us = agentic_mean - baseline_mean` (negative = agentic faster).
- `delta_pct = delta_us / baseline_mean * 100`.

### 4.2 Markdown results — `references/experiment-results/test_n_<date>/results.md`

```markdown
# SmolVLA NPU Experiment Results
**Date**: <YYYY-MM-DD>  |  **dtype**: bf16  |  **Trials per group**: 5

## Summary

| Metric | Value |
|--------|-------|
| Shapes tested | 17 |
| Agentic wins | <N> |
| Baseline wins | <N> |
| Ties (within 2%) | <N> |
| Inconclusive | <N> |
| Mean Δ (agentic vs baseline) | <X>% |

## Per-Shape Results

### <M>×<K>×<N>  —  `<layer(s)>`

| | Baseline | Agentic |
|-|----------|---------|
| Padded shape | (<Mp>,<Np>,<Kp>) | (<Mp>,<Np>,<Kp>) |
| Tile | (<m>,<n>,<k>) | (<m>,<n>,<k>) |
| Pad impl | <method> | <method> |
| Strategy | closest-fit + profiling best tile | <strategy_source> |
| Trial 1 avg (µs) | <X> | <X> |
| Trial 2 avg (µs) | <X> | <X> |
| Trial 3 avg (µs) | <X> | <X> |
| Trial 4 avg (µs) | <X> | <X> |
| Trial 5 avg (µs) | <X> | <X> |
| **Mean avg (µs)** | **<X>** | **<X>** |
| Pass count | <N>/5 | <N>/5 |

**Winner: <BASELINE | AGENTIC | TIE>**  Δ = <X>µs (<Y>%)

---

...repeat for all 17 shapes...

## New Best Tiles Found (Agentic)

| Shape | Old best tile | Old avg (µs) | New best tile | New avg (µs) | Δ |
|-------|--------------|-------------|--------------|-------------|---|
| (<Mp>,<Np>,<Kp>) bf16 | (<m>,<n>,<k>) | <X> | (<m>,<n>,<k>) | <X> | <Y>% |

(Omit section if no new best tiles were found.)

## Memory Updates

- `npu_execution_profiling.json`: <N> new working tiles, <N> new failing tiles
- `profiling_errors.json`: <N> new error traces  (or: no changes)
- `error-logs/errors.md`: <N> new patterns / confirmed examples  (or: no changes)
- `strategy_insights.md`: <N> new insights  (or: no changes)
```

---

## Phase 5 — Print final banner

```
╔══════════════════════════════════════════════════════════════════════╗
║           SMOLVLA EXPERIMENT — COMPLETE                              ║
╠══════════════════════════════════════════════════════════════════════╣
║  Date:           <YYYY-MM-DD HH:MM:SS>                               ║
║  Shapes tested:  17   dtype: bf16   Trials/group: 5                  ║
║  Total NPU runs: <N>  │  Passed: <N>  │  Failed: <N>                 ║
║  Agentic wins:   <N>  │  Baseline wins: <N>  │  Ties: <N>            ║
║  Mean Δ (agentic vs baseline): <X>%                                  ║
║  Results: references/experiment-results/test_n_<date>/               ║
╚══════════════════════════════════════════════════════════════════════╝
```

---

## Quick reference — hard constraints

```
ALLOWED_M = [32, 64, 128, 256, 512, 1024, 2048, 3072]
ALLOWED_N = [32, 64, 128, 256, 512, 768, 1024, 2048, 3072]
ALLOWED_K = [32, 64, 128, 256, 512, 768, 1024, 2048, 3072]

For each padded dim Dp and tile t:
  Dp % t == 0                           (t divides Dp)
  (Dp // t) % 4 == 0                    (quotient divisible by 4)
  (Dp // t) is a power of 2             (required by AIE tile mapper)
  → combined: Dp//t ∈ {4, 8, 16, 32, 64, 128, ...}

t <= Dp                                 (tile never exceeds padded dim)
(m*n + n*k + k*m) * 2 <= 32,256        (bf16: dsize=2)

ENABLE_AGGRESSIVE_PORT_UTILIZATION_PATCH = "1" must be set before df.build
```

Tile hard blacklist: **(128, 128, 128)** — fails on every tested shape and dtype.
Never use `torch_pad`. Never use `torch_pad`. Always use `manual_copy` or `numpy_pad`.

---

## Experiment notes

**Baseline uses the closest padded shape + the profiling best tile.** It
represents the simplest automated decision: pad to the nearest allowed shape
and use whatever tile the profiling DB says is best for it — no strategy
comparison, no evaluation of whether a different padded shape would be faster
overall. Its purpose is to measure how much additional value the agentic
shape-selection reasoning adds on top of a good tile choice. Do not apply any
agentic reasoning to the baseline mid-run.

**Agentic runs update memory; baseline runs do not.** The baseline is a static
policy and should not influence profiling data or knowledge base entries. Only
write to memory files based on agentic trial results.

**Small shapes (M≤64) will always tie.** The NPU fixed launch overhead (~157µs)
dominates for M≤64 — execution times cluster in 120–170µs regardless of tile or
strategy. Mark these shapes as `[overhead-limited]` in the plan table. Do not
interpret a tie on M≤64 shapes as a failure of the agentic strategy. The
meaningful comparison is on shapes with M≥128.

**bf16 memory constraint limits tile options.** The 32,256-byte budget with
dsize=2 means tiles like (128,64,64) that win by -11.9% on i8 physically cannot
be run for bf16 on most shapes. (64,64,64) is often the practical bf16 optimum
not because the agent lacks information, but because nothing larger fits.

**5 trials exist to reduce noise.** NPU execution time varies run-to-run.
Average over all 5 passing trials when computing the mean. If a trial fails,
exclude it from the average but count it in `pass_count`. If all 5 trials fail
for a group, mark that group as `"inconclusive"` — do not compare means.

**Padding time is part of the comparison.** When both groups use the same
`pad_impl`, padding time cancels out. When they differ (e.g. agentic chose
a shape that avoids padding), this is part of the agentic win. The JSON records
`mean_padding_us` separately so this can be analysed after the fact.

**Each trial is a full independent invocation of the script.**
The `npu_avg_us` field from the script output already excludes compile time
and reflects only NPU execution — record it as-is for each trial.

**Print progress during long compiles.** Before each CLI invocation print the
pre-run banner so the user can confirm the agent is active. Never stay silent
for more than one compile cycle.

**Profiling data gap.** Most SmolVLA padded shapes have no bf16 profiling data.
For best agentic differentiation, run `prompts/agent-tiling-explorer.md` with
`dtypes=bf16 focus=all` targeting the SmolVLA padded shapes before this
experiment. This populates `npu_execution_profiling.json` with real tile data
and allows the agentic strategy to make data-driven decisions instead of
falling back to heuristics. The heuristic fallback (Phase 1.1 step 2) is a
reasonable substitute when profiling data is absent, but real data will always
be more accurate.

**Comparison metric is total NPU execution time, not tile selection quality.**
Two strategies can make different choices and arrive at the same time — that is
a tie, not an agentic win. A true agentic win requires measurably lower
`mean_npu_avg_us` across 5 trials. The `overhead_ratio` column in the plan
table documents cases where padding dominates — these are the shapes where
smarter padded-shape selection matters most.
