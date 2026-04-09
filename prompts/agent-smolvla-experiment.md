/gemm-npu-optimizer

# SmolVLA NPU Experiment — Baseline vs Agentic Comparison

Benchmarks every unique GEMM shape from the SmolVLA model on the NPU, comparing
three decision strategies side-by-side:

- **Baseline 0** — fixed tile only: always uses (64,64,64) or (32,32,32) with
  no profiling lookup. Represents the floor — what you get with zero data.
- **Baseline 1** — profiling-best tile: closest-fit padded shape + best tile
  from `npu_execution_profiling.json`. Fails gracefully when data is missing.
- **Agentic** — full reasoning: padded shape selection across three strategies,
  k_min accuracy pre-validation, cost model fallback, and strategy insights.

All experiments use **bf16** dtype. Each (shape, group) pair is run **3 trials**
and results are averaged. Results are written to both JSON and Markdown under
`references/experiment-results/`.

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
trials        = 3             # number of NPU executions per (shape, group)
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

### 0.5 Determine output paths and resume state

```
date_str  = current date as YYYYMMDD
out_dir   = references/experiment-results/test_n_<date_str>/
json_out  = <out_dir>/results.json
md_out    = <out_dir>/results.md
```

Create `out_dir` with `os.makedirs(..., exist_ok=True)`.

**Resume logic**: if `results.json` already exists in `out_dir`, load it and
extract the set of already-completed shapes:

```python
import json, os
completed = set()
if os.path.exists(json_out):
    existing = json.load(open(json_out))
    for s in existing.get('shapes', []):
        if s.get('winner') not in (None,):   # any recorded result counts
            completed.add((s['M'], s['K'], s['N']))
    print(f"[RESUME] Found {len(completed)} already-completed shapes in {json_out}")
    print(f"[RESUME] Remaining: {len(all_shapes) - len(completed)} shapes to run")
```

Skip any shape in Phase 2 whose `(M, K, N)` is in `completed`. Preserve existing
results in the JSON — do not overwrite them. Append only new results.

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

### 1.1 For each shape: compute all three strategies

Work through all 17 shapes in `smolvla_gemm_shapes.json`. For each (M, N, K):

#### Baseline 0 strategy — fixed tile, no data lookup

The simplest possible policy: pick the closest padded shape and use a fixed
default tile with no profiling lookup whatsoever.

1. **Padded shape**: same closest-fit rule as Baseline 1 (see below).
2. **Tile**: always **(64, 64, 64)** unless any padded dimension < 64, in which
   case use **(32, 32, 32)**. No profiling lookup, no cost model, no heuristics.
   Verify divisibility (§ hard constraints). If the chosen tile fails divisibility,
   step down: (64,64,64) → (32,32,32) → (16,32,32).
   Do NOT apply k_min or any accuracy rule — Baseline 0 is intentionally naive.
3. **Padding impl**: same threshold rule as all other strategies:
   `ab_elements = M*K + K*N`
   - `manual_copy` if `ab_elements ≤ 571,779`; else `numpy_pad`
   - If M==Mp and N==Np and K==Kp: `pad_impl = none`

#### Baseline 1 strategy — profiling best tile

Picks the closest padded shape and uses the best tile from profiling data. Does
not reason about whether a different padded shape might be faster overall.

1. **Padded shape**: for each dimension, take the smallest value from ALLOWED
   lists that is ≥ the dimension — always, with no further comparison.
   ```
   ALLOWED_M = [32, 64, 96, 128, 160, 192, 256, 288, 320, 384, 448, 480, 512, 576, 640, 672, 960, 1024, 1152, 1280, 1344, 1536, 1600, 1728, 1792, 1920, 2048, 2112, 2240, 2304, 2496, 2560, 2688, 2816, 2880, 3072, 3200, 3264, 3328, 3456, 3584, 3648, 3840, 4032, 4096]
   ALLOWED_N = [32, 64, 96, 128, 160, 192, 256, 288, 320, 384, 448, 480, 512, 576, 640, 672, 768, 960, 1024, 1152, 1280, 1344, 1536, 1600, 1728, 1792, 1920, 2048, 2112, 2240, 2304, 2496, 2560, 2688, 2816, 2880, 3072, 3200, 3264, 3328, 3456, 3584, 3648, 3840, 4032, 4096]
   ALLOWED_K = [32, 64, 128, 256, 512, 768, 960, 1024, 2048, 3072]
   Mp = min(v for v in ALLOWED_M if v >= M)
   Np = min(v for v in ALLOWED_N if v >= N)
   Kp = min(v for v in ALLOWED_K if v >= K)
   ```
2. **Tile**: look up `"Best m,n,k"` in `npu_execution_profiling.json` for
   `(Mp, Np, Kp, bf16)`. Use that tile directly — Baseline 1 does **not**
   validate it against accuracy constraints; it trusts the profiling DB as-is.
   If no profile entry exists, fall back to:
   **(64, 64, 64)** if `Np >= 64`, else **(32, 32, 32)**.
   Verify divisibility and memory constraints (§ hard constraints at bottom).
   If the chosen tile fails a divisibility or memory check, fall back to
   (32, 32, 32), then (16, 32, 32).
3. **Padding impl**: same threshold rule as above.

#### Agentic strategy

The agent makes all decisions autonomously using whatever data and insights are
available. There is no fixed algorithm to follow — the agent reads the profiling
DB, knowledge base, error logs, and cost model output, then chooses the padded
shape and tile it believes will be fastest for this specific shape.

Available information (loaded in Phase 0):
- `npu_execution_profiling.json` — real measured latencies for (padded shape, tile, dtype)
- `strategy_insights.md` — accumulated patterns and rules from prior sessions
- `errors.md` — known failure modes to avoid
- Cost model — tile predictor for shapes with no profiling data

The agent should reason about:
- Which padded shape (from ALLOWED lists) minimises both padding waste and NPU time
- Which tile the data suggests will be fastest for that padded shape
- Whether the profiling DB has data for this shape, and if so what it says
- Any patterns from strategy_insights.md relevant to this shape's dimensions

**Constraints** (non-negotiable):
- Padded dims must be in ALLOWED_M / ALLOWED_N / ALLOWED_K
- Tile must satisfy divisibility, bundling constraint, and memory budget (§ hard constraints)
- If no profiling data exists for a shape, use the cost model first (Phase 0.3).
  If the cost model also returns no useful prediction, reason from first principles:
  - Enumerate valid tiles (divisibility + bundling + memory budget all satisfied)
  - For overhead-limited shapes (Mp ≤ 64): tile choice has low impact; pick the largest
    valid square tile that fits (e.g. m=Mp if Mp%3=0 or auto-adjust applies)
  - For medium shapes (64 < Mp ≤ 256): prefer larger m (reduces Pm, fewer loop iterations)
  - Prefer square-ish tiles; avoid m=16 unless forced (low parallelism for M≤128)
  - Never emit a bare heuristic fallback without explaining the reasoning in `reasoning_summary`

**Padding impl**: deterministic threshold rule — same as all other strategies:
`ab_elements = M*K + K*N`; `manual_copy` if ≤ 571,779, else `numpy_pad`; `none` if no padding needed.

### 1.2 Print the plan table

Before running anything:

```
╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                                    SMOLVLA EXPERIMENT PLAN  —  dtype=bf16                                           ║
╠══════════════╦═══════════════════╦════════════════════════════╦════════════════════════════╦════════════════════════╣
║  Shape M,K,N ║ Layer(s)          ║ Baseline 0 (fixed tile)    ║ Baseline 1 (profiling)     ║ Agentic                ║
║              ║                   ║ Padded / Tile / Impl       ║ Padded / Tile / Impl       ║ Padded / Tile / Impl   ║
╠══════════════╬═══════════════════╬════════════════════════════╬════════════════════════════╬════════════════════════╣
║ 1024,768,768 ║ vision q/k/v/o    ║ 1024,768,768 (64,64,64) np ║ 1024,768,768 (64,64,64) np ║ ...                   ║
║ ...          ║ ...               ║ ...                        ║ ...                        ║ ...                   ║
╚══════════════╩═══════════════════╩════════════════════════════╩════════════════════════════╩════════════════════════╝

Total runs: <291 shapes × 3 groups × 3 trials = 2619 NPU executions>
Estimated wall time: ~<2619 × avg_compile_time> min  (30–120s per compile)
```

---

## Phase 2 — Execute all runs

Work through every shape. For each shape, run **Baseline 0 first** (3 trials),
then **Baseline 1** (3 trials), then the **agentic group** (3 trials). Do not
ask for confirmation before any run.

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
- `padding_us` from `[bf16] Padding time: <X>us` (0 if pad_impl=none)
- `unpadding_us` from `[bf16] Unpadding time: <X>us` (0 if pad_impl=none)
- `status`: `passed` or `failed`
- Any error message

**Timing includes padding + NPU execution + unpadding only.**
Do NOT include profiling lookup time, cost model inference time, agent reasoning
time, or compilation time in any measurement.
`total_us = padding_us + npu_avg_us + unpadding_us` is the comparison metric.
`npu_avg_us` and `padding_us` come directly from the script output and already
exclude everything else — record them as-is.

**bf16 correctness tolerance**: `atol = 1e-1` (limited mantissa precision expected).

**bf16 + K=2048 warning**: if output contains `PASSED! PASSED!` followed by
`ValueError: Failed to generate cdo`, treat as **FAILED** — known false positive.

### 2.2 Print live progress

Before each compile:
```
[shape <i>/291 | <group> trial <t>/3]  M=<M> K=<K> N=<N>  padded=(<Mp>,<Np>,<Kp>)  tile=(<m>,<n>,<k>)  compiling...
# <group> is one of: baseline_0 | baseline_1 | agentic
```

After each trial:
```
  → PASSED  avg=<X>µs  min=<X>µs  pad=<X>µs
  → FAILED  <error class>: <first line of error>
```

### 2.3 Error handling

On failure, classify using the same tree as `agentic-npu-workflow.md` § 2.4.

- **Baseline 0 failures**: apply fallback (32,32,32) → (16,32,32). Record tile used.
- **Baseline 1 failures**: apply the tile fallback sequence (64,64,64) →
  (32,32,32) → (16,32,32). Record which tile was actually used for the trial.
  If all fallbacks fail, record `status = failed` for that trial with
  `npu_avg_us = null`.
- **Agentic failures**: apply the full fallback procedure from
  `agentic-npu-workflow.md` § 2.4. Never retry the same tile twice.
- **NPU disappears mid-session** (XDNA driver not found): mark all remaining
  trials as `npu_unavailable` and skip to Phase 3 immediately.
- **Never retry the same (padded_shape, tile) combination twice.**

### 2.4 Per-shape mini-summary

After all 9 trials for a shape (3 × 3 groups) are complete, print:

```
── M=<M> K=<K> N=<N>  [<layer(s)>] ──────────────────────────────
  BASELINE_0  padded=(<Mp>,<Np>,<Kp>)  tile=(<m>,<n>,<k>)  [fixed]
    Trial 1: npu=<X>µs pad=<P>µs total=<T>µs  ...  Mean total: <X>µs
  BASELINE_1  padded=(<Mp>,<Np>,<Kp>)  tile=(<m>,<n>,<k>)  [profiling]
    Trial 1: npu=<X>µs pad=<P>µs total=<T>µs  ...  Mean total: <X>µs
  AGENTIC     padded=(<Mp>,<Np>,<Kp>)  tile=(<m>,<n>,<k>)
    Trial 1: npu=<X>µs pad=<P>µs total=<T>µs  ...  Mean total: <X>µs
  Best: <BASELINE_0|BASELINE_1|AGENTIC>  (by mean_total_us)
  vs B0: Δ(B1)=<X>µs (<Y>%)   Δ(AG)=<X>µs (<Y>%)
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

**JSON serialization rule**: before calling `json.dump`, replace every `float('nan')`
and `math.nan` value with `None` (serializes as JSON `null`). Never write bare `NaN`
to JSON — it is not valid JSON and breaks parsers.
```python
import math
def sanitize(obj):
    if isinstance(obj, float) and math.isnan(obj): return None
    if isinstance(obj, dict): return {k: sanitize(v) for k,v in obj.items()}
    if isinstance(obj, list): return [sanitize(v) for v in obj]
    return obj
json.dump(sanitize(data), f, indent=2)
```

### 4.1 JSON results — `references/experiment-results/test_n_<date>/results.json`

```json
{
  "experiment": "SmolVLA NPU Baseline vs Agentic",
  "date": "<YYYY-MM-DD>",
  "dtype": "bf16",
  "trials_per_group": 3,
  "shapes": [
    {
      "M": <M>, "K": <K>, "N": <N>,
      "layers": [ "<section>/<name>", ... ],
      "baseline_0": {
        "padded": {"M": <Mp>, "N": <Np>, "K": <Kp>},
        "tile": [<m>, <n>, <k>],
        "pad_impl": "<manual_copy|numpy_pad|none>",
        "tile_source": "fixed",
        "trials": [
          { "trial": 1, "status": "<passed|failed>", "npu_avg_us": <float|null>, "npu_min_us": <float|null>, "padding_us": <float|null>, "unpadding_us": <float|null> }
        ],
        "mean_npu_avg_us": <float|null>,
        "mean_padding_us": <float|null>,
        "mean_unpadding_us": <float|null>,
        "mean_total_us": <float|null>,
        "pass_count": <N>
      },
      "baseline_1": {
        "padded": {"M": <Mp>, "N": <Np>, "K": <Kp>},
        "tile": [<m>, <n>, <k>],
        "pad_impl": "<manual_copy|numpy_pad|none>",
        "tile_source": "<profiling|fallback_fixed>",
        "trials": [
          { "trial": 1, "status": "<passed|failed>", "npu_avg_us": <float|null>, "npu_min_us": <float|null>, "padding_us": <float|null>, "unpadding_us": <float|null> }
        ],
        "mean_npu_avg_us": <float|null>,
        "mean_padding_us": <float|null>,
        "mean_unpadding_us": <float|null>,
        "mean_total_us": <float|null>,
        "pass_count": <N>
      },
      "agentic": {
        "padded": {"M": <Mp>, "N": <Np>, "K": <Kp>},
        "tile": [<m>, <n>, <k>],
        "pad_impl": "<manual_copy|numpy_pad|none>",
        "reasoning_summary": "<one sentence: why this padded shape and tile>",
        "tile_source": "<profiling|model|heuristic>",
        "trials": [
          { "trial": 1, "status": "<passed|failed>", "npu_avg_us": <float|null>, "npu_min_us": <float|null>, "padding_us": <float|null>, "unpadding_us": <float|null> }
        ],
        "mean_npu_avg_us": <float|null>,
        "mean_padding_us": <float|null>,
        "mean_unpadding_us": <float|null>,
        "mean_total_us": <float|null>,
        "pass_count": <N>
      },
      "winner": "<baseline_0|baseline_1|agentic|tie|inconclusive|prescreened_skip>",
      "delta_b1_vs_b0_us": <float|null>,
      "delta_b1_vs_b0_pct": <float|null>,
      "delta_ag_vs_b0_us": <float|null>,
      "delta_ag_vs_b0_pct": <float|null>,
      "delta_ag_vs_b1_us": <float|null>,
      "delta_ag_vs_b1_pct": <float|null>
    }
  ],
  "summary": {
    "shapes_tested": 17,
    "winner_counts": {"baseline_0": <N>, "baseline_1": <N>, "agentic": <N>, "tie": <N>, "inconclusive": <N>},
    "mean_delta_ag_vs_b0_pct": <float>,
    "mean_delta_ag_vs_b1_pct": <float>,
    "mean_delta_b1_vs_b0_pct": <float>
  }
}
```

`winner` logic:
- All three groups with ≥ 1 passing trial → winner is the group with lowest
  `mean_total_us` (= mean_padding_us + mean_npu_avg_us + mean_unpadding_us).
  "tie" if top two are within 2% of each other.
- For shapes with pad_impl=none in all groups: mean_total_us = mean_npu_avg_us.
- Any group has 0 passing trials → that group is `"inconclusive"` for that shape.
- Shape was flagged in Phase 1.0 → `"prescreened_skip"` (no trial data).
- All `delta_*` fields compare `mean_total_us`; use the reference group as
  denominator; negative = faster.

### 4.2 Markdown results — `references/experiment-results/test_n_<date>/results.md`

```markdown
# SmolVLA NPU Experiment Results
**Date**: <YYYY-MM-DD>  |  **dtype**: bf16  |  **Trials per group**: 3

## Summary

| Metric | Value |
|--------|-------|
| Shapes tested | 17 |
| Baseline 0 wins | <N> |
| Baseline 1 wins | <N> |
| Agentic wins | <N> |
| Ties (within 2%) | <N> |
| Inconclusive | <N> |
| Mean Δ agentic vs Baseline 0 | <X>% |
| Mean Δ agentic vs Baseline 1 | <X>% |
| Mean Δ Baseline 1 vs Baseline 0 | <X>% |

## Per-Shape Results

### <M>×<K>×<N>  —  `<layer(s)>`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (<Mp>,<Np>,<Kp>) | (<Mp>,<Np>,<Kp>) | (<Mp>,<Np>,<Kp>) |
| Tile | (<m>,<n>,<k>) | (<m>,<n>,<k>) | (<m>,<n>,<k>) |
| Tile source | fixed | profiling/fallback | <source> |
| Pad impl | <method> | <method> | <method> |
| Trial 1 total (µs) | <X> | <X> | <X> |
| Trial 2 total (µs) | <X> | <X> | <X> |
| Trial 3 total (µs) | <X> | <X> | <X> |
| Mean NPU (µs) | <X> | <X> | <X> |
| Mean pad (µs) | <X> | <X> | <X> |
| **Mean total (µs)** | **<X>** | **<X>** | **<X>** |
| Pass count | <N>/3 | <N>/3 | <N>/3 |

**Winner: <BASELINE_0 \| BASELINE_1 \| AGENTIC \| TIE>**
Δ(B2 vs B0)=<X>µs (<Y>%)   Δ(AG vs B0)=<X>µs (<Y>%)   Δ(AG vs B1)=<X>µs (<Y>%)

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

## Phase 5 — Generate plots

After writing results files, generate comparison plots using matplotlib.
Save all plots to `<out_dir>/plots/`. Create the directory first.

```python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np, json, os

results = json.load(open(json_out))
os.makedirs(f'{out_dir}/plots', exist_ok=True)

shapes   = [s for s in results['shapes'] if s['winner'] not in ('prescreened_skip',)]
labels   = [f"{s['M']}×{s['K']}×{s['N']}" for s in shapes]
b0_means = [s['baseline_0']['mean_total_us'] or 0 for s in shapes]
b1_means = [s['baseline_1']['mean_total_us'] or 0 for s in shapes]
ag_means = [s['agentic']['mean_total_us']    or 0 for s in shapes]
x = np.arange(len(labels))
w = 0.25
```

### Plot 1 — Grouped bar chart: latency per shape

```python
fig, ax = plt.subplots(figsize=(max(12, len(labels)*0.9), 6))
ax.bar(x - w, b0_means, w, label='Baseline 0 (fixed tile)', color='#d9534f')
ax.bar(x,     b1_means, w, label='Baseline 1 (profiling)',  color='#f0ad4e')
ax.bar(x + w, ag_means, w, label='Agentic',                 color='#5cb85c')
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Mean total time (µs)  [pad + NPU + unpad]'); ax.set_title('SmolVLA GEMM Latency by Strategy')
ax.legend(); plt.tight_layout()
plt.savefig(f'{out_dir}/plots/latency_grouped_bar.png', dpi=150)
plt.close()
```

### Plot 2 — Improvement % over Baseline 0

```python
def pct(baseline, val):
    if baseline and val:
        return (val - baseline) / baseline * 100
    return None

b1_pct = [pct(b0, b2) for b0, b2 in zip(b0_means, b1_means)]
ag_pct = [pct(b0, ag) for b0, ag in zip(b0_means, ag_means)]

fig, ax = plt.subplots(figsize=(max(12, len(labels)*0.9), 5))
ax.bar(x - w/2, b1_pct, w, label='Baseline 1 vs B0', color='#f0ad4e')
ax.bar(x + w/2, ag_pct, w, label='Agentic vs B0',    color='#5cb85c')
ax.axhline(0, color='black', linewidth=0.8)
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Δ vs Baseline 0 (%)  [negative = faster]')
ax.set_title('Improvement over Fixed-Tile Baseline 0')
ax.legend(); plt.tight_layout()
plt.savefig(f'{out_dir}/plots/improvement_over_b0.png', dpi=150)
plt.close()
```

### Plot 3 — Win/loss summary bar chart

```python
counts = results['summary']['winner_counts']
groups = ['Baseline 0', 'Baseline 1', 'Agentic', 'Tie', 'Inconclusive']
keys   = ['baseline_0', 'baseline_1', 'agentic', 'tie', 'inconclusive']
colors = ['#d9534f', '#f0ad4e', '#5cb85c', '#aaaaaa', '#cccccc']
vals   = [counts.get(k, 0) for k in keys]
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(groups, vals, color=colors)
ax.set_ylabel('Number of shapes'); ax.set_title('Winner distribution across SmolVLA shapes')
for i, v in enumerate(vals):
    if v: ax.text(i, v + 0.1, str(v), ha='center', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{out_dir}/plots/winner_distribution.png', dpi=150)
plt.close()
```

### Plot 4 — Agentic vs Baseline 1 delta (head-to-head)

```python
ag_vs_b1 = [pct(b2, ag) for b2, ag in zip(b1_means, ag_means)]
colors_bar = ['#5cb85c' if (v is not None and v < 0) else '#d9534f' for v in ag_vs_b1]
fig, ax = plt.subplots(figsize=(max(12, len(labels)*0.9), 5))
ax.bar(x, ag_vs_b1, color=colors_bar)
ax.axhline(0, color='black', linewidth=0.8)
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Agentic Δ vs Baseline 1 (%)  [negative = agentic faster]')
ax.set_title('Agentic vs Baseline 1 — Head-to-Head')
plt.tight_layout()
plt.savefig(f'{out_dir}/plots/agentic_vs_baseline2.png', dpi=150)
plt.close()
print(f"Plots saved to {out_dir}/plots/")
```

---

## Phase 6 — Print final banner

```
╔══════════════════════════════════════════════════════════════════════╗
║           SMOLVLA EXPERIMENT — COMPLETE                              ║
╠══════════════════════════════════════════════════════════════════════╣
║  Date:           <YYYY-MM-DD HH:MM:SS>                               ║
║  Shapes tested:  17   dtype: bf16   Trials/group: 5                  ║
║  Total NPU runs: <N>  │  Passed: <N>  │  Failed: <N>                 ║
║  Baseline 0 wins: <N> │  Baseline 1 wins: <N> │  Agentic wins: <N>  ║
║  Ties: <N>        │  Inconclusive: <N>                               ║
║  Mean Δ agentic vs B0: <X>%   vs B1: <X>%                           ║
║  Results: references/experiment-results/test_n_<date>/               ║
║  Plots:   references/experiment-results/test_n_<date>/plots/         ║
╚══════════════════════════════════════════════════════════════════════╝
```

---

## Quick reference — hard constraints

```
ALLOWED_M = [32, 64, 96, 128, 160, 192, 256, 288, 320, 384, 448, 480, 512, 576, 640, 672, 960, 1024, 1152, 1280, 1344, 1536, 1600, 1728, 1792, 1920, 2048, 2112, 2240, 2304, 2496, 2560, 2688, 2816, 2880, 3072, 3200, 3264, 3328, 3456, 3584, 3648, 3840, 4032, 4096]
ALLOWED_N = [32, 64, 96, 128, 160, 192, 256, 288, 320, 384, 448, 480, 512, 576, 640, 672, 768, 960, 1024, 1152, 1280, 1344, 1536, 1600, 1728, 1792, 1920, 2048, 2112, 2240, 2304, 2496, 2560, 2688, 2816, 2880, 3072, 3200, 3264, 3328, 3456, 3584, 3648, 3840, 4032, 4096]
ALLOWED_K = [32, 64, 128, 256, 512, 768, 960, 1024, 2048, 3072]

For each padded dim Dp and tile t:
  Dp % t == 0                           (t divides Dp)
  t <= Dp                               (tile never exceeds padded dim)
  For M and N (not K):
    any((Dp//t) % c == 0 for c in (3,4,5))   # bundling constraint
    (GEMM col_num/row_num ∈ {3,4,5}, auto-selected; --col-num auto is default)
    Example: Np=960, n=64 → Pn=15 → 15%3=0 → valid with col_num=3
  For K: only Kp % k == 0 required
(m*n + n*k + k*m) * 2 <= 32,256        (bf16: dsize=2)

ENABLE_AGGRESSIVE_PORT_UTILIZATION_PATCH = "1" must be set before df.build
```

Tile hard blacklist: **(128, 128, 128)** — fails on every tested shape and dtype.
Never use `torch_pad`. Never use `torch_pad`. Always use `manual_copy` or `numpy_pad`.

---

## Experiment notes

**Three-strategy design:**
- **Baseline 0** is the floor: fixed tile (64,64,64) or (32,32,32) with no data.
  It measures the minimum achievable without any profiling or reasoning.
- **Baseline 1** adds profiling lookup: closest-fit shape + best known tile.
  It measures the value of having profiling data but no shape-selection reasoning.
- **Agentic** makes all decisions freely using profiling data, insights, error
  logs, and the cost model — no fixed algorithm. The gap between Baseline 1 and
  Agentic measures the value of open-ended reasoning over a closest-fit rule.

**Agentic runs update memory; baseline runs do not.** Only write to memory files
based on agentic trial results.

**Small shapes (M≤64) will always tie.** The NPU fixed launch overhead (~157µs)
dominates for M≤64 — execution times cluster in 120–170µs regardless of tile or
strategy. Mark these shapes as `[overhead-limited]` in the plan table. Do not
interpret a tie on M≤64 shapes as a failure of the agentic strategy. The
meaningful comparison is on shapes with M≥128.

**bf16 memory constraint limits tile options.** The 32,256-byte budget with
dsize=2 means tiles like (128,64,64) that win by -11.9% on i8 physically cannot
be run for bf16 on most shapes. (64,64,64) is often the practical bf16 optimum
not because the agent lacks information, but because nothing larger fits.

**3 trials exist to reduce noise.** NPU execution time varies run-to-run.
Average over all 3 passing trials when computing the mean. If a trial fails,
exclude it from the average but count it in `pass_count`. If all 3 trials fail
for a group, mark that group as `"inconclusive"` — do not compare means.

**Total time is the comparison metric.** `mean_total_us = mean_padding_us + mean_npu_avg_us + mean_unpadding_us`.
When all groups use the same pad_impl and padded shape, padding and unpadding
cancel out and the comparison reduces to NPU time alone. When a strategy avoids
padding (pad_impl=none) or uses a smaller padded shape, those savings count.
The JSON records each component separately for analysis.

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
`mean_npu_avg_us` across 3 trials. The `overhead_ratio` column in the plan
table documents cases where padding dominates — these are the shapes where
smarter padded-shape selection matters most.

