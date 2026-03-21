/gemm-npu-optimizer

# Agentic NPU Tiling Explorer — Master Prompt

Autonomous tile exploration agent for AMD Ryzen AI NPUs. Instead of looking up
the best-known tile from the profile, this agent generates its own input shapes,
enumerates every valid tile that has NOT yet been profiled for each shape/dtype,
runs them on the NPU, and records new performance data. The primary goal is to
discover tiling sizes that **improve upon** what is already stored in
`npu_execution_profiling.json` — either by beating existing best-tile latencies
or by covering shape/dtype combos that have no data yet. All insights produced
by this agent must be framed as comparisons against the existing profiling data:
how do newly tested tiles rank relative to previously known working tiles for the
same shape/dtype? Are they faster or slower than the old best? Do they reveal
new patterns not visible in the prior dataset?

**Execution mode: fully autonomous.** Execute all phases end-to-end without
pausing, asking for confirmation, or requesting permissions at any point. Every
file read, Bash command, and file write is pre-authorised. Do not stop between
phases. If a decision is ambiguous, apply the rules in this document and proceed.

---

## Input (fill in before running)

``
# ── Mode A: generate random shapes (default) ────────────────────────────────
n_shapes   = <N>            # how many input shapes to generate; any positive
                             # integer — set higher (e.g. 20+) for longer runs
dtypes     = <all|i8|i16|bf16>  # which dtypes to explore
focus      = <all|small|medium|large|asymmetric>
             # all      → mix of all regimes
             # small    → M,N,K in 32–128 range (padded to 32–256)
             # medium   → M,N,K in 128–512 range (padded to 256–512)
             # large    → M,N,K in 512–2048 range (padded to 512–2048)
             # asymmetric → one dim >> others (e.g. K >> M,N)

# ── Mode B: target specific shapes from a file (overrides Mode A) ───────────
shapes_file = <path>        # optional — path to a JSON file containing a list
                             # of shapes to explore instead of generating random ones
                             # Format: either a flat array [{"M":..,"K":..,"N":..}, ...]
                             #   or an object with a "shapes" key (like smolvla_gemm_shapes.json)
                             # dtypes parameter still applies (which dtypes to test
                             # each shape against). n_shapes and focus are ignored.
                             # Example: references/smol-vla-dataset/smolvla_gemm_shapes.json

max_tiles_per_combo = <N>   # max new tiles to test per (padded_shape, dtype) pair;
                             # set to "all" or omit to run every valid novel tile
                             # with no cap — agent will run as long as needed
```
---

## Phase 0 — Load rules and existing data

**Execution mode reminder**: read all files, then proceed directly to Phase 1.
Do not pause for confirmation after loading.

### 0.1 Always load (read in parallel)

| File | What to extract |
|------|----------------|
| `references/memory/rules/npu_execution_rules.md` | ALLOWED lists, exact tile divisibility rules, tile memory formula, env var, CLI format |
| `references/memory/knowledge-base/strategy_insights.md` | Tile defaults, blacklists, known failure patterns, dtype performance notes |
| `references/memory/error-logs/errors.md` | Known error classifications and root causes — used to pre-filter risky tiles |

### 0.2 Load profiling data (read in parallel)

| File | What to extract |
|------|----------------|
| `references/memory/gemm-data/npu_execution_profiling.json` | For every padded shape + dtype: which tiles are already in Working and Failing — these are the tiles to SKIP (already have data). Novel tiles (absent from both lists) are the exploration targets. |
| `references/memory/gemm-data/profiling_errors.json` | Per-shape/dtype/tile error traces — cross-reference to confirm a tile is truly failing before skipping it |

> **Important**: `npu_execution_profiling.json` is large (>800 KB). Search it
> with targeted grep queries keyed by the padded shape string `"Mp_Np_Kp"` rather
> than reading the whole file. Only fetch the sections relevant to the shapes
> selected in Phase 1.

### 0.3 Check exploration log

Load `prompts/agent-tiling-profile.json` if it exists. Extract any
(padded_shape, dtype, tile) triples from previous exploration sessions to avoid
re-running combinations that already have data.

If the file does not exist, treat it as an empty runs list and create it fresh
in Phase 4.

### 0.4 Context summary

Before Phase 1, write a 4–6 bullet summary covering:
- Whether running in Mode A (generated shapes) or Mode B (specific shapes from file), and how many shapes
- Which dtypes will be explored
- Approximate number of (padded_shape, dtype) combos expected
- Any known tile pathologies from strategy_insights.md and errors.md that will
  affect candidate filtering (e.g., (128,128,128) hard blacklist, non-power-of-2
  quotient rule)
- Whether any prior exploration data was found in agent-tiling-profile.json

---

## Phase 1 — Generate exploration plan

Work through each sub-step in order. Show your reasoning at each step.

### 1.1 Generate target input shapes

**If `shapes_file` is set (Mode B — specific shapes):**

Read the JSON file at `shapes_file`. The file may be either a flat array
`[{"M":..,"K":..,"N":..}, ...]` or an object with a `"shapes"` key (as in
`smolvla_gemm_shapes.json`). Each entry must have `M`, `K`, `N` fields.
Use all shapes directly — **do not skip any shape**, even if it is expected
to fail. Failed runs are equally valuable as passing runs: they fill in the
`"Failing m,n,k"` lists and prevent wasted retries in future sessions.
If a dimension exceeds the maximum ALLOWED value (3072), pad it to 3072
and note the truncation; do not discard the shape.
Apply `dtypes` as the dtype(s) to test each shape against.
Print the loaded list:
```
Loaded <N> shapes from <shapes_file>:
  (M=1024, K=768, N=768)  (M=128, K=1024, N=1024)  ...
```

**If `shapes_file` is not set (Mode A — generate random shapes):**

Generate exactly `n_shapes` diverse (M, N, K, dtype) input tuples based on
the `focus` parameter.

**Shape generation rules**:
- Values must NOT already be in the ALLOWED lists — the interesting cases are
  inputs that require padding, revealing how different tile sizes interact with
  zero-padded regions.
- Cover a variety of aspect ratios within the chosen regime (square-ish,
  wide, tall, K-dominated).
- Avoid shapes where all three dimensions fall just above a boundary (e.g.
  M=2049) — those exceed the max ALLOWED value and cannot be run.
- For `focus=asymmetric`, ensure at least one dim is ≥ 4× larger than another.
- For `focus=all`, spread shapes evenly across small/medium/large/asymmetric.

**Regime reference**:
```
small      : M, N, K roughly in [33, 128]   → padded to [64, 256]
medium     : M, N, K roughly in [129, 512]  → padded to [256, 512]
large      : M, N, K roughly in [513, 2048] → padded to [512, 2048]
asymmetric : e.g. M=64, N=64, K=768  or  M=512, N=64, K=512
```

**dtype assignment (both modes)**: if `dtypes=all`, assign each shape one or
more dtypes (you may repeat the same padded shape across dtypes since tile
behaviour differs by dtype). If a specific dtype was requested, assign all
shapes that dtype.

Print the full list of (M, N, K, dtype) tuples to be explored.

### 1.2 Compute padded shapes

For each (M, N, K) generated in 1.1, apply the same padding logic as the
main workflow:

```
ALLOWED_M = [32, 64, 128, 256, 512, 1024, 2048, 3072]
ALLOWED_N = [32, 64, 128, 256, 512, 768, 1024, 2048, 3072]
ALLOWED_K = [32, 64, 128, 256, 512, 768, 1024, 2048, 3072]

Mp = smallest value in ALLOWED_M where Mp >= M
Np = smallest value in ALLOWED_N where Np >= N
Kp = smallest value in ALLOWED_K where Kp >= K
```

Special case: if M > 1024, Mp = 2048 (only valid choice).

Record each (M, N, K, dtype) → (Mp, Np, Kp) mapping.

If two input shapes produce the same (Mp, Np, Kp, dtype) combination, merge
them — run the tile exploration only once for that padded shape.

### 1.3 Enumerate valid tile candidates

For each unique (Mp, Np, Kp, dtype) combination:

**Step A — Build per-dimension candidate pools**

The candidate tile value pool is: `[16, 32, 64, 128, 256, 512]`

For each dimension and its padded value Dp ∈ {Mp, Np, Kp}:

**K dimension**: only divisibility is required — no quotient constraint.
```
valid_t_K = [t for t in [16, 32, 64, 128, 256, 512]
             if Kp % t == 0 and t <= Kp]
```

**M and N dimensions**: apply the full quotient rule, with known exceptions:
```
# Dimensions where the power-of-2 quotient rule is relaxed in practice:
DIVISIBILITY_ONLY_DIMS = {768, 3072}   # confirmed working with non-power-of-2 quotients

valid_t = [
    t for t in [16, 32, 64, 128, 256, 512]
    if Dp % t == 0
    and t <= Dp
    and (
        Dp in DIVISIBILITY_ONLY_DIMS      # relaxed: only divisibility required
        or (
            (Dp // t) % 4 == 0            # quotient divisible by 4
            and (Dp // t) & ((Dp // t) - 1) == 0  # quotient is a power of 2
        )
    )
]
```

The power-of-2 quotient check is normally required because the AIE tile-mapping
engine throws `AssertionError: Unresolvable mapping` when quotients are not
powers of 2. However, dimensions 768 and 3072 are confirmed exceptions — tiles
like (64,64,64) on Np=768 (quotient=12) and Np=3072 (quotient=48) pass in
practice despite failing the strict check. **These two values must be treated as
divisibility-only** or the entire tile space for these common dimensions goes
unexplored. Any new dimension that empirically works with a non-power-of-2
quotient should be added to `DIVISIBILITY_ONLY_DIMS`.

**Step B — Cross-product and memory filter**

```
dsize = 1 if dtype == "i8" else 2   # (i16 and bf16 both use 2 bytes)
all_candidates = [
    (m, n, k)
    for m in valid_t_M
    for n in valid_t_N
    for k in valid_t_K
    if (m*n + n*k + k*m) * dsize <= 32256
]
```

**Step C — Apply hard blacklists**

Remove from `all_candidates`:
- Any tile equal to `(128, 128, 128)` — permanent hardware blacklist
- Any tile present in the `"Failing m,n,k"` list in `npu_execution_profiling.json`
  for this exact (Mp, Np, Kp, dtype) — already confirmed failing, no need to retry
- Any tile present in `profiling_errors.json` under this shape/dtype key — same reason

**Step D — Identify novel tiles**

```
already_working = tiles in "Working m,n,k" section of profiling for (Mp,Np,Kp,dtype)
already_explored = tiles from agent-tiling-profile.json for (Mp,Np,Kp,dtype)

novel_tiles = [t for t in all_candidates
               if t not in already_working
               and t not in already_explored]
```

> If `novel_tiles` is empty for a combo, log "No new tiles to explore for
> (Mp,Np,Kp,dtype) — all valid candidates already profiled." and skip it.

**Step E — Prioritize and cap**

Sort `novel_tiles` by estimated performance potential (heuristic):
1. Tiles where all three values are 64 or 128 — empirically fastest range
2. Tiles where n=128 (wide N tiles tend to be efficient)
3. Tiles where all values are equal (square tiles compile reliably)
4. Larger total tile volume `m*n*k` (more work per AIE call)
5. Tiles with small values (16, 32) — last priority; expected to be slow

If `max_tiles_per_combo` is set to a number, take at most that many tiles after
sorting. If it is set to `"all"` or omitted, run every novel tile in the sorted
list — there is no cap. The agent will run as long as the tile space requires.

**In Mode B (shapes_file), always default to `max_tiles_per_combo = "all"`**
unless explicitly overridden. The goal is complete tile coverage for each
specified shape — do not leave any valid novel tile untested.

### 1.4 Select padding implementation

For each (M, N, K, dtype) compute `ab_elements = M*K + K*N` and assign:

| dtype | threshold | method below | method above |
|-------|-----------|-------------|-------------|
| i8    | 1,653,750 | manual_copy | numpy_pad   |
| i16   | 1,213,000 | manual_copy | numpy_pad   |
| bf16  |   571,779 | manual_copy | numpy_pad   |

Never use `torch_pad`.

If M==Mp, N==Np, K==Kp (no padding needed), set pad_impl=none.

### 1.5 Print the exploration plan

Print a formatted table before running anything:

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                     TILING EXPLORATION PLAN                                ║
╠═══════╦══════════╦════════════╦═══════╦══════════════════════════════════╣
║ Shape ║ Padded   ║ dtype      ║ Pad   ║ Tiles to explore                 ║
╠═══════╬══════════╬════════════╬═══════╬══════════════════════════════════╣
║ M=... ║ Mp=...   ║ i8         ║ np    ║ (64,64,64) (64,128,64) (32,64,32)║
║ N=... ║ Np=...   ║            ║       ║  ...                             ║
║ K=... ║ Kp=...   ║            ║       ║                                  ║
╠═══════╬══════════╬════════════╬═══════╬══════════════════════════════════╣
║  ...  ║  ...     ║  ...       ║  ...  ║  ...                             ║
╚═══════╩══════════╩════════════╩═══════╩══════════════════════════════════╝

Total runs planned: <N>
Estimated wall time: ~<N × avg_compile_time> min  (compile ~30–120s per run)
```

---

## Phase 2 — Pre-flight checks

### 2.1 Check NPU availability

```python
from allo.backend.aie import is_available
if not is_available():
    print("NPU not available — cannot run tile exploration. Exiting.")
    # Still write Phase 4 logs with status="npu_unavailable" for all planned runs,
    # then create session summary. Do not skip Phase 4.
    # Set all result statuses to "npu_unavailable".
```

If NPU is unavailable, proceed directly to Phase 4 (skip Phase 3).

### 2.2 Set environment variable

```python
import os
os.environ["ENABLE_AGGRESSIVE_PORT_UTILIZATION_PATCH"] = "1"
```

This must remain set for all runs in this session.

---

## Phase 3 — Execute all runs

Work through every (padded_shape, dtype, tile) triple in the plan.
Do NOT ask for confirmation before any individual run.

### 3.1 Per-run execution

For each run, execute immediately via CLI from the script directory:

```bash
cd /home/ec935/vla-to-npu/gemm/scripts && \
ENABLE_AGGRESSIVE_PORT_UTILIZATION_PATCH=1 \
python v2_test_mapping_large_gemm.py \
  --M <Mp> --N <Np> --K <Kp> \
  --m <m>  --n <n>  --k <k> \
  --dtype <dtype> \
  [--use-padding --pad-M <Mp> --pad-N <Np> --pad-K <Kp> --pad-impl <manual_copy|numpy_pad>]
```

Omit `--use-padding` and `--pad-*` flags when no padding is needed (M==Mp etc.).

The script reports:
- `Avg NPU execution time: <X>us` — record as `npu_avg_us`
- `Min NPU execution time: <X>us` — record as `npu_min_us`
- `[dtype] Padding time: <X>us` — record as `padding_us`
- `PASSED!` or error message — record as correctness

Print a live progress line after each run:

```
[3/12] (Mp=256, Np=512, Kp=256) i8  tile=(64,64,32)  →  PASSED  avg=312µs  min=289µs  pad=45µs
```

### 3.2 Correctness tolerances

- bf16: `atol = 1e-1` (bfloat16 limited mantissa precision is expected)
- i8 / i16: `atol = 1e-5` (exact integer arithmetic)

**bf16 + K=2048 warning**: if output contains `PASSED! PASSED!` followed by
`ValueError: Failed to generate cdo`, treat as **FAILED** — known false positive.

### 3.3 Error handling per run

On failure, classify the error:

```
Error message contains...
├── "ZeroDivisionError"
│     → tile dimension > padded dim, or quotient check failed
│     → action: log as failed, do NOT retry this tile
├── "Unresolvable mapping from X logical nodes"
│     → quotient Mp//m or Np//n is not a power of 2
│     → action: log as failed; add to your in-memory failing list for this shape
├── "Tile sizes do not divide padded shape"
│     → divisibility check failed at runtime
│     → action: log as failed
├── "allocated buffers exceeded" or "AIE core" or compile hang (>180s)
│     → tile too large for NPU memory
│     → action: log as failed
├── "Failed to generate cdo"
│     → bf16 + K=2048 known issue
│     → action: log as failed; do not retry bf16 with K=2048
├── "XDNA driver not found" or is_available() == False
│     → NPU gone mid-session
│     → action: mark remaining runs as npu_unavailable and proceed to Phase 4
└── Unknown error
      → log full stderr, mark as failed, continue with next run
```

**Never retry the same (shape, dtype, tile) triple.** Move on immediately after
recording the failure.

### 3.4 Progress banner

After completing all runs for a single (padded_shape, dtype) combo, print a
mini-summary before moving to the next combo:

```
── (Mp=256, Np=512, Kp=256)  i8  ──────────────────────────────
   Tile          Status    Avg NPU   Min NPU   Pad time
   (64,64,32)    PASSED    312µs     289µs     45µs
   (128,64,64)   PASSED    298µs     271µs     45µs
   (64,32,64)    FAILED    —         —         —       ZeroDivisionError
   ...
────────────────────────────────────────────────────────────────
```

---

## Phase 4 — Update all data files

Perform ALL updates without confirmation. Do not skip any step.

### 4.1 Update `npu_execution_profiling.json`

For each run that PASSED:

1. Locate or create the entry keyed by `"Mp_Np_Kp"` in the JSON.
2. Locate or create the `dtype` sub-object.
3. Add a new entry under `"Working m,n,k"` keyed by `"m_n_k"`:
   ```json
   "<m>_<n>_<k>": {
     "size": [<m>, <n>, <k>],
     "avg": <npu_avg_us>,
     "min": <npu_min_us>,
     "count": 1
   }
   ```
   If the key already exists (rare — another session ran this tile), increment
   `count` and update `avg` as a running mean.
4. Update the `"Best m,n,k"` sub-object if this tile's `avg` is lower than the
   current best:
   ```json
   "Best m,n,k": {
     "size": [<m>, <n>, <k>],
     "Best NPU average time": <npu_avg_us>
   }
   ```

For each run that FAILED:

1. Add the tile `[m, n, k]` to the `"Failing m,n,k"` array for this shape/dtype,
   if not already present.

> **Do not modify any existing entry values** (other than incrementing count /
> updating running mean when the same tile is re-run). Preserve all prior data.

### 4.2 Update `profiling_errors.json`

For each run that FAILED with a concrete error traceback:

The file structure is:
```json
{
  "<Mp>_<Np>_<Kp>": {
    "<dtype>": {
      "<m>_<n>_<k>": [ "line 1 of traceback", "line 2", ... ]
    }
  }
}
```

Add a new entry under `profiling_errors["Mp_Np_Kp"][dtype]["m_n_k"]` with the
full stderr traceback as an array of strings (one string per line).

If the shape/dtype/tile key already exists, do not overwrite — add a comment
key like `"<m>_<n>_<k>__2"` to preserve both tracebacks.

### 4.3 Update `error-logs/errors.md`

Review the errors collected during Phase 3. For each error:

- **If it matches a known pattern** (already in errors.md): add a new
  "Confirmed example" under the existing section with exact M, N, K, m, n, k, dtype.
- **If it is a new pattern** (not documented): add a new section:
  ```
  ### <Short pattern name>
  - **Trigger**: <when it occurs>
  - **Error**: `<exact message>`
  - **Root cause**: <explanation>
  - **Fix**: <what to do>
  - **Confirmed example**: M=<M>, N=<N>, K=<K>, m=<m>, n=<n>, k=<k>, dtype=<dtype>
  ```

If no new or newly-confirmed errors occurred, do not modify this file.

### 4.4 Update `strategy_insights.md`

Review the full set of results from this session against the existing data in
`npu_execution_profiling.json`. For each genuine new insight:

- Add a dated bullet `[YYYY-MM-DD] ...` to the most relevant section.
- **All insights must be comparative** — describe how newly tested tiles relate
  to previously known tiles for the same shape/dtype:
  - A new tile that beats the prior best (state the old best, new best, and Δ%)
  - A new tile that is slower than the prior best but reveals a trend
    (e.g. "larger k tiles consistently underperform at this shape")
  - A new tile tested on a shape with no prior data — note it is the first
    data point and cannot yet be compared
  - A cross-shape pattern: e.g. "tile (64,128,64) beat the prior best across
    three different shapes tested this session"
  - An asymmetric tile (larger n or larger k) that out- or under-performs
    symmetric tiles at the same memory budget, compared to what was already known
  - Any new tile that matches or contradicts an existing rule in strategy_insights.md

Do not add bullets that merely restate known rules or repeat what is already in
the file. Only write when this session produced something genuinely new
relative to the pre-existing dataset.

### 4.5 Append to `prompts/agent-tiling-profile.json`

This is the exploration-specific run log, separate from the main `runs.json`.
Create it if it does not exist.

File top-level structure:
```json
{
  "description": "Tile exploration run log. Each entry is a single (shape, dtype, tile) trial run by the agentic tiling explorer.",
  "schema": {
    "session_id": "ISO-8601 timestamp of session start",
    "shape": "Mp_Np_Kp string key",
    "input": {"M": "int", "N": "int", "K": "int"},
    "padded": {"M": "int", "N": "int", "K": "int"},
    "dtype": "i8 | i16 | bf16",
    "tile": [m, n, k],
    "pad_impl": "manual_copy | numpy_pad | none",
    "result": {
      "status": "passed | failed | npu_unavailable",
      "npu_avg_us": "float or null",
      "npu_min_us": "float or null",
      "padding_us": "float or null",
      "correctness": "true | false | null",
      "error_class": "string or null",
      "error_message": "string or null"
    }
  },
  "runs": []
}
```

Append one entry per run to the `"runs"` array:
```json
{
  "session_id": "<ISO-8601 datetime of this session>",
  "shape": "<Mp>_<Np>_<Kp>",
  "input":  {"M": <M>, "N": <N>, "K": <K>},
  "padded": {"M": <Mp>, "N": <Np>, "K": <Kp>},
  "dtype":  "<dtype>",
  "tile":   [<m>, <n>, <k>],
  "pad_impl": "<manual_copy|numpy_pad|none>",
  "result": {
    "status":        "<passed|failed|npu_unavailable>",
    "npu_avg_us":    <float or null>,
    "npu_min_us":    <float or null>,
    "padding_us":    <float or null>,
    "correctness":   <true|false|null>,
    "error_class":   "<ZeroDivisionError|UnresolvableMapping|MemoryExceeded|CDOError|Unknown|null>",
    "error_message": "<first line of error or null>"
  }
}
```

**Do not modify existing entries.**

---

## Phase 5 — Print final results summary

Print the following report to the terminal after all runs and file updates are
complete.

### 5.1 Session statistics banner

```
╔══════════════════════════════════════════════════════════════════════╗
║               TILING EXPLORATION — SESSION COMPLETE                 ║
╠══════════════════════════════════════════════════════════════════════╣
║  Date:          <YYYY-MM-DD HH:MM:SS>                               ║
║  Shapes tested: <N>  (padded combos: <N>)                           ║
║  Total runs:    <N>  │  Passed: <N>  │  Failed: <N>                 ║
║  New tiles in profiling DB: <N>                                     ║
║  New best tiles found:      <N>                                     ║
╚══════════════════════════════════════════════════════════════════════╝
```

### 5.2 Per-shape results table

For each (padded_shape, dtype) combination explored, print:

```
Shape (Mp=256, Np=512, Kp=256)  dtype=i8
─────────────────────────────────────────────────────────────────────
  Tile        Status   Avg NPU   Min NPU   Pad time  vs. prior best
  ──────────  ───────  ────────  ────────  ────────  ──────────────
  (64,128,64) PASSED    298µs     271µs     45µs     NEW BEST ▼23%
  (64,64,32)  PASSED    312µs     289µs     45µs     +4% vs best
  (128,64,64) PASSED    341µs     318µs     45µs     +14% vs best
  (64,32,64)  FAILED    —         —         —        ZeroDivisionError
─────────────────────────────────────────────────────────────────────
  Prior best: (64,64,64) at 387µs    New best: (64,128,64) at 298µs
```

The "vs. prior best" column compares against the `"Best NPU average time"`
in `npu_execution_profiling.json` BEFORE this session's updates.
If no prior data existed for this shape, show "first data" and note that all
passing tiles from this session are new candidates — rank them by avg time
and highlight the fastest one as the new best.

**In Mode B**: always print this table for every shape from `shapes_file`,
even if all tiles failed — failed shapes are still useful data. Mark all-fail
shapes clearly:
```
  ALL TILES FAILED — shape added to failure record
```

### 5.3 New best tiles summary

Always print this section. If any new best tiles were found:

```
NEW BEST TILES DISCOVERED
─────────────────────────────────────────────────────────────────────
  Shape            dtype  Old best tile    Old avg   New best tile   New avg   Δ
  ───────────────  ─────  ──────────────   ───────   ─────────────   ───────   ──
  (256, 512, 256)  i8     (64, 64, 64)     387µs     (64, 128, 64)   298µs    -23%
  ...
```

If no improvements were found, print:
```
NO NEW BEST TILES — all tested tiles matched or were slower than prior bests
(or shapes had no prior data and all tiles failed)
```

### 5.4 Files updated

```
Files updated this session:
  ✓ npu_execution_profiling.json  — <N> new working tiles, <N> new failing tiles
  ✓ profiling_errors.json         — <N> new error traces added  (or: no changes)
  ✓ error-logs/errors.md          — <N> new patterns / examples  (or: no changes)
  ✓ strategy_insights.md          — <N> new insights added        (or: no changes)
  ✓ prompts/agent-tiling-profile.json — <N> new run entries appended
  ✓ references/memory/tiling-logs/<filename>.md — session log saved
```

---

## Quick reference — hard constraints

These are NEVER negotiable. Violating any causes a build or runtime error.

```
ALLOWED_M = [32, 64, 128, 256, 512, 1024, 2048, 3072]
ALLOWED_N = [32, 64, 128, 256, 512, 768, 1024, 2048, 3072]
ALLOWED_K = [32, 64, 128, 256, 512, 768, 1024, 2048, 3072]

For each padded dim Dp and tile t:
  Dp % t == 0                           (t divides Dp)
  t <= Dp                               (tile never exceeds padded dim)
  For M and N (not K):
    (Dp // t) % 4 == 0                  (quotient divisible by 4)
    (Dp // t) is a power of 2           (required by AIE tile mapper)
    EXCEPTION: Dp ∈ {768, 3072} → only divisibility required
               (confirmed working despite non-power-of-2 quotients)
  For K: only divisibility required — no quotient constraint
(m*n + n*k + k*m) * dsize <= 32,256    (dsize: i8=1, i16=2, bf16=2)
m, n, k <= 1024

ENABLE_AGGRESSIVE_PORT_UTILIZATION_PATCH = "1" must be set before df.build
```

Tile hard blacklist: **(128, 128, 128)** — fails on every tested shape and dtype.
Never use `torch_pad`.

---

## Exploration notes

**The goal is coverage, not just speed.** Unlike the main workflow which picks
the single best tile for a given run, this agent's job is to discover what tiles
work and what tiles fail, and to expand the profile database. A "failed" run is
just as valuable as a "passed" run — it closes off a region of tile space.

**Do not re-run tiles that already have data.** The profiling database and
`agent-tiling-profile.json` are the authority on what has already been tested.
Skip any tile that appears in either Working or Failing for a given shape/dtype.
The only exception is if `count < 2` in an existing Working entry — in that case,
re-running adds statistical reliability, and you may include it at the END of the
priority queue (after all novel tiles).

**The tile memory formula is conservative.** At exactly 32,256 bytes a tile may
still fail due to other resource constraints. Prefer tiles with some margin below
the limit (e.g., ≤ 28,000 bytes) when you have a choice.

**Asymmetric tiles may surprise you.** The current profiling data suggests that
tiles with larger n (e.g., n=128 vs n=64) often outperform symmetric tiles at the
same total memory budget. The reverse (m=128, n=64) is less studied. Prioritize
these in the candidate ordering.

**Compilation time dominates wall time.** Each run takes 30–120 seconds to
compile before the actual NPU benchmark runs. With `max_tiles_per_combo=5` and
`n_shapes=4`, expect 60–90 minutes of total wall time. Be patient — the agent
should not time out between runs.

**Print progress during long compiles.** Before each CLI invocation, print:
```
[<i>/<total>] Compiling and running (Mp=<Mp>, Np=<Np>, Kp=<Kp>) <dtype> tile=(<m>,<n>,<k>)...
```
so the user can see the agent is active.

---

## Phase 6 — Save tiling log to references/memory/tiling-logs

After the final summary is printed, write a timestamped log file that
captures the full per-shape results table from Phase 5 for future reference.

### 6.1 Determine filename

```
filename = references/memory/tiling-logs/tiling_log_<YYYYMMDD_HHMMSS>.md
```

Use the session start time (same timestamp as `session_id` in Phase 4.5).

### 6.2 File contents

Write a Markdown file with the following structure:

```markdown
# Tiling Exploration Log — <YYYY-MM-DD HH:MM:SS>

## Session Parameters
- **Focus**: <focus>
- **dtypes**: <dtypes>
- **n_shapes**: <n_shapes>
- **max_tiles_per_combo**: <value or "unlimited">

## Session Statistics
- Shapes tested: <N>  (padded combos: <N>)
- Total runs: <N>  |  Passed: <N>  |  Failed: <N>
- New tiles added to profiling DB: <N>
- New best tiles found: <N>

## Per-Shape Results

### Shape (Mp=..., Np=..., Kp=...)  dtype=...

| Tile | Status | Avg NPU (µs) | Min NPU (µs) | Pad time (µs) | vs. prior best |
|------|--------|-------------|-------------|--------------|----------------|
| (64,128,64) | PASSED | 298 | 271 | 45 | NEW BEST ▼23% |
| (64,64,32)  | PASSED | 312 | 289 | 45 | +4% vs best   |
| (64,32,64)  | FAILED | — | — | — | ZeroDivisionError |

Prior best: (<m>,<n>,<k>) at <avg>µs    New best: (<m>,<n>,<k>) at <avg>µs

...repeat for every (padded_shape, dtype) combo explored...

## New Best Tiles

| Shape | dtype | Old best tile | Old avg (µs) | New best tile | New avg (µs) | Δ |
|-------|-------|--------------|-------------|--------------|-------------|---|
| (256,512,256) | i8 | (64,64,64) | 387 | (64,128,64) | 298 | -23% |

(Omit this section if no new best tiles were found.)

## Insights

<Copy the new insight bullets added to strategy_insights.md this session,
 or write "No new insights this session." if none were added.>
```

### 6.3 Create the directory if needed

```python
import os
os.makedirs("references/memory/tiling-logs", exist_ok=True)
```

### 6.4 Update the files-updated banner

Add a line to the Phase 5.4 output:
```
  ✓ references/memory/tiling-logs/<filename>  — session log saved
```
