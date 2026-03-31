/gemm-npu-optimizer

# Agentic NPU Workflow — Master Prompt

Step-by-step instructions for the agentic NPU GEMM optimization agent.
Copy this file into a Claude Code session, fill in the input block, and run.

**Execution mode: fully autonomous.** Execute all four phases end-to-end without
pausing, asking for confirmation, or requesting permissions at any point. Every
file read, Bash command, and file write in this workflow is pre-authorised. Do not
stop between phases. Do not ask the user to approve the strategy, the tile, the
CLI command, or any memory update. If a decision is ambiguous, apply the rules in
this document and proceed.

---

## Input (fill in before running)

```
M     = 1999
N     = 2043
K     = 1777
dtype = i8
```

---

## Phase 0 — Load memory and context

Read files in two passes. Pass 1 is always required. Pass 2 is conditional —
only load those files if the condition applies.

### Pass 1 — Always load (read in parallel)

| File | What to extract |
|------|----------------|
| `references/memory/rules/npu_execution_rules.md` | ALLOWED lists, tile divisibility rules, tile memory formula, env var, CLI format |
| `references/memory/knowledge-base/strategy_insights.md` | ab_elements thresholds, tile defaults/blacklists, hybrid shape rule, shape pathologies, any dated bullets relevant to this M/N/K/dtype |
| `references/memory/error-logs/errors.md` | Any known failure for this shape, tile, or dtype combination |
| `references/memory/gemm-data/runs.json` | Recent run outcomes — any prior runs on the same or similar shape |

### Pass 2 — Load when needed

| File | Load when |
|------|-----------|
| `references/memory/gemm-data/npu_execution_profiling.json` | Always load in Phase 1 to look up the best tile and execution time for the chosen padded shape |
| `references/memory/gemm-data/padding/padding_transition_thresholds.json` | Load if the ab_elements thresholds from strategy_insights.md are insufficient or need exact confirmation |
| `references/memory/gemm-data/profiling_errors.json` | Load when a tile candidate looks risky (e.g. candidate is (64,128,64), or the shape has a small N) |
| `references/memory/gemm-data/baseline1/results.json` | Load only if you need a baseline latency reference for comparison |

### 0.5 Context summary

Before moving to Phase 1, write a concise summary (3–6 bullets) of:
- Whether M, N, K are already in ALLOWED lists or need padding
- Which padded shape candidates exist (from ALLOWED lists)
- Whether a profile entry likely exists for the padded shape (check runs.json history)
- Any pathologies or warnings that apply to this input
- The most relevant knowledge-base rules for this specific case

---

## Phase 1 — Propose a strategy

Work through each sub-step in order. Show your reasoning at each decision point.

### 1.1 Determine padded shape candidates

For each dimension, find all valid padded values from the ALLOWED lists where
`Xp >= X`. These are the candidates.

```
ALLOWED_M = [32, 64, 128, 256, 512, 960, 1024, 2048, 3072]
ALLOWED_N = [32, 64, 128, 256, 512, 768, 960, 1024, 2048, 3072]
ALLOWED_K = [32, 64, 128, 256, 512, 768, 1024, 2048, 3072]
```

If a dimension is already in its ALLOWED list, it is its own candidate (no padding
needed for that dimension).

### 1.2 Run all three selection strategies

**Strategy A — `sorted_first_fit`**
Evaluate all valid (Mp, Np, Kp) combinations. Look up each in
`npu_execution_profiling.json`. Sort by NPU execution time ascending and pick the
first tuple where Mp ≥ M, Np ≥ N, Kp ≥ K. Record the selected shape and its
estimated NPU execution time. Note: this may produce large padding.

**Strategy B — `hashmap_27`**
Take the 3 smallest valid ALLOWED values ≥ M for M, ≥ N for N, and ≥ K for K.
This gives at most 27 candidate (Mp, Np, Kp) combinations. Look up each in
`npu_execution_profiling.json`. Pick the combination with the lowest estimated NPU
execution time. Record the selected shape. Note: if M > 1024, only 1 M candidate
(2048) exists, reducing the search to at most 6 candidates.

**Strategy C — `preprocessed_map`**
Find the closest valid (Mp, Np, Kp) — i.e. the smallest ALLOWED value ≥ each
dimension — and look up any existing profile-based mapping that points to a
faster shape. If the exact shape is in the profile, return it directly. This is
O(1) and is the preferred default when the shape is profiled.

### 1.3 Choose the padded shape

Apply the **hybrid selection rule**:

- **Small input regime** (M, N, K all roughly ≤ 64): prefer the padded shape with
  the fastest NPU execution time even if it adds more padding — padding cost is
  ~constant at this scale and extra zeros are negligible.

- **Large input regime** (any of M, N, K > 64): prefer the closest (smallest)
  valid (Mp, Np, Kp). Both padding time and NPU execution time grow with shape
  size; the closest shape wins on both fronts. Only deviate if a larger shape is
  substantially faster (>15%) AND the padding overhead is affordable.

- **M > 1024 special case**: only `Mp = 2048` is valid. All three strategies
  converge to the same padded shape. Use `preprocessed_map` and move on without
  further comparison.

**Padding overhead check**: Before committing to a shape, estimate the padding
volume as `Mp*Np*Kp - M*N*K` (in elements). If padding cost looks high relative
to execution time (based on prior runs or knowledge base), reconsider.

**Check `sorted_first_fit` result for padding waste**: if it picks a much larger
shape than `hashmap_27`, prefer the `hashmap_27` result unless it is clearly
slower on the NPU.

Record the final chosen (Mp, Np, Kp).

### 1.4 Select tile (m, n, k)

1. Look up (Mp, Np, Kp) and dtype in `npu_execution_profiling.json`.
   - If a "Best m,n,k" entry exists: use that tile.
   - Also check the "Failing m,n,k" list and blacklist any tile listed there.

2. If no profile entry exists, apply defaults in this priority order:
   - If `Np ≤ 64`: use **(32, 32, 32)** — larger tiles fail divisibility on small N.
   - Otherwise: use **(64, 64, 64)** — most consistently optimal across all dtypes.

3. Before finalising the tile, verify ALL of the following:
   - `Mp % m == 0` and `any((Mp//m) % c == 0 for c in (3,4,5))`
   - `Np % n == 0` and `any((Np//n) % c == 0 for c in (3,4,5))`
   - `Kp % k == 0` (K has no bundling constraint)
   - `m ≤ Mp`, `n ≤ Np`, `k ≤ Kp`
   - `(m*n + n*k + k*m) * dsize ≤ 32,256 bytes`
     (dsize: i8=1, i16=2, bf16=2)
   - The tile is NOT in the known failing list for this shape/dtype
   - The tile is not (128, 128, 128) — hard blacklist

4. If the chosen tile fails any check, fall back in this order:
   (64, 64, 32) → (64, 64, 64) → (32, 32, 32)
   Re-verify all checks after each fallback.

5. Note if the tile is (64, 128, 64) — it is unreliable. Only use it if it
   explicitly appears as "Best" in the profile for this exact shape/dtype.

### 1.5 Select padding implementation

Compute `ab_elements = M*K + K*N`.

Apply thresholds from `padding_transition_thresholds.json`:
- **i8**: use `manual_copy` if `ab_elements ≤ 1,653,750`; else `numpy_pad`
- **i16**: use `manual_copy` if `ab_elements ≤ 1,213,000`; else `numpy_pad`
- **bf16**: use `manual_copy` if `ab_elements ≤ 571,779`; else `numpy_pad`

**Never use `torch_pad`** — it is consistently the slowest method by a large margin.

If M, N, K are already equal to Mp, Np, Kp (no padding needed), set
`pad_impl = none` and omit all padding flags.

### 1.6 Output strategy block

Print the proposed strategy in this exact format:

```
=== PROPOSED STRATEGY ===
Input:          M=<M>, N=<N>, K=<K>, dtype=<dtype>
Padded shape:   Mp=<Mp>, Np=<Np>, Kp=<Kp>
Tile:           m=<m>, n=<n>, k=<k>
Pad impl:       <manual_copy|numpy_pad|none>
Strategy src:   <hashmap_27|sorted_first_fit|preprocessed_map>
Est. NPU time:  <X> µs  (from profile: <yes/no>)
Est. pad time:  <Y> µs  (from knowledge base: approx)

Checks passed:
  [ ] Mp % m == 0 and any((Mp//m)%c==0 for c in (3,4,5))
  [ ] Np % n == 0 and any((Np//n)%c==0 for c in (3,4,5))
  [ ] Kp % k == 0  (no bundling constraint on K)
  [ ] tile memory ≤ 32,256 bytes
  [ ] tile not in known failing list
  [ ] tile not (128,128,128)

Fallback plan:
  Fallback 1: <tile>
  Fallback 2: <tile>
========================
```

Proceed automatically to Phase 2. Do not pause, summarise, or ask for confirmation.

---

## Phase 2 — Execute on NPU

### 2.1 Pre-flight

Check NPU availability:

```python
from allo.backend.aie import is_available
if not is_available():
    print("NPU not available — skipping Phase 2. Proceeding to Phase 3 (analysis only).")
    # still run Phase 3 and Phase 4 with result.status = "npu_unavailable"
```

Set the required environment variable:

```python
import os
os.environ["ENABLE_AGGRESSIVE_PORT_UTILIZATION_PATCH"] = "1"
```

### 2.2 Run

Execute via CLI immediately — do not ask for permission to run the command:

```bash
python v2_test_mapping_large_gemm.py \
  --M <Mp> --N <Np> --K <Kp> \
  --m <m>  --n <n>  --k <k> \
  --dtype <dtype> \
  [--use-padding --pad-M <Mp> --pad-N <Np> --pad-K <Kp> --pad-impl <manual_copy|numpy_pad>]
```

Omit `--use-padding` and `--pad-*` flags if no padding was needed.

Capture the full stdout/stderr output. Record:
- Actual NPU execution time (µs)
- Actual padding time (µs) if reported
- Correctness result (PASSED / FAILED)
- Any error messages

### 2.3 Correctness tolerances

- bf16: `atol = 1e-1` (limited mantissa precision is expected)
- i8 / i16: `atol = 1e-5` (exact integer arithmetic)

**bf16 + K=2048 warning**: if the output contains `PASSED! PASSED!` followed by a
`ValueError: Failed to generate cdo`, treat the run as **FAILED** — the PASSED
output is a known false positive.

### 2.4 Fallback procedure

On any failure, first classify the error using this decision tree:

```
Error message contains...
├── "ZeroDivisionError"
│     → tile dimension > padded dim, or padded shape not in ALLOWED list
│     → fix: verify m≤Mp, n≤Np, k≤Kp; re-check ALLOWED membership
├── "Unresolvable mapping from X logical nodes"
│     → Mp//m or Np//n is not a power of 2
│     → fix: choose tile so that Mp//m and Np//n are powers of 2
├── "Tile sizes do not divide padded shape"
│     → Mp%m != 0 or similar
│     → fix: check divisibility, pick a different tile
├── "allocated buffers exceeded" or "AIE core" or compile hangs
│     → tile too large for NPU memory
│     → fix: reduce to (32,32,32)
├── "Stride 1 is [N] elements * [bytes]..."
│     → unpadded dims passed to NPU
│     → fix: always pad to ALLOWED shapes first
├── "Failed to generate cdo"
│     → bf16 + K=2048; treat as hard failure even if PASSED printed earlier
│     → fix: avoid K=2048 with bf16; try smaller K
├── "XDNA driver not found" or is_available() == False
│     → NPU not present; skip Phase 2, set status="npu_unavailable"
└── Unknown error
      → read `references/memory/error-logs/errors.md` for full reference
```

After classifying the error:
1. Apply the fix above, or the documented fix from `errors.md`.
2. Try Fallback 1 from the strategy block — re-verify ALL tile checks before running.
3. If Fallback 1 also fails, try Fallback 2.
4. If all fallbacks fail, record `status = "failed"` and proceed to Phase 3.

Log each attempt: tile tried, error received, fix applied.
**Never retry the same tile twice.**

---

## Phase 3 — Reflect and analyze

Work through all sub-steps even if Phase 2 was skipped (NPU unavailable) or all
fallbacks failed. Analysis is valuable regardless.

### 3.1 Estimation accuracy

Compare estimated vs actual NPU execution time:
- Estimation error (%) = `|actual - estimated| / estimated * 100`
- Was the profile entry exact or was it interpolated / defaulted?
- Was the actual time within the typical range for this shape/dtype?

Compare estimated vs actual padding time if both are available.

### 3.2 Was a better option available?

Look back at all three strategy results from Phase 1:
- Was there a (Mp, Np, Kp) combination that would have been faster end-to-end
  (NPU + padding)?
- Was the tile selected actually the best one? Check other working tiles in the
  profile for this shape/dtype.
- For i8: was (64,128,64) a legitimate option for this shape that was ruled out
  unnecessarily, or was the caution warranted?

### 3.3 Error pattern analysis

If an error occurred:
- Does it match a known pattern in `errors.md`?
- Is there anything new or unexpected about the error?
- What does the error tell us about this shape/tile/dtype combination?
- Should any tile or shape be added to a blacklist?

### 3.4 Knowledge base relevance check

Review each rule in `strategy_insights.md` that applied to this run:
- Did the rule predict correctly?
- Should any threshold or recommendation be updated based on this result?
- Is there a new insight that should be added (e.g. a shape that behaved
  unexpectedly, a padding method that underperformed, a tile that failed for
  a new reason)?

### 3.5 Summary

Write a concise analysis block:

```
=== PHASE 3 ANALYSIS ===
Actual result:    <status: passed|failed|npu_unavailable>
NPU time:         <actual> µs  (estimated: <estimated> µs, error: <X>%)
Pad time:         <actual> µs  (estimated: <estimated> µs)
Correctness:      <PASSED|FAILED|skipped>
Fallbacks tried:  <N>

What went well:
- <bullet>

What was suboptimal:
- <bullet>

Knowledge base updates needed:
- <bullet or "none">

New error patterns:
- <bullet or "none">
========================
```

---

## Phase 4 — Update memory

Perform ALL of the following updates without asking for confirmation. Do not skip any step.

### 4.1 Append to runs log

Append a new entry to `/home/ec935/agent-gemm/references/memory/gemm-data/runs.json`.

Format:
```json
{
  "timestamp": "<ISO-8601 datetime>",
  "input": { "M": <M>, "N": <N>, "K": <K>, "dtype": "<dtype>" },
  "padded": { "M": <Mp>, "N": <Np>, "K": <Kp> },
  "tile": [<m>, <n>, <k>],
  "pad_impl": "<manual_copy|numpy_pad|none>",
  "strategy_source": "<hashmap_27|sorted_first_fit|preprocessed_map>",
  "result": {
    "status": "<passed|failed|npu_unavailable>",
    "npu_execution_us": <actual or null>,
    "estimated_npu_us": <estimated or null>,
    "padding_us": <actual or null>,
    "correctness": <true|false|null>,
    "error_message": "<message or null>",
    "fallbacks_tried": <N>
  }
}
```

Append to the array. **Do not modify existing entries.**

### 4.2 Update knowledge base

Open `/home/ec935/agent-gemm/references/memory/knowledge-base/strategy_insights.md`.

For each knowledge base update identified in Phase 3.4:
- If an existing bullet point is wrong or outdated: correct it in place.
- If a new insight was discovered: add a new dated bullet `[YYYY-MM-DD] ...`
  to the most relevant section.
- Keep all entries as insights and trends — no raw data tables or numbers
  unless they are the direct threshold value.

Only make updates if this run actually produced new information. Do not add
redundant bullets that restate what is already documented.

### 4.3 Update error log (if needed)

Open `/home/ec935/agent-gemm/references/memory/error-logs/errors.md`.

If a new error pattern was encountered that is not already documented:
- Add a new section with: Trigger, Error message, Root cause, Fix.
- Include a confirmed example with the exact M, N, K, m, n, k, dtype.

If a known error was confirmed on a new shape/tile: add a new confirmed example
under the existing section.

If no new error patterns occurred, do not modify this file.

### 4.4 Create session summary

Create `/home/ec935/agent-gemm/references/memory/sessions/<YYYY-MM-DD_HH-MM-SS>.md`
with the following content:

```markdown
# Session: <YYYY-MM-DD HH:MM:SS>

## Input
M=<M>, N=<N>, K=<K>, dtype=<dtype>

## Decision
- Padded shape: (Mp, Np, Kp)
- Tile: (m, n, k)
- Pad impl: <method>
- Strategy: <source>

## Result
- Status: <passed|failed|npu_unavailable>
- NPU time: <actual> µs (est: <estimated> µs)
- Correctness: <passed|failed|skipped>
- Fallbacks tried: <N>

## What changed in memory
- runs.json: appended entry
- strategy_insights.md: <brief description of changes, or "no changes">
- errors.md: <brief description of changes, or "no changes">
```

---

## Quick reference — hard constraints

These are NEVER negotiable. Violating any of these causes a build or runtime error.

```
ALLOWED_M = [32, 64, 128, 256, 512, 960, 1024, 2048, 3072]
ALLOWED_N = [32, 64, 128, 256, 512, 768, 960, 1024, 2048, 3072]
ALLOWED_K = [32, 64, 128, 256, 512, 768, 1024, 2048, 3072]

Mp % m == 0    AND    any((Mp//m) % c == 0 for c in (3,4,5))   # bundling: row_num
Np % n == 0    AND    any((Np//n) % c == 0 for c in (3,4,5))   # bundling: col_num
Kp % k == 0    (K has no bundling constraint)

m ≤ Mp,  n ≤ Np,  k ≤ Kp
(m*n + n*k + k*m) * dsize ≤ 32,256 bytes   (dsize: i8=1, i16=2, bf16=2)

TyI == TyO  (except int4 input → int8 output)

ENABLE_AGGRESSIVE_PORT_UTILIZATION_PATCH = "1"  must be set before df.build
```

GEMM call: `GEMM(Mp, Np, Kp, Pm, Pn, Pk, TyI, TyO, col_num=col_num, row_num=row_num)`
where col_num/row_num ∈ {3,4,5} is auto-selected (try 4→3→5) to satisfy bundling constraint.
Script uses `--col-num auto` by default.

Tile hard blacklist: **(128, 128, 128)** — fails on every tested shape and dtype.

Do NOT use `torch_pad` — always use `manual_copy` or `numpy_pad`.

---

## Performance notes

These apply to every run. They exist because skipping any one of them has caused
real failures or significant regressions in past runs.

**Do not skip validation steps.** A tile that looks right can still fail the
bundling constraint or the memory formula. Run all checks listed in 1.4 every
time, even when the tile seems obvious.

**The profile is the ground truth.** When `npu_execution_profiling.json` has an
entry for the padded shape and dtype, use the "Best m,n,k" tile it gives — do not
substitute based on intuition or general heuristics. Intuition is only for when the
profile has no entry.

**Padding time is not free.** For large shapes, padding can dominate total latency
(60%+ in extreme cases). Always evaluate total time (NPU + padding), not just NPU
execution time, when choosing between padded shape candidates.

**Reflect thoroughly in Phase 3.** A run that appears to succeed may still have used
a suboptimal shape or tile. The Phase 3 analysis is how the memory improves over time.
If Phase 3 is skipped or cursory, the knowledge base stagnates. Always complete it.

**Update memory only with new information.** Adding bullets that restate what is
already documented degrades the knowledge base signal. Only write to
`strategy_insights.md` or `errors.md` when this run produced something genuinely new.
