---
name: gemm-npu-optimizer
description: Optimizes GEMM matrix multiplications for AMD Ryzen AI NPUs. Selects the best padded shape, tile size, and padding implementation to minimize end-to-end latency. Use this skill whenever the user provides M, N, K dimensions and a dtype (i8, i16, bf16) for NPU execution — even if phrased as "run my matrix multiply", "what tile should I use", "optimize this kernel", "my GEMM is slow", "fix my NPU alignment error", or "determine tiling dimensions". Also activates for any NPU compilation failure, tiling error, or padding strategy question on AMD Ryzen AI hardware.
---

# GEMM NPU Optimizer

You are an agentic optimizer for GEMM on AMD Ryzen AI NPUs. Your goal is to select
the strategy that minimizes total end-to-end latency: padding time + NPU execution
time + unpadding time.

**Execution mode: fully autonomous.** Run all phases (0 → 1 → 2 → 3 → 4) without
stopping to ask for confirmation, permission, or approval at any point. Read files,
write files, and execute commands as needed without pausing for user input.

Output quality is more important than speed. Never skip a validation step —
a wrong tile or padded shape causes a hard compilation failure or a 3-4x regression.

---

## Directory layout

All persistent data lives under `references/memory/`. Read from here; write back
here after each run.

```
agent-gemm/
├── SKILL.md                          this file
├── references/
│   ├── memory/
│   │   ├── rules/npu_execution_rules.md          ALWAYS read — hard constraints
│   │   ├── knowledge-base/strategy_insights.md   ALWAYS read — padding + tile heuristics
│   │   ├── error-logs/errors.md                  ALWAYS read — known failure patterns
│   │   ├── gemm-data/
│   │   │   ├── npu_execution_profiling.json      read Phase 1 — best tile lookup
│   │   │   ├── padding/padding_transition_thresholds.json  read Phase 1
│   │   │   ├── profiling_errors.json             read when tile looks risky
│   │   │   └── runs.json                         read Phase 0; append Phase 4
│   │   └── sessions/                             write session summary Phase 4
│   └── documentation/                            AMD/Allo hardware docs
├── scripts/                          helper scripts (parse profiling JSON, etc.)
└── prompts/agentic-npu-workflow.md   full step-by-step reference (load if needed)
```

---

## Phase 0 — Load memory

Read these files in parallel before doing anything else:

| File | Purpose |
|------|---------|
| `references/memory/rules/npu_execution_rules.md` | ALLOWED shape lists, tile divisibility rules, tile memory formula, env var, CLI format |
| `references/memory/knowledge-base/strategy_insights.md` | Padding thresholds, tile defaults and blacklists, hybrid shape-selection rule, shape pathologies |
| `references/memory/error-logs/errors.md` | Known failure patterns for this shape, tile, or dtype |
| `references/memory/gemm-data/runs.json` | Prior run outcomes on the same or similar shape |

Load these only when the condition applies:

| File | Load when |
|------|-----------|
| `references/memory/gemm-data/npu_execution_profiling.json` | Phase 1 — looking up best tile and execution time |
| `references/memory/gemm-data/padding/padding_transition_thresholds.json` | Exact ab_elements threshold needed |
| `references/memory/gemm-data/profiling_errors.json` | Tile candidate is (64,128,64) or Np is small |
| `prompts/agentic-npu-workflow.md` | You need the detailed sub-step decision logic |

Write a 3-6 bullet context summary before proceeding:
- Are M, N, K already in ALLOWED lists?
- What are the closest padded shape candidates?
- Does a profile entry likely exist for the padded shape?
- Any shape pathologies (M > 1024, small N, power-of-2 boundary, K >> M or N)?
- Most relevant heuristics from strategy_insights.md for this input

---

## Phase 1 — Propose a strategy

For the full sub-step decision logic, read `prompts/agentic-npu-workflow.md`.

At a high level:

**1. Find padded shape candidates** — for each dim, all ALLOWED values >= the input.
```
ALLOWED_M = [32, 64, 128, 256, 512, 1024, 2048, 3072]
ALLOWED_N = [32, 64, 128, 256, 512, 768, 1024, 2048, 3072]
ALLOWED_K = [32, 64, 128, 256, 512, 768, 1024, 2048, 3072]
```

**2. Run all three strategies** and record their chosen (Mp, Np, Kp):
- `hashmap_27` — checks 27 nearest candidates; best for reducing padding overhead
- `sorted_first_fit` — guarantees fastest NPU time; may pick large padding
- `preprocessed_map` — O(1) lookup; best default when shape is already profiled

**3. Apply hybrid shape-selection rule:**
- Small inputs (all dims roughly 64 or below): pick shape with fastest NPU time — padding cost is ~constant at this scale
- Large inputs (any dim above 64): pick the closest (smallest) valid shape — padding and NPU time both grow with size
- M > 1024: only Mp=2048 is valid; all strategies agree — use preprocessed_map and move on

**4. Select tile (m, n, k):**

Look up (Mp, Np, Kp) + dtype in `npu_execution_profiling.json`. Use the "Best m,n,k"
entry if it exists. If no profile entry:
- Np <= 64: use (32,32,32) — larger tiles fail divisibility on small N
- Otherwise: use (64,64,64)

CRITICAL — verify ALL checks before finalizing the tile:
```
CHECK 1:  Mp % m == 0   AND   (Mp//m) % 4 == 0
CHECK 2:  Np % n == 0   AND   (Np//n) % 4 == 0
CHECK 3:  Kp % k == 0   AND   (Kp//k) % 4 == 0
CHECK 4:  m <= Mp,  n <= Np,  k <= Kp
CHECK 5:  (m*n + n*k + k*m) * dsize <= 32256   (dsize: i8=1, i16=2, bf16=2)
CHECK 6:  tile not in Failing list for this shape/dtype in profiling_errors.json
CHECK 7:  tile != (128,128,128)   hard blacklist — fails on every shape
```
If any check fails, fall back: (64,64,32) → (64,64,64) → (32,32,32). Re-verify all
checks after each fallback.

Do not use (64,128,64) as a default — it accounts for 35% of documented tile failures.
Only use it if it explicitly appears as "Best" in the profile for this exact shape.

**5. Select padding implementation** using `ab_elements = M*K + K*N`:
- i8:  manual_copy if ab_elements <= 1653750, else numpy_pad
- i16: manual_copy if ab_elements <= 1213000, else numpy_pad
- bf16: manual_copy if ab_elements <= 571779,  else numpy_pad

Never use torch_pad — it is 4-15x slower than manual_copy across all dtypes.
If M==Mp, N==Np, K==Kp (no padding needed): set pad_impl = none.

**6. Output the strategy block** — exact format in `prompts/agentic-npu-workflow.md` §1.6.

Proceed automatically to Phase 2 without asking for confirmation.

---

## Phase 2 — Execute on NPU

**Pre-flight:**
```python
from allo.backend.aie import is_available
# If False: skip Phase 2, set status="npu_unavailable", proceed to Phase 3

import os
os.environ["ENABLE_AGGRESSIVE_PORT_UTILIZATION_PATCH"] = "1"  # required before df.build
```

**Run:**
```bash
python v2_test_mapping_large_gemm.py \
  --M Mp --N Np --K Kp --m m --n n --k k --dtype DTYPE \
  [--use-padding --pad-M Mp --pad-N Np --pad-K Kp --pad-impl manual_copy|numpy_pad]
```

**Correctness tolerances:** bf16: atol=1e-1 — i8/i16: atol=1e-5

**On failure — classify the error, then apply the fix:**

```
Error contains...
"ZeroDivisionError"
  → tile > padded dim, or shape not in ALLOWED list
  → fix: verify m<=Mp, n<=Np, k<=Kp; check ALLOWED membership

"Unresolvable mapping from X logical nodes"
  → Mp//m or Np//n is not a power of 2
  → fix: choose tile so Mp//m, Np//n, Kp//k are all powers of 2

"Tile sizes do not divide"
  → Mp%m != 0 or similar
  → fix: verify divisibility before retrying

"allocated buffers exceeded" or "AIE core" or compile hang
  → tile too large for NPU memory
  → fix: reduce to (32,32,32)

"Stride 1 is [N] elements * [bytes]..."
  → raw unpadded dims passed to NPU
  → fix: always pad to ALLOWED shapes first

"Failed to generate cdo"
  → bf16 + K=2048 known failure; preceding PASSED output is a false positive
  → fix: treat as hard failure; avoid K=2048 with bf16

"XDNA driver not found"
  → NPU unavailable; skip Phase 2, set status="npu_unavailable"

Unknown
  → read references/memory/error-logs/errors.md for full reference
```

After applying fix: try Fallback 1 (re-verify all tile checks first), then Fallback 2.
If all fallbacks fail: record status="failed", proceed to Phase 3.
Never retry the same tile twice.

---

## Phase 3 — Reflect and analyze

Complete this phase even when Phase 2 was skipped or failed. This is how memory improves.

Answer all of the following:
1. **Estimation accuracy** — what was the % error between estimated and actual NPU time? Was the profile entry exact or defaulted?
2. **Better option available?** — was there a (Mp, Np, Kp) or tile that would have been faster end-to-end? Check all three strategy results.
3. **Error analysis** (if failure occurred) — does it match a known pattern in errors.md? Anything new?
4. **Knowledge base check** — did any rule in strategy_insights.md predict correctly? Should any threshold or recommendation be updated?

Write the Phase 3 analysis block — format in `prompts/agentic-npu-workflow.md` §3.5.

---

## Phase 4 — Update memory

Complete all four steps. Do not skip any.

1. **Append to `references/memory/gemm-data/runs.json`** — format in `prompts/agentic-npu-workflow.md` §4.1. Never modify existing entries.
2. **Update `references/memory/knowledge-base/strategy_insights.md`** — add a dated bullet [YYYY-MM-DD] for any new insight. Correct wrong bullets in place. Do not add bullets that restate what is already there.
3. **Update `references/memory/error-logs/errors.md`** — only if a new error pattern was seen or a new confirmed example for an existing pattern.
4. **Create `references/memory/sessions/YYYY-MM-DD_HH-MM-SS.md`** — brief session summary (format in `prompts/agentic-npu-workflow.md` §4.4).

---

## Quick-reference constraints

```
ALLOWED_M = [32, 64, 128, 256, 512, 1024, 2048, 3072]
ALLOWED_N = [32, 64, 128, 256, 512, 768, 1024, 2048, 3072]
ALLOWED_K = [32, 64, 128, 256, 512, 768, 1024, 2048, 3072]

Mp%m==0  AND  (Mp//m)%4==0       (same for N and K)
(m*n + n*k + k*m) * dsize <= 32256
TyI == TyO   (except int4 input accumulates to int8 output)
ENABLE_AGGRESSIVE_PORT_UTILIZATION_PATCH = "1" before df.build
```

Tile blacklist: (128,128,128) always fails. Never use torch_pad.

---

## Performance notes

**Do not skip tile validation.** All 7 checks in Phase 1 step 4 must pass. A tile
that looks correct can still fail (Mp//m)%4==0 or the memory formula. Run every
check every time.

**The profile is the ground truth.** When npu_execution_profiling.json has a "Best
m,n,k" entry for the padded shape and dtype, use it. Do not substitute based on
general heuristics. Heuristics are only for when no profile entry exists.

**Padding time is not free.** For large shapes it can exceed NPU execution time
(observed 60%+ of total latency). Always evaluate NPU time + padding time together
when comparing candidate shapes — never NPU time alone.

**Phase 3 is not optional.** A run that succeeds may still have used a suboptimal
shape or tile. The reflection in Phase 3 is what makes memory useful over time.
Skipping it means the knowledge base stagnates.

**Memory updates: quality over quantity.** Only write to strategy_insights.md or
errors.md when this run produced genuinely new information. Restating existing
bullets degrades signal.
