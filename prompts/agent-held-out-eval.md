# Held-Out Tiling Inference Evaluation

You are being evaluated on your ability to **infer the best bf16 tile** for 50
NPU GEMM shapes that have no profiling data. Your only output is a JSON and
Markdown file of tile recommendations.

**You will not run any NPU experiments.** No hardware is required.

**Execution mode: fully autonomous. No time limit.**

---

## Setup

```
profiling_db = references/held-out-eval/npu_execution_profiling_reduced.json
input_shapes = references/held-out-eval/input_shapes.json
output_json  = references/held-out-eval/agent_recommendations.json
output_md    = references/held-out-eval/agent_recommendations.md
dtype        = bf16 
```

> The 50 shapes in `input_shapes.json` were deliberately removed from the
> reduced DB. They will not match any key. That is expected.
>
> Do NOT open `held_out_shapes.json` — it contains ground-truth answers.

---

## Phase 0 — Load context

Read these files before doing anything else:

1. **`references/held-out-eval/input_shapes.json`** — the 50 shapes to evaluate
2. **`references/held-out-eval/npu_execution_profiling_reduced.json`** — the profiling DB. Read the entire file into context now. All Phase 1 pattern extraction must be done by scanning this in-context data, not by issuing grep commands.
3. **`references/memory/rules/npu_execution_rules.md`** — hard validity constraints
4. **`references/memory/knowledge-base/strategy_insights.md`** — read this
   carefully from top to bottom. This file contains both original observations
   and later corrections to those observations. Where entries contradict each
   other, the **more recent entry** (later date) is authoritative. Pay
   particular attention to any entry marked CORRECTION or containing an
   override of a prior rule.
5. **`references/memory/error-logs/errors.md`** — known failing tile patterns

---

## Phase 1 — Extract patterns from the profiling DB

Do not evaluate any individual shape yet. First build a pattern library by
scanning the profiling DB you loaded in Phase 0. The DB is keyed by `"Mp_Np_Kp"`.

### 1.1 Per-(Mp, Kp) tile family

For each unique (Mp, Kp) pair that appears in your 50 input shapes, scan the
in-context DB for all entries with that Mp and Kp (varying Np).

For each pair, extract:
- What m values win most often?
- What n values win for small / medium / large Np?
- What k value wins — is it consistent across Np, or does it vary?

### 1.2 Per-Kp k distribution

For each unique Kp value in your 50 input shapes, scan the in-context DB for
ALL entries with that Kp across all Mp and Np.

Tally the best k values. Is there a dominant k? Does it depend on Mp or Np, or
is it stable regardless?

### 1.3 Pattern consolidation (write this out before phase 2)

After your queries, write a pattern summary structured as:

```
Kp=X: dominant best k is ?, (N shapes confirm, M exceptions). Notes: ...
Kp=Y: ...

Mp=32, Kp=X: m tends to be ?, n tends to be ? for small/large Np. Notes: ...
Mp=64, Kp=X: ...
...
```

This summary is your decision guide for Phase 2. If you skip this step, your
per-shape reasoning will be inconsistent.

---

## Phase 2 — Per-shape tile selection

Work through all 50 shapes. For each `(Mp, Np, Kp)`:

### Step A — Enumerate valid tiles

A tile `(m, n, k)` is valid if ALL hold (use power-of-2 candidates only):
1. `m ≤ Mp`, `n ≤ Np`, `k ≤ Kp`
2. `Mp % m == 0`, `Np % n == 0`, `Kp % k == 0`
3. `any((Mp // m) % c == 0 for c in (3, 4, 5))` — bundling
4. `(m*n + n*k + k*m) * 2 ≤ 32256` — DMEM budget
5. `k ≥ Kp / 12` — bf16 accuracy (soft rule; check strategy_insights.md for regime-specific exceptions)
6. NOT `(m=64 AND n=16 AND Np ≥ 480)`
7. NOT `(m, n, k) == (128, 128, 128)`
8. Not in the known-failing patterns from strategy_insights.md and errors.md

### Step B — Score candidates against evidence

For each valid tile candidate, ask:

1. **DB pattern match**: Is this tile consistent with what your Phase 1.1 pattern
   library shows for the same (Mp, Kp) pair? Does it match the dominant m, n, k
   family you found?

2. **Kp k-distribution match**: Does the k value match what your Phase 1.2
   analysis found as dominant for this Kp?

3. **strategy_insights.md alignment**: Do any entries in strategy_insights.md
   support or contradict this tile for this (Mp, Np, Kp) regime? Use the most
   recent entry when multiple entries conflict.

### Step C — Select one tile

Choose the single best tile based on the strongest combined support from:
DB proxy patterns + strategy_insights.md + cost model (if run).

If the selected tile is supported by all three sources, assign `confidence: high`.
If two of three agree, assign `medium`. If only one source supports it, assign
`low`.

### Step D — Write reasoning_summary

2–4 sentences. Must include:
- Which specific DB shapes (by key, e.g. `64_480_960`) you used as proxies
- Which pattern from Phase 1.1/1.2 applies
- What you ruled out and why

Stating a rule without naming the proxy shapes that demonstrate it does not
count as evidence.

---

## Phase 3 — Write output files

### JSON: `references/held-out-eval/agent_recommendations.json`

```json
{
  "description": "Agent tiling recommendations for 50 held-out SmolVLA shapes",
  "dtype": "bf16",
  "generated": "<ISO timestamp>",
  "shapes": [
    {
      "padded_key": "64_480_960",
      "Mp": 64, "Np": 480, "Kp": 960,
      "recommended_tile": [32, 32, 64],
      "confidence": "high",
      "reasoning_summary": "Phase 1: DB shapes 64_384_960 and 64_288_960 both show [32,32,64] as best tile. Phase 1.2: for Kp=960, all DB entries use k=64 (verified across N shapes). n=32 from Np=480 proxy pattern. Rejected m=64 because DB shapes with Np≥480 and Kp=960 all use m=32."
    }
  ]
}
```

All 50 shapes. Sort by `padded_key` alphabetically.

### Markdown: `references/held-out-eval/agent_recommendations.md`

```markdown
# Agent Held-Out Tiling Recommendations
**Generated:** <timestamp>  
**Dtype:** bf16  
**Shapes evaluated:** 50

## Pattern library (from Phase 1)
<Paste your Phase 1.3 consolidation summary here>

## Per-Shape Recommendations

| Shape | m | n | k | Conf | Proxy shapes used |
|-------|---|---|---|------|-------------------|
...

## Confidence distribution
- High: N  
- Medium: N  
- Low: N

## Low-confidence shapes and open questions
```

---

## Phase 4 — Self-review

For each recommendation, verify:

1. **Did you actually run a Phase 1.1 query for this (Mp, Kp) pair?** If not, do
   it now and check whether the result changes your recommendation.

2. **Did you actually run a Phase 1.2 query for this Kp?** If not, do it now.

3. **Is the recommended tile consistent with the Phase 1.3 summary you wrote?**
   If it deviates, explain why in the reasoning_summary.

4. **Does the tile pass all 8 validity checks from Phase 2 Step A?**

5. **Is there any entry in strategy_insights.md (including correction notes) that
   contradicts this tile for this regime?**

Fix any violation before writing files.

---

## Constraints

- You CANNOT look up the exact padded key in the reduced DB
- You CAN look up similar keys (same Mp, same Np, same Kp, nearby values)
- You CAN run the cost model as a secondary signal
- You CANNOT use `held_out_shapes.json`
- Every `reasoning_summary` must cite actual DB shapes by key name
- The Phase 1.3 pattern summary must appear in the Markdown output
