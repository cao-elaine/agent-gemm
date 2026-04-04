/gemm-npu-optimizer

# Agentic NPU Dimension Explorer — Master Prompt

Discovers which M and N dimensions beyond the current ALLOWED lists are valid on
the AMD Ryzen AI NPU, using the generalised bundling constraint
`(D//n) % col_num == 0` for some `col_num ∈ {3, 4, 5}` and tile size `n`.

The current ALLOWED lists are conservative (mostly powers of 2 plus 768, 960,
3072). With col_num ∈ {3,4,5}, many more dimensions are theoretically valid
(e.g. 320, 1600, 2560). This agent empirically confirms which ones actually
compile and produce correct results on the NPU.

**Execution mode: fully autonomous.** Execute all phases end-to-end without
pausing, asking for confirmation, or requesting permissions. Every Bash command
and file write is pre-authorised. If a decision is ambiguous, apply the rules
in this document and proceed.

---

## Input (fill in before running)

```
dim_max      = <N>      # upper bound of dimension range to explore; default 4096
dtypes       = <bf16|i8|i16|all>  # which dtypes to test each candidate against
tiles_per_dim = <N>     # tile combos to try per (dimension, dtype); default "all"
                         # set to "all" to try every valid tile, any integer to cap
test_M_fixed = 128      # fixed M used when probing new N dimensions (must be in
                         # current ALLOWED_M). Change only if 128 causes failures.
test_N_fixed = 512      # fixed N used when probing new M dimensions
test_K_fixed = 512      # fixed K used for all probe shapes (must be in ALLOWED_K)
```

---

## Phase 0 — Load rules and context

Read the following files in parallel before proceeding:

| File | What to extract |
|---|---|
| `references/memory/rules/npu_execution_rules.md` | Current ALLOWED_M, ALLOWED_N, ALLOWED_K; bundling constraint; memory budget formula |
| `references/memory/knowledge-base/strategy_insights.md` | Known tile pathologies, bf16 k_min rule, accuracy constraints |
| `references/memory/gemm-data/npu_execution_profiling.json` | Existing profiling data — used to check if a (shape, dtype) already has working tiles |

Write a brief context summary (3–5 bullets):
- Current ALLOWED_M and ALLOWED_N lists
- Memory budget formula and dsize per dtype
- bf16 k_min rule: `k_min = max(32, Kp//12)`
- Any known issues with specific tile sizes

---

## Phase 1 — Enumerate candidate dimensions

Run the following Python snippet (inline with Bash) to generate all candidate
dimensions:

```python
def enumerate_candidates(dim_max=4096, tile_pool=(16, 32, 64, 128, 256)):
    """Return all D in [32, dim_max] valid under the bundling constraint
    for at least one tile size in tile_pool, excluding already-ALLOWED dims."""
    ALLOWED_M = [32, 64, 128, 256, 512, 960, 1024, 2048, 3072]
    ALLOWED_N = [32, 64, 128, 256, 512, 768, 960, 1024, 2048, 3072]
    already_known = set(ALLOWED_M) | set(ALLOWED_N)

    def is_valid_quotient(q):
        # q < 3: GEMM auto-adjusts col_num down — always valid
        if q < 3:
            return True
        return any(q % c == 0 for c in (3, 4, 5))

    candidates = {}  # D -> list of (n, col_num) pairs that make it valid
    for n in tile_pool:
        for D in range(n, dim_max + 1, n):
            if D < 32:
                continue
            q = D // n
            if is_valid_quotient(q):
                col = next((c for c in (4, 3, 5) if q >= c and q % c == 0), 4)
                candidates.setdefault(D, []).append((n, col))

    new_only = {D: v for D, v in candidates.items() if D not in already_known}
    return dict(sorted(new_only.items()))

import json
result = enumerate_candidates(dim_max=4096)
# Print summary grouped by n=64 (primary) and n<64 (fallback)
primary   = {D: v for D, v in result.items() if any(n == 64 for n,c in v)}
secondary = {D: v for D, v in result.items() if D not in primary}
print(f"New candidates with n=64 valid tile: {len(primary)}")
for D, pairs in sorted(primary.items()):
    best = next((n,c) for n,c in pairs if n==64)
    print(f"  D={D:5d}  Pn={D//best[0]:3d}  col_num={best[1]}")
print(f"\nNew candidates only via n<64 tile: {len(secondary)}")
for D, pairs in sorted(secondary.items()):
    n,c = pairs[0]
    print(f"  D={D:5d}  n={n}  Pn={D//n:3d}  col_num={c}")
```

After printing the list, compile **two separate test plans**:

### Plan A — New N dimensions
For each candidate D in the primary list (valid with n=64):
- Probe shape: M=`test_M_fixed`, N=D, K=`test_K_fixed`
- Tiles to test: all valid (m, 64, k) combos satisfying:
  - `D % 64 == 0` ✓ (guaranteed by construction)
  - `(D//64) % col_num == 0` for col_num = auto-selected value
  - Memory budget: `(m*64 + 64*k + k*m) * dsize ≤ 32256`
  - bf16 k_min rule if dtype=bf16
  - `(test_K_fixed % k == 0)` and `(test_M_fixed % m == 0)`
  - Additional tiles using n=32 or n=16 if valid for that D
- Include any secondary-list D where the smallest valid n produces a quotient
  divisible by 4 (more likely to be efficient)

### Plan B — New M dimensions
Same logic but swap: probe shape M=D, N=`test_N_fixed`, K=`test_K_fixed`.
Test tiles (m=best_m_for_D, n, k) where m is the largest tile that satisfies
the M bundling constraint.

Print a summary table:
```
Plan A: <N> new N dimensions to probe   (~<X> tile runs)
Plan B: <N> new M dimensions to probe   (~<X> tile runs)
Total:  ~<Y> hardware runs
```

---

## Phase 2 — Probe execution

For each (shape, dtype, tile) in Plans A and B, run:

```bash
cd /home/ec935/vla-to-npu/gemm/scripts
/opt/anaconda3/envs/allo-base/bin/python3 v2_test_mapping_large_gemm.py \
  --M <M> --N <N> --K <K> \
  --m <m> --n <n> --k <k> \
  --dtype <dtype>
# Do NOT pass --use-padding — we are testing raw dimensions, not padded ones
```

The script auto-selects col_num/row_num. Record pass/fail and latency.

### Error handling

| Error / output | Action |
|---|---|
| `PASSED!` with latency | Record tile as working, note latency |
| `bfloat16 have accuracy issue` | Mark tile as bf16-fail (accuracy); try i8/i16 if in scope |
| `ValueError: Tile sizes ... do not divide` | Tile is invalid — skip, do not retry |
| `ValueError: Pn=... is not divisible` | Bundling constraint violated — tile invalid, skip |
| Build timeout or hang >120s | Mark shape as timeout; move on |
| Any other exception | Mark tile as fail; continue |

Do NOT stop on failures. Every failure is useful data.
Continue through all runs without pausing.

### Efficiency rules

- If a dimension's first 3 tiles all fail for the same error (e.g. all build-fail,
  not accuracy-fail), mark the dimension as "likely invalid" and skip remaining tiles.
- If a dimension gets at least 1 PASSED: mark it as **confirmed valid** and continue
  testing remaining tiles for full coverage.
- Log latency for every PASSED run.

---

## Phase 3 — Aggregate results per dimension

After all probes, build a result table:

```
=== NEW N DIMENSIONS — PROBE RESULTS ===
D     | Status         | Best tile      | Best latency | col_num | Notes
------|----------------|----------------|--------------|---------|------
192   | CONFIRMED      | (32, 64, 64)   | 312 us       | 3       |
320   | CONFIRMED      | (64, 64, 64)   | 287 us       | 5       |
1600  | CONFIRMED      | (64, 64, 64)   | 1823 us      | 5       |
1344  | ALL FAIL       | —              | —            | 3       | build error
...

=== NEW M DIMENSIONS — PROBE RESULTS ===
(same format)

Summary:
  Confirmed new N values: [192, 320, 384, ...]
  Confirmed new M values: [192, 320, 384, ...]
  Failed N values:        [...]
  Failed M values:        [...]
```

For each confirmed dimension, also record:
- Which tile size(s) are valid (n values that work)
- Which col_num was used
- Whether it's better or worse than the nearest existing ALLOWED value
  (e.g. "D=320 is 320 us, saves 160 us vs padding to N=512 at 480 us")

---

## Phase 4 — Cross-reference with practical model shapes

After confirming new dimensions, check which of them appear in real model
shapes that previously had to pad unnecessarily. Reference the SmolVLA dataset:

```bash
cat /home/ec935/agent-gemm/references/smol-vla-dataset/smolvla_gemm_sizes.md
```

For each confirmed new dimension, note if it matches any SmolVLA N or K value
that was previously being over-padded. Example:
- N=320: k_proj/v_proj output was padded to 512 — now 320 is usable directly
- N=2560: gate_proj/up_proj output was padded to 3072? — check

---

## Phase 5 — Save results

### 5.1 Write dimension discovery log

Save to `references/memory/tiling-logs/dimension_discovery_<YYYYMMDD_HHMM>.md`:

```markdown
# NPU Dimension Discovery Log — <date>

## Parameters
- dim_max: <N>
- dtypes tested: <...>
- test_M_fixed: <N>, test_N_fixed: <N>, test_K_fixed: <N>

## New confirmed N dimensions
| D | Best tile | Latency | col_num | Valid n values |
|---|-----------|---------|---------|----------------|
...

## New confirmed M dimensions
(same table)

## Failed dimensions
| D | Reason |
|---|--------|
...

## Impact on SmolVLA shapes
- <list any shapes that now avoid over-padding>

## Raw probe data
(full pass/fail table for every run)
```

### 5.2 Update profiling JSON

For every tile run that produced a timing result, append to
`references/memory/gemm-data/npu_execution_profiling.json` using the
same schema as existing entries. Key format: `<M>_<N>_<K>`.

### 5.3 Update knowledge base

Append to `references/memory/knowledge-base/strategy_insights.md`:

```markdown
- [<date>] **Dimension discovery run (dim_max=<N>)**:
  Confirmed new N dimensions: [<list>]
  Confirmed new M dimensions: [<list>]
  Key finding: <e.g. "N=320 works with col_num=5, saves ~200us vs pad-to-512">
  Failed: [<list with reason>]
```

### 5.4 Update ALLOWED lists in npu_execution_rules.md

Add all confirmed new dimensions to `ALLOWED_M` and `ALLOWED_N` in
`references/memory/rules/npu_execution_rules.md`.

**Do NOT** add failed dimensions even if they are theoretically valid — only
empirically confirmed ones belong in the ALLOWED list.

---

## Phase 6 — Final summary

Print a concise final report:

```
============================================================
DIMENSION EXPLORER — FINAL REPORT
============================================================
New confirmed N dimensions (<count>): [<sorted list>]
New confirmed M dimensions (<count>): [<sorted list>]
Dimensions that failed:               [<list>]

Top 3 practical wins (latency saved vs nearest old ALLOWED value):
  1. N=<D>: saves ~<X> us vs N=<old> (<layer> in <model>)
  2. ...
  3. ...

Files written:
  - references/memory/tiling-logs/dimension_discovery_<timestamp>.md
  - references/memory/gemm-data/npu_execution_profiling.json (updated)
  - references/memory/knowledge-base/strategy_insights.md (appended)
  - references/memory/rules/npu_execution_rules.md (ALLOWED lists updated)
============================================================
```

---

## Quick-reference constraints

```
Memory budget (per tile):
  (m*n + n*k + k*m) * dsize ≤ 32,256 bytes
  dsize: bf16=2, i8=1, i16=2, i4=0.5

bf16 k_min (accuracy):
  k ≥ max(32, Kp // 12)
  k_min=43 for Kp=512 → round up to 64

Bundling constraint (M and N only):
  (Np // n) % col_num == 0  for some col_num ∈ {3, 4, 5}
  (Mp // m) % row_num == 0  for some row_num ∈ {3, 4, 5}
  If quotient < 3: GEMM auto-adjusts — always valid
  K has NO bundling constraint — only Kp % k == 0

Tile bounds:
  m, n, k ≤ 1024
  m ≤ Mp, n ≤ Np, k ≤ Kp

Auto col_num/row_num selection:
  Try 4 first, then 3, then 5; fallback to 4 if quotient < 3
```
