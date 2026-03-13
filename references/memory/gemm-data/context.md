# Context: `gemm-data/` Directory

This directory contains all profiling data and run logs used by the agentic NPU optimizer to make strategy decisions. It is the primary data source for Phase 0 (memory loading) and Phase 4 (memory updates).

---

## Directory Structure

```
gemm-data/
├── baseline1/                     # Profiling results for Baseline 1 strategy
│   ├── context.md                 # Detailed schema and description (read this)
│   └── results.json               # Per-case timing results from Baseline 1 runs
│
├── padding/                       # Padding heuristics and profiling data
│   ├── context.md                 # Detailed schema and description (read this)
│   ├── padding_sweep_copy.json    # Profiling: manual_copy padding times by shape
│   ├── padding_sweep_numpy.json   # Profiling: numpy_pad times by shape
│   ├── padding_transition_db.json # Head-to-head: best method per input shape
│   └── padding_transition_thresholds.json  # O(1) cutoff thresholds (active policy)
│
├── npu_execution_profiling.json   # NPU execution times by (M,N,K) and tile (m,n,k)
├── profiling_errors.json          # Error tracebacks from profiling runs, keyed by shape
├── runs.json                      # Live run log appended by agent Phase 4
└── context.md                     # This file
```

---

## File Descriptions

### `baseline1/` subdirectory
Contains profiling results for **Baseline 1**, which pads inputs to the closest valid (M,N,K) shape rather than the globally optimal one. Each entry records original shape, padded shape, tile used, per-component timings (select, pad, NPU execution, unpad), padding function used, and pass/fail status.

See [baseline1/context.md](baseline1/context.md) for the full schema and field definitions.

---

### `padding/` subdirectory
Contains all data needed to choose the correct software padding implementation (`manual_copy` vs `numpy_pad`) and the cutoff thresholds that define that decision.

Key file for agent use: **`padding_transition_thresholds.json`** — this is the active O(1) runtime policy. It gives exact element-count thresholds per dtype at which the optimal padding strategy flips from `manual_copy` to `numpy_pad`.

See [padding/context.md](padding/context.md) for the full schema and field definitions for all four files.

---

### `npu_execution_profiling.json`
The core NPU execution timing database. Keyed by padded shape `"M_N_K"`, then by dtype (`i8`, `i16`, `bf16`), this file records:

- **`Best m,n,k`**: the tile that achieved the lowest average NPU execution time, and that time in microseconds
- **`Working m,n,k`**: all tile sizes that ran successfully, with their average time and trial count
- **`Failing m,n,k`**: tile sizes that failed for this shape/dtype (compile error, resource overflow, etc.)

**Agent use**: look up the padded shape key to get the best tile and its expected execution time. If a tile is listed under `Failing`, do not attempt it. If no exact key exists, fall back to the nearest shape or use the default tile `(32,32,32)`.

---

### `profiling_errors.json`
Contains raw error tracebacks captured during profiling runs. Keyed by padded shape `"M_N_K"`, then dtype, then tile size `"m_n_k"`. Values are arrays of traceback lines.

**Agent use**: cross-reference this file when a tile/shape/dtype combination fails at runtime to confirm whether it is a known failure mode. Also useful for understanding which tile sizes to avoid for a given shape before attempting execution.

---

### `runs.json`
The live run log maintained by the agent. Each entry is appended during Phase 4 and records:

- `timestamp`: ISO-8601 datetime of the run
- `input`: original M, N, K, dtype
- `padded`: padded M, N, K
- `tile`: chosen m, n, k
- `pad_impl`: padding function used (`none`, `manual_copy`, `numpy_pad`)
- `strategy_source`: how the strategy was selected (e.g. `preprocessed_map`, `hashmap_27`, `sorted_first_fit`)
- `result`: status, actual vs. estimated execution time in µs, correctness, error message if any, number of fallbacks tried

**Agent use**: read on startup (Phase 0) to review recent decisions and outcomes. Append a new entry at the end of every run (Phase 4). Do not modify existing entries.
