# Session: smolvla-baseline-agentic (2026-03-30)

**Goal**: Run the full SmolVLA NPU Baseline vs Agentic experiment across all 291 unique
GEMM shapes from `references/smol-vla-dataset/smolvla_gemm_shapes.json`, dtype=bf16,
3 trials per shape per strategy group (Baseline 0, Baseline 1, Agentic).

**Outcome**: BLOCKED — NPU hardware in persistent error state from prior retile session.

---

## Session parameters

- `shapes_file`: `references/smol-vla-dataset/smolvla_gemm_shapes.json`
- `dtypes`: bf16
- `trials_per_group`: 3
- `total_shapes`: 291
- `shapes_prescreened`: 9
- `shapes_npu_unavailable`: 282
- `shapes_run`: 0

## Pre-screening rules applied (bf16)

9 shapes skipped before any NPU attempt:
- `K > 2048` → CDO/accuracy categorical failure (confirmed)
- `M ≤ 50 AND N ≤ 96 AND K ≥ 960` → bf16 accumulation error categorical failure (confirmed)

## Strategy plan summary (computed, not executed)

All 282 runnable shapes had full strategies computed and stored in results.json:

| Metric | Count |
|--------|-------|
| Total NPU runs planned | 2538 (282 × 3 groups × 3 trials) |
| Agentic shape overrides (B2 shape ≠ Agentic shape) | 1 |
| Shapes where B0 tile ≠ B2 tile | 212 |
| Shapes where B2 tile ≠ Agentic tile | 9 |

The one Agentic shape override: M=235, K=960, N=320 → Agentic selects Np=384 (not Np=320)
because Np=320 has col_num=5 constraints that limit tile choices.

## NPU Status

**NPU was not accessible.** The C++ binary launched by the test script crashed immediately:

```
terminate called after throwing an instance of 'xrt_core::system_error'
  what():  Failed to open KMQ device (err=22): Invalid argument
Aborted (core dumped)
```

`/sys/class/accel/accel0/device/power/runtime_status` = "error"

**Root cause**: The earlier smolvla_retile_m32_m64 session (same day) ran multiple m=64,n=16
tiles on large-N shapes, causing ≥3 successive transient NPU hardware crashes. These
accumulated into a persistent error state (unlike single transient crashes which self-recover
in ~30s). NPU did not recover after 30 minutes of waiting.

**Fix required**: `sudo modprobe -r amdxdna && sudo modprobe amdxdna` or full power cycle
(root privileges required).

## Tile guard added

The `run_smolvla_bf16_experiment.py` script was updated to preemptively exclude m=64,n=16
tiles from the `failing` set when Np≥480, preventing future crash-inducing runs:

```python
# In strategy_agentic():
if Np >= 480:
    for k_val in [16, 32, 64, 128, 256, 512, 768, 1024]:
        failing.add((64, 16, k_val))
```

## Data files updated

- `references/experiment-results/test_n_20260330/results.json`: Created with all 291 shapes
  (9 prescreened_skip, 282 npu_unavailable). Full strategy plans stored for all runnable shapes.
- `references/memory/error-logs/errors.md`: New section documenting the persistent KMQ failure
  caused by accumulated transient crashes.
- `references/memory/gemm-data/npu_execution_profiling.json`: No changes (no runs executed).
- `references/memory/knowledge-base/strategy_insights.md`: No changes.

## Next steps

1. Reset NPU hardware (power cycle or `sudo modprobe -r amdxdna && sudo modprobe amdxdna`)
2. Re-run experiment — all 282 shapes are marked `npu_unavailable` in results.json so a resume
   run will re-attempt all of them
3. The m=64,n=16 crash guard is already in place — no further prep needed
4. Also pending: smolvla_retile_m32_m64 shapes (1843 tiles, from 2026-03-29 session) remain
   unrun for the same hardware reason
