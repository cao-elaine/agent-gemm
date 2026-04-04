# Session: smolvla-retile-m32-m64 (2026-03-29)

**Goal**: Re-profile SmolVLA M=32 and M=64 shapes with m=32 and m=64 tile sizes, previously untested before the `_auto_row_num` fix. Target coverage of all novel bf16 tiles for 90 shape combos from `smolvla_retile_m32_m64.json`.

**Outcome**: BLOCKED — NPU hardware unavailable for entire session.

---

## Session parameters

- `shapes_file`: `references/smol-vla-dataset/smolvla_retile_m32_m64.json`
- `dtypes`: bf16
- `max_tiles_per_combo`: all

## Shapes

- 90 unique padded combos (all M and N values already in ALLOWED lists, no padding needed)
  - 45 M=32 shapes, 895 novel tiles
  - 45 M=64 shapes, 948 novel tiles
- Total novel tiles planned: 1843
- All shapes targeted new m≥32 tiles (m=16 tiles mostly already profiled)

## NPU Status

**NPU was not accessible.** The C++ binary launched by the test script crashed with:

```
terminate called after throwing an instance of 'xrt_core::system_error'
  what():  Failed to open KMQ device (err=22): Invalid argument
Aborted (core dumped)
```

`xrt-smi` reported "0 devices found" despite `/dev/accel/accel0` existing (world-writable).
`allo.backend.aie.is_available()` returned `True` — this function only checks the `MLIR_AIE_INSTALL_DIR` env var and is NOT a reliable NPU availability check.

Likely cause: firmware crash or driver state corruption from prior session. Requires hardware reset (root privileges).

## Data files updated

- `prompts/agent-tiling-profile.json`: 1843 new entries added with `status="npu_unavailable"`
- `references/memory/error-logs/errors.md`: New section `[2026-03-29] NPU hardware unavailable — KMQ device error` documenting the failure mode and detection guidance.
- `references/memory/gemm-data/npu_execution_profiling.json`: No changes.
- `references/memory/gemm-data/profiling_errors.json`: No changes.
- `references/memory/knowledge-base/strategy_insights.md`: No changes.

## Next steps

1. Reset NPU hardware (power cycle or `sudo modprobe -r amdxdna && sudo modprobe amdxdna`)
2. Re-run this session — all 1843 tiles are marked `npu_unavailable` in agent-tiling-profile.json, so they will appear as "novel" targets again in the next session
3. Priority tiles: m=32 and m=64 variants for M=64 shapes (these are the primary goal per the shapes file description)
