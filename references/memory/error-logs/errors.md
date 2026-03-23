# NPU GEMM Error Log

Maintained by the agentic NPU optimizer. New entries appended by Phase 4.

---

## Known error patterns

### Tile divisibility
- **Trigger**: tile (m,n,k) does not evenly divide padded shape (Mp,Np,Kp)
- **Error**: `ValueError: Tile sizes (m,n,k) do not divide padded shape (Mp,Np,Kp)`
- **Fix**: ensure Mp % m == 0, Np % n == 0, Kp % k == 0

### Shape out of bounds
- **Trigger**: padded dim not in ALLOWED list
- **Error**: `ValueError: X exceeds max supported X (2048)`
- **Fix**: choose Mp, Np, Kp from ALLOWED_M, ALLOWED_N, ALLOWED_K respectively

### Type mismatch
- **Trigger**: TyI != TyO (except int4 → int8)
- **Error**: `AssertionError: TyI == TyO`
- **Fix**: use matching types; only int4 input can accumulate to int8

### Missing env var
- **Trigger**: `ENABLE_AGGRESSIVE_PORT_UTILIZATION_PATCH` not set
- **Symptom**: resource allocation failure, cryptic AIE runtime error
- **Fix**: `os.environ["ENABLE_AGGRESSIVE_PORT_UTILIZATION_PATCH"] = "1"` before build

### NPU unavailable
- **Trigger**: running on a machine without the XDNA driver
- **Error**: `XDNA driver not found` or `is_available()` returns False
- **Fix**: strategy and memory phases still run; skip Phase 2 hardware execution

### Large tile resource exhaustion
- **Trigger**: tile too large for available AIE compute resources
- **Error**: `RuntimeError: AIE core ...` or compilation hangs
- **Fix**: reduce tile size (try half the failing dimension); (64,64,64) tends
  to fail on small NPU configs; fall back to (32,32,32)

### ZeroDivisionError
- **Trigger**: tile dimension exceeds input dimension (e.g. m > M)
- **Error**: `ZeroDivisionError: integer division or modulo by zero`
- **Fix**: ensure tile size <= input size for all three dimensions

### AssertionError
- **Trigger**: Occurs when M//m is not power of 2 
- **Error**: `AssertionError: Unresolvable mapping from [x] logical nodes to [y] physical nodes`
- **Root cause**: N // n must be a power of 2; 6 or 15 is not, causing unresolvable AIE tile mapping
- **Fix**: avoid tile sizes that produce a non-power-of-2 quotient for any dimension
- **Confirmed example 1**: M=960, N=960, K=960, m=64, n=64, k=64
  (`Unresolvable mapping from 97 logical nodes to 16 physical nodes`)
  ```
  File "v2_test_mapping_large_gemm.py", line 163, in <module>
      _test_pingpong_gemm(M, N, K, M // m, N // n, K // k, int8, int8)
  File "allo/dataflow.py", line 491, in build
      aie_mod.build(...)
  File "allo/backend/aie/__init__.py", line 769, in build
      logical_node_num <= physical_node_num
  AssertionError: Unresolvable mapping from 97 logical nodes to 16 physical nodes
  ```
- **Confirmed example 2**: all cases where N=768, n=128
  (`Unresolvable mapping from 80 logical nodes to 16 physical nodes`)
  ```
  File "v2_test_mapping_large_gemm.py", line 163, in <module>
      _test_pingpong_gemm(M, N, K, M // m, N // n, K // k, int8, int8)
  File "allo/dataflow.py", line 491, in build
      aie_mod.build(...)
  File "allo/backend/aie/__init__.py", line 769, in build
      logical_node_num <= physical_node_num
  AssertionError: Unresolvable mapping from 97 logical nodes to 16 physical nodes
  ```

### RuntimeError — MLIR-AIE compilation failure
- **Trigger**: tile sizes cause buffer allocation to exceed per-core AIE data memory
- **Error**:
  ```
  error: "-":13:17: 'aie.tile' op allocated buffers exceeded available memory
  RuntimeError: Failed to compile the MLIR-AIE code
  ```
- **Sub-causes**:
  1. **Program memory overflow**: `[AIE ERROR] _XAie_LoadProgMemSection():231: Overflow of program memory` — generated program too large; reduce tile size or excessive chaining
  2. **Data memory overflow**: total buffer allocation exceeds AIE core capacity; adjust tiling strategy or use smaller tiles
- **Note**: the remaining failing tests in this category give vague errors with no additional information
- **Confirmed example**: M=241, N=241, K=241, m=64, n=64, k=64 triggers this error
  (241 is not in ALLOWED dims — this shape should be padded before tiling)

### DMA stride alignment error (new Allo commit)
- **Trigger**: input dimension is not divisible by 4 for i8, or not divisible by 2 for i16/bf16; occurs when passing unpadded dimensions directly to the NPU
- **Error**: `'aiex.npu.dma_memcpy_nd' op Stride 1 is [N] elements * [bytes] = [X] bytes, which is not divisible by 4`
- **Root cause**: AIE DMA requires all strides to be 4-byte aligned. For i8 (1 byte/element), dimensions must be divisible by 4. For i16/bf16 (2 bytes/element), dimensions must be even.
- **Fix**: always pad to ALLOWED shapes before calling the NPU — ALLOWED values are all multiples of 32, satisfying this constraint. Never pass raw unpadded dimensions.
- **Note**: this error message is more specific than earlier Allo versions; previously the same unpadded input would give a vague RuntimeError or buffer overflow error.

### ValueError
- **Trigger**: bf16 dtype with large workloads (K=2048)
- **Error**: `ValueError: Failed to generate cdo because: ...`
- **Quirk**: output may print `PASSED! PASSED! [NOTE]: bfloat16 have accuracy issue` before the ValueError — do NOT treat PASSED as success in bf16 at K=2048
- **Fix**: treat any `ValueError: Failed to generate cdo` as a hard failure regardless of preceding PASSED output; avoid K=2048 with bf16 if possible

---

## Entries added by agent runs

### [2026-03-19] Small-tile compilation failures: program memory overflow from high stage counts

- **Trigger**: tiles with very small values (m=16, n=16) on medium-to-large shapes, causing total pipeline stage count (M//m × N//n × K//k) to exceed ~256
- **Error**: Manifests as a clang++ invocation visible at the start of stderr — full MLIR-AIE compilation failure
- **Root cause**: the AIE program memory is fixed; when the total tile count is very large, the generated pipeline program overflows it. `[AIE ERROR] _XAie_LoadProgMemSection():231: Overflow of program memory` is the typical sub-cause.
- **Fix**: avoid tiles where m=16 or n=16 on shapes ≥ 256 in any dimension. Prefer m,n ≥ 32 on medium shapes and m,n ≥ 64 on large shapes.
- **Confirmed examples** (2026-03-19 exploration session):
  - (256,256,256) i8 tile=(16,16,64): M//m×N//n×K//k = 16×16×4 = 1024 stages → failed
  - (256,256,256) i8 tile=(16,32,32): 16×8×8 = 1024 stages → failed
  - (512,256,512) i8 tile=(16,16,64): 32×16×8 = 4096 stages → failed
  - (512,512,256) bf16 tile=(16,16,64): 32×32×4 = 4096 stages → failed
  - (128,128,128) bf16 tile=(32,16,16): 4×8×8 = 256 stages → failed (smallest confirmed failing count)
  - Contrast: (512,512,512) i16 tile=(32,128,64): 16×4×8 = 512 stages → PASSED; (64,64,64): 8×8×8 = 512 stages → PASSED

### [2026-03-19] AssertionError — new confirmed examples: N=768 n=128 on 1024-sized shapes

- **Shape**: (1024, 768, 768) and (1024, 768, 1024), dtype=i8 (also expected for i16/bf16)
- **Failing tiles**: (32,128,128), (128,128,32), (16,128,128), (32,128,64), (64,128,32), (128,128,16) — all have n=128 on Np=768 → N//n = 6 (not power of 2)
- **Pattern**: n=128 on Np=768 gives N//n=6; confirmed to fail with `AssertionError: Unresolvable mapping from X logical nodes to 16 physical nodes`
- **Safe alternatives**: n=32 (N//n=24) and n=64 (N//n=12) both work on N=768 — non-power-of-2 quotients are acceptable for N//n as long as n ≠ 128

### [2026-03-13] CDOError: k=16 on K=768 for bf16

- **Shape**: (1024, 768, 768), dtype=bf16
- **Tiles**: (64,64,16), (32,64,16), (32,32,16) — all failed with `ValueError: Failed to generate cdo`
- **Pattern**: K=768 with k=16 → 48 k-tiles. CDO generation fails — not solely a k=2048 issue.
- **Confirmed CDO-safe tiles**: k=32 or k=64 on K=768 with bf16 work if the tile passes memory constraints.

### [2026-03-13] bf16 accuracy failures on (1024,768,768): tile-dependent

- **Shape**: (1024, 768, 768), dtype=bf16
- **Failing tiles (accuracy)**: (128,64,32), (128,32,32), (64,32,32) — execution completes but correctness check fails (atol=1e-1).
- **Passing tiles**: (64,32,64), (128,32,64), (32,32,32), (64,64,64), (32,64,32), (32,32,64), (64,64,32) all pass.
- **Pattern**: Tiles with very small k (k=32) combined with larger m seem more prone to bf16 accuracy failures on this shape. Tiles with k≥64 were more reliable.

### [2026-03-20] bf16 CDO failure confirmed for K=3072 across all tile sizes

- **Shape**: M=1024, K=3072, N=768 (padded K=3072), dtype=bf16
- **All tiles tried**: (64,64,64), (64,64,32), (32,32,32) — all fail with `ValueError: Failed to generate cdo`
- **Pattern**: K=3072 is beyond the bf16 CDO limit (previously documented for K=2048; now confirmed extends to K=3072)
- **Fix**: bf16 with K>2048 is unsupported on this hardware configuration; avoid or use i8/i16 instead

### [2026-03-20] bf16 accuracy failure on K-dominated shapes with very small M×N

- **Shapes**: M=1, K=960, N=32 padded to (32,1024,32); M=50, K=960, N=96 padded to (64,1024,128)
- **Pattern**: When M and N are very small (M≤50, N≤128) but K is large (K≥960), bf16 accumulation accuracy degrades beyond atol=1e-1 tolerance across ALL tile sizes
- **Hypothesis**: floating point accumulation error grows with K; at K=960–1024 with small output matrices, the rounding errors exceed the bfloat16 tolerance
- **Fix**: no known tile workaround; consider i8 or i16 for shapes with K>>M, K>>N

### [2026-03-20] bf16 accuracy failure on K=2560 (K > 2048)

- **Shape**: M=128, K=2560, N=960 (no padding possible, K=2560 exceeds ALLOWED_K max), dtype=bf16
- **Pattern**: K=2560 causes bf16 accuracy failure for all tested tiles (64,64,64) and (32,32,32)
- **Fix**: consistent with K>2048 accuracy boundary for bf16; use i8/i16 or restructure computation

### [2026-03-21] bf16 accuracy failure on (1024,768,3072): all tiles fail accuracy, no CDO error

- **Shape**: M=1024, K=3072, N=768 (padded (1024,768,3072)), dtype=bf16
- **Pattern**: Most tiles that fit in memory complete execution but produce incorrect results (AccuracyFail). Very large-stage tiles (stages>20K) timeout (MemoryExceeded). CDOError does NOT appear for all tiles — some compile and execute but fail numerically.
- **Prior entry** (2026-03-20) said "all fail with CDO" — this was based on 3 tiles. With 30 tiles tested: 13 AccuracyFail, 17 MemoryExceeded, 0 CDOError, 0 passes.
- **Correction**: K=3072 bf16 fails accuracy (not CDO) for tiles that compile successfully. CDO failures only occur for very small-k tiles on K=768; at K=3072 the CDO can compile but results are numerically wrong.
- **Confirmed examples** (session 2026-03-21): (32,64,128) AccuracyFail 3965µs, (64,32,128) AccuracyFail 3972µs, (16,64,128) AccuracyFail 6356µs, (32,32,128) AccuracyFail 5185µs

### [2026-03-21] bf16 accuracy failure on (128,1024,1024) and (128,512,1024): K=1024 with K>>M

- **Shapes**: (128,1024,1024) padded from (M=128,K=960,N=960); (128,512,1024) padded from (M=128,K=960,N=320)
- **Pattern**: Tiles with k≤64 consistently fail accuracy. Tiles with k≥128 tend to pass. Tiles with n=128 frequently fail accuracy even with k=64.
- **Confirmed passing tiles on (128,1024,1024)**: (32,64,128), (16,64,128), (32,16,256), (32,32,128), (16,16,256), (16,32,128), (32,16,128), (16,16,128) — all have k≥128
- **Confirmed passing tiles on (128,512,1024)**: (32,64,128), (16,64,128), (32,16,256), (32,32,128), (16,32,256), (32,16,128), (16,16,128), (32,64,64), (16,64,64) — mix of k=64 and k≥128
- **Pattern**: bf16 accuracy on K=1024 shapes requires k≥64; smaller k values accumulate rounding error beyond atol=1e-1

### [2026-03-21] bf16 accuracy failure on (128,1024,3072): K=3072 all tiles fail

- **Shape**: (128,1024,3072) padded from (M=128,K=2560,N=960), dtype=bf16
- **All 36 tiles tested**: all fail with AccuracyFail. Best avg measured was ~313µs but all numerically incorrect.
- **Pattern**: Consistent with K>2048 accuracy boundary for bf16 across all shapes and tile sizes

### [2026-03-21] bf16 accuracy failure on (128,3072,1024): large Np + Kp=1024 all tiles fail

- **Shape**: (128,3072,1024) padded from (M=128,K=960,N=2560), dtype=bf16
- **Tiles tried**: (32,64,128), (32,64,256), (32,64,512), (32,64,1024), (64,64,64), (32,32,32) — all fail with AccuracyError
- **Pattern**: Even with k≥128 (which works on other Kp=1024 shapes), the combination of Np=3072 (N=2560 padded) with K=960 and M=128 causes bf16 accuracy failure
- **Hypothesis**: The very large N dimension (3072) changes the accumulation behavior such that even per-row errors accumulate beyond atol=1e-1
- **Fix**: No tile workaround found; use i8/i16 for (M=128, K=960, N=2560) bf16 workloads
- **Extended (2026-03-21 session)**: All 34 remaining novel tile candidates exhaustively tested. 34/34 AccuracyFail. No exceptions — confirmed categorical failure for all (m,n,k) combinations with m∈{16,32}, n∈{16,32,64,128,256,512} k∈{16,32,64,128,256,512}.

### [2026-03-21] bf16 MemoryExceeded (timeout) patterns on large Np shapes

- **Shapes affected**: (1024,768,768), (1024,768,3072), (1024,3072,768) — all large Np
- **Trigger**: Small tile sizes (especially m=16 or n=16 with k≤64) on shapes with large Np (768 or 3072). Stage count (M//m × N//n × K//k) exceeds compilation limit.
- **Examples**:
  - (1024,768,768) tile=(16,16,16): 64×48×48=147456 stages → MemoryExceeded (timeout)
  - (1024,768,768) tile=(32,16,32): 32×48×24=36864 stages → MemoryExceeded (timeout)
  - (1024,3072,768): tiles with m=16 or n≤32 nearly all timeout
- **Pattern**: Confirms that for Np=768 or Np=3072, tiles with very small m or n values cause massive stage counts due to the large N quotient. The n=128 tile gives N//n=6 or 24, keeping stage count manageable; n=16 gives N//n=48 or 192, exploding stage count.
- **Fix**: For Np=768 and Np=3072 shapes, use m≥32 and n≥64 to keep stage counts reasonable.

### [2026-03-22] i8 correctness failures on tiles with very small m×n product (m≤16, n≤32 or m≤32, n≤16)

- **Trigger**: Tiles where both m and n are very small (m≤16 AND n≤32, or m≤32 AND n≤16) on i8 shapes with any shape size
- **Error**: Script completes and produces output, but correctness check fails — stdout shows array mismatch values like `[ 53, -40, -15, ..., 21, 94, 99],...`; no "PASSED!" printed
- **Root cause**: Likely the same program memory overflow described above for small tiles. When the pipeline program overflows memory, instead of a hard compile-time error, it partially executes and produces silently corrupted results. i8 should be exact (atol=1e-5), so any deviation is a hard failure.
- **Distinguishing feature**: Unlike the documented MemoryExceeded (which is a compile/runtime crash), this failure mode allows the script to run to completion but produces numerically wrong output.
- **Fix**: Avoid tiles with m≤16 AND n≤32 (or m≤32 AND n≤16) on i8. Use m≥32 and n≥32 minimum, or m≥64/n≥64 for larger shapes.
- **Confirmed examples** (2026-03-22 exploration session, 25 failures total):
  - (64,128,128) i8 tile=(16,16,64) → correctness fail
  - (64,128,128) i8 tile=(16,32,32) → correctness fail
  - (128,128,64) i8 tile=(16,16,64), (16,32,32), (32,16,32), (16,32,16) → all correctness fail
  - (256,256,512) i8 tile=(16,16,64), (16,32,32), (32,16,32) → correctness fail
  - (512,512,64) i8 tile=(16,16,64), (16,32,32), (32,16,32), (16,32,16) → correctness fail
- **Note**: i16 and bf16 tiles with the same stage counts fail with a hard error (MemoryExceeded/timeout); i8 uniquely produces silent corruption. This makes i8 correctness failures harder to detect if the error parser only checks for "PASSED!".

### [2026-03-22] bf16 "CMake developer warning" failure on certain tile/shape combos

- **Trigger**: Specific bf16 tile + shape combinations, especially medium-to-large shapes with small m or n values; also large asymmetric shapes (Mp=2048, small Np)
- **Error**: `This warning is for project developers. Use -Wno-dev to suppress it.` appears as the last stderr line; no PASSED! output; script fails
- **Root cause**: A CMake/build warning is printed after a deeper build failure (likely program memory overflow or resource exhaustion), drowning out the actual error message. The underlying failure is the same as the MemoryExceeded/program-overflow pattern but the error text is hidden by CMake output.
- **Confirmed examples** (2026-03-22, 22 failures total):
  - (256,256,256) bf16 tile=(16,16,64); (256,512,512) bf16 tile=(16,16,64), (16,16,32)
  - (2048,768,768) bf16 tiles with small n: (128,64,32), (128,32,32), (64,32,32), (128,16,32), (256,16,32)
  - (1024,64,768) bf16 tiles: (16,16,16), (256,16,32), (16,16,64), (16,16,32)
  - (2048,64,1024) bf16: 9 tiles including (128,16,64), (64,16,64), (128,16,32), (64,16,32), (32,16,64) etc.
- **Fix**: Treat this error the same as MemoryExceeded. For bf16 on large shapes, apply the same rule: use m≥32 and n≥64 to keep stage counts manageable.

### [2026-03-22] Timeout failures on large i16 shapes with very small tiles

- **Trigger**: Tiles with m=16 or n=16 on large i16 shapes (Mp=1024, Np=768, Kp=768 or larger)
- **Pattern**: Compilation exceeds the 180s timeout — confirmed for (1024,768,768) i16 with (16,16,16), (16,16,256), (16,16,128), (16,16,64), (16,32,32) and many others (109 timeout failures this session)
- **Contrast**: The same (16,16,k) tiles compile in ~10s on small shapes (64–256 padded dims). The compile time grows with stage count: (1024,768,768) with m=16, n=16 → 64×48×k stages → massive pipeline graph.
- **Extends existing MemoryExceeded documentation**: confirms the stage-count overflow issue applies to i16 large shapes just as it does to bf16. Previously confirmed only for bf16 shapes.
- **Fix**: Same as general small-tile rule — use m≥64 and n≥64 on shapes with any dimension ≥768.

### [2026-03-21] n=16 tiles unexpectedly pass on (1024,768,768) bf16

- **Shape**: (1024,768,768), dtype=bf16
- **Passing tiles with n=16**: (128,16,64)=1543µs, (64,16,64)=1773µs, (64,16,32)=1816µs, (32,16,256)=2108µs, (32,16,64)=2159µs
- **Anomaly**: These have N//n = 768//16 = 48, which is non-power-of-2 and not in DIVISIBILITY_ONLY_DIMS. Per the standard rules these should fail UnresolvableMapping, but they pass.
- **Hypothesis**: The NPU may accept non-power-of-2 N//n quotients for smaller n values (n=16) where the mapping constraint is satisfied by a different mechanism.
- **Performance**: All these tiles are 2–3× slower than (64,64,64)=757µs. They provide coverage but are not optimal choices.

### [2026-03-22] Accuracy failure mode differs between padded-run and direct-dims-run for (128,3072,1024) bf16

- **Shape**: (128,3072,1024), dtype=bf16
- **Padded-run failure** (using --use-padding, original M=128,K=960,N=2560 → padded Mp=128,Kp=1024,Np=3072): All tested tiles including (32,64,128), (32,64,256), (32,32,32) fail accuracy (confirmed 2026-03-21, 40+ tiles exhaustively tested)
- **Direct-dims-run behavior** (passing (128,3072,1024) directly without --use-padding): tile (32,64,128) PASSES accuracy, 5/5 trials at 794.2µs avg. Tile (32,32,32) still fails accuracy even in direct mode.
- **Root cause**: The correctness check compares against different ground-truth sizes. Padded run: A@B where A=128×960, B=960×2560 (original small matrices, accumulated in bf16 over K=960 → accumulation error). Direct run: A@B where A=128×1024, B=1024×3072 (full padded size, different random inputs, k=128 tile gives 8 K-tiles → lower accumulation error).
- **Practical implication**: For SmolVLA M=128,K=960,N=2560 workloads, bf16 is unreliable due to accumulation errors on the original dimensions. The direct-dims pass is for a different computation (padded shape, different inputs).
- **Script limitation**: N=3072 cannot be used with --use-padding flag (exceeds script's ALLOWED_N max of 2048). Direct-dims mode required for any testing of this shape.
- **Confirmed passing tile on direct dims** (2026-03-22): (32,64,128) @ 794.2µs avg / 739µs min. Tile (32,32,32) fails accuracy even in direct mode.
- **Fix**: No tile workaround for the padded/original-size case. Use i8 or i16 for (M=128, K=960, N=2560) in production.
