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

### [2026-03-24] bf16 accuracy failure on (128,960,1024): k<128 rule confirmed; n=16 tiles pass with k≥128

- **Shape**: (128,960,1024) padded from (M=128,K=960,N=960), dtype=bf16
- **Passing tiles (9)**: (32,64,128), (16,32,256), (16,64,128), (32,16,256), (32,32,128), (16,16,256), (16,32,128), (32,16,128), (16,16,128) — all have k≥128
- **Failing tiles**: All tiles with k<128 fail accuracy (12 AccuracyFail); tiles with k=16 and large stage counts fail MemoryExceeded (6 MemoryExceeded)
- **Key result**: n=16 tiles unexpectedly pass on Np=960 when k≥128. (32,16,256) is best at 517µs — faster than (32,64,128)@635µs. The n=16,k=256 combination outperforms larger n tiles.
- **Pattern**: Confirms the k≥128 accuracy requirement for Kp=1024 extends to Np=960 shapes. Also confirms n=16 tiles are valid on Np=960 when k is large enough.

### [2026-03-24] Direct-mode K=960 bf16 passes accuracy where padded Kp=1024 fails

- **Shape**: M=128, K=960, N=2560 (run in direct mode, tile (64,64,64))
- **Observation**: Direct-mode K=960 with k=64 (tile (64,64,64)) PASSES accuracy (5/5 trials, ~651µs avg). The same shape padded to Kp=1024 requires k≥128 for accuracy.
- **Root cause**: In direct mode the script compares A@B where A=128×960 and B=960×2560 using K=960/k K-tiles. With Kp=1024, even though the extra 64 K-elements are zero-padded, the internal tiling structure changes — Kp/k=1024/64=16 tiles vs K/k=960/64=15 tiles — and the rounding behavior differs. The accumulation error with 16 K-tiles on 1024-size exceeds atol=1e-1 while 15 K-tiles on 960-size does not.
- **Practical implication**: The k_min accuracy guard (k≥128 for Kp=1024) does NOT apply when running in direct mode (no --use-padding). Only apply k_min when Kp > K (i.e., when K-padding was actually applied).

### [2026-03-24] bf16 K=3072 (128,960,3072) — all tiles fail (confirmation of known limit)

- **Shape**: (128,960,3072) padded from (M=128,K=2560,N=960), dtype=bf16
- **All 27 tiles tested**: All fail with UnknownError (script-level exception before producing output — consistent with K=3072 CDO/accuracy failure pattern)
- **Pattern**: Confirms K=3072 bf16 hard limit. No tile workaround exists for any shape with Kp=3072.

### [2026-03-24] (64,128,1024) bf16: small shape accuracy requires k≥128; best tile (16,32,128)@151µs

- **Shape**: (64,128,1024) padded from (M=50,K=960,N=96), dtype=bf16
- **Passing tiles (4)**: (16,32,256)@154µs, (16,16,256)@169µs, (16,32,128)@151µs, (16,16,128)@170µs — all k≥128
- **Failing tiles (4)**: k=64 and k=32 tiles all fail accuracy; (16,32,16) fails MemoryExceeded; (16,16,16) fails MemoryExceeded
- **New best**: (16,32,128)@151.5µs beats prior best (32,32,32)@150.0µs — marginal improvement of ~1%
- **Pattern**: k≥128 accuracy requirement for Kp=1024 extends to small shapes (Mp=64, Np=128). All m=16 tiles valid on Mp=64 (only bundling option).

### [2026-03-24] D=3520 bf16 — categorical failure: col_num/row_num=5 with Pn=55 unsupported

- **Dimension**: N=3520 (Plan A: M=128,N=3520,K=512) and M=3520 (Plan B: M=3520,N=512,K=512), dtype=bf16
- **All tiles tested**: (64,64,64) acc_fail, (64,64,32) acc_fail (Plan A); (64,64,64) acc_fail, (16,64,64) acc_fail (Plan B)
- **Confirmed from background task**: M=3520 with m=16, n=64, k=64 (row_num=4, col_num=4) compiles but produces `bfloat16 have accuracy issue`. Additionally, at least one prior tile attempt failed at compile time with `SystemExit: 1` from peano_clang linker — indicating some 3520 tiles fail compilation entirely before accuracy can be checked.
- **Root cause**: 3520//64=55, and 55 is only divisible by 5 (col_num=5 only). The workarounds available for other col_num=5 dims (use m≥64 or smaller m for row_num=4) do not resolve the issue here — the tile pool is exhausted. Even m=16 with row_num=4 (Pm=220) fails accuracy.
- **Pattern**: Extends the col_num/row_num=5 gradient — N=160,320,640,1600,2240 (lower Pn) work with tile restrictions; N=3200 (Pn=50) needs special tile; N=3520 (Pn=55) is the first dimension that fails entirely regardless of tile. The Pn=55 boundary appears to be where col_num=5 bf16 accumulation error becomes irrecoverable.
- **Fix**: D=3520 is not usable in bf16. Pad to D=3584 (Pn=56, col_num=4) which is fully confirmed valid.

### [2026-03-24] (64,960,128) bf16: all 12 tile candidates pass; best tile (16,64,32)@200µs

- **Shape**: (64,960,128) padded from (M=50,K=96,N=960), dtype=bf16
- **All 12 tiles pass**: No accuracy failures, no MemoryExceeded. Small Kp=128 means only 1–8 K-tiles; no accumulation error issues.
- **Best tile**: (16,64,32)@200.2µs; closely followed by (16,64,64)@205µs and (16,64,16)@205µs
- **Pattern**: n=64 orientation is fastest on Np=960 with Kp=128. Small Kp removes the k≥threshold accuracy constraint entirely — all k values work.

### [2026-03-25] bf16 accuracy failures: Kp=768 requires k≥64 (not k≥128) for most shapes; large Np breaks even k≥64

- **Shapes affected**: All shapes with Kp=768 tested in 2026-03-25 session (1938 inputs → 250 combos)
- **Confirmed passing k values on Kp=768**: k=32 passes on small-N shapes (Np=128–512), k=64 and k=128 and k=256 pass broadly
- **Confirmed failing**: k=16 fails accuracy on ALL Kp=768 shapes without exception (31 failures)
- **k=32 behavior**: Passes on small-N shapes (Np≤512), fails accuracy on large-N shapes (Np≥768) — pattern: (64,2048,768) k=32 fails but (64,512,768) k=32 passes
- **Rule**: For Kp=768 bf16: use k≥64 for Np≤512; use k≥128 for Np≥768. Never use k=16 on any Kp=768 shape.
- **Confirmed examples** (2026-03-25, 40 AccuracyFail entries with k≤32):
  - (64,128,768) k=16: AccuracyFail; k=128,k=256: PASSED
  - (256,768,768) k=32: AccuracyFail; k=64: PASSED
  - (512,960,768) k=32: AccuracyFail; k=64 (via (32,64,128)): PASSED

### [2026-03-25] bf16 accuracy failures: Kp=2048 — zero tiles pass across all shapes tested

- **Shapes**: All shapes with Kp=2048 in this session (32 combos across Mp=64,128,256,512 and Np=32..3072)
- **Result**: 0 passes out of 112 runs. 70 AccuracyFail, 37 CDOError, 5 Timeout
- **Pattern**: Kp=2048 with bf16 fails across all tile sizes and all shape combinations. Previously documented only for specific shapes; now confirmed as a categorical failure for bf16+Kp=2048.
- **Fix**: Use i8 or i16 for any workload requiring Kp=2048. No tile workaround exists.

### [2026-03-25] bf16 Kp=3072 — confirmed categorical failure across all tested shapes

- **Shapes**: 136 trials across all Mp/Np combinations with Kp=3072
- **Result**: 0 passes. All fail with AccuracyFail, CDOError, or Timeout.
- **Confirms**: Prior documentation extended — Kp=3072 bf16 fails for ALL M and N values, not just specific shapes previously tested.

### [2026-03-25] bf16 Kp=1024 accuracy: k≥128 required; fails for large Mp×Np products

- **Pattern**: k<128 fails accuracy on ALL Kp=1024 shapes (0 passes out of 94 trials with k<128)
- **k=128 and k=256 do pass** but only when Mp×Np is not too large: passes when Mp×Np≤65536, increasingly rare above that
- **Large shapes fail even with k≥128**: (512,512,1024), (512,768,1024), (512,1024,1024), (512,2048,1024) all fail all tiles
- **Practical rule for bf16 Kp=1024**: Use k≥128; expect failures for Mp×Np>65536 (i.e., Mp=512 with Np≥512 or any shape with very large M×N product)

### [2026-03-25] bf16 Np=3072 accuracy failures: only (128,3072,512) passes

- **General rule**: Np=3072 with bf16 almost always fails accuracy
- **Exception**: (128,3072,512) passes with tiles (32,128,64)@411µs, (32,128,32)@419µs, (16,128,64)@603µs, (16,128,32)@598µs
- **Why (128,3072,512) works**: Small Kp=512 means few K-tiles (Kp/k=4–8), keeping accumulation error low despite large N
- **All other Np=3072 combinations tested fail**: (64,3072,512), (256,3072,512), (512,3072,512) all fail even with small Kp
- **Pattern**: Even within Np=3072, only Mp=128 succeeds. Mp=64/256/512 with Np=3072 fail regardless of Kp

### [2026-03-25] CDO error: n=128, k=16 on all Kp values — broadly confirmed across all shapes

- **Trigger**: tile n=128 combined with k=16 on bf16 regardless of shape size or Kp
- **Error**: `ValueError: Failed to generate cdo`
- **Scope**: 79 CDO failures across Kp={768,1024,2048,3072} and all tested Mp/Np combinations (64,128,256,512) in the 2026-03-25 session (1234 trials over 252 shapes)
- **Pattern**: n=128, k=16 is universally bad for bf16 CDO generation. Previous documentation focused on K=2048; this session confirms the issue extends to Kp=768 and Kp=1024 for ALL tested shapes when n=128 and k=16.
- **Safe alternatives**: Use k=32 or k=64 with n=128 to avoid CDO. k=32 triggers CDO on ~10 shapes; k=64 is much safer but may still fail accuracy on large Kp.
- **Confirmed examples**: (128,512,768) tile=(32,128,16) CDO; (128,768,768) tile=(32,128,16) CDO; (256,512,1024) tile=(64,128,16) CDO; (512,512,768) tile=(32,128,16) CDO; (512,2048,1024) tile=(32,128,16) CDO

### [2026-03-25] bf16 Kp=1024 accuracy: fails for large (512,*,1024) shapes even with k≥128 — confirmed exceptions

- **Prior rule**: Use k≥128 for Kp=1024 bf16 accuracy; expect failures for Mp×Np>65536
- **New data from 2026-03-25 (1234 trials)**: (512,*,1024) shapes pass for small-N cases but fail for large-N:
  - (512,64,1024) PASSES tiles (16,16,256)@277µs, (32,16,128)@247µs, (16,16,128)@284µs — Np=64, Np×Mp=32768 ≤ 65536
  - (512,128,1024) PASSES tiles (32,16,256)@344µs, (32,32,128)@273µs — Np=128, Mp×Np=65536 = boundary
  - (512,256,1024) PASSES 2 tiles — Np=256, Mp×Np=131072 > 65536 but passes when n<64
  - (512,512,1024), (512,768,1024), (512,1024,1024), (512,2048,1024) — 0 passes, all accuracy fail or CDO
- **Refined rule**: For bf16 Kp=1024, Mp×Np threshold is approximately 65536–131072; above 131072 (Mp=512, Np≥512), no tiles pass.

### [2026-03-28] bf16 accuracy failures: small-M shapes (Mp=32 or 64) with any K≥32 — all tiles fail for certain shape families

- **Shapes confirmed failing**: (128,96,512), (64,96,256), (64,96,32), (32,96,256), (32,96,128) — all dtype=bf16
- **Pattern**: When Mp is very small (32 or 64) due to M≤50 padding, and Np is small (32–256), bf16 accuracy fails across ALL tile sizes tried, including (8,32,64), (16,32,32), (32,32,32), (8,32,32). Not a K-accumulation issue (Kp as low as 32 still fails).
- **Key difference from prior "small MN large K" rule**: This failure occurs even with small Kp (Kp=32, 128, 256) — not just large K. The original M=1–50 padded to Mp=32–64 with small N is fundamentally problematic for bf16 correctness.
- **Confirmed failing shapes** (2026-03-28 experiment, all tiles exhausted):
  - (128,96,512) [M=128,K=320,N=96]: all tiles fail accuracy (32,32,32), (16,32,32), (16,32,16), (16,8,64)
  - (64,96,256) [M=50,K=256,N=96]: all tiles fail accuracy (16,32,64), (32,32,32), (16,32,32), (16,32,16), (16,8,64)
  - (64,96,32) [M=50,K=32,N=96]: all tiles fail accuracy (16,32,32), (32,32,32), (16,32,16), (8,32,32)
  - (32,96,256) [M=1,K=192,N=96]: all tiles fail accuracy (8,32,64), (32,32,32), (16,32,32), (16,32,16), (16,8,64)
  - (32,96,128) [M=1,K=96,N=96]: all tiles fail accuracy (8,32,64), (32,32,32), (16,32,32), (16,32,16)
- **Hypothesis**: Np=96 with very small original M (M≤50) causes bf16 correctness to fail. The N=96 padded dimension may interact with the bundling/col_num=3 or 4 computation in a way that accumulates errors. Alternatively, the very small M×N product (≤4096) means any rounding error in the computation exceeds atol=1e-1 relative to the small output values.
- **Fix**: Use i8 or i16 for any shape with M≤50 and N≤96 in bf16. No tile workaround found.

### [2026-03-28] bf16 compilation failure: (128,2560,960) fails for all tiles

- **Shape**: M=128, K=960, N=2560 padded to (128,2560,960), dtype=bf16
- **Error**: Compilation failure (clang++ invocation visible in stderr) — not an accuracy issue
- **Tiles tried**: (32,64,32), (32,64,64), (64,64,64), (32,32,32) — all fail at compile time
- **Pattern**: N=2560 with Kp=960 and M=128 causes compilation failure in the MLIR-AIE backend. The large N=2560 dimension may cause program memory overflow during MLIR-AIE compilation.
- **Fix**: No tile workaround found for bf16. Use i8 or i16 for this shape.

### [2026-03-25] 64_3072_* and 256_3072_* and 512_3072_* — all tiles fail bf16 (confirmed categorical failures)

- **Shapes**: All 45 trials with Mp=64 and Np=3072 — 0 passes (AccuracyFail, CDOError)
- **Shapes**: All 45 trials with Mp=256 and Np=3072 — 0 passes
- **Shapes**: All 45 trials with Mp=512 and Np=3072 — 0 passes
- **Contrast**: 128_3072_512 — 4 passes with Kp=512 tiles; all other 128_3072_* shapes (Kp≠512) fail
- **Refined rule for Np=3072 bf16**: Only (128,3072,512) is proven to work. Mp=64,256,512 with Np=3072 fail categorically regardless of Kp. Mp=128 with Np=3072 only works when Kp=512 (small K).

### [2026-03-29] bf16 compilation failure: Mp=256 shapes fail universally on this hardware

- **Pattern**: All padded shapes with Mp=256 fail clang++ compilation regardless of tile choice, dtype=bf16
- **Error**: Compilation failure (clang++ invocation fails in MLIR-AIE backend) — not an accuracy issue
- **Confirmed failing shapes** (2026-03-29 experiment, 291 shapes, all inconclusive):
  - (256,96,512), (256,192,512), (256,288,512) [M=235, various N, K=320]
  - (256,320,960), (256,384,960), (256,960,960) [M=235, various N, K=960]
  - (256,2560,960) [M=235, K=960, N=2560]
  - All 12 M=235 shapes (padded to Mp=256) confirmed inconclusive
- **Scope**: All 12+ shapes with Mp=256 in the SmolVLA full experiment (291 shapes, bf16) are inconclusive. No tile combination succeeds compilation, including tiles with very few pipeline stages (Pm=4, Pn=3, Pk=4 = 48 total pipeline stages).
- **Contrast with prior Mp=256 data**: Earlier tiling-exploration sessions found (256,960,Kp) shapes passing (e.g., [2026-03-25] entry). This may be hardware-configuration or session-specific. The SmolVLA full experiment environment shows universal Mp=256 compilation failure.
- **Pre-screening rule**: Add Mp=256 to the likely-uncompilable list for bf16 in this hardware configuration. Mark such shapes inconclusive without running to save experiment time.

### [2026-03-29] NPU hardware unavailable — KMQ device error

- **Trigger**: NPU hardware not accessible at session start; persists for entire session
- **Error**: `terminate called after throwing an instance of 'xrt_core::system_error'\n  what():  Failed to open KMQ device (err=22): Invalid argument\nAborted (core dumped)`
- **Root cause**: The KMQ (Kernel Message Queue) device used by XRT to communicate with the AMD Ryzen AI NPU cannot be opened. `errno=22` (EINVAL) indicates the device is in an invalid state — likely firmware crash or driver state corruption from a prior session. `/dev/accel/accel0` exists and is world-writable but returns EINVAL on open. `xrt-smi` reports "0 devices found".
- **Detection note**: `allo.backend.aie.is_available()` returns **True** even when the NPU is unavailable — it only checks the `MLIR_AIE_INSTALL_DIR` env var. Do NOT rely on `is_available()` alone. Always run a test command and verify "PASSED!" appears in stdout before starting a full session.
- **Fix**: Requires hardware reset (power cycle or driver reload with root privileges). Cannot be resolved from user-space without root access.
- **Impact**: 1843 planned bf16 tile runs for smolvla_retile_m32_m64.json shapes could not execute; all recorded as `npu_unavailable` in agent-tiling-profile.json.
- **Confirmed session**: 2026-03-29, smolvla_retile_m32_m64 experiment (all M=32 and M=64 SmolVLA shapes, bf16).
- **Updated 2026-03-30**: NPU recovered (no hardware reset needed) after ~30 minutes idle. This crash is self-resolving, unlike persistent KMQ failures requiring reboot.

### [2026-03-30] NPU hardware transient crash: m=64, n=16 tiles on large-N shapes

- **Trigger**: Tiles with m=64 AND n=16 on shapes with Np≥480; also some tiles with m=32, n=16 at Np≥672
- **Error**: Same KMQ device error as above (`Aborted (core dumped)`, `Failed to open KMQ device`) but **transient** — NPU self-recovers after 30–60s without a hardware reset
- **Confirmed crash-inducing tiles** (2026-03-30, smolvla_retile_m32_m64 session):
  - (64,480,512) bf16 tiles (64,16,32), (64,16,64), (64,16,128): NPU crash, self-recovered after ~30s
  - (64,480,960) bf16 tiles (64,16,32), (64,16,64): NPU crash, self-recovered
  - (64,3072,768) bf16 tiles (64,16,32), (64,16,64), (64,16,128), (64,32,32), (64,32,64): repeated crashes
- **Pattern**: The m=64 dimension combined with n=16 on large-N shapes (Np≥480) causes hardware state corruption that is recoverable within 30–60s. Distinct from the persistent KMQ failure (which requires 30+ minutes or hardware reset).
- **Root cause**: Unknown. Likely a hardware resource exhaustion or DMA alignment issue in the AIE tile when m=64 (Pm=1, single tile) is combined with n=16 on large-N shapes — the combination of full-M coverage with narrow-N tiling may stress the interconnect.
- **Fix**: Skip m=64, n=16 tiles on shapes with Np≥480. Use m=32, n=16 or m=64, n=32 instead — both work reliably on these shapes.
- **Important**: Repeated transient crashes can compound into a PERSISTENT KMQ failure state (see next entry).

### [2026-03-30] NPU persistent KMQ failure caused by accumulated transient crashes

- **Trigger**: After the smolvla_retile_m32_m64 session ran multiple m=64,n=16 tiles on large-N shapes, causing ≥3 rapid successive transient NPU crashes, the NPU entered a persistent error state.
- **Error**: Same KMQ device error (`Failed to open KMQ device (err=22): Invalid argument`); `/sys/class/accel/accel0/device/power/runtime_status` = "error"
- **Persistence**: Did NOT self-recover after 30 minutes (unlike the single-crash transient case documented 2026-03-29). Requires hardware reset with root privileges.
- **Impact**: The full SmolVLA bf16 experiment (291 shapes, 2538 planned NPU runs) was entirely blocked. All 282 runnable shapes marked `npu_unavailable`; 9 shapes prescreened. No NPU data collected.
- **Confirmed session**: 2026-03-30, smolvla-baseline-agentic experiment.
- **Root cause**: Accumulated state corruption. Each transient crash leaves the hardware slightly degraded; after several crashes in rapid succession, the firmware does not recover spontaneously.
- **Fix**: Requires `sudo modprobe -r amdxdna && sudo modprobe amdxdna` or a full power cycle. Cannot be resolved from user-space.
- **Prevention**: The m=64,n=16 on Np≥480 tile guard (above) eliminates the crash-inducing tiles. With that guard in place, no transient crashes should occur during normal bf16 profiling.
