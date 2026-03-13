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

(none yet — will be appended as sessions run)
