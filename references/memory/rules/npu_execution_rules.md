# NPU Execution Rules

These rules govern how to call the allo GEMM NPU backend correctly.
Violations cause build-time or runtime failures.

---

## Working dimensions

```python
ALLOWED_M = [32, 64, 128, 256, 512, 1024, 2048, 3072]
ALLOWED_N = [32, 64, 128, 256, 512, 768, 1024, 2048, 3072]
ALLOWED_K = [32, 64, 128, 256, 512, 768, 1024, 2048, 3072]
```

The padded outer dims (Mp, Np, Kp) MUST be members of these lists.

---

## Tile divisibility

```
Mp % m == 0
Np % n == 0
Kp % k == 0

M/m % 4 == 0
N/n % 4 == 0
K/k % 4 == 0
```

This is checked at build time by allo. If violated you get:
`ValueError: Tile sizes (m,n,k) do not divide padded shape (Mp,Np,Kp)`

---

## Tile size

```
m, n, k <= 1024
m <= Mp
n <= Np
k <= Kp
```

This is checked at build time by allo. If violated you get:
`ValueError: Tile sizes (m,n,k) are too large`

---

## Tile memory

```
(m * n + n * k + k * m) * (dsize) <= 32,256 bytes
```

This is checked at build time by allo. If violated you get:
`ValueError: Tile sizes (m,n,k) exceed memory`

---

## Type rules

```python
TyI == TyO    # always, except:
int4 → int8   # int4 input accumulates into int8 output
```

---

## Environment variable (required)

```python
import os
os.environ["ENABLE_AGGRESSIVE_PORT_UTILIZATION_PATCH"] = "1"
```

Set this before calling `df.build`. Without it, some tiling configs fail with
resource allocation errors.

---

## Build call

```python
mod = df.build(
    top,
    project=str(project_dir),
    target="aie",
    mapping_primitives=mapping_primitives,
    profile=True,
    warmup=200,     # warmup iterations (200 is standard)
    num_iters=1000, # measurement iterations
)
```

---

## GEMM module construction

```python
top, mapping_primitives = GEMM(
    Mp, Np, Kp,          # padded outer dims
    Mp//m, Np//n, Kp//k, # num tiles per dim
    TyI, TyO             # input/output types
)
```

---

## NPU availability check

```python
from allo.backend.aie import is_available
if not is_available():
    print("NPU not available — cannot run hardware test")
    # still valid to propose strategy and update memory
```

---

## Project directory reuse

Passing the same `project_dir` to multiple `df.build` calls with different
parameters forces recompilation. This is expected behavior.
Reuse the same project dir within a run to avoid redundant recompiles.

Default project dir: `/home/ec935/vla-to-npu/gemm/top.prj`

---

## CLI invocation (via v2_test_mapping_large_gemm.py)

```bash
python v2_test_mapping_large_gemm.py \
  --M <Mp> --N <Np> --K <Kp> \
  --m <m>  --n <n>  --k <k> \
  --dtype <i8|i16|bf16> \
  --use-padding \
  --pad-M <Mp> --pad-N <Np> --pad-K <Kp> \
  --pad-impl <manual_copy|numpy_pad|torch_pad>
```

If M, N, K are already in ALLOWED lists and equal Mp, Np, Kp, omit
`--use-padding` and the `--pad-*` flags.
