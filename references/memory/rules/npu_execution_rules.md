# NPU Execution Rules

These rules govern how to call the allo GEMM NPU backend correctly.
Violations cause build-time or runtime failures.

---

## Working dimensions

```python
ALLOWED_M = [32, 64, 96, 128, 160, 192, 256, 288, 320, 384, 448, 480, 512, 576, 640, 672, 960, 1024, 1152, 1280, 1344, 1536, 1600, 1728, 1792, 1920, 2048, 2112, 2240, 2304, 2496, 2560, 2688, 2816, 2880, 3072, 3200, 3264, 3328, 3456, 3584, 3648, 3840, 4032, 4096]
ALLOWED_N = [32, 64, 96, 128, 160, 192, 256, 288, 320, 384, 448, 480, 512, 576, 640, 672, 768, 960, 1024, 1152, 1280, 1344, 1536, 1600, 1728, 1792, 1920, 2048, 2112, 2240, 2304, 2496, 2560, 2688, 2816, 2880, 3072, 3200, 3264, 3328, 3456, 3584, 3648, 3840, 4032, 4096]
ALLOWED_K = [32, 64, 128, 256, 512, 768, 960, 1024, 2048, 3072]
```

**Note (2026-03-24 session 1)**: New M and N values added: 192, 320, 384, 576, 640, 1152, 1280, 1344, 1536, 1600, 1728, 1792, 1920.

**Note (2026-03-24 session 2)**: New M and N values added: 96, 160, 288, 448, 480, 672 (secondary small dims) and 2112, 2240, 2304, 2496, 2560, 2688, 2816, 2880, 3200, 3264, 3328, 3456, 3584, 3648, 3840, 4032, 4096 (primary large dims). N=3520 and M=3520 were tested and FAILED — do NOT add them to ALLOWED lists.

**col_num=5 / row_num=5 tile workarounds (bf16 only)**:
- N=160/320/640/1600 (Pn=5 or 10 or 25): use m=64 (not m=32) for bf16 accuracy
- N=2240 (Pn=35): use m=64 with k≤64 — m=32 fails accuracy
- N=3200 (Pn=50): use m=64, k=32 only — k=64 fails even with m=64
- N=3520 (Pn=55): ALL tiles fail bf16 accuracy — dimension unusable
- M=160/320/640/1600: use m=16 (Pm=10/20/40/100, gives row_num=4 or 5 → pick m that gives row_num∈{3,4})
- M=2240 (Pm=35 with m=64): use m=16 (Pm=140, 140%4=0 → row_num=4) ✓
- M=3200 (Pm=50 with m=64): use m=32 (Pm=100, 100%4=0 → row_num=4) ✓
- M=3520 (Pm=55 with m=64): ALL tiles fail bf16 accuracy — dimension unusable

**General rule for bf16**: For any D where auto_col_num(D//n) or auto_row_num(D//m) returns 5, prefer a different m or n such that the quotient is divisible by 3 or 4.

The padded outer dims (Mp, Np, Kp) MUST be members of these lists.

---

## Tile divisibility

```
Mp % m == 0
Np % n == 0
Kp % k == 0
```

### Bundling constraint (M and N dims only)

The GEMM function accepts `col_num` and `row_num` parameters (default 4, valid: 3, 4, 5):

```python
GEMM(Mp, Np, Kp, Pm, Pn, Pk, TyI, TyO, col_num=col_num, row_num=row_num)
```

The bundling constraint requires:
```
Pn % col_num == 0   where Pn = Np // n, col_num ∈ {3, 4, 5}
Pm % row_num == 0   where Pm = Mp // m, row_num ∈ {3, 4, 5}
```

A tile (m, n) is valid if there exists *any* col_num ∈ {3, 4, 5} satisfying the constraint.
Use `col_num="auto"` in the script to auto-select the best value.

**Key implication**: N=960 is directly usable (no padding to 1024) with n=64:
- Pn = 960//64 = 15, and 15%3=0 and 15%5=0 → valid with col_num=3 or 5.

K dimension has no bundling constraint — only `Kp % k == 0` is required.

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
    Mp//m, Np//n, Kp//k, # num tiles per dim (Pm, Pn, Pk)
    TyI, TyO,            # input/output types
    col_num=col_num,      # ∈ {3, 4, 5}, must divide Pn; default 4
    row_num=row_num,      # ∈ {3, 4, 5}, must divide Pm; default 4
)
```

Auto-select col_num/row_num (try 4 first, then 3, then 5):
```python
def _auto_col_num(Pn): return next(c for c in (4,3,5) if Pn % c == 0)
def _auto_row_num(Pm): return next(r for r in (4,3,5) if Pm % r == 0)
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
  [--use-padding] \
  [--pad-M <Mp>] [--pad-N <Np>] [--pad-K <Kp>] \
  [--pad-impl <manual_copy|numpy_pad|torch_pad>] \
  [--col-num <auto|3|4|5>] \
  [--row-num <auto|3|4|5>]
```

If M, N, K are already in ALLOWED lists and equal Mp, Np, Kp, omit
`--use-padding` and the `--pad-*` flags.

`--col-num` and `--row-num` default to `auto` (tries 4 first, then 3, then 5).
Example for N=960, n=64 (Pn=15 → col_num=3):
```bash
python v2_test_mapping_large_gemm.py --M 128 --N 960 --K 512 --m 64 --n 64 --k 64 --dtype bf16
# col_num auto-selects 3 (15%3=0)
```
