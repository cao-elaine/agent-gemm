# NPU Dimension Discovery Log — 2026-03-24

## Parameters
- dim_max: 4096
- dtypes tested: bf16
- test_M_fixed: 128, test_N_fixed: 512, test_K_fixed: 512
- Memory budget: (m*n + n*k + k*m) * 2 <= 32,256 bytes
- k_min (UPDATED): max(32, Kp//16) — for Kp=512: k_min=32

---

## New confirmed N dimensions

| D    | Best tile   | Latency (avg) | col_num | Valid n values | Notes |
|------|-------------|---------------|---------|----------------|-------|
| 192  | (32,64,64)  | 176µs         | 3       | n=64, n=32     | All 4 tiles pass |
| 320  | (64,64,64)  | 213µs         | 5       | n=64 (m=64 only) | m=32 accuracy fails with col_num=5; m=64 required |
| 384  | (32,32,128) | 187µs         | 3/4     | n=64, n=32     | All 4 tiles pass; n=32 faster than n=64 |
| 576  | (32,64,64)  | 283µs         | 3       | n=64, n=32     | All 4 tiles pass |
| 640  | (32,32,128) | 229µs         | 5/4     | n=64 (m=64), n=32 (col_num=4) | col_num=5 with m=32 fails accuracy |
| 1152 | (32,32,128) | 299µs         | 3/4     | n=64, n=32     | All 4 tiles pass |
| 1280 | (32,64,64)  | 275µs         | 4       | n=64, n=32     | All 4 tiles pass |
| 1344 | (32,64,32)  | 479µs         | 3       | n=64, n=32     | All 4 tiles pass |
| 1536 | (32,64,64)  | 301µs         | 4       | n=64, n=32     | All 4 tiles pass |
| 1600 | (64,64,64)  | 527µs         | 5       | n=64 (m=64 only) | col_num=5 with m=32 fails accuracy |
| 1728 | (32,64,32)  | 585µs         | 3       | n=64, n=32     | All 4 tiles pass |
| 1792 | (32,64,128) | 326µs         | 4       | n=64, n=32     | All 4 tiles pass |
| 1920 | (32,32,128) | 390µs         | 3/4     | n=64, n=32     | All 4 tiles pass |

---

## New confirmed M dimensions

| D    | Best tile    | Latency (avg) | row_num | Valid m values | Notes |
|------|--------------|---------------|---------|----------------|-------|
| 192  | (64,64,32)   | 233µs         | 3       | m=64, m=32     | All 4 tiles pass |
| 320  | (16,64,64)   | 352µs         | 4       | m=16 (row_num=4) | m=32/m=64 with row_num=5 fail accuracy |
| 384  | (32,64,64)   | 300µs         | 3/4     | m=64, m=32     | All 4 tiles pass |
| 576  | (64,64,32)   | 435µs         | 3       | m=64, m=32     | All 4 tiles pass |
| 640  | (32,64,128)  | 398µs         | 4       | m=32 (row_num=4) | m=64 with row_num=5 fails accuracy |
| 1152 | (32,64,128)  | 601µs         | 3/4     | m=64, m=32     | All 4 tiles pass |
| 1280 | (64,64,64)   | 499µs         | 4       | m=64, m=32     | All 4 tiles pass |
| 1344 | (64,64,32)   | 845µs         | 3       | m=64, m=32     | All 4 tiles pass |
| 1536 | (64,64,64)   | 558µs         | 4       | m=64, m=32     | All 4 tiles pass |
| 1600 | (16,64,64)   | 1246µs        | 4       | m=16 (row_num=4) | m=64/m=32 with row_num=5 fail accuracy |
| 1728 | (64,64,64)   | 1061µs        | 3       | m=64, m=32     | All 4 tiles pass |
| 1792 | (64,64,64)   | 621µs         | 4       | m=64, m=32     | All 4 tiles pass |
| 1920 | (32,64,128)  | 904µs         | 3/4     | m=64, m=32     | All 4 tiles pass |

---

## Failed dimensions

None confirmed as "likely invalid". All 13 candidate dimensions (both N and M) were confirmed valid with at least one passing tile. However, accuracy failures occurred for specific tile/bundling combinations:

| D | Context | Failing pattern | Root cause |
|---|---------|-----------------|------------|
| 320 (N) | m=32, col_num=5 (Pn=5) | All m=32 tiles fail accuracy | col_num=5 with m=32 causes bf16 accumulation error |
| 640 (N) | m=32, n=64, col_num=5 | m=32 tiles fail; m=64 or n=32 work | col_num=5 with m=32+n=64 fails accuracy |
| 1600 (N) | m=32, col_num=5 | m=32 tiles fail; m=64 passes | Same col_num=5 pattern |
| 320 (M) | m=64/m=32, row_num=5 (Pm=5/10) | row_num=5 fails accuracy | row_num=5 with medium m causes bf16 error |
| 640 (M) | m=64, row_num=5 (Pm=10) | row_num=5 fails; m=32 row_num=4 works | Same row_num=5 pattern |
| 1600 (M) | m=64/m=32, row_num=5 | row_num=5 fails; m=16 row_num=4 works | Same row_num=5 pattern |

**Key finding**: Dimensions where D//64 is divisible by 5 but not by 3 or 4 (i.e., col_num/row_num must be 5) exhibit bf16 accuracy failures when m (for M) or the row tile count (for N) is 32. Using m=64 for N-dims or m=16/32 with row_num=4 for M-dims resolves the issue.

---

## Impact on SmolVLA shapes

### Direct dimension matches

| SmolVLA shape | Layer | New dimension | Padding savings |
|---------------|-------|---------------|-----------------|
| M=128, K=960, N=320 | k_proj, v_proj (Text Layers x16) | N=320 now VALID | Previously padded N=320 → N=512 (60% padding); now run N=320 directly, saving 37.5% of N computation |

### Indirect benefits
- N=192, N=384, N=576 are now valid; these cover potential future SmolVLA attention head sub-dimensions
- M=192, M=384 etc. could cover batched inference scenarios
- N=1280, N=1536, N=1920 may be relevant for larger LM decoder hidden dimensions

---

## Key discovery: col_num=5 / row_num=5 accuracy failure pattern

**New rule identified**: For bf16 GEMMs, when the bundling quotient (Pn=Np//n or Pm=Mp//m) is only divisible by 5 (not by 3 or 4), the auto-selected col_num/row_num=5 causes bf16 accumulation accuracy failures for small tile values. The workaround is:
- For N dimensions: use m=64 (larger row tile) instead of m=32
- For M dimensions: use a different m tile that achieves row_num=3 or 4 (e.g., m=16 with Pm=D//16 divisible by 4)

Affected dimensions: those where D//64 mod 3 != 0 and D//64 mod 4 != 0 and D//64 mod 5 == 0 (i.e., Pn is a multiple of 5 but not 3 or 4).

Examples: 320 (Pn=5), 640 (Pn=10), 1600 (Pn=25).

---

## Raw probe data

### Plan A — New N dimensions (M=128, N=D, K=512)

| M   | N    | K   | m  | n  | k   | Status       | Avg (µs) |
|-----|------|-----|----|----|-----|--------------|----------|
| 128 | 192  | 512 | 32 | 64 | 128 | PASSED       | 187.5    |
| 128 | 192  | 512 | 32 | 64 | 64  | PASSED       | 176.3    |
| 128 | 192  | 512 | 32 | 64 | 32  | PASSED       | 177.6    |
| 128 | 192  | 512 | 32 | 32 | 128 | PASSED       | 185.3    |
| 128 | 320  | 512 | 32 | 64 | 128 | ACCURACY_FAIL| —        |
| 128 | 320  | 512 | 32 | 64 | 64  | ACCURACY_FAIL| —        |
| 128 | 320  | 512 | 32 | 64 | 32  | ACCURACY_FAIL| —        |
| 128 | 320  | 512 | 32 | 32 | 128 | ACCURACY_FAIL| —        |
| 128 | 320  | 512 | 64 | 64 | 64  | PASSED       | 212.9    |
| 128 | 384  | 512 | 32 | 64 | 128 | PASSED       | 241.5    |
| 128 | 384  | 512 | 32 | 64 | 64  | PASSED       | 229.1    |
| 128 | 384  | 512 | 32 | 64 | 32  | PASSED       | 234.7    |
| 128 | 384  | 512 | 32 | 32 | 128 | PASSED       | 187.2    |
| 128 | 576  | 512 | 32 | 64 | 128 | PASSED       | 294.2    |
| 128 | 576  | 512 | 32 | 64 | 64  | PASSED       | 283.4    |
| 128 | 576  | 512 | 32 | 64 | 32  | PASSED       | 284.8    |
| 128 | 576  | 512 | 32 | 32 | 128 | PASSED       | 293.9    |
| 128 | 640  | 512 | 32 | 64 | 128 | ACCURACY_FAIL| —        |
| 128 | 640  | 512 | 32 | 64 | 64  | ACCURACY_FAIL| —        |
| 128 | 640  | 512 | 64 | 64 | 64  | PASSED       | 260.3    |
| 128 | 640  | 512 | 32 | 32 | 128 | PASSED       | 229.2    |
| 128 | 1152 | 512 | 32 | 64 | 128 | PASSED       | 444.4    |
| 128 | 1152 | 512 | 32 | 64 | 64  | PASSED       | 432.2    |
| 128 | 1152 | 512 | 32 | 64 | 32  | PASSED       | 458.8    |
| 128 | 1152 | 512 | 32 | 32 | 128 | PASSED       | 299.5    |
| 128 | 1280 | 512 | 32 | 64 | 128 | PASSED       | 280.6    |
| 128 | 1280 | 512 | 32 | 64 | 64  | PASSED       | 275.3    |
| 128 | 1280 | 512 | 32 | 64 | 32  | PASSED       | 282.7    |
| 128 | 1280 | 512 | 32 | 32 | 128 | PASSED       | 317.6    |
| 128 | 1344 | 512 | 32 | 64 | 128 | PASSED       | 499.5    |
| 128 | 1344 | 512 | 32 | 64 | 64  | PASSED       | 484.6    |
| 128 | 1344 | 512 | 32 | 64 | 32  | PASSED       | 479.5    |
| 128 | 1344 | 512 | 32 | 32 | 128 | PASSED       | 497.9    |
| 128 | 1536 | 512 | 32 | 64 | 128 | PASSED       | 305.5    |
| 128 | 1536 | 512 | 32 | 64 | 64  | PASSED       | 301.2    |
| 128 | 1536 | 512 | 32 | 64 | 32  | PASSED       | 305.4    |
| 128 | 1536 | 512 | 32 | 32 | 128 | PASSED       | 342.3    |
| 128 | 1600 | 512 | 32 | 64 | 128 | ACCURACY_FAIL| —        |
| 128 | 1600 | 512 | 32 | 64 | 64  | ACCURACY_FAIL| —        |
| 128 | 1600 | 512 | 64 | 64 | 64  | PASSED       | 526.9    |
| 128 | 1600 | 512 | 32 | 32 | 128 | ACCURACY_FAIL| —        |
| 128 | 1728 | 512 | 32 | 64 | 128 | PASSED       | 600.1    |
| 128 | 1728 | 512 | 32 | 64 | 64  | PASSED       | 589.1    |
| 128 | 1728 | 512 | 32 | 64 | 32  | PASSED       | 584.8    |
| 128 | 1728 | 512 | 32 | 32 | 128 | PASSED       | 609.6    |
| 128 | 1792 | 512 | 32 | 64 | 128 | PASSED       | 325.9    |
| 128 | 1792 | 512 | 32 | 64 | 64  | PASSED       | 327.5    |
| 128 | 1792 | 512 | 32 | 64 | 32  | PASSED       | 327.3    |
| 128 | 1792 | 512 | 32 | 32 | 128 | PASSED       | 371.5    |
| 128 | 1920 | 512 | 32 | 64 | 128 | PASSED       | 643.4    |
| 128 | 1920 | 512 | 32 | 64 | 64  | PASSED       | 626.4    |
| 128 | 1920 | 512 | 32 | 64 | 32  | PASSED       | 637.5    |
| 128 | 1920 | 512 | 32 | 32 | 128 | PASSED       | 389.9    |

### Plan B — New M dimensions (M=D, N=512, K=512)

| M    | N   | K   | m  | n  | k   | Status       | Avg (µs) |
|------|-----|-----|----|----|-----|--------------|----------|
| 192  | 512 | 512 | 64 | 64 | 64  | PASSED       | 236.3    |
| 192  | 512 | 512 | 64 | 64 | 32  | PASSED       | 232.8    |
| 192  | 512 | 512 | 32 | 64 | 128 | PASSED       | 244.5    |
| 192  | 512 | 512 | 32 | 64 | 64  | PASSED       | 250.8    |
| 320  | 512 | 512 | 64 | 64 | 64  | ACCURACY_FAIL| —        |
| 320  | 512 | 512 | 64 | 64 | 32  | ACCURACY_FAIL| —        |
| 320  | 512 | 512 | 32 | 64 | 128 | ACCURACY_FAIL| —        |
| 320  | 512 | 512 | 32 | 64 | 64  | ACCURACY_FAIL| —        |
| 320  | 512 | 512 | 16 | 64 | 64  | PASSED       | 351.7    |
| 320  | 512 | 512 | 16 | 64 | 128 | PASSED       | 364.0    |
| 384  | 512 | 512 | 64 | 64 | 64  | PASSED       | 333.7    |
| 384  | 512 | 512 | 64 | 64 | 32  | PASSED       | 338.9    |
| 384  | 512 | 512 | 32 | 64 | 128 | PASSED       | 301.9    |
| 384  | 512 | 512 | 32 | 64 | 64  | PASSED       | 299.6    |
| 576  | 512 | 512 | 64 | 64 | 64  | PASSED       | 438.1    |
| 576  | 512 | 512 | 64 | 64 | 32  | PASSED       | 435.2    |
| 576  | 512 | 512 | 32 | 64 | 128 | PASSED       | 474.6    |
| 576  | 512 | 512 | 32 | 64 | 64  | PASSED       | 449.2    |
| 640  | 512 | 512 | 64 | 64 | 64  | ACCURACY_FAIL| —        |
| 640  | 512 | 512 | 64 | 64 | 32  | ACCURACY_FAIL| —        |
| 640  | 512 | 512 | 32 | 64 | 64  | PASSED       | 401.5    |
| 640  | 512 | 512 | 32 | 64 | 128 | PASSED       | 398.4    |
| 1152 | 512 | 512 | 64 | 64 | 64  | PASSED       | 761.4    |
| 1152 | 512 | 512 | 64 | 64 | 32  | PASSED       | 762.1    |
| 1152 | 512 | 512 | 32 | 64 | 128 | PASSED       | 601.2    |
| 1152 | 512 | 512 | 32 | 64 | 64  | PASSED       | 605.7    |
| 1280 | 512 | 512 | 64 | 64 | 64  | PASSED       | 498.7    |
| 1280 | 512 | 512 | 64 | 64 | 32  | PASSED       | 511.3    |
| 1280 | 512 | 512 | 32 | 64 | 128 | PASSED       | 659.0    |
| 1280 | 512 | 512 | 32 | 64 | 64  | PASSED       | 654.5    |
| 1344 | 512 | 512 | 64 | 64 | 64  | PASSED       | 851.1    |
| 1344 | 512 | 512 | 64 | 64 | 32  | PASSED       | 844.6    |
| 1344 | 512 | 512 | 32 | 64 | 128 | PASSED       | 938.0    |
| 1344 | 512 | 512 | 32 | 64 | 64  | PASSED       | 903.3    |
| 1536 | 512 | 512 | 64 | 64 | 64  | PASSED       | 558.0    |
| 1536 | 512 | 512 | 64 | 64 | 32  | PASSED       | 582.1    |
| 1536 | 512 | 512 | 32 | 64 | 128 | PASSED       | 748.4    |
| 1536 | 512 | 512 | 32 | 64 | 64  | PASSED       | 737.9    |
| 1600 | 512 | 512 | 64 | 64 | 64  | ACCURACY_FAIL| —        |
| 1600 | 512 | 512 | 32 | 64 | 64  | ACCURACY_FAIL| —        |
| 1600 | 512 | 512 | 16 | 64 | 64  | PASSED       | 1246.5   |
| 1728 | 512 | 512 | 64 | 64 | 64  | PASSED       | 1060.8   |
| 1728 | 512 | 512 | 64 | 64 | 32  | PASSED       | 1078.3   |
| 1728 | 512 | 512 | 32 | 64 | 128 | PASSED       | 1208.5   |
| 1728 | 512 | 512 | 32 | 64 | 64  | PASSED       | 1138.9   |
| 1792 | 512 | 512 | 64 | 64 | 64  | PASSED       | 621.0    |
| 1792 | 512 | 512 | 64 | 64 | 32  | PASSED       | 658.9    |
| 1792 | 512 | 512 | 32 | 64 | 128 | PASSED       | 877.8    |
| 1792 | 512 | 512 | 32 | 64 | 64  | PASSED       | 840.4    |
| 1920 | 512 | 512 | 64 | 64 | 64  | PASSED       | 1227.4   |
| 1920 | 512 | 512 | 64 | 64 | 32  | PASSED       | 1211.9   |
| 1920 | 512 | 512 | 32 | 64 | 128 | PASSED       | 903.5    |
| 1920 | 512 | 512 | 32 | 64 | 64  | PASSED       | 903.6    |
