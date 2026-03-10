# GEMM Strategy Comparison: M=1537, N=257, K=769, bf16

## Problem Statement

Find the best tiling and padding strategy for a GEMM operation on AMD Ryzen NPU with:
- **Original dimensions**: M=1537, N=257, K=769
- **Data type**: bf16 (2 bytes per element)
- **Useful compute**: 2 × 1537 × 257 × 769 = ~606M ops

### Constraints
- `ALLOWED_M` = [32, 64, 128, 256, 512, 1024, 2048]
- `ALLOWED_N` = [32, 64, 128, 256, 512, 768, 1024, 2048]
- `ALLOWED_K` = [32, 64, 128, 256, 512, 768, 1024, 2048]
- Tile (m, n, k) must evenly divide padded (Mp, Np, Kp)
- Each padded dimension must itself be a multiple of at least one allowed tile size

---

## Step 1: Determine Valid Padded Dimension Ranges

### M dimension (original = 1537)
- 1537 > 1024, so the smallest allowed value ≥ 1537 is **Mp = 2048**
- No smaller ALLOWED_M value works: 1024 < 1537
- Only option: **Mp = 2048**

### N dimension (original = 257)
- 257 > 256, so 256 is NOT valid
- Next ALLOWED_N value ≥ 257: **Np = 512**
- Larger options: 768, 1024, 2048 (all valid but cause more waste)

### K dimension (original = 769)
- 769 > 768, so 768 is NOT valid
- Next ALLOWED_K value ≥ 769: **Kp = 1024**
- Larger options: 1536 (=2×768), 2048 (all valid but cause more waste)

---

## Step 2: Candidate Padded Shapes and Tile Selection

For each candidate padded shape, the best tile is the **largest (m,n,k) from ALLOWED sets that evenly divides (Mp,Np,Kp)**.

### Strategy A: Mp=2048, Np=512, Kp=1024 — WINNER

| Parameter | Value |
|-----------|-------|
| Padded shape | (2048, 512, 1024) |
| Best tile | (2048, 512, 1024) |
| Number of tiles | **1** |
| Tile dispatches | 1×1×1 |
| Total padded ops | 2 × 2048 × 512 × 1024 = **2,147,483,648** |
| Useful ops | 2 × 1537 × 257 × 769 = **605,985,842** |
| Wasted ops | 1,541,497,806 |
| Waste % | **254.4%** |
| Tile memory (A+B+C) | (2048×1024 + 1024×512 + 2048×512) × 2 = **7,340,032 bytes** |
| Arithmetic intensity | 2,147,483,648 / 7,340,032 = **292.6 ops/byte** |

**Tile divisibility check:**
- 2048 / 2048 = 1 ✓ (ALLOWED_M contains 2048)
- 512 / 512 = 1 ✓ (ALLOWED_N contains 512)
- 1024 / 1024 = 1 ✓ (ALLOWED_K contains 1024)

---

### Strategy B: Mp=2048, Np=768, Kp=1024

| Parameter | Value |
|-----------|-------|
| Padded shape | (2048, 768, 1024) |
| Best tile | (2048, 768, 1024) |
| Number of tiles | 1 |
| Total padded ops | 2 × 2048 × 768 × 1024 = **3,221,225,472** |
| Wasted ops | 2,615,239,630 |
| Waste % | **431.6%** |
| Tile memory (A+B+C) | (2048×1024 + 1024×768 + 2048×768) × 2 = **9,175,040 bytes** |
| Arithmetic intensity | 3,221,225,472 / 9,175,040 = **351.1 ops/byte** |

**Analysis:** Arithmetic intensity is higher due to larger tile, but the N=257→768 padding triples the N dimension, wasting 511 rows out of 768 (66% waste in N). The total wasted compute is 431.6% vs Strategy A's 254.4%. Despite the single tile dispatch, the excess compute cost makes this strictly worse.

---

### Strategy C: Mp=2048, Np=512, Kp=1024, Tile=(1024, 512, 1024)

| Parameter | Value |
|-----------|-------|
| Padded shape | (2048, 512, 1024) |
| Tile | (1024, 512, 1024) |
| Number of tiles | **2** (2×1×1) |
| Total padded ops | 2,147,483,648 (same as A) |
| Waste % | **254.4%** (same as A) |
| Tile memory (A+B+C) | (1024×1024 + 1024×512 + 1024×512) × 2 = **4,194,304 bytes** |
| Arithmetic intensity | 1,073,741,824 / 4,194,304 = **256.0 ops/byte** |

**Analysis:** Same padded shape as Strategy A but uses a smaller tile, resulting in 2 tile dispatches and reduced arithmetic intensity (256.0 vs 292.6). No benefit over Strategy A.

---

### Strategy D: Mp=2048, Np=512, Kp=1024, Tile=(2048, 512, 512)

| Parameter | Value |
|-----------|-------|
| Padded shape | (2048, 512, 1024) |
| Tile | (2048, 512, 512) |
| Number of tiles | **2** (1×1×2) |
| Waste % | **254.4%** (same as A) |
| Tile memory (A+B+C) | (2048×512 + 512×512 + 2048×512) × 2 = **4,718,592 bytes** |
| Arithmetic intensity | 1,073,741,824 / 4,718,592 = **227.6 ops/byte** |

**Analysis:** Splitting the K dimension into 2 tiles reduces arithmetic intensity to 227.6 ops/byte (vs 292.6 for Strategy A) and doubles K-direction dispatches. Strictly worse than A.

---

### Strategy E: Mp=2048, Np=512, Kp=1536

| Parameter | Value |
|-----------|-------|
| Padded shape | (2048, 512, 1536) |
| Best tile | (2048, 512, 768) |
| Number of tiles | **2** (1×1×2) |
| Total padded ops | 2 × 2048 × 512 × 1536 = **3,221,225,472** |
| Waste % | **431.6%** |
| Tile memory | (2048×768 + 768×512 + 2048×512) × 2 = **5,505,024 bytes** |
| Arithmetic intensity | 1,610,612,736 / 5,505,024 = **292.6 ops/byte** |

**Analysis:** Kp=1536 is unnecessary since Kp=1024 already fits K=769. Using Kp=1536 inflates K by 50% more than needed, wasting 767 K-values. The 431.6% waste ratio is far worse despite matching arithmetic intensity.

---

### Strategy F: Mp=2048, Np=512, Kp=2048

| Parameter | Value |
|-----------|-------|
| Padded shape | (2048, 512, 2048) |
| Best tile | (2048, 512, 2048) |
| Number of tiles | 1 |
| Total padded ops | 2 × 2048 × 512 × 2048 = **4,294,967,296** |
| Waste % | **608.8%** |
| Arithmetic intensity | 390.8 ops/byte |

**Analysis:** Single tile dispatch with highest arithmetic intensity, but 608.8% waste is prohibitive. K=769 requires only Kp=1024; expanding to 2048 doubles K padding cost unnecessarily.

---

## Step 3: Comparison Summary Table

| Strategy | Mp | Np | Kp | Tile (m,n,k) | Tiles | Waste% | Arith. Int. |
|----------|----|----|-----|--------------|-------|--------|-------------|
| **A (BEST)** | **2048** | **512** | **1024** | **(2048,512,1024)** | **1** | **254.4%** | **292.6** |
| B | 2048 | 768 | 1024 | (2048,768,1024) | 1 | 431.6% | 351.1 |
| C (A, small tile) | 2048 | 512 | 1024 | (1024,512,1024) | 2 | 254.4% | 256.0 |
| D (A, small K tile) | 2048 | 512 | 1024 | (2048,512,512) | 2 | 254.4% | 227.6 |
| E | 2048 | 512 | 1536 | (2048,512,768) | 2 | 431.6% | 292.6 |
| F | 2048 | 512 | 2048 | (2048,512,2048) | 1 | 608.8% | 390.8 |

---

## Step 4: Decision Rationale

### Why Strategy A wins

**1. Mp=2048 is the only option** — there is no freedom here. M=1537 exceeds all smaller ALLOWED_M values, forcing Mp=2048 with 33.2% M-dimension padding.

**2. Np=512 is strictly better than larger values:**
- N=257 only needs 255 values of padding (padding 49.8% of N)
- Going to Np=768 would pad 511 values (padding 198.8% of N), wasting 2x more N-compute
- The gain in arithmetic intensity from larger N tile (351.1 vs 292.6 ops/byte) does NOT compensate for 77.2 percentage points more wasted compute

**3. Kp=1024 is strictly better than larger values:**
- K=769 only needs 255 values of K-padding (33.2% K overhead)
- Kp=1536 would pad 767 extra K values, inflating K compute by 50% more for no benefit
- Kp=2048 doubles the needed K, wasting 1279 K-values

**4. Maximum tile size (2048, 512, 1024) is optimal within Strategy A:**
- Single tile dispatch minimizes kernel launch overhead on NPU
- Highest arithmetic intensity (292.6 ops/byte) among all valid tiles for this padded shape
- Full NPU compute array utilization — no partial fills

### Estimated Latency Ordering (fastest to slowest)

Assuming NPU latency is dominated by compute ops and inversely proportional to arithmetic intensity:
1. **Strategy A** — minimum wasted compute (254.4%), single dispatch, high intensity
2. Strategy C/D — same compute cost as A, but 2x dispatches, lower intensity
3. Strategy B/E — same dispatch count as A, but 77% more wasted compute
4. Strategy F — highest intensity but 140% more wasted compute than A

---

## Conclusion

**Winner: Strategy A — padded shape (Mp=2048, Np=512, Kp=1024), tile (2048, 512, 1024)**

This strategy achieves the minimum possible total work while maintaining maximum arithmetic intensity and the lowest possible dispatch overhead (1 tile). The key insight is that for dimensions just above an allowed boundary (M=1537 just above 1024, N=257 just above 256, K=769 just above 768), the "next valid" padded size is the right choice — using larger padded sizes to gain tile flexibility is never worth the extra computation cost.
