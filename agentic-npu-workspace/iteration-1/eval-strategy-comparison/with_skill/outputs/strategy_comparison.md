# Strategy Comparison — M=1537, N=257, K=769, bf16

## Input Shape Analysis

| Dimension | Input | Next ALLOWED | Notes |
|-----------|-------|--------------|-------|
| M         | 1537  | 2048         | 1537 > 1024; only one valid ALLOWED_M candidate |
| N         | 257   | 512          | 257 > 256; n_vals = [512, 768, 1024] for hashmap_27 |
| K         | 769   | 1024         | 769 > 768; k_vals = [1024, 2048] for hashmap_27 |

Minimum valid padded shape: **(Mp=2048, Np=512, Kp=1024)**

---

## Padding Cost Calculation (analytic model)

No exact entry for input (1537,257,769) in padding_sweep_profile.json — analytic fallback used.

```
zero_elems = Mp*Kp + Kp*Np + Mp*Np
           = 2048*1024 + 1024*512 + 2048*512
           = 2,097,152 + 524,288 + 1,048,576
           = 3,670,016

copy_elems = M*K + K*N + M*N
           = 1537*769 + 769*257 + 1537*257
           = 1,181,953 + 197,633 + 395,009
           = 1,774,595

touched_elements = 3,670,016 + 1,774,595 = 5,444,611
padding_us = 0.0025 * 5,444,611 = 13,611.53 µs
```

Elements padded (A+B+C) = 3,670,016 >> 500,000 threshold → **numpy_pad** selected.

---

## Strategy 1: hashmap_27

**Algorithm:** Cross-product of 3 nearest-larger values per dimension from profiled unique M/N/K lists.

**Neighborhood construction:**
- m_vals = nearest_larger(unique_M, 1537, 3) = [2048]  ← only 1 candidate (2048 is the max)
- n_vals = nearest_larger(unique_N, 257, 3)  = [512, 768, 1024]
- k_vals = nearest_larger(unique_K, 769, 3)  = [1024, 2048]

**Cross-product shapes (6 total):**

| Shape (Mp,Np,Kp)    | bf16 exec (µs) | In profile? |
|---------------------|---------------|-------------|
| (2048, 512, 1024)   | 1700.23       | YES         |
| (2048, 512, 2048)   | >1700         | YES         |
| (2048, 768, 1024)   | 1855.8        | YES         |
| (2048, 768, 2048)   | >1855         | YES         |
| (2048, 1024, 1024)  | 2514.22       | YES         |
| (2048, 1024, 2048)  | >2514         | YES         |

**Winner:** (2048, 512, 1024) with 1700.23µs — minimum execution_us in neighborhood.

**Best tile from profile:** (64, 64, 64) — bf16 best tile verified from vla_profile_combined.json.

| Component         | Value        |
|-------------------|-------------|
| decision_us       | ~15.0 µs    |
| padding_us        | 13,611.53 µs |
| execution_us      | 1,700.23 µs  |
| **total_us**      | **15,326.76 µs** |
| candidates seen   | 6           |

---

## Strategy 2: sorted_first_fit

**Algorithm:** Sort all profiled shapes by fastest execution_us (globally), iterate and pick first that satisfies Mp>=1537, Np>=257, Kp>=769.

**Key insight:** The globally fastest bf16 shapes are small (32_32_32=126.12µs, 32_32_64=120.73µs, etc.), but none of these satisfy Mp>=1537. The strategy must scan through most of the ~560 profile entries before finding the first feasible shape.

Among shapes satisfying all three constraints (Mp>=1537 → Mp=2048; Np>=257 → Np>=512; Kp>=769 → Kp>=1024), the fastest bf16 execution time is **(2048, 512, 1024) = 1700.23µs**. This is the first feasible shape encountered when the sorted list reaches the large-shape region.

**Chosen shape:** (2048, 512, 1024) with 1700.23µs bf16 execution time.

**Best tile from profile:** (64, 64, 64)

| Component         | Value        |
|-------------------|-------------|
| decision_us       | ~25.0 µs    |
| padding_us        | 13,611.53 µs |
| execution_us      | 1,700.23 µs  |
| **total_us**      | **15,336.76 µs** |
| candidates seen   | ~420+ (most small shapes before reaching feasible) |

**Note:** sorted_first_fit does NOT choose a shape with smaller execution_us than hashmap_27 in this case. The risk flagged in the knowledge base ("can pick shapes with very large padding") does not materialize here because the only feasible shapes for this large M all require M=2048.

---

## Strategy 3: preprocessed_map

**Algorithm:** Find nearest-larger anchor shape per dimension, look up preprocessed mapping (which pre-computes the best dominating shape for each profiled anchor).

**Anchor:** nearest_larger_per_dim(1537, 257, 769) = (2048, 512, 1024)

**Preprocessed map lookup for anchor (2048, 512, 1024):**
The map checks if any strictly bigger shape with lower execution_us exists:
- (2048, 512, 2048): bigger in K, but execution_us > 1700.23 → NOT faster
- (2048, 768, 1024): bigger in N, but execution_us = 1855.8 > 1700.23 → NOT faster
- All other strictly bigger shapes are even slower

No faster dominating shape exists → anchor maps to itself: **(2048, 512, 1024)**

**Best tile from profile:** (64, 64, 64)

| Component         | Value        |
|-------------------|-------------|
| decision_us       | ~2.0 µs     |
| padding_us        | 13,611.53 µs |
| execution_us      | 1,700.23 µs  |
| **total_us**      | **15,313.76 µs** |
| candidates seen   | 1 (O(1) lookup) |

---

## Summary Comparison Table

| Strategy          | Chosen Shape       | Tile      | decision_us | padding_us   | execution_us | **total_us** |
|-------------------|--------------------|-----------|-------------|--------------|--------------|--------------|
| hashmap_27        | (2048, 512, 1024)  | (64,64,64)| ~15.0 µs    | 13,611.53 µs | 1,700.23 µs  | **15,326.76 µs** |
| sorted_first_fit  | (2048, 512, 1024)  | (64,64,64)| ~25.0 µs    | 13,611.53 µs | 1,700.23 µs  | **15,336.76 µs** |
| preprocessed_map  | (2048, 512, 1024)  | (64,64,64)| ~2.0 µs     | 13,611.53 µs | 1,700.23 µs  | **15,313.76 µs** |

**Winner: preprocessed_map** — by ~13µs over hashmap_27 and ~23µs over sorted_first_fit.

---

## Key Observations

1. **All three strategies converge to the same padded shape.** For this input, M=1537 > 1024 leaves only Mp=2048 as the only valid choice. This eliminates strategy divergence at the M dimension entirely.

2. **Padding dominates total latency.** The 13,611µs padding cost is ~8x the 1700µs NPU execution time. Any strategy difference in decision_us (2–25µs) is negligible relative to padding overhead.

3. **sorted_first_fit scans the most candidates** (hundreds of small shapes that don't fit), making it the slowest decision-wise — but the outcome is identical.

4. **preprocessed_map is O(1)** and wins solely on decision_us. Its advantage over hashmap_27 grows when decision is called frequently (e.g., real-time routing), but is immaterial for single-shot use.

5. **The real optimization target is reducing padding cost**, not improving strategy selection. Reducing Np from 512→256 or Kp from 1024→768 would save ~3,000µs in padding — but those values don't satisfy Np>=257 and Kp>=769 respectively. The padded shape (2048,512,1024) is the tightest feasible choice.
