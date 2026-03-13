# Strategy Insights — Knowledge Base

Maintained by the agentic NPU optimizer. 

---

## Padding strategy

- The threshold for switching from `manual_copy` to `numpy_pad` is determined by
  `ab_elements = M×K + K×N` (total elements in A and B matrices). Exact thresholds
  per dtype (from `padding_transition_thresholds.json`):
  - i8: switch to `numpy_pad` when ab_elements > **1,653,750**
  - i16: switch to `numpy_pad` when ab_elements > **1,213,000**
  - bf16: switch to `numpy_pad` when ab_elements > **571,779**
  bf16 switches to numpy_pad much sooner than i8 because numpy's vectorized backend
  benefits more from wider element widths.

- `manual_copy` wins in approximately 90% of all shape/dtype combinations. `numpy_pad`
  only outperforms it in ~10% of cases — almost exclusively large, geometrically
  irregular shapes. When in doubt, default to `manual_copy`.

- The `numpy_pad` win rate is strongly dtype-dependent: bf16 shows the most numpy_pad
  wins (~17% of shapes), i16 ~9%, and i8 only ~5%. This is consistent with the lower
  bf16 threshold above. For i8, `manual_copy` is almost always correct.

- **`torch_pad` (`torch.nn.ZeroPad2d`) is consistently the slowest padding method
  across all dtypes and all input sizes.** Median latency is roughly 4–15× higher
  than `manual_copy` depending on dtype. Eliminate it from consideration entirely.

- For very small inputs (M,N,K all < 64), padding to the next 64
  is nearly free — NPU launch overhead dominates anyway.

- **Padding time is not negligible and can exceed NPU execution time.** On average
  it accounts for ~17% of total end-to-end time, but in edge cases (e.g. shapes
  with large padding ratios) it can exceed 60% of total time. Always account for
  padding cost when evaluating a candidate padded shape.

- **The bf16 padding threshold carries ~8–17× more regret than i8/i16** (~8.4 µs vs
  ~0.5–0.8 µs). For bf16 inputs near the 571K ab_elements boundary, both methods are
  close — either choice is acceptable. For i8 below 1.65M, the margin strongly favors
  `manual_copy` and the threshold is highly reliable.

- **Possible hybrid padded-shape selection rule**: for small inputs (M, N, K roughly in the
  32–64 range), prefer the padded shape with the fastest NPU execution time even if
  it adds more padding — padding cost is ~constant at this scale so extra zeros are
  free. For larger inputs, always pad to the closest valid (smallest) shape — both
  padding cost and NPU execution time grow with shape size, so the closest shape wins
  on both fronts and there is rarely benefit to padding further. This is a possible
  heuristic to use when deciding what padding shape to pad to.

---

## Tile selection

- Default safe tile: (32, 32, 32) works for all dtypes and all
  legal (Mp, Np, Kp) combinations.

- Larger tiles are not always faster — they increase register
  pressure. Profile data is the ground truth; always prefer profile lookup
  over intuition.

- (64,64,64) is the most consistently optimal tile across all shapes
  and dtypes — it wins the most cases overall. (64,128,64) can edge it out
  on very large shapes but the gap is small (~2%). Default to (64,64,64)
  when there is no profile hit.

- **Shapes with Np < 128 (padded N = 32 or 64) frequently only support (32,32,32).**
  Larger tiles fail divisibility checks on small N values. If the padded shape has
  N ≤ 64, immediately fall back to (32,32,32) and skip larger tile attempts.

- Asymmetric tiles where n or k drops significantly below 64 are
  strongly penalized — performance can degrade 2× or more. Avoid tiles
  that reduce n or k below 64.

- **For i8, tile selection is more contested than other dtypes.** Both (64,64,64)
  and (64,128,64) win on roughly equal numbers of shapes. Always prefer a profile
  lookup for i8 rather than relying on the (64,64,64) default — the gain from the
  right tile is larger for i8 than for bf16.

- (32,32,64) and (32,32,32) have the highest failure rates across
  profiled shapes and are also among the slowest. 64-based tiles fail
  far less often. When falling back from a failed tile, prefer 
  (64,64,32) or (64,64,64) over anything in the 32-family.

- Small tiles like (16,16,16) are among the slowest even when they
  succeed. Only use as a last resort.

- **(128,128,128) never works — hard blacklist.** It accounts for ~42% of all
  documented tile failures across the profiled shape set. Do not attempt it.

- **(64,128,64) appears fast on average but is unreliable.** It accounts for ~35%
  of all documented tile failures — second only to (128,128,128). It only succeeds
  on shapes where N is cleanly divisible by 128, making its average time artificially
  low. Do not use it as a general default.

- **The root cause of most tile failures is integer divisibility**: the NPU requires
  M % m == 0 and N % n == 0. When these don't divide evenly, a ZeroDivisionError
  occurs in the tile mapping computation. Always verify divisibility before selecting
  a tile.

---

## Strategy selection rules

- All three strategy decision times are negligible relative to padding and NPU
  execution time. Do not choose a strategy primarily to save decision overhead —
  the choice of *which padded shape* is selected matters far more than how fast
  the decision is made.

- `hashmap_27` is the most reliable all-around strategy for inputs
  that fall near the interior of the profiled shape grid. Its main advantage is
  limiting candidate search to the 27 closest shapes, which reduces padding overhead.

- `sorted_first_fit` guarantees the fastest NPU execution time among profiled shapes
  but can pick shapes with very large padding. Always check the padding overhead
  before committing to its result.

- `preprocessed_map` is both the fastest at decision time (< 1µs) and guaranteed to
  find the fastest NPU execution time — it is the best default when the exact shape
  is already profiled.

---

## Data type performance

- int8 is consistently ~2× faster than int16 and bf16 across all
  shapes and tiles. bf16 and int16 perform nearly identically to each other.
  If the use case allows quantization to int8, the speedup is significant.

- bf16 has a lower median latency than i16 despite having a similar or
  higher mean. The i16 distribution has a longer upper tail. When the input
  is small-to-medium, bf16 often wins over i16.

- Execution time scales roughly with total compute volume (M×N×K).
  Doubling all three dimensions increases NPU time by roughly 5–7×, not 8×,
  suggesting some parallelism benefit at larger scales.

- The NPU has a fixed launch overhead of ~157µs. For small matrices (bottom
  25% by M×N×K volume), execution time barely changes with work size — the
  NPU is mostly idling. This overhead is unavoidable and should be factored
  into latency estimates for small inputs.

- [2026-03-12] **The profile "Best NPU average time" is not a reliable comparator for
  live measured averages.** The measured average swings bidirectionally around the
  profile estimate depending on system load: observed range across 4 runs is −6.5% to
  +73% of the profile value. The NPU *minimum* execution time is far more stable —
  consistently 10–15% below the profile average across all tested shapes and dtypes.
  When diagnosing apparent profile mismatches, always check the minimum first. If the
  minimum tracks the profile (within ~15%), any deviation in the average is system
  jitter, not a strategy or tile error.

---

## Correctness

- bf16 correctness tolerance: atol=1e-1 (bfloat16 has limited
  mantissa precision).
- i8 / i16 correctness tolerance: atol=1e-5 (exact integer arithmetic).

---

## When no profile entry exists

- Fall back to minimum ALLOWED shape that satisfies Mp >= M, Np >= N,
  Kp >= K with tile (32, 32, 32).

---

## Shape pathologies

- K >> M and K >> N (e.g. M=64, N=64, K=1024): padding K is expensive.
  Prefer the smallest feasible Kp; `hashmap_27` tends to handle this better than
  `sorted_first_fit`.

- M just above a power-of-2 boundary (e.g. M=1025): `sorted_first_fit`
  often picks a 2× larger padded shape (M'=2048). Check `hashmap_27` result — it
  usually finds a smaller option.

- **Shapes just above any power-of-2 boundary (e.g. M=129, M=257, M=513) incur
  the worst padding volume ratios (~7.8–7.9× element volume increase).** These are
  the most expensive inputs to pad. If the application can avoid generating such
  inputs, it should.

- **Extremely unbalanced shapes with very small N (e.g. Np=32) can have no working
  tile at all** — all tile sizes fail divisibility or resource constraints. If a shape
  produces no valid tile, fall back to (32,32,32) and verify it divides the padded shape.

- **Tile choice matters most at large shapes.** For shapes near 2048×2048×2048,
  choosing the wrong tile can cause 3–4.5× performance degradation. Profile lookup
  is especially critical in this regime.

- [2026-03-09] When M > 1024, only Mp=2048 is valid (it is the maximum ALLOWED_M).
  In this regime, all three strategies converge to the same padded shape — strategy
  selection overhead is the only differentiator. Use `preprocessed_map` as default
  for M > 1024 inputs (O(1) decision, identical result).

- [2026-03-09] For large-M inputs (M > 1024), padding overhead routinely exceeds
  NPU execution time by 5–10x. For M=1537, N=257, K=769, bf16:
  padding=13,611µs vs execution=1,700µs (ratio ~8x). Strategy selection matters
  far less than ensuring the padded shape is minimal. The padded shape is already
  at minimum when M > 1024 since Mp=2048 is forced.

- [2026-03-09] hashmap_27 neighborhood collapses for M > 1024: only 1 valid M
  candidate (2048), reducing the 27-candidate cross-product to at most 1×3×2=6
  candidates. The strategy still works correctly but loses its main advantage
  (broad neighborhood search).
