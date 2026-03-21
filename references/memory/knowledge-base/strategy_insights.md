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

- **[2026-03-21] bf16 memory constraint prevents tiles that win on i8.** The
  tile memory budget is 32,256 bytes. bf16 uses 2 bytes/element vs i8's 1 byte,
  so the effective tile element budget is halved. Tiles like (128,64,64) that
  beat (64,64,64) by -11.9% on i8 for shape (1024,768,768) require
  (128×64 + 64×64 + 64×128)×2 = 40,960 bytes — exceeding the bf16 limit.
  For bf16, (64,64,64) is often the largest tile that fits, making it the
  practical optimum even when larger tiles would be faster if they fit.

- **[2026-03-21] Small shapes (M≤64) always tie regardless of tile or strategy.**
  The NPU has a fixed launch overhead of ~157µs. For shapes where M≤64, all
  execution times cluster in the 120–170µs range regardless of tile choice.
  Tile selection has no measurable effect here. Do not waste trial runs
  comparing strategies for M≤64 shapes — the result will always be a tie.

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

- [2026-03-21] **Tile divisibility filter is too strict for Np=768 and Np=3072.**
  The power-of-2 quotient rule (Dp//t must be a power of 2) is generally correct
  but empirically fails for Np=768 and Np=3072 — both dimensions are multiples of
  3×2^k, so no standard tile gives a power-of-2 quotient, yet tiles like (64,64,64)
  pass on both. For these two N values, only divisibility (Np % n == 0) is required.
  Applying the strict filter leaves these shapes almost entirely unexplored.

- [2026-03-13] **CDO errors extend beyond K=2048 for bf16**: k=16 on K=768 (48 k-tiles)
  triggers `ValueError: Failed to generate cdo` for bf16. The CDO failure boundary
  is not solely K=2048 — avoid small k values (k≤16) on large K dimensions with bf16.
  k=32 and k=64 on K=768 are CDO-safe.

- [2026-03-13] **bf16 accuracy is tile-dependent on large shapes**: On (1024,768,768),
  tiles (128,64,32), (128,32,32), (64,32,32) produced incorrect results (atol=1e-1
  violation) while (64,32,64) and (128,32,64) passed. Tiles with k≥64 appear more
  reliable for bf16 accuracy on K=768. Avoid tiles with k=32 combined with large m on
  large K bf16 shapes unless specifically validated.

- [2026-03-19] **Tiles with m=16 or n=16 frequently fail on medium/large shapes.** When total pipeline stage count (M//m × N//n × K//k) exceeds ~256, AIE program memory overflows. On shapes ≥256 in any dim, avoid m=16 or n=16. Prefer m,n ≥ 32 on 256-512 shapes and m,n ≥ 64 on 1024+ shapes. Confirmed across 19 failures this session — all involved m=16 or n=16 on shapes ≥128.

- [2026-03-19] **Major speedup on (1024,768,768) i8: tile (128,64,64) beats (64,64,64) by -11.9%.** Prior best was (64,64,64) @ 445.9µs; new best is (128,64,64) @ 393µs. For shapes where Np=768, n=64 is safe (N//n=12 works despite not being power of 2), and larger m (m=128) gives significant improvement. This refutes the assumption that (64,64,64) is optimal for all 1024-class shapes with non-standard N.

- [2026-03-19] **For (512,512,256) bf16, tile (64,128,32) beats (64,64,64) by -2.2%.** Prior best (64,64,64) @ 225.6µs; new best (64,128,32) @ 220.6µs. Asymmetric n=128 tile wins over symmetric tile for this shape — consistent with the general heuristic that n=128 tiles tend to outperform symmetric tiles on large N shapes.

- [2026-03-19] **For (64,64,128) i8, tile (64,32,32) beats (64,64,64) by -1.6%.** Prior best (64,64,64) @ 122.8µs; new best (64,32,32) @ 120.8µs. Asymmetric tile with smaller n and k shows marginal gain on this K-dominated shape (Kp=128 vs Mp=Np=64).

- [2026-03-19] **For (128,128,64) i16, tile (32,64,16) beats (32,32,32) by -0.7%.** Prior best (32,32,32) @ 126.7µs; new best (32,64,16) @ 125.9µs. Larger n with small k produces marginal gain.

- [2026-03-13] **Asymmetric tiles with reduced k (k=32, k=64) on K-dominated shapes**:
  On (128,128,768)/i8, tiles with reduced k (e.g. (64,32,64), (128,32,64)) perform
  competitively with the standard (64,64,64) tile. (64,32,64)=139.9µs vs
  (64,64,64)=139.3µs — essentially identical. The (128,32,128) tile also performs
  well at 162µs. For K-dominated shapes, exploring varied k-tile sizes is worthwhile.

- [2026-03-20] **bf16 accuracy degrades on K-dominated shapes with very small M×N (M≤50, N≤128, K≥960).** Confirmed across shapes (1,960,32) padded to (32,1024,32) and (50,960,96) padded to (64,1024,128) — all tile sizes fail accuracy check (atol=1e-1). No tile workaround found. These shapes should use i8/i16 dtype instead of bf16.

- [2026-03-20] **K>2048 is a hard bf16 limit for both CDO generation and numerical accuracy.** K=3072 fails CDO for all tile sizes. K=2560 passes CDO but fails accuracy. Combined with prior observations (K=2048 CDO failure), the practical bf16 K-limit appears to be K≤2048 for CDO safety and K≤768 for accuracy safety on very small output matrices.

- [2026-03-20] **Shapes (M=1024, N=3072, K=768) and (M=128, N=2560, K=960) run successfully without padding when passed directly to the NPU.** N=3072 and N=2560 are not in the script ALLOWED_N list (max 2048), but the underlying GEMM module accepts them as long as N%n==0. The ALLOWED lists in the script only gate the `--use-padding` path. When no padding is needed (dimensions already cleanly divide by tile), raw out-of-range values can be passed directly.

- [2026-03-20] **For SmolVLA bf16 workloads, the most reliable tile across all 13 successful shapes was (64,64,64) for large shapes (M≥128, any K, any N) and (32,32,32) or (32,64,32) for small shapes (M≤64).** The profiling-recommended tiles sometimes failed accuracy on live runs for shapes with K≥960 and small M/N; (64,64,64) was more robust in practice for the large shapes.

- [2026-03-21] **bf16 accuracy on K=1024 shapes requires k≥128 for reliable correctness.** Across 8 passing tiles on (128,1024,1024) (from M=128,K=960,N=960) and 9 passing tiles on (128,512,1024) (from M=128,K=960,N=320), tiles with k=16 or k=32 consistently fail accuracy; tiles with k≥64 mostly pass on the (512,1024) shape and tiles with k≥128 pass reliably on both. Best tile on (128,1024,1024): (32,64,128) at 343µs. Best tile on (128,512,1024): (32,64,128) at 239µs. The pattern: larger k reduces accumulation error over many K-tiles.

- [2026-03-21] **K=3072 for bf16 causes accuracy failure (not CDO failure) for tiles that compile.** Revised from prior entry: initial testing of 3 tiles on (1024,768,3072) reported CDOError; full testing of 30 tiles shows 13 AccuracyFail + 17 MemoryExceeded + 0 passes. CDO generation succeeds for k≥32 tiles; the failure mode is numerical — bf16 accumulation over 3072/k K-tiles produces unacceptably large error. K>2048 is a hard accuracy limit for bf16 regardless of whether CDO compilation succeeds.

- [2026-03-21] **For small SmolVLA shapes (M≤64, K≤256, N≤1024), bf16 accuracy is reliable across nearly all tile sizes.** Shapes (64,128,32), (64,128,256), (64,256,128), (64,1024,128) showed 100% pass rate (43/43 tiles). Shape (64,128,1024) showed 4/10 passing — failures occur for k≤64. This confirms the k-threshold rule: for K≤256, all k values work; for K=1024, k≥128 is required.

- [2026-03-21] **On (1024,768,768) bf16, (64,64,64) is the best tile at 762µs avg / 723µs min, beating (32,64,64) at 1056µs by 28%.** CDO failures appear for tiles with k=16 on Kp=768 (CDOError confirmed for 5 tiles: (128,64,16), (256,32,16), (128,32,16), (64,32,16), (16,64,16)). Accuracy failures for tiles with very small k combined with small n (e.g. (64,64,32), (32,32,32), (32,64,32), (16,64,32)) — consistent with prior observations on this shape.

- [2026-03-21] **For (128,128,512) bf16, best tile is (32,32,128) at 145µs, followed closely by (32,32,32) at 147µs.** All 17 tested tiles pass except 2 AccuracyFail ((16,16,64) and (32,16,16)). Pattern: tiles with very small m×n product (m=16,n=16 or m=32,n=16 on small K) fail accuracy — likely due to reduced accumulation precision per tile.

- [2026-03-21] **SmolVLA bf16 experiment: bf16 accuracy on Kp=1024 shapes consistently requires k≥128.** Live 5-trial run across 13 SmolVLA shapes confirmed that tiles with k<128 fail accuracy on shapes padded to Kp=1024 (M=128,K=960,N=960 → (128,1024,1024) and M=128,K=960,N=320 → (128,512,1024)). The agentic strategy using (32,64,128) passed all 5 trials on both shapes; baseline (profiling-recommended) tiles (64,64,64) and (32,64,32) failed all accuracy checks. Profiling data updated: (128,1024,1024) and (128,512,1024) bf16 best tile changed to (32,64,128).

- [2026-03-21] **Shape (128,3072,1024) bf16 fails accuracy for all tile sizes including k≥128.** Live testing of (128,3072,1024) padded from (M=128,K=960,N=2560): tiles (32,64,128), (32,64,256), (32,64,512), (32,64,1024) all fail accuracy. Pattern: when N is very large (Np=3072) combined with K=1024 and small M=128, bf16 accumulation error exceeds tolerance across all k values. This may be specific to extremely unbalanced N-dominated shapes with K=1024.
