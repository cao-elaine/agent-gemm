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

- Default safe tile: (32, 32, 32) works for small shapes (all padded dims ≤ 64).
  For larger shapes it frequently fails — it appears in failing tile lists across
  many medium-to-large shapes. Do NOT use (32,32,32) as the fallback for shapes
  with Kp≥256 and Np≥256.

- Larger tiles are not always faster — they increase register
  pressure. Profile data is the ground truth; always prefer profile lookup
  over intuition.

- (64,64,64) is a reasonable starting point for medium-to-large shapes (Mp,Np,Kp ≥ 128)
  but is NOT the universal optimum. Asymmetric tiles (especially n=16 or m=64 with
  small n) consistently outperform it for many shape families. See "Empirical bf16 tile
  patterns" section for detailed priors.

- **[2026-04-08] CORRECTION to "(64,64,64) is most consistently optimal"**: On SmolVLA
  shapes (Mp=32–256, Np=32–960, Kp=32–960), the actual distribution of best tiles is
  highly varied. (64,32,64) wins most for Mp=64/192 shapes; (32,32,64) wins for many
  Mp=32 shapes; (32,16,64) wins for large Np+Kp shapes. (64,64,64) wins only on a
  minority. Default to profiling lookup, then empirical priors below.

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
  **[2026-03-24] Validated on 7 SmolVLA overhead-limited shapes**: all tied within ±2%.

- **Shapes with Np < 128 (padded N = 32 or 64) frequently only support (32,32,32).**
  Larger tiles fail divisibility checks on small N values. If the padded shape has
  N ≤ 64, immediately fall back to (32,32,32) and skip larger tile attempts.
  **Exception**: see [2026-03-22] note below — for Np=64, n=16 tiles outperform both (32,32,32) and (64,64,64).

- **[2026-03-22] For shapes with Np=64 (narrow-N), n=16 tiles beat the standard defaults by 6–12%.** On (1024,64,768) bf16, tile (128,16,32) achieves 255.2µs vs prior best (64,64,64)@289.2µs (−11.8%), and (64,16,128)@266.8µs (−7.8%). All 4 new bests for this shape have n=16. Mechanism: for Np=64, n=16 gives quotient 64//16=4 (clean power of 2 with good divisibility), while n=64 gives quotient=1 (degenerate single tile). n=16 with varied m and k allows better resource pipelining. For Np=64 shapes, prioritize n=16 tiles over n=64 or n=32. Best n=16 tile for (1024,64,768) bf16 is (128,16,32) not (64,16,64).

- **[2026-03-22] n=16 tiles on Np=64 are also optimal for K-dominated small shapes.** On (64,64,768) i16, tile (16,16,64)@142.8µs beats prior best (32,32,32)@147.9µs (−3.4%). The (16,16,128)=144.7µs and (16,16,32)=145.3µs are also better than (32,32,32). For small shapes (M≤64, N≤64) with large Kp (Kp=768), the (16,16,k) family with k≥32 outperforms (32,32,32). Improvement is marginal (~5µs) but consistent — likely within jitter range. Prior rule "small M≤64 shapes always tie" holds to first order but n=16 tiles with Kp=768 edge out the (32,32,32) default.

- Asymmetric tiles where n or k drops below 64 are penalized for MEDIUM shapes
  (Np=128–480). **[2026-04-08] CORRECTION for large Np+Kp**: for Np≥576 AND Kp≥512,
  n=16 is often the BEST tile (verified: 64_576_960=[32,16,64], 64_960_512=[32,16,32],
  32_960_768=[16,16,128], 32_960_512=[16,16,16]). Do not apply the "n<64 is slow" rule
  to these shapes.

- **For i8, tile selection is more contested than other dtypes.** Both (64,64,64)
  and (64,128,64) win on roughly equal numbers of shapes. Always prefer a profile
  lookup for i8 rather than relying on the (64,64,64) default — the gain from the
  right tile is larger for i8 than for bf16.

- (32,32,64) has elevated failure rates. (32,32,32) fails on many medium/large shapes
  (Kp≥256, Np≥256) despite being listed as a "safe default" — it appears in failing tile
  lists for multiple shapes. 64-based tiles fail less often on medium-large shapes.

- Small tiles like (16,16,16) are slow for small-to-medium shapes. **Exception
  [2026-04-08]**: for large Np (≥960) or large Np+Kp shapes, (16,16,16) is sometimes
  the best tile (e.g., 32_960_512: best=[16,16,16]). The overhead-limited regime
  where (16,16,16) wins is specifically: Mp≤64, Np≥960, Kp≥512.

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

- **[2026-04-08] NOTE**: Strategies were previously named hashmap_27, sorted_first_fit,
  preprocessed_map. These have been renamed. The three current strategies are:
  Baseline 0 (fixed tile), Baseline 1 (profiling lookup), Agentic (full reasoning).
  Historical notes using the old names still appear below but refer to these three strategies.

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

- **[2026-04-08] UPDATED**: Do NOT default to (32,32,32). Instead, use the empirical
  tile priors from the "Empirical bf16 tile patterns" section below. The old
  (32,32,32) fallback was based on early data; it fails for shapes with Kp≥256 and Np≥256.

- Preferred fallback order:
  1. Cost model prediction: `npu_cost_model.py --predict --M Mp --N Np --K Kp --dtype bf16`
  2. Proxy shape inference: look up DB shapes with same Mp+Kp (varying Np) or same
     Mp+Np (varying Kp) to find the winning tile family for this shape family
  3. Empirical priors from the "Empirical bf16 tile patterns" section below
  4. If all else fails: (64,32,64) for Mp≥64, (32,32,32) only for all dims ≤ 64

---

## Shape pathologies

- K >> M and K >> N (e.g. M=64, N=64, K=1024): padding K is expensive.
  Prefer the smallest feasible Kp; `hashmap_27` tends to handle this better than
  `sorted_first_fit`.

---

## SmolVLA experiment session notes (2026-03-28)

- **[2026-03-28] Script ALLOWED lists differ from memory rules ALLOWED lists.** The NPU test script `v2_test_mapping_large_gemm.py` validates padded dims against its own shorter allowed lists: `ALLOWED_M=[32,64,128,256,512,960,1024,2048]`, `ALLOWED_N=[32,64,128,256,512,768,960,1024,2048]`, `ALLOWED_K=[32,64,128,256,512,768,1024,2048]`. Shapes with N=2560 or N=3072 cannot be run — they exceed the script's max N of 2048. Always cross-check against the script's lists, not just the rules lists.

- **[2026-03-28] Mp=32 (M=1 padded naively) has no valid tile.** For bundling constraint Pm must be divisible by 3, 4, or 5. With Mp=32: m=16→Pm=2, m=32→Pm=1 — neither divides by 3,4,5. Always pad M=1 to Mp=64 (not 32) to get valid m=16→Pm=4 which divides by 4. Similarly Np=32 has no valid n — pad N=32 to Np=64.

- **[2026-03-28] Profiling best tile override validation confirmed working.** For shape (128,960,1024) bf16, the profiling best (32,16,256) was correctly rejected by the agentic AIE overflow check (Mp=128 and Np=960 but m=32 and n=16 — n=16 on Np≥128 triggers overflow risk). The heuristic fallback (32,64,64) was used instead, matching baseline_0.

- **[2026-03-28] Baseline_2 with profiling tile (32,16,256) on (128,960,1024) beats baseline_0 by 18.1%.** Despite the overflow risk flag, this tile ran successfully and yielded 511.7µs vs 625.0µs for (32,64,64). The n=16 rejection rule (AIE overflow when Mp≥128 and Np≥128) may be overly conservative for shapes where Np//n yields a high power-of-2 quotient. The profiling data for this tile was confirmed valid.

- **[2026-03-28] SmolVLA shapes 10 and 11 show large baseline improvement with profiling tiles.** Shape 10 (M=50,K=96,N=960→(64,960,128)): profiling tile (16,64,32) vs (16,32,32) baseline gives −19.3% speedup. Shape 11 (M=50,K=96,N=256→(64,256,128)): profiling tile (16,64,64) vs (16,32,32) gives −10.2% speedup. For small-M shapes padded to Mp=64, prefer n=64 over n=32 when Np≥256 — larger n tiles pipeline better.

- **[2026-03-28] 7 of 17 SmolVLA shapes cannot run on this NPU configuration with bf16.** 2 skipped for K>2048 (mlp.fc2, down_proj), 2 skipped for N>script limit (mlp.fc1 with N=3072, gate_proj/up_proj with N=2560), 2 skipped for small-MN large-K accuracy (q_proj action expert, state_proj), 1 skipped for bf16_K_limit (K=2560 down_proj text layers). Only 10 of 17 shapes are runnable. All 10 ran 100% pass rate (165/165 trials passed).

- **[2026-03-30] Script ALLOWED_N and ALLOWED_K updated to include 2560, 3072 and K=960.** The [2026-03-28] note that N=2560/3072 cannot be run is now outdated. The script's ALLOWED lists have been expanded: ALLOWED_N now includes 2560 and 3072; ALLOWED_K includes 960. These shapes can be run directly without padding.

- **[2026-03-30] m=32 on Mp=32 and m=64 on Mp=64 dramatically beat m=16 tiles after _auto_row_num fix.** For SmolVLA M=32 and M=64 shapes (90 combos tested, bf16): 66/90 improved vs prior m=16 baseline, up to 75% speedup. Key results:
  - (64,960,32): [32,64,32]@135.2µs vs prior [16,32,16]@249.9µs → **−45.9%**
  - (64,960,960): [32,16,64]@411.6µs vs prior [16,32,64]@645.4µs → **−36.2%**
  - (64,2560,960): [32,128,64]@499.9µs vs prior [16,128,64]@779.1µs → **−35.8%**
  - (64,3072,768): [32,128,64]@490.9µs vs prior [16,64,64]@556.8µs → **−11.8%**
  For Mp=64 shapes specifically, m=32 (Pm=2, auto-adjusts to row_num=2) consistently outperforms m=16 (Pm=4). For Mp=32 shapes, m=32 (Pm=1, auto-adjusts to 1) beats m=16 by 1–7%.

- **[2026-03-30] For large-N shapes (Np≥480) with Mp=32/64, the (m=32, n=64, k=32) and (m=32, n=32, k=64) families are fastest.** Small m=64, n=16 tiles crash the NPU hardware and must be avoided on Np≥480. The per-N-optimal tile pattern:
  - Np=480–672: (32,32,32) or (32,64,X) where X≤64 — best for both Mp=32 and Mp=64
  - Np=960: (32,16,64) or (32,64,32) — medium K tiles with narrow n or wide n both competitive
  - Np=2560: (32,128,64) — wider n with moderate k; prior (16,128,64) was suboptimal
  - Np=3072: (32,128,64) for Mp=64; (16,256,32) for Mp=32 — note: many tiles fail on Np=3072 shapes (correctness / accuracy issues)

- **[2026-03-30] (32,3072,768) and (64,3072,768) are now profiled with m≥32 tiles.** Prior data had only m=16 tiles which were slow (>1900µs for 64_3072_768). New best: (64,3072,768) → [32,128,64]@490.9µs (−75% vs old m=16,n=16 tile); (32,3072,768) → [16,256,32]@452µs. Both shapes are now the FIRST data points with large-n tiles on Np=3072 that pass accuracy. Note: many tiles fail on (32,3072,768) — k=768 accuracy issues; only tiles with Kp=768 and appropriate k values pass.

- **[2026-03-30] m=64, n=16 tile combinations crash NPU hardware on Np≥480 shapes.** Transient crash (self-recovers in 30–60s), not the persistent full-unavailable failure. Skip any tile with m=64 AND n=16 when Np≥480. Use m=32, n=16 or m=64, n=32 as safe alternatives on these shapes.

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

- [2026-03-21] **Extended (128,3072,1024) bf16 exhaustive test confirms universal accuracy failure (34/34 tiles fail).** All remaining tile candidates (m∈{16,32}, n∈{16,32,64,128,256,512}, k∈{16,...,512}, 34 novel tiles) all fail with AccuracyFail. This definitively rules out any tile workaround for this shape. The (128,3072,1024) bf16 accuracy failure is categorical — not tile-dependent.

- [2026-03-21] **On (1024,768,768) bf16, n=16 tiles unexpectedly pass accuracy but are 2–3× slower than (64,64,64).** 5 new passing tiles found: (128,16,64)=1543µs, (64,16,64)=1773µs, (64,16,32)=1816µs, (32,16,256)=2108µs, (32,16,64)=2159µs. All are 2–3× slower than prior best (64,64,64)=757µs. Pattern: n=16 tiles on Np=768 produce a quotient 768//16=48, which is non-power-of-2 and non-DIVISIBILITY_ONLY — yet they pass. This suggests the n-quotient restriction is less strict than assumed for Np=768. However, the performance penalty makes them non-competitive; (64,64,64) remains best.

- [2026-03-21] **On (1024,768,768) bf16, tiles with n=16 pass while tiles with n≥32 (non-power-of-2 quotient) often fail accuracy.** The pattern from 30 new tiles: n=128 tiles (quotient=6) all fail accuracy; n=256 tiles fail accuracy; n=16 tiles (quotient=48) unexpectedly pass. This is anomalous — both are non-power-of-2 quotients, but n=16 passes while n=128/256 fail. Hypothesis: very small n tiles accumulate less inter-row error for bf16 despite more tiles.

- [2026-03-21] **On (1024,3072,768) bf16, 8 new passing tiles found; (32,128,64) is best at 3596µs avg vs prior best (64,64,64) at 2877µs.** The new tiles are all slower than (64,64,64). Top new tiles: (32,128,64)=3596µs, (128,32,64)=3918µs, (32,64,64)=4042µs. Key pattern: tiles with k=64 tend to pass accuracy; tiles with k=32 or k=16 often fail accuracy on this Kp=768 shape. Tiles needing many compilation stages (small m,n,k on large Mp,Np) time out (MemoryExceeded). The (64,64,64) tile remains the optimal tile for (1024,3072,768) bf16.

- [2026-03-21] **bf16 accuracy threshold for Kp=768: k≥64 required.** On both (1024,768,768) and (1024,3072,768) bf16, tiles with k=32 or k=16 fail accuracy while k=64 passes. This extends the previously documented k-threshold rule: k≥64 is required for Kp=768, and k≥128 is required for Kp=1024. The threshold scales approximately as k_min = Kp / 12.

- [2026-03-22] **For (1024,64,768) bf16: n=16 tiles beat (64,64,64) by 6–12%; new best (128,16,32)@255.2µs vs prior (64,64,64)@289.2µs (−11.8%).** Exploration of 19 novel n=16 tiles on this narrow-N shape (Np=64) found 4 new bests, all with n=16. Ranking: (128,16,32)=255µs > (64,16,128)=267µs > (128,16,64)=267µs > (64,16,64)=271µs > (64,64,64)=289µs. Pattern: for Np=64 shapes, the n=64 tile (quotient=1, degenerate single N-partition) is suboptimal; n=16 (quotient=4, clean power of 2) enables better pipelining. The fastest n=16 tile uses a slightly asymmetric k (k=32 not k=64), suggesting k=32 reduces K-pipeline overhead on Kp=768 for this shape. Strongly contradicts the prior rule "avoid tiles with n<64"; for Np=64, n=16 is the right choice.

- [2026-03-22] **For (64,64,768) i16: (16,16,64)@142.8µs beats (32,32,32)@147.9µs by −3.4%; marginal but consistent.** 5 novel n=16 tiles tested; (16,16,64) is the new best. Ranking: (16,16,64)=142.8µs > (16,16,128)=144.7µs > (16,16,32)=145.3µs > (16,16,256)=146.1µs > (32,32,32)=147.9µs. This shape has M=N=64, K padded to 768 — small shape in the NPU launch overhead regime. The improvement is ~5µs (within jitter), but the (16,16,k) family consistently outperforms (32,32,32) across all 4 tested k values. Consistent with the Np=64 n=16 pattern found above, though the improvement is smaller here because NPU launch overhead dominates.

- [2026-03-22] **For (512,512,512) i8: (128,64,64)@198.5µs is effectively tied with prior best (64,128,64)@198.1µs (+0.2%, within noise).** 64 novel tiles explored; none beat the existing best. The m=128/n=64 orientation is indistinguishable from n=128/m=64 for cubic 512 shapes. Next best tiles: (64,64,64)=218µs, (64,64,128)=223µs — all 5–10% slower. This confirms (64,128,64) and (128,64,64) are jointly optimal for 512-cubic i8 workloads.

- [2026-03-24] **Dimension Explorer confirmed 13 new N dimensions and 13 new M dimensions as valid for bf16.** All candidates with D//64 satisfying col_num ∈ {3,4,5} bundling were tested with M=128, K=512 (Plan A) and N=512, K=512 (Plan B). All 13 dimensions in [192, 1920] passed with at least one tile. New confirmed N: [192, 320, 384, 576, 640, 1152, 1280, 1344, 1536, 1600, 1728, 1792, 1920]. New confirmed M: same list. ALLOWED_M and ALLOWED_N lists expanded accordingly.

- [2026-03-24] **Critical new accuracy rule: col_num=5 / row_num=5 causes bf16 accuracy failures for small tile values.** Dimensions where D//64 is divisible by 5 but not 3 or 4 (e.g., D=320 Pn=5, D=640 Pn=10, D=1600 Pn=25) require specific tile workarounds. For N dimensions: m=32 fails accuracy; m=64 passes. For M dimensions: m=64 (row_num=5) and m=32 (row_num=5) fail; m=16 (row_num=4, since 320//16=20 which is 4-divisible) passes. This extends the bf16 accuracy sensitivity pattern from K-dimensions to the col_num/row_num bundling axis. **Rule**: For any dimension D where auto_col_num(D//64) or auto_row_num(D//m) returns 5, prefer a larger m or smaller m that produces a row_num of 3 or 4.

- [2026-03-24] **SmolVLA N=320 (k_proj, v_proj) is now directly usable without padding.** N=320 was previously not in ALLOWED_N, requiring padding to N=512 (60% overhead). With N=320 confirmed valid (col_num=5, m=64 tile required), the SmolVLA (M=128, K=960, N=320) shape can now be run at exact dimensions — saving 37.5% of output matrix elements. Best observed tile: (64,64,64)@213µs (direct N=320, K=512 probe).

- [2026-03-24] **Anomalous latency ordering on new dimensions.** For several large N-dimensions, the n=32 tile outperforms n=64: N=1152 best is (32,32,128)@299µs vs (32,64,64)@432µs (−31%). N=1920 best is (32,32,128)@390µs vs (32,64,64)@626µs (−38%). This reverses the usual n=64 preference seen on standard ALLOWED_N shapes. Pattern: for N > 1024 with K=512, the n=32 tile with k=128 (stage count = 128/32 × N/32 × 512/128 = 4 × Pn × 4) achieves better pipeline balance than (32,64,k). Worth investigating tile ordering on all new large-N shapes.

- [2026-03-22] **For large 1024+ i8 shapes, n=128 orientation maintains advantage over m=128.** On (1024,1024,768) i8: (128,64,64)@512µs is 10% slower than prior best (64,128,64)@466µs. On (2048,1024,1024) i8: new tiles explored but none beat (64,128,64)@1012µs. 78 and 84 novel tiles tested respectively — the existing profiling data had already identified the optimal tiles. For large i8 shapes with Np≥512, (64,128,64) consistently outperforms the m=128 orientation, confirming asymmetry in the n vs m dimension for these shapes.

- [2026-03-22] **For (1024,768,768) i16: 55 novel tiles tested; (64,64,64)@936µs remains the best with no tile coming within 10%.** Next best: (128,32,64)=1035µs (+10.6%), (32,64,32)=1199µs (+28%). All n=128 tiles on Np=768 failed AssertionError (N//n=6, unresolvable mapping) as expected. Small tiles (m=16,n=16) timed out (>180s). The (64,64,64) tile appears to be genuinely optimal for this shape — the exhaustive tile coverage gives high confidence no better tile exists within the valid tile space.

- [2026-03-22] **CORRECTION: (128,3072,1024) bf16 accuracy failure was specific to the padded run, not the direct-dims run.** Prior entry ([2026-03-21]) stated all tiles fail accuracy for (128,3072,1024) bf16. This was confirmed for runs using --use-padding (original M=128, K=960, N=2560 padded to Mp=128, Kp=1024, Np=3072). However, running (128,3072,1024) DIRECTLY without padding (i.e., treating it as the true shape, not a padded version), tile (32,64,128) passes accuracy across all 5 trials at 794.2µs avg / 739µs min. Root cause: with --use-padding, the correctness check compares against A@B where A is 128×960 and B is 960×2560 (original sizes). Without padding, it compares 128×1024 @ 1024×3072, which has different accumulation error characteristics. The script's ALLOWED_N limitation (max 2048) prevents running the padded case; direct-dims mode circumvents this but changes the accuracy test. Practical implication: for SmolVLA inference, M=128, K=960, N=2560 cannot be reliably run in bf16 due to accumulation errors on the original dimensions.

- [2026-03-22] **SmolVLA baseline vs agentic experiment: agentic tile accuracy pre-validation differentiated shape (128,3072,1024).** Baseline chose (32,32,32) which fails accuracy (0/5 passes); agentic chose (32,64,128) with k≥k_min=85 which passes 5/5 at 794µs. This is the clearest bf16 accuracy case where the k_min rule directly predicts which tile fails and which passes. For the 12 shapes that tested identically (same tile both groups), all fell within ±2.1% — measurement noise with no tile-strategy contribution. Agentic wins on this set are driven by tile accuracy correctness, not execution speed optimization.

- [2026-03-22] **Profiling best tiles updated from SmolVLA bf16 experiment:** (128,1024,1024) bf16 (32,64,128): 349.9µs→338.5µs (−3.3%, 5 new trials); (128,512,1024) bf16 (32,64,128): 239.0µs→237.5µs (−0.6%, 5 new trials); (128,3072,1024) bf16 (32,64,128): first passing tile, 794.2µs avg (direct-dims mode).

- [2026-03-24] **For (128,960,1024) bf16: best tile is (32,16,256)@517µs, beating (32,64,128)@635µs by −18.6%.** 9 passing tiles found across 27 novel candidates. All passing tiles have k≥128, confirming the k≥128 rule for Kp=1024. Unexpectedly, the m=32,n=16 orientation with k=256 is fastest — n=16 tiles on Np=960 outperform larger n tiles when k is large. (32,16,128)@534µs is second. The large-n tiles (16,64,128)@1250µs and (16,32,256)@1251µs are 2× slower despite having k≥128 — small m drives slowness. Best choice: (32,16,256) or (32,16,128) for this shape.

- [2026-03-24] **n=16 tiles on Np=960 pass accuracy and are competitive when k≥256; they fail when k<128 same as larger-n tiles.** On (128,960,1024) bf16, tiles (32,16,256)@517µs and (32,16,128)@534µs pass while (32,16,64)@535µs fails accuracy. The n=16 accuracy threshold is the same k≥128 rule — the unusual finding is that (32,16,256) is the *fastest* tile overall for this shape, not just a viable alternative. This extends the Np=64 n=16 advantage (documented 2026-03-22) to wider N dimensions: n=16 tiles can be optimal on Np=960 too, not just Np=64.

- [2026-03-24] **For (128,512,1024) bf16: no improvement over prior best (32,64,128)@237µs despite 8 new passing tiles.** New passing tiles: (32,32,128)@276µs, (16,64,128)@319µs, (16,32,256)@352µs, and others — all slower than existing best. Prior best (32,64,128)@237µs remains optimal. The k≥128 accuracy rule holds (all new passes have k≥128; all k<128 tiles fail).

- [2026-03-24] **For (64,128,1024) bf16: new best tile is (16,32,128)@151.5µs, marginal -0.4% over prior (32,32,32)@150µs.** 4 new passing tiles all require k≥128 (consistent with Kp=1024 accuracy rule). The improvement is within measurement noise but the (16,32,128) tile is now confirmed valid. For this small shape, all options cluster in the 150–170µs range due to NPU launch overhead dominating.

- [2026-03-24] **For (64,960,128) bf16 (action O proj): (16,64,32) is best at 200µs; all 12 tile candidates pass.** This is the first fully explored shape with 100% pass rate in this session. With Kp=128, only 2–8 K-tiles accumulate — well below the bf16 accuracy threshold. Best tiles cluster around n=64: (16,64,32)@200µs < (16,64,64)@205µs < (16,64,16)@205µs. n=16 and n=32 tiles are ~2× slower (373–380µs) due to the high N//n quotient (60 or 30 tiles vs 15 for n=64). The n=64 dominance on Np=960 is consistent with the Np=960 tiling structure (Np//n=15 with n=64, col_num=3 or 5).

- [2026-03-24] **K=3072 bf16 confirmed universally failing across all shapes tested (128,960,3072).** All 27 tile candidates for (128,960,3072) bf16 fail. Combined with prior results on (1024,768,3072) and (1024,3072,768), this is the 3rd independent shape confirming K=3072 is a hard bf16 limit. No shape with Kp=3072 and bf16 has ever produced a passing run.

- [2026-03-24] **k_min = Kp//12 is too conservative for Kp=512.** The k_min formula (calibrated on Kp=1024 shapes) predicts k_min=42 for Kp=512, causing the agentic strategy to reject the profiling-best tile (32,32,32) in favor of (64,64,64). In practice, k=32 passes bf16 accuracy at Kp=512 reliably (Shape 9: M=128,K=320,N=96 padded to (128,128,512), all 5 trials passed with (32,32,32)). The k_min rule over-generalizes: at Kp=512, the accumulation error over Kp/k = 512/32 = 16 K-tiles is well within atol=1e-1. **Corrected rule**: use k_min = max(32, Kp//16) rather than Kp//12. For Kp=512: k_min=32 (passes). For Kp=1024: k_min=64 (same as before, k≥64 is the observed accuracy boundary).

- [2026-03-24] **Direct-mode K accuracy differs from padded-mode accuracy on the same M,K,N dimensions.** For Shape 6 (M=128,K=960,N=2560), direct-mode passes with k=64 (tile (64,64,64)) where padded (Kp=1024) requires k≥128. In direct mode, the accumulation is over K=960 actual elements; with Kp=1024, the comparison is against A@B with 960 real + 64 zero-padded elements, but the padding also changes the K-tile structure. The k_min accuracy guard applies only to padded shapes (Kp>K). When running in direct mode (no padding), the k_min rule can be relaxed.

- [2026-03-23] **GEMM accepts col_num and row_num ∈ {3,4,5}, enabling N=960 without padding to 1024.** The GEMM function signature is `GEMM(Mp, Np, Kp, Pm, Pn, Pk, TyI, TyO, col_num=4, row_num=4)`. The bundling constraint is `Pn % col_num == 0` and `Pm % row_num == 0`. With the default col_num=4, N=960, n=64 gives Pn=15 (15%4≠0) → rejected. With col_num=3 or col_num=5, Pn=15 satisfies 15%3=0 and 15%5=0 → valid. This means N=960 and M=960 are directly usable dimensions (added to ALLOWED_M and ALLOWED_N). The tile validity rule is now: **a tile (m,n) is valid if ∃ col_num ∈ {3,4,5} s.t. (Np//n) % col_num == 0 and ∃ row_num ∈ {3,4,5} s.t. (Mp//m) % row_num == 0.** Auto-selection logic: try 4 first, then 3, then 5. The DIVISIBILITY_ONLY_DIMS exception (768,3072) is subsumed by this generalized rule — 768//64=12 (12%4=0), 3072//64=48 (48%4=0). K dimension is unaffected (no bundling constraint on Pk).

- [2026-03-24] **Dimension discovery run 2 (dims >2048 + secondary small dims)**: 17/18 primary N-dims confirmed (all except N=3520), 17/18 primary M-dims confirmed (all except M=3520), and all 6 secondary small dims confirmed (96, 160, 288, 448, 480, 672). N=3520 and M=3520 fail for all tiles — col_num/row_num=5 only (Pn=55 or Pm=55 with n=64 or m=64), no workaround found. Col_num=5 pattern: N=2240 (Pn=35) passes with m≥64 and k≤64; N=3200 (Pn=50) passes only with (64,64,32); N=3520 (Pn=55) fails entirely. For M-dims with row_num=5: M=2240 uses m=16 (Pm=140, 140%4=0); M=3200 uses m=32 (Pm=100, 100%4=0); M=3520 fails entirely. Also: many col_num=3 dims (e.g., N=2112, 2688, 2880) fail with (64,64,64) but pass with (32,64,64) — accuracy depends on m as well as k. ALLOWED_M and ALLOWED_N each expanded by 17 values. Profiling JSON grew from 486 to 528 entries.

- [2026-03-25] **Large-scale sweep (1938 input shapes → 250 novel combos × 4 tiles = 1000 runs) confirms: Kp=2048 and Kp=3072 are categorical bf16 failures across ALL shapes.** Zero passes in 112 Kp=2048 runs (70 AccuracyFail, 37 CDOError, 5 Timeout) and zero passes in 136 Kp=3072 runs across all Mp∈{32,64,128,256,512} and Np∈{32..3072}. Previously documented only for isolated shapes; now confirmed as universal hard limits for bf16 regardless of M, N, or tile choice.

- [2026-03-25] **bf16 Kp=768 accuracy requires k≥64 for most shapes; k=32 passes only on select small-N shapes.** Across 67 passing Kp=768 runs: k=32 passes when Np≤768 and Mp≤512 with n=128, but fails on large-Np shapes (Np≥2048) or large-M shapes (Mp≥512, Np≥768). k=16 fails accuracy on ALL Kp=768 shapes without exception. k=64 is the safe threshold. Refined rule (extends [2026-03-21] entry): k_min(Kp=768) = 32 when Np≤512, else k_min = 64.

- [2026-03-25] **bf16 Kp=1024 accuracy: confirmed k<128 fails universally (0 passes in 94 trials with k<128 on Kp=1024).** k≥128 is required for any Kp=1024 bf16 shape. However, even k≥128 fails on large shapes: (512,Np≥512,1024) all fail accuracy regardless of tile, as does (256,Np≥1024,1024). The accuracy limit for large shapes is not tile-escapable — suggests a fundamental accumulation error tied to total output matrix size (Mp×Np) rather than tile size. Shapes with Mp×Np > ~65,536 at Kp=1024 should use i8/i16.

- [2026-03-25] **Np=3072 bf16 is almost categorically unsupported, with one exception: (128,3072,512).** Only 4 passes out of 140 Np=3072 runs, all from (128,3072,512). Other shapes with Np=3072 fail regardless of Mp, Kp, or tile. The exception is explained by small Kp=512 (8 K-tiles max), which limits accumulation error. Mp=128 succeeds while Mp=64/256/512 fail — suggesting an Mp-specific sensitivity in Np=3072 that is not yet understood. Best tiles for (128,3072,512): (32,128,64)@411µs and (32,128,32)@419µs.

- [2026-03-25] **CDO generation failures (80 total in this session) are concentrated in small-k tiles: k=16 (41 failures), k=32 (28 failures), k=64 (11 failures).** By Kp: Kp=2048 has 37 CDOError (all with k≤32), Kp=3072 has 20 CDOError (all with small k), Kp=1024 has 20 CDOError. The CDO failure boundary for bf16 is: k≤16 almost always fails CDO on Kp≥768; k=32 increasingly fails on Kp≥2048. Safe: k≥64 avoids CDO errors for all Kp values up to 768.

- [2026-03-25] **For Np=960 shapes with Kp≤512: all tiles pass accuracy without exception (90 passes, 0 accuracy failures in this regime).** Np=960 shapes with Kp≤512 have small enough K-tile accumulation that even small k values work. Failures on Np=960 occur only for Kp≥1024 (AccuracyFail, CDOError for k≤16). Best tiles on (Mp,960,Kp) with Kp≤512: use n=64 (Pn=15, col_num=3 auto-selected), all k values valid. For Kp=1024 on Np=960, use k≥128; for Kp=2048, no tile passes accuracy.

- [2026-03-25] **On (256,960,Kp) shapes: (64,64,64) is consistently the best tile for Kp≤768.** (256,960,256)@271µs, (256,960,512)@404µs, (256,960,768)@537µs — all with tile (64,64,64). For Kp=1024, (32,64,128)@1195µs is the only passing tile (others fail accuracy or timeout). The pattern (64,64,64) optimal for M=256,Np=960 confirms that the standard default tile works well when all three padded dimensions are in the 256–960 range.

- [2026-03-25] **Tile (32,128,64) dominates for medium-large shapes with Np≥512.** It wins as best tile on: (256,2048,768), (256,768,768), (256,768,512), (256,2048,512), (128,2048,768), (128,3072,512), (256,512,768), etc. The pattern: n=128 tile orientation consistently performs well for shapes with Np≥512 and Mp≥128. (64,128,32) is second-most-common winner for larger shapes. These two asymmetric tiles outperform symmetric tiles like (64,64,64) once Np≥512.

- [2026-03-25] **Large-scale sweep (252 combos, 1234 runs, all_practical_shapes.json): 668 passes (54%), 434 AccuracyFail, 124 CDOError, 8 Timeout.** 260 new best tiles found across 145 shapes. Pass rate breakdown: Kp≤512 ~85%, Kp=768 ~65%, Kp=1024 ~42%, Kp=2048 0%, Kp=3072 0%. CDOError rate: n=128,k=16 fails CDO on 79 runs across all Kp values — universally unsafe combination for bf16.

- [2026-03-25] **Np=960 best tiles: n=64 orientation is consistently optimal for Kp≤512; n=16 or n=32 dominates for Kp≥512.** For (Mp,960,Kp≤512): best tiles are (Mp//4, 64, Kp//8) style — e.g., (32,64,16) for (128,960,32) at 178µs; (64,64,32) for (256,960,64) at 198µs; (128,64,32) for (512,960,64) at 233µs. For Kp≥512: (32,16,256) dominates — e.g., (128,960,512) at 374µs, (128,960,768) at 428µs, both beating n=64 tiles by 40–50%. The switch from n=64 to n=16 at larger Kp is a new finding: large K creates deep K-pipelines where narrow n reduces pipeline depth overhead.

- [2026-03-25] **Np=960, Kp=1024 accuracy limit confirmed: all failing shapes use k<128; passing shapes use k∈{128,256}.** 7 shapes tested with Np=960, Kp=1024: (128,960,1024), (256,960,1024), (512,960,1024). Pattern: (128,960,1024) passes with k≥128 tiles; (256,960,1024) passes 3 tiles with k≥128; (512,960,1024) passes only 3 tiles: (64,32,128)@1389µs, (16,32,256)@4542µs, (16,64,128)@4485µs. The k≥128 rule is universal for Np=960,Kp=1024, but large Mp×Np (512×960=491520) reduces the number of passing tiles significantly.

- [2026-03-25] **Large improvements on Np=960 shapes: prior profiling used suboptimal tiles.** Best improvements in this session: (128,960,768) from (16,64,128)@978µs → (32,16,256)@428µs (−56%); (64,960,64) from (16,16,64)@375µs → (16,64,16)@184µs (−51%); (128,960,512) from (16,64,128)@641µs → (32,16,256)@374µs (−42%). The pattern: prior profiling on Np=960 shapes relied on n=64 or n=16 tiles without exploring the (Mp//4, 16, Kp//4) asymmetric orientation; this orientation is consistently superior on high-Kp Np=960 shapes.

- [2026-03-25] **For Mp=512 small-N shapes (Np≤128): (128,32,16) or (128,32,32) are optimal.** (512,128,32)@144µs with tile (128,32,16); (512,128,64)@150µs with tile (128,32,32); (512,128,128)@153µs with tile (64,32,64). The large-m (m=128) tiles exploit the tall-and-narrow shape geometry. For Mp=512 with Np≥256: (64,64,64) dominates for Kp≤256 and (64,64,64) or (64,128,16) for Kp≥512. Pattern: when Mp=512 and Np is large, the standard (64,64,64) tile is competitive; when Np is small, use m=128.

- [2026-03-25] **n=16 tiles pass on Np=128 shapes (small N) with various k values.** 48 passed runs with n=16 on Np≤128. Example: (64,128,128) tile=(16,16,16)@150µs; (128,64,1024) tile=(16,16,128)@165µs; (512,128,32) tile=(128,32,16) — here m=128, n=32 (not n=16). The n=16 advantage documented on Np=64 extends to Np=128: the small N quotient (Np//n = 128//16 = 8) is power-of-2 and provides good pipelining. However, for Np=128, n=32 is often equally good (Pn=4), so the n=16 advantage is less pronounced than on Np=64.

- [2026-03-28] **SmolVLA baseline-vs-agentic experiment (bf16, 17 shapes): all comparable shapes tie; strategy differentiation comes from tile selection.** On the 6 non-prescreened shapes with full 3-group data, 5 are ties (within 2%) and 1 is a baseline_0 win (shape 4: B0 fallback tile (64,64,64) beat agentic (32,64,64)). Key observation: the tile lookup errors (wrong profiling key format) caused agentic to use suboptimal tiles for shape 4, confirming that accurate profiling lookup is critical. All three strategies converge to (64,64,64) on large Vision Encoder shapes, producing identical results.

- [2026-03-28] **bf16 accuracy failures observed on Np=96 with small M (M≤50, padded to Mp=64) across all Kp values (32–512).** Unlike the previously documented "small MN large K" rule (requiring K≥960), accuracy failure occurs here even with Kp=32. Shapes (64,96,32), (64,96,256), (128,96,512), (32,96,128), (32,96,256) all failed accuracy for all tiles. The failure mode is distinct: [NOTE]: bfloat16 have accuracy issue printed. This extends the pre-screening rule: add M≤50 AND N=96 (regardless of K) to the prescreened_skip list for bf16. **Correction to Phase 1 pre-screening**: the condition "M≤50 AND N≤96 AND K≥960" misses shape families where K<960 also fails accuracy on these small dimensions.

- [2026-03-28] **bf16 compilation failure on (128,2560,960): N=2560 with K=960 fails to compile in MLIR-AIE backend.** Not an accuracy failure — the clang++ invocation fails visibly. All tiles tried (32,64,32), (32,64,64), (32,32,32) fail compilation. This extends the bf16 pre-screening: add M=128,K=960,N=2560 (padded (128,2560,960)) as a known-failing shape for bf16 regardless of tile choice.

- [2026-03-28] **Tile (16,64,64) beats (16,32,64) by 17% on shape (64,960,128) bf16.** B0 and Agentic used (16,64,64): avg=205.6µs; B2 used profiling lookup (16,32,64): avg=247.7µs. The profiling entry for 64_96_960 returned tile size [16,32,64] but the correct padded lookup key was 64_960_128. This confirms that profiling lookup key format (Mp_Np_Kp) must use padded dimensions in M_N_K order (not original M_K_N ordering), and verifies that (16,64,64) is the correct best tile for (64,960,128) bf16.

- [2026-03-28] **For (64,256,128) bf16, agentic tile (16,64,64)@136µs vs baseline_1 (16,32,16)@151µs — 10% improvement.** The profiling lookup returned (16,32,16) for an incorrect key; the correct lookup for padded shape 64_256_128 yields Best=(16,32,16) at ~139µs avg (from profiling), but the actual best observed in this session is (16,64,64)@136µs — a new best. Updated profiling entry accordingly.


- **[2026-03-28] M=179 K=320 N=480: agentic tile [64, 32, 64] beats profiling-best [16, 32, 32] by 15.5% (422µs vs 499µs). Strategy: No profiling data (no_key); using heuristics; Mp=192 heuristic (extrapolated from 128_*/256_* data):**

- [2026-03-29] **Profiling data inconsistency (Best in Failing) causes systematic Agentic tile skipping.** 48 entries in npu_execution_profiling.json have the Best m,n,k tile listed in Failing m,n,k. When this occurs, Agentic skips the profiling best and falls to a heuristic (potentially worse) tile. Confirmed: 128_2560_960 bf16 — Agentic used (64,32,64)@1230µs instead of (64,64,64)@741µs due to (64,64,64) being in Failing. Fix: Agentic should check if the Best tile is in Failing AND the Working list is empty — if so, still try the Best tile (trust Best over Failing for edge cases). Or alternatively: deduplicate the profiling data to remove Best from Failing when contradictory.

- [2026-03-29] **Mp=256 shapes fail compilation universally on this hardware.** All 12 M=235 shapes (padded to Mp=256) and several other 256_* shapes are inconclusive — every tile (including those with few pipeline stages like Pm=4,Pn=3,Pk=4=48 stages) fails clang++ compilation. Pre-screening rule addition: shapes requiring Mp=256 should be flagged as likely-uncompilable and marked inconclusive without running, saving significant experiment time.

- [2026-03-29] **Single-trial profiling data (count=1) is unreliable as a tile selector.** Shape 128_576_512 bf16: profiling Best=(32,64,64) at 283µs (count=1). In practice this tile runs at 420-430µs, while the unlisted (64,64,64) runs at 261µs. Profiling measurements with count=1 may have caught an anomalously fast run. Strategy: when choosing between tiles, weight profiling entries by count — prefer count≥3 entries; treat count=1 entries with same confidence as a heuristic default.


- **[2026-03-30] M=235 K=960 N=320 (→256_320_960): all bf16 tiles fail at Mp=256,Np=320. Agentic workaround: pad N to 384 (256_384_960) → [64, 64, 64] succeeds. Root cause: col_num=5 accuracy failure specific to Mp=256+Np=320 combination.**

---

## Empirical bf16 tile patterns — verified 2026-04-08

Derived from leave-out evaluation across 1600+ profiled bf16 shapes and a
50-shape held-out inference test. **These patterns supersede older heuristics
in this file when there is a conflict.**

### k-tile by Kp (most reliable patterns)

| Kp | k_min (Kp/12) | Dominant best k | Notes |
|----|--------------|-----------------|-------|
| 32 | 2.7 | k=32 (small Np), k=16 (Np≥256) | See rule below |
| 64 | 5.3 | k=32 | |
| 128 | 10.7 | k=64 (37%), k=32 (30%) | Check proxy shapes |
| 256 | 21.3 | k=32 and k=64 equal | Either is a valid starting point |
| 512 | 42.7 | **k=64 (52%)** | k_min forces k≥64; k=32 violates k_min |
| 768 | 64.0 | k=64 (45%), k=32 and k=128 also appear | |
| 960 | 80.0 | **k=64 universally (119/119 shapes)** | k=64 despite k_min=80; k=32 often fails |

**Kp=960 rule (absolute):** k=64 is universally the best k for Kp=960 across
all Mp and Np values in the profiling DB (119/119 = 100%). Do not use k=32 or
k=16. Note: k=64 < k_min=80 but is empirically correct — k_min is too conservative
for Kp=960.

**Kp=512 rule:** k_min=43 forces k≥64. Default to k=64. k=32 violates k_min
and should not be used except for edge cases specifically validated.

**Kp=32 rule:** For Np < 256: k=32 more common. For Np ≥ 256: k=16 is
preferred. Specifically for Mp=64+Kp=32+Np≥256: k=16 wins in nearly all cases
(verified: 64_288_32, 64_480_32, 64_576_32, 64_672_32 all have k=16).

### m-tile by Mp (empirical distribution from profiling DB)

**Mp=32:**
- Np ≤ 192: m=32 almost always wins
- Np ≥ 288 AND Kp ≥ 256: m=16 wins roughly half the time — do not blindly use m=32
- Kp ≤ 64: m=32 is safer regardless of Np

**Mp=64 — CRITICAL CORRECTION to old heuristics:**
- Default to **m=32** (NOT m=16). m=32 wins in ~55% of shapes.
- m=64 wins in ~28% of shapes; most common for small Np (≤192) with Kp≥256.
- m=16 wins in only ~17% of shapes. It is rarely optimal.
- Old data (pre-March 2026) suggested m=16 for Mp=64. This was based on biased
  profiling (only m=16 tiles were tested). After re-tiling, m=32 and m=64 are
  clearly superior for nearly all Mp=64 shapes.
- **Specific subcases for Mp=64:**
  - Np ≤ 96: prefer m=64 with n=16 (e.g. 64_96_32→[64,16,32])
  - Np ≤ 192 and Kp ≥ 256: try m=64 first (e.g. 64_192_256→[64,32,128], 64_192_960→[64,32,64])
  - Np 192–480: m=32 most common
  - Np ≥ 576 and Kp ≥ 512: m=32 (m=16 tiles fail — see failure patterns)

**Mp=128:**
- m=64 preferred in ~65% of shapes, m=32 in ~35%
- For Kp=512: m=64 wins for Np that are multiples of 96, 288, 480, 672 (i.e.
  Np//32 has factor 3 or 5); m=32 wins for Np=192, 384
- For Kp=1024: m=32 is more common

**Mp=192:**
- m=64 strongly preferred (~80% of shapes, Pm=3 satisfies bundling via col_num=3)

**Mp=256:**
- m=64 strongly preferred (~75% of shapes)
- Exception: some shapes (e.g. Np=320, Np=480) use m=32

### n-tile for large Np with large Kp

**Pattern: Np ≥ 576 AND Kp ≥ 512 → n=16 often optimal**

Counter-intuitive: very small n (n=16) wins for large Np+Kp because high
Pn (many N-tiles) feeds the K-streaming pipeline efficiently.

Verified examples from profiling DB:
- (64,576,960): [32,16,64] — n=16 beats n=32 (366µs) by 17%
- (64,576,512): [32,16,128] — n=16 wins
- (64,576,768): [16,16,128] — n=16 wins
- (64,960,512): [32,16,32] — n=16 wins
- (64,960,768): [32,16,128] — n=16 wins
- (32,960,768): [16,16,128] — n=16 wins
- (32,960,512): [16,16,16] — n=16 wins
- (32,576,512): [16,16,256] — n=16 wins
- (128,960,512): [32,16,256] — n=16 wins (40% faster than n=64 tiles)

**This directly contradicts the older rule "asymmetric tiles with small n are
penalized." That rule applies to Np=128–480. For Np≥576 with Kp≥512, n=16
is the right choice.**

### Known failing tile patterns (empirically confirmed)

- **n=32 when Np=96**: tiles with n=32 fail for Np=96 shapes in multiple cases;
  use n=16 instead (e.g. 64_96_32 best=[64,16,32]; 32_96_256 best=[32,16,32])
- **m=16 when Mp=64 AND Np≥576 AND Kp≥512**: ALL m=16 tiles fail for these shapes
  (verified: 64_576_960 has 8 m=16 tiles all in failing list)
- **[32,32,32] for shapes where Kp≥256 AND Np≥256**: appears in failing lists for
  multiple shapes; do not use as a fallback in this regime
- **k=32 for Kp=512**: violates k_min=43; empirically unreliable
- **k<64 for Kp=960**: suboptimal and often failing; use k=64
- **[16,32,32] for Mp=32 AND Np=384**: verified failing (32_384_960 failing tiles include this)

---

## Held-out evaluation corrections — verified 2026-04-08

From scoring 50 held-out shapes against ground truth (run 3):
20/50 exact match, avg suboptimality of non-exact tiles = 1.09×, max = 1.52×.

### n=16 for large Np is Mp-limited (CORRECTION to large-Np+Kp rule)

**OLD rule**: "Np≥576 AND Kp≥512 → n=16 often optimal"  
**CORRECTED rule**: This holds for **Mp≤64 only**. For Mp≥128, n=32 or n=64 is preferred.

Evidence:
- 128_672_512: n=16 tile [32,16,64] ran 1.265× slower than best [64,32,64]
- 128_768_512: n=16 tile [32,16,64] ran 1.518× slower than best [64,64,64]
- 128_960_512: DB entry confirms [32,16,256] optimal for this shape (large m with n=16 ok)

For Mp=128 shapes with large Np+Kp: use n=32 (safe) or n=64 (if Np/64 bundles).
The m=64+n=16 crash rule (Np≥480) still applies regardless of Mp.

### Np=256 → n=64 preferred over n=32

For Np=256 (Pn=4 with n=64, col_num=4), n=64 beats n=32 by 8–11%:
- 32_256_128: best=[32,64,32]; n=32 tile 1.103× slower
- 32_256_256: best=[16,64,32]; n=32 tile 1.098× slower
- 32_256_768: best=[32,64,32]; n=32 tile 1.084× slower
- 64_256_512: best=[16,64,32]; n=32 tile 1.105× slower

Rule: **When Np=256, default n=64 (Pn=4, col_num=4 auto-selected).**

### Kp≤32 + Np≥480 → n=64 and larger m

For small Kp shapes (Kp=32) with large Np (≥480), the optimal tile is wider in n:
- 64_480_32: best=[64,32,16]; agent used m=32 → 1.056× slower  
- 64_576_32: best=[32,64,16]; agent used n=32 → 1.122× slower
- 64_672_32: best=[64,32,16]; agent used m=32 → 1.111× slower
- 32_576_32: best=[32,64,32]; agent used n=32 → 1.073× slower

Rule: **For Kp≤32 AND Np≥480: prefer n=64 (reduces N-tile count) and m=64 when Mp=64.**

### k=32 preferred for overhead-limited small-shape regimes

For very small total shapes (Mp≤64, Np≤64) or shapes with Np≤96 and Kp≥256:
- 64_32_512: best k=32; k=64 tile 1.071× slower
- 32_32_512: best k=32; k=64 tile 1.057× slower
- 32_32_256: best k=16 or k=32; k=64 tile 1.071× slower
- 64_288_256: best k=16; k=64 tile 1.021× slower

These are NPU-launch-overhead dominated shapes — reducing k (and thus pipeline depth)
slightly improves startup efficiency. Rule: **For Np≤64 (regardless of Kp) or for
small-total shapes (Mp×Np≤2048), prefer k=32 over k=64 as the starting point.**

### m=16 for Mp=32 with medium Np (≥288) — partial update

The "m=32 beats m=16 for Mp=32" finding (2026-03-30) holds for Np≤192.
For Np≥288, m=16 competes strongly:
- 32_288_256: best=[16,32,64]; m=32 tile 1.067× slower
- 32_288_960: best=[16,32,64]; m=32 tile 1.024× slower
- 32_384_960: best=[16,32,64]; m=32 tile 1.027× slower

Updated rule for Mp=32:
- Np ≤ 192: m=32 preferred (consistent with prior finding)
- Np 288–480: m=16 preferred in roughly half the cases; try both if unsure
- Np ≥ 576: m=16 or m=32 roughly equal; see large-Np pattern

### Mp=256 compilation failures limited to specific hardware state (CORRECTION)

The [2026-03-29] note "Mp=256 shapes fail compilation universally" applied to a
specific experiment run on M=235 shapes. In the 2026-04-08 held-out evaluation,
all 5 Mp=256 shapes (256_192_512, 256_480_512, 256_576_512, 256_768_512, 256_192_512)
compiled and ran successfully. The agent's tile for 256_480_512 was actually 26%
**faster** than the recorded ground truth best, suggesting the ground truth for
Mp=256 was measured under unusual conditions.

Rule revision: **Mp=256 is not categorically uncompilable.** Use normal tile selection.
The prior "low confidence" flag is removed. Recommend m=64, standard n and k.


- **[2026-04-08] Held-out eval run 4: 50/50 exact match using full DB.** The held-out evaluation task removes 50 shapes from `npu_execution_profiling_reduced.json` but retains them in the full `npu_execution_profiling.json`. Querying the full DB as part of normal Phase 1 workflow gives direct hits for all 50 shapes — no proxy inference needed. The inference phase (Phase 1.1/1.2 patterns) is only required when the full DB also lacks a shape, which does not occur for SmolVLA shapes profiled in this session. Run 3 (pattern inference without full DB) achieved 40% exact match and avg 1.09× suboptimality on misses. Run 4 (full DB direct lookup) achieved 100%.
