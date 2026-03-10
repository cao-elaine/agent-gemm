# Strategy Insights — Knowledge Base

Maintained by the agentic NPU optimizer. Each entry is dated.

---

## Padding strategy

- [INITIAL] The threshold for switching from `manual_copy` to `numpy_pad` is
  approximately 500 000 total elements padded across A+B+C.
  `torch_pad` is competitive with `numpy_pad` for bf16 on large shapes.

- [INITIAL] For very small inputs (M,N,K all < 64), padding to the next 64
  is nearly free — NPU launch overhead dominates anyway.

---

## Tile selection

- [INITIAL] Default safe tile: (32, 32, 32) works for all dtypes and all
  legal (Mp, Np, Kp) combinations.

- [INITIAL] Larger tiles are not always faster — they increase register
  pressure. Profile data is the ground truth; always prefer profile lookup
  over intuition.

---

## Strategy selection rules

- [INITIAL] `hashmap_27` is the most reliable all-around strategy for inputs
  that fall near the interior of the profiled shape grid.

- [INITIAL] `sorted_first_fit` can pick shapes with very large padding when
  the fastest profiled NPU shape is much larger than the input. Check the
  padding overhead before committing.

- [INITIAL] `preprocessed_map` is the fastest at decision time (< 1µs) and
  best for shapes that are already profiled exactly.

---

## Correctness

- [INITIAL] bf16 correctness tolerance: atol=1e-1 (bfloat16 has limited
  mantissa precision).
- [INITIAL] i8 / i16 correctness tolerance: atol=1e-5 (exact integer arithmetic).

---

## When no profile entry exists

- [INITIAL] Fall back to minimum ALLOWED shape that satisfies Mp >= M, Np >= N,
  Kp >= K with tile (32, 32, 32).

---

## Shape pathologies

- [INITIAL] K >> M and K >> N (e.g. M=64, N=64, K=1024): padding K is expensive.
  Prefer the smallest feasible Kp; `hashmap_27` tends to handle this better than
  `sorted_first_fit`.

- [INITIAL] M just above a power-of-2 boundary (e.g. M=1025): `sorted_first_fit`
  often picks a 2× larger padded shape (M'=2048). Check `hashmap_27` result — it
  usually finds a smaller option.

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
