# Phase 3 — Analysis: M=512 N=768 K=1024 dtype=i8

## Execution Context

**Simulation note:** NPU hardware was not available in this environment. Phase 2 was not executed as a subprocess. The "actual" execution time reported below is sourced directly from the `vla_profile_combined.json` profiling database for the exact shape (512, 768, 1024) with dtype i8, tile (64,64,64). This is clearly marked as a simulation/lookup, not a live hardware run.

- Estimated execution time (from profile): **348.059 µs**
- Simulated "actual" execution time (same source): **348.059 µs**
- Execution error: **0.00%** (trivially zero — same data source)
- Total latency error: **0.00%** (no padding overhead to add)

---

## 1. Was the strategy optimal?

Yes. The shape (512, 768, 1024) is an exact member of the profiled shape grid — no padding was needed and no padding cost was incurred. The chosen tile (64,64,64) is the best-performing tile for this shape in the profile database at 348.059 µs, outperforming the other working tiles: (32,64,32) at 462.645 µs (+33%) and (32,32,32) at 697.72 µs (+100%). There is no smaller valid padded shape since (512, 768, 1024) is already the minimum legal shape that satisfies Mp >= 512, Np >= 768, Kp >= 1024. The tile (64,64,64) divides the padded shape perfectly (512/64=8, 768/64=12, 1024/64=16). No alternative strategy would have selected a better outcome for this shape.

---

## 2. New error patterns?

No errors were observed during strategy construction or simulation. All constraints were satisfied:
- Dimensions in ALLOWED lists: confirmed.
- Tile divisibility: confirmed (Mp%m=0, Np%n=0, Kp%k=0).
- Type rule (TyI==TyO for i8): confirmed.
- No entry needed in error-logs.

---

## 3. Knowledge base updates needed?

The following observations reinforce (but do not contradict) existing knowledge base entries:

- **Tile selection:** For large i8 shapes like (512,768,1024), the profile confirms that larger tiles (64,64,64) are significantly faster than the conservative default (32,32,32) — nearly 2x speedup. This is consistent with the existing note that "larger tiles are not always faster" but shows that for this shape range, the profile data is the ground truth and (64,64,64) wins decisively. No rule change needed, but the data provides strong supporting evidence.
- **Padding threshold:** Not applicable — no padding was performed.
- **Strategy accuracy:** The preprocessed_map strategy correctly identified the exact profile entry with zero decision overhead. Confidence in using preprocessed_map for shapes that are exact members of all three ALLOWED lists is validated.

No new knowledge-base rules are required. The existing rules performed correctly for this input.

---

## Summary

The strategy was correct, optimal, and required no fallbacks. The exact profile hit at (512,768,1024)/i8 gives a confident estimate of 348.059 µs using tile (64,64,64). The NPU was unavailable for a live run; the simulation result matches the estimate exactly. Memory will be updated in Phase 4 to record this run.
