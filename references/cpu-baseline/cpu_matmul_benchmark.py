"""
CPU matmul benchmark for SmolVLA GEMM shapes.

Runs numpy matmul for every shape in smolvla_gemm_shapes.json, profiles
the raw matrix multiplication time (no padding, no data movement overhead),
and saves results to cpu_matmul_profiling.json in the same directory.

Output format mirrors npu_execution_profiling.json for easy comparison:
  {
    "M_N_K": {
      "bf16": {
        "trials": [t1_us, t2_us, t3_us],
        "avg_us": <float>,
        "min_us": <float>,
        "layers": [...]
      }
    },
    ...
  }
"""

import json
import os
import time
import numpy as np
from ml_dtypes import bfloat16 as np_bf16

TRIALS = 3
SHAPES_FILE = os.path.join(os.path.dirname(__file__), "../../references/smol-vla-dataset/smolvla_gemm_shapes.json")
OUT_FILE    = os.path.join(os.path.dirname(__file__), "cpu_matmul_profiling.json")

DTYPE_MAP = {
    "bf16": np_bf16,
}


def run_shape(M: int, K: int, N: int, dtype, trials: int) -> dict:
    """Run matmul M×K @ K×N for `trials` repetitions. Returns timing in µs."""
    # Allocate once, reuse across trials
    A = np.random.randn(M, K).astype(dtype)
    B = np.random.randn(K, N).astype(dtype)

    times_us = []
    for _ in range(trials):
        t0 = time.perf_counter()
        _ = np.matmul(A, B)
        t1 = time.perf_counter()
        times_us.append((t1 - t0) * 1e6)

    return {
        "trials": [round(t, 3) for t in times_us],
        "avg_us": round(sum(times_us) / len(times_us), 3),
        "min_us": round(min(times_us), 3),
    }


def main():
    with open(SHAPES_FILE) as f:
        data = json.load(f)
    shapes = data["shapes"]

    # Load existing results so the script is re-runnable / resumable
    if os.path.exists(OUT_FILE):
        with open(OUT_FILE) as f:
            results = json.load(f)
        print(f"[RESUME] Loaded {len(results)} existing entries from {OUT_FILE}")
    else:
        results = {}

    total = len(shapes)
    skipped = 0

    for i, shape in enumerate(shapes):
        M, K, N = shape["M"], shape["K"], shape["N"]
        layers  = shape.get("layers", [])
        key     = f"{M}_{N}_{K}"

        if key in results:
            skipped += 1
            continue

        results[key] = {}

        for dtype_name, dtype in DTYPE_MAP.items():
            timing = run_shape(M, K, N, dtype, TRIALS)
            timing["layers"] = layers
            results[key][dtype_name] = timing

            print(
                f"[{i+1:>3}/{total}]  {key:<18}  {dtype_name}  "
                f"avg={timing['avg_us']:>8.1f}µs  "
                f"min={timing['min_us']:>8.1f}µs  "
                f"trials={timing['trials']}"
            )

        # Save after every shape so progress is never lost
        with open(OUT_FILE, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\nDone. {total - skipped} shapes run, {skipped} skipped (already in file).")
    print(f"Results saved to: {OUT_FILE}")


if __name__ == "__main__":
    main()
