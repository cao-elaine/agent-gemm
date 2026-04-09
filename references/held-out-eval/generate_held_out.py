"""
generate_held_out.py

Creates the held-out evaluation dataset for testing the agent's tiling inference ability.

Outputs (all in the same directory as this script):
  held_out_shapes.json          -- 50 held-out padded shapes + their full profiling data
  npu_execution_profiling_reduced.json -- Full profiling DB minus the 50 held-out keys

Usage:
  python generate_held_out.py [--seed SEED] [--n N]

The full profiling DB is NOT modified. The reduced DB is a separate file the agent sees.
"""

import argparse
import json
import os
import random

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(SCRIPT_DIR, "../..")

PROFILING_FILE = os.path.join(BASE, "references/memory/gemm-data/npu_execution_profiling.json")
SMOLVLA_FILE   = os.path.join(BASE, "references/smol-vla-dataset/smolvla_gemm_shapes.json")
OUT_HELD_OUT   = os.path.join(SCRIPT_DIR, "held_out_shapes.json")
OUT_REDUCED    = os.path.join(SCRIPT_DIR, "npu_execution_profiling_reduced.json")

ALLOWED_M = [32, 64, 96, 128, 160, 192, 256, 288, 320, 384, 448, 480, 512, 576,
             640, 672, 960, 1024, 1152, 1280, 1344, 1536, 1600, 1728, 1792, 1920,
             2048, 2112, 2240, 2304, 2496, 2560, 2688, 2816, 2880, 3072, 3200,
             3264, 3328, 3456, 3584, 3648, 3840, 4032, 4096]
ALLOWED_N = [32, 64, 96, 128, 160, 192, 256, 288, 320, 384, 448, 480, 512, 576,
             640, 672, 768, 960, 1024, 1152, 1280, 1344, 1536, 1600, 1728, 1792,
             1920, 2048, 2112, 2240, 2304, 2496, 2560, 2688, 2816, 2880, 3072,
             3200, 3264, 3328, 3456, 3584, 3648, 3840, 4032, 4096]
ALLOWED_K = [32, 64, 128, 256, 512, 768, 960, 1024, 2048, 3072]

DTYPE = "bf16"


def pad_dim(v, allowed):
    return min(x for x in allowed if x >= v)


def get_best_tile(prof_dtype_entry):
    """Extract ([m,n,k], avg_us) from a profiling dtype entry, handling schema variants."""
    bt = prof_dtype_entry.get("Best m,n,k")
    if not isinstance(bt, dict) or not bt:
        return None, None
    if "size" in bt:
        return bt["size"], bt.get("Best NPU average time")
    if "tile" in bt:
        return bt["tile"], bt.get("avg")
    return None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n",    type=int, default=50)
    args = parser.parse_args()

    print(f"[generate_held_out] seed={args.seed}  n={args.n}")

    with open(PROFILING_FILE) as f:
        profiling = json.load(f)
    with open(SMOLVLA_FILE) as f:
        smolvla = json.load(f)

    shapes = smolvla["shapes"]

    # Map each SmolVLA raw shape to its padded key
    padded_to_raw = {}
    for s in shapes:
        M, K, N = s["M"], s["K"], s["N"]
        Mp = pad_dim(M, ALLOWED_M)
        Np = pad_dim(N, ALLOWED_N)
        Kp = pad_dim(K, ALLOWED_K)
        key = f"{Mp}_{Np}_{Kp}"
        if key not in padded_to_raw:
            padded_to_raw[key] = []
        padded_to_raw[key].append({"M": M, "K": K, "N": N,
                                   "Mp": Mp, "Np": Np, "Kp": Kp,
                                   "layers": s.get("layers", [])})

    # Keep only padded keys that have a valid bf16 best tile in the profiling DB
    eligible = []
    for key in padded_to_raw:
        if key not in profiling or DTYPE not in profiling[key]:
            continue
        tile, avg = get_best_tile(profiling[key][DTYPE])
        if tile is not None and avg is not None:
            eligible.append(key)
    print(f"  Eligible padded keys (have bf16 Best m,n,k): {len(eligible)}")
    assert len(eligible) >= args.n, f"Not enough eligible keys ({len(eligible)}) to hold out {args.n}"

    # Shuffle and pick N keys to hold out (stratified by Mp for diversity)
    # Group by padded M value, then sample proportionally
    from collections import defaultdict
    by_mp = defaultdict(list)
    for key in eligible:
        mp = int(key.split("_")[0])
        by_mp[mp].append(key)

    print(f"  Distribution by Mp: {dict(sorted((k, len(v)) for k,v in by_mp.items()))}")

    # Proportional stratified sampling
    rng = random.Random(args.seed)
    total_eligible = len(eligible)
    held_out_keys = []

    mp_values = sorted(by_mp.keys())
    # Calculate how many to take from each bucket (round-robin the remainder)
    quotas = {mp: int(len(by_mp[mp]) / total_eligible * args.n) for mp in mp_values}
    remainder = args.n - sum(quotas.values())
    # Give remainder to largest buckets
    sorted_by_size = sorted(mp_values, key=lambda mp: len(by_mp[mp]), reverse=True)
    for i in range(remainder):
        quotas[sorted_by_size[i]] += 1

    print(f"  Stratified quotas by Mp: {dict(sorted(quotas.items()))}")

    for mp in mp_values:
        bucket = by_mp[mp][:]
        rng.shuffle(bucket)
        held_out_keys.extend(bucket[:quotas[mp]])

    # Shuffle final list for presentation
    rng.shuffle(held_out_keys)
    assert len(held_out_keys) == args.n

    # Build held-out shapes file
    held_out = {
        "description": (
            f"50 held-out SmolVLA padded shapes removed from the profiling DB for agent evaluation. "
            f"Seed={args.seed}. The agent should NOT have access to this file."
        ),
        "total": len(held_out_keys),
        "dtype": DTYPE,
        "shapes": []
    }

    for key in held_out_keys:
        parts = key.split("_")
        Mp, Np, Kp = int(parts[0]), int(parts[1]), int(parts[2])
        prof_entry = profiling[key][DTYPE]
        best_tile, best_avg = get_best_tile(prof_entry)

        # Collect all working tiles, sorted by avg time
        working = []
        for tile_key, tile_data in prof_entry.get("Working m,n,k", {}).items():
            working.append({
                "tile": tile_data["size"],
                "avg_us": tile_data["avg"],
                "count": tile_data["count"]
            })
        working.sort(key=lambda x: x["avg_us"])

        held_out["shapes"].append({
            "padded_key": key,
            "Mp": Mp, "Np": Np, "Kp": Kp,
            "raw_shapes": padded_to_raw[key],
            "actual_best_tile": best_tile,
            "actual_best_avg_us": best_avg,
            "actual_working_tiles": working,
            "actual_failing_tiles": prof_entry.get("Failing m,n,k", [])
        })

    with open(OUT_HELD_OUT, "w") as f:
        json.dump(held_out, f, indent=2)
    print(f"  Wrote {OUT_HELD_OUT} ({len(held_out['shapes'])} shapes)")

    # Build reduced profiling DB
    held_out_set = set(held_out_keys)
    reduced = {k: v for k, v in profiling.items() if k not in held_out_set}
    print(f"  Reduced DB: {len(reduced)} keys (removed {len(profiling) - len(reduced)})")

    with open(OUT_REDUCED, "w") as f:
        json.dump(reduced, f, indent=2)
    print(f"  Wrote {OUT_REDUCED}")

    # Print summary of held-out shapes
    print(f"\n  Held-out shape summary:")
    print(f"  {'Key':<20} {'Best tile':<20} {'Best avg µs':>12}")
    print(f"  {'-'*20} {'-'*20} {'-'*12}")
    for entry in sorted(held_out["shapes"], key=lambda x: x["padded_key"]):
        tile_str = str(entry["actual_best_tile"])
        print(f"  {entry['padded_key']:<20} {tile_str:<20} {entry['actual_best_avg_us']:>12.1f}")


if __name__ == "__main__":
    main()
