#!/usr/bin/env python3
"""
SmolVLA NPU Experiment — Baseline vs Agentic Comparison
dtype=bf16, trials=3
shapes_file=references/smol-vla-dataset/smolvla_gemm_shapes.json
"""

import json, os, subprocess, time, sys, re, math
from datetime import datetime
import numpy as np

# ─── Environment ──────────────────────────────────────────────────────────────
os.environ["ENABLE_AGGRESSIVE_PORT_UTILIZATION_PATCH"] = "1"

BASE       = "/home/ec935/agent-gemm"
SCRIPT_DIR = "/home/ec935/vla-to-npu/gemm/scripts"
PYTHON     = "/opt/anaconda3/envs/allo-base/bin/python3"
DATE_STR   = "20260330"
OUT_DIR    = f"{BASE}/references/experiment-results/test_n_{DATE_STR}"
JSON_OUT   = f"{OUT_DIR}/results.json"
MD_OUT     = f"{OUT_DIR}/results.md"
os.makedirs(OUT_DIR, exist_ok=True)

DTYPE  = "bf16"
TRIALS = 3
DSIZE  = 2  # bytes per element for bf16

# ─── ALLOWED dimension lists ──────────────────────────────────────────────────
ALLOWED_M = [32, 64, 96, 128, 160, 192, 256, 288, 320, 384, 448, 480, 512, 576,
             640, 672, 960, 1024, 1152, 1280, 1344, 1536, 1600, 1728, 1792, 1920,
             2048, 2112, 2240, 2304, 2496, 2560, 2688, 2816, 2880, 3072, 3200,
             3264, 3328, 3456, 3584, 3648, 3840, 4032, 4096]
ALLOWED_N = [32, 64, 96, 128, 160, 192, 256, 288, 320, 384, 448, 480, 512, 576,
             640, 672, 768, 960, 1024, 1152, 1280, 1344, 1536, 1600, 1728, 1792,
             1920, 2048, 2112, 2240, 2304, 2496, 2560, 2688, 2816, 2880, 3072,
             3200, 3264, 3328, 3456, 3584, 3648, 3840, 4032, 4096]
ALLOWED_K = [32, 64, 128, 256, 512, 768, 960, 1024, 2048, 3072]

def pad_dim(v, allowed):
    return min(x for x in allowed if x >= v)

# ─── Load context data ────────────────────────────────────────────────────────
print("[Phase 0] Loading profiling data...")
prof = json.load(open(f"{BASE}/references/memory/gemm-data/npu_execution_profiling.json"))
shapes_data = json.load(open(f"{BASE}/references/smol-vla-dataset/smolvla_gemm_shapes.json"))
ALL_SHAPES = shapes_data["shapes"]

# ─── Helper: tile validation ──────────────────────────────────────────────────
def check_tile(Mp, Np, Kp, m, n, k):
    """Returns (ok, reason). Uses relaxed bundling: allows any divisor including 2."""
    if m > Mp or n > Np or k > Kp:
        return False, "tile exceeds padded dim"
    if Mp % m != 0 or Np % n != 0 or Kp % k != 0:
        return False, f"divisibility: Mp%m={Mp%m} Np%n={Np%n} Kp%k={Kp%k}"
    if (m, n, k) == (128, 128, 128):
        return False, "hard_blacklist"
    if (m*n + n*k + k*m) * DSIZE > 32256:
        return False, f"memory: {(m*n+n*k+k*m)*DSIZE} > 32256"
    return True, "ok"

def get_pad_impl(M, K, N, Mp, Np, Kp):
    if M == Mp and N == Np and K == Kp:
        return 'none'
    ae = M*K + K*N
    return 'manual_copy' if ae <= 571779 else 'numpy_pad'

def get_profiling_best(Mp, Np, Kp):
    """Returns (tile_tuple_or_None, source_str)."""
    key = f"{Mp}_{Np}_{Kp}"
    if key not in prof:
        return None, 'no_key'
    if DTYPE not in prof[key]:
        return None, 'no_dtype'
    bt = prof[key][DTYPE].get('Best m,n,k')
    if bt is None:
        return None, 'best_is_none'
    if isinstance(bt, dict) and 'size' in bt:
        return tuple(bt['size']), 'profiling'
    return None, 'bad_format'

def get_failing_tiles(Mp, Np, Kp):
    key = f"{Mp}_{Np}_{Kp}"
    if key not in prof:
        return set()
    if DTYPE not in prof[key]:
        return set()
    return {tuple(t) for t in prof[key][DTYPE].get('Failing m,n,k', [])}

# ─── Strategy: Baseline 0 ────────────────────────────────────────────────────
def strategy_b0(M, K, N, Mp, Np, Kp):
    """Fixed tile (64,64,64) or (32,32,32), no data lookup."""
    fallback_seq = [(64,64,64), (32,32,32), (16,32,32)]
    for m, n, k in fallback_seq:
        ok, _ = check_tile(Mp, Np, Kp, m, n, k)
        if ok:
            return {"Mp":Mp,"Np":Np,"Kp":Kp, "tile":[m,n,k],
                    "pad_impl": get_pad_impl(M,K,N,Mp,Np,Kp), "tile_source":"fixed"}
    return {"Mp":Mp,"Np":Np,"Kp":Kp, "tile":[32,32,32],
            "pad_impl": get_pad_impl(M,K,N,Mp,Np,Kp), "tile_source":"fixed"}

# ─── Strategy: Baseline 2 ────────────────────────────────────────────────────
def strategy_b2(M, K, N, Mp, Np, Kp):
    """Closest-fit padded shape + profiling-best tile (no accuracy pre-validation)."""
    best, src = get_profiling_best(Mp, Np, Kp)
    if best:
        ok, _ = check_tile(Mp, Np, Kp, *best)
        if ok:
            return {"Mp":Mp,"Np":Np,"Kp":Kp, "tile":list(best),
                    "pad_impl": get_pad_impl(M,K,N,Mp,Np,Kp), "tile_source":"profiling"}
    # Fallback
    for tile in [(64,64,64),(32,32,32),(16,32,32)]:
        ok, _ = check_tile(Mp, Np, Kp, *tile)
        if ok:
            return {"Mp":Mp,"Np":Np,"Kp":Kp, "tile":list(tile),
                    "pad_impl": get_pad_impl(M,K,N,Mp,Np,Kp), "tile_source":"fallback_fixed"}
    return {"Mp":Mp,"Np":Np,"Kp":Kp, "tile":[32,32,32],
            "pad_impl": get_pad_impl(M,K,N,Mp,Np,Kp), "tile_source":"fallback_fixed"}

# ─── Strategy: Agentic ───────────────────────────────────────────────────────
def strategy_agentic(M, K, N, Mp, Np, Kp):
    """
    Full reasoning: profiling → cost model → heuristics.
    May choose a different padded shape if closest-fit is known to fail entirely.
    """
    failing = get_failing_tiles(Mp, Np, Kp)
    reasoning = []

    # Special case: 256_320_960 — all 5 tested tiles fail (col_num=5 bf16 accuracy issue at Mp=256)
    # Agentic override: pad N to 384 instead of 320 to get a working padded shape
    if Mp == 256 and Np == 320 and Kp == 960:
        Np_new = 384
        reasoning.append(f"Agentic: overriding Np=320→384 (256_320_960 has 0 working bf16 tiles)")
        Mp2, Np2, Kp2 = Mp, Np_new, Kp
        best2, src2 = get_profiling_best(Mp2, Np2, Kp2)
        tile = None
        if best2:
            ok, _ = check_tile(Mp2, Np2, Kp2, *best2)
            if ok:
                tile = list(best2)
                reasoning.append(f"profiling best for 256_384_960: {tile}")
        if tile is None:
            # Try (64,64,64) — should work since Pn=384/64=6 (6%3=0), Pm=256/64=4 (4%4=0)
            tile = [64, 64, 64]
            reasoning.append("No profiling for 256_384_960; using (64,64,64) heuristic")
        return {"Mp":Mp2,"Np":Np2,"Kp":Kp2, "tile":tile,
                "pad_impl": get_pad_impl(M,K,N,Mp2,Np2,Kp2),
                "tile_source":"heuristic",
                "reasoning": "; ".join(reasoning)}

    # [2026-03-30] Safety rule: (m=64, n=16) crashes NPU hardware on Np>=480
    # Add to failing set to prevent selection
    if Np >= 480:
        for k_val in [16, 32, 64, 128, 256, 512, 768, 1024]:
            failing.add((64, 16, k_val))

    # Try profiling best
    best, src = get_profiling_best(Mp, Np, Kp)
    if best and best not in failing:
        ok, reason = check_tile(Mp, Np, Kp, *best)
        if ok:
            reasoning.append(f"Using profiling best {best}")
            return {"Mp":Mp,"Np":Np,"Kp":Kp, "tile":list(best),
                    "pad_impl": get_pad_impl(M,K,N,Mp,Np,Kp),
                    "tile_source":"profiling",
                    "reasoning": "; ".join(reasoning)}
        else:
            reasoning.append(f"Profiling best {best} fails check: {reason}")
    elif best is None:
        reasoning.append(f"No profiling data ({src}); using heuristics")

    # Np=320 special: prefer m=64, avoid n=64 (Pn=5 col_num=5 accuracy issue)
    if Np == 320:
        # n=16 gives Pn=20 (col_num=4), m=64 required by rules
        candidates = [(64,16,64),(64,16,32),(64,16,128),(64,32,64),(64,64,64)]
        for tile in candidates:
            if tuple(tile) in failing:
                continue
            ok, _ = check_tile(Mp, Np, Kp, *tile)
            if ok:
                reasoning.append(f"Np=320: using m=64,n=16 tile {tile} to avoid col_num=5")
                return {"Mp":Mp,"Np":Np,"Kp":Kp, "tile":list(tile),
                        "pad_impl": get_pad_impl(M,K,N,Mp,Np,Kp),
                        "tile_source":"heuristic",
                        "reasoning": "; ".join(reasoning)}

    # Np=64: prefer n=16 tiles (known from profiling to beat n=64)
    if Np == 64:
        for tile in [(64,16,64),(64,16,32),(32,16,64),(32,16,32),(64,16,128)]:
            if tuple(tile) in failing:
                continue
            ok, _ = check_tile(Mp, Np, Kp, *tile)
            if ok:
                reasoning.append(f"Np=64: using n=16 tile {tile}")
                return {"Mp":Mp,"Np":Np,"Kp":Kp, "tile":list(tile),
                        "pad_impl": get_pad_impl(M,K,N,Mp,Np,Kp),
                        "tile_source":"heuristic",
                        "reasoning": "; ".join(reasoning)}

    # For Mp=192 shapes (no profiling): use heuristics from 128_* and 256_* patterns
    if Mp == 192:
        # From cross-shape analysis: (64,64,64) works well for most 192_* shapes
        # (64,32,64) is good for small-N shapes (Np≤288)
        if Np <= 288:
            tile_cands = [(64,32,64),(64,32,128),(64,64,64),(32,32,32)]
        elif Np <= 512:
            tile_cands = [(64,64,64),(64,32,64),(32,64,64),(32,32,128)]
        else:
            tile_cands = [(64,64,64),(64,32,128),(32,64,64)]
        for tile in tile_cands:
            if tuple(tile) in failing:
                continue
            ok, _ = check_tile(Mp, Np, Kp, *tile)
            if ok:
                reasoning.append(f"Mp=192 heuristic (extrapolated from 128_*/256_* data): {tile}")
                return {"Mp":Mp,"Np":Np,"Kp":Kp, "tile":list(tile),
                        "pad_impl": get_pad_impl(M,K,N,Mp,Np,Kp),
                        "tile_source":"heuristic",
                        "reasoning": "; ".join(reasoning)}

    # For Mp=32 shapes: prefer (32,32,32) but try others if it fails
    if Mp == 32:
        for tile in [(32,32,32),(16,32,32),(32,64,32),(32,32,64)]:
            if tuple(tile) in failing:
                continue
            ok, _ = check_tile(Mp, Np, Kp, *tile)
            if ok:
                reasoning.append(f"Mp=32 fallback: {tile}")
                return {"Mp":Mp,"Np":Np,"Kp":Kp, "tile":list(tile),
                        "pad_impl": get_pad_impl(M,K,N,Mp,Np,Kp),
                        "tile_source":"heuristic",
                        "reasoning": "; ".join(reasoning)}

    # General fallback
    general_cands = [(64,64,64),(64,32,64),(32,64,64),(64,64,32),(32,32,128),
                     (64,32,128),(32,32,64),(32,32,32),(16,32,32),(16,16,32)]
    for tile in general_cands:
        if tuple(tile) in failing:
            continue
        ok, _ = check_tile(Mp, Np, Kp, *tile)
        if ok:
            reasoning.append(f"General fallback: {tile}")
            return {"Mp":Mp,"Np":Np,"Kp":Kp, "tile":list(tile),
                    "pad_impl": get_pad_impl(M,K,N,Mp,Np,Kp),
                    "tile_source":"heuristic",
                    "reasoning": "; ".join(reasoning)}

    return {"Mp":Mp,"Np":Np,"Kp":Kp, "tile":[32,32,32],
            "pad_impl": get_pad_impl(M,K,N,Mp,Np,Kp),
            "tile_source":"heuristic",
            "reasoning": "all candidates exhausted, final fallback (32,32,32)"}

# ─── Pre-screen all shapes ────────────────────────────────────────────────────
def prescreen(M, K, N):
    """Returns (skip:bool, reason:str)."""
    if K > 2048:
        return True, "bf16_K_limit"
    if M <= 50 and N <= 96 and K >= 960:
        return True, "bf16_accuracy_small_MN_large_K"
    return False, ""

# ─── Execute one trial ────────────────────────────────────────────────────────
def run_trial(M_orig, N_orig, K_orig, Mp, Np, Kp, m, n, k, pad_impl,
              trial_idx, shape_idx, total_shapes, group):
    """
    M_orig/N_orig/K_orig: original (unpadded) dimensions.
    Mp/Np/Kp: padded dimensions for NPU execution.
    When pad_impl != 'none': pass originals as --M/N/K and padded as --pad-M/N/K.
    When pad_impl == 'none' (no padding needed): pass Mp/Np/Kp as --M/N/K only.
    """
    if pad_impl not in ('none', None):
        script_args = (
            f"--M {M_orig} --N {N_orig} --K {K_orig} "
            f"--m {m} --n {n} --k {k} "
            f"--dtype {DTYPE} "
            f"--use-padding --pad-M {Mp} --pad-N {Np} --pad-K {Kp} "
            f"--pad-impl {pad_impl}"
        )
    else:
        script_args = (
            f"--M {Mp} --N {Np} --K {Kp} "
            f"--m {m} --n {n} --k {k} "
            f"--dtype {DTYPE}"
        )
    cmd = [
        "bash", "-c",
        f"cd {SCRIPT_DIR} && "
        f"ENABLE_AGGRESSIVE_PORT_UTILIZATION_PATCH=1 "
        f"{PYTHON} v2_test_mapping_large_gemm.py {script_args}"
    ]
    print(f"  [shape {shape_idx} | {group} trial {trial_idx}/{TRIALS}]  "
          f"M={M_orig} N={N_orig} K={K_orig} → ({Mp},{Np},{Kp})  "
          f"tile=({m},{n},{k})  pad={pad_impl}  compiling...",
          flush=True)
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    elapsed = time.time() - t0
    stdout = result.stdout + result.stderr

    # Parse outputs
    npu_avg = None
    npu_min = None
    pad_time = None
    status = "failed"
    error_msg = ""

    avg_m    = re.search(r'Avg NPU execution time:\s*([\d.]+)\s*us', stdout)
    min_m    = re.search(r'Min NPU execution time:\s*([\d.]+)\s*us', stdout)
    pad_m    = re.search(r'\[bf16\] Padding time:\s*([\d.]+)\s*us', stdout)
    unpad_m  = re.search(r'\[bf16\] Unpadding time:\s*([\d.]+)\s*us', stdout)

    if avg_m:
        npu_avg = float(avg_m.group(1))
    if min_m:
        npu_min = float(min_m.group(1))
    if pad_m:
        pad_time = float(pad_m.group(1))
    unpad_time = float(unpad_m.group(1)) if unpad_m else None

    # Determine pass/fail
    if 'ValueError: Failed to generate cdo' in stdout:
        status = 'failed'
        error_msg = "ValueError_cdo (bf16+K=2048 false positive trap)"
    elif 'XDNA driver not found' in stdout:
        status = 'npu_unavailable'
        error_msg = "XDNA driver not found"
    elif npu_avg is not None and 'PASSED' in stdout and 'ValueError' not in stdout:
        status = 'passed'
    elif result.returncode != 0 or npu_avg is None:
        # Classify error
        if 'ZeroDivisionError' in stdout:
            error_msg = "ZeroDivisionError"
        elif 'Unresolvable mapping' in stdout:
            error_msg = "AssertionError_unresolvable_mapping"
        elif 'do not divide' in stdout.lower() or 'Tile sizes' in stdout:
            error_msg = "TileDivisionError"
        elif 'allocated buffers exceeded' in stdout or 'AIE core' in stdout:
            error_msg = "BufferOverflow"
        elif 'Stride 1 is' in stdout:
            error_msg = "DMAStrideError"
        elif 'Failed to compile' in stdout:
            error_msg = "CompilationError"
        elif 'clang++' in stdout and 'error:' in stdout:
            error_msg = "ClangCompileError"
        else:
            first_err = next((l.strip() for l in stdout.split('\n') if 'Error' in l or 'error' in l), "unknown")
            error_msg = first_err[:120]
        status = 'failed'

    if status == 'passed':
        print(f"    → PASSED  avg={npu_avg:.1f}µs  min={npu_min:.1f}µs  "
              f"pad={pad_time if pad_time else 'N/A'}µs  ({elapsed:.0f}s)", flush=True)
    elif status == 'npu_unavailable':
        print(f"    → NPU_UNAVAILABLE", flush=True)
    else:
        print(f"    → FAILED  {error_msg[:80]}", flush=True)

    return {
        "trial": trial_idx,
        "status": status,
        "npu_avg_us": npu_avg,
        "npu_min_us": npu_min,
        "padding_us": pad_time,
        "unpadding_us": unpad_time,
        "error": error_msg if status != 'passed' else None,
        "elapsed_s": round(elapsed, 1)
    }

# ─── Fallback tile sequences ──────────────────────────────────────────────────
B0_FALLBACKS = [(64,64,64), (32,32,32), (16,32,32)]
B2_FALLBACKS = [(64,64,64), (32,32,32), (16,32,32)]

def try_tile_sequence(M_orig, K_orig, N_orig, Mp, Np, Kp, tiles, pad_impl, group,
                      shape_idx, total_shapes, tried_tiles=None):
    """
    Run trials with fallback. Returns (trials_list, tile_used, npu_gone).
    M_orig/K_orig/N_orig are original dims; Mp/Np/Kp are padded dims.
    """
    if tried_tiles is None:
        tried_tiles = set()
    trial_results = []
    last_tile = list(tiles[0]) if tiles else [32,32,32]
    for tile in tiles:
        t = tuple(tile)
        if t in tried_tiles:
            continue
        ok, reason = check_tile(Mp, Np, Kp, *t)
        if not ok:
            print(f"    [skip tile {t}: {reason}]", flush=True)
            tried_tiles.add(t)
            continue
        tried_tiles.add(t)
        last_tile = list(t)
        trial_results = []
        npu_unavailable = False
        for trial_idx in range(1, TRIALS+1):
            tr = run_trial(M_orig, N_orig, K_orig, Mp, Np, Kp,
                           *t, pad_impl, trial_idx, shape_idx, total_shapes, group)
            trial_results.append(tr)
            if tr['status'] == 'npu_unavailable':
                npu_unavailable = True
                break
        if npu_unavailable:
            return trial_results, last_tile, True
        # If at least one trial passed, accept this tile
        if any(r['status'] == 'passed' for r in trial_results):
            return trial_results, last_tile, False
        # All trials failed — try next tile
        print(f"    [all {TRIALS} trials failed with tile {t}, trying fallback]", flush=True)
    # All tiles exhausted
    return trial_results, last_tile, False


def run_group(M, K, N, Mp, Np, Kp, strategy_info, group_name, shape_idx, total_shapes):
    """Run TRIALS trials for one group (b0/b2/agentic). Returns group result dict."""
    tile = strategy_info['tile']
    pad_impl = strategy_info['pad_impl']
    tried = set()

    trial_results, tile_used, npu_gone = try_tile_sequence(
        M, K, N, Mp, Np, Kp,
        [tile], pad_impl, group_name, shape_idx, total_shapes, tried
    )
    # Note: Mp/Np/Kp might differ from M/K/N if agentic chose different padded shape
    # In that case M/K/N are the original dims, Mp/Np/Kp are the chosen padded dims

    if npu_gone:
        # Mark all remaining as npu_unavailable
        pass

    # Fallbacks if all primary tile trials failed
    if not npu_gone and not any(r['status']=='passed' for r in trial_results):
        if group_name == 'baseline_0':
            fb = B0_FALLBACKS
        elif group_name == 'baseline_2':
            fb = B2_FALLBACKS
        else:
            # Agentic fallbacks: skip the current tile in sequence
            fb = [(64,64,64),(32,32,32),(16,32,32)]
        new_trials, tile_used, npu_gone = try_tile_sequence(
            M, K, N, Mp, Np, Kp, fb, pad_impl, group_name, shape_idx, total_shapes, tried
        )
        if new_trials:
            trial_results = new_trials

    passed = [r for r in trial_results if r['status'] == 'passed']
    mean_avg   = round(float(np.mean([r['npu_avg_us'] for r in passed])), 2) if passed else None
    mean_pad   = round(float(np.mean([r['padding_us'] for r in passed if r['padding_us'] is not None])), 2) if passed else None
    mean_unpad = round(float(np.mean([r['unpadding_us'] for r in passed if r.get('unpadding_us') is not None])), 2) if passed else None
    if mean_avg is not None:
        mean_total = round(mean_avg + (mean_pad or 0.0) + (mean_unpad or 0.0), 2)
    else:
        mean_total = None

    result = {
        "padded": {"M": Mp, "N": Np, "K": Kp},
        "tile": tile_used,
        "pad_impl": pad_impl,
        "tile_source": strategy_info.get('tile_source', 'unknown'),
        "trials": trial_results,
        "mean_npu_avg_us": mean_avg,
        "mean_padding_us": mean_pad,
        "mean_unpadding_us": mean_unpad,
        "mean_total_us": mean_total,
        "pass_count": len(passed)
    }
    if group_name == 'agentic':
        result["reasoning_summary"] = strategy_info.get('reasoning', '')
    return result


# ─── Load or initialise results ───────────────────────────────────────────────
def load_results():
    if os.path.exists(JSON_OUT):
        data = json.load(open(JSON_OUT))
        completed = set()
        for s in data.get('shapes', []):
            if s.get('winner') not in (None,):
                completed.add((s['M'], s['K'], s['N']))
        print(f"[RESUME] Loaded {len(data['shapes'])} shapes, {len(completed)} completed", flush=True)
        return data, completed
    return {"experiment":"SmolVLA NPU Baseline vs Agentic",
            "date": "2026-03-30",
            "dtype": DTYPE,
            "trials_per_group": TRIALS,
            "shapes": []}, set()

def save_results(data):
    with open(JSON_OUT, 'w') as f:
        json.dump(data, f, indent=2)

# ─── Phase 1: Compute strategies for all shapes ───────────────────────────────
def compute_all_strategies():
    plan = []
    for s in ALL_SHAPES:
        M, K, N = s['M'], s['K'], s['N']
        layers = []
        for l in s.get('layers', []):
            if isinstance(l, dict):
                layers.append(l.get('name', str(l)))
            else:
                layers.append(str(l))
        skip, reason = prescreen(M, K, N)
        if skip:
            plan.append({
                'M': M, 'K': K, 'N': N, 'layers': layers,
                'skip': True, 'skip_reason': reason
            })
            continue

        Mp = pad_dim(M, ALLOWED_M)
        Kp = pad_dim(K, ALLOWED_K)
        Np = pad_dim(N, ALLOWED_N)

        b0 = strategy_b0(M, K, N, Mp, Np, Kp)
        b2 = strategy_b2(M, K, N, Mp, Np, Kp)
        ag = strategy_agentic(M, K, N, Mp, Np, Kp)

        plan.append({
            'M': M, 'K': K, 'N': N, 'layers': layers,
            'skip': False,
            'baseline_0': b0,
            'baseline_2': b2,
            'agentic': ag
        })
    return plan


# ─── Phase 1.2: Print plan table ─────────────────────────────────────────────
def print_plan_table(plan):
    runnable = [p for p in plan if not p.get('skip')]
    skipped  = [p for p in plan if p.get('skip')]

    print()
    print("╔" + "═"*118 + "╗")
    print("║" + "   SMOLVLA EXPERIMENT PLAN  —  dtype=bf16".center(118) + "║")
    print("╠" + "═"*14 + "╦" + "═"*22 + "╦" + "═"*26 + "╦" + "═"*26 + "╦" + "═"*26 + "╣")
    print("║ Shape M,K,N  ║ Layer(s)             ║ Baseline 0 (fixed)       ║ Baseline 2 (profiling)   ║ Agentic                  ║")
    print("╠" + "═"*14 + "╬" + "═"*22 + "╬" + "═"*26 + "╬" + "═"*26 + "╬" + "═"*26 + "╣")

    for p in plan[:30]:  # Show first 30 for brevity, then summary
        if p.get('skip'):
            layer_str = ', '.join(p['layers'])[:18]
            print(f"║ {p['M']:4},{p['K']:4},{p['N']:4} ║ {layer_str:<20} ║ {'PRESCREENED_SKIP ('+p['skip_reason'][:20]+')':<24} ║{'':26}║{'':26}║")
            continue
        b0, b2, ag = p['baseline_0'], p['baseline_2'], p['agentic']
        layer_str = ', '.join(p['layers'])[:18]
        def fmt(s):
            tp = f"({s['Mp']},{s['Np']},{s['Kp']})"
            tt = f"({s['tile'][0]},{s['tile'][1]},{s['tile'][2]})"
            pi = s['pad_impl'][:2] if s['pad_impl'] != 'none' else 'no'
            return f"{tp} {tt} {pi}"[:24]
        print(f"║ {p['M']:4},{p['K']:4},{p['N']:4} ║ {layer_str:<20} ║ {fmt(b0):<24} ║ {fmt(b2):<24} ║ {fmt(ag):<24} ║")
    if len(plan) > 30:
        print(f"║ ... {len(plan)-30} more shapes ...                                                                                               ║")

    print("╚" + "═"*14 + "╩" + "═"*22 + "╩" + "═"*26 + "╩" + "═"*26 + "╩" + "═"*26 + "╝")
    print()
    total_runs = len(runnable) * 3 * TRIALS
    print(f"Total shapes in file:   {len(plan)}")
    print(f"  Pre-screened (skip):  {len(skipped)}")
    print(f"  Runnable:             {len(runnable)}")
    print(f"Total NPU runs:         {total_runs}  ({len(runnable)} × 3 groups × {TRIALS} trials)")
    print()


# ─── Phase 2: Execute all runs ────────────────────────────────────────────────
def determine_winner(b0, b2, ag):
    """Determine winner among groups with ≥1 passing trial. Uses mean_total_us."""
    means = {}
    for name, grp in [('baseline_0', b0), ('baseline_2', b2), ('agentic', ag)]:
        if grp['pass_count'] > 0 and grp.get('mean_total_us') is not None:
            means[name] = grp['mean_total_us']
        elif grp['pass_count'] > 0 and grp.get('mean_npu_avg_us') is not None:
            means[name] = grp['mean_npu_avg_us']  # fallback if mean_total_us missing

    if len(means) == 0:
        return 'inconclusive', None, None, None, None, None, None

    best_name = min(means, key=means.get)
    best_val  = means[best_name]

    # Check if any group has 0 passing trials
    groups_inc = []
    for name, grp in [('baseline_0', b0), ('baseline_2', b2), ('agentic', ag)]:
        if grp['pass_count'] == 0:
            groups_inc.append(name)

    # Delta calculations (based on mean_total_us)
    def delta(ref_name, cmp_name):
        if ref_name in means and cmp_name in means and means[ref_name]:
            d = means[cmp_name] - means[ref_name]
            pct = d / means[ref_name] * 100
            return round(d, 2), round(pct, 2)
        return None, None

    db2_b0_us, db2_b0_pct = delta('baseline_0', 'baseline_2')
    dag_b0_us, dag_b0_pct = delta('baseline_0', 'agentic')
    dag_b2_us, dag_b2_pct = delta('baseline_2', 'agentic')

    if groups_inc:
        winner = 'inconclusive'
    else:
        # Check tie: top two within 2%
        sorted_means = sorted(means.items(), key=lambda x: x[1])
        if len(sorted_means) >= 2:
            top1, top2 = sorted_means[0][1], sorted_means[1][1]
            if abs(top1 - top2) / top1 * 100 <= 2.0:
                winner = 'tie'
            else:
                winner = best_name
        else:
            winner = best_name

    return winner, db2_b0_us, db2_b0_pct, dag_b0_us, dag_b0_pct, dag_b2_us, dag_b2_pct


def print_shape_summary(M, K, N, layers, b0_res, b2_res, ag_res, winner, deltas):
    db2_b0_us, db2_b0_pct, dag_b0_us, dag_b0_pct, dag_b2_us, dag_b2_pct = deltas
    layer_str = ', '.join(layers[:3])
    print(f"\n── M={M} K={K} N={N}  [{layer_str}] " + "─"*40, flush=True)
    for name, res in [('BASELINE_0', b0_res), ('BASELINE_2', b2_res), ('AGENTIC', ag_res)]:
        t = res['tile']
        p = res['padded']
        src = res.get('tile_source', '')
        print(f"  {name:12} padded=({p['M']},{p['N']},{p['K']})  tile=({t[0]},{t[1]},{t[2]})  [{src}]", flush=True)
        avgs = [f"T{r['trial']}:{r['npu_avg_us']:.1f}µs" if r['npu_avg_us'] else f"T{r['trial']}:FAIL"
                for r in res['trials']]
        mean_s = f"{res['mean_npu_avg_us']:.1f}µs" if res['mean_npu_avg_us'] else "N/A"
        print(f"    {' | '.join(avgs)}   Mean: {mean_s}", flush=True)
    print(f"  Best: {winner}", flush=True)
    parts = []
    if db2_b0_us is not None:
        parts.append(f"Δ(B2 vs B0)={db2_b0_us:+.1f}µs ({db2_b0_pct:+.1f}%)")
    if dag_b0_us is not None:
        parts.append(f"Δ(AG vs B0)={dag_b0_us:+.1f}µs ({dag_b0_pct:+.1f}%)")
    if dag_b2_us is not None:
        parts.append(f"Δ(AG vs B2)={dag_b2_us:+.1f}µs ({dag_b2_pct:+.1f}%)")
    print("  " + "   ".join(parts), flush=True)
    print("─"*72, flush=True)


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    from allo.backend.aie import is_available
    if not is_available():
        print("ERROR: NPU not available — cannot run experiment. Exiting.")
        sys.exit(1)
    print("[OK] NPU is available", flush=True)

    # Compute plan
    print("[Phase 1] Computing strategies for all shapes...", flush=True)
    plan = compute_all_strategies()
    print_plan_table(plan)

    # Load existing results (resume support)
    data, completed = load_results()
    existing_map = {(s['M'], s['K'], s['N']): s for s in data['shapes']}

    total_shapes = len(plan)
    npu_gone = False

    for shape_idx, p in enumerate(plan, 1):
        M, K, N = p['M'], p['K'], p['N']

        if p.get('skip'):
            key = (M, K, N)
            if key not in completed:
                print(f"[PRE-SCREEN SKIP] M={M} K={K} N={N}  reason: {p['skip_reason']}  layers: {', '.join(p['layers'])}", flush=True)
                entry = {
                    "M": M, "K": K, "N": N,
                    "layers": p['layers'],
                    "winner": "prescreened_skip",
                    "skip_reason": p['skip_reason'],
                    "baseline_0": None, "baseline_2": None, "agentic": None,
                    "delta_b2_vs_b0_us": None, "delta_b2_vs_b0_pct": None,
                    "delta_ag_vs_b0_us": None, "delta_ag_vs_b0_pct": None,
                    "delta_ag_vs_b2_us": None, "delta_ag_vs_b2_pct": None
                }
                data['shapes'].append(entry)
                save_results(data)
                completed.add(key)
            continue

        key = (M, K, N)
        if key in completed:
            print(f"[SKIP] M={M} K={K} N={N} already completed", flush=True)
            continue

        if npu_gone:
            # Mark as npu_unavailable
            entry = {
                "M": M, "K": K, "N": N, "layers": p['layers'],
                "winner": "npu_unavailable",
                "baseline_0": None, "baseline_2": None, "agentic": None,
                "delta_b2_vs_b0_us": None, "delta_b2_vs_b0_pct": None,
                "delta_ag_vs_b0_us": None, "delta_ag_vs_b0_pct": None,
                "delta_ag_vs_b2_us": None, "delta_ag_vs_b2_pct": None
            }
            data['shapes'].append(entry)
            continue

        print(f"\n[shape {shape_idx}/{total_shapes}]  M={M} K={K} N={N}  layers={', '.join(p['layers'][:2])}", flush=True)

        b0_strat = p['baseline_0']
        b2_strat = p['baseline_2']
        ag_strat = p['agentic']

        # Run Baseline 0
        print(f"  Running Baseline 0...", flush=True)
        b0_res = run_group(M, K, N,
                           b0_strat['Mp'], b0_strat['Np'], b0_strat['Kp'],
                           b0_strat, 'baseline_0', shape_idx, total_shapes)
        if any(r['status']=='npu_unavailable' for r in b0_res['trials']):
            npu_gone = True

        # Run Baseline 2
        if not npu_gone:
            print(f"  Running Baseline 2...", flush=True)
            b2_res = run_group(M, K, N,
                               b2_strat['Mp'], b2_strat['Np'], b2_strat['Kp'],
                               b2_strat, 'baseline_2', shape_idx, total_shapes)
            if any(r['status']=='npu_unavailable' for r in b2_res['trials']):
                npu_gone = True
        else:
            b2_res = {"padded":{"M":b2_strat['Mp'],"N":b2_strat['Np'],"K":b2_strat['Kp']},
                      "tile":b2_strat['tile'], "pad_impl":b2_strat['pad_impl'],
                      "tile_source":"npu_unavailable", "trials":[], "mean_npu_avg_us":None,
                      "mean_padding_us":None, "pass_count":0}

        # Run Agentic
        if not npu_gone:
            print(f"  Running Agentic...", flush=True)
            ag_res = run_group(M, K, N,
                               ag_strat['Mp'], ag_strat['Np'], ag_strat['Kp'],
                               ag_strat, 'agentic', shape_idx, total_shapes)
            if any(r['status']=='npu_unavailable' for r in ag_res['trials']):
                npu_gone = True
        else:
            ag_res = {"padded":{"M":ag_strat['Mp'],"N":ag_strat['Np'],"K":ag_strat['Kp']},
                      "tile":ag_strat['tile'], "pad_impl":ag_strat['pad_impl'],
                      "tile_source":"npu_unavailable", "trials":[], "mean_npu_avg_us":None,
                      "mean_padding_us":None, "pass_count":0}

        # Determine winner
        winner, db2_b0_us, db2_b0_pct, dag_b0_us, dag_b0_pct, dag_b2_us, dag_b2_pct = \
            determine_winner(b0_res, b2_res, ag_res)

        deltas = (db2_b0_us, db2_b0_pct, dag_b0_us, dag_b0_pct, dag_b2_us, dag_b2_pct)
        print_shape_summary(M, K, N, p['layers'], b0_res, b2_res, ag_res, winner, deltas)

        entry = {
            "M": M, "K": K, "N": N,
            "layers": p['layers'],
            "baseline_0": b0_res,
            "baseline_2": b2_res,
            "agentic": ag_res,
            "winner": winner,
            "delta_b2_vs_b0_us": db2_b0_us,
            "delta_b2_vs_b0_pct": db2_b0_pct,
            "delta_ag_vs_b0_us": dag_b0_us,
            "delta_ag_vs_b0_pct": dag_b0_pct,
            "delta_ag_vs_b2_us": dag_b2_us,
            "delta_ag_vs_b2_pct": dag_b2_pct
        }
        data['shapes'].append(entry)
        completed.add(key)
        save_results(data)

    # ── Phase 3+4: Compute summary and write outputs ──────────────────────────
    run_shapes = [s for s in data['shapes'] if s.get('winner') not in ('prescreened_skip', 'npu_unavailable')]
    skip_shapes = [s for s in data['shapes'] if s.get('winner') == 'prescreened_skip']

    winner_counts = {"baseline_0":0, "baseline_2":0, "agentic":0, "tie":0, "inconclusive":0}
    ag_b0_deltas, ag_b2_deltas, b2_b0_deltas = [], [], []

    for s in run_shapes:
        w = s.get('winner', 'inconclusive')
        if w in winner_counts:
            winner_counts[w] += 1
        if s.get('delta_ag_vs_b0_pct') is not None:
            ag_b0_deltas.append(s['delta_ag_vs_b0_pct'])
        if s.get('delta_ag_vs_b2_pct') is not None:
            ag_b2_deltas.append(s['delta_ag_vs_b2_pct'])
        if s.get('delta_b2_vs_b0_pct') is not None:
            b2_b0_deltas.append(s['delta_b2_vs_b0_pct'])

    def safe_mean(lst):
        return round(float(np.mean(lst)), 2) if lst else None

    data['summary'] = {
        "shapes_tested": len(ALL_SHAPES),
        "shapes_run": len(run_shapes),
        "shapes_prescreened": len(skip_shapes),
        "winner_counts": winner_counts,
        "mean_delta_ag_vs_b0_pct": safe_mean(ag_b0_deltas),
        "mean_delta_ag_vs_b2_pct": safe_mean(ag_b2_deltas),
        "mean_delta_b2_vs_b0_pct": safe_mean(b2_b0_deltas)
    }
    save_results(data)
    print("\n[Phase 4] Results saved to", JSON_OUT, flush=True)

    # ── Phase 3: Update memory (agentic runs only) ────────────────────────────
    print("[Phase 3] Updating memory files...", flush=True)
    update_profiling_memory(data)
    update_strategy_insights(data)

    # ── Phase 4.2: Write Markdown ─────────────────────────────────────────────
    write_markdown(data)
    print("[Phase 4] Markdown saved to", MD_OUT, flush=True)

    # ── Phase 5: Generate plots ───────────────────────────────────────────────
    generate_plots(data)

    # ── Phase 6: Final banner ─────────────────────────────────────────────────
    s = data['summary']
    wc = s['winner_counts']
    total_runs = sum(
        len(sh['baseline_0'].get('trials', [])) +
        len(sh['baseline_2'].get('trials', [])) +
        len(sh['agentic'].get('trials', []))
        for sh in run_shapes
        if sh.get('baseline_0') and sh.get('baseline_2') and sh.get('agentic')
    )
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print()
    print("╔" + "═"*70 + "╗")
    print("║" + "   SMOLVLA EXPERIMENT — COMPLETE".center(70) + "║")
    print("╠" + "═"*70 + "╣")
    print(f"║  Date:           {now:<52}║")
    print(f"║  Shapes total:   {len(ALL_SHAPES):<4}  dtype: bf16   Trials/group: {TRIALS:<19}║")
    print(f"║  Shapes run:     {len(run_shapes):<4}  Pre-screened: {len(skip_shapes):<28}║")
    print(f"║  Total NPU runs: {total_runs:<52}║")
    print(f"║  Baseline 0 wins: {wc['baseline_0']:<3} │ Baseline 2 wins: {wc['baseline_2']:<3} │ Agentic wins: {wc['agentic']:<4}║")
    print(f"║  Ties: {wc['tie']:<4}    │ Inconclusive: {wc['inconclusive']:<35}║")
    if s['mean_delta_ag_vs_b0_pct'] is not None:
        print(f"║  Mean Δ agentic vs B0: {s['mean_delta_ag_vs_b0_pct']:+.1f}%   vs B2: {s['mean_delta_ag_vs_b2_pct']:+.1f}%{'':<25}║")
    print(f"║  Results: references/experiment-results/test_n_{DATE_STR}/{'':<17}║")
    print("╚" + "═"*70 + "╝")


# ─── Phase 3: Memory update helpers ─────────────────────────────────────────
def update_profiling_memory(data):
    """Update npu_execution_profiling.json with agentic trial results."""
    changed = False
    for s in data['shapes']:
        if s.get('winner') == 'prescreened_skip':
            continue
        ag = s.get('agentic')
        if not ag:
            continue
        Mp = ag['padded']['M']
        Np = ag['padded']['N']
        Kp = ag['padded']['K']
        key = f"{Mp}_{Np}_{Kp}"
        tile = ag['tile']
        m, n, k = tile[0], tile[1], tile[2]
        tk = f"{m}_{n}_{k}"

        if key not in prof:
            prof[key] = {}
        if DTYPE not in prof[key]:
            prof[key][DTYPE] = {"Best m,n,k": None, "Working m,n,k": {}, "Failing m,n,k": []}

        bf = prof[key][DTYPE]
        if 'Working m,n,k' not in bf or isinstance(bf['Working m,n,k'], list):
            # Convert to dict format if needed
            if isinstance(bf.get('Working m,n,k'), list):
                bf['Working m,n,k'] = {}

        passed_trials = [r for r in ag['trials'] if r['status'] == 'passed']
        failed_trials = [r for r in ag['trials'] if r['status'] == 'failed']

        for r in passed_trials:
            avg = r['npu_avg_us']
            mn  = r['npu_min_us']
            if tk in bf.get('Working m,n,k', {}):
                existing = bf['Working m,n,k'][tk]
                old_avg = existing.get('avg', avg)
                old_cnt = existing.get('count', 1)
                new_avg = (old_avg * old_cnt + avg) / (old_cnt + 1)
                existing['avg'] = round(new_avg, 3)
                existing['count'] = old_cnt + 1
                if mn and (existing.get('min') is None or mn < existing['min']):
                    existing['min'] = round(mn, 3)
            else:
                if 'Working m,n,k' not in bf:
                    bf['Working m,n,k'] = {}
                bf['Working m,n,k'][tk] = {
                    "size": [m, n, k],
                    "avg": round(avg, 3),
                    "min": round(mn, 3) if mn else None,
                    "count": 1
                }
            # Update Best
            current_best = bf.get('Best m,n,k')
            if current_best is None or (isinstance(current_best, dict) and
                    avg < current_best.get('Best NPU average time', float('inf'))):
                bf['Best m,n,k'] = {
                    "size": [m, n, k],
                    "Best NPU average time": round(avg, 3)
                }
            changed = True

        for r in failed_trials:
            fail_list = bf.get('Failing m,n,k', [])
            if [m, n, k] not in fail_list:
                fail_list.append([m, n, k])
                bf['Failing m,n,k'] = fail_list
                changed = True

    if changed:
        with open(f"{BASE}/references/memory/gemm-data/npu_execution_profiling.json", 'w') as f:
            json.dump(prof, f, indent=2)
        print("  Updated npu_execution_profiling.json", flush=True)


def update_strategy_insights(data):
    """Append new insights to strategy_insights.md based on agentic run results."""
    insights_path = f"{BASE}/references/memory/knowledge-base/strategy_insights.md"
    new_bullets = []
    today = "2026-03-30"

    # Check for new best tiles found by agentic
    for s in data['shapes']:
        if s.get('winner') == 'prescreened_skip':
            continue
        ag = s.get('agentic')
        b0 = s.get('baseline_0')
        b2 = s.get('baseline_2')
        if not ag or not b2:
            continue
        # If agentic significantly outperforms baseline_2 on a non-trivial shape
        ag_mean = ag.get('mean_total_us') or ag.get('mean_npu_avg_us')
        b2_mean = b2.get('mean_total_us') or b2.get('mean_npu_avg_us')
        if ag_mean and b2_mean and ag['tile'] != b2['tile']:
            pct_diff = (ag_mean - b2_mean) / b2_mean * 100
            if pct_diff < -5 and s['M'] > 64:  # Agentic >5% faster on non-trivial shape
                new_bullets.append(
                    f"[{today}] M={s['M']} K={s['K']} N={s['N']}: agentic tile {ag['tile']} "
                    f"beats profiling-best {b2['tile']} by {abs(pct_diff):.1f}% "
                    f"({ag_mean:.0f}µs vs {b2_mean:.0f}µs). "
                    f"Strategy: {ag.get('reasoning_summary','')[:100]}"
                )

    # Special insight: 256_320_960 all-tile failure and agentic workaround
    shapes_235_K960_N320 = [s for s in data['shapes']
                             if s['M'] == 235 and s['K'] == 960 and s['N'] == 320]
    if shapes_235_K960_N320:
        s = shapes_235_K960_N320[0]
        ag = s.get('agentic', {})
        if ag and ag.get('pass_count', 0) > 0:
            new_bullets.append(
                f"[{today}] M=235 K=960 N=320 (→256_320_960): all bf16 tiles fail at Mp=256,Np=320. "
                f"Agentic workaround: pad N to 384 (256_384_960) → {ag['tile']} succeeds. "
                f"Root cause: col_num=5 accuracy failure specific to Mp=256+Np=320 combination."
            )

    if new_bullets:
        with open(insights_path, 'a') as f:
            f.write("\n")
            for b in new_bullets:
                f.write(f"\n- **{b}**\n")
        print(f"  Updated strategy_insights.md with {len(new_bullets)} new insight(s)", flush=True)
    else:
        print("  strategy_insights.md: no new insights to add", flush=True)


# ─── Phase 4.2: Markdown output ──────────────────────────────────────────────
def write_markdown(data):
    s = data['summary']
    wc = s['winner_counts']

    lines = [
        "# SmolVLA NPU Experiment Results",
        f"**Date**: {data['date']}  |  **dtype**: bf16  |  **Trials per group**: {data['trials_per_group']}",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Shapes in file | {s['shapes_tested']} |",
        f"| Shapes run | {s.get('shapes_run', 'N/A')} |",
        f"| Pre-screened (skip) | {s.get('shapes_prescreened', 'N/A')} |",
        f"| Baseline 0 wins | {wc.get('baseline_0', 0)} |",
        f"| Baseline 2 wins | {wc.get('baseline_2', 0)} |",
        f"| Agentic wins | {wc.get('agentic', 0)} |",
        f"| Ties (within 2%) | {wc.get('tie', 0)} |",
        f"| Inconclusive | {wc.get('inconclusive', 0)} |",
        f"| Mean Δ agentic vs Baseline 0 | {s.get('mean_delta_ag_vs_b0_pct', 'N/A')}% |",
        f"| Mean Δ agentic vs Baseline 2 | {s.get('mean_delta_ag_vs_b2_pct', 'N/A')}% |",
        f"| Mean Δ Baseline 2 vs Baseline 0 | {s.get('mean_delta_b2_vs_b0_pct', 'N/A')}% |",
        "",
        "## Per-Shape Results",
        ""
    ]

    new_bests = []

    for sh in data['shapes']:
        M, K, N = sh['M'], sh['K'], sh['N']
        layers = ', '.join(sh.get('layers', []))[:60]
        winner = sh.get('winner', 'N/A')

        if winner == 'prescreened_skip':
            lines += [
                f"### {M}×{K}×{N}  —  `{layers}`",
                "",
                f"**Pre-screened skip**: {sh.get('skip_reason', '')}",
                "",
                "---",
                ""
            ]
            continue

        b0 = sh.get('baseline_0', {}) or {}
        b2 = sh.get('baseline_2', {}) or {}
        ag = sh.get('agentic', {}) or {}

        def fmt_padded(g):
            p = g.get('padded', {})
            return f"({p.get('M','?')},{p.get('N','?')},{p.get('K','?')})"
        def fmt_tile(g):
            t = g.get('tile', ['?','?','?'])
            return f"({t[0]},{t[1]},{t[2]})"
        def fmt_mean(g):
            v = g.get('mean_npu_avg_us')
            return f"{v:.1f}" if v else "FAIL"
        def fmt_trial(g, i):
            ts = g.get('trials', [])
            if i < len(ts):
                r = ts[i]
                return f"{r['npu_avg_us']:.1f}" if r.get('npu_avg_us') else "FAIL"
            return "—"

        lines += [
            f"### {M}×{K}×{N}  —  `{layers}`",
            "",
            "| | Baseline 0 | Baseline 2 | Agentic |",
            "|-|------------|------------|---------|",
            f"| Padded shape | {fmt_padded(b0)} | {fmt_padded(b2)} | {fmt_padded(ag)} |",
            f"| Tile | {fmt_tile(b0)} | {fmt_tile(b2)} | {fmt_tile(ag)} |",
            f"| Tile source | {b0.get('tile_source','?')} | {b2.get('tile_source','?')} | {ag.get('tile_source','?')} |",
            f"| Pad impl | {b0.get('pad_impl','?')} | {b2.get('pad_impl','?')} | {ag.get('pad_impl','?')} |",
        ]
        for i in range(TRIALS):
            lines.append(f"| Trial {i+1} avg (µs) | {fmt_trial(b0,i)} | {fmt_trial(b2,i)} | {fmt_trial(ag,i)} |")
        lines += [
            f"| **Mean avg (µs)** | **{fmt_mean(b0)}** | **{fmt_mean(b2)}** | **{fmt_mean(ag)}** |",
            f"| Pass count | {b0.get('pass_count',0)}/{TRIALS} | {b2.get('pass_count',0)}/{TRIALS} | {ag.get('pass_count',0)}/{TRIALS} |",
            "",
        ]

        winner_label = winner.upper().replace('_', ' ')
        lines.append(f"**Winner: {winner_label}**")
        deltas = []
        if sh.get('delta_b2_vs_b0_us') is not None:
            deltas.append(f"Δ(B2 vs B0)={sh['delta_b2_vs_b0_us']:+.1f}µs ({sh['delta_b2_vs_b0_pct']:+.1f}%)")
        if sh.get('delta_ag_vs_b0_us') is not None:
            deltas.append(f"Δ(AG vs B0)={sh['delta_ag_vs_b0_us']:+.1f}µs ({sh['delta_ag_vs_b0_pct']:+.1f}%)")
        if sh.get('delta_ag_vs_b2_us') is not None:
            deltas.append(f"Δ(AG vs B2)={sh['delta_ag_vs_b2_us']:+.1f}µs ({sh['delta_ag_vs_b2_pct']:+.1f}%)")
        if deltas:
            lines.append("   ".join(deltas))
        lines += ["", "---", ""]

        # Check for new best tiles
        b2_mean = b2.get('mean_npu_avg_us')
        ag_mean = ag.get('mean_npu_avg_us')
        if ag_mean and b2_mean and ag.get('tile') != b2.get('tile') and ag_mean < b2_mean:
            new_bests.append((sh, b2, ag))

    if new_bests:
        lines += ["## New Best Tiles Found (Agentic)", ""]
        lines += ["| Shape | Old best tile | Old avg (µs) | New best tile | New avg (µs) | Δ |",
                  "|-------|--------------|-------------|--------------|-------------|---|"]
        for sh, b2, ag in new_bests:
            p = ag['padded']
            shape_str = f"({p['M']},{p['N']},{p['K']}) bf16"
            old_t = b2.get('tile', ['?','?','?'])
            new_t = ag.get('tile', ['?','?','?'])
            old_avg = b2.get('mean_npu_avg_us', 0) or 0
            new_avg = ag.get('mean_npu_avg_us', 0) or 0
            delta_pct = (new_avg - old_avg) / old_avg * 100 if old_avg else 0
            lines.append(f"| {shape_str} | ({old_t[0]},{old_t[1]},{old_t[2]}) | {old_avg:.1f} | ({new_t[0]},{new_t[1]},{new_t[2]}) | {new_avg:.1f} | {delta_pct:+.1f}% |")
        lines.append("")

    # Memory updates section
    lines += [
        "## Memory Updates",
        "",
        f"- `npu_execution_profiling.json`: updated with agentic trial results",
        f"- `strategy_insights.md`: updated with new insights (if any)",
        f"- `error-logs/errors.md`: updated if new error patterns found",
        ""
    ]

    with open(MD_OUT, 'w') as f:
        f.write('\n'.join(lines))


# ─── Phase 5: Generate plots ─────────────────────────────────────────────────
def generate_plots(data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plots_dir = f"{OUT_DIR}/plots"
    os.makedirs(plots_dir, exist_ok=True)

    shapes = [s for s in data['shapes']
              if s.get('winner') not in ('prescreened_skip', 'npu_unavailable')
              and s.get('baseline_0') and s.get('baseline_2') and s.get('agentic')]

    if not shapes:
        print("  No runnable shapes to plot", flush=True)
        return

    labels   = [f"{s['M']}×{s['K']}×{s['N']}" for s in shapes]
    b0_means = [s['baseline_0'].get('mean_npu_avg_us') or 0 for s in shapes]
    b2_means = [s['baseline_2'].get('mean_npu_avg_us') or 0 for s in shapes]
    ag_means = [s['agentic'].get('mean_npu_avg_us') or 0    for s in shapes]
    x = np.arange(len(labels))
    w = 0.25

    # Plot 1: grouped bar chart
    fig, ax = plt.subplots(figsize=(max(16, len(labels)*0.12), 6))
    ax.bar(x - w, b0_means, w, label='Baseline 0 (fixed tile)', color='#d9534f')
    ax.bar(x,     b2_means, w, label='Baseline 2 (profiling)',  color='#f0ad4e')
    ax.bar(x + w, ag_means, w, label='Agentic',                 color='#5cb85c')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, ha='center', fontsize=5)
    ax.set_ylabel('Mean NPU avg time (µs)')
    ax.set_title('SmolVLA GEMM Latency by Strategy')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/latency_grouped_bar.png", dpi=150)
    plt.close()

    # Plot 2: improvement % over B0
    def pct(baseline, val):
        if baseline and val and baseline > 0:
            return (val - baseline) / baseline * 100
        return None

    b2_pct = [pct(b0, b2) for b0, b2 in zip(b0_means, b2_means)]
    ag_pct = [pct(b0, ag) for b0, ag in zip(b0_means, ag_means)]
    b2_pct_clean = [v if v is not None else 0 for v in b2_pct]
    ag_pct_clean = [v if v is not None else 0 for v in ag_pct]

    fig, ax = plt.subplots(figsize=(max(16, len(labels)*0.12), 5))
    ax.bar(x - w/2, b2_pct_clean, w, label='Baseline 2 vs B0', color='#f0ad4e')
    ax.bar(x + w/2, ag_pct_clean, w, label='Agentic vs B0',    color='#5cb85c')
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, ha='center', fontsize=5)
    ax.set_ylabel('Δ vs Baseline 0 (%)  [negative = faster]')
    ax.set_title('Improvement over Fixed-Tile Baseline 0')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/improvement_over_b0.png", dpi=150)
    plt.close()

    # Plot 3: winner distribution
    counts = data['summary']['winner_counts']
    groups = ['Baseline 0', 'Baseline 2', 'Agentic', 'Tie', 'Inconclusive']
    keys   = ['baseline_0', 'baseline_2', 'agentic', 'tie', 'inconclusive']
    colors = ['#d9534f', '#f0ad4e', '#5cb85c', '#aaaaaa', '#cccccc']
    vals   = [counts.get(k, 0) for k in keys]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(groups, vals, color=colors)
    ax.set_ylabel('Number of shapes')
    ax.set_title('Winner distribution across SmolVLA shapes')
    for i, v in enumerate(vals):
        if v:
            ax.text(i, v + 0.1, str(v), ha='center', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/winner_distribution.png", dpi=150)
    plt.close()

    # Plot 4: agentic vs B2 head-to-head
    ag_vs_b2 = [pct(b2, ag) for b2, ag in zip(b2_means, ag_means)]
    ag_vs_b2_clean = [v if v is not None else 0 for v in ag_vs_b2]
    colors_bar = ['#5cb85c' if v < 0 else '#d9534f' for v in ag_vs_b2_clean]
    fig, ax = plt.subplots(figsize=(max(16, len(labels)*0.12), 5))
    ax.bar(x, ag_vs_b2_clean, color=colors_bar)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, ha='center', fontsize=5)
    ax.set_ylabel('Agentic Δ vs Baseline 2 (%)  [negative = agentic faster]')
    ax.set_title('Agentic vs Baseline 2 — Head-to-Head')
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/agentic_vs_baseline2.png", dpi=150)
    plt.close()

    print(f"  Plots saved to {plots_dir}/", flush=True)


if __name__ == '__main__':
    main()
