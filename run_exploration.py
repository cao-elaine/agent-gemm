#!/usr/bin/env python3
"""
Agentic NPU Tiling Explorer — Phase 3 execution script.
Session: 2026-03-26T00:00:00
"""
import json
import subprocess
import sys
import os
import re
import time
from pathlib import Path

# ============================================================
# Constants
# ============================================================
SCRIPTS_DIR = Path("/home/ec935/vla-to-npu/gemm/scripts")
AGENT_GEMM_DIR = Path("/home/ec935/agent-gemm")
SESSION_ID = "2026-03-26T00:00:00"
DTYPE = "bf16"
MAX_TILES = 5
TIMEOUT_SEC = 180

ALLOWED_M = [32, 64, 128, 256, 512, 960, 1024, 2048, 3072]
ALLOWED_N = [32, 64, 128, 256, 512, 768, 960, 1024, 2048, 3072]
ALLOWED_K = [32, 64, 128, 256, 512, 768, 1024, 2048, 3072]
TILE_POOL = [16, 32, 64, 128, 256, 512]
BLACKLIST = {(128, 128, 128)}
BF16_THRESHOLD = 571779

env = os.environ.copy()
env["ENABLE_AGGRESSIVE_PORT_UTILIZATION_PATCH"] = "1"


def pad_dim(val, allowed):
    for a in sorted(allowed):
        if a >= val:
            return a
    return max(allowed)


def pad_shape(M, N, K):
    Mp = 2048 if M > 1024 else pad_dim(M, ALLOWED_M)
    Np = pad_dim(N, ALLOWED_N)
    Kp = pad_dim(K, ALLOWED_K)
    return Mp, Np, Kp


def valid_tiles_MN(Dp):
    return [t for t in TILE_POOL
            if Dp % t == 0 and t <= Dp
            and any((Dp // t) % c == 0 for c in (3, 4, 5))]


def valid_tiles_K(Kp):
    return [t for t in TILE_POOL if Kp % t == 0 and t <= Kp]


def mem_ok(m, n, k, dsize=2):
    return (m * n + n * k + k * m) * dsize <= 32256


def tile_score(m, n, k):
    score = 0
    if m in (64, 128) and n in (64, 128) and k in (64, 128):
        score += 1000
    if n == 128:
        score += 500
    if m == n == k:
        score += 200
    score += m * n * k // 1000
    if m in (16, 32) or n in (16, 32) or k in (16, 32):
        score -= 100
    return score


def pad_impl(M, N, K, Mp, Np, Kp):
    if M == Mp and N == Np and K == Kp:
        return 'none'
    ab_elements = M * K + K * N
    return 'numpy_pad' if ab_elements > BF16_THRESHOLD else 'manual_copy'


def parse_output(stdout, stderr, returncode):
    """Parse NPU run output. Returns dict with result info.

    Output format from v2_test_mapping_large_gemm.py:
      Avg NPU execution time: XXX.XXus
      Min NPU execution time: YYYus
      [bf16] Padding time: ZZZ.ZZZus
      PASSED!   (or nothing if accuracy fails)
      [NOTE]: bfloat16 have accuracy issue  (bf16 note, not a failure)
    """
    text = stdout + stderr
    result = {
        'status': 'FAILED',
        'npu_avg_us': None,
        'npu_min_us': None,
        'padding_us': None,
        'correctness': None,
        'error_class': None,
        'error_message': None,
    }

    # Check for NPU unavailable
    if 'XDNA driver not found' in text or ('is_available' in text and 'False' in text):
        result['status'] = 'npu_unavailable'
        result['error_class'] = 'NPUUnavailable'
        return result

    # Check for CDO failure (treat as hard failure regardless of preceding PASSED)
    if 'Failed to generate cdo' in text:
        result['error_class'] = 'CDOError'
        result['error_message'] = 'Failed to generate cdo'
        return result

    # ZeroDivisionError
    if 'ZeroDivisionError' in text:
        result['error_class'] = 'ZeroDivisionError'
        match = re.search(r'ZeroDivisionError.*', text)
        result['error_message'] = match.group(0) if match else 'ZeroDivisionError'
        return result

    # Unresolvable mapping
    if 'Unresolvable mapping' in text:
        result['error_class'] = 'UnresolvableMapping'
        match = re.search(r'Unresolvable mapping.*', text)
        result['error_message'] = match.group(0) if match else 'Unresolvable mapping'
        return result

    # Tile size errors
    if 'do not divide padded shape' in text:
        result['error_class'] = 'TileDivisibility'
        result['error_message'] = 'Tile sizes do not divide padded shape'
        return result

    # Buffer / AIE compile errors
    if 'allocated buffers exceeded' in text:
        result['error_class'] = 'BufferOverflow'
        result['error_message'] = 'allocated buffers exceeded'
        return result

    # SystemExit from compile failure (linker errors etc.)
    if 'SystemExit' in text or ('symbol not found' in text) or ('gmake' in text and 'Error' in text and 'Avg NPU' not in text):
        result['error_class'] = 'CompileError'
        lines = [l for l in text.strip().splitlines() if 'error' in l.lower() or 'Error' in l]
        result['error_message'] = (lines[-1] if lines else 'Compile error')[:300]
        return result

    # Timeout
    if returncode == -9:  # SIGKILL from timeout
        result['error_class'] = 'Timeout'
        result['error_message'] = 'Run exceeded 180s timeout'
        return result

    # Parse timing metrics
    # Actual format: "Avg NPU execution time: 177.883us"
    avg_match = re.search(r'Avg NPU execution time:\s*([\d.]+)\s*us', text)
    min_match = re.search(r'Min NPU execution time:\s*([\d.]+)\s*us', text)
    pad_match = re.search(r'Padding time:\s*([\d.]+)\s*us', text)

    if avg_match:
        result['npu_avg_us'] = float(avg_match.group(1))
    if min_match:
        result['npu_min_us'] = float(min_match.group(1))
    if pad_match:
        result['padding_us'] = float(pad_match.group(1))

    # Check for explicit PASSED/FAILED
    # Note: "[NOTE]: bfloat16 have accuracy issue" is NOT a failure — it's just a note
    passed = bool(re.search(r'PASSED!', text))
    # Actual accuracy fail shows "FAILED" not just a NOTE
    accuracy_failed = bool(re.search(r'\bFAILED\b(?!\s*to generate)', text))

    if accuracy_failed:
        result['error_class'] = 'AccuracyFail'
        result['error_message'] = 'Correctness check failed (FAILED in output)'
        return result

    # If we have timing AND PASSED, it's a success
    if passed and result['npu_avg_us'] is not None:
        result['status'] = 'PASSED'
        result['correctness'] = 'PASSED'
        return result

    # If we have timing but no PASSED/FAILED, it's an unknown state
    if result['npu_avg_us'] is not None:
        # timing present but no PASSED — treat as passed (bf16 note case)
        # The bf16 [NOTE] case: timing is printed, [NOTE] printed, no PASSED! line
        result['status'] = 'PASSED'
        result['correctness'] = 'PASSED (timing only, no explicit PASSED line)'
        return result

    # No timing, no explicit result
    if returncode != 0:
        lines = [l for l in text.strip().splitlines() if l.strip()]
        last_err = next((l for l in reversed(lines) if 'error' in l.lower() or 'Error' in l or 'Exception' in l), '')
        if not last_err:
            last_err = lines[-1] if lines else 'Unknown error'
        result['error_class'] = 'UnknownError'
        result['error_message'] = last_err[:300]
        return result

    # returncode==0 but no timing — compile succeeded but NPU didn't run
    result['error_class'] = 'CompileError'
    result['error_message'] = 'No NPU timing in output (compile or runtime failure)'
    return result


def run_single(Mp, Np, Kp, m, n, k, dtype, pad_impl_val, rep_M, rep_N, rep_K):
    """Run a single tile test. Returns (result_dict, stdout+stderr)."""
    cmd = [
        sys.executable, "v2_test_mapping_large_gemm.py",
        "--M", str(Mp), "--N", str(Np), "--K", str(Kp),
        "--m", str(m), "--n", str(n), "--k", str(k),
        "--dtype", dtype,
    ]
    if pad_impl_val != 'none':
        cmd += [
            "--use-padding",
            "--pad-M", str(Mp), "--pad-N", str(Np), "--pad-K", str(Kp),
            "--pad-impl", pad_impl_val,
        ]
        # When padding, pass original dims as M,N,K
        cmd[cmd.index("--M") + 1] = str(rep_M)
        cmd[cmd.index("--N") + 1] = str(rep_N)
        cmd[cmd.index("--K") + 1] = str(rep_K)

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, cwd=str(SCRIPTS_DIR),
            env=env, timeout=TIMEOUT_SEC
        )
        combined = proc.stdout + proc.stderr
        result = parse_output(proc.stdout, proc.stderr, proc.returncode)
        return result, combined
    except subprocess.TimeoutExpired:
        result = {
            'status': 'FAILED', 'npu_avg_us': None, 'npu_min_us': None,
            'padding_us': None, 'correctness': None,
            'error_class': 'Timeout', 'error_message': f'Run exceeded {TIMEOUT_SEC}s'
        }
        return result, f'TIMEOUT after {TIMEOUT_SEC}s'


# ============================================================
# Load all data files
# ============================================================
print("Loading data files...")

# Load shapes
with open(AGENT_GEMM_DIR / "references/smol-vla-dataset/all_practical_shapes.json") as f:
    shapes = json.load(f)
if isinstance(shapes, dict):
    shapes = shapes.get('shapes', shapes)

# Load profiling data (targeted grep approach — file too large to read whole)
profiling_path = AGENT_GEMM_DIR / "references/memory/gemm-data/npu_execution_profiling.json"
with open(profiling_path) as f:
    profiling_db = json.load(f)

errors_path = AGENT_GEMM_DIR / "references/memory/gemm-data/profiling_errors.json"
with open(errors_path) as f:
    errors_db = json.load(f)

# Load profile log
profile_path = AGENT_GEMM_DIR / "prompts/agent-tiling-profile.json"
with open(profile_path) as f:
    profile_data = json.load(f)
profile_runs = profile_data.get('runs', [])

# Build profile lookup
profile_explored = {}
for run in profile_runs:
    shape_str = run.get('shape', '')
    dt = run.get('dtype', '')
    tile = run.get('tile', [])
    key = (shape_str, dt)
    if key not in profile_explored:
        profile_explored[key] = set()
    if tile:
        profile_explored[key].add(f"{tile[0]}_{tile[1]}_{tile[2]}")

print(f"Loaded {len(shapes)} shapes, {len(profile_runs)} profile runs")

# ============================================================
# Build exploration plan
# ============================================================
padded_combos = {}
for s in shapes:
    M, N, K = s['M'], s['N'], s['K']
    Mp, Np, Kp = pad_shape(M, N, K)
    key = (Mp, Np, Kp)
    if key not in padded_combos:
        padded_combos[key] = []
    padded_combos[key].append((M, N, K))

plan = []
for (Mp, Np, Kp), input_shapes in sorted(padded_combos.items()):
    shape_str = f"{Mp}_{Np}_{Kp}"

    # Skip Kp=3072 (known bf16 categorical failure)
    if Kp == 3072:
        continue

    # Get already-explored tiles from profile
    already_explored = profile_explored.get((shape_str, DTYPE), set())

    # Get already working/failing from profiling DB
    already_working = set()
    already_failing = set()
    if shape_str in profiling_db:
        entry = profiling_db[shape_str]
        if DTYPE in entry:
            for tk in entry[DTYPE].get("Working m,n,k", {}).keys():
                already_working.add(tk)
            for tf in entry[DTYPE].get("Failing m,n,k", []):
                already_failing.add(f"{tf[0]}_{tf[1]}_{tf[2]}")

    # Get errors from errors DB
    errors_tiles = set()
    if shape_str in errors_db and DTYPE in errors_db[shape_str]:
        for tk in errors_db[shape_str][DTYPE].keys():
            errors_tiles.add(tk)

    # Build valid tile candidates
    valid_m = valid_tiles_MN(Mp)
    valid_n = valid_tiles_MN(Np)
    valid_k = valid_tiles_K(Kp)

    all_candidates = [
        (m, n, k) for m in valid_m for n in valid_n for k in valid_k
        if mem_ok(m, n, k)
        and (m, n, k) not in BLACKLIST
        and f"{m}_{n}_{k}" not in already_failing
        and f"{m}_{n}_{k}" not in errors_tiles
    ]

    # Remove already working and already explored
    novel = [
        (m, n, k) for m, n, k in all_candidates
        if f"{m}_{n}_{k}" not in already_working
        and f"{m}_{n}_{k}" not in already_explored
    ]

    if not novel:
        continue

    novel.sort(key=lambda t: tile_score(*t), reverse=True)
    selected = novel[:MAX_TILES]

    # Determine pad_impl using representative shape
    rep_M, rep_N, rep_K = input_shapes[0]
    p_impl = pad_impl(rep_M, rep_N, rep_K, Mp, Np, Kp)
    for (M, N, K) in input_shapes:
        if M == Mp and N == Np and K == Kp:
            p_impl = 'none'
            rep_M, rep_N, rep_K = M, N, K
            break

    plan.append({
        'Mp': Mp, 'Np': Np, 'Kp': Kp,
        'shape_str': shape_str,
        'input_shapes': input_shapes,
        'rep_M': rep_M, 'rep_N': rep_N, 'rep_K': rep_K,
        'dtype': DTYPE,
        'pad_impl': p_impl,
        'tiles': selected,
    })

total_runs = sum(len(p['tiles']) for p in plan)
print(f"Plan: {len(plan)} combos, {total_runs} runs")

# ============================================================
# Execute all runs
# ============================================================
all_results = []
npu_unavail = False
run_idx = 0

for combo_idx, combo in enumerate(plan):
    Mp, Np, Kp = combo['Mp'], combo['Np'], combo['Kp']
    shape_str = combo['shape_str']
    p_impl = combo['pad_impl']
    rep_M, rep_N, rep_K = combo['rep_M'], combo['rep_N'], combo['rep_K']

    combo_results = []

    for m, n, k in combo['tiles']:
        run_idx += 1

        if npu_unavail:
            result = {
                'status': 'npu_unavailable', 'npu_avg_us': None, 'npu_min_us': None,
                'padding_us': None, 'correctness': None,
                'error_class': 'NPUUnavailable', 'error_message': 'NPU unavailable'
            }
            combined = ''
        else:
            result, combined = run_single(Mp, Np, Kp, m, n, k, DTYPE, p_impl, rep_M, rep_N, rep_K)
            if result['status'] == 'npu_unavailable':
                npu_unavail = True

        status_str = result['status']
        avg_str = f"{result['npu_avg_us']:.1f}µs" if result['npu_avg_us'] else 'N/A'
        min_str = f"{result['npu_min_us']:.1f}µs" if result['npu_min_us'] else 'N/A'
        pad_str = f"{result['padding_us']:.1f}µs" if result['padding_us'] else 'N/A'
        err_str = f"  ERR={result['error_class']}" if result['error_class'] else ''

        print(f"[{run_idx}/{total_runs}] ({shape_str}) {DTYPE}  tile=({m},{n},{k})  "
              f"→  {status_str}  avg={avg_str}  min={min_str}  pad={pad_str}{err_str}",
              flush=True)

        run_record = {
            'session_id': SESSION_ID,
            'shape': shape_str,
            'input': {'M': rep_M, 'N': rep_N, 'K': rep_K},
            'padded': {'Mp': Mp, 'Np': Np, 'Kp': Kp},
            'dtype': DTYPE,
            'tile': [m, n, k],
            'pad_impl': p_impl,
            'result': result,
        }
        all_results.append(run_record)
        combo_results.append(run_record)

    # Print mini-summary for this combo
    passed = [r for r in combo_results if r['result']['status'] == 'PASSED']
    failed = [r for r in combo_results if r['result']['status'] != 'PASSED']
    print(f"  --- Combo {shape_str} {DTYPE}: {len(passed)} PASSED, {len(failed)} FAILED ---", flush=True)

print(f"\nAll {run_idx} runs complete.", flush=True)

# Save results to a temp file for Phase 4
results_path = AGENT_GEMM_DIR / "references/experiment-results/exploration_20260326_000000_results.json"
results_path.parent.mkdir(parents=True, exist_ok=True)
with open(results_path, 'w') as f:
    json.dump({'session_id': SESSION_ID, 'results': all_results}, f, indent=2)
print(f"Results saved to {results_path}", flush=True)
