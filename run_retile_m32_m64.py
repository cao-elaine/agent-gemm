#!/usr/bin/env python3
"""
Run all novel bf16 tiles for smolvla_retile_m32_m64 shapes.
Session: 2026-03-29 17:10 — NPU available after prior npu_unavailable session.
"""

import json
import subprocess
import sys
import os
import re
import time
from copy import deepcopy
from datetime import datetime

SESSION_ID = "2026-03-29T17:10:00"
SCRIPT_DIR = "/home/ec935/vla-to-npu/gemm/scripts"
PROFILING_FILE = "/home/ec935/agent-gemm/references/memory/gemm-data/npu_execution_profiling.json"
ERRORS_FILE = "/home/ec935/agent-gemm/references/memory/gemm-data/profiling_errors.json"
PROFILE_LOG = "/home/ec935/agent-gemm/prompts/agent-tiling-profile.json"
SHAPES_FILE = "/home/ec935/agent-gemm/references/smol-vla-dataset/smolvla_retile_m32_m64.json"

TILE_POOL = [16, 32, 64, 128, 256, 512]
DSIZE = 2  # bf16

os.environ["ENABLE_AGGRESSIVE_PORT_UTILIZATION_PATCH"] = "1"

def auto_row_col(Pm):
    if Pm < 3:
        return True
    return any(Pm % c == 0 for c in (3, 4, 5))

def valid_tiles(Mp, Np, Kp):
    tm_pool = [t for t in TILE_POOL if Mp % t == 0 and t <= Mp and auto_row_col(Mp // t)]
    tn_pool = [t for t in TILE_POOL if Np % t == 0 and t <= Np and auto_row_col(Np // t)]
    tk_pool = [t for t in TILE_POOL if Kp % t == 0 and t <= Kp]
    candidates = []
    for m in tm_pool:
        for n in tn_pool:
            for k in tk_pool:
                if (m*n + n*k + k*m) * DSIZE <= 32256:
                    if (m, n, k) != (128, 128, 128):
                        candidates.append((m, n, k))
    return candidates

def priority(t):
    m, n, k = t
    score = 0
    if all(64 <= x <= 128 for x in (m, n, k)): score += 4
    if n == 128: score += 3
    if m == n == k: score += 2
    score += min(m*n*k // 10000, 3)
    if min(m, n, k) <= 32: score -= 2
    return -score

def classify_error(output):
    if "ZeroDivisionError" in output:
        return "ZeroDivisionError", "ZeroDivisionError: tile dimension issue"
    if "Unresolvable mapping" in output:
        return "UnresolvableMapping", output.split("Unresolvable mapping")[1][:80].strip()
    if "Tile sizes do not divide" in output:
        return "TileDivisibility", "Tile sizes do not divide padded shape"
    if "allocated buffers exceeded" in output or ("AIE core" in output and "error" in output.lower()):
        return "MemoryExceeded", "allocated buffers exceeded"
    if "Failed to generate cdo" in output:
        return "CDOError", "Failed to generate cdo"
    if "XDNA driver not found" in output or "KMQ device" in output:
        return "NPUUnavailable", "NPU unavailable"
    if "Overflow of program memory" in output or "CMake developer warning" in output:
        return "MemoryExceeded", "program memory overflow"
    if "terminate called" in output or "Aborted" in output:
        return "NPUUnavailable", "NPU crash"
    return "Unknown", output[-200:].strip() if output else "no output"

def parse_result(output):
    avg_us = None
    min_us = None
    pad_us = None
    passed = False

    m = re.search(r'Avg NPU execution time:\s+([\d.]+)us', output)
    if m: avg_us = float(m.group(1))
    m = re.search(r'Min NPU execution time:\s+([\d.]+)us', output)
    if m: min_us = float(m.group(1))
    m = re.search(r'Padding time:\s+([\d.]+)us', output)
    if m: pad_us = float(m.group(1))

    # bf16 K=2048 false positive check
    if "Failed to generate cdo" in output:
        passed = False
    elif "PASSED!" in output:
        passed = True

    return avg_us, min_us, pad_us, passed

def run_tile(M, N, K, m, n, k, dtype="bf16", timeout=200):
    cmd = [
        "python", "v2_test_mapping_large_gemm.py",
        "--M", str(M), "--N", str(N), "--K", str(K),
        "--m", str(m), "--n", str(n), "--k", str(k),
        "--dtype", dtype,
    ]
    env = os.environ.copy()
    env["ENABLE_AGGRESSIVE_PORT_UTILIZATION_PATCH"] = "1"
    try:
        result = subprocess.run(
            cmd, cwd=SCRIPT_DIR, capture_output=True, text=True,
            timeout=timeout, env=env
        )
        output = result.stdout + "\n" + result.stderr
        return output, False  # False = not timed out
    except subprocess.TimeoutExpired:
        return "TIMEOUT", True

def run_tile_with_retry(M, N, K, m, n, k, dtype="bf16", timeout=200):
    """Run a tile. On NPU crash: wait 30s and retry once.
    If still unavailable, return npu_unavail=True so the caller can skip this tile
    and try the next (rather than exhausting retries on a crash-inducing tile).
    If 3 consecutive tiles are npu_unavail, caller should stop the session."""
    output, timed_out = run_tile(M, N, K, m, n, k, dtype, timeout)
    if timed_out:
        return output, timed_out, False
    error_class, _ = classify_error(output)
    if error_class == "NPUUnavailable":
        # Single short wait then one retry
        print("  [NPU unavailable, waiting 30s before single retry]")
        sys.stdout.flush()
        time.sleep(30)
        output2, timed_out2 = run_tile(M, N, K, m, n, k, dtype, timeout)
        if timed_out2:
            return output2, True, False
        error_class2, _ = classify_error(output2)
        if error_class2 == "NPUUnavailable":
            return output2, False, True  # still unavail → skip this tile
        return output2, False, False
    return output, False, False

def load_json(path):
    with open(path) as f:
        return json.load(f)

def save_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def main():
    # Load data
    prof = load_json(PROFILING_FILE)
    errors = load_json(ERRORS_FILE)
    profile_log = load_json(PROFILE_LOG)
    shapes_data = load_json(SHAPES_FILE)
    shapes = shapes_data['shapes']

    # Get npu_unavailable tiles (treat as novel)
    npu_unavail = {}
    for run in profile_log.get('runs', []):
        if run.get('result', {}).get('status') == 'npu_unavailable' and run.get('dtype') == 'bf16':
            key = run['shape']
            t = tuple(run['tile'])
            if key not in npu_unavail:
                npu_unavail[key] = set()
            npu_unavail[key].add(t)

    # Build full plan
    plan = []
    for s in shapes:
        M, N, K = s['M'], s['N'], s['K']
        Mp, Np, Kp = M, N, K  # no padding needed
        key = '%d_%d_%d' % (M, N, K)

        # Get already working/failing from profiling
        already_working = set()
        already_failing = set()
        if key in prof and 'bf16' in prof[key] and prof[key]['bf16']:
            bf16 = prof[key]['bf16']
            for tk in (bf16.get('Working m,n,k') or {}):
                parts = tk.split('_')
                already_working.add(tuple(int(x) for x in parts))
            for tf in (bf16.get('Failing m,n,k') or []):
                already_failing.add(tuple(tf))

        # Get all valid tiles, filter novel
        all_valid = valid_tiles(Mp, Np, Kp)
        novel = [t for t in all_valid if t not in already_working and t not in already_failing]
        novel.sort(key=priority)

        if novel:
            plan.append({
                'M': M, 'N': N, 'K': K,
                'key': key,
                'tiles': novel,
                'already_working': already_working,
                'already_failing': already_failing,
                'prior_best_avg': None,
                'prior_best_tile': None,
            })
            # Extract prior best
            if key in prof and 'bf16' in prof[key] and prof[key]['bf16']:
                best = prof[key]['bf16'].get('Best m,n,k') or {}
                if best:
                    plan[-1]['prior_best_avg'] = best.get('Best NPU average time')
                    plan[-1]['prior_best_tile'] = best.get('size')

    total_runs = sum(len(p['tiles']) for p in plan)
    print("="*70)
    print("TILING EXPLORATION PLAN")
    print("Session: %s" % SESSION_ID)
    print("Shapes: %d  |  Total tile runs: %d  |  dtype: bf16" % (len(plan), total_runs))
    print("="*70)
    sys.stdout.flush()

    # Track results
    run_num = 0
    session_results = []  # (key, tile, status, avg, min, pad, error_class, error_msg)
    new_working = 0
    new_failing = 0
    new_best = 0

    # Track per-shape results for summary
    shape_results = {}

    for shape_plan in plan:
        M, N, K = shape_plan['M'], shape_plan['N'], shape_plan['K']
        key = shape_plan['key']
        tiles = shape_plan['tiles']
        prior_best_avg = shape_plan['prior_best_avg']
        prior_best_tile = shape_plan['prior_best_tile']

        shape_results[key] = {
            'tiles': [], 'new_best_tile': None, 'new_best_avg': None,
            'prior_best_avg': prior_best_avg, 'prior_best_tile': prior_best_tile
        }

        print("\n── (%d,%d,%d) bf16 (%d novel tiles, prior best: %s @ %.1fµs) ──" % (
            M, N, K, len(tiles),
            str(prior_best_tile) if prior_best_tile else "none",
            prior_best_avg if prior_best_avg else 0.0
        ))
        print("   %-20s  %-8s  %-10s  %-10s  %-10s  %s" % (
            "Tile", "Status", "Avg NPU", "Min NPU", "Pad time", "vs prior best"
        ))
        sys.stdout.flush()

        shape_current_best_avg = prior_best_avg
        shape_current_best_tile = prior_best_tile

        for tile in tiles:
            run_num += 1
            m, n, k = tile

            output, timed_out, npu_tile_unavail = run_tile_with_retry(M, N, K, m, n, k, timeout=200)

            if timed_out:
                status = "failed"
                avg_us = min_us = pad_us = None
                error_class = "MemoryExceeded"
                error_msg = "TIMEOUT >200s"
                passed = False
            elif npu_tile_unavail:
                # Tile caused NPU crash; mark as npu_unavailable and continue to next tile
                status = "npu_unavailable"
                avg_us = min_us = pad_us = None
                error_class = "NPUUnavailable"
                error_msg = "NPU unavailable after single retry; tile skipped"
                passed = False
                consecutive_npu_failures = getattr(main, '_consec_npu', 0) + 1
                main._consec_npu = consecutive_npu_failures
                print("[%d/%d] (%d,%d,%d) bf16 tile=(%d,%d,%d) → NPU UNAVAIL (tile skipped, consec=%d)" % (
                    run_num, total_runs, M, N, K, m, n, k, consecutive_npu_failures))
                sys.stdout.flush()
                if consecutive_npu_failures >= 5:
                    print("  5 consecutive NPU failures — NPU is down. Stopping session.")
                    _flush_results(prof, errors, profile_log, session_results, new_working, new_failing, new_best)
                    return
                # Record and continue to next tile (don't add to profile_log, let it be retried)
                continue
            else:
                main._consec_npu = 0  # reset consecutive counter on any success/non-npu-fail
                avg_us, min_us, pad_us, passed = parse_result(output)
                if passed:
                    status = "passed"
                    error_class = None
                    error_msg = None
                else:
                    status = "failed"
                    error_class, error_msg = classify_error(output)

            # vs prior best
            vs_str = ""
            if status == "passed" and avg_us is not None:
                if shape_current_best_avg is None:
                    vs_str = "FIRST DATA"
                    shape_current_best_avg = avg_us
                    shape_current_best_tile = list(tile)
                elif avg_us < shape_current_best_avg:
                    pct = 100.0 * (avg_us - shape_current_best_avg) / shape_current_best_avg
                    vs_str = "NEW BEST %.1f%%" % pct
                    shape_current_best_avg = avg_us
                    shape_current_best_tile = list(tile)
                else:
                    pct = 100.0 * (avg_us - shape_current_best_avg) / shape_current_best_avg
                    vs_str = "+%.1f%% vs best" % pct

            print("[%d/%d] (%d,%d,%d) bf16 tile=(%d,%d,%d) → %s  avg=%s  min=%s  pad=%s  %s" % (
                run_num, total_runs, M, N, K, m, n, k,
                "PASSED" if status=="passed" else ("UNAVAIL" if status=="npu_unavailable" else "FAILED"),
                ("%.1fµs" % avg_us) if avg_us else "—",
                ("%.1fµs" % min_us) if min_us else "—",
                ("%.1fµs" % pad_us) if pad_us else "—",
                vs_str
            ))
            sys.stdout.flush()

            shape_results[key]['tiles'].append({
                'tile': tile, 'status': status, 'avg': avg_us, 'min': min_us,
                'pad': pad_us, 'error_class': error_class, 'error_msg': error_msg,
                'vs_str': vs_str
            })

            # Update profiling.json in-memory
            if key not in prof:
                prof[key] = {}
            if 'bf16' not in prof[key]:
                prof[key]['bf16'] = {'Working m,n,k': {}, 'Failing m,n,k': [], 'Best m,n,k': None}
            bf16_entry = prof[key]['bf16']
            if bf16_entry is None:
                prof[key]['bf16'] = {'Working m,n,k': {}, 'Failing m,n,k': [], 'Best m,n,k': None}
                bf16_entry = prof[key]['bf16']

            tile_key = '%d_%d_%d' % (m, n, k)
            if status == "passed" and avg_us is not None:
                working = bf16_entry.get('Working m,n,k') or {}
                if tile_key in working:
                    # Update running mean
                    old = working[tile_key]
                    old_count = old.get('count', 1)
                    old['avg'] = (old['avg'] * old_count + avg_us) / (old_count + 1)
                    old['min'] = min(old.get('min', min_us), min_us) if min_us else old.get('min')
                    old['count'] = old_count + 1
                else:
                    if 'Working m,n,k' not in bf16_entry or bf16_entry['Working m,n,k'] is None:
                        bf16_entry['Working m,n,k'] = {}
                    bf16_entry['Working m,n,k'][tile_key] = {
                        'size': list(tile), 'avg': avg_us, 'min': min_us, 'count': 1
                    }
                    new_working += 1

                # Update best
                current_best = bf16_entry.get('Best m,n,k')
                if current_best is None or avg_us < current_best.get('Best NPU average time', float('inf')):
                    bf16_entry['Best m,n,k'] = {
                        'size': list(tile), 'Best NPU average time': avg_us
                    }
                    if prior_best_avg and avg_us < prior_best_avg:
                        new_best += 1
                    elif not prior_best_avg:
                        new_best += 1

            elif status == "failed":
                failing = bf16_entry.get('Failing m,n,k') or []
                if list(tile) not in failing:
                    if 'Failing m,n,k' not in bf16_entry or bf16_entry['Failing m,n,k'] is None:
                        bf16_entry['Failing m,n,k'] = []
                    bf16_entry['Failing m,n,k'].append(list(tile))
                    new_failing += 1

                # Update profiling_errors.json
                if error_msg and error_class not in ("ZeroDivisionError", "TileDivisibility"):
                    if key not in errors:
                        errors[key] = {}
                    if 'bf16' not in errors[key]:
                        errors[key]['bf16'] = {}
                    if tile_key not in errors[key]['bf16']:
                        errors[key]['bf16'][tile_key] = [error_class + ": " + str(error_msg)]

            # Append to profile log
            profile_log['runs'].append({
                'session_id': SESSION_ID,
                'shape': key,
                'input': {'M': M, 'N': N, 'K': K},
                'padded': {'M': M, 'N': N, 'K': K},
                'dtype': 'bf16',
                'tile': list(tile),
                'pad_impl': 'none',
                'result': {
                    'status': status,
                    'npu_avg_us': avg_us,
                    'npu_min_us': min_us,
                    'padding_us': pad_us,
                    'correctness': True if status == "passed" else (False if status == "failed" else None),
                    'error_class': error_class,
                    'error_message': error_msg
                }
            })

            # Save every 20 runs
            if run_num % 20 == 0:
                save_json(PROFILING_FILE, prof)
                save_json(ERRORS_FILE, errors)
                save_json(PROFILE_LOG, profile_log)
                print("  [saved checkpoint at run %d]" % run_num)
                sys.stdout.flush()

        # Shape done - update current best
        shape_results[key]['new_best_tile'] = shape_current_best_tile
        shape_results[key]['new_best_avg'] = shape_current_best_avg

    # Final save
    save_json(PROFILING_FILE, prof)
    save_json(ERRORS_FILE, errors)
    save_json(PROFILE_LOG, profile_log)

    # Print final summary
    print("\n" + "="*70)
    print("TILING EXPLORATION — SESSION COMPLETE")
    print("Date: %s" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("Shapes tested: %d  |  Total runs: %d" % (len(plan), run_num))
    print("New working tiles: %d  |  New failing tiles: %d  |  New bests: %d" % (
        new_working, new_failing, new_best))
    print("="*70)

    # Per-shape results table
    print("\nPER-SHAPE RESULTS:")
    for shape_plan in plan:
        key = shape_plan['key']
        M, N, K = shape_plan['M'], shape_plan['N'], shape_plan['K']
        sr = shape_results[key]
        prior_best_avg = sr['prior_best_avg']
        passed_tiles = [t for t in sr['tiles'] if t['status'] == 'passed']
        failed_tiles = [t for t in sr['tiles'] if t['status'] == 'failed']
        print("Shape (%d,%d,%d) bf16 — %d passed, %d failed" % (M, N, K, len(passed_tiles), len(failed_tiles)))
        if passed_tiles:
            best = min(passed_tiles, key=lambda t: t['avg'] or float('inf'))
            print("  New best: tile=%s  avg=%.1fµs%s" % (
                str(best['tile']),
                best['avg'],
                ("  IMPROVEMENT from %.1fµs" % prior_best_avg) if prior_best_avg and best['avg'] < prior_best_avg else ""
            ))

    print("\nFiles saved:")
    print("  npu_execution_profiling.json — %d new working, %d new failing" % (new_working, new_failing))
    print("  profiling_errors.json — updated")
    print("  agent-tiling-profile.json — %d new entries" % run_num)

def _flush_results(prof, errors, profile_log, session_results, new_working, new_failing, new_best):
    save_json(PROFILING_FILE, prof)
    save_json(ERRORS_FILE, errors)
    save_json(PROFILE_LOG, profile_log)
    print("Emergency save complete.")

if __name__ == '__main__':
    main()
