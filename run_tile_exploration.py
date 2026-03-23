#!/usr/bin/env python3
"""
NPU Tile Exploration Orchestrator
Session: 2026-03-22
Runs all novel (shape, dtype, tile) combinations and updates profiling data.
"""
import json, os, re, subprocess, sys, time
from datetime import datetime
from pathlib import Path

SESSION_ID = "2026-03-22T12:00:00"
SCRIPT_DIR = "/home/ec935/vla-to-npu/gemm/scripts"
BASE_DIR   = "/home/ec935/agent-gemm"
PROFILING_JSON  = f"{BASE_DIR}/references/memory/gemm-data/npu_execution_profiling.json"
ERRORS_JSON     = f"{BASE_DIR}/references/memory/gemm-data/profiling_errors.json"
AGENT_LOG_JSON  = f"{BASE_DIR}/prompts/agent-tiling-profile.json"
RUN_PLAN_JSON   = "/tmp/npu_run_plan.json"
RESULTS_LOG     = "/tmp/npu_tile_results.jsonl"

os.environ["ENABLE_AGGRESSIVE_PORT_UTILIZATION_PATCH"] = "1"

def load_json(path):
    with open(path) as f:
        return json.load(f)

def save_json(path, obj):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)

def parse_output(stdout, stderr, dtype):
    """Parse script output and return (status, npu_avg, npu_min, pad_us, error_class, error_msg)"""
    # Check for CDO false-positive: PASSED followed by ValueError
    if "Failed to generate cdo" in (stdout + stderr):
        # Find error message
        for line in (stdout + stderr).splitlines():
            if "Failed to generate cdo" in line:
                return "failed", None, None, None, "CDOError", line.strip()
        return "failed", None, None, None, "CDOError", "Failed to generate cdo"

    # Check for PASSED
    passed = "PASSED!" in stdout

    npu_avg = None
    npu_min = None
    pad_us  = None

    for line in stdout.splitlines():
        m = re.search(r'Avg NPU execution time:\s*([\d.]+)\s*us', line)
        if m: npu_avg = float(m.group(1))
        m = re.search(r'Min NPU execution time:\s*([\d.]+)\s*us', line)
        if m: npu_min = float(m.group(1))
        m = re.search(r'Padding time:\s*([\d.]+)\s*us', line)
        if m: pad_us = float(m.group(1))

    if passed and npu_avg is not None:
        return "passed", npu_avg, npu_min, pad_us, None, None

    # Classify failure
    combined = stdout + "\n" + stderr
    if "ZeroDivisionError" in combined:
        lines = [l for l in combined.splitlines() if l.strip()]
        return "failed", None, None, None, "ZeroDivisionError", "ZeroDivisionError"
    if "Unresolvable mapping" in combined:
        for line in combined.splitlines():
            if "Unresolvable mapping" in line:
                return "failed", None, None, None, "UnresolvableMapping", line.strip()
        return "failed", None, None, None, "UnresolvableMapping", "Unresolvable mapping"
    if "do not divide padded shape" in combined:
        return "failed", None, None, None, "DivisibilityError", "Tile sizes do not divide padded shape"
    if "allocated buffers exceeded" in combined or "AIE core" in combined or "Overflow of program memory" in combined:
        return "failed", None, None, None, "MemoryExceeded", "AIE buffer/memory overflow"
    if "XDNA driver not found" in combined:
        return "npu_unavailable", None, None, None, "NPUUnavailable", "XDNA driver not found"

    if passed:
        return "passed", npu_avg, npu_min, pad_us, None, None

    # Unknown failure
    err_lines = [l for l in (stdout + "\n" + stderr).splitlines() if l.strip()]
    first_err = err_lines[-1] if err_lines else "Unknown error"
    return "failed", None, None, None, "Unknown", first_err[:200]

def run_one(Mp, Np, Kp, m, n, k, dtype, pad_impl):
    cmd = [
        "python", "v2_test_mapping_large_gemm.py",
        "--M", str(Mp), "--N", str(Np), "--K", str(Kp),
        "--m", str(m),  "--n", str(n),  "--k", str(k),
        "--dtype", dtype,
    ]
    if pad_impl != 'none':
        cmd += ["--use-padding",
                "--pad-M", str(Mp), "--pad-N", str(Np), "--pad-K", str(Kp),
                "--pad-impl", pad_impl]

    env = os.environ.copy()
    env["ENABLE_AGGRESSIVE_PORT_UTILIZATION_PATCH"] = "1"

    try:
        result = subprocess.run(
            cmd, cwd=SCRIPT_DIR, env=env,
            capture_output=True, text=True, timeout=180
        )
        return result.stdout, result.stderr
    except subprocess.TimeoutExpired as e:
        return "", f"TIMEOUT after 180s: {str(e)[:200]}"
    except Exception as e:
        return "", f"Exception: {str(e)[:200]}"

def update_profiling(profiling, shape_key, dtype, m, n, k, status, npu_avg, npu_min):
    if shape_key not in profiling:
        profiling[shape_key] = {}
    if dtype not in profiling[shape_key]:
        profiling[shape_key][dtype] = {"Best m,n,k": {}, "Working m,n,k": {}, "Failing m,n,k": []}

    entry = profiling[shape_key][dtype]
    tile_key = f"{m}_{n}_{k}"

    if status == "passed":
        working = entry.setdefault("Working m,n,k", {})
        if tile_key in working:
            # Update running mean
            old = working[tile_key]
            cnt = old.get("count", 1)
            old["avg"] = (old["avg"] * cnt + npu_avg) / (cnt + 1)
            old["count"] = cnt + 1
        else:
            working[tile_key] = {"size": [m, n, k], "avg": npu_avg, "min": npu_min, "count": 1}

        # Update best
        best = entry.get("Best m,n,k", {})
        if not best or npu_avg < best.get("Best NPU average time", float('inf')):
            entry["Best m,n,k"] = {"size": [m, n, k], "Best NPU average time": npu_avg}

    elif status == "failed":
        failing = entry.setdefault("Failing m,n,k", [])
        tile_list = [m, n, k]
        if tile_list not in failing:
            failing.append(tile_list)

def append_agent_log(agent_log, run_entry):
    agent_log["runs"].append(run_entry)

def update_errors_json(errors_data, shape_key, dtype, m, n, k, stderr):
    if not stderr or len(stderr.strip()) < 10:
        return
    if shape_key not in errors_data:
        errors_data[shape_key] = {}
    if dtype not in errors_data[shape_key]:
        errors_data[shape_key][dtype] = {}
    tile_key = f"{m}_{n}_{k}"
    if tile_key in errors_data[shape_key][dtype]:
        tile_key = f"{m}_{n}_{k}__2"
    errors_data[shape_key][dtype][tile_key] = [l for l in stderr.splitlines()[-30:] if l.strip()]

def main():
    session_start = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    SESSION_ID_LIVE = session_start
    print(f"=== NPU Tile Exploration Session: {session_start} ===")
    print(f"Loading run plan from {RUN_PLAN_JSON}...")

    run_plan = load_json(RUN_PLAN_JSON)
    total = len(run_plan)
    print(f"Total runs planned: {total}")

    # Load data files
    profiling = load_json(PROFILING_JSON)

    try:
        errors_data = load_json(ERRORS_JSON)
    except:
        errors_data = {}

    # Load or init agent log
    try:
        agent_log = load_json(AGENT_LOG_JSON)
        if "runs" not in agent_log:
            agent_log["runs"] = []
    except:
        agent_log = {
            "description": "Tile exploration run log. Each entry is a single (shape, dtype, tile) trial run by the agentic tiling explorer.",
            "schema": {
                "session_id": "ISO-8601 timestamp of session start",
                "shape": "Mp_Np_Kp string key",
                "input": {"M": "int", "N": "int", "K": "int"},
                "padded": {"M": "int", "N": "int", "K": "int"},
                "dtype": "i8 | i16 | bf16",
                "tile": "list[m,n,k]",
                "pad_impl": "manual_copy | numpy_pad | none",
                "result": {
                    "status": "passed | failed | npu_unavailable",
                    "npu_avg_us": "float or null",
                    "npu_min_us": "float or null",
                    "padding_us": "float or null",
                    "correctness": "true | false | null",
                    "error_class": "string or null",
                    "error_message": "string or null"
                }
            },
            "runs": []
        }

    # Check already-done from results log
    done_set = set()
    results_so_far = []
    if os.path.exists(RESULTS_LOG):
        with open(RESULTS_LOG) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    key = (r['shape'], r['dtype'], tuple(r['tile']))
                    done_set.add(key)
                    results_so_far.append(r)
                except:
                    pass
        print(f"Resuming: {len(done_set)} runs already done.")

    passed_count = sum(1 for r in results_so_far if r.get('status') == 'passed')
    failed_count = sum(1 for r in results_so_far if r.get('status') == 'failed')
    npu_gone = False
    new_bests = []

    # Track prior bests (before this session)
    prior_bests = {}
    for run in run_plan:
        M,N,K,dtype,Mp,Np,Kp,pad_impl,m,n,k = run
        sk = f"{Mp}_{Np}_{Kp}"
        key = (sk, dtype)
        if key not in prior_bests and sk in profiling and dtype in profiling[sk]:
            b = profiling[sk][dtype].get("Best m,n,k", {})
            prior_bests[key] = (b.get("size"), b.get("Best NPU average time"))

    for i, run in enumerate(run_plan):
        M,N,K,dtype,Mp,Np,Kp,pad_impl,m,n,k = run
        sk = f"{Mp}_{Np}_{Kp}"
        run_key = (sk, dtype, (m,n,k))

        if run_key in done_set:
            continue

        if npu_gone:
            status, npu_avg, npu_min, pad_us = "npu_unavailable", None, None, None
            err_class, err_msg = "NPUUnavailable", "NPU gone mid-session"
        else:
            print(f"[{i+1}/{total}] Compiling and running ({Mp},{Np},{Kp}) {dtype} tile=({m},{n},{k})...", flush=True)
            t0 = time.time()
            stdout, stderr = run_one(Mp, Np, Kp, m, n, k, dtype, pad_impl)
            elapsed = time.time() - t0
            status, npu_avg, npu_min, pad_us, err_class, err_msg = parse_output(stdout, stderr, dtype)

            if status == "npu_unavailable":
                npu_gone = True

            # Print live progress
            if status == "passed":
                avg_str = f"avg={npu_avg:.1f}µs" if npu_avg else ""
                min_str = f"min={npu_min:.1f}µs" if npu_min else ""
                pad_str = f"pad={pad_us:.1f}µs" if pad_us else ""
                print(f"[{i+1}/{total}] ({Mp},{Np},{Kp}) {dtype} tile=({m},{n},{k}) → PASSED  {avg_str}  {min_str}  {pad_str}  [{elapsed:.0f}s]", flush=True)
                passed_count += 1
            else:
                print(f"[{i+1}/{total}] ({Mp},{Np},{Kp}) {dtype} tile=({m},{n},{k}) → {status.upper()}  {err_class}: {err_msg}  [{elapsed:.0f}s]", flush=True)
                failed_count += 1

        # Update profiling
        if status in ("passed", "failed"):
            prior_best_avg = prior_bests.get((sk, dtype), (None, None))[1]
            update_profiling(profiling, sk, dtype, m, n, k, status, npu_avg, npu_min)

            # Check for new best
            if status == "passed" and prior_best_avg is not None and npu_avg < prior_best_avg:
                pb = prior_bests[(sk, dtype)]
                delta_pct = (npu_avg - prior_best_avg) / prior_best_avg * 100
                new_bests.append((sk, dtype, pb[0], prior_best_avg, [m,n,k], npu_avg, delta_pct))
                print(f"  *** NEW BEST for ({sk}) {dtype}: ({m},{n},{k})={npu_avg:.1f}µs vs prior ({pb[0]})={prior_best_avg:.1f}µs ({delta_pct:+.1f}%)", flush=True)

        # Update errors
        if status == "failed" and err_class not in ("ZeroDivisionError", "UnresolvableMapping", "DivisibilityError"):
            update_errors_json(errors_data, sk, dtype, m, n, k, stderr if 'stderr' in dir() else "")

        # Build run entry
        run_entry = {
            "session_id": SESSION_ID_LIVE,
            "shape": sk,
            "input": {"M": M, "N": N, "K": K},
            "padded": {"M": Mp, "N": Np, "K": Kp},
            "dtype": dtype,
            "tile": [m, n, k],
            "pad_impl": pad_impl,
            "result": {
                "status": status,
                "npu_avg_us": npu_avg,
                "npu_min_us": npu_min,
                "padding_us": pad_us,
                "correctness": True if status == "passed" else (False if status == "failed" else None),
                "error_class": err_class,
                "error_message": err_msg
            }
        }

        # Append to results log (for crash recovery)
        with open(RESULTS_LOG, 'a') as f:
            f.write(json.dumps(run_entry) + "\n")
        done_set.add(run_key)

        # Append to agent log
        append_agent_log(agent_log, run_entry)

        # Periodically save data files (every 10 runs)
        if (i + 1) % 10 == 0:
            save_json(PROFILING_JSON, profiling)
            save_json(ERRORS_JSON, errors_data)
            save_json(AGENT_LOG_JSON, agent_log)
            print(f"  [CHECKPOINT] Saved data files at run {i+1}", flush=True)

    # Final save
    save_json(PROFILING_JSON, profiling)
    save_json(ERRORS_JSON, errors_data)
    save_json(AGENT_LOG_JSON, agent_log)

    print(f"\n=== SESSION COMPLETE ===")
    print(f"Total runs: {len(done_set)} | Passed: {passed_count} | Failed: {failed_count}")
    print(f"New best tiles found: {len(new_bests)}")
    for sk, dtype, old_tile, old_avg, new_tile, new_avg, delta in new_bests:
        print(f"  {sk} {dtype}: {old_tile}@{old_avg:.1f}µs → {new_tile}@{new_avg:.1f}µs ({delta:+.1f}%)")

    # Write summary to /tmp for Phase 5 processing
    summary = {
        "session_id": SESSION_ID_LIVE,
        "total_runs": len(done_set),
        "passed": passed_count,
        "failed": failed_count,
        "new_bests": new_bests
    }
    with open('/tmp/npu_session_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary written to /tmp/npu_session_summary.json")

if __name__ == "__main__":
    main()
