"""
score_recommendations.py

Scores the agent's tiling recommendations against ground-truth profiling data.

For each of the 50 held-out shapes:
  - If agent tile == actual best tile  → record as exact match. No NPU run.
  - If agent tile != actual best tile  → run the NPU once with the agent's tile.
                                         Compare measured time to best-tile time.

Each mismatch costs 0 or 1 NPU run (3 trials → avg).

Outputs (in runs/<N>/ subdirectory, auto-incremented):
  runs/<N>/agent_recommendations.json
  runs/<N>/agent_recommendations.md
  runs/<N>/eval_results.json
  runs/<N>/eval_results.md

Usage:
  cd /home/ec935/agent-gemm
  python3 references/held-out-eval/score_recommendations.py
"""

import json
import os
import re
import shutil
import subprocess
import time
from datetime import datetime

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
BASE_DIR    = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
RUNS_ROOT   = os.path.join(SCRIPT_DIR, "runs")
os.makedirs(RUNS_ROOT, exist_ok=True)

AGENT_FILE  = os.path.join(SCRIPT_DIR, "agent_recommendations.json")
GROUND_FILE = os.path.join(SCRIPT_DIR, "held_out_shapes.json")
INPUT_FILE  = os.path.join(SCRIPT_DIR, "input_shapes.json")

PYTHON      = "/opt/anaconda3/envs/allo-base/bin/python3"
NPU_SCRIPT  = "/home/ec935/vla-to-npu/gemm/scripts/v2_test_mapping_large_gemm.py"
DTYPE       = "bf16"
TRIALS      = 3


def _next_run_index():
    i = 1
    while os.path.isdir(os.path.join(RUNS_ROOT, str(i))):
        i += 1
    return i


# ── NPU execution ──────────────────────────────────────────────────────────────

def run_npu_trial(Mp, Np, Kp, m, n, k, trial_idx):
    """Run one trial directly on the padded shape (no padding step)."""
    cmd = [
        "bash", "-c",
        f"cd /home/ec935/vla-to-npu/gemm/scripts && "
        f"ENABLE_AGGRESSIVE_PORT_UTILIZATION_PATCH=1 "
        f"{PYTHON} v2_test_mapping_large_gemm.py "
        f"--M {Mp} --N {Np} --K {Kp} "
        f"--m {m} --n {n} --k {k} "
        f"--dtype {DTYPE}"
    ]
    print(f"    trial {trial_idx}/{TRIALS}  ({Mp},{Np},{Kp}) tile=({m},{n},{k})",
          flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    stdout = result.stdout + result.stderr

    avg_m = re.search(r'Avg NPU execution time:\s*([\d.]+)\s*us', stdout)
    if avg_m and 'PASSED' in stdout and 'ValueError' not in stdout:
        return float(avg_m.group(1)), 'passed', ''

    # Classify failure
    if 'Failed to open KMQ device' in stdout or 'XDNA driver' in stdout:
        return None, 'npu_unavailable', 'NPU unavailable'
    if 'ValueError: Failed to generate cdo' in stdout:
        return None, 'failed', 'CDOError'
    if 'ZeroDivisionError' in stdout:
        return None, 'failed', 'ZeroDivisionError'
    if 'Unresolvable mapping' in stdout:
        return None, 'failed', 'UnresolvableMapping'
    if 'FAILED' in stdout:
        return None, 'failed', 'AccuracyFail'
    return None, 'failed', stdout[-300:].strip()


def run_npu(Mp, Np, Kp, m, n, k):
    """Run TRIALS trials; return (avg_us, status, error)."""
    times = []
    for i in range(1, TRIALS + 1):
        t, status, err = run_npu_trial(Mp, Np, Kp, m, n, k, i)
        if status == 'npu_unavailable':
            return None, 'npu_unavailable', err
        if status == 'passed':
            times.append(t)
        else:
            return None, 'failed', err
    avg = round(sum(times) / len(times), 2) if times else None
    return avg, 'passed', ''


# ── Scoring ────────────────────────────────────────────────────────────────────

def score_shape(agent_rec, ground_entry, input_entry):
    rec_tile  = list(agent_rec["recommended_tile"])
    best_tile = list(ground_entry["actual_best_tile"])
    best_avg  = ground_entry["actual_best_avg_us"]

    Mp = ground_entry["Mp"]
    Np = ground_entry["Np"]
    Kp = ground_entry["Kp"]

    result = {
        "padded_key":       agent_rec["padded_key"],
        "Mp": Mp, "Np": Np, "Kp": Kp,
        "recommended_tile": rec_tile,
        "actual_best_tile": best_tile,
        "actual_best_avg_us": best_avg,
        "confidence":       agent_rec.get("confidence", "unknown"),
        "exact_match":      rec_tile == best_tile,
        "npu_run":          False,
        "agent_avg_us":     None,
        "suboptimality":    None,
        "npu_status":       None,
        "npu_error":        None,
    }

    if result["exact_match"]:
        print(f"  ✓ MATCH  {agent_rec['padded_key']}  {rec_tile}")
        return result

    # Mismatch → run NPU
    print(f"  ✗ MISS   {agent_rec['padded_key']}  rec={rec_tile}  best={best_tile}  → running NPU...")
    m, n, k = rec_tile
    avg, status, err = run_npu(Mp, Np, Kp, m, n, k)

    result["npu_run"]    = True
    result["npu_status"] = status
    result["npu_error"]  = err or None

    if avg is not None:
        result["agent_avg_us"]  = avg
        result["suboptimality"] = round(avg / best_avg, 4) if best_avg else None
        gap = f"{result['suboptimality']:.3f}×"
        print(f"    agent={avg:.1f}µs  best={best_avg:.1f}µs  ratio={gap}")
    else:
        print(f"    NPU run failed: {status} — {err}")

    return result


# ── Report ─────────────────────────────────────────────────────────────────────

def write_json(results, summary, run_dir):
    out = {"summary": summary, "per_shape": results}
    path = os.path.join(run_dir, "eval_results.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {path}")


def write_md(results, summary, run_dir, run_idx):
    lines = [
        f"# Held-Out Eval Results — Run {run_idx}",
        f"**Generated:** {summary['generated']}  ",
        f"**Shapes scored:** {summary['n_shapes']}  ",
        f"**NPU runs:** {summary['npu_runs']} (mismatches only)  ",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Exact match | {summary['exact_match']}/{summary['n_shapes']} ({summary['exact_match_pct']}%) |",
        f"| Avg suboptimality (mismatches with NPU data) | {summary['avg_suboptimality']}× |",
        f"| Max suboptimality | {summary['max_suboptimality']}× |",
        f"| NPU run failed | {summary['npu_failed']} |",
        f"| NPU unavailable | {summary['npu_unavailable']} |",
        "",
        "## Per-Shape Results",
        "",
        "| Shape | Rec tile | Best tile | Match | Agent µs | Best µs | Subopt |",
        "|-------|----------|-----------|-------|----------|---------|--------|",
    ]

    for r in sorted(results, key=lambda x: x["padded_key"]):
        match  = "✓" if r["exact_match"] else "✗"
        a_avg  = f"{r['agent_avg_us']:.1f}" if r["agent_avg_us"] else ("—" if not r["npu_run"] else r["npu_status"])
        b_avg  = f"{r['actual_best_avg_us']:.1f}"
        sub    = f"{r['suboptimality']:.3f}×" if r["suboptimality"] else "—"
        lines.append(
            f"| {r['padded_key']} | {r['recommended_tile']} | {r['actual_best_tile']} "
            f"| {match} | {a_avg} | {b_avg} | {sub} |"
        )

    path = os.path.join(run_dir, "eval_results.md")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    run_idx = _next_run_index()
    run_dir = os.path.join(RUNS_ROOT, str(run_idx))
    os.makedirs(run_dir, exist_ok=True)
    print(f"[score_recommendations] run={run_idx}  → runs/{run_idx}/")

    # Move agent output files into the run subdirectory, then remove root-level
    # copies so the agent cannot read previous recommendations on the next run.
    for ext in ("json", "md"):
        src = os.path.join(SCRIPT_DIR, f"agent_recommendations.{ext}")
        dst = os.path.join(run_dir, f"agent_recommendations.{ext}")
        if os.path.exists(src):
            shutil.move(src, dst)
            print(f"Moved agent_recommendations.{ext} → runs/{run_idx}/")

    with open(os.path.join(run_dir, "agent_recommendations.json")) as f: agent  = json.load(f)
    with open(GROUND_FILE) as f: ground = json.load(f)
    with open(INPUT_FILE)  as f: inp    = json.load(f)

    gt_index    = {s["padded_key"]: s for s in ground["shapes"]}
    input_index = {s["padded_key"]: s for s in inp["shapes"]}

    results = []
    npu_unavailable = False

    for rec in sorted(agent["shapes"], key=lambda x: x["padded_key"]):
        key = rec["padded_key"]
        if key not in gt_index:
            print(f"  WARNING: {key} not in ground truth — skipping")
            continue

        r = score_shape(rec, gt_index[key], input_index.get(key, {}))
        results.append(r)

        if r["npu_status"] == "npu_unavailable":
            print("  NPU unavailable — stopping NPU runs for remaining mismatches.")
            npu_unavailable = True
            # Continue scoring remaining shapes as exact/miss but skip NPU runs
            for rec2 in sorted(agent["shapes"], key=lambda x: x["padded_key"]):
                key2 = rec2["padded_key"]
                if key2 not in gt_index or any(r2["padded_key"] == key2 for r2 in results):
                    continue
                rec_tile  = list(rec2["recommended_tile"])
                best_tile = list(gt_index[key2]["actual_best_tile"])
                results.append({
                    "padded_key": key2,
                    "Mp": gt_index[key2]["Mp"], "Np": gt_index[key2]["Np"], "Kp": gt_index[key2]["Kp"],
                    "recommended_tile": rec_tile,
                    "actual_best_tile": best_tile,
                    "actual_best_avg_us": gt_index[key2]["actual_best_avg_us"],
                    "confidence": rec2.get("confidence", "unknown"),
                    "exact_match": rec_tile == best_tile,
                    "npu_run": False, "agent_avg_us": None, "suboptimality": None,
                    "npu_status": "skipped_npu_unavailable", "npu_error": None,
                })
            break

    n = len(results)
    n_exact  = sum(1 for r in results if r["exact_match"])
    n_runs   = sum(1 for r in results if r["npu_run"])
    n_failed = sum(1 for r in results if r["npu_status"] == "failed")
    n_unavail = sum(1 for r in results if r["npu_status"] == "npu_unavailable")

    sub_vals = [r["suboptimality"] for r in results if r["suboptimality"] is not None]
    avg_sub = round(sum(sub_vals) / len(sub_vals), 4) if sub_vals else None
    max_sub = round(max(sub_vals), 4) if sub_vals else None

    summary = {
        "generated":        datetime.now().isoformat(timespec="seconds"),
        "run":              run_idx,
        "n_shapes":         n,
        "exact_match":      n_exact,
        "exact_match_pct":  round(100 * n_exact / n, 1) if n else 0,
        "npu_runs":         n_runs,
        "npu_failed":       n_failed,
        "npu_unavailable":  n_unavail,
        "avg_suboptimality": avg_sub,
        "max_suboptimality": max_sub,
    }

    write_json(results, summary, run_dir)
    write_md(results, summary, run_dir, run_idx)

    print(f"\n{'='*55}")
    print(f"RUN {run_idx} SUMMARY")
    print(f"{'='*55}")
    print(f"Exact match:      {n_exact}/{n} ({summary['exact_match_pct']}%)")
    print(f"NPU runs:         {n_runs}  (failed: {n_failed})")
    print(f"Avg suboptimality:{avg_sub}×")
    print(f"Max suboptimality:{max_sub}×")


if __name__ == "__main__":
    main()
