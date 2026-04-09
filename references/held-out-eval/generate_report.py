"""
generate_report.py

Generates a self-contained HTML eval report across all completed runs in runs/.

Usage:
  cd /home/ec935/agent-gemm
  python3 references/held-out-eval/generate_report.py

Output: references/held-out-eval/report.html
"""

import json
import os
import sys

import numpy as np
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import plotly.offline as pyo

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RUNS_ROOT  = os.path.join(SCRIPT_DIR, "runs")
OUT_PATH   = os.path.join(SCRIPT_DIR, "report.html")


# ── Data loading ───────────────────────────────────────────────────────────────

def load_runs():
    runs = []
    i = 1
    while True:
        path = os.path.join(RUNS_ROOT, str(i), "eval_results.json")
        if not os.path.exists(path):
            break
        with open(path) as f:
            d = json.load(f)
        d["run_folder"] = i
        runs.append(d)
        i += 1
    if not runs:
        print("No runs found in runs/. Exiting.")
        sys.exit(1)
    print(f"Loaded {len(runs)} run(s): {[r['run_folder'] for r in runs]}")
    return runs


def classify_shape(s):
    """Return 'match', 'close' (≤1.1×), 'far' (>1.1×), or 'failed'."""
    if s["exact_match"]:
        return "match"
    if s["npu_status"] == "failed":
        return "failed"
    sub = s.get("suboptimality")
    if sub is None:
        return "untested"
    return "close" if sub <= 1.10 else "far"


def wrong_dims(rec, best):
    """Return list of which tile dimensions differ: 'm', 'n', 'k'."""
    dims = []
    for label, r, b in zip(["m", "n", "k"], rec, best):
        if r != b:
            dims.append(label)
    return dims


# ── Plot builders ──────────────────────────────────────────────────────────────

COLORS = {
    "match":    "#2ecc71",
    "close":    "#f39c12",
    "far":      "#e74c3c",
    "failed":   "#8e44ad",
    "untested": "#95a5a6",
}

RUN_PALETTE = ["#3498db", "#e67e22", "#9b59b6", "#1abc9c", "#e74c3c"]


def fig_summary_table(runs):
    """Plot 1 — cross-run summary table."""
    headers = ["Run", "Date", "Exact Match", "Exact %",
               "Avg Subopt ×", "Max Subopt ×", "NPU Runs", "Failed"]
    rows = {h: [] for h in headers}
    for r in runs:
        s = r["summary"]
        rows["Run"].append(r["run_folder"])
        rows["Date"].append(s["generated"][:16].replace("T", " "))
        rows["Exact Match"].append(f"{s['exact_match']}/{s['n_shapes']}")
        rows["Exact %"].append(f"{s['exact_match_pct']}%")
        rows["Avg Subopt ×"].append(
            f"{s['avg_suboptimality']:.4f}" if s["avg_suboptimality"] else "—")
        rows["Max Subopt ×"].append(
            f"{s['max_suboptimality']:.4f}" if s["max_suboptimality"] else "—")
        rows["NPU Runs"].append(s["npu_runs"])
        rows["Failed"].append(s["npu_failed"])

    fig = go.Figure(go.Table(
        header=dict(values=headers,
                    fill_color="#2c3e50", font=dict(color="white", size=13),
                    align="center"),
        cells=dict(values=[rows[h] for h in headers],
                   fill_color=[["#ecf0f1" if i % 2 == 0 else "white"
                                 for i in range(len(runs))]],
                   align="center", font=dict(size=12)),
    ))
    fig.update_layout(title="Run Summary", margin=dict(t=40, b=10))
    return fig


def fig_stacked_bar(runs):
    """Plot 2 — stacked bar: match / close / far / failed per run."""
    categories = ["match", "close", "far", "failed", "untested"]
    labels = {
        "match":    "Exact match",
        "close":    "Miss ≤1.1×",
        "far":      "Miss >1.1×",
        "failed":   "NPU failed",
        "untested": "Not tested",
    }
    counts = {c: [] for c in categories}
    run_labels = [f"Run {r['run_folder']}" for r in runs]

    for r in runs:
        cats = [classify_shape(s) for s in r["per_shape"]]
        for c in categories:
            counts[c].append(cats.count(c))

    fig = go.Figure()
    for c in categories:
        fig.add_trace(go.Bar(
            name=labels[c], x=run_labels, y=counts[c],
            marker_color=COLORS[c],
            text=counts[c], textposition="inside",
        ))
    fig.update_layout(
        barmode="stack", title="Shape outcome breakdown per run",
        yaxis_title="Number of shapes", legend_title="Outcome",
        xaxis_tickfont_size=13,
    )
    return fig


def fig_box_strip(runs):
    """Plot 5 — box + strip of suboptimality ratios per run."""
    fig = go.Figure()
    for idx, r in enumerate(runs):
        label = f"Run {r['run_folder']}"
        vals = [s["suboptimality"] for s in r["per_shape"]
                if s.get("suboptimality") is not None]
        keys = [s["padded_key"] for s in r["per_shape"]
                if s.get("suboptimality") is not None]
        color = RUN_PALETTE[idx % len(RUN_PALETTE)]

        fig.add_trace(go.Box(
            y=vals, name=label, marker_color=color,
            boxpoints=False, line_width=2,
        ))
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(vals))
        fig.add_trace(go.Scatter(
            x=[label] * len(vals) + jitter.tolist(),  # jitter ignored on categorical
            y=vals, mode="markers",
            marker=dict(color=color, size=8, opacity=0.7,
                        line=dict(width=1, color="white")),
            name=label, showlegend=False,
            text=keys, hovertemplate="%{text}<br>subopt: %{y:.4f}<extra></extra>",
        ))

    fig.add_hline(y=1.0, line_dash="dot", line_color="green",
                  annotation_text="Optimal (1.0×)")
    fig.add_hline(y=1.1, line_dash="dot", line_color="orange",
                  annotation_text="10% overhead")
    fig.update_layout(
        title="Suboptimality ratio distribution (mismatches with NPU data)",
        yaxis_title="Agent time / Best time",
        showlegend=False,
    )
    return fig


def fig_cdf(runs):
    """Plot 6 — CDF of suboptimality ratios."""
    fig = go.Figure()
    for idx, r in enumerate(runs):
        vals = sorted(s["suboptimality"] for s in r["per_shape"]
                      if s.get("suboptimality") is not None)
        n = len(vals)
        cdf_y = [(i + 1) / n for i in range(n)]
        color = RUN_PALETTE[idx % len(RUN_PALETTE)]
        fig.add_trace(go.Scatter(
            x=vals, y=cdf_y, mode="lines+markers",
            name=f"Run {r['run_folder']}",
            line=dict(color=color, width=2),
            marker=dict(size=6),
        ))

    fig.add_vline(x=1.0, line_dash="dot", line_color="green",
                  annotation_text="1.0×")
    fig.add_vline(x=1.1, line_dash="dot", line_color="orange",
                  annotation_text="1.1×")
    fig.update_layout(
        title="CDF of suboptimality (mismatches with NPU data)",
        xaxis_title="Agent time / Best time",
        yaxis_title="Fraction of mismatched shapes",
        yaxis_tickformat=".0%",
    )
    return fig


def fig_paired_dot(runs):
    """Plot 7 — paired dot plot: agent µs vs best µs per mismatch shape."""
    fig = go.Figure()
    for idx, r in enumerate(runs):
        misses = [s for s in r["per_shape"]
                  if not s["exact_match"] and s.get("agent_avg_us") is not None]
        if not misses:
            continue

        misses_sorted = sorted(misses, key=lambda s: s["actual_best_avg_us"])
        keys = [s["padded_key"] for s in misses_sorted]
        best_vals = [s["actual_best_avg_us"] for s in misses_sorted]
        agent_vals = [s["agent_avg_us"] for s in misses_sorted]
        color = RUN_PALETTE[idx % len(RUN_PALETTE)]
        label = f"Run {r['run_folder']}"

        # Connecting lines
        for k, b, a in zip(keys, best_vals, agent_vals):
            fig.add_trace(go.Scatter(
                x=[b, a], y=[k, k],
                mode="lines",
                line=dict(color=color, width=1.5),
                showlegend=False,
                hoverinfo="skip",
            ))
        fig.add_trace(go.Scatter(
            x=best_vals, y=keys, mode="markers",
            name=f"{label} — best tile",
            marker=dict(color=color, size=10, symbol="circle",
                        line=dict(width=1.5, color="white")),
            text=[f"{k}<br>Best tile: {s['actual_best_tile']}<br>{b:.1f}µs"
                  for k, s, b in zip(keys, misses_sorted, best_vals)],
            hovertemplate="%{text}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=agent_vals, y=keys, mode="markers",
            name=f"{label} — agent tile",
            marker=dict(color=color, size=10, symbol="diamond",
                        line=dict(width=1.5, color="white")),
            text=[f"{k}<br>Agent tile: {s['recommended_tile']}<br>{a:.1f}µs"
                  for k, s, a in zip(keys, misses_sorted, agent_vals)],
            hovertemplate="%{text}<extra></extra>",
        ))

    fig.update_layout(
        title="Paired timing: agent tile vs best tile (mismatches only)",
        xaxis_title="Execution time (µs)",
        height=max(400, len(misses_sorted) * 28 + 100),
        legend_tracegroupgap=5,
    )
    return fig


def fig_scatter_timing(runs):
    """Plot 8 — scatter: agent µs vs best µs."""
    fig = go.Figure()
    all_vals = []

    for idx, r in enumerate(runs):
        misses = [s for s in r["per_shape"]
                  if not s["exact_match"] and s.get("agent_avg_us") is not None]
        if not misses:
            continue
        best_vals  = [s["actual_best_avg_us"] for s in misses]
        agent_vals = [s["agent_avg_us"] for s in misses]
        all_vals.extend(best_vals + agent_vals)
        color = RUN_PALETTE[idx % len(RUN_PALETTE)]

        fig.add_trace(go.Scatter(
            x=best_vals, y=agent_vals, mode="markers",
            name=f"Run {r['run_folder']}",
            marker=dict(color=color, size=10,
                        line=dict(width=1, color="white")),
            text=[f"{s['padded_key']}<br>rec={s['recommended_tile']}<br>"
                  f"best={s['actual_best_tile']}<br>"
                  f"agent={s['agent_avg_us']:.1f}µs  best={s['actual_best_avg_us']:.1f}µs<br>"
                  f"subopt={s['suboptimality']:.3f}×" for s in misses],
            hovertemplate="%{text}<extra></extra>",
        ))

    if all_vals:
        lo, hi = min(all_vals) * 0.9, max(all_vals) * 1.1
        fig.add_trace(go.Scatter(
            x=[lo, hi], y=[lo, hi], mode="lines",
            line=dict(color="green", dash="dot", width=1.5),
            name="y = x (perfect)", showlegend=True,
        ))
        fig.update_layout(
            xaxis_range=[lo, hi], yaxis_range=[lo, hi],
        )

    fig.update_layout(
        title="Agent timing vs best timing (mismatches only)",
        xaxis_title="Best tile time (µs)",
        yaxis_title="Agent tile time (µs)",
    )
    return fig


def fig_timing_delta(runs):
    """Plot 9 — bar chart of (agent_us - best_us) per shape, sorted by delta."""
    fig = go.Figure()
    for idx, r in enumerate(runs):
        misses = [s for s in r["per_shape"]
                  if not s["exact_match"] and s.get("agent_avg_us") is not None]
        if not misses:
            continue
        misses_sorted = sorted(misses,
                               key=lambda s: s["agent_avg_us"] - s["actual_best_avg_us"],
                               reverse=True)
        keys   = [s["padded_key"] for s in misses_sorted]
        deltas = [s["agent_avg_us"] - s["actual_best_avg_us"] for s in misses_sorted]
        color  = RUN_PALETTE[idx % len(RUN_PALETTE)]

        fig.add_trace(go.Bar(
            x=keys, y=deltas,
            name=f"Run {r['run_folder']}",
            marker_color=color,
            text=[f"+{d:.1f}µs" if d > 0 else f"{d:.1f}µs" for d in deltas],
            textposition="outside",
            hovertemplate="%{x}<br>Δ = %{y:.1f}µs<extra></extra>",
        ))

    fig.add_hline(y=0, line_color="black", line_width=1)
    fig.update_layout(
        title="Timing overhead: agent tile vs best tile (Δ µs, mismatches only)",
        xaxis_title="Shape", yaxis_title="Agent time − Best time (µs)",
        barmode="group", xaxis_tickangle=-45,
    )
    return fig


def fig_heatmap_correctness(runs):
    """Plot 10 — heatmap of match rate by (Mp, Kp)."""
    # Collect all Mp/Kp combos
    all_mp = sorted({s["Mp"] for r in runs for s in r["per_shape"]})
    all_kp = sorted({s["Kp"] for r in runs for s in r["per_shape"]})

    fig = make_subplots(
        rows=1, cols=len(runs),
        subplot_titles=[f"Run {r['run_folder']}" for r in runs],
        shared_yaxes=True,
    )

    for col_idx, r in enumerate(runs, start=1):
        # Build match rate grid
        z = []
        text = []
        for mp in all_mp:
            row_z = []
            row_t = []
            for kp in all_kp:
                shapes = [s for s in r["per_shape"]
                          if s["Mp"] == mp and s["Kp"] == kp]
                if not shapes:
                    row_z.append(None)
                    row_t.append("")
                else:
                    rate = sum(1 for s in shapes if s["exact_match"]) / len(shapes)
                    row_z.append(rate)
                    row_t.append(f"{int(rate*100)}%<br>n={len(shapes)}")
            z.append(row_z)
            text.append(row_t)

        fig.add_trace(go.Heatmap(
            z=z,
            x=[str(kp) for kp in all_kp],
            y=[str(mp) for mp in all_mp],
            text=text,
            texttemplate="%{text}",
            colorscale="RdYlGn",
            zmin=0, zmax=1,
            showscale=(col_idx == len(runs)),
            colorbar=dict(title="Match rate", tickformat=".0%"),
        ), row=1, col=col_idx)

    fig.update_layout(
        title="Exact match rate by (Mp, Kp)",
        height=350,
    )
    fig.update_xaxes(title_text="Kp")
    fig.update_yaxes(title_text="Mp", col=1)
    return fig


def fig_scatter_mp_np(runs):
    """Plot 11 — scatter Mp vs Np colored by match/miss per run."""
    fig = make_subplots(
        rows=1, cols=len(runs),
        subplot_titles=[f"Run {r['run_folder']}" for r in runs],
        shared_xaxes=True, shared_yaxes=True,
    )

    for col_idx, r in enumerate(runs, start=1):
        for cat, color in COLORS.items():
            shapes = [s for s in r["per_shape"] if classify_shape(s) == cat]
            if not shapes:
                continue
            fig.add_trace(go.Scatter(
                x=[s["Mp"] for s in shapes],
                y=[s["Np"] for s in shapes],
                mode="markers",
                name=cat,
                marker=dict(color=color, size=10,
                            line=dict(width=1, color="white")),
                showlegend=(col_idx == 1),
                text=[f"{s['padded_key']}<br>rec={s['recommended_tile']}<br>"
                      f"best={s['actual_best_tile']}" for s in shapes],
                hovertemplate="%{text}<extra></extra>",
            ), row=1, col=col_idx)

    fig.update_layout(
        title="Shape space: Mp vs Np colored by outcome",
        legend_title="Outcome",
        height=420,
    )
    fig.update_xaxes(title_text="Mp")
    fig.update_yaxes(title_text="Np", col=1)
    return fig


def fig_wrong_dims(runs):
    """Plot 13 — which tile dimension(s) were wrong per run."""
    dim_combos = ["m only", "n only", "k only", "m+n", "m+k", "n+k", "m+n+k"]
    combo_map = {
        frozenset(["m"]):       "m only",
        frozenset(["n"]):       "n only",
        frozenset(["k"]):       "k only",
        frozenset(["m", "n"]):  "m+n",
        frozenset(["m", "k"]):  "m+k",
        frozenset(["n", "k"]):  "n+k",
        frozenset(["m","n","k"]):"m+n+k",
    }

    fig = go.Figure()
    for idx, r in enumerate(runs):
        counts = {c: 0 for c in dim_combos}
        for s in r["per_shape"]:
            if s["exact_match"]:
                continue
            dims = wrong_dims(s["recommended_tile"], s["actual_best_tile"])
            key = frozenset(dims)
            label = combo_map.get(key, str(sorted(dims)))
            if label in counts:
                counts[label] += 1

        color = RUN_PALETTE[idx % len(RUN_PALETTE)]
        fig.add_trace(go.Bar(
            name=f"Run {r['run_folder']}",
            x=dim_combos,
            y=[counts[c] for c in dim_combos],
            marker_color=color,
            text=[counts[c] if counts[c] > 0 else "" for c in dim_combos],
            textposition="outside",
        ))

    fig.update_layout(
        title="Which tile dimension(s) were wrong (mismatches only)",
        xaxis_title="Wrong dimension(s)",
        yaxis_title="Number of shapes",
        barmode="group",
    )
    return fig


def fig_consistency_matrix(runs):
    """Plot 16 — shape-level consistency matrix across runs."""
    all_keys = sorted({s["padded_key"] for r in runs for s in r["per_shape"]})
    run_labels = [f"Run {r['run_folder']}" for r in runs]

    # Build z as numeric (1=match, 0=miss, 0.5=failed/untested), text for hover
    z = []
    hover = []
    for key in all_keys:
        row_z = []
        row_h = []
        for r in runs:
            s_map = {s["padded_key"]: s for s in r["per_shape"]}
            s = s_map.get(key)
            if s is None:
                row_z.append(None)
                row_h.append("—")
            elif s["exact_match"]:
                row_z.append(1.0)
                row_h.append(f"✓ {s['recommended_tile']}")
            elif s.get("suboptimality") is not None:
                row_z.append(1.0 - min(s["suboptimality"] - 1.0, 0.5))
                row_h.append(f"✗ {s['recommended_tile']} → best {s['actual_best_tile']}<br>"
                             f"subopt {s['suboptimality']:.3f}×")
            else:
                row_z.append(0.3)
                row_h.append(f"✗ {s['recommended_tile']} → best {s['actual_best_tile']}<br>"
                             f"({s.get('npu_status', '?')})")
        z.append(row_z)
        hover.append(row_h)

    fig = go.Figure(go.Heatmap(
        z=z,
        x=run_labels,
        y=all_keys,
        text=hover,
        texttemplate="",
        hovertemplate="%{y}  %{x}<br>%{text}<extra></extra>",
        colorscale=[
            [0.0, "#e74c3c"],
            [0.5, "#f39c12"],
            [1.0, "#2ecc71"],
        ],
        zmin=0, zmax=1,
        showscale=True,
        colorbar=dict(
            title="Outcome",
            tickvals=[0.15, 0.5, 1.0],
            ticktext=["Miss / failed", "Near miss", "Exact match"],
        ),
    ))
    fig.update_layout(
        title="Per-shape consistency across runs",
        xaxis_title="Run",
        yaxis_title="Shape (padded key)",
        height=max(600, len(all_keys) * 14 + 100),
        yaxis=dict(autorange="reversed", tickfont=dict(size=10)),
    )
    return fig


def fig_recommendation_consistency(runs):
    """Table + bar: which tile did the agent pick per run, and do they agree?"""
    if len(runs) < 2:
        return None

    all_keys = sorted({s["padded_key"] for r in runs for s in r["per_shape"]})
    run_labels = [f"Run {r['run_folder']}" for r in runs]

    # Build per-shape rec lookup
    rec = {}  # key -> {run_folder: tile_str}
    for r in runs:
        for s in r["per_shape"]:
            rec.setdefault(s["padded_key"], {})[r["run_folder"]] = str(s["recommended_tile"])

    # Compute agreement
    all_same, some_diff, missing = [], [], []
    for key in all_keys:
        tiles = [rec[key].get(r["run_folder"]) for r in runs]
        if None in tiles:
            missing.append(key)
        elif len(set(tiles)) == 1:
            all_same.append(key)
        else:
            some_diff.append(key)

    n = len(all_keys)
    pct = round(100 * len(all_same) / n, 1) if n else 0

    # ── Part 1: summary bar ──
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.28, 0.72],
        subplot_titles=[
            f"Agreement summary ({pct}% fully consistent)",
            "Per-shape recommendation across runs",
        ],
        specs=[[{"type": "bar"}, {"type": "table"}]],
    )

    fig.add_trace(go.Bar(
        x=["All agree", "Some differ", "Missing data"],
        y=[len(all_same), len(some_diff), len(missing)],
        marker_color=["#2ecc71", "#e74c3c", "#95a5a6"],
        text=[len(all_same), len(some_diff), len(missing)],
        textposition="outside",
        showlegend=False,
    ), row=1, col=1)

    # ── Part 2: detail table ──
    # Columns: shape | run1_tile | run2_tile | ... | consistent?
    tile_cols = [[rec[key].get(r["run_folder"], "—") for key in all_keys]
                 for r in runs]
    agree_col = []
    agree_colors = []
    for key in all_keys:
        tiles = [rec[key].get(r["run_folder"]) for r in runs]
        if None in tiles:
            agree_col.append("?")
            agree_colors.append("#ecf0f1")
        elif len(set(tiles)) == 1:
            agree_col.append("✓")
            agree_colors.append("#d5f5e3")
        else:
            agree_col.append("✗")
            agree_colors.append("#fadbd8")

    # Row background: alternate for readability, but override agree column
    row_bg = ["#ecf0f1" if i % 2 == 0 else "white" for i in range(len(all_keys))]

    header_vals = ["Shape"] + run_labels + ["Consistent?"]
    cell_vals   = [all_keys] + tile_cols + [agree_col]
    cell_colors = [row_bg] + [row_bg for _ in runs] + [agree_colors]

    fig.add_trace(go.Table(
        header=dict(
            values=header_vals,
            fill_color="#2c3e50", font=dict(color="white", size=12),
            align="center",
        ),
        cells=dict(
            values=cell_vals,
            fill_color=cell_colors,
            align="center", font=dict(size=11),
        ),
    ), row=1, col=2)

    fig.update_layout(
        title=f"Agent recommendation consistency across runs — {len(all_same)}/{n} shapes always agree ({pct}%)",
        height=max(500, len(all_keys) * 22 + 120),
    )
    fig.update_yaxes(title_text="Shapes", row=1, col=1)
    return fig


# ── HTML assembly ──────────────────────────────────────────────────────────────

SECTION_STYLE = """
    <div style="margin: 40px 0 10px 0; padding: 8px 16px;
                background:#2c3e50; color:white; border-radius:4px;
                font-family:sans-serif; font-size:1.1em; font-weight:bold;">
        {title}
    </div>
"""

def section(title):
    return SECTION_STYLE.format(title=title)


def build_html(runs):
    plots = [
        ("1 — Run summary table",                 fig_summary_table(runs)),
        ("2 — Outcome breakdown per run",          fig_stacked_bar(runs)),
        ("5 — Suboptimality distribution",         fig_box_strip(runs)),
        ("6 — CDF of suboptimality",               fig_cdf(runs)),
        ("7 — Paired timing: agent vs best",       fig_paired_dot(runs)),
        ("8 — Scatter: agent µs vs best µs",       fig_scatter_timing(runs)),
        ("9 — Timing overhead Δµs",                fig_timing_delta(runs)),
        ("10 — Match rate heatmap (Mp × Kp)",      fig_heatmap_correctness(runs)),
        ("11 — Shape space (Mp vs Np)",            fig_scatter_mp_np(runs)),
        ("13 — Wrong tile dimensions",             fig_wrong_dims(runs)),
        ("16 — Per-shape consistency matrix",      fig_consistency_matrix(runs)),
        ("Recommendation consistency across runs", fig_recommendation_consistency(runs)),
    ]

    # Render each figure to an HTML div (no full page, include plotly once)
    plotly_js = pyo.get_plotlyjs()
    divs = []
    for title, fig in plots:
        if fig is None:
            continue
        div_html = fig.to_html(full_html=False, include_plotlyjs=False)
        divs.append(section(title) + div_html)

    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>GEMM NPU Agent Held-Out Eval Report</title>
  <script>{plotly_js}</script>
  <style>
    body {{ font-family: sans-serif; max-width: 1400px; margin: 0 auto; padding: 20px; background: #f8f9fa; }}
    h1   {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
    p.meta {{ color: #7f8c8d; font-size: 0.9em; }}
  </style>
</head>
<body>
  <h1>GEMM NPU Agent — Held-Out Eval Report</h1>
  <p class="meta">
    {len(runs)} run(s) evaluated &nbsp;|&nbsp;
    50 held-out bf16 shapes &nbsp;|&nbsp;
    Profiling DB: npu_execution_profiling_reduced.json
  </p>
  {''.join(divs)}
</body>
</html>"""
    return html


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    runs = load_runs()
    html = build_html(runs)
    with open(OUT_PATH, "w") as f:
        f.write(html)
    print(f"\nReport written to: {OUT_PATH}")
    print(f"Open with:  xdg-open {OUT_PATH}")


if __name__ == "__main__":
    main()
