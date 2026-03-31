#!/usr/bin/env python3
"""
NPU GEMM Cost Regression Model
================================
Predicts NPU execution time for any (M, N, K, dtype) + tile (m, n, k) combination,
then recommends the best tile for unseen shapes.

Implemented with numpy only (no sklearn/xgboost required).

Model: Ridge regression on hand-crafted log-scale features + interaction terms.
Evaluated via:
  1. Random 80/20 holdout — R², MAE, MAPE
  2. Leave-one-shape-out — "best tile accuracy" (did we recommend the right tile?)

Usage
-----
  # Train and save model
  python npu_cost_model.py --train

  # Evaluate model accuracy
  python npu_cost_model.py --evaluate

  # Predict best tile for a shape
  python npu_cost_model.py --predict --M 128 --N 960 --K 960 --dtype bf16

  # All three at once
  python npu_cost_model.py --train --evaluate --predict --M 128 --N 960 --K 960 --dtype bf16

Paths (relative to project root /home/ec935/agent-gemm)
-----
  Input:  references/memory/gemm-data/npu_execution_profiling.json
  Output: references/cost_model/model.npz
          references/cost_model/evaluation_report.md
"""

import argparse
import json
import math
import os
import sys
import numpy as np
from datetime import datetime

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
PROFILING_JSON = os.path.join(PROJECT_ROOT, "references", "memory", "gemm-data",
                               "npu_execution_profiling.json")
MODEL_FILE     = os.path.join(SCRIPT_DIR, "model.npz")
EVAL_REPORT    = os.path.join(SCRIPT_DIR, "evaluation_report.md")

# ── Hardware constants ─────────────────────────────────────────────────────────
ALLOWED_M = [32, 64, 128, 256, 512, 960, 1024, 2048, 3072]
ALLOWED_N = [32, 64, 128, 256, 512, 768, 960, 1024, 2048, 3072]
ALLOWED_K = [32, 64, 128, 256, 512, 768, 1024, 2048, 3072]
TILE_POOL  = [16, 32, 64, 128, 256, 512]
MAX_TILE_MEM_BYTES = 32256
DSIZE = {"i8": 1, "i16": 2, "bf16": 2}
# Dimensions where the strict power-of-2 quotient rule is relaxed (empirically confirmed)
DIVISIBILITY_ONLY_DIMS = {768, 3072}

DTYPE_ENC = {"i8": 0, "i16": 1, "bf16": 2}

# ── Tile validity ──────────────────────────────────────────────────────────────

def valid_tiles_for_dim(Dp: int, is_K: bool = False) -> list:
    """Return all valid tile values for a padded dimension.

    K dimension: only divisibility required (Kp % k == 0).
    M/N dimensions: bundling constraint — (Dp//t) must be divisible by
    at least one col_num/row_num ∈ {3, 4, 5}.
    """
    result = []
    for t in TILE_POOL:
        if t > Dp or Dp % t != 0:
            continue
        if is_K:
            result.append(t)
        else:
            q = Dp // t
            # q < 3: GEMM auto-adjusts col_num/row_num down to q → always valid
            if q < 3 or any(q % c == 0 for c in (3, 4, 5)):
                result.append(t)
    return result

def enumerate_valid_tiles(Mp: int, Np: int, Kp: int, dtype: str) -> list:
    """Enumerate all valid (m, n, k) tiles for a padded shape + dtype."""
    ds = DSIZE[dtype]
    vm = valid_tiles_for_dim(Mp, is_K=False)
    vn = valid_tiles_for_dim(Np, is_K=False)
    vk = valid_tiles_for_dim(Kp, is_K=True)
    candidates = []
    for m in vm:
        for n in vn:
            for k in vk:
                if (m, n, k) == (128, 128, 128):
                    continue  # hard blacklist
                mem = (m * n + n * k + k * m) * ds
                if mem <= MAX_TILE_MEM_BYTES:
                    candidates.append((m, n, k))
    return candidates

def pad_dim(v: int, allowed: list) -> int:
    """Return smallest allowed value >= v."""
    for a in allowed:
        if a >= v:
            return a
    raise ValueError(f"Value {v} exceeds max allowed {allowed[-1]}")

def compute_padded(M, N, K):
    return pad_dim(M, ALLOWED_M), pad_dim(N, ALLOWED_N), pad_dim(K, ALLOWED_K)

# ── Feature engineering ────────────────────────────────────────────────────────

def make_features(Mp, Np, Kp, m, n, k, dtype) -> np.ndarray:
    """
    Compute feature vector for one (shape, tile, dtype) observation.

    Features (19 total):
      Shape:       log(Mp), log(Np), log(Kp)
      Tile:        log(m),  log(n),  log(k)
      Tiles/dim:   log(Mp/m), log(Np/n), log(Kp/k)
      Compute:     log(Mp*Np*Kp)
      Tile volume: log(m*n*k)
      N tiles:     log(Mp*Np*Kp / (m*n*k))
      Tile mem:    (m*n + n*k + k*m)*dsize / 32256   [0..1]
      dtype:       dtype_enc / 2                      [0, 0.5, 1]
      Symmetry:    float(m==n and n==k)
      N/M ratio:   log(n/m)
      K/N ratio:   log(k/n)
      Interaction: log(Mp*Np*Kp) * log(m*n*k) / 100  [scaled cross-term]
      Shape ratio: log(max(Mp,Np,Kp) / min(Mp,Np,Kp))
    """
    ds = DSIZE[dtype]
    log = math.log

    lMp, lNp, lKp  = log(Mp), log(Np), log(Kp)
    lm,  ln,  lk   = log(m),  log(n),  log(k)
    lMm, lNn, lKk  = log(Mp / m), log(Np / n), log(Kp / k)
    compute        = Mp * Np * Kp
    tile_vol       = m * n * k
    n_tiles        = compute / tile_vol
    tile_mem_frac  = (m * n + n * k + k * m) * ds / MAX_TILE_MEM_BYTES
    dtype_enc      = DTYPE_ENC[dtype] / 2.0
    is_square      = float(m == n == k)
    log_n_m        = log(n / m)
    log_k_n        = log(k / n)
    interaction    = log(compute) * log(tile_vol) / 100.0
    shape_ratio    = log(max(Mp, Np, Kp) / min(Mp, Np, Kp))

    return np.array([
        lMp, lNp, lKp,
        lm,  ln,  lk,
        lMm, lNn, lKk,
        log(compute), log(tile_vol), log(n_tiles),
        tile_mem_frac, dtype_enc,
        is_square, log_n_m, log_k_n,
        interaction, shape_ratio,
    ], dtype=np.float64)

def make_feature_names() -> list:
    return [
        "log_Mp", "log_Np", "log_Kp",
        "log_m",  "log_n",  "log_k",
        "log_Mp/m", "log_Np/n", "log_Kp/k",
        "log_compute", "log_tile_vol", "log_n_tiles",
        "tile_mem_frac", "dtype_enc",
        "is_square", "log_n/m", "log_k/n",
        "interaction", "shape_ratio",
    ]

# ── Data loading ───────────────────────────────────────────────────────────────

def load_records(profiling_path: str) -> list:
    """
    Parse npu_execution_profiling.json into a flat list of records:
      {"Mp", "Np", "Kp", "m", "n", "k", "dtype", "avg_us", "shape_key"}
    Only includes entries with avg > 0 (real measurements).
    """
    with open(profiling_path) as f:
        data = json.load(f)

    records = []
    for shape_key, shape_data in data.items():
        parts = shape_key.split("_")
        if len(parts) != 3:
            continue
        try:
            Mp, Np, Kp = int(parts[0]), int(parts[1]), int(parts[2])
        except ValueError:
            continue

        for dtype, entry in shape_data.items():
            if not isinstance(entry, dict):
                continue
            working = entry.get("Working m,n,k", {})
            if not isinstance(working, dict):
                continue
            for tile_key, tile_data in working.items():
                if not isinstance(tile_data, dict):
                    continue
                avg = tile_data.get("avg", 0)
                size = tile_data.get("size", [])
                if avg <= 0 or len(size) != 3:
                    continue
                m, n, k = size
                try:
                    feat = make_features(Mp, Np, Kp, m, n, k, dtype)
                    if not np.all(np.isfinite(feat)):
                        continue
                except Exception:
                    continue
                records.append({
                    "Mp": Mp, "Np": Np, "Kp": Kp,
                    "m": m, "n": n, "k": k,
                    "dtype": dtype,
                    "avg_us": avg,
                    "shape_key": shape_key,
                })
    return records

# ── Model: Ridge regression via numpy ─────────────────────────────────────────

def ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float = 1.0):
    """
    Fit Ridge regression: min ||Xw - y||^2 + alpha*||w||^2
    Solved via augmented system using numpy.linalg.lstsq.
    Returns (weights, feature_mean, feature_std, y_mean, y_std).
    """
    # Standardise features
    feat_mean = X.mean(axis=0)
    feat_std  = X.std(axis=0) + 1e-8
    Xs = (X - feat_mean) / feat_std

    # Standardise target (fit on log scale)
    y_mean = y.mean()
    y_std  = y.std() + 1e-8
    ys = (y - y_mean) / y_std

    n, d = Xs.shape
    # Augment with L2 regularisation rows
    reg = np.sqrt(alpha) * np.eye(d)
    X_aug = np.vstack([Xs, reg])
    y_aug = np.concatenate([ys, np.zeros(d)])

    w, _, _, _ = np.linalg.lstsq(X_aug, y_aug, rcond=None)
    return w, feat_mean, feat_std, y_mean, y_std

def ridge_predict(X: np.ndarray, w, feat_mean, feat_std, y_mean, y_std) -> np.ndarray:
    Xs = (X - feat_mean) / feat_std
    return Xs @ w * y_std + y_mean

def save_model(path: str, w, feat_mean, feat_std, y_mean, y_std, alpha: float):
    np.savez(path,
             w=w, feat_mean=feat_mean, feat_std=feat_std,
             y_mean=np.array([y_mean]), y_std=np.array([y_std]),
             alpha=np.array([alpha]))
    print(f"Model saved → {path}")

def load_model(path: str):
    d = np.load(path)
    return (d["w"], d["feat_mean"], d["feat_std"],
            float(d["y_mean"].item()), float(d["y_std"].item()), float(d["alpha"].item()))

# ── Training ───────────────────────────────────────────────────────────────────

def train(alpha: float = 1.0, verbose: bool = True):
    records = load_records(PROFILING_JSON)
    if not records:
        print("ERROR: No records loaded from profiling JSON.")
        sys.exit(1)

    X = np.array([make_features(r["Mp"], r["Np"], r["Kp"],
                                r["m"],  r["n"],  r["k"], r["dtype"])
                  for r in records])
    y = np.log(np.array([r["avg_us"] for r in records]))  # fit on log scale

    w, feat_mean, feat_std, y_mean, y_std = ridge_fit(X, y, alpha=alpha)
    save_model(MODEL_FILE, w, feat_mean, feat_std, y_mean, y_std, alpha)

    if verbose:
        # Quick train-set R²
        y_pred = ridge_predict(X, w, feat_mean, feat_std, y_mean, y_std)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot
        mae_log = np.mean(np.abs(y - y_pred))
        # Back to µs
        avg_pred = np.exp(y_pred)
        avg_true = np.exp(y)
        mape = np.mean(np.abs(avg_pred - avg_true) / avg_true) * 100
        print(f"Training set  — R²={r2:.4f}  MAE(log)={mae_log:.4f}  MAPE={mape:.1f}%")
        print(f"Records used: {len(records)}  Features: {X.shape[1]}")
        print(f"Shapes covered: {len(set(r['shape_key'] for r in records))}")

    return w, feat_mean, feat_std, y_mean, y_std

# ── Prediction ─────────────────────────────────────────────────────────────────

def predict_best_tile(M: int, N: int, K: int, dtype: str,
                      model=None, top_k: int = 5,
                      verbose: bool = True) -> dict:
    """
    Given raw (M, N, K, dtype), enumerate all valid tiles for the padded shape
    and return the predicted best tile plus top-k ranked candidates.

    Returns a dict with keys:
      padded, best_tile, predicted_us, top_candidates,
      profiling_best (if data exists), source ("profiling"|"model")
    """
    # Load model if not provided
    if model is None:
        if not os.path.exists(MODEL_FILE):
            print("Model not found. Training now...")
            model = train(verbose=False)
        else:
            model = load_model(MODEL_FILE)
    w, feat_mean, feat_std, y_mean, y_std = model[:5]

    Mp, Np, Kp = compute_padded(M, N, K)
    candidates = enumerate_valid_tiles(Mp, Np, Kp, dtype)

    if not candidates:
        return {"error": f"No valid tiles for ({Mp},{Np},{Kp}) {dtype}"}

    # Check profiling DB first
    shape_key = f"{Mp}_{Np}_{Kp}"
    profiling_best = None
    with open(PROFILING_JSON) as f:
        prof = json.load(f)
    if shape_key in prof and dtype in prof[shape_key]:
        best_entry = prof[shape_key][dtype].get("Best m,n,k", {})
        if best_entry and best_entry.get("size"):
            profiling_best = {
                "tile": tuple(best_entry["size"]),
                "avg_us": best_entry.get("Best NPU average time"),
            }

    # Predict for all candidates
    feats = np.array([make_features(Mp, Np, Kp, m, n, k, dtype)
                      for m, n, k in candidates])
    log_pred = ridge_predict(feats, w, feat_mean, feat_std, y_mean, y_std)
    pred_us  = np.exp(log_pred)

    order    = np.argsort(pred_us)
    ranked   = [(candidates[i], float(pred_us[i])) for i in order]
    best_tile, best_us = ranked[0]

    source = "profiling" if profiling_best else "model"

    if verbose:
        print(f"\n{'='*60}")
        print(f"  NPU TILE RECOMMENDATION")
        print(f"  Input:   M={M}  N={N}  K={K}  dtype={dtype}")
        print(f"  Padded:  Mp={Mp}  Np={Np}  Kp={Kp}")
        print(f"  Candidates evaluated: {len(candidates)}")
        print(f"{'='*60}")
        if profiling_best:
            pt = profiling_best['tile']
            pu = profiling_best['avg_us']
            print(f"  Profiling best:  tile={pt}  measured_avg={pu:.1f}µs  [USE THIS]")
        print(f"  Model best:      tile={best_tile}  predicted={best_us:.1f}µs")
        print(f"  Source:          {source}")
        print(f"\n  Top {min(top_k, len(ranked))} model predictions:")
        for i, (tile, us) in enumerate(ranked[:top_k]):
            marker = " ◄ best" if i == 0 else ""
            print(f"    {i+1}. tile={tile}  predicted={us:.1f}µs{marker}")
        print(f"{'='*60}\n")

    return {
        "M": M, "N": N, "K": K, "dtype": dtype,
        "padded": (Mp, Np, Kp),
        "best_tile": best_tile if not profiling_best else profiling_best["tile"],
        "best_tile_model": best_tile,
        "predicted_us": best_us,
        "profiling_best": profiling_best,
        "source": source,
        "top_candidates": ranked[:top_k],
    }

# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate(alpha: float = 1.0, n_loso_shapes: int = 50, seed: int = 42):
    """
    Two evaluation modes:
      1. Random 80/20 holdout on all (shape, tile) entries → R², MAE, MAPE
      2. Leave-one-shape-out (LOSO) on `n_loso_shapes` random shapes →
         "best tile hit rate" (did the model recommend the same best tile as profiling?)
    Writes evaluation_report.md.
    """
    rng = np.random.default_rng(seed)
    records = load_records(PROFILING_JSON)
    print(f"Loaded {len(records)} records from profiling DB")

    X_all = np.array([make_features(r["Mp"], r["Np"], r["Kp"],
                                    r["m"],  r["n"],  r["k"], r["dtype"])
                      for r in records])
    y_all = np.log(np.array([r["avg_us"] for r in records]))

    # ── 1. Random 80/20 holdout ────────────────────────────────────────────────
    idx = rng.permutation(len(records))
    split = int(0.8 * len(idx))
    tr, te = idx[:split], idx[split:]

    X_tr, y_tr = X_all[tr], y_all[tr]
    X_te, y_te = X_all[te], y_all[te]

    w, feat_mean, feat_std, y_mean, y_std = ridge_fit(X_tr, y_tr, alpha=alpha)
    y_pred = ridge_predict(X_te, w, feat_mean, feat_std, y_mean, y_std)

    ss_res = np.sum((y_te - y_pred) ** 2)
    ss_tot = np.sum((y_te - y_te.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    mae_log = float(np.mean(np.abs(y_te - y_pred)))
    avg_pred = np.exp(y_pred)
    avg_true = np.exp(y_te)
    mape = float(np.mean(np.abs(avg_pred - avg_true) / avg_true) * 100)
    mae_us  = float(np.mean(np.abs(avg_pred - avg_true)))

    print(f"\n── Holdout evaluation (80/20 split, {len(te)} test records) ──")
    print(f"  R²:      {r2:.4f}  (1.0 = perfect)")
    print(f"  MAE(µs): {mae_us:.1f}µs")
    print(f"  MAPE:    {mape:.1f}%")

    # Per-dtype breakdown
    dtype_results = {}
    for dtype in ["i8", "i16", "bf16"]:
        mask = np.array([records[i]["dtype"] == dtype for i in te])
        if mask.sum() == 0:
            continue
        yt = y_te[mask]
        yp = y_pred[mask]
        ss_r = np.sum((yt - yp) ** 2)
        ss_t = np.sum((yt - yt.mean()) ** 2)
        r2_d = 1 - ss_r / ss_t if ss_t > 0 else float("nan")
        mape_d = float(np.mean(np.abs(np.exp(yp) - np.exp(yt)) / np.exp(yt)) * 100)
        mae_d  = float(np.mean(np.abs(np.exp(yp) - np.exp(yt))))
        dtype_results[dtype] = {"r2": r2_d, "mape": mape_d, "mae_us": mae_d, "n": int(mask.sum())}
        print(f"  {dtype:>4}:  R²={r2_d:.4f}  MAE={mae_d:.1f}µs  MAPE={mape_d:.1f}%  (n={mask.sum()})")

    # ── 2. Leave-one-shape-out (LOSO) best-tile accuracy ──────────────────────
    # Group records by (shape_key, dtype)
    groups = {}
    for i, r in enumerate(records):
        key = (r["shape_key"], r["dtype"])
        groups.setdefault(key, []).append(i)

    # Only evaluate groups with ≥3 tile measurements (need meaningful "best tile")
    eligible = [(k, v) for k, v in groups.items() if len(v) >= 3]
    rng.shuffle(eligible)
    loso_sample = eligible[:n_loso_shapes]

    hits = 0
    top3_hits = 0
    loso_details = []
    print(f"\n── Leave-one-shape-out (LOSO) on {len(loso_sample)} shapes ──")

    for (shape_key, dtype), indices in loso_sample:
        # Train on everything except this shape+dtype
        train_mask = np.ones(len(records), dtype=bool)
        for i in indices:
            train_mask[i] = False

        X_tr2 = X_all[train_mask]
        y_tr2 = y_all[train_mask]
        w2, fm2, fs2, ym2, ys2 = ridge_fit(X_tr2, y_tr2, alpha=alpha)

        # True best tile (lowest avg_us in profiling)
        group_records = [records[i] for i in indices]
        true_best = min(group_records, key=lambda r: r["avg_us"])
        true_best_tile = (true_best["m"], true_best["n"], true_best["k"])

        # Predicted best tile for this shape
        X_test = X_all[np.array(indices)]
        y_pred2 = ridge_predict(X_test, w2, fm2, fs2, ym2, ys2)
        best_idx = int(np.argmin(y_pred2))
        pred_best_tile = (group_records[best_idx]["m"],
                          group_records[best_idx]["n"],
                          group_records[best_idx]["k"])

        # Top-3 predicted tiles
        top3_idx = np.argsort(y_pred2)[:3]
        top3_tiles = [(group_records[i]["m"],
                       group_records[i]["n"],
                       group_records[i]["k"]) for i in top3_idx]

        hit   = pred_best_tile == true_best_tile
        top3h = true_best_tile in top3_tiles
        hits     += int(hit)
        top3_hits += int(top3h)

        loso_details.append({
            "shape_key": shape_key,
            "dtype": dtype,
            "n_tiles": len(indices),
            "true_best": true_best_tile,
            "true_best_us": true_best["avg_us"],
            "pred_best": pred_best_tile,
            "pred_best_us": group_records[best_idx]["avg_us"],  # actual time of pred tile
            "hit": hit,
            "top3_hit": top3h,
        })

    best_tile_acc = hits / len(loso_sample) * 100 if loso_sample else 0
    top3_acc      = top3_hits / len(loso_sample) * 100 if loso_sample else 0
    # Among misses, what is the avg suboptimality (predicted tile time vs true best time)?
    misses = [d for d in loso_details if not d["hit"]]
    subopt = [d["pred_best_us"] / d["true_best_us"] for d in misses if d["true_best_us"] > 0]
    avg_subopt = float(np.mean(subopt)) if subopt else 1.0

    print(f"  Best-tile accuracy (exact match): {hits}/{len(loso_sample)} = {best_tile_acc:.1f}%")
    print(f"  Top-3 accuracy:                   {top3_hits}/{len(loso_sample)} = {top3_acc:.1f}%")
    print(f"  Avg suboptimality on misses:      {avg_subopt:.3f}x  (1.0 = perfect)")

    # ── 3. Feature importance (|weight| after standardisation) ────────────────
    feat_names = make_feature_names()
    # Refit on all data for importance
    w_full, fm_full, fs_full, ym_full, ys_full = ridge_fit(X_all, y_all, alpha=alpha)
    importance = np.abs(w_full)
    order_imp  = np.argsort(importance)[::-1]
    print("\n── Feature importance (|weight|, top 10) ──")
    for i in order_imp[:10]:
        print(f"  {feat_names[i]:20s}  {importance[i]:.4f}")

    # ── Write report ──────────────────────────────────────────────────────────
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"# NPU Cost Model Evaluation Report",
        f"",
        f"**Generated:** {date_str}  ",
        f"**Model:** Ridge regression (numpy)  ",
        f"**Records:** {len(records)}  ",
        f"**Alpha:** {alpha}  ",
        f"",
        f"---",
        f"",
        f"## 1. Random 80/20 Holdout ({len(te)} test records)",
        f"",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| R² | {r2:.4f} |",
        f"| MAE (µs) | {mae_us:.1f} |",
        f"| MAPE | {mape:.1f}% |",
        f"",
        f"### Per-dtype breakdown",
        f"",
        f"| dtype | R² | MAE (µs) | MAPE | n |",
        f"|-------|----|---------|------|---|",
    ]
    for dtype, res in dtype_results.items():
        lines.append(f"| {dtype} | {res['r2']:.4f} | {res['mae_us']:.1f} | {res['mape']:.1f}% | {res['n']} |")
    lines += [
        f"",
        f"---",
        f"",
        f"## 2. Leave-One-Shape-Out Best-Tile Accuracy ({len(loso_sample)} shapes)",
        f"",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Best-tile exact match | {best_tile_acc:.1f}% ({hits}/{len(loso_sample)}) |",
        f"| Top-3 tile accuracy | {top3_acc:.1f}% ({top3_hits}/{len(loso_sample)}) |",
        f"| Avg suboptimality on misses | {avg_subopt:.3f}x |",
        f"",
        f"> **Interpretation**: Best-tile accuracy = fraction of held-out shapes where the",
        f"> model recommends the exact same tile as the profiling DB's best.  ",
        f"> Top-3 accuracy = fraction where the true best tile is in the model's top 3.  ",
        f"> Suboptimality = ratio of predicted tile's measured time vs true best time (1.0 = no loss).",
        f"",
        f"### LOSO detail (first 20 samples)",
        f"",
        f"| Shape | dtype | n tiles | True best | True avg | Pred best | Pred tile avg | Hit | Top3 |",
        f"|-------|-------|---------|-----------|----------|-----------|--------------|-----|------|",
    ]
    for d in loso_details[:20]:
        hit_str  = "✓" if d["hit"] else "✗"
        top3_str = "✓" if d["top3_hit"] else "✗"
        lines.append(
            f"| {d['shape_key']} | {d['dtype']} | {d['n_tiles']} "
            f"| {d['true_best']} | {d['true_best_us']:.0f}µs "
            f"| {d['pred_best']} | {d['pred_best_us']:.0f}µs "
            f"| {hit_str} | {top3_str} |"
        )
    lines += [
        f"",
        f"---",
        f"",
        f"## 3. Feature Importance (top 10)",
        f"",
        f"| Feature | |weight| |",
        f"|---------|---------|",
    ]
    for i in order_imp[:10]:
        lines.append(f"| {feat_names[i]} | {importance[i]:.4f} |")
    lines += [
        f"",
        f"---",
        f"",
        f"## Usage notes for the agent",
        f"",
        f"- If `npu_execution_profiling.json` has a `Best m,n,k` entry for the exact",
        f"  `(Mp, Np, Kp, dtype)` — **use profiling data directly**. It is always more",
        f"  accurate than the model.",
        f"- If profiling data is missing, call `predict_best_tile(M, N, K, dtype)` to",
        f"  get the model's recommendation. Check `source` field — `'model'` means",
        f"  no profiling data existed for this padded shape.",
        f"- Retrain the model after adding new profiling data:",
        f"  `python references/cost_model/npu_cost_model.py --train`",
    ]

    with open(EVAL_REPORT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nEvaluation report → {EVAL_REPORT}")

    return {
        "r2": r2, "mae_us": mae_us, "mape": mape,
        "best_tile_acc": best_tile_acc, "top3_acc": top3_acc,
        "avg_subopt": avg_subopt,
    }

# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="NPU GEMM cost regression model — train, evaluate, predict"
    )
    parser.add_argument("--train",    action="store_true", help="Train and save model")
    parser.add_argument("--evaluate", action="store_true", help="Run cross-validation evaluation")
    parser.add_argument("--predict",  action="store_true", help="Predict best tile for a shape")
    parser.add_argument("--M",     type=int,   help="M dimension (raw, not padded)")
    parser.add_argument("--N",     type=int,   help="N dimension (raw, not padded)")
    parser.add_argument("--K",     type=int,   help="K dimension (raw, not padded)")
    parser.add_argument("--dtype", type=str,   default="bf16", choices=["i8", "i16", "bf16"])
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Ridge regularisation strength (default 1.0)")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Number of top candidates to show in --predict")
    parser.add_argument("--loso-shapes", type=int, default=50,
                        help="Number of shapes for leave-one-shape-out evaluation")
    args = parser.parse_args()

    if not any([args.train, args.evaluate, args.predict]):
        parser.print_help()
        sys.exit(0)

    model = None

    if args.train:
        print("Training model...")
        model = train(alpha=args.alpha)

    if args.evaluate:
        evaluate(alpha=args.alpha, n_loso_shapes=args.loso_shapes)

    if args.predict:
        if not (args.M and args.N and args.K):
            print("ERROR: --predict requires --M, --N, --K")
            sys.exit(1)
        predict_best_tile(args.M, args.N, args.K, args.dtype,
                          model=model, top_k=args.top_k)


if __name__ == "__main__":
    main()
