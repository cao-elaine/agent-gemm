# NPU Cost Model Evaluation Report

**Generated:** 2026-03-28 03:40:49  
**Model:** Ridge regression (numpy)  
**Records:** 10204  
**Alpha:** 1.0  

---

## 1. Random 80/20 Holdout (2041 test records)

| Metric | Value |
|--------|-------|
| R² | 0.7691 |
| MAE (µs) | 298.8 |
| MAPE | 35.3% |

### Per-dtype breakdown

| dtype | R² | MAE (µs) | MAPE | n |
|-------|----|---------|------|---|
| i8 | 0.7147 | 241.4 | 40.7% | 432 |
| i16 | 0.7972 | 557.8 | 35.6% | 362 |
| bf16 | 0.7682 | 243.6 | 33.3% | 1247 |

---

## 2. Leave-One-Shape-Out Best-Tile Accuracy (50 shapes)

| Metric | Value |
|--------|-------|
| Best-tile exact match | 52.0% (26/50) |
| Top-3 tile accuracy | 94.0% (47/50) |
| Avg suboptimality on misses | 1.245x |

> **Interpretation**: Best-tile accuracy = fraction of held-out shapes where the
> model recommends the exact same tile as the profiling DB's best.  
> Top-3 accuracy = fraction where the true best tile is in the model's top 3.  
> Suboptimality = ratio of predicted tile's measured time vs true best time (1.0 = no loss).

### LOSO detail (first 20 samples)

| Shape | dtype | n tiles | True best | True avg | Pred best | Pred tile avg | Hit | Top3 |
|-------|-------|---------|-----------|----------|-----------|--------------|-----|------|
| 256_768_1024 | bf16 | 3 | (64, 64, 64) | 340µs | (64, 64, 64) | 340µs | ✓ | ✓ |
| 128_2048_2048 | i8 | 5 | (64, 128, 64) | 533µs | (64, 128, 64) | 533µs | ✓ | ✓ |
| 128_128_256 | i8 | 4 | (32, 64, 32) | 125µs | (64, 128, 64) | 135µs | ✗ | ✓ |
| 64_512_2048 | i16 | 3 | (32, 32, 32) | 392µs | (64, 64, 64) | 477µs | ✗ | ✓ |
| 64_1024_256 | i16 | 3 | (32, 64, 32) | 200µs | (64, 64, 64) | 231µs | ✗ | ✓ |
| 2048_1024_1024 | i16 | 5 | (64, 64, 64) | 3108µs | (64, 64, 64) | 3108µs | ✓ | ✓ |
| 32_768_768 | bf16 | 3 | (32, 64, 32) | 215µs | (32, 64, 32) | 215µs | ✓ | ✓ |
| 64_768_2048 | i8 | 4 | (32, 64, 32) | 278µs | (64, 128, 64) | 330µs | ✗ | ✓ |
| 128_1792_32 | bf16 | 5 | (32, 64, 16) | 232µs | (32, 64, 32) | 264µs | ✗ | ✓ |
| 512_256_128 | i16 | 3 | (64, 64, 64) | 173µs | (64, 64, 64) | 173µs | ✓ | ✓ |
| 64_4096_32 | bf16 | 5 | (16, 256, 32) | 197µs | (16, 256, 16) | 327µs | ✗ | ✓ |
| 64_512_64 | bf16 | 12 | (16, 128, 32) | 132µs | (64, 64, 64) | 139µs | ✗ | ✓ |
| 64_672_128 | bf16 | 8 | (16, 32, 32) | 204µs | (16, 32, 128) | 263µs | ✗ | ✓ |
| 128_256_64 | i16 | 3 | (32, 64, 32) | 132µs | (64, 64, 64) | 142µs | ✗ | ✓ |
| 64_384_128 | bf16 | 10 | (16, 128, 16) | 157µs | (16, 128, 32) | 160µs | ✗ | ✓ |
| 256_256_256 | i8 | 80 | (64, 64, 64) | 138µs | (64, 256, 32) | 178µs | ✗ | ✗ |
| 1024_768_768 | i16 | 40 | (64, 64, 64) | 936µs | (64, 64, 64) | 936µs | ✓ | ✓ |
| 64_320_512 | bf16 | 3 | (16, 16, 128) | 201µs | (16, 16, 256) | 204µs | ✗ | ✓ |
| 256_768_64 | i8 | 3 | (64, 64, 64) | 163µs | (64, 64, 64) | 163µs | ✓ | ✓ |
| 256_1024_64 | bf16 | 3 | (64, 64, 64) | 202µs | (64, 64, 64) | 202µs | ✓ | ✓ |

---

## 3. Feature Importance (top 10)

| Feature | |weight| |
|---------|---------|
| interaction | 0.2909 |
| log_n_tiles | 0.2400 |
| tile_mem_frac | 0.2194 |
| log_tile_vol | 0.2055 |
| log_Np/n | 0.1955 |
| log_Mp/m | 0.1568 |
| log_n | 0.1359 |
| log_Np | 0.1294 |
| log_m | 0.1204 |
| log_compute | 0.1166 |

---

## Usage notes for the agent

- If `npu_execution_profiling.json` has a `Best m,n,k` entry for the exact
  `(Mp, Np, Kp, dtype)` — **use profiling data directly**. It is always more
  accurate than the model.
- If profiling data is missing, call `predict_best_tile(M, N, K, dtype)` to
  get the model's recommendation. Check `source` field — `'model'` means
  no profiling data existed for this padded shape.
- Retrain the model after adding new profiling data:
  `python references/cost_model/npu_cost_model.py --train`
