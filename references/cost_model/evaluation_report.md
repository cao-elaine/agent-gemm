# NPU Cost Model Evaluation Report

**Generated:** 2026-03-22 01:36:24  
**Model:** Ridge regression (numpy)  
**Records:** 5129  
**Alpha:** 1.0  

---

## 1. Random 80/20 Holdout (1026 test records)

| Metric | Value |
|--------|-------|
| R² | 0.7872 |
| MAE (µs) | 351.2 |
| MAPE | 38.0% |

### Per-dtype breakdown

| dtype | R² | MAE (µs) | MAPE | n |
|-------|----|---------|------|---|
| i8 | 0.7622 | 286.8 | 39.2% | 416 |
| i16 | 0.7958 | 505.2 | 36.2% | 325 |
| bf16 | 0.8011 | 269.4 | 38.4% | 285 |

---

## 2. Leave-One-Shape-Out Best-Tile Accuracy (50 shapes)

| Metric | Value |
|--------|-------|
| Best-tile exact match | 64.0% (32/50) |
| Top-3 tile accuracy | 92.0% (46/50) |
| Avg suboptimality on misses | 1.116x |

> **Interpretation**: Best-tile accuracy = fraction of held-out shapes where the
> model recommends the exact same tile as the profiling DB's best.  
> Top-3 accuracy = fraction where the true best tile is in the model's top 3.  
> Suboptimality = ratio of predicted tile's measured time vs true best time (1.0 = no loss).

### LOSO detail (first 20 samples)

| Shape | dtype | n tiles | True best | True avg | Pred best | Pred tile avg | Hit | Top3 |
|-------|-------|---------|-----------|----------|-----------|--------------|-----|------|
| 256_64_768 | bf16 | 3 | (64, 64, 64) | 163µs | (64, 64, 64) | 163µs | ✓ | ✓ |
| 128_2048_256 | i8 | 4 | (64, 128, 64) | 205µs | (64, 128, 64) | 205µs | ✓ | ✓ |
| 2048_1024_256 | bf16 | 3 | (64, 64, 64) | 747µs | (64, 64, 64) | 747µs | ✓ | ✓ |
| 1024_64_768 | bf16 | 4 | (64, 64, 64) | 289µs | (64, 64, 64) | 289µs | ✓ | ✓ |
| 1024_512_768 | i8 | 6 | (64, 128, 64) | 306µs | (64, 128, 64) | 306µs | ✓ | ✓ |
| 2048_768_128 | i8 | 3 | (64, 64, 64) | 528µs | (64, 64, 64) | 528µs | ✓ | ✓ |
| 1024_512_128 | bf16 | 6 | (64, 64, 64) | 261µs | (64, 64, 64) | 261µs | ✓ | ✓ |
| 1024_768_64 | bf16 | 6 | (64, 64, 64) | 241µs | (64, 64, 64) | 241µs | ✓ | ✓ |
| 64_2048_768 | i8 | 4 | (64, 128, 64) | 281µs | (64, 128, 64) | 281µs | ✓ | ✓ |
| 64_128_2048 | i8 | 4 | (32, 32, 32) | 167µs | (64, 64, 64) | 175µs | ✗ | ✗ |
| 512_512_768 | bf16 | 3 | (64, 64, 64) | 352µs | (64, 64, 64) | 352µs | ✓ | ✓ |
| 512_256_512 | i16 | 3 | (64, 64, 64) | 234µs | (64, 64, 64) | 234µs | ✓ | ✓ |
| 256_512_256 | i8 | 4 | (64, 128, 64) | 149µs | (64, 128, 64) | 149µs | ✓ | ✓ |
| 1024_512_64 | bf16 | 6 | (64, 64, 64) | 197µs | (64, 64, 64) | 197µs | ✓ | ✓ |
| 64_512_256 | i16 | 3 | (32, 64, 32) | 163µs | (64, 64, 64) | 176µs | ✗ | ✓ |
| 2048_1024_128 | i16 | 3 | (64, 64, 64) | 709µs | (64, 64, 64) | 709µs | ✓ | ✓ |
| 512_768_1024 | bf16 | 3 | (64, 64, 64) | 561µs | (64, 64, 64) | 561µs | ✓ | ✓ |
| 1024_128_512 | i16 | 6 | (32, 32, 32) | 279µs | (64, 64, 64) | 320µs | ✗ | ✗ |
| 128_128_512 | i8 | 4 | (32, 64, 32) | 135µs | (64, 64, 64) | 137µs | ✗ | ✓ |
| 1024_768_128 | i8 | 6 | (64, 64, 64) | 229µs | (64, 64, 64) | 229µs | ✓ | ✓ |

---

## 3. Feature Importance (top 10)

| Feature | |weight| |
|---------|---------|
| interaction | 0.3242 |
| log_n_tiles | 0.2617 |
| log_compute | 0.2534 |
| tile_mem_frac | 0.2509 |
| log_Np/n | 0.2014 |
| log_Mp | 0.1983 |
| log_Np | 0.1969 |
| log_Mp/m | 0.1955 |
| log_Kp/k | 0.0805 |
| log_Kp | 0.0800 |

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
