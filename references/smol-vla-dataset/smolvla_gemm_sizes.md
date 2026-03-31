# SmolVLA GEMM Size Reference

> This file catalogues every GEMM operation in the SmolVLA model (vision encoder,
> connector, text layers, action expert, and projection heads). It lists each
> layer's M, K, N dimensions, defines the variables used to express them (B, L, C,
> S_conn), and provides a deduplicated summary of unique weight shapes (K × N) with
> their resolved M values. Use this as the authoritative reference for deciding
> which GEMM shapes to benchmark on the NPU.

Convention: **Y[M, N] = X[M, K] × W[K, N]**

## Variable Definitions

| Variable | Meaning |
|----------|---------|
| B | Batch size = 1|
| L | Text sequence length = 115, 128, 179, 235| num_image * 64 + 1 (state token) + 48 (text token)
| C | Action chunk length = 50, 30, 20, 60, 32|
| S_conn | Connector output tokens = 64 (~64, if 16-patch groups from 1024 patches) |
| A_dim | action expert hidden dimension 960 * factor, factor = 0.1,... ,0.75|

---

## Vision Encoder *(per layer, x1 reduced)*

| Layer | M | K | N | Notes |
|-------|---|---|---|-------|
| q_proj | B×1024 | 768 | 768 | |
| k_proj | B×1024 | 768 | 768 | |
| v_proj | B×1024 | 768 | 768 | |
| out_proj | B×1024 | 768 | 768 | |
| patch_embedding | B×1024 | 768 | 768 | im2col flattened |
| mlp.fc1 | B×1024 | 768 | 3072 | |
| mlp.fc2 | B×1024 | 3072 | 768 | |

---

## Connector

| Layer | M | K | N | Notes |
|-------|---|---|---|-------|
| modality_projection | B×S_conn | 12288 | 960 | Vision → text dim |

---

## Text Embeddings

| Layer | M | K | N | Notes |
|-------|---|---|---|-------|
| embed_tokens | B×L | 960 | 49280 | Row-gather (embedding lookup), not a GEMM |

---

## Text Layers *(per layer, x16)*

| Layer | M | K | N | Notes |
|-------|---|---|---|-------|
| q_proj | B×L | 960 | 960 | |
| k_proj | B×L | 960 | 320 | GQA: fewer KV heads |
| v_proj | B×L | 960 | 320 | GQA: fewer KV heads |
| o_proj | B×L | 960 | 960 | |
| gate_proj | B×L | 960 | 2560 | SwiGLU |
| up_proj | B×L | 960 | 2560 | SwiGLU |
| down_proj | B×L | 2560 | 960 | |

---

## LM Head

| Layer | M | K | N | Notes |
|-------|---|---|---|-------|
| lm_head | B×L | 960 | 49280 | Tied with embed_tokens weight |

---

## Action Expert *(per layer, x16, cross-attention into text KV)*

| Layer | M | K | N | Notes |
|-------|---|---|---|-------|
| q_proj | B×C | 960 | A_dim | Q from action tokens |
| k_proj | B×L | 320 | A_dim | KV reused from text stream |
| v_proj | B×L | 320 | A_dim | KV reused from text stream |
| o_proj | B×C | A_dim | 960 | |
| gate_proj | B×C | A_dim | 256 | SwiGLU |
| up_proj | B×C | A_dim | 256 | SwiGLU |
| down_proj | B×C | 256 | A_dim | |

---

## Projections

| Layer | M | K | N | Notes |
|-------|---|---|---|-------|
| state_proj | B | 960 | 32 | Per-sample, no token dim |
| action_in_proj | B×C | 32 | A_dim | Action input projection |
| action_out_proj | B×C | A_dim | 32 | Action output projection |
| action_time_mlp_in | B | 192 | A_dim | Time embedding MLP input |
| action_time_mlp_out | B | A_dim | A_dim | Time embedding MLP output |

---

## Unique Weight Shapes Summary *(K × N)*

M values resolved using B=1, L=128, C=50, S_conn=64.

| M | K | N | Used In | Count |
|---|---|---|---------|-------|
| 1024 | 768 | 768 | Vision Q/K/V/O proj, patch embedding | 5 per layer |
| 1024 | 768 | 3072 | Vision MLP fc1 | 1 per layer |
| 1024 | 3072 | 768 | Vision MLP fc2 | 1 per layer |
| 64 | 12288 | 960 | Connector modality_projection | 1 |
| 128 | 960 | 960 | Text Q/O proj | 2 per layer × 16 |
| 128 | 960 | 320 | Text K/V proj | 2 per layer × 16 |
| 128 | 960 | 2560 | Text gate/up proj | 2 per layer × 16 |
| 128 | 2560 | 960 | Text down proj | 1 per layer × 16 |
| 128 | 960 | 49280 | LM head | 1 |
| 50 | 960 | A_dim | Action Q proj | 1 per layer × 16 |
| 128 | 320 | A_dim | Action K/V proj (KV from text stream) | 2 per layer × 16 |
| 50 | A_dim | 960 | Action O proj | 1 per layer × 16 |
| 50 | A_dim | 256 | Action gate/up proj | 2 per layer × 16 |
| 50 | 256 | A_dim | Action down proj | 1 per layer × 16 |
| 1 | 960 | 32 | state_proj | 1 |
| 50 | 32 | A_dim | action_in_proj | 1 |
| 50 | A_dim | 32 | action_out_proj | 1 |
| 1 | 192 | A_dim | action_time_mlp_in | 1 |
| 1 | A_dim | A_dim | action_time_mlp_out | 1 |

---

## Notes

- All weights are **int4** in the reduced model; activations may be computed in higher precision.
- Layer norms (RMSNorm, LayerNorm) are element-wise — **no GEMM required**.
- `embed_tokens` is a **row-gather** (embedding lookup) sharing the same weight shape as `lm_head` [49280 × 960], but is not a GEMM.
- `rope`, `softmax`, `silu`, `gelu`, residual adds, and masking are **non-GEMM** ops.
- Vision encoder token count (1024) is fixed for a given input resolution (16×16 patches over a 512×512 or similar image).
- `S_conn` depends on connector pooling strategy; verify from model config.
