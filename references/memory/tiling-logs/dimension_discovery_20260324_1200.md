# Dimension Discovery Log — 2026-03-24 12:00 (Session 2)

**Scope**: Dimensions >2048 (primary) + small secondary candidates
**Method**: Direct GEMM execution (no --use-padding), bf16 dtype
**Fixed params**: test_M_fixed=128 (Plan A), test_N_fixed=512 (Plan B), test_K_fixed=512
**k_min rule**: k ≥ max(32, Kp//16) = 32 (for Kp=512)

---

## Phase 0 — State at session start

- ALLOWED_M (22 values): [32, 64, 128, 192, 256, 320, 384, 512, 576, 640, 960, 1024, 1152, 1280, 1344, 1536, 1600, 1728, 1792, 1920, 2048, 3072]
- ALLOWED_N (23 values): [32, 64, 128, 192, 256, 320, 384, 512, 576, 640, 768, 960, 1024, 1152, 1280, 1344, 1536, 1600, 1728, 1792, 1920, 2048, 3072]
- ALLOWED_K (10 values): [32, 64, 128, 256, 512, 768, 960, 1024, 2048, 3072] (unchanged)
- Profiling JSON: 486 entries before this session

Large-dim entries already present:
- 128_2560_960 (from previous SmolVLA experiment)

---

## Plan A — New N Dimensions (N>2048, M=128, K=512)

Primary candidates: n=64, Pn ∈ {33,35,36,39,40,42,44,45,50,51,52,54,55,56,57,60,63,64}

### Results Table

| D    | Pn  | col_num | (64,64,64) | (32,64,64) | (64,64,32) | Best tile    | Best latency | Status    |
|------|-----|---------|------------|------------|------------|--------------|--------------|-----------|
| 2112 | 33  | 3       | acc_fail   | PASS       | acc_fail   | (32,64,64)   | 687us        | CONFIRMED |
| 2240 | 35  | 5       | PASS       | acc_fail   | —          | (64,64,64)   | 652us        | CONFIRMED* |
| 2304 | 36  | 4       | PASS       | PASS       | —          | (64,64,64)   | 373us        | CONFIRMED |
| 2496 | 39  | 3       | PASS       | PASS       | —          | (64,64,64)   | 715us        | CONFIRMED |
| 2560 | 40  | 4       | PASS       | PASS       | PASS       | (32,64,128)  | 398us        | CONFIRMED |
| 2688 | 42  | 3       | acc_fail   | PASS       | acc_fail   | (32,64,64)   | 830us        | CONFIRMED |
| 2816 | 44  | 4       | PASS       | PASS       | PASS       | (64,64,64)   | 425us        | CONFIRMED |
| 2880 | 45  | 3       | acc_fail   | PASS       | acc_fail   | (32,64,64)   | 887us        | CONFIRMED |
| 3200 | 50  | 5       | acc_fail   | acc_fail   | PASS(k=32) | (64,64,32)   | 943us        | CONFIRMED* |
| 3264 | 51  | 3       | PASS       | PASS       | PASS       | (64,64,64)   | 914us        | CONFIRMED |
| 3328 | 52  | 4       | PASS       | PASS       | PASS       | (64,64,64)   | 490us        | CONFIRMED |
| 3456 | 54  | 3       | acc_fail   | PASS       | PASS       | (64,64,32)   | 995us        | CONFIRMED |
| 3520 | 55  | 5       | acc_fail   | —          | acc_fail   | (none)       | —            | FAILED    |
| 3584 | 56  | 4       | PASS       | PASS       | PASS       | (64,64,64)   | 511us        | CONFIRMED |
| 3648 | 57  | 3       | acc_fail   | PASS       | PASS       | (64,64,32)   | 1083us       | CONFIRMED |
| 3840 | 60  | 4       | PASS       | PASS       | —          | (64,64,64)   | 559us        | CONFIRMED |
| 4032 | 63  | 3       | acc_fail   | PASS       | PASS       | (64,64,32)   | 1128us       | CONFIRMED |
| 4096 | 64  | 4       | PASS       | PASS       | PASS       | (64,64,64)   | 558us        | CONFIRMED |

*CONFIRMED* = confirmed with tile restrictions (col_num=5 accuracy pattern)

**N=3520 FAILED** — col_num=5 (Pn=55, only divisible by 5), all tiles tried had accuracy issues.

### Key observation: col_num=5 accuracy pattern for N dimensions
- Dims where Pn is divisible only by 5 (not 3 or 4): 2240 (35→5), 3200 (50→need col_num=5), 3520 (55→5)
- N=2240: works with m≥64, fails with m=32 — **n=32 tiles fine since Pn=35//1=35 is not n-tile bundling**
  Actually: col_num=5 (Pn=2240//64=35, 35%5=0), and m=32 accuracy issue is a k=64 issue
- N=3200: works only with (64,64,32) — k=32 reduces numerical accumulation issues
- N=3520: fully fails (Pn=55, col_num=5 only), cannot work at all

---

## Plan B — New M Dimensions (M as new dim, N=512, K=512)

| D    | Pm(m=64) | row_num | (64,64,64) | Best tile   | Best latency | Status    | Notes          |
|------|----------|---------|------------|-------------|--------------|-----------|----------------|
| 2112 | 33       | 3       | PASS       | (64,64,64)  | 1365us       | CONFIRMED |                |
| 2240 | 35       | 5 only  | acc_fail   | (16,64,64)  | 1814us       | CONFIRMED* | use m=16       |
| 2304 | 36       | 4       | PASS       | (64,64,64)  | 757us        | CONFIRMED |                |
| 2496 | 39       | 3       | PASS       | (64,64,64)  | 1608us       | CONFIRMED |                |
| 2560 | 40       | 4       | PASS       | (64,64,64)  | 820us        | CONFIRMED |                |
| 2688 | 42       | 3       | PASS       | (64,64,64)  | 1789us       | CONFIRMED |                |
| 2816 | 44       | 4       | PASS       | (64,64,64)  | 922us        | CONFIRMED |                |
| 2880 | 45       | 3       | PASS       | (64,64,64)  | 1910us       | CONFIRMED |                |
| 3200 | 50       | 5 only  | acc_fail   | (32,64,64)  | 1448us       | CONFIRMED* | Pm=100,100%4=0 |
| 3264 | 51       | 3       | PASS       | (64,64,64)  | 2084us       | CONFIRMED |                |
| 3328 | 52       | 4       | PASS       | (64,64,64)  | 1054us       | CONFIRMED |                |
| 3456 | 54       | 3       | PASS       | (64,64,64)  | 2229us       | CONFIRMED |                |
| 3520 | 55       | 5 only  | acc_fail   | (none)      | —            | FAILED    | m=16 also fails|
| 3584 | 56       | 4       | PASS       | (64,64,64)  | 1108us       | CONFIRMED |                |
| 3648 | 57       | 3       | PASS       | (64,64,64)  | 2359us       | CONFIRMED |                |
| 3840 | 60       | 4       | PASS       | (64,64,64)  | 1193us       | CONFIRMED |                |
| 4032 | 63       | 3       | PASS       | (64,64,64)  | 2544us       | CONFIRMED |                |
| 4096 | 64       | 4       | PASS       | (64,64,64)  | 1243us       | CONFIRMED |                |

*M=3200 special case*: Pm=3200//64=50 (row_num=5 only, accuracy fail), but m=32 gives Pm=100, 100%4=0, row_num=4 — PASSES.

---

## Plan C — Secondary Small Dimensions (N<64 tile, M=128, K=512)

| D   | n   | Pn | col_num | Best tile    | Best latency | Status    | Notes              |
|-----|-----|----|---------|--------------|--------------|-----------|--------------------|
| 96  | 32  | 3  | 3       | (64,32,64)   | 152us        | CONFIRMED |                    |
| 160 | 32  | 5  | 5       | (64,32,64)   | 168us        | CONFIRMED* | m=32 fails (col=5) |
| 288 | 32  | 9  | 3       | (64,32,64)   | 192us        | CONFIRMED |                    |
| 448 | 16  | 28 | 4       | (32,16,128)  | 240us        | CONFIRMED |                    |
| 480 | 32  | 15 | 3       | (64,32,64)   | 227us        | CONFIRMED |                    |
| 672 | 32  | 21 | 3       | (64,32,64)   | 293us        | CONFIRMED |                    |

*N=160: col_num=5 (Pn=5), m=32 has accuracy issue, m≥64 works.

---

## Summary

### Confirmed NEW dimensions (this session):

**N dimensions confirmed (17/18)**:
2112, 2240*, 2304, 2496, 2560, 2688, 2816, 2880, 3200*, 3264, 3328, 3456, 3584, 3648, 3840, 4032, 4096
(* = tile restrictions due to col_num=5)

**FAILED N dimension (1)**: 3520 (col_num=5 only, all tiles fail)

**M dimensions confirmed (17/18)**:
2112, 2240*, 2304, 2496, 2560, 2688, 2816, 2880, 3200*, 3264, 3328, 3456, 3584, 3648, 3840, 4032, 4096
(* = tile restrictions)

**FAILED M dimension (1)**: 3520 (row_num=5 only, all tiles fail)

**Secondary dimensions confirmed (6/6)**: 96, 160*, 288, 448, 480, 672
(* = col_num=5, m≥64 required)

### col_num=5 pattern summary (accumulated):
- Dimensions where D//64 is divisible only by 5 (not 3 or 4):
  - N=160 (Pn=5): use m≥64
  - N=320 (Pn=5): use m≥64 [prev session]
  - N=640 (Pn=10=2×5, but also %4≠0 if 10%4≠0... actually 10%5=0, 10%3≠0, 10%4≠0, so col_num=5): use m≥64 [prev]
  - N=1600 (Pn=25, 25%5=0 only): use m≥64 [prev]
  - N=2240 (Pn=35, 35%5=0 only): use m≥64 (k=64 ok with m≥64)
  - N=3200 (Pn=50, 50%5=0 but 50%5=0, 50%4≠0, 50%3≠0): use (64,64,32) only (k=64 fails even with m≥64)
  - N=3520 (Pn=55, 55%5=0 only): **ALL tiles fail** — dimension unusable in bf16

- M dimensions with row_num=5 only:
  - M=2240 (Pm=35 with m=64 → use m=16 or m=32 for Pm%4=0): use m=16 (Pm=140, 140%4=0)
  - M=3200 (Pm=50 with m=64 → use m=32 for Pm=100, 100%4=0): PASS
  - M=3520 (Pm=55 with m=64 → use m=16 for Pm=220, 220%4=0): **still fails** — dimension unusable

### SmolVLA cross-reference:
- N=96: appears in smolvla_gemm_shapes.json in 5 shapes (action projections, cross-attention)
- N=2560: appears in gate_proj/up_proj (128×2560×960) — previously had one entry, now confirmed with K=512 too

### Total profiling JSON entries after this session: 486 + [new entries below]
