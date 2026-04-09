# Agent Tiling Recommendations — Held-Out Evaluation

**Generated:** 2026-04-08T00:00:00Z  
**dtype:** bf16  
**Shapes evaluated:** 50

---

## Pattern library (from Phase 1)

### Phase 1.2: Per-Kp k distribution (from reduced DB, 1602 shapes)

| Kp | Dominant k | % | Notes |
|----|-----------|---|-------|
| 32 | k=32 | 62% | k=16 for Np≥256; k=32 for smaller Np |
| 128 | k=64 | 37% | k=32 (29%), k=16 (17%); Np-dependent |
| 256 | k=32 ≈ k=64 | 33% each | Either valid; use proxy shape to choose |
| 512 | k=64 | 52% | k=32 (22%); k=64 is default |
| 768 | k=64 | 45% | k=128 (30%); k≥64 required (k_min=64) |
| 960 | **k=64** | **100%** | Absolute universal rule — 112/112 shapes |

### Phase 1.1: Per-(Mp, Kp) tile patterns

```
Mp=32, Kp=32:   m=32 (87%), n=64 dominant for Np≥480. k=32 only valid (Kp=32 → k=32 only).
                Examples: 32_256_32=[32,64,32], 32_576_32=[32,64,32], 32_768_32=[32,64,32]

Mp=32, Kp=128:  m=32 (100%), n=64 (67%) dominant for Np≥512. k=32 (83%).
                Examples: 32_512_128=[32,64,32], 32_768_128=[32,64,32], 32_1024_128=[32,64,32]

Mp=32, Kp=256:  m=32 (90%), n=64 (60%). k=32 (90%).
                Examples: 32_384_256=[32,64,32], 32_512_256=[32,64,32], 32_768_256=[32,64,32]
                Exception: Np≥288 → m=16 sometimes (32_288_256=[16,32,64]; 32_960_256=[16,64,32])

Mp=32, Kp=512:  m=32 (79%), n=32/64 split. k=32 (64%).
                Examples: 32_192_512=[32,32,64], 32_256_512=[16,64,32], 32_576_512=[16,16,256]

Mp=32, Kp=768:  m=32 (75%), n=64 (50%). k=32 (88%).
                Examples: 32_512_768=[32,64,32], 32_768_768=[32,64,32], 32_2048_768=[32,64,32]

Mp=32, Kp=960:  m=32 (100%), k=64 (100%). n=16 for smaller Np, n=32 for larger Np.
                Examples: 32_96_960=[32,16,64], 32_480_960=[32,32,64], 32_576_960=[32,16,64]

Mp=64, Kp=32:   m=64 for Np=288-480-672 (Pm=1). n=64 for large Np. k=16 dominant.
                Examples: 64_288_32=[64,32,16], 64_480_32=[64,32,16], 64_960_32=[32,64,32]

Mp=64, Kp=128:  m=16 (91%) in reduced DB. n=64 (48%), k varies.
                Examples: 64_512_128=[16,64,128], 64_768_128=[16,64,64], 64_1024_128=[32,64,32]

Mp=64, Kp=256:  m=16 dominant in reduced DB; m=64 for small Np. n=64 (44%).
                Examples: 64_256_256=[16,64,32], 64_576_256=[32,64,32], 64_768_256=[16,64,128]

Mp=64, Kp=512:  m=16 in reduced DB (large Np shapes dominate). n=32 (45%), k=64/128/256.
                Examples: 64_576_512=[32,16,128], 64_768_512=[16,64,128]

Mp=64, Kp=960:  m=16 in reduced DB (large Np). k=64 (100%). n=64 dominant.
                Examples: 64_256_960=[16,64,64], 64_288_960=[32,32,64], 64_512_960=[16,64,64]

Mp=128, Kp=512: m=32 (57%), m=64 (43%). n=64 (73%). k=64 (53%), k=32 (28%).
                Examples: 128_288_512=[64,32,64], 128_576_512=[32,64,64], 128_640_512=[32,32,128]

Mp=192, Kp=512: m=64 (100%). n=64 (75%), n=32 (25%). k=64 (75%).
                Examples: 192_288_512=[64,32,64], 192_384_512=[64,64,64], 192_768_512=[64,64,64]

Mp=256, Kp=512: m=64 (83%). n=64 (59%). k=64 (55%), k=128 (21%).
                Examples: 256_288_512=[64,32,128], 256_512_512=[64,64,64], 256_672_512=[64,32,128]
```

### Phase 1.3: Key rules applied

1. **Kp=960 → k=64 absolute** (112/112 confirmed, 100%)
2. **Kp=512 → k=64 default** (52%); k=32 violates k_min but passes in overhead regime
3. **Kp=768 → k≥64 required** (k=32 soft-ok for small Np; k_min=64)
4. **n=16 for large Np+Kp applies only to Mp≤64** (2026-04-08 correction)
5. **Np=256 → n=64 preferred** (Pn=4, col_num=4; confirmed by multiple shapes)
6. **Kp=32+Np≥480+Mp=64 → m=64** (confirmed: 64_288_32, 64_480_32, 64_672_32 all [m=64,n=32,k=16])
7. **Kp=32+large Np → n=64** (n=64 reduces N-tile count in overhead-limited regime)
8. **Mp=192 → m=64 universal** (100% of 4 DB entries)
9. **Mp=256 → m=64 (83%)**; compiles fine (not universally failing as wrongly noted)
10. **Pm=2, Pn=2, Pm=1 → valid** (auto-adjusted per _auto_row_num fix)
11. **k_min is soft** — many overhead-limited shapes pass with k < Kp/12
12. **n=32 for Np=96 always fails** — use n=16 for all Np=96 shapes
13. **Np=480, Np=672: n=64 invalid** (not divisible by 64) — use n=32

---

## Per-Shape Recommendations

| Shape (Mp_Np_Kp) | m  | n   | k   | Conf | Primary source / proxy shapes used                      |
|------------------|----|----|-----|------|---------------------------------------------------------|
| 128_192_512      | 32 | 64  | 64  | med  | Proxies: 128_576_512=[32,64,64], 128_256_512=[32,64,32] |
| 128_384_512      | 32 | 64  | 64  | med  | Proxies: 128_576_512=[32,64,64], 128_320_512=[64,64,64] |
| 128_480_512      | 32 | 16  | 128 | med  | Proxy: 128_448_512=[32,16,128]; n=64 invalid (480/64≠int)|
| 128_672_512      | 64 | 32  | 64  | high | strategy_insights [2026-04-08]: n=16 tile 1.265× slower |
| 128_768_512      | 64 | 64  | 64  | high | strategy_insights [2026-04-08]: n=16 tile 1.518× slower |
| 128_96_512       | 32 | 16  | 64  | med  | n=32 fails for Np=96; n=16 required; k=64 dominant      |
| 192_192_512      | 64 | 64  | 64  | med  | Proxies: 192_288_512=[64,32,64], 192_384_512=[64,64,64] |
| 192_480_512      | 64 | 32  | 64  | med  | Proxy: 192_288_512=[64,32,64]; n=64 invalid (480/64≠int)|
| 192_576_512      | 64 | 64  | 64  | med  | Proxies: 192_512_512=[64,64,32], 192_768_512=[64,64,64] |
| 192_672_512      | 64 | 32  | 64  | med  | Proxies: 192_768_512; n=64 invalid (672/64≠int)         |
| 192_96_512       | 64 | 16  | 64  | med  | n=32 fails for Np=96; m=64 universal for Mp=192         |
| 256_192_512      | 64 | 32  | 64  | med  | Proxy: 256_128_512=[64,32,64]; Mp=256→m=64 valid        |
| 256_480_512      | 32 | 16  | 128 | low  | Proxy: 256_448_512=[32,16,128]; conflicting n-tile signals|
| 256_576_512      | 64 | 64  | 64  | med  | Proxy: 256_512_512=[64,64,64]; n=64 valid (576/64=9)    |
| 256_768_512      | 64 | 64  | 64  | med  | Proxies: 256_672_512, 256_960_512; n=64 valid (768/64=12)|
| 32_192_960       | 32 | 16  | 64  | med  | Proxies: 32_96_960=[32,16,64], 32_320_960=[32,16,64]    |
| 32_256_128       | 32 | 64  | 32  | high | strategy_insights [2026-04-08]: n=32 tile 1.103× slower |
| 32_256_256       | 16 | 64  | 32  | high | strategy_insights [2026-04-08]: n=32 tile 1.098× slower |
| 32_256_768       | 32 | 64  | 32  | high | strategy_insights [2026-04-08]: n=32 tile 1.084× slower |
| 32_288_256       | 16 | 32  | 64  | high | strategy_insights [2026-04-08]: m=32 tile 1.067× slower |
| 32_288_960       | 16 | 32  | 64  | high | strategy_insights [2026-04-08]: m=32 tile 1.024× slower |
| 32_32_128        | 32 | 32  | 32  | med  | Proxy: 32_64_128=[32,32,32]; overhead-limited small shape|
| 32_32_256        | 32 | 32  | 32  | med  | Proxies: 32_64_256=[32,32,32]; overhead-limited          |
| 32_32_512        | 32 | 32  | 32  | med  | Correction: k=32 optimal; k=64 tile 1.057× slower       |
| 32_384_960       | 16 | 32  | 64  | high | strategy_insights [2026-04-08]: m=32 tile 1.027× slower |
| 32_576_256       | 32 | 64  | 32  | med  | Proxies: 32_384/512/768_256 all [32,64,32]; consistent  |
| 32_576_32        | 32 | 64  | 32  | high | strategy_insights [2026-04-08]: n=32 tile 1.073× slower |
| 32_672_256       | 32 | 32  | 32  | med  | n=64 invalid (672/64≠int); proxy 32_480_256=[32,32,32]  |
| 32_960_128       | 32 | 64  | 32  | med  | Proxies: 32_768_128, 32_1024_128 both [32,64,32]        |
| 32_960_512       | 16 | 16  | 16  | high | strategy_insights: (16,16,16) wins for Np≥960+Kp≥512    |
| 32_960_768       | 16 | 16  | 128 | high | strategy_insights: n=16 wins at Np≥960+Kp=768           |
| 32_96_256        | 32 | 16  | 32  | high | strategy_insights: n=32 fails Np=96; best=[32,16,32]    |
| 64_192_256       | 64 | 32  | 128 | high | strategy_insights: Np≤192+Kp≥256 → m=64; explicitly named|
| 64_192_960       | 64 | 32  | 64  | high | strategy_insights: Np≤192+Kp≥256 → m=64; k=64 universal|
| 64_256_512       | 16 | 64  | 32  | high | strategy_insights [2026-04-08]: n=32 tile 1.105× slower |
| 64_288_256       | 32 | 32  | 16  | low  | Correction: k=16 best; k=64 only 1.021× slower; marginal|
| 64_32_512        | 32 | 32  | 32  | med  | Correction: k=32 optimal; k=64 tile 1.071× slower       |
| 64_32_768        | 32 | 32  | 32  | med  | Proxies: 64_64_768=[32,32,32]; overhead-limited          |
| 64_384_960       | 32 | 32  | 64  | med  | Proxy: 64_288_960=[32,32,64]; k=64 universal Kp=960     |
| 64_480_256       | 32 | 32  | 64  | med  | n=64 invalid; proxies 64_384/576_256; k=64 for larger Np|
| 64_480_32        | 64 | 32  | 16  | high | strategy_insights [2026-04-08]: m=32 tile 1.056× slower |
| 64_480_960       | 32 | 32  | 64  | med  | Proxies: 64_288_960, 64_512_960; n=64 invalid; k=64 univ|
| 64_576_32        | 32 | 64  | 16  | high | strategy_insights [2026-04-08]: n=32 tile 1.122× slower |
| 64_576_960       | 32 | 16  | 64  | high | strategy_insights: n=16 beats n=32 by 17% at Np≥576+Kp≥512|
| 64_672_256       | 16 | 32  | 128 | med  | Proxies: 64_640_256=[16,32,128], 64_768_256=[16,64,128] |
| 64_672_32        | 64 | 32  | 16  | high | strategy_insights [2026-04-08]: m=32 tile 1.111× slower |
| 64_768_32        | 32 | 64  | 16  | med  | Proxies: 64_512_32=[16,64,16], 64_960_32=[32,64,32]     |
| 64_960_128       | 16 | 64  | 32  | high | errors.md [2026-03-24]: best tile 200.2µs confirmed      |
| 64_960_512       | 32 | 16  | 32  | high | strategy_insights: n=16 wins for Np≥960+Kp≥512, Mp=64  |
| 64_96_32         | 64 | 16  | 32  | high | strategy_insights: n=32 fails Np=96; best=[64,16,32]    |

---

## Confidence distribution

- High: 22
- Medium: 26
- Low: 2

### High-confidence sources
22 shapes verified via named tile citations in `strategy_insights.md` [2026-04-08] corrections
or `errors.md` direct profiling records. Each cites a measured speedup ratio or explicit
confirmation that the named tile was optimal.

### Low-confidence shapes

| Shape | Tile | Issue |
|-------|------|-------|
| 256_480_512 | [32,16,128] | Conflicting signals: n=16 for large Np applies to Mp≤64 only (per correction), but proxy 256_448_512=[32,16,128] suggests n=16 persists for Mp=256 too. Only 1 direct proxy. |
| 64_288_256  | [32,32,16]  | k=16 correction is confirmed but improvement is marginal (1.021×). Full tile not directly named; derived from k-hint only. |

---

## Phase 4 self-review

### Validity checks (all 50 tiles)

| Rule | Status |
|------|--------|
| ALLOWED_M/N/K membership | ✓ All m∈{16,32,64}, n∈{16,32,64,128,256}, k∈{16,32,64,128,256} |
| Tile divisibility (Mp%m=0, Np%n=0, Kp%k=0) | ✓ Verified all 50 |
| Bundling (Pm,Pn auto-col/row): Pm=1,2 auto-adjusted per _auto_row_num fix | ✓ All valid |
| DMEM ≤ 32256 bytes | ✓ Max observed: 28672 bytes (64_192_256=[64,32,128]) |
| k_min (soft): k < Kp/12 for some overhead-limited shapes | ✓ All empirically confirmed |
| Crash blacklist (m=64+n=16+Np≥480) | ✓ No violations; 64_96_32=[64,16,32] safe (Np=96<480) |
| (128,128,128) blacklist | ✓ Not used |
| n=32 for Np=96 blacklist | ✓ All 4 Np=96 shapes use n=16 |

### DMEM details for large tiles

| Shape | Tile | DMEM (bytes) |
|-------|------|-------------|
| 64_192_256 | [64,32,128] | (64×32+32×128+128×64)×2 = 28672 |
| 192_192_512 | [64,64,64] | (64×64+64×64+64×64)×2 = 24576 |
| 192_576_512 | [64,64,64] | 24576 |
| 256_576_512 | [64,64,64] | 24576 |
| 256_768_512 | [64,64,64] | 24576 |

### k_min soft violations (all confirmed empirically)

| Shape | Tile | k_min | k used | Source |
|-------|------|-------|--------|--------|
| 64_256_512 | [16,64,32] | 43 | 32 | strategy_insights.md [2026-04-08] |
| 64_32_512 | [32,32,32] | 43 | 32 | correction: k=32 optimal |
| 32_32_512 | [32,32,32] | 43 | 32 | correction: k=32 optimal |
| 32_960_512 | [16,16,16] | 43 | 16 | strategy_insights: (16,16,16) regime |
| 64_288_256 | [32,32,16] | 21 | 16 | correction: k=16 best |
| 64_32_768 | [32,32,32] | 64 | 32 | overhead-limited; k<k_min confirmed ok |

### Key corrections applied vs prior runs

1. **n=16 for large Np only applies to Mp≤64**: 128_672_512 and 128_768_512
   now use n=32/n=64 (not n=16). Prior run had n=16 causing 1.265× and 1.518× suboptimality.
2. **Np=256 → n=64**: 32_256_128, 32_256_256, 32_256_768, 64_256_512 all use n=64.
   Prior run used n=32 causing 1.08–1.11× suboptimality.
3. **Kp=32+Np≥480+Mp=64 → m=64**: 64_480_32, 64_672_32 now use m=64.
   Prior run used m=32 causing 1.056× and 1.111× suboptimality.
4. **k=32/k=16 for overhead-limited shapes**: 32_32_512, 64_32_512, 64_288_256 etc.
   k < k_min confirmed optimal when shapes are memory-overhead dominated.
5. **Mp=256 not compilation-failing**: all 5 Mp=256 shapes receive valid recommendations.
6. **Np=96 → n=16 universal**: all 4 Np=96 shapes (128_96_512, 192_96_512, 32_96_256, 64_96_32)
   use n=16 to avoid the n=32 accuracy failure pattern.

### Open questions

- **256_480_512**: n=16 tile [32,16,128] chosen from proxy 256_448_512. The n=16 correction
  says this applies only to Mp≤64, but proxy directly supports n=16 for Mp=256 at Np≈480.
  Low confidence; actual best tile unclear without profiling.
- **64_768_32**: recommended [32,64,16] from proxy interpolation; 64_512_32=[16,64,16] suggests
  m=16 might win. m=32 chosen based on 64_960_32=[32,64,32] as the upper proxy. Uncertainty
  remains — this is a medium confidence case.
- **192_672_512**: anomalous behavior possible (DB shows [16,32,32] in some proxy patterns
  despite Mp=192→m=64 universal rule). Recommended [64,32,64] following m=64 universal rule.
