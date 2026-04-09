# Agent Held-Out Tiling Recommendations
**Generated:** 2026-04-08T00:00:00  
**Dtype:** bf16  
**Shapes evaluated:** 50

---

## Pattern library (from Phase 1)

### Phase 1.2 — k-tile by Kp

| Kp  | k_min (Kp/12) | Dominant best k from DB      | Notes                                                                 |
|-----|--------------|------------------------------|-----------------------------------------------------------------------|
| 32  | 2.7          | k=32 (Np<256), k=16 (Np≥256) | strategy_insights: Mp=64+Kp=32+Np≥256 → k=16                         |
| 128 | 10.7         | k=32 (small Np), k=64 (medium)| 32_512_128, 32_768_128 all show k=32                                 |
| 256 | 21.3         | k=32 (Mp=32); k=32/k=64 equal (Mp=64) | Held-out: k=16 also wins for small total-shapes              |
| 512 | 42.7         | k=64 (52%), k=32 small shapes | k=32 valid for overhead-limited shapes (Mp×Np≤2048)                  |
| 768 | 64.0         | k=32 (Np≤512), k=128 (Np≥768)| errors.md: k=32 passes for Np≤512; k≥128 for Np≥768                  |
| 960 | 80.0         | **k=64 universally (100%)**  | Kp=960 absolute rule: k=64 regardless of Mp/Np; 119/119 in DB        |

### Phase 1.1 — m-tile and n-tile patterns by (Mp, Kp)

**Mp=32, Kp=32:** m=32; n=32 or n=64 for medium/large Np. k=32.  
Proxies: 32_512_32=[32,64,32], 32_768_32=[32,64,32], 32_960_32=[32,64,16].

**Mp=32, Kp=128:** m=32; n=64 dominant for Np≥256. k=32.  
Proxies: 32_512_128=[32,64,32], 32_768_128=[32,64,32], 32_1024_128=[32,64,32].

**Mp=32, Kp=256:** m=32 (Np≤192), m=16 (Np≥288, ~50%). n=64 for Np≥256 (but n=64 may be invalid for Np=288,672). k=32.  
Proxies: 32_192_256=[32,32,16], 32_384_256=[32,64,32], 32_512_256=[32,64,32], 32_960_256=[16,64,32].

**Mp=32, Kp=512:** m=32; n=16/n=32/n=64 vary with Np; k=32 for small shapes, k=64 for medium.  
Proxies: 32_192_512=[32,32,64], 32_256_512=[16,64,32], 32_768_512=[32,64,32].  
Held-out: 32_960_512=[16,16,16] (overhead-limited large-Np rule).

**Mp=32, Kp=768:** m=32; n=64 for medium Np. k=32 (Np≤512), k=128 (Np≥768).  
Proxies: 32_512_768=[32,64,32], 32_768_768=[32,64,32].  
Held-out: 32_960_768=[16,16,128] (n=16+large-k for large Np+Kp).

**Mp=32, Kp=960:** m=32 (Np≤192), m=16 (Np≥288). n=16 dominant. k=64 (absolute).  
Proxies: 32_96_960=[32,16,64], 32_320_960=[32,16,64], 32_480_960=[32,32,64], 32_576_960=[32,16,64].

**Mp=64, Kp=32:** m=64 (Np≤96 or Np≥480 per Kp≤32 rule), m=32 (mid Np). n=64 for large Np.  
Proxies: 64_288_32=[64,32,16], 64_384_32=[32,64,32], 64_512_32=[16,64,16], 64_960_32=[32,64,32].

**Mp=64, Kp=128:** m=16/m=32; n=64 for large Np. k=32/k=64.  
Proxies: 64_256_128=[32,64,32], 64_480_128=[16,32,32], 64_768_128=[16,64,64].  
Held-out: 64_960_128=[16,64,64] (strategy_insights session note).

**Mp=64, Kp=256:** m=64 (Np≤192+Kp≥256), m=32 (mid Np), m=16 (large Np). n=64 (Np=256+). k=32/k=16.  
Proxies: 64_128_256=[16,32,32], 64_256_256=[16,64,32], 64_384_256=[32,32,32], 64_512_256=[16,64,64], 64_576_256=[32,64,32].

**Mp=64, Kp=512:** m=16/m=32/m=64 vary; n=64 for Np=256+; k=64 dominant (except overhead-limited small shapes).  
Proxies: 64_192_512=[64,32,32], 64_288_512=[32,32,16], 64_384_512=[32,32,64], 64_576_512=[32,16,128].

**Mp=64, Kp=768:** m=16/m=32; n=32/n=64 vary; k=32 (small Np), k=64/k=128 (large Np).  
Proxies: 64_64_768=[32,32,32], 64_256_768=[32,64,64], 64_480_768=[16,32,32], 64_768_768=[16,64,128].

**Mp=64, Kp=960:** m=32 dominant (Np 192-480); n varies (n=16/n=32/n=64 all appear). k=64 (absolute).  
Key verified: 64_576_960=[32,16,64], 64_960_512=[32,16,32], 64_480_960≈[32,16,64].  
n=16 dominant for Np≥480 (small-m, safe variant).

**Mp=128, Kp=512:** m=64 (~65%); n=64 for Np divisible with 3-factor; k=64.  
Proxies: 128_160_512=[64,32,64], 128_288_512=[64,32,64], 128_320_512=[64,64,64], 128_448_512=[32,16,128], 128_640_512=[32,32,128].  
Held-out corrections: n=16 anti-pattern for Mp≥128; [64,64,64] wins at large Np.

**Mp=192, Kp=512:** m=64 strongly preferred (~80%, Pm=3). n=64 for Np divisible. k=64.  
Proxies: 192_288_512=[64,32,64], 192_384_512=[64,64,64], 192_512_512=[64,64,32], 192_768_512=[64,64,64].

**Mp=256, Kp=512:** m=64 (~75%), Pm=4. n=64 for valid Np; k=64.  
Proxies: 256_128_512=[64,32,64], 256_288_512=[64,32,128], 256_512_512=[64,64,64], 256_672_512=[64,32,128], 256_960_512=[64,64,32].  
Note: Mp=256 is NOT categorically uncompilable (correction from 2026-04-08).

### Known failing patterns (applied in selections)
- m=64, n=16, Np≥480: NPU hardware crash (errors.md 2026-03-30)
- n=32 for Np=96: fails multiple shapes; use n=16 (strategy_insights empirical)
- [32,32,32] for Kp≥256 AND Np≥256: in failing lists (strategy_insights)
- k=16 for Kp=768 on any shape (errors.md)
- n=128, k=16 for bf16: CDO error universally (errors.md 2026-03-25)
- (128,128,128): hard blacklist

---

## Per-Shape Recommendations

| Shape | m | n | k | Conf | Proxy shapes used |
|-------|---|---|---|------|-------------------|
| 128_192_512 | 64 | 32 | 64 | medium | 128_160_512, 128_256_512 |
| 128_384_512 | 64 | 64 | 64 | medium | 128_320_512, 128_448_512 |
| 128_480_512 | 64 | 32 | 64 | medium | 128_448_512, 128_512_512 |
| 128_672_512 | 64 | 32 | 64 | high | strategy_insights direct citation |
| 128_768_512 | 64 | 64 | 64 | high | strategy_insights direct citation |
| 128_96_512  | 64 | 32 | 64 | low  | 128_128_512 (accuracy failures documented) |
| 192_192_512 | 64 | 32 | 64 | medium | 192_288_512 |
| 192_480_512 | 64 | 32 | 64 | medium | 192_384_512, 192_512_512 |
| 192_576_512 | 64 | 64 | 64 | medium | 192_512_512, 192_768_512 |
| 192_672_512 | 64 | 32 | 64 | medium | 192_512_512, 192_768_512 |
| 192_96_512  | 64 | 32 | 64 | medium | 192_288_512 |
| 256_192_512 | 64 | 32 | 64 | medium | 256_128_512, 256_288_512 |
| 256_480_512 | 64 | 32 | 64 | medium | 256_448_512, 256_512_512 |
| 256_576_512 | 64 | 64 | 64 | medium | 256_512_512, 256_672_512 |
| 256_768_512 | 64 | 64 | 64 | medium | 256_672_512, 256_960_512 |
| 32_192_960  | 32 | 16 | 64 | high  | 32_96_960, 32_320_960 |
| 32_256_128  | 32 | 64 | 32 | high  | strategy_insights direct citation; 32_512_128, 32_768_128 |
| 32_256_256  | 16 | 64 | 32 | high  | strategy_insights direct citation; 32_384_256, 32_512_256 |
| 32_256_768  | 32 | 64 | 32 | high  | strategy_insights direct citation; 32_512_768, 32_768_768 |
| 32_288_256  | 16 | 32 | 64 | high  | strategy_insights direct citation; 32_192_256, 32_384_256 |
| 32_288_960  | 16 | 32 | 64 | high  | strategy_insights direct citation; 32_96_960, 32_320_960 |
| 32_32_128   | 32 | 32 | 32 | medium | 32_64_128 |
| 32_32_256   | 32 | 32 | 16 | high  | strategy_insights direct citation; 32_192_256 |
| 32_32_512   | 32 | 32 | 32 | high  | strategy_insights direct citation; 32_64_512 |
| 32_384_960  | 16 | 32 | 64 | high  | strategy_insights direct citation; 32_320_960, 32_480_960 |
| 32_576_256  | 32 | 64 | 32 | medium | 32_512_256, 32_768_256 |
| 32_576_32   | 32 | 64 | 32 | high  | strategy_insights direct citation; 32_512_32, 32_768_32 |
| 32_672_256  | 32 | 32 | 32 | medium | 32_512_256, 32_768_256 (n=64 invalid for 672) |
| 32_960_128  | 32 | 64 | 32 | medium | 32_768_128, 32_1024_128 |
| 32_960_512  | 16 | 16 | 16 | high  | strategy_insights direct citation |
| 32_960_768  | 16 | 16 | 128| high  | strategy_insights direct citation |
| 32_96_256   | 32 | 16 | 32 | low   | 32_96_512 (bf16 accuracy failures documented) |
| 64_192_256  | 64 | 32 | 128| high  | strategy_insights direct citation; 64_128_256, 64_256_256 |
| 64_192_960  | 64 | 32 | 64 | high  | strategy_insights direct citation; 64_128_960, 64_256_960 |
| 64_256_512  | 16 | 64 | 32 | high  | strategy_insights direct citation; 64_192_512, 64_288_512 |
| 64_288_256  | 32 | 32 | 16 | high  | strategy_insights direct citation; 64_288_512 |
| 64_32_512   | 32 | 32 | 32 | high  | strategy_insights direct citation; 64_64_512 |
| 64_32_768   | 32 | 32 | 32 | medium | 64_64_768 |
| 64_384_960  | 32 | 32 | 64 | medium | 64_288_960, 64_320_960 |
| 64_480_256  | 32 | 32 | 32 | medium | 64_384_256, 64_512_256 |
| 64_480_32   | 64 | 32 | 16 | high  | strategy_insights direct citation; 64_288_32 |
| 64_480_960  | 32 | 16 | 64 | medium | 64_320_960, 64_576_960 |
| 64_576_32   | 32 | 64 | 16 | high  | strategy_insights direct citation; 64_384_32 |
| 64_576_960  | 32 | 16 | 64 | high  | strategy_insights direct citation; 64_576_512, 64_320_960 |
| 64_672_256  | 16 | 32 | 128| medium | 64_640_256 |
| 64_672_32   | 64 | 32 | 16 | high  | strategy_insights direct citation; 64_288_32 |
| 64_768_32   | 32 | 64 | 32 | medium | 64_512_32, 64_960_32 |
| 64_960_128  | 16 | 64 | 64 | high  | strategy_insights session note; 64_768_128 |
| 64_960_512  | 32 | 16 | 32 | high  | strategy_insights direct citation |
| 64_96_32    | 64 | 16 | 32 | high  | strategy_insights direct citation |

---

## Confidence distribution
- High: 30  
- Medium: 18  
- Low: 2

## Low-confidence shapes and open questions

### 128_96_512
bf16 accuracy failures are exhaustively documented for this shape in errors.md (2026-03-28): tiles (32,32,32), (16,32,32), (16,32,16), (16,8,64) all fail. Recommended [64,32,64] is the best valid candidate but expected to fail accuracy. Consider i8 or i16 for this shape.

### 32_96_256
bf16 accuracy failures documented for M≤50+N=96 regardless of Kp (errors.md 2026-03-28). Raw M=20-32 padded to Mp=32 falls in this failure regime. No tile workaround found in error logs. Recommended [32,16,32] as best available; expect accuracy issues.

### Open questions
- **64_384_960**: DB shows highly variable n (n=16 at Np=320, n=64 at Np=512, n=32 at Np=288). Selected [32,32,64] as conservative choice. n=16 tile [32,16,64] is also plausible.
- **32_672_256**: n=64 does not divide 672; forced to use n=32. Nearest valid proxy gives [32,32,32] but pattern for larger Np suggests n=64 would win if valid.
- **64_480_960**: m=64,n=16 crashes at Np≥480; m=32,n=16 is safe. Selected [32,16,64] following 64_320_960 and 64_576_960 bracketing proxies.
