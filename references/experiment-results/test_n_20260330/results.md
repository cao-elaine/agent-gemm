# SmolVLA NPU Experiment Results
**Date**: 2026-03-30  |  **dtype**: bf16  |  **Trials per group**: 3

## Summary

| Metric | Value |
|--------|-------|
| Shapes in file | 291 |
| Shapes run | 282 |
| Pre-screened (skip) | 9 |
| Baseline 0 wins | 39 |
| Baseline 1 wins | 39 |
| Agentic wins | 51 |
| Ties (within 2%) | 118 |
| Inconclusive | 35 |
| Mean Δ agentic vs Baseline 0 | nan% |
| Mean Δ agentic vs Baseline 1 | nan% |
| Mean Δ Baseline 1 vs Baseline 0 | nan% |

## Per-Shape Results

### 20×32×96  —  `action_in_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,96,32) | (32,96,32) | (32,96,32) |
| Tile | (32,32,32) | (32,16,16) | (32,16,16) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 127.4 | 122.5 | 125.0 |
| Trial 2 avg (µs) | 125.9 | 123.6 | 124.9 |
| Trial 3 avg (µs) | 125.6 | 126.6 | 128.9 |
| **Mean avg (µs)** | **126.3** | **124.2** | **126.2** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: AGENTIC**
Δ(B2 vs B0)=+0.1µs (+0.1%)   Δ(AG vs B0)=-3.6µs (-2.3%)   Δ(AG vs B1)=-3.7µs (-2.4%)

---

### 20×32×192  —  `action_in_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,192,32) | (32,192,32) | (32,192,32) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 126.7 | 126.9 | 125.5 |
| Trial 2 avg (µs) | 127.0 | 123.8 | 125.6 |
| Trial 3 avg (µs) | 129.1 | 127.2 | 125.1 |
| **Mean avg (µs)** | **127.6** | **126.0** | **125.4** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: AGENTIC**
Δ(B2 vs B0)=-0.8µs (-0.5%)   Δ(AG vs B0)=-8.2µs (-5.3%)   Δ(AG vs B1)=-7.4µs (-4.8%)

---

### 20×32×288  —  `action_in_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,288,32) | (32,288,32) | (32,288,32) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 129.2 | 129.6 | 127.8 |
| Trial 2 avg (µs) | 129.0 | 133.0 | 132.8 |
| Trial 3 avg (µs) | 132.7 | 128.7 | 133.0 |
| **Mean avg (µs)** | **130.3** | **130.4** | **131.2** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-2.8µs (-1.7%)   Δ(AG vs B0)=-4.0µs (-2.4%)   Δ(AG vs B1)=-1.2µs (-0.8%)

---

### 20×32×384  —  `action_in_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,384,32) | (32,384,32) | (32,384,32) |
| Tile | (32,32,32) | (16,64,16) | (16,64,16) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 158.8 | 127.7 | 130.2 |
| Trial 2 avg (µs) | 152.5 | 128.5 | 128.5 |
| Trial 3 avg (µs) | 152.3 | 127.2 | 128.0 |
| **Mean avg (µs)** | **154.5** | **127.8** | **128.9** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: BASELINE 2**
Δ(B2 vs B0)=-30.2µs (-16.4%)   Δ(AG vs B0)=-25.8µs (-14.0%)   Δ(AG vs B1)=+4.4µs (+2.9%)

---

### 20×32×480  —  `action_in_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,480,32) | (32,480,32) | (32,480,32) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 135.3 | 138.0 | 137.1 |
| Trial 2 avg (µs) | 133.7 | 130.4 | 133.6 |
| Trial 3 avg (µs) | 136.0 | 136.1 | 137.0 |
| **Mean avg (µs)** | **135.0** | **134.8** | **135.9** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=+2.8µs (+1.8%)   Δ(AG vs B0)=+8.2µs (+5.2%)   Δ(AG vs B1)=+5.5µs (+3.4%)

---

### 20×32×576  —  `action_in_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,576,32) | (32,576,32) | (32,576,32) |
| Tile | (32,32,32) | (32,64,32) | (32,64,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 139.2 | 131.7 | 141.9 |
| Trial 2 avg (µs) | 142.6 | 131.5 | 132.8 |
| Trial 3 avg (µs) | 139.7 | 132.3 | 140.7 |
| **Mean avg (µs)** | **140.5** | **131.8** | **138.5** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: BASELINE 2**
Δ(B2 vs B0)=-5.2µs (-3.2%)   Δ(AG vs B0)=+2.7µs (+1.7%)   Δ(AG vs B1)=+7.9µs (+5.1%)

---

### 20×32×672  —  `action_in_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,672,32) | (32,672,32) | (32,672,32) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 150.2 | 151.5 | 148.1 |
| Trial 2 avg (µs) | 144.4 | 146.3 | 145.8 |
| Trial 3 avg (µs) | 145.6 | 145.9 | 147.7 |
| **Mean avg (µs)** | **146.7** | **147.9** | **147.2** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: BASELINE 2**
Δ(B2 vs B0)=-3.5µs (-2.0%)   Δ(AG vs B0)=+3.8µs (+2.2%)   Δ(AG vs B1)=+7.3µs (+4.3%)

---

### 20×32×720  —  `action_in_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,768,32) | (32,768,32) | (32,768,32) |
| Tile | (32,32,32) | (32,64,32) | (32,64,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 194.8 | 154.7 | 155.7 |
| Trial 2 avg (µs) | 196.7 | 154.5 | 160.0 |
| Trial 3 avg (µs) | 195.6 | 155.0 | 154.0 |
| **Mean avg (µs)** | **195.7** | **154.7** | **156.5** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: BASELINE 2**
Δ(B2 vs B0)=-35.7µs (-16.1%)   Δ(AG vs B0)=-29.7µs (-13.4%)   Δ(AG vs B1)=+5.9µs (+3.2%)

---

### 20×96×32  —  `action_out_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,32,128) | (32,32,128) | (32,32,128) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 123.2 | 119.2 | 121.5 |
| Trial 2 avg (µs) | 122.8 | 116.4 | 118.9 |
| Trial 3 avg (µs) | 124.8 | 120.9 | 120.3 |
| **Mean avg (µs)** | **123.6** | **118.8** | **120.2** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=+2.0µs (+1.4%)   Δ(AG vs B0)=+0.8µs (+0.6%)   Δ(AG vs B1)=-1.2µs (-0.8%)

---

### 20×96×256  —  `gate_proj, up_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,256,128) | (32,256,128) | (32,256,128) |
| Tile | (32,32,32) | (32,64,32) | (32,64,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 141.1 | 136.4 | 130.6 |
| Trial 2 avg (µs) | 141.0 | 131.6 | 130.2 |
| Trial 3 avg (µs) | 142.2 | 131.4 | 136.1 |
| **Mean avg (µs)** | **141.4** | **133.1** | **132.3** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: BASELINE 2**
Δ(B2 vs B0)=-11.9µs (-7.0%)   Δ(AG vs B0)=-8.4µs (-4.9%)   Δ(AG vs B1)=+3.5µs (+2.2%)

---

### 20×96×960  —  `o_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,960,128) | (32,960,128) | (32,960,128) |
| Tile | (32,32,32) | (16,64,64) | (16,64,64) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 188.5 | 181.2 | 178.1 |
| Trial 2 avg (µs) | 185.0 | 180.2 | 178.6 |
| Trial 3 avg (µs) | 187.3 | 178.6 | 175.4 |
| **Mean avg (µs)** | **186.9** | **180.0** | **177.3** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-6.9µs (-3.2%)   Δ(AG vs B0)=-8.1µs (-3.8%)   Δ(AG vs B1)=-1.2µs (-0.6%)

---

### 20×192×32  —  `action_out_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,32,256) | (32,32,256) | (32,32,256) |
| Tile | (32,32,32) | (32,16,32) | (32,16,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 124.2 | 125.5 | 123.8 |
| Trial 2 avg (µs) | 127.4 | 125.0 | 124.2 |
| Trial 3 avg (µs) | 128.2 | 124.4 | 122.8 |
| **Mean avg (µs)** | **126.6** | **125.0** | **123.6** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: AGENTIC**
Δ(B2 vs B0)=-0.5µs (-0.3%)   Δ(AG vs B0)=-6.2µs (-4.1%)   Δ(AG vs B1)=-5.7µs (-3.8%)

---

### 20×192×256  —  `gate_proj, up_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,256,256) | (32,256,256) | (32,256,256) |
| Tile | (32,32,32) | (16,64,32) | (16,64,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 143.3 | 132.6 | 134.9 |
| Trial 2 avg (µs) | 142.8 | 131.7 | 131.3 |
| Trial 3 avg (µs) | 144.4 | 131.2 | 134.2 |
| **Mean avg (µs)** | **143.5** | **131.8** | **133.5** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: AGENTIC**
Δ(B2 vs B0)=+3.4µs (+2.0%)   Δ(AG vs B0)=-10.8µs (-6.4%)   Δ(AG vs B1)=-14.2µs (-8.3%)

---

### 20×192×960  —  `o_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,960,256) | (32,960,256) | (32,960,256) |
| Tile | (16,32,32) | (16,64,32) | (16,64,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 241.9 | 221.5 | 223.6 |
| Trial 2 avg (µs) | FAIL | 226.5 | 221.6 |
| Trial 3 avg (µs) | FAIL | 221.1 | 224.4 |
| **Mean avg (µs)** | **241.9** | **223.1** | **223.2** |
| Pass count | 1/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-18.6µs (-6.8%)   Δ(AG vs B0)=-17.7µs (-6.5%)   Δ(AG vs B1)=+0.9µs (+0.3%)

---

### 20×256×96  —  `down_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,96,256) | (32,96,256) | (32,96,256) |
| Tile | (32,32,32) | (32,16,32) | (32,16,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 134.2 | 122.5 | 125.3 |
| Trial 2 avg (µs) | 133.6 | 125.7 | 122.4 |
| Trial 3 avg (µs) | 130.5 | 122.9 | 123.4 |
| **Mean avg (µs)** | **132.7** | **123.7** | **123.7** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: AGENTIC**
Δ(B2 vs B0)=-5.6µs (-3.5%)   Δ(AG vs B0)=-14.0µs (-8.8%)   Δ(AG vs B1)=-8.3µs (-5.4%)

---

### 20×256×192  —  `down_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,192,256) | (32,192,256) | (32,192,256) |
| Tile | (32,32,32) | (32,32,16) | (32,32,16) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 135.4 | 134.7 | 135.0 |
| Trial 2 avg (µs) | 132.7 | 133.1 | 134.3 |
| Trial 3 avg (µs) | 132.5 | 135.2 | 136.3 |
| **Mean avg (µs)** | **133.5** | **134.3** | **135.2** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: AGENTIC**
Δ(B2 vs B0)=-7.1µs (-4.1%)   Δ(AG vs B0)=-15.9µs (-9.2%)   Δ(AG vs B1)=-8.8µs (-5.3%)

---

### 20×256×288  —  `down_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,288,256) | (32,288,256) | (32,288,256) |
| Tile | (32,32,32) | (16,32,64) | (16,32,64) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 157.0 | 152.4 | 154.9 |
| Trial 2 avg (µs) | 150.9 | 150.2 | 154.1 |
| Trial 3 avg (µs) | 154.5 | 163.7 | 157.1 |
| **Mean avg (µs)** | **154.1** | **155.4** | **155.3** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=+1.3µs (+0.7%)   Δ(AG vs B0)=+7.2µs (+4.0%)   Δ(AG vs B1)=+5.9µs (+3.3%)

---

### 20×256×384  —  `down_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,384,256) | (32,384,256) | (32,384,256) |
| Tile | (32,32,32) | (32,64,32) | (32,64,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 160.6 | 150.4 | 149.3 |
| Trial 2 avg (µs) | 157.4 | 154.7 | 149.2 |
| Trial 3 avg (µs) | 159.1 | 154.3 | 151.6 |
| **Mean avg (µs)** | **159.0** | **153.2** | **150.1** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-6.5µs (-3.5%)   Δ(AG vs B0)=-8.7µs (-4.7%)   Δ(AG vs B1)=-2.2µs (-1.2%)

---

### 20×256×480  —  `down_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,480,256) | (32,480,256) | (32,480,256) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 176.7 | 174.7 | 175.2 |
| Trial 2 avg (µs) | 176.9 | 171.2 | 176.2 |
| Trial 3 avg (µs) | 174.5 | 177.5 | 173.5 |
| **Mean avg (µs)** | **176.0** | **174.5** | **174.9** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=+0.3µs (+0.1%)   Δ(AG vs B0)=+0.7µs (+0.3%)   Δ(AG vs B1)=+0.4µs (+0.2%)

---

### 20×256×576  —  `down_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,576,256) | (32,576,256) | (32,576,256) |
| Tile | (32,32,32) | (16,64,16) | (16,64,16) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 192.2 | 187.9 | 187.5 |
| Trial 2 avg (µs) | 191.3 | 186.7 | 189.8 |
| Trial 3 avg (µs) | 194.3 | 189.8 | 187.7 |
| **Mean avg (µs)** | **192.6** | **188.1** | **188.3** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-4.0µs (-1.8%)   Δ(AG vs B0)=-4.7µs (-2.1%)   Δ(AG vs B1)=-0.7µs (-0.3%)

---

### 20×256×672  —  `down_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,672,256) | (32,672,256) | (32,672,256) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 204.5 | 206.5 | 205.2 |
| Trial 2 avg (µs) | 205.4 | 214.8 | 204.1 |
| Trial 3 avg (µs) | 212.6 | 203.6 | 215.4 |
| **Mean avg (µs)** | **207.5** | **208.3** | **208.2** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=+2.3µs (+1.0%)   Δ(AG vs B0)=-0.1µs (-0.0%)   Δ(AG vs B1)=-2.4µs (-1.0%)

---

### 20×256×720  —  `down_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,768,256) | (32,768,256) | (32,768,256) |
| Tile | (32,32,32) | (32,64,32) | (32,64,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 195.0 | 162.3 | 163.7 |
| Trial 2 avg (µs) | 199.5 | 163.2 | 162.9 |
| Trial 3 avg (µs) | 198.4 | 163.5 | 165.6 |
| **Mean avg (µs)** | **197.6** | **163.0** | **164.1** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: BASELINE 2**
Δ(B2 vs B0)=-34.5µs (-15.0%)   Δ(AG vs B0)=-21.7µs (-9.4%)   Δ(AG vs B1)=+12.8µs (+6.5%)

---

### 20×288×32  —  `action_out_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,32,512) | (32,32,512) | (32,32,512) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 132.3 | 134.3 | 129.4 |
| Trial 2 avg (µs) | 128.2 | 132.7 | 134.4 |
| Trial 3 avg (µs) | 135.3 | 132.2 | 126.1 |
| **Mean avg (µs)** | **131.9** | **133.1** | **130.0** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=+10.4µs (+6.8%)   Δ(AG vs B0)=-1.0µs (-0.7%)   Δ(AG vs B1)=-11.4µs (-7.0%)

---

### 20×288×256  —  `gate_proj, up_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,256,512) | (32,256,512) | (32,256,512) |
| Tile | (32,32,32) | (16,64,32) | (16,64,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 155.8 | 148.4 | 148.0 |
| Trial 2 avg (µs) | 157.3 | 147.0 | 150.9 |
| Trial 3 avg (µs) | 155.1 | 145.5 | 146.1 |
| **Mean avg (µs)** | **156.0** | **147.0** | **148.3** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-5.8µs (-3.1%)   Δ(AG vs B0)=-2.9µs (-1.6%)   Δ(AG vs B1)=+2.9µs (+1.6%)

---

### 20×288×960  —  `o_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,960,512) | (32,960,512) | (32,960,512) |
| Tile | (16,32,32) | (16,16,16) | (16,16,16) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | FAIL | 303.7 | 301.0 |
| Trial 2 avg (µs) | 352.1 | 307.8 | 303.2 |
| Trial 3 avg (µs) | FAIL | 306.0 | 307.0 |
| **Mean avg (µs)** | **352.1** | **305.8** | **303.8** |
| Pass count | 1/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-42.7µs (-8.0%)   Δ(AG vs B0)=-42.0µs (-7.9%)   Δ(AG vs B1)=+0.7µs (+0.1%)

---

### 20×384×32  —  `action_out_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,32,512) | (32,32,512) | (32,32,512) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 131.9 | 133.9 | 133.5 |
| Trial 2 avg (µs) | 133.0 | 133.0 | 130.9 |
| Trial 3 avg (µs) | 130.2 | 128.2 | 130.4 |
| **Mean avg (µs)** | **131.7** | **131.7** | **131.6** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: BASELINE 0**
Δ(B2 vs B0)=+9.0µs (+5.9%)   Δ(AG vs B0)=+13.7µs (+8.9%)   Δ(AG vs B1)=+4.7µs (+2.9%)

---

### 20×384×256  —  `gate_proj, up_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,256,512) | (32,256,512) | (32,256,512) |
| Tile | (32,32,32) | (16,64,32) | (16,64,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 158.3 | 149.4 | 148.1 |
| Trial 2 avg (µs) | 153.0 | 147.1 | 147.9 |
| Trial 3 avg (µs) | 157.6 | 145.6 | 146.2 |
| **Mean avg (µs)** | **156.3** | **147.4** | **147.4** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-8.1µs (-4.4%)   Δ(AG vs B0)=-9.2µs (-5.0%)   Δ(AG vs B1)=-1.0µs (-0.6%)

---

### 20×384×960  —  `o_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,960,512) | (32,960,512) | (32,960,512) |
| Tile | (16,32,32) | (16,16,16) | (16,16,16) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | FAIL | 304.6 | 311.5 |
| Trial 2 avg (µs) | FAIL | 304.9 | 303.2 |
| Trial 3 avg (µs) | FAIL | 304.2 | 307.0 |
| **Mean avg (µs)** | **FAIL** | **304.6** | **307.3** |
| Pass count | 0/3 | 3/3 | 3/3 |

**Winner: INCONCLUSIVE**
Δ(AG vs B1)=-4.4µs (-0.8%)

---

### 20×480×32  —  `action_out_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,32,512) | (32,32,512) | (32,32,512) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 132.0 | 130.3 | 130.4 |
| Trial 2 avg (µs) | 130.5 | 132.7 | 127.9 |
| Trial 3 avg (µs) | 128.4 | 132.1 | 132.8 |
| **Mean avg (µs)** | **130.3** | **131.7** | **130.4** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=+1.1µs (+0.7%)   Δ(AG vs B0)=+2.5µs (+1.6%)   Δ(AG vs B1)=+1.5µs (+0.9%)

---

### 20×480×256  —  `gate_proj, up_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,256,512) | (32,256,512) | (32,256,512) |
| Tile | (32,32,32) | (16,64,32) | (16,64,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 153.0 | 148.1 | 148.4 |
| Trial 2 avg (µs) | 157.1 | 144.2 | 148.6 |
| Trial 3 avg (µs) | 155.0 | 147.9 | 144.3 |
| **Mean avg (µs)** | **155.0** | **146.7** | **147.1** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: BASELINE 2**
Δ(B2 vs B0)=-9.1µs (-4.9%)   Δ(AG vs B0)=-3.3µs (-1.8%)   Δ(AG vs B1)=+5.8µs (+3.3%)

---

### 20×480×960  —  `o_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,960,512) | (32,960,512) | (32,960,512) |
| Tile | (16,32,32) | (16,16,16) | (16,16,16) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 354.4 | 306.6 | 305.0 |
| Trial 2 avg (µs) | FAIL | 311.7 | 309.6 |
| Trial 3 avg (µs) | FAIL | 308.9 | 307.6 |
| **Mean avg (µs)** | **354.4** | **309.1** | **307.4** |
| Pass count | 1/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-68.0µs (-10.1%)   Δ(AG vs B0)=-59.5µs (-8.9%)   Δ(AG vs B1)=+8.5µs (+1.4%)

---

### 20×576×32  —  `action_out_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,32,768) | (32,32,768) | (32,32,768) |
| Tile | (32,32,32) | (16,16,256) | (16,16,256) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 139.6 | 133.1 | 130.4 |
| Trial 2 avg (µs) | 142.2 | 127.0 | 130.0 |
| Trial 3 avg (µs) | 135.5 | 126.7 | 133.0 |
| **Mean avg (µs)** | **139.1** | **128.9** | **131.1** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-20.5µs (-11.4%)   Δ(AG vs B0)=-22.2µs (-12.4%)   Δ(AG vs B1)=-1.7µs (-1.1%)

---

### 20×576×256  —  `gate_proj, up_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,256,768) | (32,256,768) | (32,256,768) |
| Tile | (32,32,32) | (32,64,32) | (32,64,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 166.8 | 156.1 | 158.9 |
| Trial 2 avg (µs) | 162.0 | 161.2 | 158.3 |
| Trial 3 avg (µs) | 167.0 | 156.7 | 160.3 |
| **Mean avg (µs)** | **165.3** | **158.0** | **159.2** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-10.1µs (-5.1%)   Δ(AG vs B0)=-8.1µs (-4.1%)   Δ(AG vs B1)=+2.0µs (+1.1%)

---

### 20×576×960  —  `o_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,960,768) | (32,960,768) | (32,960,768) |
| Tile | (16,32,32) | (16,16,128) | (16,16,128) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | FAIL | 301.6 | 308.6 |
| Trial 2 avg (µs) | FAIL | 301.5 | 300.7 |
| Trial 3 avg (µs) | FAIL | 305.1 | 307.8 |
| **Mean avg (µs)** | **FAIL** | **302.8** | **305.7** |
| Pass count | 0/3 | 3/3 | 3/3 |

**Winner: INCONCLUSIVE**
Δ(AG vs B1)=+16.5µs (+1.8%)

---

### 20×672×32  —  `action_out_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,32,768) | (32,32,768) | (32,32,768) |
| Tile | (32,32,32) | (16,16,256) | (16,16,256) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 132.4 | 128.2 | 127.0 |
| Trial 2 avg (µs) | 132.8 | 126.2 | 125.5 |
| Trial 3 avg (µs) | 134.2 | 123.2 | 125.7 |
| **Mean avg (µs)** | **133.1** | **125.9** | **126.1** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: BASELINE 2**
Δ(B2 vs B0)=-12.5µs (-7.6%)   Δ(AG vs B0)=-8.2µs (-5.0%)   Δ(AG vs B1)=+4.3µs (+2.8%)

---

### 20×672×256  —  `gate_proj, up_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,256,768) | (32,256,768) | (32,256,768) |
| Tile | (32,32,32) | (32,64,32) | (32,64,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 158.6 | 159.6 | 157.4 |
| Trial 2 avg (µs) | 197.3 | 158.1 | 158.0 |
| Trial 3 avg (µs) | 197.4 | 157.9 | 156.5 |
| **Mean avg (µs)** | **184.5** | **158.6** | **157.3** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-31.4µs (-14.2%)   Δ(AG vs B0)=-30.6µs (-13.8%)   Δ(AG vs B1)=+0.8µs (+0.4%)

---

### 20×672×960  —  `o_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,960,768) | (32,960,768) | (32,960,768) |
| Tile | (16,32,32) | (16,16,128) | (16,16,128) |
| Tile source | fixed | profiling | profiling |
| Pad impl | numpy_pad | numpy_pad | numpy_pad |
| Trial 1 avg (µs) | FAIL | 304.1 | 303.5 |
| Trial 2 avg (µs) | 467.3 | 304.6 | 303.4 |
| Trial 3 avg (µs) | FAIL | 301.6 | 304.2 |
| **Mean avg (µs)** | **467.3** | **303.4** | **303.7** |
| Pass count | 1/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-123.7µs (-9.0%)   Δ(AG vs B0)=-145.5µs (-10.6%)   Δ(AG vs B1)=-21.7µs (-1.7%)

---

### 20×720×32  —  `action_out_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,32,768) | (32,32,768) | (32,32,768) |
| Tile | (32,32,32) | (16,16,256) | (16,16,256) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 139.5 | 132.3 | 129.3 |
| Trial 2 avg (µs) | 136.3 | 130.9 | 125.3 |
| Trial 3 avg (µs) | 140.0 | 131.5 | 130.0 |
| **Mean avg (µs)** | **138.6** | **131.6** | **128.2** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: AGENTIC**
Δ(B2 vs B0)=+1.6µs (+1.0%)   Δ(AG vs B0)=-7.4µs (-4.6%)   Δ(AG vs B1)=-9.1µs (-5.5%)

---

### 20×720×256  —  `gate_proj, up_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,256,768) | (32,256,768) | (32,256,768) |
| Tile | (32,32,32) | (32,64,32) | (32,64,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 162.1 | 154.6 | 158.4 |
| Trial 2 avg (µs) | 157.3 | 159.6 | 157.8 |
| Trial 3 avg (µs) | 158.8 | 158.8 | 158.3 |
| **Mean avg (µs)** | **159.4** | **157.7** | **158.2** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: BASELINE 2**
Δ(B2 vs B0)=-5.8µs (-3.0%)   Δ(AG vs B0)=-1.7µs (-0.8%)   Δ(AG vs B1)=+4.2µs (+2.2%)

---

### 20×720×960  —  `o_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,960,768) | (32,960,768) | (32,960,768) |
| Tile | (16,32,32) | (16,16,128) | (16,16,128) |
| Tile source | fixed | profiling | profiling |
| Pad impl | numpy_pad | numpy_pad | numpy_pad |
| Trial 1 avg (µs) | FAIL | 301.8 | 302.3 |
| Trial 2 avg (µs) | FAIL | 308.8 | 306.4 |
| Trial 3 avg (µs) | 469.3 | 306.2 | 307.3 |
| **Mean avg (µs)** | **469.3** | **305.6** | **305.3** |
| Pass count | 1/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-160.1µs (-14.7%)   Δ(AG vs B0)=-144.9µs (-13.3%)   Δ(AG vs B1)=+15.1µs (+1.6%)

---

### 20×960×96  —  `q_proj`

**Pre-screened skip**: bf16_accuracy_small_MN_large_K

---

### 20×960×192  —  `q_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,192,960) | (32,192,960) | (32,192,960) |
| Tile | (16,32,32) | (32,32,64) | (32,32,64) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 167.5 | 169.6 | 174.2 |
| Trial 2 avg (µs) | 174.7 | 172.0 | 169.6 |
| Trial 3 avg (µs) | 175.8 | 167.9 | 166.1 |
| **Mean avg (µs)** | **FAIL** | **169.8** | **170.0** |
| Pass count | 0/3 | 3/3 | 3/3 |

**Winner: INCONCLUSIVE**
Δ(AG vs B1)=+3.0µs (+1.5%)

---

### 20×960×288  —  `q_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,288,960) | (32,288,960) | (32,288,960) |
| Tile | (16,32,32) | (16,32,64) | (16,32,64) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 242.9 | 242.5 | 251.8 |
| Trial 2 avg (µs) | 245.3 | 243.4 | 242.5 |
| Trial 3 avg (µs) | 243.5 | 245.3 | 244.1 |
| **Mean avg (µs)** | **FAIL** | **243.7** | **246.1** |
| Pass count | 0/3 | 3/3 | 3/3 |

**Winner: INCONCLUSIVE**
Δ(AG vs B1)=+8.7µs (+2.0%)

---

### 20×960×384  —  `q_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,384,960) | (32,384,960) | (32,384,960) |
| Tile | (16,32,32) | (16,32,64) | (16,32,64) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 192.9 | 191.8 | 190.8 |
| Trial 2 avg (µs) | 193.5 | 192.2 | 191.2 |
| Trial 3 avg (µs) | 192.5 | 189.7 | 191.8 |
| **Mean avg (µs)** | **FAIL** | **191.2** | **191.3** |
| Pass count | 0/3 | 3/3 | 3/3 |

**Winner: INCONCLUSIVE**
Δ(AG vs B1)=-4.3µs (-1.0%)

---

### 20×960×480  —  `q_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,480,960) | (32,480,960) | (32,480,960) |
| Tile | (16,32,32) | (32,32,64) | (32,32,64) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 303.5 | 301.0 | 304.0 |
| Trial 2 avg (µs) | 306.9 | 302.3 | 317.8 |
| Trial 3 avg (µs) | 307.8 | 306.2 | 305.9 |
| **Mean avg (µs)** | **FAIL** | **303.2** | **309.2** |
| Pass count | 0/3 | 3/3 | 3/3 |

**Winner: INCONCLUSIVE**
Δ(AG vs B1)=+1.2µs (+0.2%)

---

### 20×960×576  —  `q_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,576,960) | (32,576,960) | (32,576,960) |
| Tile | (16,32,32) | (32,16,64) | (32,16,64) |
| Tile source | fixed | profiling | profiling |
| Pad impl | numpy_pad | numpy_pad | numpy_pad |
| Trial 1 avg (µs) | 366.6 | 278.6 | 279.4 |
| Trial 2 avg (µs) | 367.7 | 278.3 | 275.1 |
| Trial 3 avg (µs) | 368.3 | 276.1 | 275.5 |
| **Mean avg (µs)** | **FAIL** | **277.7** | **276.7** |
| Pass count | 0/3 | 3/3 | 3/3 |

**Winner: INCONCLUSIVE**
Δ(AG vs B1)=-0.6µs (-0.1%)

---

### 20×960×672  —  `q_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,672,960) | (32,672,960) | (32,672,960) |
| Tile | (16,32,32) | (32,32,64) | (32,32,64) |
| Tile source | fixed | profiling | profiling |
| Pad impl | numpy_pad | numpy_pad | numpy_pad |
| Trial 1 avg (µs) | 426.3 | 428.5 | 425.7 |
| Trial 2 avg (µs) | 424.6 | 429.3 | 557.2 |
| Trial 3 avg (µs) | 424.1 | 430.2 | 557.6 |
| **Mean avg (µs)** | **FAIL** | **429.4** | **513.5** |
| Pass count | 0/3 | 3/3 | 3/3 |

**Winner: INCONCLUSIVE**
Δ(AG vs B1)=+89.7µs (+6.7%)

---

### 20×960×720  —  `q_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,768,960) | (32,768,960) | (32,768,960) |
| Tile | (16,32,32) | (16,32,32) | (16,32,32) |
| Tile source | fixed | fallback_fixed | heuristic |
| Pad impl | numpy_pad | numpy_pad | numpy_pad |
| Trial 1 avg (µs) | 364.7 | 366.5 | 353.8 |
| Trial 2 avg (µs) | 359.4 | 371.7 | 358.1 |
| Trial 3 avg (µs) | 355.2 | 375.1 | 362.2 |
| **Mean avg (µs)** | **FAIL** | **FAIL** | **FAIL** |
| Pass count | 0/3 | 0/3 | 0/3 |

**Winner: INCONCLUSIVE**

---

### 30×32×96  —  `action_in_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,96,32) | (32,96,32) | (32,96,32) |
| Tile | (32,32,32) | (32,16,16) | (32,16,16) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 152.3 | 163.7 | 164.7 |
| Trial 2 avg (µs) | 152.3 | 161.8 | 164.0 |
| Trial 3 avg (µs) | 152.1 | 161.3 | 159.6 |
| **Mean avg (µs)** | **152.2** | **161.3** | **159.6** |
| Pass count | 3/3 | 1/3 | 1/3 |

**Winner: BASELINE 0**
Δ(B2 vs B0)=+6.2µs (+3.5%)   Δ(AG vs B0)=+9.1µs (+5.2%)   Δ(AG vs B1)=+2.9µs (+1.6%)

---

### 30×32×192  —  `action_in_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,192,32) | (32,192,32) | (32,192,32) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 162.7 | 164.2 | 169.5 |
| Trial 2 avg (µs) | 174.7 | 168.7 | 174.5 |
| Trial 3 avg (µs) | 165.1 | 163.7 | 162.5 |
| **Mean avg (µs)** | **167.5** | **165.5** | **168.9** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-3.5µs (-1.8%)   Δ(AG vs B0)=-3.0µs (-1.6%)   Δ(AG vs B1)=+0.4µs (+0.2%)

---

### 30×32×288  —  `action_in_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,288,32) | (32,288,32) | (32,288,32) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 167.1 | 160.8 | 160.1 |
| Trial 2 avg (µs) | 167.2 | 167.3 | 168.2 |
| Trial 3 avg (µs) | 168.1 | 160.7 | 160.4 |
| **Mean avg (µs)** | **167.5** | **163.0** | **162.9** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=+1.9µs (+1.0%)   Δ(AG vs B0)=+2.1µs (+1.1%)   Δ(AG vs B1)=+0.2µs (+0.1%)

---

### 30×32×384  —  `action_in_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,384,32) | (32,384,32) | (32,384,32) |
| Tile | (32,32,32) | (16,64,16) | (16,64,16) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 183.4 | 183.5 | 195.0 |
| Trial 2 avg (µs) | 182.9 | 183.0 | 184.4 |
| Trial 3 avg (µs) | 189.3 | 195.5 | 192.6 |
| **Mean avg (µs)** | **185.2** | **187.3** | **190.7** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=+2.1µs (+1.0%)   Δ(AG vs B0)=+26.6µs (+12.6%)   Δ(AG vs B1)=+24.5µs (+11.4%)

---

### 30×32×480  —  `action_in_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,480,32) | (32,480,32) | (32,480,32) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 182.3 | 169.3 | 180.9 |
| Trial 2 avg (µs) | 171.8 | 181.8 | 173.0 |
| Trial 3 avg (µs) | 183.3 | 184.8 | 173.1 |
| **Mean avg (µs)** | **179.1** | **178.6** | **175.7** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-1.4µs (-0.7%)   Δ(AG vs B0)=-5.3µs (-2.5%)   Δ(AG vs B1)=-4.0µs (-1.9%)

---

### 30×32×576  —  `action_in_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,576,32) | (32,576,32) | (32,576,32) |
| Tile | (32,32,32) | (32,64,32) | (32,64,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 180.0 | 161.3 | 163.6 |
| Trial 2 avg (µs) | 180.9 | 162.9 | 210.9 |
| Trial 3 avg (µs) | 182.2 | 164.6 | 179.6 |
| **Mean avg (µs)** | **181.0** | **163.0** | **184.7** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: BASELINE 2**
Δ(B2 vs B0)=-17.3µs (-8.4%)   Δ(AG vs B0)=+3.8µs (+1.9%)   Δ(AG vs B1)=+21.2µs (+11.2%)

---

### 30×32×672  —  `action_in_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,672,32) | (32,672,32) | (32,672,32) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 184.8 | 177.4 | 183.9 |
| Trial 2 avg (µs) | 182.6 | 185.4 | 184.0 |
| Trial 3 avg (µs) | 182.4 | 184.8 | 183.1 |
| **Mean avg (µs)** | **183.3** | **182.5** | **183.7** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-2.2µs (-1.1%)   Δ(AG vs B0)=+3.2µs (+1.5%)   Δ(AG vs B1)=+5.4µs (+2.6%)

---

### 30×32×720  —  `action_in_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,768,32) | (32,768,32) | (32,768,32) |
| Tile | (32,32,32) | (32,64,32) | (32,64,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 238.8 | 190.5 | 184.5 |
| Trial 2 avg (µs) | 238.8 | 186.9 | 192.0 |
| Trial 3 avg (µs) | 236.0 | 185.7 | 186.1 |
| **Mean avg (µs)** | **237.8** | **187.7** | **187.5** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: AGENTIC**
Δ(B2 vs B0)=-48.9µs (-18.0%)   Δ(AG vs B0)=-59.2µs (-21.8%)   Δ(AG vs B1)=-10.3µs (-4.6%)

---

### 30×96×32  —  `action_out_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,32,128) | (32,32,128) | (32,32,128) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 132.0 | 129.8 | 130.2 |
| Trial 2 avg (µs) | 131.7 | 132.5 | 125.5 |
| Trial 3 avg (µs) | 132.1 | 131.0 | 130.9 |
| **Mean avg (µs)** | **131.9** | **131.1** | **128.9** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: AGENTIC**
Δ(B2 vs B0)=+0.6µs (+0.4%)   Δ(AG vs B0)=-4.9µs (-3.1%)   Δ(AG vs B1)=-5.5µs (-3.5%)

---

### 30×96×256  —  `gate_proj, up_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,256,128) | (32,256,128) | (32,256,128) |
| Tile | (32,32,32) | (32,64,32) | (32,64,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 167.8 | 164.9 | 160.5 |
| Trial 2 avg (µs) | 175.9 | 162.0 | 153.1 |
| Trial 3 avg (µs) | 174.8 | 127.7 | 157.4 |
| **Mean avg (µs)** | **172.8** | **151.5** | **157.0** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: BASELINE 2**
Δ(B2 vs B0)=-13.9µs (-7.2%)   Δ(AG vs B0)=-0.9µs (-0.5%)   Δ(AG vs B1)=+13.0µs (+7.3%)

---

### 30×96×960  —  `o_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,960,128) | (32,960,128) | (32,960,128) |
| Tile | (32,32,32) | (16,64,64) | (16,64,64) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 237.0 | 289.5 | 266.5 |
| Trial 2 avg (µs) | 241.0 | 285.6 | 275.3 |
| Trial 3 avg (µs) | 237.0 | 267.1 | 280.4 |
| **Mean avg (µs)** | **238.3** | **280.8** | **274.1** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: BASELINE 0**
Δ(B2 vs B0)=+42.1µs (+15.8%)   Δ(AG vs B0)=+43.2µs (+16.2%)   Δ(AG vs B1)=+1.1µs (+0.3%)

---

### 30×192×32  —  `action_out_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,32,256) | (32,32,256) | (32,32,256) |
| Tile | (32,32,32) | (32,16,32) | (32,16,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 132.3 | 134.8 | 133.0 |
| Trial 2 avg (µs) | 133.3 | 134.8 | 133.8 |
| Trial 3 avg (µs) | 132.6 | 134.3 | 146.9 |
| **Mean avg (µs)** | **132.7** | **134.6** | **137.9** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=+2.5µs (+1.6%)   Δ(AG vs B0)=-0.2µs (-0.1%)   Δ(AG vs B1)=-2.6µs (-1.6%)

---

### 30×192×256  —  `gate_proj, up_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,256,256) | (32,256,256) | (32,256,256) |
| Tile | (32,32,32) | (16,64,32) | (16,64,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 165.4 | 178.3 | 177.6 |
| Trial 2 avg (µs) | 169.0 | 175.2 | 169.5 |
| Trial 3 avg (µs) | 184.9 | 182.2 | 179.4 |
| **Mean avg (µs)** | **173.1** | **178.6** | **175.5** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=+13.5µs (+6.9%)   Δ(AG vs B0)=+3.8µs (+2.0%)   Δ(AG vs B1)=-9.6µs (-4.6%)

---

### 30×192×960  —  `o_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,960,256) | (32,960,256) | (32,960,256) |
| Tile | (16,32,32) | (16,64,32) | (16,64,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | FAIL | 221.8 | 220.4 |
| Trial 2 avg (µs) | FAIL | 221.3 | 225.1 |
| Trial 3 avg (µs) | FAIL | 221.2 | 223.9 |
| **Mean avg (µs)** | **FAIL** | **221.4** | **223.1** |
| Pass count | 0/3 | 3/3 | 3/3 |

**Winner: INCONCLUSIVE**
Δ(AG vs B1)=-3.1µs (-1.2%)

---

### 30×256×96  —  `down_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,96,256) | (32,96,256) | (32,96,256) |
| Tile | (32,32,32) | (32,16,32) | (32,16,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 130.7 | 127.3 | 122.1 |
| Trial 2 avg (µs) | 132.5 | 125.0 | 124.3 |
| Trial 3 avg (µs) | 130.0 | 123.5 | 125.3 |
| **Mean avg (µs)** | **131.1** | **125.3** | **123.9** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-13.1µs (-7.9%)   Δ(AG vs B0)=-15.4µs (-9.3%)   Δ(AG vs B1)=-2.3µs (-1.5%)

---

### 30×256×192  —  `down_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,192,256) | (32,192,256) | (32,192,256) |
| Tile | (32,32,32) | (32,32,16) | (32,32,16) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 132.7 | 131.8 | 134.6 |
| Trial 2 avg (µs) | 134.3 | 135.8 | 135.7 |
| Trial 3 avg (µs) | 132.4 | 137.6 | 134.1 |
| **Mean avg (µs)** | **133.1** | **135.1** | **134.8** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=+2.0µs (+1.3%)   Δ(AG vs B0)=+1.3µs (+0.8%)   Δ(AG vs B1)=-0.7µs (-0.5%)

---

### 30×256×288  —  `down_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,288,256) | (32,288,256) | (32,288,256) |
| Tile | (32,32,32) | (16,32,64) | (16,32,64) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 153.6 | 156.8 | 153.6 |
| Trial 2 avg (µs) | 155.6 | 154.0 | 161.7 |
| Trial 3 avg (µs) | 156.2 | 153.4 | 154.6 |
| **Mean avg (µs)** | **155.2** | **154.7** | **156.6** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-0.3µs (-0.2%)   Δ(AG vs B0)=+6.6µs (+3.7%)   Δ(AG vs B1)=+6.9µs (+3.9%)

---

### 30×256×384  —  `down_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,384,256) | (32,384,256) | (32,384,256) |
| Tile | (32,32,32) | (32,64,32) | (32,64,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 150.3 | 146.9 | 153.6 |
| Trial 2 avg (µs) | 149.0 | 151.6 | 148.6 |
| Trial 3 avg (µs) | 150.6 | 147.3 | 154.3 |
| **Mean avg (µs)** | **149.9** | **148.6** | **152.2** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: BASELINE 2**
Δ(B2 vs B0)=-10.2µs (-5.5%)   Δ(AG vs B0)=+0.3µs (+0.2%)   Δ(AG vs B1)=+10.5µs (+6.0%)

---

### 30×256×480  —  `down_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,480,256) | (32,480,256) | (32,480,256) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 170.7 | 173.3 | 173.3 |
| Trial 2 avg (µs) | 173.4 | 176.1 | 172.1 |
| Trial 3 avg (µs) | 178.3 | 174.6 | 178.2 |
| **Mean avg (µs)** | **174.2** | **174.7** | **174.6** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=+0.5µs (+0.3%)   Δ(AG vs B0)=+1.4µs (+0.7%)   Δ(AG vs B1)=+0.8µs (+0.4%)

---

### 30×256×576  —  `down_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,576,256) | (32,576,256) | (32,576,256) |
| Tile | (32,32,32) | (16,64,16) | (16,64,16) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 195.1 | 187.6 | 190.6 |
| Trial 2 avg (µs) | 191.2 | 186.6 | 188.3 |
| Trial 3 avg (µs) | 190.0 | 185.8 | 188.3 |
| **Mean avg (µs)** | **192.1** | **186.7** | **189.1** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: BASELINE 2**
Δ(B2 vs B0)=-5.4µs (-2.5%)   Δ(AG vs B0)=+6.2µs (+2.8%)   Δ(AG vs B1)=+11.6µs (+5.4%)

---

### 30×256×672  —  `down_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,672,256) | (32,672,256) | (32,672,256) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 207.0 | 207.5 | 207.6 |
| Trial 2 avg (µs) | 207.7 | 206.2 | 213.4 |
| Trial 3 avg (µs) | 206.8 | 207.0 | 205.9 |
| **Mean avg (µs)** | **207.2** | **206.9** | **209.0** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=+0.1µs (+0.0%)   Δ(AG vs B0)=+2.1µs (+0.9%)   Δ(AG vs B1)=+2.0µs (+0.8%)

---

### 30×256×720  —  `down_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,768,256) | (32,768,256) | (32,768,256) |
| Tile | (32,32,32) | (32,64,32) | (32,64,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 198.7 | 163.1 | 160.3 |
| Trial 2 avg (µs) | 196.6 | 164.3 | 161.7 |
| Trial 3 avg (µs) | 199.2 | 161.2 | 159.8 |
| **Mean avg (µs)** | **198.2** | **162.9** | **160.6** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-35.2µs (-15.3%)   Δ(AG vs B0)=-37.2µs (-16.2%)   Δ(AG vs B1)=-2.0µs (-1.0%)

---

### 30×288×32  —  `action_out_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,32,512) | (32,32,512) | (32,32,512) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 130.4 | 128.2 | 128.1 |
| Trial 2 avg (µs) | 129.9 | 135.9 | 128.2 |
| Trial 3 avg (µs) | 133.3 | 131.7 | 132.5 |
| **Mean avg (µs)** | **131.2** | **132.0** | **129.6** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-3.4µs (-2.0%)   Δ(AG vs B0)=-1.3µs (-0.8%)   Δ(AG vs B1)=+2.0µs (+1.2%)

---

### 30×288×256  —  `gate_proj, up_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,256,512) | (32,256,512) | (32,256,512) |
| Tile | (32,32,32) | (16,64,32) | (16,64,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 155.1 | 149.1 | 147.6 |
| Trial 2 avg (µs) | 155.9 | 145.4 | 146.7 |
| Trial 3 avg (µs) | 154.4 | 143.9 | 150.9 |
| **Mean avg (µs)** | **155.2** | **146.1** | **148.4** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-15.2µs (-8.1%)   Δ(AG vs B0)=-14.6µs (-7.8%)   Δ(AG vs B1)=+0.7µs (+0.4%)

---

### 30×288×960  —  `o_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,960,512) | (32,960,512) | (32,960,512) |
| Tile | (16,32,32) | (16,16,16) | (16,16,16) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | FAIL | 387.9 | 349.4 |
| Trial 2 avg (µs) | FAIL | 503.0 | 380.9 |
| Trial 3 avg (µs) | 536.5 | 368.5 | 503.4 |
| **Mean avg (µs)** | **536.5** | **419.8** | **411.2** |
| Pass count | 1/3 | 3/3 | 3/3 |

**Winner: AGENTIC**
Δ(B2 vs B0)=-110.7µs (-15.4%)   Δ(AG vs B0)=-126.5µs (-17.6%)   Δ(AG vs B1)=-15.8µs (-2.6%)

---

### 30×384×32  —  `action_out_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,32,512) | (32,32,512) | (32,32,512) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 134.1 | 131.4 | 131.7 |
| Trial 2 avg (µs) | 131.4 | 132.2 | 132.0 |
| Trial 3 avg (µs) | 131.6 | 130.1 | 129.7 |
| **Mean avg (µs)** | **132.4** | **131.2** | **131.1** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: BASELINE 0**
Δ(B2 vs B0)=+3.2µs (+2.0%)   Δ(AG vs B0)=+4.1µs (+2.6%)   Δ(AG vs B1)=+0.9µs (+0.6%)

---

### 30×384×256  —  `gate_proj, up_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,256,512) | (32,256,512) | (32,256,512) |
| Tile | (32,32,32) | (16,64,32) | (16,64,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 156.7 | 148.0 | 144.6 |
| Trial 2 avg (µs) | 156.9 | 144.8 | 151.2 |
| Trial 3 avg (µs) | 156.7 | 146.2 | 144.7 |
| **Mean avg (µs)** | **156.8** | **146.3** | **146.8** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-12.0µs (-6.5%)   Δ(AG vs B0)=-11.4µs (-6.2%)   Δ(AG vs B1)=+0.6µs (+0.3%)

---

### 30×384×960  —  `o_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,960,512) | (32,960,512) | (32,960,512) |
| Tile | (16,32,32) | (16,16,16) | (16,16,16) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 356.3 | 308.9 | 419.4 |
| Trial 2 avg (µs) | 354.9 | 322.0 | 307.6 |
| Trial 3 avg (µs) | FAIL | 308.1 | 309.0 |
| **Mean avg (µs)** | **355.6** | **313.0** | **345.3** |
| Pass count | 2/3 | 3/3 | 3/3 |

**Winner: BASELINE 2**
Δ(B2 vs B0)=-54.0µs (-9.0%)   Δ(AG vs B0)=-15.8µs (-2.6%)   Δ(AG vs B1)=+38.2µs (+7.0%)

---

### 30×480×32  —  `action_out_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,32,512) | (32,32,512) | (32,32,512) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 131.9 | 133.9 | 131.2 |
| Trial 2 avg (µs) | 130.9 | 132.7 | 132.1 |
| Trial 3 avg (µs) | 128.7 | 132.9 | 129.3 |
| **Mean avg (µs)** | **130.5** | **133.2** | **130.8** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-2.2µs (-1.4%)   Δ(AG vs B0)=+4.7µs (+3.0%)   Δ(AG vs B1)=+7.0µs (+4.4%)

---

### 30×480×256  —  `gate_proj, up_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,256,512) | (32,256,512) | (32,256,512) |
| Tile | (32,32,32) | (16,64,32) | (16,64,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 154.8 | 145.8 | 144.9 |
| Trial 2 avg (µs) | 155.9 | 150.8 | 149.6 |
| Trial 3 avg (µs) | 153.7 | 147.7 | 149.3 |
| **Mean avg (µs)** | **154.8** | **148.1** | **147.9** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-6.5µs (-3.6%)   Δ(AG vs B0)=-6.9µs (-3.8%)   Δ(AG vs B1)=-0.3µs (-0.2%)

---

### 30×480×960  —  `o_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,960,512) | (32,960,512) | (32,960,512) |
| Tile | (16,32,32) | (16,16,16) | (16,16,16) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 350.8 | 310.6 | 310.7 |
| Trial 2 avg (µs) | 355.4 | 311.2 | 309.9 |
| Trial 3 avg (µs) | 350.7 | 309.2 | 311.3 |
| **Mean avg (µs)** | **352.3** | **310.3** | **310.6** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-38.2µs (-6.0%)   Δ(AG vs B0)=-33.7µs (-5.3%)   Δ(AG vs B1)=+4.5µs (+0.8%)

---

### 30×576×32  —  `action_out_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,32,768) | (32,32,768) | (32,32,768) |
| Tile | (32,32,32) | (16,16,256) | (16,16,256) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 135.9 | 132.7 | 130.2 |
| Trial 2 avg (µs) | 137.0 | 131.4 | 130.6 |
| Trial 3 avg (µs) | 138.6 | 130.5 | 131.2 |
| **Mean avg (µs)** | **137.2** | **131.5** | **130.7** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: AGENTIC**
Δ(B2 vs B0)=-8.4µs (-5.0%)   Δ(AG vs B0)=-13.9µs (-8.2%)   Δ(AG vs B1)=-5.4µs (-3.4%)

---

### 30×576×256  —  `gate_proj, up_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,256,768) | (32,256,768) | (32,256,768) |
| Tile | (32,32,32) | (32,64,32) | (32,64,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 165.3 | 156.9 | 157.3 |
| Trial 2 avg (µs) | 166.9 | 159.8 | 156.9 |
| Trial 3 avg (µs) | 167.1 | 156.1 | 158.6 |
| **Mean avg (µs)** | **166.4** | **157.6** | **157.6** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-10.9µs (-5.5%)   Δ(AG vs B0)=-10.7µs (-5.4%)   Δ(AG vs B1)=+0.2µs (+0.1%)

---

### 30×576×960  —  `o_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,960,768) | (32,960,768) | (32,960,768) |
| Tile | (16,32,32) | (16,16,128) | (16,16,128) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | FAIL | 390.1 | 303.8 |
| Trial 2 avg (µs) | 695.2 | 308.8 | 306.8 |
| Trial 3 avg (µs) | FAIL | 304.9 | 490.8 |
| **Mean avg (µs)** | **695.2** | **334.6** | **367.1** |
| Pass count | 1/3 | 3/3 | 3/3 |

**Winner: BASELINE 2**
Δ(B2 vs B0)=-352.0µs (-26.9%)   Δ(AG vs B0)=-310.5µs (-23.8%)   Δ(AG vs B1)=+41.5µs (+4.3%)

---

### 30×672×32  —  `action_out_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,32,768) | (32,32,768) | (32,32,768) |
| Tile | (32,32,32) | (16,16,256) | (16,16,256) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 137.7 | 127.8 | 129.2 |
| Trial 2 avg (µs) | 133.4 | 121.3 | 133.5 |
| Trial 3 avg (µs) | 137.6 | 125.3 | 129.3 |
| **Mean avg (µs)** | **136.2** | **124.8** | **130.7** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: BASELINE 2**
Δ(B2 vs B0)=-14.0µs (-8.6%)   Δ(AG vs B0)=-5.9µs (-3.6%)   Δ(AG vs B1)=+8.2µs (+5.5%)

---

### 30×672×256  —  `gate_proj, up_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,256,768) | (32,256,768) | (32,256,768) |
| Tile | (32,32,32) | (32,64,32) | (32,64,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 163.1 | 159.9 | 158.9 |
| Trial 2 avg (µs) | 164.0 | 160.7 | 157.9 |
| Trial 3 avg (µs) | 159.1 | 154.9 | 156.8 |
| **Mean avg (µs)** | **162.1** | **158.5** | **157.9** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-0.8µs (-0.4%)   Δ(AG vs B0)=-4.1µs (-2.1%)   Δ(AG vs B1)=-3.3µs (-1.7%)

---

### 30×672×960  —  `o_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,960,768) | (32,960,768) | (32,960,768) |
| Tile | (16,32,32) | (16,16,128) | (16,16,128) |
| Tile source | fixed | profiling | profiling |
| Pad impl | numpy_pad | numpy_pad | numpy_pad |
| Trial 1 avg (µs) | FAIL | 306.0 | 309.0 |
| Trial 2 avg (µs) | FAIL | 308.6 | 308.1 |
| Trial 3 avg (µs) | FAIL | 304.3 | 305.2 |
| **Mean avg (µs)** | **FAIL** | **306.3** | **307.4** |
| Pass count | 0/3 | 3/3 | 3/3 |

**Winner: INCONCLUSIVE**
Δ(AG vs B1)=+25.2µs (+2.1%)

---

### 30×720×32  —  `action_out_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,32,768) | (32,32,768) | (32,32,768) |
| Tile | (32,32,32) | (16,16,256) | (16,16,256) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 135.0 | 132.3 | 132.2 |
| Trial 2 avg (µs) | 138.2 | 132.4 | 131.7 |
| Trial 3 avg (µs) | 134.8 | 130.6 | 130.0 |
| **Mean avg (µs)** | **136.0** | **131.8** | **131.3** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: AGENTIC**
Δ(B2 vs B0)=-3.8µs (-2.3%)   Δ(AG vs B0)=-10.2µs (-6.3%)   Δ(AG vs B1)=-6.5µs (-4.1%)

---

### 30×720×256  —  `gate_proj, up_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,256,768) | (32,256,768) | (32,256,768) |
| Tile | (32,32,32) | (32,64,32) | (32,64,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 162.8 | 158.8 | 158.9 |
| Trial 2 avg (µs) | 167.2 | 156.5 | 159.9 |
| Trial 3 avg (µs) | 164.8 | 160.1 | 156.2 |
| **Mean avg (µs)** | **164.9** | **158.5** | **158.3** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-11.5µs (-5.7%)   Δ(AG vs B0)=-12.2µs (-6.0%)   Δ(AG vs B1)=-0.7µs (-0.4%)

---

### 30×720×960  —  `o_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,960,768) | (32,960,768) | (32,960,768) |
| Tile | (16,32,32) | (16,16,128) | (16,16,128) |
| Tile source | fixed | profiling | profiling |
| Pad impl | numpy_pad | numpy_pad | numpy_pad |
| Trial 1 avg (µs) | 466.7 | 311.1 | 311.6 |
| Trial 2 avg (µs) | FAIL | 310.1 | 307.4 |
| Trial 3 avg (µs) | FAIL | 308.4 | 311.2 |
| **Mean avg (µs)** | **466.7** | **309.9** | **310.1** |
| Pass count | 1/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-159.7µs (-14.9%)   Δ(AG vs B0)=-163.1µs (-15.2%)   Δ(AG vs B1)=-3.4µs (-0.4%)

---

### 30×960×96  —  `q_proj`

**Pre-screened skip**: bf16_accuracy_small_MN_large_K

---

### 30×960×192  —  `q_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,192,960) | (32,192,960) | (32,192,960) |
| Tile | (16,32,32) | (32,32,64) | (32,32,64) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 174.1 | 170.7 | 172.5 |
| Trial 2 avg (µs) | 172.8 | 168.7 | 169.5 |
| Trial 3 avg (µs) | 173.2 | 171.6 | 173.9 |
| **Mean avg (µs)** | **FAIL** | **170.3** | **171.9** |
| Pass count | 0/3 | 3/3 | 3/3 |

**Winner: INCONCLUSIVE**
Δ(AG vs B1)=+2.3µs (+1.1%)

---

### 30×960×288  —  `q_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,288,960) | (32,288,960) | (32,288,960) |
| Tile | (16,32,32) | (16,32,64) | (16,32,64) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 254.0 | 243.5 | 243.2 |
| Trial 2 avg (µs) | 241.4 | 248.9 | 245.8 |
| Trial 3 avg (µs) | 240.2 | 243.5 | 252.1 |
| **Mean avg (µs)** | **FAIL** | **245.3** | **247.0** |
| Pass count | 0/3 | 3/3 | 3/3 |

**Winner: INCONCLUSIVE**
Δ(AG vs B1)=+6.2µs (+1.4%)

---

### 30×960×384  —  `q_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,384,960) | (32,384,960) | (32,384,960) |
| Tile | (16,32,32) | (16,32,64) | (16,32,64) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 197.6 | 191.3 | 193.3 |
| Trial 2 avg (µs) | 198.3 | 190.3 | 191.2 |
| Trial 3 avg (µs) | 199.7 | 195.3 | 193.0 |
| **Mean avg (µs)** | **FAIL** | **192.3** | **192.5** |
| Pass count | 0/3 | 3/3 | 3/3 |

**Winner: INCONCLUSIVE**
Δ(AG vs B1)=+2.3µs (+0.5%)

---

### 30×960×480  —  `q_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,480,960) | (32,480,960) | (32,480,960) |
| Tile | (16,32,32) | (32,32,64) | (32,32,64) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 303.5 | 306.5 | 305.4 |
| Trial 2 avg (µs) | 333.2 | 302.4 | 307.0 |
| Trial 3 avg (µs) | 344.8 | 308.6 | 307.5 |
| **Mean avg (µs)** | **FAIL** | **305.8** | **306.6** |
| Pass count | 0/3 | 3/3 | 3/3 |

**Winner: INCONCLUSIVE**
Δ(AG vs B1)=+7.6µs (+1.3%)

---

### 30×960×576  —  `q_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,576,960) | (32,576,960) | (32,576,960) |
| Tile | (16,32,32) | (32,16,64) | (32,16,64) |
| Tile source | fixed | profiling | profiling |
| Pad impl | numpy_pad | numpy_pad | numpy_pad |
| Trial 1 avg (µs) | 367.7 | 275.9 | 358.5 |
| Trial 2 avg (µs) | 371.9 | 278.7 | 280.3 |
| Trial 3 avg (µs) | 406.6 | 279.7 | 280.1 |
| **Mean avg (µs)** | **FAIL** | **278.1** | **306.3** |
| Pass count | 0/3 | 3/3 | 3/3 |

**Winner: INCONCLUSIVE**
Δ(AG vs B1)=+45.3µs (+4.4%)

---

### 30×960×672  —  `q_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,672,960) | (32,672,960) | (32,672,960) |
| Tile | (16,32,32) | (32,32,64) | (32,32,64) |
| Tile source | fixed | profiling | profiling |
| Pad impl | numpy_pad | numpy_pad | numpy_pad |
| Trial 1 avg (µs) | 484.8 | 457.4 | 423.0 |
| Trial 2 avg (µs) | 490.5 | 429.0 | 425.0 |
| Trial 3 avg (µs) | 428.4 | 429.5 | 426.2 |
| **Mean avg (µs)** | **FAIL** | **429.3** | **424.8** |
| Pass count | 0/3 | 2/3 | 3/3 |

**Winner: INCONCLUSIVE**
Δ(AG vs B1)=+15.0µs (+1.2%)

---

### 30×960×720  —  `q_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,768,960) | (32,768,960) | (32,768,960) |
| Tile | (16,32,32) | (16,32,32) | (16,32,32) |
| Tile source | fixed | fallback_fixed | heuristic |
| Pad impl | numpy_pad | numpy_pad | numpy_pad |
| Trial 1 avg (µs) | 262.3 | 263.3 | 259.7 |
| Trial 2 avg (µs) | 265.4 | 264.1 | 261.5 |
| Trial 3 avg (µs) | 263.5 | 259.6 | 427.1 |
| **Mean avg (µs)** | **FAIL** | **FAIL** | **FAIL** |
| Pass count | 0/3 | 0/3 | 0/3 |

**Winner: INCONCLUSIVE**

---

### 32×32×96  —  `action_in_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,96,32) | (32,96,32) | (32,96,32) |
| Tile | (32,32,32) | (32,16,16) | (32,16,16) |
| Tile source | fixed | profiling | profiling |
| Pad impl | none | none | none |
| Trial 1 avg (µs) | 127.8 | 125.1 | 125.6 |
| Trial 2 avg (µs) | 128.3 | 124.9 | 121.9 |
| Trial 3 avg (µs) | 130.2 | 124.4 | 125.8 |
| **Mean avg (µs)** | **128.8** | **124.8** | **124.4** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: BASELINE 0**
Δ(B2 vs B0)=+nanµs (+nan%)   Δ(AG vs B0)=+nanµs (+nan%)   Δ(AG vs B1)=+nanµs (+nan%)

---

### 32×32×192  —  `action_in_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,192,32) | (32,192,32) | (32,192,32) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | none | none | none |
| Trial 1 avg (µs) | 124.9 | 124.4 | 125.7 |
| Trial 2 avg (µs) | 127.1 | 127.2 | 124.4 |
| Trial 3 avg (µs) | 129.9 | 127.6 | 124.8 |
| **Mean avg (µs)** | **127.3** | **126.4** | **125.0** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: BASELINE 0**
Δ(B2 vs B0)=+nanµs (+nan%)   Δ(AG vs B0)=+nanµs (+nan%)   Δ(AG vs B1)=+nanµs (+nan%)

---

### 32×32×288  —  `action_in_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,288,32) | (32,288,32) | (32,288,32) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | none | none | none |
| Trial 1 avg (µs) | 130.5 | 132.7 | 133.4 |
| Trial 2 avg (µs) | 133.2 | 134.0 | 130.2 |
| Trial 3 avg (µs) | 132.4 | 131.9 | 133.8 |
| **Mean avg (µs)** | **132.0** | **132.9** | **132.5** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: BASELINE 0**
Δ(B2 vs B0)=+nanµs (+nan%)   Δ(AG vs B0)=+nanµs (+nan%)   Δ(AG vs B1)=+nanµs (+nan%)

---

### 32×32×384  —  `action_in_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,384,32) | (32,384,32) | (32,384,32) |
| Tile | (32,32,32) | (16,64,16) | (16,64,16) |
| Tile source | fixed | profiling | profiling |
| Pad impl | none | none | none |
| Trial 1 avg (µs) | 157.2 | 129.8 | 132.1 |
| Trial 2 avg (µs) | 158.1 | 129.5 | 130.4 |
| Trial 3 avg (µs) | 156.7 | 128.0 | 133.9 |
| **Mean avg (µs)** | **157.3** | **129.1** | **132.1** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: BASELINE 0**
Δ(B2 vs B0)=+nanµs (+nan%)   Δ(AG vs B0)=+nanµs (+nan%)   Δ(AG vs B1)=+nanµs (+nan%)

---

### 32×32×480  —  `action_in_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,480,32) | (32,480,32) | (32,480,32) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | none | none | none |
| Trial 1 avg (µs) | 134.8 | 134.7 | 135.6 |
| Trial 2 avg (µs) | 137.0 | 137.5 | 137.0 |
| Trial 3 avg (µs) | 135.1 | 137.2 | 131.6 |
| **Mean avg (µs)** | **135.6** | **136.4** | **134.7** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: BASELINE 0**
Δ(B2 vs B0)=+nanµs (+nan%)   Δ(AG vs B0)=+nanµs (+nan%)   Δ(AG vs B1)=+nanµs (+nan%)

---

### 32×32×576  —  `action_in_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,576,32) | (32,576,32) | (32,576,32) |
| Tile | (32,32,32) | (32,64,32) | (32,64,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | none | none | none |
| Trial 1 avg (µs) | 139.7 | 139.8 | 133.9 |
| Trial 2 avg (µs) | 139.5 | 141.2 | 140.2 |
| Trial 3 avg (µs) | 138.9 | 135.3 | 143.7 |
| **Mean avg (µs)** | **139.4** | **138.8** | **139.3** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: BASELINE 0**
Δ(B2 vs B0)=+nanµs (+nan%)   Δ(AG vs B0)=+nanµs (+nan%)   Δ(AG vs B1)=+nanµs (+nan%)

---

### 32×32×672  —  `action_in_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,672,32) | (32,672,32) | (32,672,32) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | none | none | none |
| Trial 1 avg (µs) | 148.2 | 150.1 | 149.4 |
| Trial 2 avg (µs) | 141.8 | 148.9 | 149.4 |
| Trial 3 avg (µs) | 147.2 | 149.8 | 148.6 |
| **Mean avg (µs)** | **145.7** | **149.6** | **149.1** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: BASELINE 0**
Δ(B2 vs B0)=+nanµs (+nan%)   Δ(AG vs B0)=+nanµs (+nan%)   Δ(AG vs B1)=+nanµs (+nan%)

---

### 32×32×720  —  `action_in_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,768,32) | (32,768,32) | (32,768,32) |
| Tile | (32,32,32) | (32,64,32) | (32,64,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 194.5 | 156.2 | 155.7 |
| Trial 2 avg (µs) | 195.7 | 156.2 | 154.4 |
| Trial 3 avg (µs) | 194.8 | 157.8 | 154.5 |
| **Mean avg (µs)** | **195.0** | **156.8** | **154.9** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-42.2µs (-18.9%)   Δ(AG vs B0)=-38.8µs (-17.4%)   Δ(AG vs B1)=+3.4µs (+1.9%)

---

### 32×96×32  —  `action_out_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,32,128) | (32,32,128) | (32,32,128) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 124.4 | 125.1 | 119.7 |
| Trial 2 avg (µs) | 121.5 | 123.3 | 120.2 |
| Trial 3 avg (µs) | 124.3 | 119.7 | 123.7 |
| **Mean avg (µs)** | **123.4** | **122.7** | **121.2** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-0.4µs (-0.3%)   Δ(AG vs B0)=+1.6µs (+1.1%)   Δ(AG vs B1)=+2.0µs (+1.4%)

---

### 32×96×256  —  `gate_proj, up_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,256,128) | (32,256,128) | (32,256,128) |
| Tile | (32,32,32) | (32,64,32) | (32,64,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 142.9 | 134.3 | 129.7 |
| Trial 2 avg (µs) | 141.2 | 128.9 | 136.0 |
| Trial 3 avg (µs) | 142.0 | 130.7 | 132.1 |
| **Mean avg (µs)** | **142.0** | **131.3** | **132.6** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: BASELINE 2**
Δ(B2 vs B0)=-15.8µs (-9.2%)   Δ(AG vs B0)=-9.9µs (-5.7%)   Δ(AG vs B1)=+5.9µs (+3.8%)

---

### 32×96×960  —  `o_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,960,128) | (32,960,128) | (32,960,128) |
| Tile | (32,32,32) | (16,64,64) | (16,64,64) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 187.5 | 180.6 | 175.7 |
| Trial 2 avg (µs) | 187.1 | 178.1 | 178.7 |
| Trial 3 avg (µs) | 183.7 | 178.3 | 182.8 |
| **Mean avg (µs)** | **186.1** | **179.0** | **179.1** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: BASELINE 2**
Δ(B2 vs B0)=-13.5µs (-6.2%)   Δ(AG vs B0)=-7.4µs (-3.4%)   Δ(AG vs B1)=+6.1µs (+3.0%)

---

### 32×192×32  —  `action_out_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,32,256) | (32,32,256) | (32,32,256) |
| Tile | (32,32,32) | (32,16,32) | (32,16,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 128.6 | 126.6 | 123.6 |
| Trial 2 avg (µs) | 128.1 | 125.3 | 123.8 |
| Trial 3 avg (µs) | 128.7 | 125.8 | 126.7 |
| **Mean avg (µs)** | **128.5** | **125.9** | **124.7** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-12.0µs (-7.4%)   Δ(AG vs B0)=-13.1µs (-8.1%)   Δ(AG vs B1)=-1.1µs (-0.7%)

---

### 32×192×256  —  `gate_proj, up_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,256,256) | (32,256,256) | (32,256,256) |
| Tile | (32,32,32) | (16,64,32) | (16,64,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 143.8 | 133.2 | 137.1 |
| Trial 2 avg (µs) | 149.6 | 135.7 | 133.4 |
| Trial 3 avg (µs) | 143.1 | 135.8 | 132.4 |
| **Mean avg (µs)** | **145.5** | **134.9** | **134.3** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: AGENTIC**
Δ(B2 vs B0)=+0.6µs (+0.4%)   Δ(AG vs B0)=-5.5µs (-3.3%)   Δ(AG vs B1)=-6.1µs (-3.6%)

---

### 32×192×960  —  `o_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,960,256) | (32,960,256) | (32,960,256) |
| Tile | (16,32,32) | (16,64,32) | (16,64,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | FAIL | 224.3 | 225.8 |
| Trial 2 avg (µs) | FAIL | 223.4 | 226.8 |
| Trial 3 avg (µs) | 241.8 | 219.8 | 222.5 |
| **Mean avg (µs)** | **241.8** | **222.5** | **225.1** |
| Pass count | 1/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-20.2µs (-7.4%)   Δ(AG vs B0)=-17.6µs (-6.4%)   Δ(AG vs B1)=+2.6µs (+1.0%)

---

### 32×256×96  —  `down_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,96,256) | (32,96,256) | (32,96,256) |
| Tile | (32,32,32) | (32,16,32) | (32,16,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | none | none | none |
| Trial 1 avg (µs) | 135.1 | 126.0 | 125.4 |
| Trial 2 avg (µs) | 131.5 | 126.7 | 127.4 |
| Trial 3 avg (µs) | 134.3 | 124.9 | 122.2 |
| **Mean avg (µs)** | **133.6** | **125.8** | **125.0** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: BASELINE 0**
Δ(B2 vs B0)=+nanµs (+nan%)   Δ(AG vs B0)=+nanµs (+nan%)   Δ(AG vs B1)=+nanµs (+nan%)

---

### 32×256×192  —  `down_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,192,256) | (32,192,256) | (32,192,256) |
| Tile | (32,32,32) | (32,32,16) | (32,32,16) |
| Tile source | fixed | profiling | profiling |
| Pad impl | none | none | none |
| Trial 1 avg (µs) | 132.0 | 131.5 | 133.2 |
| Trial 2 avg (µs) | 132.3 | 133.6 | 135.2 |
| Trial 3 avg (µs) | 133.9 | 134.5 | 132.8 |
| **Mean avg (µs)** | **132.7** | **133.2** | **133.7** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: BASELINE 0**
Δ(B2 vs B0)=+nanµs (+nan%)   Δ(AG vs B0)=+nanµs (+nan%)   Δ(AG vs B1)=+nanµs (+nan%)

---

### 32×256×288  —  `down_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,288,256) | (32,288,256) | (32,288,256) |
| Tile | (32,32,32) | (16,32,64) | (16,32,64) |
| Tile source | fixed | profiling | profiling |
| Pad impl | none | none | none |
| Trial 1 avg (µs) | 161.4 | 153.9 | 153.0 |
| Trial 2 avg (µs) | 154.1 | 153.6 | 157.0 |
| Trial 3 avg (µs) | 161.4 | 153.1 | 154.2 |
| **Mean avg (µs)** | **159.0** | **153.5** | **154.7** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: BASELINE 0**
Δ(B2 vs B0)=+nanµs (+nan%)   Δ(AG vs B0)=+nanµs (+nan%)   Δ(AG vs B1)=+nanµs (+nan%)

---

### 32×256×384  —  `down_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,384,256) | (32,384,256) | (32,384,256) |
| Tile | (32,32,32) | (32,64,32) | (32,64,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | none | none | none |
| Trial 1 avg (µs) | 154.6 | 152.3 | 151.3 |
| Trial 2 avg (µs) | 156.3 | 152.7 | 149.8 |
| Trial 3 avg (µs) | 155.3 | 151.8 | 150.8 |
| **Mean avg (µs)** | **155.4** | **152.3** | **150.7** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: BASELINE 0**
Δ(B2 vs B0)=+nanµs (+nan%)   Δ(AG vs B0)=+nanµs (+nan%)   Δ(AG vs B1)=+nanµs (+nan%)

---

### 32×256×480  —  `down_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,480,256) | (32,480,256) | (32,480,256) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | none | none | none |
| Trial 1 avg (µs) | 176.9 | 174.9 | 174.1 |
| Trial 2 avg (µs) | 173.4 | 173.9 | 175.4 |
| Trial 3 avg (µs) | 176.7 | 172.7 | 174.9 |
| **Mean avg (µs)** | **175.7** | **173.8** | **174.8** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: BASELINE 0**
Δ(B2 vs B0)=+nanµs (+nan%)   Δ(AG vs B0)=+nanµs (+nan%)   Δ(AG vs B1)=+nanµs (+nan%)

---

### 32×256×576  —  `down_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,576,256) | (32,576,256) | (32,576,256) |
| Tile | (32,32,32) | (16,64,16) | (16,64,16) |
| Tile source | fixed | profiling | profiling |
| Pad impl | none | none | none |
| Trial 1 avg (µs) | 195.4 | 186.9 | 191.7 |
| Trial 2 avg (µs) | 192.3 | 189.4 | 189.2 |
| Trial 3 avg (µs) | 189.8 | 186.9 | 186.1 |
| **Mean avg (µs)** | **192.5** | **187.8** | **189.0** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: BASELINE 0**
Δ(B2 vs B0)=+nanµs (+nan%)   Δ(AG vs B0)=+nanµs (+nan%)   Δ(AG vs B1)=+nanµs (+nan%)

---

### 32×256×672  —  `down_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,672,256) | (32,672,256) | (32,672,256) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | none | none | none |
| Trial 1 avg (µs) | 208.6 | 205.3 | 213.9 |
| Trial 2 avg (µs) | 214.3 | 213.4 | 204.0 |
| Trial 3 avg (µs) | 209.8 | 211.9 | 203.7 |
| **Mean avg (µs)** | **210.9** | **210.2** | **207.2** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: BASELINE 0**
Δ(B2 vs B0)=+nanµs (+nan%)   Δ(AG vs B0)=+nanµs (+nan%)   Δ(AG vs B1)=+nanµs (+nan%)

---

### 32×256×720  —  `down_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,768,256) | (32,768,256) | (32,768,256) |
| Tile | (32,32,32) | (32,64,32) | (32,64,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 196.5 | 161.8 | 163.8 |
| Trial 2 avg (µs) | 200.7 | 167.5 | 162.4 |
| Trial 3 avg (µs) | 195.1 | 163.5 | 162.7 |
| **Mean avg (µs)** | **197.4** | **164.3** | **163.0** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-33.4µs (-14.5%)   Δ(AG vs B0)=-34.9µs (-15.2%)   Δ(AG vs B1)=-1.6µs (-0.8%)

---

### 32×288×32  —  `action_out_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,32,512) | (32,32,512) | (32,32,512) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 131.2 | 132.2 | 134.0 |
| Trial 2 avg (µs) | 132.8 | 134.0 | 133.0 |
| Trial 3 avg (µs) | 130.7 | 132.2 | 129.4 |
| **Mean avg (µs)** | **131.6** | **132.8** | **132.2** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: BASELINE 2**
Δ(B2 vs B0)=-12.5µs (-7.5%)   Δ(AG vs B0)=-4.8µs (-2.9%)   Δ(AG vs B1)=+7.7µs (+5.0%)

---

### 32×288×256  —  `gate_proj, up_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,256,512) | (32,256,512) | (32,256,512) |
| Tile | (32,32,32) | (16,64,32) | (16,64,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 156.3 | 143.7 | 148.3 |
| Trial 2 avg (µs) | 156.1 | 144.4 | 144.2 |
| Trial 3 avg (µs) | 151.1 | 141.9 | 150.3 |
| **Mean avg (µs)** | **154.5** | **143.4** | **147.6** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-3.8µs (-2.1%)   Δ(AG vs B0)=-6.0µs (-3.4%)   Δ(AG vs B1)=-2.2µs (-1.3%)

---

### 32×288×960  —  `o_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,960,512) | (32,960,512) | (32,960,512) |
| Tile | (16,32,32) | (16,16,16) | (16,16,16) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | FAIL | 307.1 | 307.5 |
| Trial 2 avg (µs) | 352.3 | 304.4 | 310.8 |
| Trial 3 avg (µs) | 354.0 | 306.9 | 308.9 |
| **Mean avg (µs)** | **353.2** | **306.1** | **309.1** |
| Pass count | 2/3 | 3/3 | 3/3 |

**Winner: AGENTIC**
Δ(B2 vs B0)=-24.0µs (-4.5%)   Δ(AG vs B0)=-38.9µs (-7.4%)   Δ(AG vs B1)=-14.9µs (-3.0%)

---

### 32×384×32  —  `action_out_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,32,512) | (32,32,512) | (32,32,512) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 132.3 | 132.6 | 134.1 |
| Trial 2 avg (µs) | 133.6 | 132.6 | 130.3 |
| Trial 3 avg (µs) | 132.8 | 132.6 | 130.6 |
| **Mean avg (µs)** | **132.9** | **132.6** | **131.7** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: AGENTIC**
Δ(B2 vs B0)=-0.7µs (-0.4%)   Δ(AG vs B0)=-6.2µs (-3.8%)   Δ(AG vs B1)=-5.4µs (-3.4%)

---

### 32×384×256  —  `gate_proj, up_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,256,512) | (32,256,512) | (32,256,512) |
| Tile | (32,32,32) | (16,64,32) | (16,64,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 156.2 | 149.2 | 149.6 |
| Trial 2 avg (µs) | 156.7 | 148.0 | 145.1 |
| Trial 3 avg (µs) | 155.8 | 152.4 | 145.1 |
| **Mean avg (µs)** | **156.2** | **149.9** | **146.6** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-6.5µs (-3.5%)   Δ(AG vs B0)=-9.5µs (-5.2%)   Δ(AG vs B1)=-3.0µs (-1.7%)

---

### 32×384×960  —  `o_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,960,512) | (32,960,512) | (32,960,512) |
| Tile | (16,32,32) | (16,16,16) | (16,16,16) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | FAIL | 309.3 | 307.2 |
| Trial 2 avg (µs) | FAIL | 326.2 | 313.0 |
| Trial 3 avg (µs) | FAIL | 306.7 | 308.8 |
| **Mean avg (µs)** | **FAIL** | **314.1** | **309.7** |
| Pass count | 0/3 | 3/3 | 3/3 |

**Winner: INCONCLUSIVE**
Δ(AG vs B1)=-6.8µs (-1.2%)

---

### 32×480×32  —  `action_out_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,32,512) | (32,32,512) | (32,32,512) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 132.2 | 129.9 | 133.0 |
| Trial 2 avg (µs) | 130.8 | 132.9 | 135.6 |
| Trial 3 avg (µs) | 132.5 | 133.6 | 132.6 |
| **Mean avg (µs)** | **131.8** | **132.2** | **133.7** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: BASELINE 2**
Δ(B2 vs B0)=-9.2µs (-5.7%)   Δ(AG vs B0)=-3.3µs (-2.0%)   Δ(AG vs B1)=+6.0µs (+3.9%)

---

### 32×480×256  —  `gate_proj, up_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,256,512) | (32,256,512) | (32,256,512) |
| Tile | (32,32,32) | (16,64,32) | (16,64,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 154.3 | 149.6 | 148.5 |
| Trial 2 avg (µs) | 155.8 | 149.8 | 146.9 |
| Trial 3 avg (µs) | 157.9 | 147.2 | 147.3 |
| **Mean avg (µs)** | **156.0** | **148.8** | **147.6** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-6.8µs (-3.7%)   Δ(AG vs B0)=-8.7µs (-4.7%)   Δ(AG vs B1)=-1.9µs (-1.1%)

---

### 32×480×960  —  `o_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,960,512) | (32,960,512) | (32,960,512) |
| Tile | (16,32,32) | (16,16,16) | (16,16,16) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 355.3 | 309.5 | 306.3 |
| Trial 2 avg (µs) | 352.9 | 308.8 | 460.8 |
| Trial 3 avg (µs) | FAIL | 310.2 | 304.4 |
| **Mean avg (µs)** | **354.1** | **309.5** | **357.2** |
| Pass count | 2/3 | 3/3 | 3/3 |

**Winner: BASELINE 2**
Δ(B2 vs B0)=-58.2µs (-8.9%)   Δ(AG vs B0)=+39.8µs (+6.1%)   Δ(AG vs B1)=+98.0µs (+16.6%)

---

### 32×576×32  —  `action_out_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,32,768) | (32,32,768) | (32,32,768) |
| Tile | (32,32,32) | (16,16,256) | (16,16,256) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 134.5 | 132.5 | 130.8 |
| Trial 2 avg (µs) | 135.7 | 131.9 | 131.4 |
| Trial 3 avg (µs) | 141.0 | 131.6 | 130.2 |
| **Mean avg (µs)** | **137.1** | **132.0** | **130.8** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: AGENTIC**
Δ(B2 vs B0)=-4.7µs (-2.9%)   Δ(AG vs B0)=-11.8µs (-7.2%)   Δ(AG vs B1)=-7.2µs (-4.5%)

---

### 32×576×256  —  `gate_proj, up_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,256,768) | (32,256,768) | (32,256,768) |
| Tile | (32,32,32) | (32,64,32) | (32,64,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 167.6 | 159.0 | 159.8 |
| Trial 2 avg (µs) | 167.3 | 161.4 | 161.0 |
| Trial 3 avg (µs) | 168.0 | 158.3 | 158.8 |
| **Mean avg (µs)** | **167.7** | **159.6** | **159.9** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-9.0µs (-4.5%)   Δ(AG vs B0)=-9.8µs (-4.9%)   Δ(AG vs B1)=-0.8µs (-0.4%)

---

### 32×576×960  —  `o_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,960,768) | (32,960,768) | (32,960,768) |
| Tile | (16,32,32) | (16,16,128) | (16,16,128) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 467.6 | 373.3 | 303.9 |
| Trial 2 avg (µs) | FAIL | 306.3 | 308.3 |
| Trial 3 avg (µs) | FAIL | 361.0 | 308.4 |
| **Mean avg (µs)** | **467.6** | **346.9** | **306.9** |
| Pass count | 1/3 | 3/3 | 3/3 |

**Winner: AGENTIC**
Δ(B2 vs B0)=-177.5µs (-15.3%)   Δ(AG vs B0)=-230.3µs (-19.9%)   Δ(AG vs B1)=-52.8µs (-5.4%)

---

### 32×672×32  —  `action_out_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,32,768) | (32,32,768) | (32,32,768) |
| Tile | (32,32,32) | (16,16,256) | (16,16,256) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 138.9 | 131.5 | 133.2 |
| Trial 2 avg (µs) | 135.7 | 130.4 | 129.3 |
| Trial 3 avg (µs) | 137.5 | 135.5 | 132.3 |
| **Mean avg (µs)** | **137.4** | **132.5** | **131.6** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: AGENTIC**
Δ(B2 vs B0)=-5.0µs (-3.0%)   Δ(AG vs B0)=-16.4µs (-9.8%)   Δ(AG vs B1)=-11.4µs (-7.0%)

---

### 32×672×256  —  `gate_proj, up_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,256,768) | (32,256,768) | (32,256,768) |
| Tile | (32,32,32) | (32,64,32) | (32,64,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 166.2 | 162.0 | 160.9 |
| Trial 2 avg (µs) | 166.3 | 162.9 | 158.2 |
| Trial 3 avg (µs) | 165.1 | 159.7 | 159.8 |
| **Mean avg (µs)** | **165.8** | **161.5** | **159.6** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-5.0µs (-2.5%)   Δ(AG vs B0)=-7.8µs (-4.0%)   Δ(AG vs B1)=-2.8µs (-1.5%)

---

### 32×672×960  —  `o_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,960,768) | (32,960,768) | (32,960,768) |
| Tile | (16,32,32) | (16,16,128) | (16,16,128) |
| Tile source | fixed | profiling | profiling |
| Pad impl | numpy_pad | numpy_pad | numpy_pad |
| Trial 1 avg (µs) | 490.5 | 308.1 | 307.3 |
| Trial 2 avg (µs) | 483.8 | 308.7 | 308.1 |
| Trial 3 avg (µs) | FAIL | 302.1 | 306.3 |
| **Mean avg (µs)** | **487.1** | **306.3** | **307.2** |
| Pass count | 2/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-177.6µs (-12.8%)   Δ(AG vs B0)=-180.4µs (-13.0%)   Δ(AG vs B1)=-2.7µs (-0.2%)

---

### 32×720×32  —  `action_out_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,32,768) | (32,32,768) | (32,32,768) |
| Tile | (32,32,32) | (16,16,256) | (16,16,256) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 142.0 | 130.7 | 130.8 |
| Trial 2 avg (µs) | 137.2 | 130.9 | 132.3 |
| Trial 3 avg (µs) | 138.4 | 129.3 | 133.1 |
| **Mean avg (µs)** | **139.2** | **130.3** | **132.1** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-4.0µs (-2.5%)   Δ(AG vs B0)=-2.6µs (-1.6%)   Δ(AG vs B1)=+1.4µs (+0.9%)

---

### 32×720×256  —  `gate_proj, up_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,256,768) | (32,256,768) | (32,256,768) |
| Tile | (32,32,32) | (32,64,32) | (32,64,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 164.0 | 158.4 | 154.4 |
| Trial 2 avg (µs) | 167.9 | 162.1 | 161.2 |
| Trial 3 avg (µs) | 168.3 | 157.6 | 159.2 |
| **Mean avg (µs)** | **166.7** | **159.4** | **158.2** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-9.1µs (-4.5%)   Δ(AG vs B0)=-11.8µs (-5.9%)   Δ(AG vs B1)=-2.7µs (-1.4%)

---

### 32×720×960  —  `o_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,960,768) | (32,960,768) | (32,960,768) |
| Tile | (16,32,32) | (16,16,128) | (16,16,128) |
| Tile source | fixed | profiling | profiling |
| Pad impl | numpy_pad | numpy_pad | numpy_pad |
| Trial 1 avg (µs) | FAIL | 305.9 | 306.8 |
| Trial 2 avg (µs) | 466.5 | 373.6 | 307.9 |
| Trial 3 avg (µs) | FAIL | 308.1 | 304.0 |
| **Mean avg (µs)** | **466.5** | **329.2** | **306.2** |
| Pass count | 1/3 | 3/3 | 3/3 |

**Winner: AGENTIC**
Δ(B2 vs B0)=-149.5µs (-13.8%)   Δ(AG vs B0)=-180.3µs (-16.7%)   Δ(AG vs B1)=-30.8µs (-3.3%)

---

### 32×960×96  —  `q_proj`

**Pre-screened skip**: bf16_accuracy_small_MN_large_K

---

### 32×960×192  —  `q_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,192,960) | (32,192,960) | (32,192,960) |
| Tile | (16,32,32) | (32,32,64) | (32,32,64) |
| Tile source | fixed | profiling | profiling |
| Pad impl | none | none | none |
| Trial 1 avg (µs) | 171.8 | 168.6 | 169.5 |
| Trial 2 avg (µs) | 173.5 | 170.2 | 168.6 |
| Trial 3 avg (µs) | 173.8 | 166.9 | 168.4 |
| **Mean avg (µs)** | **FAIL** | **168.6** | **168.8** |
| Pass count | 0/3 | 3/3 | 3/3 |

**Winner: INCONCLUSIVE**
Δ(AG vs B1)=+nanµs (+nan%)

---

### 32×960×288  —  `q_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,288,960) | (32,288,960) | (32,288,960) |
| Tile | (16,32,32) | (16,32,64) | (16,32,64) |
| Tile source | fixed | profiling | profiling |
| Pad impl | none | none | none |
| Trial 1 avg (µs) | 243.9 | 242.2 | 250.7 |
| Trial 2 avg (µs) | 241.4 | 244.8 | 243.3 |
| Trial 3 avg (µs) | 246.7 | 243.0 | 242.2 |
| **Mean avg (µs)** | **FAIL** | **243.3** | **245.4** |
| Pass count | 0/3 | 3/3 | 3/3 |

**Winner: INCONCLUSIVE**
Δ(AG vs B1)=+nanµs (+nan%)

---

### 32×960×384  —  `q_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,384,960) | (32,384,960) | (32,384,960) |
| Tile | (16,32,32) | (16,32,64) | (16,32,64) |
| Tile source | fixed | profiling | profiling |
| Pad impl | none | none | none |
| Trial 1 avg (µs) | 198.6 | 198.5 | 195.6 |
| Trial 2 avg (µs) | 199.1 | 196.7 | 198.6 |
| Trial 3 avg (µs) | 199.3 | 195.7 | 197.9 |
| **Mean avg (µs)** | **FAIL** | **197.0** | **197.4** |
| Pass count | 0/3 | 3/3 | 3/3 |

**Winner: INCONCLUSIVE**
Δ(AG vs B1)=+nanµs (+nan%)

---

### 32×960×480  —  `q_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,480,960) | (32,480,960) | (32,480,960) |
| Tile | (16,32,32) | (32,32,64) | (32,32,64) |
| Tile source | fixed | profiling | profiling |
| Pad impl | none | none | none |
| Trial 1 avg (µs) | 308.4 | 304.8 | 302.9 |
| Trial 2 avg (µs) | 312.0 | 310.7 | 310.2 |
| Trial 3 avg (µs) | 305.7 | 309.3 | 308.4 |
| **Mean avg (µs)** | **FAIL** | **308.3** | **307.2** |
| Pass count | 0/3 | 3/3 | 3/3 |

**Winner: INCONCLUSIVE**
Δ(AG vs B1)=+nanµs (+nan%)

---

### 32×960×576  —  `q_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,576,960) | (32,576,960) | (32,576,960) |
| Tile | (16,32,32) | (32,16,64) | (32,16,64) |
| Tile source | fixed | profiling | profiling |
| Pad impl | none | none | none |
| Trial 1 avg (µs) | 372.0 | 278.9 | 283.2 |
| Trial 2 avg (µs) | 371.1 | 277.9 | 282.5 |
| Trial 3 avg (µs) | 371.1 | 275.4 | 281.4 |
| **Mean avg (µs)** | **FAIL** | **277.4** | **282.3** |
| Pass count | 0/3 | 3/3 | 3/3 |

**Winner: INCONCLUSIVE**
Δ(AG vs B1)=+nanµs (+nan%)

---

### 32×960×672  —  `q_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,672,960) | (32,672,960) | (32,672,960) |
| Tile | (16,32,32) | (32,32,64) | (32,32,64) |
| Tile source | fixed | profiling | profiling |
| Pad impl | none | none | none |
| Trial 1 avg (µs) | 427.9 | 425.1 | 431.9 |
| Trial 2 avg (µs) | 512.3 | 429.6 | 424.1 |
| Trial 3 avg (µs) | 523.6 | 426.5 | 451.1 |
| **Mean avg (µs)** | **FAIL** | **427.1** | **435.7** |
| Pass count | 0/3 | 3/3 | 3/3 |

**Winner: INCONCLUSIVE**
Δ(AG vs B1)=+nanµs (+nan%)

---

### 32×960×720  —  `q_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,768,960) | (32,768,960) | (32,768,960) |
| Tile | (16,32,32) | (16,32,32) | (16,32,32) |
| Tile source | fixed | fallback_fixed | heuristic |
| Pad impl | numpy_pad | numpy_pad | numpy_pad |
| Trial 1 avg (µs) | 261.7 | 261.1 | 265.3 |
| Trial 2 avg (µs) | 263.5 | 259.5 | 265.6 |
| Trial 3 avg (µs) | 264.6 | 262.6 | 264.4 |
| **Mean avg (µs)** | **FAIL** | **FAIL** | **FAIL** |
| Pass count | 0/3 | 0/3 | 0/3 |

**Winner: INCONCLUSIVE**

---

### 50×32×96  —  `action_in_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,96,32) | (64,96,32) | (64,96,32) |
| Tile | (32,32,32) | (64,16,32) | (64,16,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 139.0 | 127.6 | 124.1 |
| Trial 2 avg (µs) | 131.0 | 125.6 | 128.8 |
| Trial 3 avg (µs) | 137.3 | 127.3 | 126.3 |
| **Mean avg (µs)** | **135.8** | **126.8** | **126.4** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-12.1µs (-7.4%)   Δ(AG vs B0)=-9.4µs (-5.8%)   Δ(AG vs B1)=+2.6µs (+1.8%)

---

### 50×32×192  —  `action_in_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,192,32) | (64,192,32) | (64,192,32) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 127.9 | 129.3 | 129.4 |
| Trial 2 avg (µs) | 127.5 | 130.2 | 130.5 |
| Trial 3 avg (µs) | 131.6 | 130.3 | 127.2 |
| **Mean avg (µs)** | **129.0** | **129.9** | **129.0** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: BASELINE 2**
Δ(B2 vs B0)=-3.7µs (-2.4%)   Δ(AG vs B0)=+7.2µs (+4.7%)   Δ(AG vs B1)=+10.9µs (+7.3%)

---

### 50×32×288  —  `action_in_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,288,32) | (64,288,32) | (64,288,32) |
| Tile | (32,32,32) | (64,32,16) | (64,32,16) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 134.1 | 132.5 | 129.8 |
| Trial 2 avg (µs) | 134.7 | 129.1 | 135.9 |
| Trial 3 avg (µs) | 131.6 | 131.1 | 131.7 |
| **Mean avg (µs)** | **133.4** | **130.9** | **132.5** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=+1.9µs (+1.2%)   Δ(AG vs B0)=+5.0µs (+3.1%)   Δ(AG vs B1)=+3.1µs (+1.9%)

---

### 50×32×384  —  `action_in_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,384,32) | (64,384,32) | (64,384,32) |
| Tile | (32,32,32) | (32,64,32) | (32,64,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 150.4 | 130.0 | 130.9 |
| Trial 2 avg (µs) | 153.2 | 134.1 | 128.4 |
| Trial 3 avg (µs) | 153.7 | 132.3 | 129.0 |
| **Mean avg (µs)** | **152.4** | **132.1** | **129.4** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: AGENTIC**
Δ(B2 vs B0)=-10.1µs (-5.8%)   Δ(AG vs B0)=-22.1µs (-12.7%)   Δ(AG vs B1)=-12.0µs (-7.3%)

---

### 50×32×480  —  `action_in_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,480,32) | (64,480,32) | (64,480,32) |
| Tile | (32,32,32) | (64,32,16) | (64,32,16) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 144.6 | 138.2 | 144.2 |
| Trial 2 avg (µs) | 141.0 | 148.1 | 140.3 |
| Trial 3 avg (µs) | 142.3 | 138.5 | 137.0 |
| **Mean avg (µs)** | **142.6** | **141.6** | **140.5** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=+1.6µs (+0.9%)   Δ(AG vs B0)=-2.7µs (-1.6%)   Δ(AG vs B1)=-4.2µs (-2.5%)

---

### 50×32×576  —  `action_in_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,576,32) | (64,576,32) | (64,576,32) |
| Tile | (32,32,32) | (32,64,16) | (32,64,16) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 147.7 | 135.4 | 136.6 |
| Trial 2 avg (µs) | 145.5 | 133.9 | 137.4 |
| Trial 3 avg (µs) | 147.1 | 135.1 | 139.4 |
| **Mean avg (µs)** | **146.8** | **134.8** | **137.8** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-11.0µs (-6.2%)   Δ(AG vs B0)=-14.0µs (-7.9%)   Δ(AG vs B1)=-3.0µs (-1.8%)

---

### 50×32×672  —  `action_in_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,672,32) | (64,672,32) | (64,672,32) |
| Tile | (32,32,32) | (64,32,16) | (64,32,16) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 162.4 | 151.0 | 148.5 |
| Trial 2 avg (µs) | 160.0 | 149.7 | 150.9 |
| Trial 3 avg (µs) | 163.8 | 149.3 | 146.6 |
| **Mean avg (µs)** | **162.1** | **150.0** | **148.7** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: BASELINE 2**
Δ(B2 vs B0)=-13.0µs (-6.7%)   Δ(AG vs B0)=-9.2µs (-4.8%)   Δ(AG vs B1)=+3.8µs (+2.1%)

---

### 50×32×720  —  `action_in_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,768,32) | (64,768,32) | (64,768,32) |
| Tile | (32,32,32) | (16,256,16) | (16,256,16) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 184.1 | 153.7 | 152.4 |
| Trial 2 avg (µs) | 182.5 | 152.9 | 152.7 |
| Trial 3 avg (µs) | 183.7 | 152.8 | 151.7 |
| **Mean avg (µs)** | **183.4** | **153.2** | **152.3** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-32.6µs (-15.5%)   Δ(AG vs B0)=-31.6µs (-15.0%)   Δ(AG vs B1)=+0.9µs (+0.5%)

---

### 50×96×32  —  `action_out_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,32,128) | (64,32,128) | (64,32,128) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 125.4 | 125.5 | 126.7 |
| Trial 2 avg (µs) | 126.2 | 123.2 | 126.5 |
| Trial 3 avg (µs) | 127.6 | 127.3 | 127.0 |
| **Mean avg (µs)** | **126.4** | **125.3** | **126.7** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-2.2µs (-1.4%)   Δ(AG vs B0)=+3.0µs (+1.9%)   Δ(AG vs B1)=+5.3µs (+3.5%)

---

### 50×96×256  —  `gate_proj, up_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,256,128) | (64,256,128) | (64,256,128) |
| Tile | (64,64,64) | (32,64,32) | (32,64,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 143.3 | 132.7 | 133.2 |
| Trial 2 avg (µs) | 141.0 | 134.9 | 132.9 |
| Trial 3 avg (µs) | 141.6 | 132.8 | 134.1 |
| **Mean avg (µs)** | **141.9** | **133.5** | **133.4** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: AGENTIC**
Δ(B2 vs B0)=-4.1µs (-2.5%)   Δ(AG vs B0)=-7.3µs (-4.5%)   Δ(AG vs B1)=-3.2µs (-2.0%)

---

### 50×96×960  —  `o_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,960,128) | (64,960,128) | (64,960,128) |
| Tile | (64,64,64) | (16,64,32) | (16,64,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 186.8 | 201.5 | 201.8 |
| Trial 2 avg (µs) | 185.7 | 202.5 | 203.6 |
| Trial 3 avg (µs) | 183.0 | 201.7 | 203.2 |
| **Mean avg (µs)** | **185.2** | **201.9** | **202.9** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: BASELINE 0**
Δ(B2 vs B0)=+22.6µs (+10.7%)   Δ(AG vs B0)=+19.4µs (+9.2%)   Δ(AG vs B1)=-3.2µs (-1.4%)

---

### 50×192×32  —  `action_out_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,32,256) | (64,32,256) | (64,32,256) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 125.4 | 125.2 | 125.0 |
| Trial 2 avg (µs) | 125.9 | 126.3 | 126.5 |
| Trial 3 avg (µs) | 127.3 | 125.9 | 124.8 |
| **Mean avg (µs)** | **126.2** | **125.8** | **125.4** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-1.8µs (-1.2%)   Δ(AG vs B0)=+2.2µs (+1.4%)   Δ(AG vs B1)=+4.0µs (+2.6%)

---

### 50×192×256  —  `gate_proj, up_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,256,256) | (64,256,256) | (64,256,256) |
| Tile | (64,64,64) | (16,64,32) | (16,64,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 151.2 | 139.8 | 138.7 |
| Trial 2 avg (µs) | 148.2 | 137.6 | 137.7 |
| Trial 3 avg (µs) | 154.7 | 135.8 | 138.6 |
| **Mean avg (µs)** | **151.3** | **137.7** | **138.3** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: AGENTIC**
Δ(B2 vs B0)=-3.3µs (-1.9%)   Δ(AG vs B0)=-13.4µs (-7.6%)   Δ(AG vs B1)=-10.1µs (-5.9%)

---

### 50×192×960  —  `o_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,960,256) | (64,960,256) | (64,960,256) |
| Tile | (64,64,64) | (32,64,32) | (32,64,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 229.6 | 223.4 | 223.1 |
| Trial 2 avg (µs) | 228.7 | 223.8 | 228.1 |
| Trial 3 avg (µs) | 228.3 | 226.2 | 226.1 |
| **Mean avg (µs)** | **228.9** | **224.5** | **225.8** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-6.3µs (-2.4%)   Δ(AG vs B0)=-3.5µs (-1.3%)   Δ(AG vs B1)=+2.8µs (+1.1%)

---

### 50×256×96  —  `down_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,96,256) | (64,96,256) | (64,96,256) |
| Tile | (32,32,32) | (32,16,64) | (32,16,64) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 137.0 | 130.6 | 130.2 |
| Trial 2 avg (µs) | 139.7 | 128.3 | 131.2 |
| Trial 3 avg (µs) | 137.1 | 131.3 | 133.1 |
| **Mean avg (µs)** | **137.9** | **130.1** | **131.5** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: AGENTIC**
Δ(B2 vs B0)=+1.2µs (+0.7%)   Δ(AG vs B0)=-5.8µs (-3.5%)   Δ(AG vs B1)=-7.0µs (-4.2%)

---

### 50×256×192  —  `down_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,192,256) | (64,192,256) | (64,192,256) |
| Tile | (64,64,64) | (64,32,128) | (64,32,128) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 158.4 | 143.7 | 138.8 |
| Trial 2 avg (µs) | 154.7 | 144.1 | 139.7 |
| Trial 3 avg (µs) | 156.7 | 144.7 | 141.6 |
| **Mean avg (µs)** | **156.6** | **144.2** | **140.0** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: AGENTIC**
Δ(B2 vs B0)=-15.6µs (-8.2%)   Δ(AG vs B0)=-20.4µs (-10.8%)   Δ(AG vs B1)=-4.9µs (-2.8%)

---

### 50×256×288  —  `down_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,288,256) | (64,288,256) | (64,288,256) |
| Tile | (32,32,32) | (32,32,16) | (32,32,16) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 157.2 | 162.6 | 154.6 |
| Trial 2 avg (µs) | 158.2 | 160.8 | 159.5 |
| Trial 3 avg (µs) | 156.3 | 157.7 | 156.5 |
| **Mean avg (µs)** | **157.2** | **160.4** | **156.8** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=+3.4µs (+1.9%)   Δ(AG vs B0)=-0.3µs (-0.1%)   Δ(AG vs B1)=-3.7µs (-2.0%)

---

### 50×256×384  —  `down_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,384,256) | (64,384,256) | (64,384,256) |
| Tile | (64,64,64) | (32,32,32) | (32,32,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 157.1 | 153.5 | 156.8 |
| Trial 2 avg (µs) | 159.3 | 153.8 | 156.7 |
| Trial 3 avg (µs) | 158.6 | 152.1 | 154.1 |
| **Mean avg (µs)** | **158.3** | **153.2** | **155.9** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-9.4µs (-5.0%)   Δ(AG vs B0)=-8.1µs (-4.2%)   Δ(AG vs B1)=+1.4µs (+0.8%)

---

### 50×256×480  —  `down_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,480,256) | (64,480,256) | (64,480,256) |
| Tile | (32,32,32) | (32,32,64) | (32,32,64) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 170.8 | 177.4 | 176.7 |
| Trial 2 avg (µs) | 175.4 | 181.4 | 175.9 |
| Trial 3 avg (µs) | 177.5 | 176.9 | 176.0 |
| **Mean avg (µs)** | **174.6** | **178.6** | **176.2** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=+4.7µs (+2.3%)   Δ(AG vs B0)=+3.2µs (+1.6%)   Δ(AG vs B1)=-1.4µs (-0.7%)

---

### 50×256×576  —  `down_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,576,256) | (64,576,256) | (64,576,256) |
| Tile | (64,64,64) | (32,64,32) | (32,64,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 199.7 | 190.2 | 194.8 |
| Trial 2 avg (µs) | 200.6 | 188.9 | 187.3 |
| Trial 3 avg (µs) | 203.2 | 192.3 | 192.9 |
| **Mean avg (µs)** | **201.2** | **190.5** | **191.7** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-8.0µs (-3.5%)   Δ(AG vs B0)=-6.8µs (-3.0%)   Δ(AG vs B1)=+1.1µs (+0.5%)

---

### 50×256×672  —  `down_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,672,256) | (64,672,256) | (64,672,256) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 206.9 | 206.1 | 209.8 |
| Trial 2 avg (µs) | 207.8 | 206.6 | 207.6 |
| Trial 3 avg (µs) | 205.0 | 208.1 | 209.6 |
| **Mean avg (µs)** | **206.6** | **206.9** | **209.0** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=+1.6µs (+0.7%)   Δ(AG vs B0)=+1.9µs (+0.8%)   Δ(AG vs B1)=+0.3µs (+0.1%)

---

### 50×256×720  —  `down_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,768,256) | (64,768,256) | (64,768,256) |
| Tile | (64,64,64) | (16,64,128) | (16,64,128) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 182.5 | 183.1 | 180.4 |
| Trial 2 avg (µs) | 182.2 | 181.3 | 180.7 |
| Trial 3 avg (µs) | 183.4 | 180.4 | 183.5 |
| **Mean avg (µs)** | **182.7** | **181.6** | **181.5** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-0.5µs (-0.2%)   Δ(AG vs B0)=-0.5µs (-0.2%)   Δ(AG vs B1)=-0.0µs (-0.0%)

---

### 50×288×32  —  `action_out_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,32,512) | (64,32,512) | (64,32,512) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 137.2 | 136.6 | 135.1 |
| Trial 2 avg (µs) | 132.6 | 132.7 | 128.5 |
| Trial 3 avg (µs) | 133.6 | 133.2 | 134.4 |
| **Mean avg (µs)** | **134.4** | **134.2** | **132.7** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=+0.3µs (+0.2%)   Δ(AG vs B0)=-2.6µs (-1.6%)   Δ(AG vs B1)=-2.9µs (-1.8%)

---

### 50×288×256  —  `gate_proj, up_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,256,512) | (64,256,512) | (64,256,512) |
| Tile | (64,64,64) | (16,64,32) | (16,64,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 162.8 | 151.1 | 151.3 |
| Trial 2 avg (µs) | 164.1 | 151.5 | 151.5 |
| Trial 3 avg (µs) | 157.3 | 146.9 | 151.8 |
| **Mean avg (µs)** | **161.4** | **149.8** | **151.6** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: BASELINE 2**
Δ(B2 vs B0)=-10.4µs (-5.5%)   Δ(AG vs B0)=-6.8µs (-3.6%)   Δ(AG vs B1)=+3.6µs (+2.0%)

---

### 50×288×960  —  `o_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,960,512) | (64,960,512) | (64,960,512) |
| Tile | (64,64,64) | (32,16,32) | (32,16,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 328.4 | 311.7 | 310.1 |
| Trial 2 avg (µs) | 329.1 | 396.2 | 310.7 |
| Trial 3 avg (µs) | 328.4 | 307.4 | 307.2 |
| **Mean avg (µs)** | **328.6** | **338.4** | **309.3** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: AGENTIC**
Δ(B2 vs B0)=+3.9µs (+0.8%)   Δ(AG vs B0)=-22.7µs (-4.3%)   Δ(AG vs B1)=-26.6µs (-5.0%)

---

### 50×384×32  —  `action_out_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,32,512) | (64,32,512) | (64,32,512) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 129.4 | 137.8 | 133.2 |
| Trial 2 avg (µs) | 133.9 | 134.3 | 133.0 |
| Trial 3 avg (µs) | 135.9 | 136.7 | 134.9 |
| **Mean avg (µs)** | **133.1** | **136.3** | **133.7** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=+8.8µs (+5.6%)   Δ(AG vs B0)=+1.9µs (+1.2%)   Δ(AG vs B1)=-7.0µs (-4.2%)

---

### 50×384×256  —  `gate_proj, up_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,256,512) | (64,256,512) | (64,256,512) |
| Tile | (64,64,64) | (16,64,32) | (16,64,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 160.4 | 150.5 | 149.3 |
| Trial 2 avg (µs) | 162.1 | 150.0 | 151.1 |
| Trial 3 avg (µs) | 161.1 | 149.7 | 150.0 |
| **Mean avg (µs)** | **161.2** | **150.1** | **150.1** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-9.4µs (-5.0%)   Δ(AG vs B0)=-8.1µs (-4.3%)   Δ(AG vs B1)=+1.3µs (+0.7%)

---

### 50×384×960  —  `o_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,960,512) | (64,960,512) | (64,960,512) |
| Tile | (64,64,64) | (32,16,32) | (32,16,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 324.6 | 309.7 | 314.7 |
| Trial 2 avg (µs) | 327.9 | 312.3 | 309.9 |
| Trial 3 avg (µs) | 328.8 | 327.7 | 312.6 |
| **Mean avg (µs)** | **327.1** | **316.6** | **312.4** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-17.8µs (-3.1%)   Δ(AG vs B0)=-23.4µs (-4.0%)   Δ(AG vs B1)=-5.7µs (-1.0%)

---

### 50×480×32  —  `action_out_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,32,512) | (64,32,512) | (64,32,512) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 133.6 | 130.8 | 130.4 |
| Trial 2 avg (µs) | 133.4 | 134.3 | 134.8 |
| Trial 3 avg (µs) | 131.0 | 129.5 | 133.9 |
| **Mean avg (µs)** | **132.7** | **131.5** | **133.1** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: AGENTIC**
Δ(B2 vs B0)=-2.4µs (-1.5%)   Δ(AG vs B0)=-8.5µs (-5.1%)   Δ(AG vs B1)=-6.1µs (-3.7%)

---

### 50×480×256  —  `gate_proj, up_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,256,512) | (64,256,512) | (64,256,512) |
| Tile | (64,64,64) | (16,64,32) | (16,64,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 161.4 | 152.0 | 151.0 |
| Trial 2 avg (µs) | 158.8 | 153.7 | 153.8 |
| Trial 3 avg (µs) | 160.8 | 152.3 | 148.4 |
| **Mean avg (µs)** | **160.3** | **152.7** | **151.1** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-8.5µs (-4.5%)   Δ(AG vs B0)=-10.3µs (-5.5%)   Δ(AG vs B1)=-1.8µs (-1.0%)

---

### 50×480×960  —  `o_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,960,512) | (64,960,512) | (64,960,512) |
| Tile | (64,64,64) | (32,16,32) | (32,16,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 326.6 | 413.0 | 310.9 |
| Trial 2 avg (µs) | 323.3 | 312.3 | 426.6 |
| Trial 3 avg (µs) | 328.5 | 309.8 | 308.1 |
| **Mean avg (µs)** | **326.1** | **345.1** | **348.5** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: BASELINE 0**
Δ(B2 vs B0)=+25.0µs (+4.0%)   Δ(AG vs B0)=+32.4µs (+5.2%)   Δ(AG vs B1)=+7.5µs (+1.2%)

---

### 50×576×32  —  `action_out_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,32,768) | (64,32,768) | (64,32,768) |
| Tile | (32,32,32) | (32,32,64) | (32,32,64) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 142.2 | 140.4 | 138.0 |
| Trial 2 avg (µs) | 139.2 | 141.0 | 141.3 |
| Trial 3 avg (µs) | 142.5 | 135.1 | 138.7 |
| **Mean avg (µs)** | **141.3** | **138.8** | **139.3** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: AGENTIC**
Δ(B2 vs B0)=+3.2µs (+1.9%)   Δ(AG vs B0)=-5.0µs (-3.0%)   Δ(AG vs B1)=-8.2µs (-4.7%)

---

### 50×576×256  —  `gate_proj, up_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,256,768) | (64,256,768) | (64,256,768) |
| Tile | (64,64,64) | (32,64,64) | (32,64,64) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 171.2 | 157.5 | 161.2 |
| Trial 2 avg (µs) | 173.4 | 157.0 | 158.6 |
| Trial 3 avg (µs) | 170.6 | 157.1 | 159.8 |
| **Mean avg (µs)** | **171.7** | **157.2** | **159.9** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-14.1µs (-6.9%)   Δ(AG vs B0)=-12.3µs (-6.0%)   Δ(AG vs B1)=+1.8µs (+1.0%)

---

### 50×576×960  —  `o_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,960,768) | (64,960,768) | (64,960,768) |
| Tile | (64,64,64) | (32,16,128) | (32,16,128) |
| Tile source | fixed | profiling | profiling |
| Pad impl | numpy_pad | numpy_pad | numpy_pad |
| Trial 1 avg (µs) | 426.8 | 458.4 | 338.4 |
| Trial 2 avg (µs) | 426.5 | 389.7 | 340.1 |
| Trial 3 avg (µs) | 424.2 | 340.3 | 342.5 |
| **Mean avg (µs)** | **425.8** | **396.1** | **340.3** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: AGENTIC**
Δ(B2 vs B0)=+13.4µs (+1.0%)   Δ(AG vs B0)=-85.1µs (-6.5%)   Δ(AG vs B1)=-98.5µs (-7.4%)

---

### 50×672×32  —  `action_out_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,32,768) | (64,32,768) | (64,32,768) |
| Tile | (32,32,32) | (32,32,64) | (32,32,64) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 143.1 | 138.7 | 140.3 |
| Trial 2 avg (µs) | 144.7 | 138.7 | 135.1 |
| Trial 3 avg (µs) | 140.9 | 140.0 | 136.5 |
| **Mean avg (µs)** | **142.9** | **139.2** | **137.3** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-13.2µs (-7.5%)   Δ(AG vs B0)=-10.4µs (-5.9%)   Δ(AG vs B1)=+2.9µs (+1.8%)

---

### 50×672×256  —  `gate_proj, up_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,256,768) | (64,256,768) | (64,256,768) |
| Tile | (64,64,64) | (32,64,64) | (32,64,64) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 172.7 | 160.0 | 159.5 |
| Trial 2 avg (µs) | 170.7 | 157.7 | 159.0 |
| Trial 3 avg (µs) | 170.0 | 157.7 | 159.3 |
| **Mean avg (µs)** | **171.2** | **158.5** | **159.3** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-13.1µs (-6.4%)   Δ(AG vs B0)=-12.9µs (-6.3%)   Δ(AG vs B1)=+0.2µs (+0.1%)

---

### 50×672×960  —  `o_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,960,768) | (64,960,768) | (64,960,768) |
| Tile | (64,64,64) | (32,16,128) | (32,16,128) |
| Tile source | fixed | profiling | profiling |
| Pad impl | numpy_pad | numpy_pad | numpy_pad |
| Trial 1 avg (µs) | 424.4 | 421.3 | 337.8 |
| Trial 2 avg (µs) | 424.9 | 341.6 | 450.1 |
| Trial 3 avg (µs) | 421.1 | 341.1 | 340.9 |
| **Mean avg (µs)** | **423.5** | **368.0** | **376.3** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: AGENTIC**
Δ(B2 vs B0)=+63.5µs (+4.7%)   Δ(AG vs B0)=-32.3µs (-2.4%)   Δ(AG vs B1)=-95.8µs (-6.8%)

---

### 50×720×32  —  `action_out_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,32,768) | (64,32,768) | (64,32,768) |
| Tile | (32,32,32) | (32,32,64) | (32,32,64) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 138.1 | 141.5 | 137.6 |
| Trial 2 avg (µs) | 139.0 | 138.2 | 139.4 |
| Trial 3 avg (µs) | 137.8 | 139.1 | 133.9 |
| **Mean avg (µs)** | **138.3** | **139.6** | **137.0** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: AGENTIC**
Δ(B2 vs B0)=+1.3µs (+0.8%)   Δ(AG vs B0)=-4.4µs (-2.7%)   Δ(AG vs B1)=-5.8µs (-3.4%)

---

### 50×720×256  —  `gate_proj, up_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,256,768) | (64,256,768) | (64,256,768) |
| Tile | (64,64,64) | (32,64,64) | (32,64,64) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 172.3 | 159.7 | 154.9 |
| Trial 2 avg (µs) | 170.0 | 159.0 | 157.9 |
| Trial 3 avg (µs) | 170.8 | 156.4 | 160.8 |
| **Mean avg (µs)** | **171.0** | **158.3** | **157.9** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-22.8µs (-10.6%)   Δ(AG vs B0)=-23.0µs (-10.7%)   Δ(AG vs B1)=-0.2µs (-0.1%)

---

### 50×720×960  —  `o_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,960,768) | (64,960,768) | (64,960,768) |
| Tile | (64,64,64) | (32,16,128) | (32,16,128) |
| Tile source | fixed | profiling | profiling |
| Pad impl | numpy_pad | numpy_pad | numpy_pad |
| Trial 1 avg (µs) | 450.4 | 338.5 | 336.6 |
| Trial 2 avg (µs) | 425.2 | 339.8 | 338.1 |
| Trial 3 avg (µs) | 429.1 | 343.4 | 342.4 |
| **Mean avg (µs)** | **434.9** | **340.6** | **339.1** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-80.7µs (-7.6%)   Δ(AG vs B0)=-91.5µs (-8.6%)   Δ(AG vs B1)=-10.8µs (-1.1%)

---

### 50×960×96  —  `q_proj`

**Pre-screened skip**: bf16_accuracy_small_MN_large_K

---

### 50×960×192  —  `q_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,192,960) | (64,192,960) | (64,192,960) |
| Tile | (64,64,64) | (64,32,64) | (64,32,64) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 222.1 | 172.3 | 172.5 |
| Trial 2 avg (µs) | 222.7 | 174.6 | 171.3 |
| Trial 3 avg (µs) | 222.4 | 172.4 | 177.2 |
| **Mean avg (µs)** | **222.4** | **173.1** | **173.7** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-49.3µs (-19.3%)   Δ(AG vs B0)=-48.9µs (-19.2%)   Δ(AG vs B1)=+0.4µs (+0.2%)

---

### 50×960×288  —  `q_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,288,960) | (64,288,960) | (64,288,960) |
| Tile | (16,32,32) | (32,32,64) | (32,32,64) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 269.1 | 266.2 | 242.0 |
| Trial 2 avg (µs) | 270.2 | 248.6 | 241.6 |
| Trial 3 avg (µs) | 268.8 | 248.7 | 242.3 |
| **Mean avg (µs)** | **FAIL** | **254.5** | **242.0** |
| Pass count | 0/3 | 3/3 | 3/3 |

**Winner: INCONCLUSIVE**
Δ(AG vs B1)=-17.5µs (-3.9%)

---

### 50×960×384  —  `q_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,384,960) | (64,384,960) | (64,384,960) |
| Tile | (64,64,64) | (32,32,64) | (32,32,64) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 224.1 | 204.4 | 202.6 |
| Trial 2 avg (µs) | 220.3 | 206.3 | 204.8 |
| Trial 3 avg (µs) | 220.5 | 206.3 | 205.8 |
| **Mean avg (µs)** | **222.3** | **205.7** | **204.4** |
| Pass count | 2/3 | 3/3 | 3/3 |

**Winner: BASELINE 2**
Δ(B2 vs B0)=-21.2µs (-4.6%)   Δ(AG vs B0)=-7.5µs (-1.6%)   Δ(AG vs B1)=+13.7µs (+3.1%)

---

### 50×960×480  —  `q_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,480,960) | (64,480,960) | (64,480,960) |
| Tile | (16,32,32) | (32,32,64) | (32,32,64) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 361.8 | 316.8 | 306.1 |
| Trial 2 avg (µs) | 365.7 | 307.7 | 306.3 |
| Trial 3 avg (µs) | 384.4 | 304.0 | 384.3 |
| **Mean avg (µs)** | **FAIL** | **309.5** | **332.2** |
| Pass count | 0/3 | 3/3 | 3/3 |

**Winner: INCONCLUSIVE**
Δ(AG vs B1)=+31.4µs (+5.0%)

---

### 50×960×576  —  `q_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,576,960) | (64,576,960) | (64,576,960) |
| Tile | (64,64,64) | (32,16,64) | (32,16,64) |
| Tile source | fixed | profiling | profiling |
| Pad impl | numpy_pad | numpy_pad | numpy_pad |
| Trial 1 avg (µs) | 477.3 | 303.5 | 303.3 |
| Trial 2 avg (µs) | 443.3 | 298.2 | 295.8 |
| Trial 3 avg (µs) | 456.4 | 295.5 | 296.6 |
| **Mean avg (µs)** | **459.0** | **299.1** | **298.5** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: AGENTIC**
Δ(B2 vs B0)=+107.6µs (+8.8%)   Δ(AG vs B0)=-126.8µs (-10.4%)   Δ(AG vs B1)=-234.4µs (-17.7%)

---

### 50×960×672  —  `q_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,672,960) | (64,672,960) | (64,672,960) |
| Tile | (16,32,32) | (32,32,64) | (32,32,64) |
| Tile source | fixed | profiling | profiling |
| Pad impl | numpy_pad | numpy_pad | numpy_pad |
| Trial 1 avg (µs) | 455.1 | 430.1 | 425.7 |
| Trial 2 avg (µs) | 459.0 | 427.0 | 433.1 |
| Trial 3 avg (µs) | 626.0 | 544.9 | 523.2 |
| **Mean avg (µs)** | **FAIL** | **467.3** | **460.7** |
| Pass count | 0/3 | 3/3 | 3/3 |

**Winner: INCONCLUSIVE**
Δ(AG vs B1)=-26.9µs (-2.0%)

---

### 50×960×720  —  `q_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,768,960) | (64,768,960) | (64,768,960) |
| Tile | (64,64,64) | (16,64,64) | (16,64,64) |
| Tile source | fixed | profiling | profiling |
| Pad impl | numpy_pad | numpy_pad | numpy_pad |
| Trial 1 avg (µs) | 272.9 | 259.4 | 254.4 |
| Trial 2 avg (µs) | 271.8 | 261.6 | 254.8 |
| Trial 3 avg (µs) | 277.7 | 258.4 | 255.7 |
| **Mean avg (µs)** | **274.1** | **259.8** | **255.0** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: BASELINE 0**
Δ(B2 vs B0)=+150.0µs (+17.4%)   Δ(AG vs B0)=+210.3µs (+24.4%)   Δ(AG vs B1)=+60.4µs (+6.0%)

---

### 60×32×96  —  `action_in_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,96,32) | (64,96,32) | (64,96,32) |
| Tile | (32,32,32) | (64,16,32) | (64,16,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 133.5 | 129.0 | 125.2 |
| Trial 2 avg (µs) | 132.6 | 129.8 | 126.6 |
| Trial 3 avg (µs) | 133.5 | 127.0 | 129.8 |
| **Mean avg (µs)** | **133.2** | **128.6** | **127.2** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-3.2µs (-2.0%)   Δ(AG vs B0)=-2.6µs (-1.7%)   Δ(AG vs B1)=+0.6µs (+0.4%)

---

### 60×32×192  —  `action_in_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,192,32) | (64,192,32) | (64,192,32) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 126.2 | 130.2 | 129.1 |
| Trial 2 avg (µs) | 127.3 | 133.7 | 131.4 |
| Trial 3 avg (µs) | 127.1 | 127.3 | 130.0 |
| **Mean avg (µs)** | **126.9** | **130.4** | **130.2** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=+7.5µs (+5.0%)   Δ(AG vs B0)=+2.9µs (+1.9%)   Δ(AG vs B1)=-4.7µs (-3.0%)

---

### 60×32×288  —  `action_in_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,288,32) | (64,288,32) | (64,288,32) |
| Tile | (32,32,32) | (64,32,16) | (64,32,16) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 137.9 | 134.5 | 130.4 |
| Trial 2 avg (µs) | 135.8 | 128.1 | 133.4 |
| Trial 3 avg (µs) | 136.7 | 128.7 | 130.4 |
| **Mean avg (µs)** | **136.8** | **130.4** | **131.4** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-2.1µs (-1.3%)   Δ(AG vs B0)=-5.0µs (-3.1%)   Δ(AG vs B1)=-2.9µs (-1.8%)

---

### 60×32×384  —  `action_in_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,384,32) | (64,384,32) | (64,384,32) |
| Tile | (32,32,32) | (32,64,32) | (32,64,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 151.5 | 130.6 | 132.6 |
| Trial 2 avg (µs) | 153.8 | 133.7 | 130.9 |
| Trial 3 avg (µs) | 151.4 | 131.1 | 130.7 |
| **Mean avg (µs)** | **152.2** | **131.8** | **131.4** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: BASELINE 2**
Δ(B2 vs B0)=-16.2µs (-9.1%)   Δ(AG vs B0)=-11.8µs (-6.6%)   Δ(AG vs B1)=+4.4µs (+2.7%)

---

### 60×32×480  —  `action_in_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,480,32) | (64,480,32) | (64,480,32) |
| Tile | (32,32,32) | (64,32,16) | (64,32,16) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 144.2 | 146.2 | 136.5 |
| Trial 2 avg (µs) | 143.9 | 137.5 | 137.6 |
| Trial 3 avg (µs) | 144.5 | 147.9 | 134.6 |
| **Mean avg (µs)** | **144.2** | **143.9** | **136.2** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: AGENTIC**
Δ(B2 vs B0)=+0.4µs (+0.2%)   Δ(AG vs B0)=-7.4µs (-4.3%)   Δ(AG vs B1)=-7.8µs (-4.5%)

---

### 60×32×576  —  `action_in_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,576,32) | (64,576,32) | (64,576,32) |
| Tile | (32,32,32) | (32,64,16) | (32,64,16) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 147.2 | 138.4 | 133.5 |
| Trial 2 avg (µs) | 147.0 | 130.6 | 135.5 |
| Trial 3 avg (µs) | 148.0 | 138.8 | 136.7 |
| **Mean avg (µs)** | **147.4** | **135.9** | **135.3** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-0.1µs (-0.1%)   Δ(AG vs B0)=-3.0µs (-1.8%)   Δ(AG vs B1)=-2.9µs (-1.7%)

---

### 60×32×672  —  `action_in_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,672,32) | (64,672,32) | (64,672,32) |
| Tile | (32,32,32) | (64,32,16) | (64,32,16) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 162.2 | 149.4 | 146.8 |
| Trial 2 avg (µs) | 163.7 | 150.3 | 147.7 |
| Trial 3 avg (µs) | 153.2 | 146.7 | 149.4 |
| **Mean avg (µs)** | **159.7** | **148.8** | **147.9** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-18.7µs (-9.9%)   Δ(AG vs B0)=-16.2µs (-8.5%)   Δ(AG vs B1)=+2.5µs (+1.5%)

---

### 60×32×720  —  `action_in_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,768,32) | (64,768,32) | (64,768,32) |
| Tile | (32,32,32) | (16,256,16) | (16,256,16) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 181.7 | 154.0 | 153.4 |
| Trial 2 avg (µs) | 182.8 | 155.0 | 149.2 |
| Trial 3 avg (µs) | 184.0 | 154.5 | 153.2 |
| **Mean avg (µs)** | **182.8** | **154.5** | **151.9** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-34.6µs (-16.5%)   Δ(AG vs B0)=-31.9µs (-15.2%)   Δ(AG vs B1)=+2.7µs (+1.5%)

---

### 60×96×32  —  `action_out_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,32,128) | (64,32,128) | (64,32,128) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 129.1 | 125.4 | 124.9 |
| Trial 2 avg (µs) | 124.0 | 125.3 | 126.4 |
| Trial 3 avg (µs) | 126.9 | 125.3 | 125.6 |
| **Mean avg (µs)** | **126.7** | **125.3** | **125.6** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: BASELINE 0**
Δ(B2 vs B0)=+6.8µs (+4.6%)   Δ(AG vs B0)=+7.2µs (+4.9%)   Δ(AG vs B1)=+0.4µs (+0.3%)

---

### 60×96×256  —  `gate_proj, up_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,256,128) | (64,256,128) | (64,256,128) |
| Tile | (64,64,64) | (32,64,32) | (32,64,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 144.1 | 135.0 | 135.3 |
| Trial 2 avg (µs) | 139.7 | 134.8 | 135.0 |
| Trial 3 avg (µs) | 140.5 | 133.5 | 132.9 |
| **Mean avg (µs)** | **141.4** | **134.4** | **134.4** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: AGENTIC**
Δ(B2 vs B0)=-0.6µs (-0.3%)   Δ(AG vs B0)=-5.3µs (-3.2%)   Δ(AG vs B1)=-4.8µs (-2.9%)

---

### 60×96×960  —  `o_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,960,128) | (64,960,128) | (64,960,128) |
| Tile | (64,64,64) | (16,64,32) | (16,64,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 186.1 | 201.6 | 201.7 |
| Trial 2 avg (µs) | 189.3 | 204.0 | 202.5 |
| Trial 3 avg (µs) | 188.2 | 207.6 | 200.9 |
| **Mean avg (µs)** | **187.9** | **204.4** | **201.7** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: BASELINE 0**
Δ(B2 vs B0)=+16.0µs (+7.5%)   Δ(AG vs B0)=+14.4µs (+6.7%)   Δ(AG vs B1)=-1.6µs (-0.7%)

---

### 60×192×32  —  `action_out_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,32,256) | (64,32,256) | (64,32,256) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 122.5 | 124.2 | 125.2 |
| Trial 2 avg (µs) | 126.5 | 126.6 | 126.8 |
| Trial 3 avg (µs) | 126.7 | 125.3 | 124.6 |
| **Mean avg (µs)** | **125.2** | **125.4** | **125.6** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: BASELINE 2**
Δ(B2 vs B0)=-11.8µs (-7.4%)   Δ(AG vs B0)=-4.3µs (-2.7%)   Δ(AG vs B1)=+7.5µs (+5.1%)

---

### 60×192×256  —  `gate_proj, up_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,256,256) | (64,256,256) | (64,256,256) |
| Tile | (64,64,64) | (16,64,32) | (16,64,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 151.5 | 136.8 | 137.5 |
| Trial 2 avg (µs) | 151.1 | 139.3 | 136.6 |
| Trial 3 avg (µs) | 154.5 | 138.4 | 135.6 |
| **Mean avg (µs)** | **152.4** | **138.2** | **136.6** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: AGENTIC**
Δ(B2 vs B0)=-8.6µs (-4.7%)   Δ(AG vs B0)=-20.6µs (-11.3%)   Δ(AG vs B1)=-12.1µs (-7.0%)

---

### 60×192×960  —  `o_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,960,256) | (64,960,256) | (64,960,256) |
| Tile | (64,64,64) | (32,64,32) | (32,64,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 230.9 | 230.4 | 226.2 |
| Trial 2 avg (µs) | 234.2 | 226.9 | 225.8 |
| Trial 3 avg (µs) | 227.1 | 225.0 | 227.6 |
| **Mean avg (µs)** | **230.7** | **227.4** | **226.5** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-3.5µs (-1.3%)   Δ(AG vs B0)=-3.4µs (-1.3%)   Δ(AG vs B1)=+0.1µs (+0.0%)

---

### 60×256×96  —  `down_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,96,256) | (64,96,256) | (64,96,256) |
| Tile | (32,32,32) | (32,16,64) | (32,16,64) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 137.5 | 129.8 | 130.0 |
| Trial 2 avg (µs) | 140.0 | 130.1 | 130.2 |
| Trial 3 avg (µs) | 135.2 | 130.5 | 130.1 |
| **Mean avg (µs)** | **137.5** | **130.1** | **130.1** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: BASELINE 2**
Δ(B2 vs B0)=-6.8µs (-4.2%)   Δ(AG vs B0)=-0.3µs (-0.2%)   Δ(AG vs B1)=+6.4µs (+4.1%)

---

### 60×256×192  —  `down_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,192,256) | (64,192,256) | (64,192,256) |
| Tile | (64,64,64) | (64,32,128) | (64,32,128) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 155.7 | 145.1 | 142.9 |
| Trial 2 avg (µs) | 154.2 | 146.2 | 142.1 |
| Trial 3 avg (µs) | 154.2 | 145.4 | 140.8 |
| **Mean avg (µs)** | **154.7** | **145.6** | **141.9** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: AGENTIC**
Δ(B2 vs B0)=-9.8µs (-5.5%)   Δ(AG vs B0)=-13.1µs (-7.3%)   Δ(AG vs B1)=-3.3µs (-2.0%)

---

### 60×256×288  —  `down_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,288,256) | (64,288,256) | (64,288,256) |
| Tile | (32,32,32) | (32,32,16) | (32,32,16) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 159.0 | 155.2 | 155.5 |
| Trial 2 avg (µs) | 158.0 | 162.0 | 156.8 |
| Trial 3 avg (µs) | 162.9 | 163.9 | 159.0 |
| **Mean avg (µs)** | **160.0** | **160.4** | **157.1** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=+2.0µs (+1.0%)   Δ(AG vs B0)=-1.4µs (-0.8%)   Δ(AG vs B1)=-3.4µs (-1.8%)

---

### 60×256×384  —  `down_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,384,256) | (64,384,256) | (64,384,256) |
| Tile | (64,64,64) | (32,32,32) | (32,32,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 156.4 | 152.8 | 154.5 |
| Trial 2 avg (µs) | 153.1 | 151.6 | 152.5 |
| Trial 3 avg (µs) | 155.6 | 156.4 | 150.6 |
| **Mean avg (µs)** | **155.1** | **153.6** | **152.5** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-1.1µs (-0.6%)   Δ(AG vs B0)=-1.2µs (-0.6%)   Δ(AG vs B1)=-0.1µs (-0.0%)

---

### 60×256×480  —  `down_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,480,256) | (64,480,256) | (64,480,256) |
| Tile | (32,32,32) | (32,32,64) | (32,32,64) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 175.8 | 174.5 | 176.6 |
| Trial 2 avg (µs) | 176.1 | 180.3 | 175.5 |
| Trial 3 avg (µs) | 173.4 | 175.2 | 176.9 |
| **Mean avg (µs)** | **175.1** | **176.7** | **176.3** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-2.0µs (-1.0%)   Δ(AG vs B0)=-1.6µs (-0.8%)   Δ(AG vs B1)=+0.5µs (+0.2%)

---

### 60×256×576  —  `down_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,576,256) | (64,576,256) | (64,576,256) |
| Tile | (64,64,64) | (32,64,32) | (32,64,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 197.9 | 187.6 | 190.6 |
| Trial 2 avg (µs) | 195.5 | 187.9 | 195.5 |
| Trial 3 avg (µs) | 196.0 | 190.3 | 186.1 |
| **Mean avg (µs)** | **196.5** | **188.6** | **190.7** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-6.7µs (-3.0%)   Δ(AG vs B0)=-4.9µs (-2.2%)   Δ(AG vs B1)=+1.7µs (+0.8%)

---

### 60×256×672  —  `down_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,672,256) | (64,672,256) | (64,672,256) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 208.3 | 210.1 | 213.3 |
| Trial 2 avg (µs) | 209.7 | 207.6 | 208.2 |
| Trial 3 avg (µs) | 207.1 | 207.1 | 209.3 |
| **Mean avg (µs)** | **208.4** | **208.3** | **210.2** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=+0.7µs (+0.3%)   Δ(AG vs B0)=+1.0µs (+0.4%)   Δ(AG vs B1)=+0.3µs (+0.1%)

---

### 60×256×720  —  `down_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,768,256) | (64,768,256) | (64,768,256) |
| Tile | (64,64,64) | (16,64,128) | (16,64,128) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 183.2 | 181.6 | 180.3 |
| Trial 2 avg (µs) | 184.3 | 182.8 | 182.8 |
| Trial 3 avg (µs) | 187.0 | 182.1 | 180.9 |
| **Mean avg (µs)** | **184.8** | **182.2** | **181.3** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=+17.1µs (+7.8%)   Δ(AG vs B0)=-2.3µs (-1.1%)   Δ(AG vs B1)=-19.4µs (-8.3%)

---

### 60×288×32  —  `action_out_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,32,512) | (64,32,512) | (64,32,512) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 129.6 | 135.8 | 132.5 |
| Trial 2 avg (µs) | 130.3 | 133.0 | 134.6 |
| Trial 3 avg (µs) | 135.2 | 133.7 | 134.3 |
| **Mean avg (µs)** | **131.7** | **134.1** | **133.8** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-2.9µs (-1.8%)   Δ(AG vs B0)=+3.1µs (+1.9%)   Δ(AG vs B1)=+6.0µs (+3.7%)

---

### 60×288×256  —  `gate_proj, up_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,256,512) | (64,256,512) | (64,256,512) |
| Tile | (64,64,64) | (16,64,32) | (16,64,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 160.8 | 151.1 | 150.9 |
| Trial 2 avg (µs) | 162.9 | 152.5 | 149.9 |
| Trial 3 avg (µs) | 161.8 | 149.6 | 149.3 |
| **Mean avg (µs)** | **161.8** | **151.1** | **150.0** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-5.4µs (-2.9%)   Δ(AG vs B0)=-6.9µs (-3.7%)   Δ(AG vs B1)=-1.6µs (-0.8%)

---

### 60×288×960  —  `o_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,960,512) | (64,960,512) | (64,960,512) |
| Tile | (64,64,64) | (32,16,32) | (32,16,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 325.3 | 322.3 | 459.0 |
| Trial 2 avg (µs) | 323.5 | 311.8 | 309.2 |
| Trial 3 avg (µs) | 382.2 | 310.4 | 310.7 |
| **Mean avg (µs)** | **343.7** | **314.8** | **359.6** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: BASELINE 2**
Δ(B2 vs B0)=-47.2µs (-8.7%)   Δ(AG vs B0)=+8.0µs (+1.5%)   Δ(AG vs B1)=+55.2µs (+11.1%)

---

### 60×384×32  —  `action_out_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,32,512) | (64,32,512) | (64,32,512) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 138.0 | 134.2 | 134.1 |
| Trial 2 avg (µs) | 137.0 | 133.8 | 132.8 |
| Trial 3 avg (µs) | 133.7 | 135.0 | 130.4 |
| **Mean avg (µs)** | **136.2** | **134.3** | **132.4** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: AGENTIC**
Δ(B2 vs B0)=-3.3µs (-1.9%)   Δ(AG vs B0)=-8.1µs (-4.8%)   Δ(AG vs B1)=-4.8µs (-2.9%)

---

### 60×384×256  —  `gate_proj, up_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,256,512) | (64,256,512) | (64,256,512) |
| Tile | (64,64,64) | (16,64,32) | (16,64,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 159.1 | 153.2 | 150.4 |
| Trial 2 avg (µs) | 159.4 | 151.3 | 148.4 |
| Trial 3 avg (µs) | 165.4 | 151.5 | 145.1 |
| **Mean avg (µs)** | **161.3** | **152.0** | **147.9** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: AGENTIC**
Δ(B2 vs B0)=-9.1µs (-4.8%)   Δ(AG vs B0)=-12.7µs (-6.8%)   Δ(AG vs B1)=-3.6µs (-2.0%)

---

### 60×384×960  —  `o_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,960,512) | (64,960,512) | (64,960,512) |
| Tile | (64,64,64) | (32,16,32) | (32,16,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 326.8 | 314.3 | 311.5 |
| Trial 2 avg (µs) | 326.5 | 309.7 | 309.0 |
| Trial 3 avg (µs) | 328.9 | 311.2 | 314.1 |
| **Mean avg (µs)** | **327.4** | **311.7** | **311.5** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-36.2µs (-6.2%)   Δ(AG vs B0)=-33.8µs (-5.7%)   Δ(AG vs B1)=+2.4µs (+0.4%)

---

### 60×480×32  —  `action_out_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,32,512) | (64,32,512) | (64,32,512) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 134.3 | 130.6 | 136.4 |
| Trial 2 avg (µs) | 132.3 | 132.0 | 133.9 |
| Trial 3 avg (µs) | 135.4 | 135.3 | 132.3 |
| **Mean avg (µs)** | **134.0** | **132.6** | **134.2** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-2.8µs (-1.8%)   Δ(AG vs B0)=-0.3µs (-0.2%)   Δ(AG vs B1)=+2.5µs (+1.6%)

---

### 60×480×256  —  `gate_proj, up_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,256,512) | (64,256,512) | (64,256,512) |
| Tile | (64,64,64) | (16,64,32) | (16,64,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 164.7 | 152.1 | 147.3 |
| Trial 2 avg (µs) | 161.5 | 151.2 | 148.9 |
| Trial 3 avg (µs) | 158.2 | 149.8 | 148.7 |
| **Mean avg (µs)** | **161.5** | **151.1** | **148.3** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: AGENTIC**
Δ(B2 vs B0)=-8.7µs (-4.6%)   Δ(AG vs B0)=-12.4µs (-6.5%)   Δ(AG vs B1)=-3.7µs (-2.0%)

---

### 60×480×960  —  `o_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,960,512) | (64,960,512) | (64,960,512) |
| Tile | (64,64,64) | (32,16,32) | (32,16,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 326.1 | 308.6 | 311.8 |
| Trial 2 avg (µs) | 328.7 | 312.4 | 312.4 |
| Trial 3 avg (µs) | 331.5 | 315.4 | 380.1 |
| **Mean avg (µs)** | **328.8** | **312.1** | **334.8** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: BASELINE 2**
Δ(B2 vs B0)=-25.6µs (-4.1%)   Δ(AG vs B0)=+14.3µs (+2.3%)   Δ(AG vs B1)=+39.9µs (+6.6%)

---

### 60×576×32  —  `action_out_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,32,768) | (64,32,768) | (64,32,768) |
| Tile | (32,32,32) | (32,32,64) | (32,32,64) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 144.5 | 139.9 | 139.0 |
| Trial 2 avg (µs) | 143.1 | 134.2 | 138.1 |
| Trial 3 avg (µs) | 139.4 | 140.6 | 138.4 |
| **Mean avg (µs)** | **142.3** | **138.3** | **138.5** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=+2.5µs (+1.6%)   Δ(AG vs B0)=-2.6µs (-1.6%)   Δ(AG vs B1)=-5.2µs (-3.1%)

---

### 60×576×256  —  `gate_proj, up_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,256,768) | (64,256,768) | (64,256,768) |
| Tile | (64,64,64) | (32,64,64) | (32,64,64) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 171.5 | 158.5 | 156.2 |
| Trial 2 avg (µs) | 170.1 | 159.2 | 156.7 |
| Trial 3 avg (µs) | 171.5 | 155.4 | 158.0 |
| **Mean avg (µs)** | **171.1** | **157.7** | **157.0** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-13.7µs (-6.7%)   Δ(AG vs B0)=-15.7µs (-7.7%)   Δ(AG vs B1)=-2.0µs (-1.1%)

---

### 60×576×960  —  `o_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,960,768) | (64,960,768) | (64,960,768) |
| Tile | (64,64,64) | (32,16,128) | (32,16,128) |
| Tile source | fixed | profiling | profiling |
| Pad impl | numpy_pad | numpy_pad | numpy_pad |
| Trial 1 avg (µs) | 424.5 | 338.4 | 339.5 |
| Trial 2 avg (µs) | 426.6 | 332.9 | 339.6 |
| Trial 3 avg (µs) | 425.1 | 338.3 | 339.4 |
| **Mean avg (µs)** | **425.4** | **336.5** | **339.5** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: AGENTIC**
Δ(B2 vs B0)=-71.9µs (-5.4%)   Δ(AG vs B0)=-98.1µs (-7.4%)   Δ(AG vs B1)=-26.2µs (-2.1%)

---

### 60×672×32  —  `action_out_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,32,768) | (64,32,768) | (64,32,768) |
| Tile | (32,32,32) | (32,32,64) | (32,32,64) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 141.3 | 135.3 | 138.9 |
| Trial 2 avg (µs) | 142.5 | 138.5 | 136.8 |
| Trial 3 avg (µs) | 141.1 | 136.3 | 140.5 |
| **Mean avg (µs)** | **141.6** | **136.7** | **138.8** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=+0.6µs (+0.4%)   Δ(AG vs B0)=+8.1µs (+4.9%)   Δ(AG vs B1)=+7.4µs (+4.5%)

---

### 60×672×256  —  `gate_proj, up_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,256,768) | (64,256,768) | (64,256,768) |
| Tile | (64,64,64) | (32,64,64) | (32,64,64) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 176.9 | 160.6 | 162.2 |
| Trial 2 avg (µs) | 172.1 | 161.2 | 159.5 |
| Trial 3 avg (µs) | 174.2 | 156.0 | 158.4 |
| **Mean avg (µs)** | **174.4** | **159.3** | **160.0** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-15.9µs (-7.6%)   Δ(AG vs B0)=-14.1µs (-6.7%)   Δ(AG vs B1)=+1.8µs (+0.9%)

---

### 60×672×960  —  `o_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,960,768) | (64,960,768) | (64,960,768) |
| Tile | (64,64,64) | (32,16,128) | (32,16,128) |
| Tile source | fixed | profiling | profiling |
| Pad impl | numpy_pad | numpy_pad | numpy_pad |
| Trial 1 avg (µs) | 423.7 | 336.2 | 338.7 |
| Trial 2 avg (µs) | 423.6 | 336.7 | 337.4 |
| Trial 3 avg (µs) | 424.4 | 332.5 | 339.0 |
| **Mean avg (µs)** | **423.9** | **335.1** | **338.4** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: BASELINE 2**
Δ(B2 vs B0)=-330.6µs (-20.6%)   Δ(AG vs B0)=-223.2µs (-13.9%)   Δ(AG vs B1)=+107.4µs (+8.4%)

---

### 60×720×32  —  `action_out_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,32,768) | (64,32,768) | (64,32,768) |
| Tile | (32,32,32) | (32,32,64) | (32,32,64) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 142.8 | 138.7 | 137.1 |
| Trial 2 avg (µs) | 143.4 | 139.8 | 136.7 |
| Trial 3 avg (µs) | 138.4 | 136.6 | 136.9 |
| **Mean avg (µs)** | **141.6** | **138.3** | **136.9** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: AGENTIC**
Δ(B2 vs B0)=+1.8µs (+1.1%)   Δ(AG vs B0)=-4.6µs (-2.8%)   Δ(AG vs B1)=-6.4µs (-3.9%)

---

### 60×720×256  —  `gate_proj, up_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,256,768) | (64,256,768) | (64,256,768) |
| Tile | (64,64,64) | (32,64,64) | (32,64,64) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 172.7 | 159.2 | 157.6 |
| Trial 2 avg (µs) | 171.6 | 157.4 | 157.8 |
| Trial 3 avg (µs) | 171.7 | 158.2 | 159.4 |
| **Mean avg (µs)** | **172.0** | **158.3** | **158.3** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-13.5µs (-6.5%)   Δ(AG vs B0)=-15.3µs (-7.4%)   Δ(AG vs B1)=-1.9µs (-1.0%)

---

### 60×720×960  —  `o_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,960,768) | (64,960,768) | (64,960,768) |
| Tile | (64,64,64) | (32,16,128) | (32,16,128) |
| Tile source | fixed | profiling | profiling |
| Pad impl | numpy_pad | numpy_pad | numpy_pad |
| Trial 1 avg (µs) | 427.8 | 337.8 | 340.7 |
| Trial 2 avg (µs) | 427.0 | 335.0 | 341.2 |
| Trial 3 avg (µs) | 419.9 | 338.6 | 338.9 |
| **Mean avg (µs)** | **424.9** | **337.2** | **340.3** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-76.9µs (-7.3%)   Δ(AG vs B0)=-92.1µs (-8.8%)   Δ(AG vs B1)=-15.3µs (-1.6%)

---

### 60×960×96  —  `q_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,96,960) | (64,96,960) | (64,96,960) |
| Tile | (16,32,32) | (32,16,64) | (32,16,64) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 171.4 | 150.6 | 154.5 |
| Trial 2 avg (µs) | 170.0 | 156.1 | 153.9 |
| Trial 3 avg (µs) | 169.8 | 153.9 | 155.8 |
| **Mean avg (µs)** | **FAIL** | **153.6** | **154.7** |
| Pass count | 0/3 | 3/3 | 3/3 |

**Winner: INCONCLUSIVE**
Δ(AG vs B1)=-0.1µs (-0.1%)

---

### 60×960×192  —  `q_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,192,960) | (64,192,960) | (64,192,960) |
| Tile | (64,64,64) | (64,32,64) | (64,32,64) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 264.6 | 174.6 | 175.5 |
| Trial 2 avg (µs) | 221.1 | 173.1 | 174.5 |
| Trial 3 avg (µs) | 219.5 | 174.1 | 172.0 |
| **Mean avg (µs)** | **235.1** | **173.9** | **174.0** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: AGENTIC**
Δ(B2 vs B0)=-52.0µs (-19.3%)   Δ(AG vs B0)=-61.5µs (-22.9%)   Δ(AG vs B1)=-9.6µs (-4.4%)

---

### 60×960×288  —  `q_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,288,960) | (64,288,960) | (64,288,960) |
| Tile | (16,32,32) | (32,32,64) | (32,32,64) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 271.5 | 246.5 | 242.8 |
| Trial 2 avg (µs) | 267.7 | 240.9 | 242.1 |
| Trial 3 avg (µs) | 271.4 | 240.1 | 242.2 |
| **Mean avg (µs)** | **FAIL** | **242.5** | **242.5** |
| Pass count | 0/3 | 3/3 | 2/3 |

**Winner: INCONCLUSIVE**
Δ(AG vs B1)=+1.2µs (+0.3%)

---

### 60×960×384  —  `q_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,384,960) | (64,384,960) | (64,384,960) |
| Tile | (64,64,64) | (32,32,64) | (32,32,64) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 223.7 | 204.9 | 205.3 |
| Trial 2 avg (µs) | 221.2 | 206.6 | 205.4 |
| Trial 3 avg (µs) | 220.7 | 206.7 | 205.8 |
| **Mean avg (µs)** | **221.9** | **206.1** | **205.5** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: AGENTIC**
Δ(B2 vs B0)=+30.9µs (+6.6%)   Δ(AG vs B0)=-18.0µs (-3.9%)   Δ(AG vs B1)=-48.9µs (-9.8%)

---

### 60×960×480  —  `q_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,480,960) | (64,480,960) | (64,480,960) |
| Tile | (16,32,32) | (32,32,64) | (32,32,64) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 362.8 | 357.6 | 301.7 |
| Trial 2 avg (µs) | 455.3 | 408.1 | 304.1 |
| Trial 3 avg (µs) | 363.4 | 388.8 | 305.7 |
| **Mean avg (µs)** | **FAIL** | **384.8** | **303.9** |
| Pass count | 0/3 | 3/3 | 3/3 |

**Winner: INCONCLUSIVE**
Δ(AG vs B1)=-84.7µs (-12.4%)

---

### 60×960×576  —  `q_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,576,960) | (64,576,960) | (64,576,960) |
| Tile | (64,64,64) | (32,16,64) | (32,16,64) |
| Tile source | fixed | profiling | profiling |
| Pad impl | numpy_pad | numpy_pad | numpy_pad |
| Trial 1 avg (µs) | 374.4 | 302.5 | 331.8 |
| Trial 2 avg (µs) | 372.6 | 298.3 | 295.8 |
| Trial 3 avg (µs) | 385.9 | 304.5 | 302.3 |
| **Mean avg (µs)** | **377.6** | **301.8** | **309.9** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: BASELINE 2**
Δ(B2 vs B0)=-72.5µs (-6.3%)   Δ(AG vs B0)=-21.0µs (-1.8%)   Δ(AG vs B1)=+51.5µs (+4.8%)

---

### 60×960×672  —  `q_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,672,960) | (64,672,960) | (64,672,960) |
| Tile | (16,32,32) | (32,32,64) | (32,32,64) |
| Tile source | fixed | profiling | profiling |
| Pad impl | numpy_pad | numpy_pad | numpy_pad |
| Trial 1 avg (µs) | 471.7 | 567.7 | 511.6 |
| Trial 2 avg (µs) | 460.5 | 430.3 | 429.6 |
| Trial 3 avg (µs) | 451.8 | 427.9 | 422.4 |
| **Mean avg (µs)** | **FAIL** | **475.3** | **454.5** |
| Pass count | 0/3 | 3/3 | 3/3 |

**Winner: INCONCLUSIVE**
Δ(AG vs B1)=+9.6µs (+0.7%)

---

### 60×960×720  —  `q_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,768,960) | (64,768,960) | (64,768,960) |
| Tile | (64,64,64) | (16,64,64) | (16,64,64) |
| Tile source | fixed | profiling | profiling |
| Pad impl | numpy_pad | numpy_pad | numpy_pad |
| Trial 1 avg (µs) | 276.7 | 434.4 | 260.2 |
| Trial 2 avg (µs) | 279.6 | 263.5 | 258.7 |
| Trial 3 avg (µs) | 274.1 | 264.5 | 258.5 |
| **Mean avg (µs)** | **276.8** | **320.8** | **259.1** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: AGENTIC**
Δ(B2 vs B0)=-96.7µs (-9.4%)   Δ(AG vs B0)=-145.0µs (-14.0%)   Δ(AG vs B1)=-48.3µs (-5.2%)

---

### 115×320×96  —  `k_proj, v_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (128,96,512) | (128,96,512) | (128,96,512) |
| Tile | (32,32,32) | (64,32,64) | (64,32,64) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 151.4 | 151.6 | 152.5 |
| Trial 2 avg (µs) | 149.7 | 155.2 | 152.0 |
| Trial 3 avg (µs) | 155.2 | 155.1 | 153.5 |
| **Mean avg (µs)** | **152.1** | **154.0** | **152.7** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: BASELINE 0**
Δ(B2 vs B0)=+7.9µs (+4.5%)   Δ(AG vs B0)=+7.6µs (+4.3%)   Δ(AG vs B1)=-0.3µs (-0.2%)

---

### 115×320×192  —  `k_proj, v_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (128,192,512) | (128,192,512) | (128,192,512) |
| Tile | (64,64,64) | (32,64,64) | (32,64,64) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 182.4 | 182.1 | 180.3 |
| Trial 2 avg (µs) | 183.1 | 178.9 | 178.6 |
| Trial 3 avg (µs) | 182.5 | 179.0 | 178.9 |
| **Mean avg (µs)** | **182.7** | **180.0** | **179.3** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-8.6µs (-4.0%)   Δ(AG vs B0)=-9.5µs (-4.4%)   Δ(AG vs B1)=-0.8µs (-0.4%)

---

### 115×320×288  —  `k_proj, v_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (128,288,512) | (128,288,512) | (128,288,512) |
| Tile | (32,32,32) | (64,32,64) | (64,32,64) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 206.1 | 188.1 | 195.5 |
| Trial 2 avg (µs) | 207.2 | 195.1 | 192.3 |
| Trial 3 avg (µs) | 207.2 | 194.0 | 194.4 |
| **Mean avg (µs)** | **206.8** | **192.4** | **194.1** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-15.8µs (-6.7%)   Δ(AG vs B0)=-13.4µs (-5.7%)   Δ(AG vs B1)=+2.4µs (+1.1%)

---

### 115×320×384  —  `k_proj, v_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (128,384,512) | (128,384,512) | (128,384,512) |
| Tile | (64,64,64) | (32,32,128) | (32,32,128) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 184.7 | 191.3 | 191.1 |
| Trial 2 avg (µs) | 184.0 | 195.2 | 191.3 |
| Trial 3 avg (µs) | 179.1 | 189.6 | 190.5 |
| **Mean avg (µs)** | **182.6** | **192.0** | **191.0** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: BASELINE 0**
Δ(B2 vs B0)=+8.8µs (+4.1%)   Δ(AG vs B0)=+11.8µs (+5.5%)   Δ(AG vs B1)=+3.0µs (+1.3%)

---

### 115×320×480  —  `k_proj, v_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (128,480,512) | (128,480,512) | (128,480,512) |
| Tile | (32,32,32) | (64,32,64) | (64,32,64) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 261.8 | 228.2 | 223.5 |
| Trial 2 avg (µs) | 306.4 | 227.6 | 229.2 |
| Trial 3 avg (µs) | 444.3 | 226.1 | 227.8 |
| **Mean avg (µs)** | **337.5** | **227.3** | **226.8** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-109.8µs (-29.6%)   Δ(AG vs B0)=-109.2µs (-29.4%)   Δ(AG vs B1)=+0.6µs (+0.2%)

---

### 115×320×576  —  `k_proj, v_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (128,576,512) | (128,576,512) | (128,576,512) |
| Tile | (64,64,64) | (32,64,64) | (32,64,64) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 267.4 | 307.2 | 284.1 |
| Trial 2 avg (µs) | 259.8 | 281.2 | 281.1 |
| Trial 3 avg (µs) | 263.0 | 280.1 | 285.4 |
| **Mean avg (µs)** | **263.4** | **289.5** | **283.5** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: BASELINE 0**
Δ(B2 vs B0)=+27.5µs (+9.1%)   Δ(AG vs B0)=+23.0µs (+7.7%)   Δ(AG vs B1)=-4.5µs (-1.4%)

---

### 115×320×672  —  `k_proj, v_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (128,672,512) | (128,672,512) | (128,672,512) |
| Tile | (32,32,32) | (64,32,64) | (64,32,64) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 308.1 | 291.1 | 326.4 |
| Trial 2 avg (µs) | 315.7 | 342.2 | 296.7 |
| Trial 3 avg (µs) | 312.9 | 294.7 | 339.8 |
| **Mean avg (µs)** | **312.2** | **309.3** | **321.0** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-2.1µs (-0.6%)   Δ(AG vs B0)=+13.6µs (+3.9%)   Δ(AG vs B1)=+15.7µs (+4.5%)

---

### 115×320×720  —  `k_proj, v_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (128,768,512) | (128,768,512) | (128,768,512) |
| Tile | (64,64,64) | (64,64,64) | (64,64,64) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 221.7 | 225.9 | 222.7 |
| Trial 2 avg (µs) | 217.6 | 250.8 | 220.4 |
| Trial 3 avg (µs) | 438.5 | 226.9 | 221.7 |
| **Mean avg (µs)** | **292.6** | **234.6** | **221.6** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: AGENTIC**
Δ(B2 vs B0)=-62.8µs (-18.5%)   Δ(AG vs B0)=-74.2µs (-21.8%)   Δ(AG vs B1)=-11.4µs (-4.1%)

---

### 115×960×320  —  `k_proj, v_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (128,320,960) | (128,320,960) | (128,320,960) |
| Tile | (64,64,64) | (64,64,64) | (64,64,64) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 282.4 | 283.3 | 282.3 |
| Trial 2 avg (µs) | 280.9 | 284.3 | 283.4 |
| Trial 3 avg (µs) | 285.8 | 283.6 | 283.2 |
| **Mean avg (µs)** | **283.0** | **283.7** | **282.9** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: AGENTIC**
Δ(B2 vs B0)=-3.5µs (-0.6%)   Δ(AG vs B0)=-94.2µs (-16.4%)   Δ(AG vs B1)=-90.6µs (-15.9%)

---

### 115×960×960  —  `q_proj, o_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (128,960,960) | (128,960,960) | (128,960,960) |
| Tile | (64,64,64) | (64,64,64) | (64,32,64) |
| Tile source | fixed | profiling | heuristic |
| Pad impl | numpy_pad | numpy_pad | numpy_pad |
| Trial 1 avg (µs) | 500.9 | 503.3 | 552.8 |
| Trial 2 avg (µs) | 530.2 | 497.1 | FAIL |
| Trial 3 avg (µs) | 495.6 | 494.2 | FAIL |
| **Mean avg (µs)** | **508.9** | **503.3** | **552.8** |
| Pass count | 3/3 | 1/3 | 1/3 |

**Winner: TIE**
Δ(B2 vs B0)=-7.8µs (-0.5%)   Δ(AG vs B0)=+422.5µs (+24.7%)   Δ(AG vs B1)=+430.4µs (+25.3%)

---

### 115×960×2560  —  `gate_proj, up_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (128,2560,960) | (128,2560,960) | (128,2560,960) |
| Tile | (64,64,64) | (64,64,64) | (64,64,64) |
| Tile source | fixed | profiling | heuristic |
| Pad impl | numpy_pad | numpy_pad | numpy_pad |
| Trial 1 avg (µs) | 656.2 | 662.4 | 669.1 |
| Trial 2 avg (µs) | 882.9 | 827.4 | 665.4 |
| Trial 3 avg (µs) | 818.9 | 653.6 | 665.1 |
| **Mean avg (µs)** | **769.6** | **714.5** | **666.5** |
| Pass count | 2/3 | 3/3 | 3/3 |

**Winner: BASELINE 0**
Δ(B2 vs B0)=+60.2µs (+2.4%)   Δ(AG vs B0)=+308.2µs (+12.2%)   Δ(AG vs B1)=+248.0µs (+9.6%)

---

### 115×2560×960  —  `down_proj`

**Pre-screened skip**: bf16_K_limit

---

### 128×320×96  —  `k_proj, v_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (128,96,512) | (128,96,512) | (128,96,512) |
| Tile | (32,32,32) | (64,32,64) | (64,32,64) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 146.1 | 147.7 | 146.9 |
| Trial 2 avg (µs) | 153.9 | 149.1 | 147.4 |
| Trial 3 avg (µs) | 146.0 | 146.3 | 145.8 |
| **Mean avg (µs)** | **148.7** | **147.7** | **146.7** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: BASELINE 2**
Δ(B2 vs B0)=-8.1µs (-4.5%)   Δ(AG vs B0)=-1.8µs (-1.0%)   Δ(AG vs B1)=+6.3µs (+3.7%)

---

### 128×320×192  —  `k_proj, v_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (128,192,512) | (128,192,512) | (128,192,512) |
| Tile | (64,64,64) | (32,64,64) | (32,64,64) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 184.9 | 174.8 | 172.4 |
| Trial 2 avg (µs) | 176.8 | 174.4 | 174.5 |
| Trial 3 avg (µs) | 178.2 | 173.1 | 174.6 |
| **Mean avg (µs)** | **179.9** | **174.1** | **173.8** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: BASELINE 2**
Δ(B2 vs B0)=-5.8µs (-2.8%)   Δ(AG vs B0)=+2.4µs (+1.1%)   Δ(AG vs B1)=+8.2µs (+4.0%)

---

### 128×320×288  —  `k_proj, v_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (128,288,512) | (128,288,512) | (128,288,512) |
| Tile | (32,32,32) | (64,32,64) | (64,32,64) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 206.7 | 190.3 | 190.5 |
| Trial 2 avg (µs) | 199.7 | 191.6 | 192.7 |
| Trial 3 avg (µs) | 205.2 | 188.0 | 187.9 |
| **Mean avg (µs)** | **203.8** | **190.0** | **190.4** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-1.9µs (-0.8%)   Δ(AG vs B0)=+2.0µs (+0.9%)   Δ(AG vs B1)=+4.0µs (+1.7%)

---

### 128×320×384  —  `k_proj, v_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (128,384,512) | (128,384,512) | (128,384,512) |
| Tile | (64,64,64) | (32,32,128) | (32,32,128) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 179.3 | 185.1 | 186.1 |
| Trial 2 avg (µs) | 180.5 | 187.1 | 189.7 |
| Trial 3 avg (µs) | 177.2 | 185.3 | 187.1 |
| **Mean avg (µs)** | **179.0** | **185.8** | **187.6** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: BASELINE 0**
Δ(B2 vs B0)=+11.2µs (+5.3%)   Δ(AG vs B0)=+29.1µs (+13.8%)   Δ(AG vs B1)=+17.9µs (+8.1%)

---

### 128×320×480  —  `k_proj, v_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (128,480,512) | (128,480,512) | (128,480,512) |
| Tile | (32,32,32) | (64,32,64) | (64,32,64) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 265.3 | 224.3 | 232.2 |
| Trial 2 avg (µs) | 262.7 | 225.8 | 235.5 |
| Trial 3 avg (µs) | 259.8 | 241.7 | 225.9 |
| **Mean avg (µs)** | **262.6** | **230.6** | **231.2** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: AGENTIC**
Δ(B2 vs B0)=-11.4µs (-3.7%)   Δ(AG vs B0)=-26.0µs (-8.4%)   Δ(AG vs B1)=-14.6µs (-4.9%)

---

### 128×320×576  —  `k_proj, v_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (128,576,512) | (128,576,512) | (128,576,512) |
| Tile | (64,64,64) | (32,64,64) | (32,64,64) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 260.8 | 423.8 | 427.8 |
| Trial 2 avg (µs) | 260.6 | 414.8 | 399.9 |
| Trial 3 avg (µs) | 364.2 | 410.4 | 416.6 |
| **Mean avg (µs)** | **295.2** | **416.4** | **414.8** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: BASELINE 0**
Δ(B2 vs B0)=+99.6µs (+26.9%)   Δ(AG vs B0)=+105.2µs (+28.4%)   Δ(AG vs B1)=+5.6µs (+1.2%)

---

### 128×320×672  —  `k_proj, v_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (128,672,512) | (128,672,512) | (128,672,512) |
| Tile | (32,32,32) | (64,32,64) | (64,32,64) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 442.0 | 423.1 | 435.7 |
| Trial 2 avg (µs) | 458.2 | 433.5 | 435.1 |
| Trial 3 avg (µs) | 452.4 | 436.0 | 437.1 |
| **Mean avg (µs)** | **450.9** | **430.9** | **436.0** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-42.6µs (-8.2%)   Δ(AG vs B0)=-35.5µs (-6.9%)   Δ(AG vs B1)=+7.1µs (+1.5%)

---

### 128×320×720  —  `k_proj, v_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (128,768,512) | (128,768,512) | (128,768,512) |
| Tile | (64,64,64) | (64,64,64) | (64,64,64) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 304.6 | 309.7 | 293.2 |
| Trial 2 avg (µs) | 314.6 | 295.9 | 323.6 |
| Trial 3 avg (µs) | 306.7 | 299.9 | 296.4 |
| **Mean avg (µs)** | **308.6** | **301.8** | **304.4** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: BASELINE 0**
Δ(B2 vs B0)=+18.0µs (+5.0%)   Δ(AG vs B0)=+31.5µs (+8.8%)   Δ(AG vs B1)=+13.4µs (+3.6%)

---

### 128×960×320  —  `k_proj, v_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (128,320,960) | (128,320,960) | (128,320,960) |
| Tile | (64,64,64) | (64,64,64) | (64,64,64) |
| Tile source | fixed | profiling | profiling |
| Pad impl | none | none | none |
| Trial 1 avg (µs) | 395.5 | 411.4 | 286.0 |
| Trial 2 avg (µs) | 403.7 | 417.5 | 285.6 |
| Trial 3 avg (µs) | 396.8 | 289.7 | 284.5 |
| **Mean avg (µs)** | **398.6** | **372.9** | **285.4** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: BASELINE 0**
Δ(B2 vs B0)=+nanµs (+nan%)   Δ(AG vs B0)=+nanµs (+nan%)   Δ(AG vs B1)=+nanµs (+nan%)

---

### 128×960×960  —  `q_proj, o_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (128,960,960) | (128,960,960) | (128,960,960) |
| Tile | (64,64,64) | (64,64,64) | (64,32,64) |
| Tile source | fixed | profiling | heuristic |
| Pad impl | none | none | none |
| Trial 1 avg (µs) | 499.1 | 495.3 | FAIL |
| Trial 2 avg (µs) | 500.7 | 499.6 | FAIL |
| Trial 3 avg (µs) | 497.8 | 494.9 | 553.2 |
| **Mean avg (µs)** | **499.2** | **496.6** | **553.2** |
| Pass count | 3/3 | 3/3 | 1/3 |

**Winner: BASELINE 0**
Δ(B2 vs B0)=+nanµs (+nan%)   Δ(AG vs B0)=+nanµs (+nan%)   Δ(AG vs B1)=+nanµs (+nan%)

---

### 128×960×2560  —  `gate_proj, up_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (128,2560,960) | (128,2560,960) | (128,2560,960) |
| Tile | (64,64,64) | (64,64,64) | (64,64,64) |
| Tile source | fixed | profiling | heuristic |
| Pad impl | none | none | none |
| Trial 1 avg (µs) | 662.4 | 756.8 | 658.3 |
| Trial 2 avg (µs) | 694.3 | 736.2 | 658.4 |
| Trial 3 avg (µs) | 656.9 | 750.3 | 659.0 |
| **Mean avg (µs)** | **678.4** | **753.5** | **658.3** |
| Pass count | 2/3 | 2/3 | 1/3 |

**Winner: BASELINE 0**
Δ(B2 vs B0)=+nanµs (+nan%)   Δ(AG vs B0)=+nanµs (+nan%)   Δ(AG vs B1)=+nanµs (+nan%)

---

### 128×2560×960  —  `down_proj`

**Pre-screened skip**: bf16_K_limit

---

### 179×320×96  —  `k_proj, v_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (192,96,512) | (192,96,512) | (192,96,512) |
| Tile | (32,32,32) | (64,32,64) | (64,32,64) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 169.1 | 182.5 | 182.6 |
| Trial 2 avg (µs) | 169.2 | 188.1 | 182.7 |
| Trial 3 avg (µs) | 169.4 | 179.5 | 179.7 |
| **Mean avg (µs)** | **169.2** | **183.4** | **181.7** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: BASELINE 0**
Δ(B2 vs B0)=+32.4µs (+16.5%)   Δ(AG vs B0)=+36.3µs (+18.4%)   Δ(AG vs B1)=+3.9µs (+1.7%)

---

### 179×320×192  —  `k_proj, v_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (192,192,512) | (192,192,512) | (192,192,512) |
| Tile | (64,64,64) | (64,32,64) | (64,32,64) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 200.5 | 219.3 | 318.6 |
| Trial 2 avg (µs) | 178.3 | 216.8 | 318.1 |
| Trial 3 avg (µs) | 179.8 | 219.5 | 327.2 |
| **Mean avg (µs)** | **186.2** | **218.5** | **321.3** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: BASELINE 0**
Δ(B2 vs B0)=+27.0µs (+12.0%)   Δ(AG vs B0)=+133.6µs (+59.4%)   Δ(AG vs B1)=+106.6µs (+42.3%)

---

### 179×320×288  —  `k_proj, v_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (192,288,512) | (192,288,512) | (192,288,512) |
| Tile | (32,32,32) | (64,32,64) | (64,32,64) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 392.3 | 387.1 | 394.7 |
| Trial 2 avg (µs) | 396.7 | 391.9 | 390.7 |
| Trial 3 avg (µs) | 401.5 | 393.1 | 392.0 |
| **Mean avg (µs)** | **396.8** | **390.7** | **392.5** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=+9.6µs (+2.2%)   Δ(AG vs B0)=-3.3µs (-0.8%)   Δ(AG vs B1)=-12.9µs (-2.9%)

---

### 179×320×384  —  `k_proj, v_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (192,384,512) | (192,384,512) | (192,384,512) |
| Tile | (64,64,64) | (64,64,64) | (64,64,64) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 335.3 | 334.2 | 334.5 |
| Trial 2 avg (µs) | 334.4 | 336.5 | 335.8 |
| Trial 3 avg (µs) | 334.9 | 323.5 | 324.3 |
| **Mean avg (µs)** | **334.9** | **331.4** | **331.6** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-1.3µs (-0.3%)   Δ(AG vs B0)=-4.3µs (-1.2%)   Δ(AG vs B1)=-3.0µs (-0.8%)

---

### 179×320×480  —  `k_proj, v_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (192,480,512) | (192,480,512) | (192,480,512) |
| Tile | (16,32,32) | (64,32,64) | (64,32,64) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 512.4 | 525.2 | 529.0 |
| Trial 2 avg (µs) | 503.9 | 524.9 | 524.0 |
| Trial 3 avg (µs) | 740.0 | 513.7 | 526.3 |
| **Mean avg (µs)** | **585.4** | **521.3** | **526.5** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-62.3µs (-9.9%)   Δ(AG vs B0)=-59.5µs (-9.5%)   Δ(AG vs B1)=+2.8µs (+0.5%)

---

### 179×320×576  —  `k_proj, v_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (192,576,512) | (192,576,512) | (192,576,512) |
| Tile | (64,64,64) | (64,64,64) | (64,64,64) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 393.0 | 394.3 | 381.1 |
| Trial 2 avg (µs) | 392.2 | 395.1 | 391.0 |
| Trial 3 avg (µs) | 389.6 | 394.8 | 387.6 |
| **Mean avg (µs)** | **391.6** | **394.7** | **386.6** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: AGENTIC**
Δ(B2 vs B0)=+58.2µs (+12.1%)   Δ(AG vs B0)=-15.7µs (-3.3%)   Δ(AG vs B1)=-73.9µs (-13.7%)

---

### 179×320×672  —  `k_proj, v_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (192,672,512) | (192,672,512) | (192,672,512) |
| Tile | (16,32,32) | (16,32,32) | (16,32,32) |
| Tile source | fixed | fallback_fixed | heuristic |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 689.0 | 686.0 | 681.3 |
| Trial 2 avg (µs) | 679.6 | 685.2 | 683.5 |
| Trial 3 avg (µs) | 684.4 | 680.8 | 680.8 |
| **Mean avg (µs)** | **684.3** | **684.0** | **681.8** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: BASELINE 2**
Δ(B2 vs B0)=-50.1µs (-6.1%)   Δ(AG vs B0)=+57.4µs (+7.0%)   Δ(AG vs B1)=+107.5µs (+13.9%)

---

### 179×320×720  —  `k_proj, v_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (192,768,512) | (192,768,512) | (192,768,512) |
| Tile | (64,64,64) | (64,64,64) | (64,64,64) |
| Tile source | fixed | fallback_fixed | heuristic |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 297.7 | 296.9 | 297.1 |
| Trial 2 avg (µs) | 295.9 | 294.9 | 304.4 |
| Trial 3 avg (µs) | 301.1 | 295.7 | 296.3 |
| **Mean avg (µs)** | **298.2** | **295.8** | **299.3** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: BASELINE 2**
Δ(B2 vs B0)=-52.5µs (-11.0%)   Δ(AG vs B0)=-34.5µs (-7.3%)   Δ(AG vs B1)=+17.9µs (+4.2%)

---

### 179×960×320  —  `k_proj, v_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (192,320,960) | (192,320,960) | (192,320,960) |
| Tile | (64,64,64) | (64,64,64) | (64,16,64) |
| Tile source | fixed | fallback_fixed | heuristic |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 300.2 | 298.6 | 614.3 |
| Trial 2 avg (µs) | 296.5 | 297.6 | 616.3 |
| Trial 3 avg (µs) | 296.9 | 298.2 | 614.5 |
| **Mean avg (µs)** | **297.9** | **298.1** | **615.0** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: BASELINE 2**
Δ(B2 vs B0)=-90.2µs (-13.0%)   Δ(AG vs B0)=+543.9µs (+78.6%)   Δ(AG vs B1)=+634.1µs (+105.4%)

---

### 179×960×960  —  `q_proj, o_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (192,960,960) | (192,960,960) | (192,960,960) |
| Tile | (64,64,64) | (64,64,64) | (64,64,64) |
| Tile source | fixed | fallback_fixed | heuristic |
| Pad impl | numpy_pad | numpy_pad | numpy_pad |
| Trial 1 avg (µs) | 621.9 | 625.2 | 624.2 |
| Trial 2 avg (µs) | 622.0 | 623.9 | 622.7 |
| Trial 3 avg (µs) | 619.3 | 623.9 | 622.2 |
| **Mean avg (µs)** | **621.1** | **624.5** | **623.0** |
| Pass count | 3/3 | 2/3 | 3/3 |

**Winner: BASELINE 0**
Δ(B2 vs B0)=+63.6µs (+2.7%)   Δ(AG vs B0)=+447.0µs (+19.2%)   Δ(AG vs B1)=+383.4µs (+16.1%)

---

### 179×960×2560  —  `gate_proj, up_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (192,2560,960) | (192,2560,960) | (192,2560,960) |
| Tile | (16,32,32) | (64,64,64) | (64,64,64) |
| Tile source | fixed | fallback_fixed | heuristic |
| Pad impl | numpy_pad | numpy_pad | numpy_pad |
| Trial 1 avg (µs) | 2698.8 | 1302.3 | 1301.6 |
| Trial 2 avg (µs) | 2687.1 | 1300.0 | 1307.5 |
| Trial 3 avg (µs) | 2701.6 | 1299.3 | 1299.7 |
| **Mean avg (µs)** | **FAIL** | **1300.5** | **1303.6** |
| Pass count | 0/3 | 3/3 | 2/3 |

**Winner: INCONCLUSIVE**
Δ(AG vs B1)=-82.5µs (-2.0%)

---

### 179×2560×960  —  `down_proj`

**Pre-screened skip**: bf16_K_limit

---

### 235×320×96  —  `k_proj, v_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (256,96,512) | (256,96,512) | (256,96,512) |
| Tile | (32,32,32) | (64,32,128) | (64,32,128) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 189.0 | 169.0 | 172.2 |
| Trial 2 avg (µs) | 187.9 | 170.0 | 168.7 |
| Trial 3 avg (µs) | 184.9 | 168.5 | 165.8 |
| **Mean avg (µs)** | **187.3** | **169.2** | **168.9** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: AGENTIC**
Δ(B2 vs B0)=+1.0µs (+0.4%)   Δ(AG vs B0)=-12.5µs (-5.2%)   Δ(AG vs B1)=-13.5µs (-5.6%)

---

### 235×320×192  —  `k_proj, v_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (256,192,512) | (256,192,512) | (256,192,512) |
| Tile | (64,64,64) | (64,64,64) | (64,64,64) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 192.6 | 196.8 | 196.2 |
| Trial 2 avg (µs) | 197.6 | 197.4 | 199.6 |
| Trial 3 avg (µs) | 198.2 | 197.2 | 197.8 |
| **Mean avg (µs)** | **196.1** | **197.1** | **197.8** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: AGENTIC**
Δ(B2 vs B0)=+11.9µs (+4.6%)   Δ(AG vs B0)=-12.9µs (-5.0%)   Δ(AG vs B1)=-24.8µs (-9.3%)

---

### 235×320×288  —  `k_proj, v_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (256,288,512) | (256,288,512) | (256,288,512) |
| Tile | (32,32,32) | (64,32,128) | (64,32,128) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 302.0 | 233.8 | 236.0 |
| Trial 2 avg (µs) | 300.7 | 238.0 | 239.0 |
| Trial 3 avg (µs) | 302.8 | 234.3 | 236.2 |
| **Mean avg (µs)** | **301.8** | **235.4** | **237.1** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: AGENTIC**
Δ(B2 vs B0)=-16.6µs (-4.5%)   Δ(AG vs B0)=-70.4µs (-18.9%)   Δ(AG vs B1)=-53.7µs (-15.1%)

---

### 235×320×384  —  `k_proj, v_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (256,384,512) | (256,384,512) | (256,384,512) |
| Tile | (64,64,64) | (64,128,16) | (64,128,16) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 254.8 | 252.8 | 242.4 |
| Trial 2 avg (µs) | 253.1 | 252.8 | 247.1 |
| Trial 3 avg (µs) | 248.8 | 252.9 | 247.9 |
| **Mean avg (µs)** | **252.2** | **252.8** | **245.8** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: AGENTIC**
Δ(B2 vs B0)=-56.7µs (-13.8%)   Δ(AG vs B0)=-131.0µs (-31.9%)   Δ(AG vs B1)=-74.3µs (-21.0%)

---

### 235×320×480  —  `k_proj, v_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (256,480,512) | (256,480,512) | (256,480,512) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 383.0 | 384.0 | 385.5 |
| Trial 2 avg (µs) | 383.8 | 384.6 | 382.5 |
| Trial 3 avg (µs) | 414.2 | 385.8 | 381.0 |
| **Mean avg (µs)** | **393.7** | **384.8** | **383.0** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-12.9µs (-3.0%)   Δ(AG vs B0)=-13.1µs (-3.0%)   Δ(AG vs B1)=-0.3µs (-0.1%)

---

### 235×320×576  —  `k_proj, v_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (256,576,512) | (256,576,512) | (256,576,512) |
| Tile | (64,64,64) | (64,32,128) | (64,32,128) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 302.5 | 323.0 | 318.3 |
| Trial 2 avg (µs) | 300.7 | 317.8 | 353.3 |
| Trial 3 avg (µs) | 300.4 | 316.9 | 332.8 |
| **Mean avg (µs)** | **301.2** | **319.2** | **334.8** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: BASELINE 0**
Δ(B2 vs B0)=+16.8µs (+5.0%)   Δ(AG vs B0)=+33.7µs (+9.9%)   Δ(AG vs B1)=+16.9µs (+4.8%)

---

### 235×320×672  —  `k_proj, v_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (256,672,512) | (256,672,512) | (256,672,512) |
| Tile | (32,32,32) | (64,32,128) | (64,32,128) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 479.9 | 348.3 | 349.1 |
| Trial 2 avg (µs) | 483.4 | 351.5 | 350.3 |
| Trial 3 avg (µs) | 570.6 | 346.8 | 375.3 |
| **Mean avg (µs)** | **511.3** | **348.9** | **358.2** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: BASELINE 2**
Δ(B2 vs B0)=-157.1µs (-28.4%)   Δ(AG vs B0)=-147.8µs (-26.8%)   Δ(AG vs B1)=+9.3µs (+2.3%)

---

### 235×320×720  —  `k_proj, v_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (256,768,512) | (256,768,512) | (256,768,512) |
| Tile | (64,64,64) | (64,64,64) | (64,64,64) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 249.1 | 251.1 | 252.3 |
| Trial 2 avg (µs) | 252.0 | 257.9 | 322.1 |
| Trial 3 avg (µs) | 248.7 | 255.6 | 249.8 |
| **Mean avg (µs)** | **249.9** | **254.8** | **274.7** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=+4.7µs (+1.6%)   Δ(AG vs B0)=+26.4µs (+9.0%)   Δ(AG vs B1)=+21.7µs (+7.3%)

---

### 235×960×320  —  `k_proj, v_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (256,320,960) | (256,320,960) | (256,384,960) |
| Tile | (16,32,32) | (16,32,32) | (64,64,64) |
| Tile source | fixed | fallback_fixed | heuristic |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | FAIL | FAIL | 382.8 |
| Trial 2 avg (µs) | FAIL | FAIL | 329.0 |
| Trial 3 avg (µs) | FAIL | FAIL | 326.4 |
| **Mean avg (µs)** | **FAIL** | **FAIL** | **346.1** |
| Pass count | 0/3 | 0/3 | 3/3 |

**Winner: INCONCLUSIVE**

---

### 235×960×960  —  `q_proj, o_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (256,960,960) | (256,960,960) | (256,960,960) |
| Tile | (64,64,64) | (64,64,64) | (64,64,64) |
| Tile source | fixed | profiling | profiling |
| Pad impl | numpy_pad | numpy_pad | numpy_pad |
| Trial 1 avg (µs) | 623.0 | 615.3 | 621.0 |
| Trial 2 avg (µs) | 743.5 | 614.7 | 653.0 |
| Trial 3 avg (µs) | 621.0 | 616.5 | 678.5 |
| **Mean avg (µs)** | **682.2** | **615.0** | **649.7** |
| Pass count | 2/3 | 2/3 | 2/3 |

**Winner: BASELINE 2**
Δ(B2 vs B0)=-78.0µs (-4.2%)   Δ(AG vs B0)=-20.1µs (-1.1%)   Δ(AG vs B1)=+57.9µs (+3.2%)

---

### 235×960×2560  —  `gate_proj, up_proj`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (256,2560,960) | (256,2560,960) | (256,2560,960) |
| Tile | (64,64,64) | (64,64,64) | (64,64,64) |
| Tile source | fixed | profiling | profiling |
| Pad impl | numpy_pad | numpy_pad | numpy_pad |
| Trial 1 avg (µs) | 919.8 | 831.3 | 805.3 |
| Trial 2 avg (µs) | 799.7 | 801.8 | 824.6 |
| Trial 3 avg (µs) | 798.4 | 842.3 | 887.1 |
| **Mean avg (µs)** | **859.8** | **825.1** | **805.3** |
| Pass count | 2/3 | 3/3 | 1/3 |

**Winner: TIE**
Δ(B2 vs B0)=+31.8µs (+1.2%)   Δ(AG vs B0)=+136.4µs (+5.2%)   Δ(AG vs B1)=+104.7µs (+3.9%)

---

### 235×2560×960  —  `down_proj`

**Pre-screened skip**: bf16_K_limit

---

### 1024×768×768  —  `q_proj, k_proj, v_proj, out_proj, patch_embedding`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (1024,768,768) | (1024,768,768) | (1024,768,768) |
| Tile | (64,64,64) | (64,64,64) | (64,64,64) |
| Tile source | fixed | profiling | profiling |
| Pad impl | none | none | none |
| Trial 1 avg (µs) | 757.0 | 759.1 | 755.2 |
| Trial 2 avg (µs) | 755.3 | 752.5 | 797.6 |
| Trial 3 avg (µs) | 752.3 | 756.2 | 772.5 |
| **Mean avg (µs)** | **754.9** | **755.9** | **775.1** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: BASELINE 0**
Δ(B2 vs B0)=+nanµs (+nan%)   Δ(AG vs B0)=+nanµs (+nan%)   Δ(AG vs B1)=+nanµs (+nan%)

---

### 1024×768×3072  —  `mlp.fc1`

| | Baseline 0 | Baseline 1 | Agentic |
|-|------------|------------|---------|
| Padded shape | (1024,3072,768) | (1024,3072,768) | (1024,3072,768) |
| Tile | (64,64,64) | (64,64,64) | (64,64,64) |
| Tile source | fixed | profiling | profiling |
| Pad impl | none | none | none |
| Trial 1 avg (µs) | 3033.1 | 3030.0 | 3041.3 |
| Trial 2 avg (µs) | 3023.9 | 3037.7 | 3049.4 |
| Trial 3 avg (µs) | 3022.1 | 3046.8 | 3043.3 |
| **Mean avg (µs)** | **3026.4** | **3038.2** | **3044.7** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: BASELINE 0**
Δ(B2 vs B0)=+nanµs (+nan%)   Δ(AG vs B0)=+nanµs (+nan%)   Δ(AG vs B1)=+nanµs (+nan%)

---

### 1024×3072×768  —  `mlp.fc2`

**Pre-screened skip**: bf16_K_limit

---

## Memory Updates

- `npu_execution_profiling.json`: updated with agentic trial results
- `strategy_insights.md`: updated with new insights (if any)
- `error-logs/errors.md`: updated if new error patterns found
