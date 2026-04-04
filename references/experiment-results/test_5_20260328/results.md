# SmolVLA NPU Experiment Results
**Date**: 2026-03-28  |  **dtype**: bf16  |  **Trials per group**: 5

## Summary

| Metric | Value |
|--------|-------|
| Shapes in file | 291 |
| Shapes run | 282 |
| Pre-screened (skip) | 12 |
| Baseline 0 wins | 38 |
| Baseline 2 wins | 7 |
| Agentic wins | 18 |
| Ties (within 2%) | 163 |
| Inconclusive | 56 |
| Mean Δ agentic vs Baseline 0 | 2.05% |
| Mean Δ agentic vs Baseline 2 | 0.45% |
| Mean Δ Baseline 2 vs Baseline 0 | 1.12% |

## Per-Shape Results

### 1024×768×768  —  `Vision Encoder/q_proj, Vision Encoder/k_proj, Vision Encoder`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (?,?,?) | (?,?,?) | (?,?,?) |
| Tile | (?,?,?) | (?,?,?) | (?,?,?) |
| Tile source | ? | ? | ? |
| Pad impl | ? | ? | ? |
| Trial 1 avg (µs) | — | — | — |
| Trial 2 avg (µs) | — | — | — |
| Trial 3 avg (µs) | — | — | — |
| **Mean avg (µs)** | **FAIL** | **FAIL** | **FAIL** |
| Pass count | 0/3 | 0/3 | 0/3 |

**Winner: TIE**
Δ(B2 vs B0)=+13.4µs (+1.8%)   Δ(AG vs B0)=+1.6µs (+0.2%)   Δ(AG vs B2)=-11.8µs (-1.5%)

---

### 1024×768×3072  —  `Vision Encoder/mlp.fc1`

**Pre-screened skip**: 

---

### 1024×3072×768  —  `Vision Encoder/mlp.fc2`

**Pre-screened skip**: 

---

### 128×960×960  —  `Text Layers/q_proj, Text Layers/o_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (?,?,?) | (?,?,?) | (?,?,?) |
| Tile | (?,?,?) | (?,?,?) | (?,?,?) |
| Tile source | ? | ? | ? |
| Pad impl | ? | ? | ? |
| Trial 1 avg (µs) | — | — | — |
| Trial 2 avg (µs) | — | — | — |
| Trial 3 avg (µs) | — | — | — |
| **Mean avg (µs)** | **FAIL** | **FAIL** | **FAIL** |
| Pass count | 0/3 | 0/3 | 0/3 |

**Winner: BASELINE 2**
Δ(B2 vs B0)=-113.3µs (-18.1%)   Δ(AG vs B0)=+1.7µs (+0.3%)   Δ(AG vs B2)=+115.0µs (+22.5%)

---

### 128×960×320  —  `Text Layers/k_proj, Text Layers/v_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (?,?,?) | (?,?,?) | (?,?,?) |
| Tile | (?,?,?) | (?,?,?) | (?,?,?) |
| Tile source | ? | ? | ? |
| Pad impl | ? | ? | ? |
| Trial 1 avg (µs) | — | — | — |
| Trial 2 avg (µs) | — | — | — |
| Trial 3 avg (µs) | — | — | — |
| **Mean avg (µs)** | **FAIL** | **FAIL** | **FAIL** |
| Pass count | 0/3 | 0/3 | 0/3 |

**Winner: TIE**
Δ(B2 vs B0)=+3.5µs (+1.5%)   Δ(AG vs B0)=+4.2µs (+1.8%)   Δ(AG vs B2)=+0.7µs (+0.3%)

---

### 128×960×2560  —  `Text Layers/gate_proj, Text Layers/up_proj`

**Pre-screened skip**: 

---

### 128×2560×960  —  `Text Layers/down_proj`

**Pre-screened skip**: 

---

### 50×960×96  —  `Action Expert/q_proj`

**Pre-screened skip**: 

---

### 128×320×96  —  `Action Expert/k_proj, Action Expert/v_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (?,?,?) | (?,?,?) | (?,?,?) |
| Tile | (?,?,?) | (?,?,?) | (?,?,?) |
| Tile source | ? | ? | ? |
| Pad impl | ? | ? | ? |
| Trial 1 avg (µs) | — | — | — |
| Trial 2 avg (µs) | — | — | — |
| Trial 3 avg (µs) | — | — | — |
| **Mean avg (µs)** | **FAIL** | **FAIL** | **FAIL** |
| Pass count | 0/3 | 0/3 | 0/3 |

**Winner: TIE**
Δ(B2 vs B0)=+4.3µs (+2.9%)   Δ(AG vs B0)=-0.2µs (-0.1%)   Δ(AG vs B2)=-4.5µs (-3.0%)

---

### 50×96×960  —  `Action Expert/o_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (?,?,?) | (?,?,?) | (?,?,?) |
| Tile | (?,?,?) | (?,?,?) | (?,?,?) |
| Tile source | ? | ? | ? |
| Pad impl | ? | ? | ? |
| Trial 1 avg (µs) | — | — | — |
| Trial 2 avg (µs) | — | — | — |
| Trial 3 avg (µs) | — | — | — |
| **Mean avg (µs)** | **FAIL** | **FAIL** | **FAIL** |
| Pass count | 0/3 | 0/3 | 0/3 |

**Winner: TIE**
Δ(B2 vs B0)=-47.3µs (-19.0%)   Δ(AG vs B0)=-48.2µs (-19.3%)   Δ(AG vs B2)=-0.9µs (-0.4%)

---

### 50×96×256  —  `Action Expert/gate_proj, Action Expert/up_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (?,?,?) | (?,?,?) | (?,?,?) |
| Tile | (?,?,?) | (?,?,?) | (?,?,?) |
| Tile source | ? | ? | ? |
| Pad impl | ? | ? | ? |
| Trial 1 avg (µs) | — | — | — |
| Trial 2 avg (µs) | — | — | — |
| Trial 3 avg (µs) | — | — | — |
| **Mean avg (µs)** | **FAIL** | **FAIL** | **FAIL** |
| Pass count | 0/3 | 0/3 | 0/3 |

**Winner: TIE**
Δ(B2 vs B0)=-17.5µs (-11.5%)   Δ(AG vs B0)=-15.4µs (-10.2%)   Δ(AG vs B2)=+2.0µs (+1.5%)

---

### 50×256×96  —  `Action Expert/down_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (?,?,?) | (?,?,?) | (?,?,?) |
| Tile | (?,?,?) | (?,?,?) | (?,?,?) |
| Tile source | ? | ? | ? |
| Pad impl | ? | ? | ? |
| Trial 1 avg (µs) | — | — | — |
| Trial 2 avg (µs) | — | — | — |
| Trial 3 avg (µs) | — | — | — |
| **Mean avg (µs)** | **FAIL** | **FAIL** | **FAIL** |
| Pass count | 0/3 | 0/3 | 0/3 |

**Winner: TIE**
Δ(B2 vs B0)=-2.3µs (-1.7%)   Δ(AG vs B0)=+0.9µs (+0.7%)   Δ(AG vs B2)=+3.2µs (+2.5%)

---

### 1×960×32  —  `Projections/state_proj`

**Pre-screened skip**: 

---

### 50×32×96  —  `Projections/action_in_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (?,?,?) | (?,?,?) | (?,?,?) |
| Tile | (?,?,?) | (?,?,?) | (?,?,?) |
| Tile source | ? | ? | ? |
| Pad impl | ? | ? | ? |
| Trial 1 avg (µs) | — | — | — |
| Trial 2 avg (µs) | — | — | — |
| Trial 3 avg (µs) | — | — | — |
| **Mean avg (µs)** | **FAIL** | **FAIL** | **FAIL** |
| Pass count | 0/3 | 0/3 | 0/3 |

**Winner: TIE**
Δ(B2 vs B0)=-0.1µs (-0.1%)   Δ(AG vs B0)=-0.9µs (-0.7%)   Δ(AG vs B2)=-0.8µs (-0.6%)

---

### 50×96×32  —  `Projections/action_out_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (?,?,?) | (?,?,?) | (?,?,?) |
| Tile | (?,?,?) | (?,?,?) | (?,?,?) |
| Tile source | ? | ? | ? |
| Pad impl | ? | ? | ? |
| Trial 1 avg (µs) | — | — | — |
| Trial 2 avg (µs) | — | — | — |
| Trial 3 avg (µs) | — | — | — |
| **Mean avg (µs)** | **FAIL** | **FAIL** | **FAIL** |
| Pass count | 0/3 | 0/3 | 0/3 |

**Winner: TIE**
Δ(B2 vs B0)=+0.4µs (+0.3%)   Δ(AG vs B0)=-0.3µs (-0.2%)   Δ(AG vs B2)=-0.7µs (-0.5%)

---

### 1×192×96  —  `Projections/action_time_mlp_in`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (?,?,?) | (?,?,?) | (?,?,?) |
| Tile | (?,?,?) | (?,?,?) | (?,?,?) |
| Tile source | ? | ? | ? |
| Pad impl | ? | ? | ? |
| Trial 1 avg (µs) | — | — | — |
| Trial 2 avg (µs) | — | — | — |
| Trial 3 avg (µs) | — | — | — |
| **Mean avg (µs)** | **FAIL** | **FAIL** | **FAIL** |
| Pass count | 0/3 | 0/3 | 0/3 |

**Winner: TIE**
Δ(B2 vs B0)=-1.5µs (-1.2%)   Δ(AG vs B0)=+0.3µs (+0.2%)   Δ(AG vs B2)=+1.9µs (+1.4%)

---

### 1×96×96  —  `Projections/action_time_mlp_out`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (?,?,?) | (?,?,?) | (?,?,?) |
| Tile | (?,?,?) | (?,?,?) | (?,?,?) |
| Tile source | ? | ? | ? |
| Pad impl | ? | ? | ? |
| Trial 1 avg (µs) | — | — | — |
| Trial 2 avg (µs) | — | — | — |
| Trial 3 avg (µs) | — | — | — |
| **Mean avg (µs)** | **FAIL** | **FAIL** | **FAIL** |
| Pass count | 0/3 | 0/3 | 0/3 |

**Winner: TIE**
Δ(B2 vs B0)=+0.4µs (+0.3%)   Δ(AG vs B0)=+0.1µs (+0.1%)   Δ(AG vs B2)=-0.3µs (-0.2%)

---

### 20×32×96  —  `action_in_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,96,32) | (32,96,32) | (32,96,32) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | fallback_fixed | heuristic |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 129.7 | 129.8 | 130.5 |
| Trial 2 avg (µs) | 127.4 | 127.6 | 126.8 |
| Trial 3 avg (µs) | 129.8 | 132.1 | 128.7 |
| **Mean avg (µs)** | **128.9** | **129.8** | **128.7** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=+0.9µs (+0.7%)   Δ(AG vs B0)=-0.3µs (-0.2%)   Δ(AG vs B2)=-1.1µs (-0.9%)

---

### 20×32×192  —  `action_in_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,192,32) | (32,192,32) | (32,192,32) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | fallback_fixed | heuristic |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 124.0 | 124.7 | 128.5 |
| Trial 2 avg (µs) | 125.7 | 125.7 | 125.7 |
| Trial 3 avg (µs) | 125.4 | 124.6 | 126.2 |
| **Mean avg (µs)** | **125.0** | **125.0** | **126.8** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=+0.0µs (+0.0%)   Δ(AG vs B0)=+1.8µs (+1.4%)   Δ(AG vs B2)=+1.8µs (+1.4%)

---

### 20×32×288  —  `action_in_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,288,32) | (32,288,32) | (32,288,32) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | fallback_fixed | heuristic |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 129.9 | 131.7 | 126.9 |
| Trial 2 avg (µs) | 134.5 | 124.6 | 124.8 |
| Trial 3 avg (µs) | 125.5 | 126.3 | 131.2 |
| **Mean avg (µs)** | **129.9** | **127.5** | **127.7** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-2.4µs (-1.9%)   Δ(AG vs B0)=-2.3µs (-1.8%)   Δ(AG vs B2)=+0.1µs (+0.1%)

---

### 20×32×384  —  `action_in_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,384,32) | (32,384,32) | (32,384,32) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | fallback_fixed | heuristic |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 157.2 | 156.5 | 155.3 |
| Trial 2 avg (µs) | 155.2 | 155.9 | 158.3 |
| Trial 3 avg (µs) | 155.2 | 155.7 | 154.7 |
| **Mean avg (µs)** | **155.9** | **156.0** | **156.1** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=+0.2µs (+0.1%)   Δ(AG vs B0)=+0.2µs (+0.1%)   Δ(AG vs B2)=+0.1µs (+0.0%)

---

### 20×32×480  —  `action_in_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,480,32) | (32,480,32) | (32,480,32) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | fallback_fixed | heuristic |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 134.6 | 134.4 | 137.9 |
| Trial 2 avg (µs) | 133.5 | 136.7 | 136.0 |
| Trial 3 avg (µs) | 137.9 | 136.0 | 137.4 |
| **Mean avg (µs)** | **135.4** | **135.7** | **137.1** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=+0.3µs (+0.2%)   Δ(AG vs B0)=+1.7µs (+1.3%)   Δ(AG vs B2)=+1.4µs (+1.0%)

---

### 20×32×576  —  `action_in_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,576,32) | (32,576,32) | (32,576,32) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | fallback_fixed | heuristic |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 139.2 | 139.3 | 138.2 |
| Trial 2 avg (µs) | 143.0 | 139.4 | 138.8 |
| Trial 3 avg (µs) | 138.0 | 136.8 | 136.7 |
| **Mean avg (µs)** | **140.0** | **138.5** | **137.9** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-1.5µs (-1.1%)   Δ(AG vs B0)=-2.1µs (-1.5%)   Δ(AG vs B2)=-0.6µs (-0.5%)

---

### 20×32×672  —  `action_in_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,672,32) | (32,672,32) | (32,672,32) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | fallback_fixed | heuristic |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 146.0 | 148.3 | 148.7 |
| Trial 2 avg (µs) | 148.3 | 148.2 | 147.1 |
| Trial 3 avg (µs) | 147.1 | 150.9 | 147.3 |
| **Mean avg (µs)** | **147.1** | **149.1** | **147.7** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=+2.0µs (+1.4%)   Δ(AG vs B0)=+0.6µs (+0.4%)   Δ(AG vs B2)=-1.4µs (-1.0%)

---

### 20×32×720  —  `action_in_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,768,32) | (32,768,32) | (32,768,32) |
| Tile | (32,32,32) | (32,64,32) | (32,64,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 200.4 | 154.5 | 158.2 |
| Trial 2 avg (µs) | 192.5 | 158.0 | 156.7 |
| Trial 3 avg (µs) | 192.7 | 153.8 | 156.6 |
| **Mean avg (µs)** | **195.2** | **155.4** | **157.2** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-39.8µs (-20.4%)   Δ(AG vs B0)=-38.0µs (-19.5%)   Δ(AG vs B2)=+1.7µs (+1.1%)

---

### 20×96×32  —  `action_out_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,32,128) | (32,32,128) | (32,32,128) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 121.2 | 121.2 | 121.7 |
| Trial 2 avg (µs) | 121.6 | 120.6 | 122.1 |
| Trial 3 avg (µs) | 121.3 | 119.8 | 121.0 |
| **Mean avg (µs)** | **121.4** | **120.5** | **121.6** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-0.9µs (-0.7%)   Δ(AG vs B0)=+0.2µs (+0.2%)   Δ(AG vs B2)=+1.1µs (+0.9%)

---

### 20×96×256  —  `gate_proj, up_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,256,128) | (32,256,128) | (32,256,128) |
| Tile | (32,32,32) | (32,64,32) | (32,64,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 141.5 | 131.1 | 130.8 |
| Trial 2 avg (µs) | 140.8 | 131.9 | 135.1 |
| Trial 3 avg (µs) | 142.3 | 130.5 | 132.1 |
| **Mean avg (µs)** | **141.5** | **131.2** | **132.7** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-10.3µs (-7.3%)   Δ(AG vs B0)=-8.9µs (-6.3%)   Δ(AG vs B2)=+1.5µs (+1.1%)

---

### 20×96×960  —  `o_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,960,128) | (32,960,128) | (32,960,128) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | fallback_fixed | heuristic |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 187.5 | 186.0 | 184.3 |
| Trial 2 avg (µs) | 187.8 | 183.5 | 184.9 |
| Trial 3 avg (µs) | 184.7 | 186.0 | 184.0 |
| **Mean avg (µs)** | **186.6** | **185.2** | **184.4** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-1.4µs (-0.8%)   Δ(AG vs B0)=-2.2µs (-1.2%)   Δ(AG vs B2)=-0.8µs (-0.4%)

---

### 20×192×32  —  `action_out_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,32,256) | (32,32,256) | (32,32,256) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 125.5 | 128.0 | 129.6 |
| Trial 2 avg (µs) | 130.9 | 128.5 | 126.0 |
| Trial 3 avg (µs) | 127.2 | 124.8 | 123.7 |
| **Mean avg (µs)** | **127.8** | **127.1** | **126.5** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-0.7µs (-0.6%)   Δ(AG vs B0)=-1.4µs (-1.1%)   Δ(AG vs B2)=-0.7µs (-0.5%)

---

### 20×192×256  —  `gate_proj, up_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,256,256) | (32,256,256) | (32,256,256) |
| Tile | (32,32,32) | (32,64,32) | (32,64,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 141.8 | 140.5 | 137.6 |
| Trial 2 avg (µs) | 144.9 | 137.6 | 139.3 |
| Trial 3 avg (µs) | 145.2 | 139.5 | 138.4 |
| **Mean avg (µs)** | **144.0** | **139.2** | **138.4** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-4.8µs (-3.3%)   Δ(AG vs B0)=-5.5µs (-3.8%)   Δ(AG vs B2)=-0.8µs (-0.6%)

---

### 20×192×960  —  `o_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,960,256) | (32,960,256) | (32,960,256) |
| Tile | (16,32,32) | (16,32,32) | (16,32,32) |
| Tile source | fixed | fallback_fixed | heuristic |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | FAIL | FAIL | 242.0 |
| Trial 2 avg (µs) | 242.5 | FAIL | 242.7 |
| Trial 3 avg (µs) | 319.6 | FAIL | 242.6 |
| **Mean avg (µs)** | **281.1** | **FAIL** | **242.4** |
| Pass count | 2/3 | 0/3 | 3/3 |

**Winner: INCONCLUSIVE**
Δ(AG vs B0)=-38.6µs (-13.7%)

---

### 20×256×96  —  `down_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,96,256) | (32,96,256) | (32,96,256) |
| Tile | (32,32,32) | (32,32,32) | (32,32,64) |
| Tile source | fixed | fallback_fixed | heuristic |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 131.6 | 132.8 | 133.3 |
| Trial 2 avg (µs) | 133.5 | 129.1 | 135.2 |
| Trial 3 avg (µs) | 135.6 | 131.9 | 134.9 |
| **Mean avg (µs)** | **133.6** | **131.2** | **134.5** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-2.3µs (-1.7%)   Δ(AG vs B0)=+0.9µs (+0.7%)   Δ(AG vs B2)=+3.2µs (+2.5%)

---

### 20×256×192  —  `down_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,192,256) | (32,192,256) | (32,192,256) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | fallback_fixed | heuristic |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 134.7 | 134.4 | 136.8 |
| Trial 2 avg (µs) | 135.6 | 133.9 | 133.6 |
| Trial 3 avg (µs) | 135.6 | 134.8 | 136.7 |
| **Mean avg (µs)** | **135.3** | **134.4** | **135.7** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-0.9µs (-0.7%)   Δ(AG vs B0)=+0.4µs (+0.3%)   Δ(AG vs B2)=+1.3µs (+1.0%)

---

### 20×256×288  —  `down_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,288,256) | (32,288,256) | (32,288,256) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | fallback_fixed | heuristic |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 153.8 | 153.9 | 151.9 |
| Trial 2 avg (µs) | 152.6 | 157.2 | 154.8 |
| Trial 3 avg (µs) | 155.1 | 168.0 | 156.6 |
| **Mean avg (µs)** | **153.8** | **159.7** | **154.4** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=+5.9µs (+3.8%)   Δ(AG vs B0)=+0.6µs (+0.4%)   Δ(AG vs B2)=-5.3µs (-3.3%)

---

### 20×256×384  —  `down_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,384,256) | (32,384,256) | (32,384,256) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | fallback_fixed | heuristic |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 154.5 | 157.5 | 156.0 |
| Trial 2 avg (µs) | 156.1 | 156.0 | 157.3 |
| Trial 3 avg (µs) | 153.0 | 159.5 | 158.5 |
| **Mean avg (µs)** | **154.5** | **157.7** | **157.3** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=+3.1µs (+2.0%)   Δ(AG vs B0)=+2.7µs (+1.8%)   Δ(AG vs B2)=-0.4µs (-0.3%)

---

### 20×256×480  —  `down_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,480,256) | (32,480,256) | (32,480,256) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | fallback_fixed | heuristic |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 176.2 | 174.7 | 173.4 |
| Trial 2 avg (µs) | 172.4 | 175.8 | 176.1 |
| Trial 3 avg (µs) | 177.3 | 175.0 | 174.8 |
| **Mean avg (µs)** | **175.3** | **175.2** | **174.8** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-0.1µs (-0.1%)   Δ(AG vs B0)=-0.5µs (-0.3%)   Δ(AG vs B2)=-0.4µs (-0.2%)

---

### 20×256×576  —  `down_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,576,256) | (32,576,256) | (32,576,256) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | fallback_fixed | heuristic |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 186.8 | 194.1 | 193.7 |
| Trial 2 avg (µs) | 190.5 | 194.8 | 190.1 |
| Trial 3 avg (µs) | 193.8 | 193.7 | 190.2 |
| **Mean avg (µs)** | **190.4** | **194.2** | **191.3** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=+3.8µs (+2.0%)   Δ(AG vs B0)=+1.0µs (+0.5%)   Δ(AG vs B2)=-2.8µs (-1.5%)

---

### 20×256×672  —  `down_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,672,256) | (32,672,256) | (32,672,256) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | fallback_fixed | heuristic |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 204.6 | 203.8 | 204.6 |
| Trial 2 avg (µs) | 204.1 | 209.0 | 206.6 |
| Trial 3 avg (µs) | 206.4 | 210.6 | 204.8 |
| **Mean avg (µs)** | **205.0** | **207.8** | **205.3** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=+2.8µs (+1.4%)   Δ(AG vs B0)=+0.3µs (+0.2%)   Δ(AG vs B2)=-2.5µs (-1.2%)

---

### 20×256×720  —  `down_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,768,256) | (32,768,256) | (32,768,256) |
| Tile | (32,32,32) | (32,64,32) | (32,64,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 193.7 | 163.7 | 161.8 |
| Trial 2 avg (µs) | 195.0 | 163.6 | 161.4 |
| Trial 3 avg (µs) | 195.7 | 163.6 | 161.8 |
| **Mean avg (µs)** | **194.8** | **163.6** | **161.7** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-31.2µs (-16.0%)   Δ(AG vs B0)=-33.1µs (-17.0%)   Δ(AG vs B2)=-1.9µs (-1.2%)

---

### 20×288×32  —  `action_out_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,32,512) | (32,32,512) | (32,32,512) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 130.6 | 131.3 | 135.2 |
| Trial 2 avg (µs) | 130.6 | 135.3 | 133.2 |
| Trial 3 avg (µs) | 131.6 | 130.8 | 130.9 |
| **Mean avg (µs)** | **130.9** | **132.5** | **133.1** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=+1.5µs (+1.2%)   Δ(AG vs B0)=+2.2µs (+1.6%)   Δ(AG vs B2)=+0.6µs (+0.5%)

---

### 20×288×256  —  `gate_proj, up_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,256,512) | (32,256,512) | (32,256,512) |
| Tile | (32,32,32) | (32,64,32) | (32,64,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 154.6 | 150.3 | 150.6 |
| Trial 2 avg (µs) | 156.2 | 147.2 | 150.1 |
| Trial 3 avg (µs) | 156.9 | 148.0 | 152.0 |
| **Mean avg (µs)** | **155.9** | **148.5** | **150.9** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-7.4µs (-4.7%)   Δ(AG vs B0)=-5.0µs (-3.2%)   Δ(AG vs B2)=+2.4µs (+1.6%)

---

### 20×288×960  —  `o_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,960,512) | (32,960,512) | (32,960,512) |
| Tile | (16,32,32) | (16,32,32) | (16,32,32) |
| Tile source | fixed | fallback_fixed | heuristic |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | FAIL | FAIL | FAIL |
| Trial 2 avg (µs) | 348.0 | 352.6 | FAIL |
| Trial 3 avg (µs) | FAIL | FAIL | FAIL |
| **Mean avg (µs)** | **348.0** | **352.6** | **FAIL** |
| Pass count | 1/3 | 1/3 | 0/3 |

**Winner: INCONCLUSIVE**
Δ(B2 vs B0)=+4.6µs (+1.3%)

---

### 20×384×32  —  `action_out_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,32,512) | (32,32,512) | (32,32,512) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 131.8 | 131.6 | 130.7 |
| Trial 2 avg (µs) | 133.2 | 135.0 | 134.5 |
| Trial 3 avg (µs) | 134.0 | 133.9 | 132.4 |
| **Mean avg (µs)** | **133.0** | **133.5** | **132.5** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=+0.5µs (+0.4%)   Δ(AG vs B0)=-0.5µs (-0.4%)   Δ(AG vs B2)=-1.0µs (-0.8%)

---

### 20×384×256  —  `gate_proj, up_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,256,512) | (32,256,512) | (32,256,512) |
| Tile | (32,32,32) | (32,64,32) | (32,64,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 157.9 | 150.4 | 148.5 |
| Trial 2 avg (µs) | 154.7 | 148.8 | 148.4 |
| Trial 3 avg (µs) | 151.5 | 152.8 | 146.5 |
| **Mean avg (µs)** | **154.7** | **150.7** | **147.8** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-4.0µs (-2.6%)   Δ(AG vs B0)=-6.9µs (-4.5%)   Δ(AG vs B2)=-2.9µs (-1.9%)

---

### 20×384×960  —  `o_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,960,512) | (32,960,512) | (32,960,512) |
| Tile | (16,32,32) | (16,32,32) | (16,32,32) |
| Tile source | fixed | fallback_fixed | heuristic |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 390.4 | 415.5 | FAIL |
| Trial 2 avg (µs) | 356.9 | FAIL | FAIL |
| Trial 3 avg (µs) | FAIL | FAIL | 431.1 |
| **Mean avg (µs)** | **373.6** | **415.5** | **431.1** |
| Pass count | 2/3 | 1/3 | 1/3 |

**Winner: BASELINE 0**
Δ(B2 vs B0)=+41.9µs (+11.2%)   Δ(AG vs B0)=+57.5µs (+15.4%)   Δ(AG vs B2)=+15.6µs (+3.8%)

---

### 20×480×32  —  `action_out_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,32,512) | (32,32,512) | (32,32,512) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 130.1 | 134.4 | 134.0 |
| Trial 2 avg (µs) | 134.1 | 132.3 | 133.7 |
| Trial 3 avg (µs) | 131.3 | 131.5 | 132.6 |
| **Mean avg (µs)** | **131.9** | **132.7** | **133.4** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=+0.9µs (+0.7%)   Δ(AG vs B0)=+1.6µs (+1.2%)   Δ(AG vs B2)=+0.7µs (+0.5%)

---

### 20×480×256  —  `gate_proj, up_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,256,512) | (32,256,512) | (32,256,512) |
| Tile | (32,32,32) | (32,64,32) | (32,64,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 156.1 | 148.6 | 149.2 |
| Trial 2 avg (µs) | 157.1 | 148.2 | 150.2 |
| Trial 3 avg (µs) | 155.7 | 152.1 | 147.3 |
| **Mean avg (µs)** | **156.3** | **149.6** | **148.9** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-6.7µs (-4.3%)   Δ(AG vs B0)=-7.4µs (-4.8%)   Δ(AG vs B2)=-0.7µs (-0.5%)

---

### 20×480×960  —  `o_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,960,512) | (32,960,512) | (32,960,512) |
| Tile | (16,32,32) | (16,32,32) | (16,32,32) |
| Tile source | fixed | fallback_fixed | heuristic |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | FAIL | 352.1 | FAIL |
| Trial 2 avg (µs) | FAIL | 355.0 | FAIL |
| Trial 3 avg (µs) | FAIL | FAIL | FAIL |
| **Mean avg (µs)** | **FAIL** | **353.5** | **FAIL** |
| Pass count | 0/3 | 2/3 | 0/3 |

**Winner: INCONCLUSIVE**

---

### 20×576×32  —  `action_out_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,32,768) | (32,32,768) | (32,32,768) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 138.9 | 139.3 | 138.9 |
| Trial 2 avg (µs) | 143.6 | 139.0 | 140.0 |
| Trial 3 avg (µs) | 137.2 | 138.2 | 135.4 |
| **Mean avg (µs)** | **139.9** | **138.8** | **138.1** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-1.1µs (-0.8%)   Δ(AG vs B0)=-1.8µs (-1.3%)   Δ(AG vs B2)=-0.7µs (-0.5%)

---

### 20×576×256  —  `gate_proj, up_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,256,768) | (32,256,768) | (32,256,768) |
| Tile | (32,32,32) | (32,64,32) | (32,64,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 164.6 | 159.7 | 159.0 |
| Trial 2 avg (µs) | 166.2 | 158.5 | 162.0 |
| Trial 3 avg (µs) | 166.0 | 157.9 | 162.9 |
| **Mean avg (µs)** | **165.6** | **158.7** | **161.3** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-6.9µs (-4.2%)   Δ(AG vs B0)=-4.3µs (-2.6%)   Δ(AG vs B2)=+2.6µs (+1.6%)

---

### 20×576×960  —  `o_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,960,768) | (32,960,768) | (32,960,768) |
| Tile | (16,32,32) | (16,32,32) | (16,32,32) |
| Tile source | fixed | fallback_fixed | heuristic |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | FAIL | 462.2 | 604.6 |
| Trial 2 avg (µs) | 537.5 | FAIL | FAIL |
| Trial 3 avg (µs) | 467.3 | 467.1 | FAIL |
| **Mean avg (µs)** | **502.4** | **464.6** | **604.6** |
| Pass count | 2/3 | 2/3 | 1/3 |

**Winner: BASELINE 2**
Δ(B2 vs B0)=-37.8µs (-7.5%)   Δ(AG vs B0)=+102.2µs (+20.3%)   Δ(AG vs B2)=+139.9µs (+30.1%)

---

### 20×672×32  —  `action_out_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,32,768) | (32,32,768) | (32,32,768) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 138.4 | 137.1 | 131.7 |
| Trial 2 avg (µs) | 137.9 | 136.4 | 136.5 |
| Trial 3 avg (µs) | 139.0 | 137.8 | 135.3 |
| **Mean avg (µs)** | **138.4** | **137.1** | **134.5** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-1.3µs (-1.0%)   Δ(AG vs B0)=-4.0µs (-2.9%)   Δ(AG vs B2)=-2.6µs (-1.9%)

---

### 20×672×256  —  `gate_proj, up_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,256,768) | (32,256,768) | (32,256,768) |
| Tile | (32,32,32) | (32,64,32) | (32,64,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 164.2 | 158.8 | 162.1 |
| Trial 2 avg (µs) | 167.8 | 156.2 | 158.2 |
| Trial 3 avg (µs) | 165.7 | 159.7 | 163.0 |
| **Mean avg (µs)** | **165.9** | **158.2** | **161.1** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-7.7µs (-4.6%)   Δ(AG vs B0)=-4.8µs (-2.9%)   Δ(AG vs B2)=+2.9µs (+1.8%)

---

### 20×672×960  —  `o_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,960,768) | (32,960,768) | (32,960,768) |
| Tile | (16,32,32) | (16,32,32) | (16,32,32) |
| Tile source | fixed | fallback_fixed | heuristic |
| Pad impl | numpy_pad | numpy_pad | numpy_pad |
| Trial 1 avg (µs) | FAIL | FAIL | FAIL |
| Trial 2 avg (µs) | FAIL | FAIL | 468.3 |
| Trial 3 avg (µs) | FAIL | FAIL | FAIL |
| **Mean avg (µs)** | **FAIL** | **FAIL** | **468.3** |
| Pass count | 0/3 | 0/3 | 1/3 |

**Winner: INCONCLUSIVE**

---

### 20×720×32  —  `action_out_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,32,768) | (32,32,768) | (32,32,768) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 139.7 | 137.3 | 134.5 |
| Trial 2 avg (µs) | 137.0 | 136.6 | 139.0 |
| Trial 3 avg (µs) | 138.7 | 134.2 | 140.2 |
| **Mean avg (µs)** | **138.5** | **136.0** | **137.9** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-2.5µs (-1.8%)   Δ(AG vs B0)=-0.6µs (-0.5%)   Δ(AG vs B2)=+1.9µs (+1.4%)

---

### 20×720×256  —  `gate_proj, up_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,256,768) | (32,256,768) | (32,256,768) |
| Tile | (32,32,32) | (32,64,32) | (32,64,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 165.2 | 161.4 | 157.1 |
| Trial 2 avg (µs) | 166.7 | 160.3 | 156.8 |
| Trial 3 avg (µs) | 165.6 | 159.1 | 160.7 |
| **Mean avg (µs)** | **165.8** | **160.2** | **158.2** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-5.6µs (-3.4%)   Δ(AG vs B0)=-7.6µs (-4.6%)   Δ(AG vs B2)=-2.1µs (-1.3%)

---

### 20×720×960  —  `o_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,960,768) | (32,960,768) | (32,960,768) |
| Tile | (16,32,32) | (16,32,32) | (16,32,32) |
| Tile source | fixed | fallback_fixed | heuristic |
| Pad impl | numpy_pad | numpy_pad | numpy_pad |
| Trial 1 avg (µs) | FAIL | FAIL | FAIL |
| Trial 2 avg (µs) | FAIL | 770.3 | FAIL |
| Trial 3 avg (µs) | FAIL | FAIL | FAIL |
| **Mean avg (µs)** | **FAIL** | **770.3** | **FAIL** |
| Pass count | 0/3 | 1/3 | 0/3 |

**Winner: INCONCLUSIVE**

---

### 20×960×96  —  `q_proj`

**Pre-screened skip**: bf16_accuracy_small_MN_large_K

---

### 20×960×192  —  `q_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,192,960) | (32,192,960) | (32,192,960) |
| Tile | (16,32,32) | (16,32,32) | (16,32,32) |
| Tile source | fixed | fallback_fixed | heuristic |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 175.4 | 174.2 | 174.6 |
| Trial 2 avg (µs) | 171.5 | 169.5 | 174.8 |
| Trial 3 avg (µs) | 174.5 | 176.4 | 169.2 |
| **Mean avg (µs)** | **FAIL** | **FAIL** | **FAIL** |
| Pass count | 0/3 | 0/3 | 0/3 |

**Winner: INCONCLUSIVE**

---

### 20×960×288  —  `q_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,288,960) | (32,288,960) | (32,288,960) |
| Tile | (16,32,32) | (16,32,32) | (16,32,32) |
| Tile source | fixed | fallback_fixed | heuristic |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 325.0 | 241.9 | 320.4 |
| Trial 2 avg (µs) | 243.1 | 330.1 | 240.3 |
| Trial 3 avg (µs) | 243.9 | 241.9 | 241.6 |
| **Mean avg (µs)** | **FAIL** | **FAIL** | **FAIL** |
| Pass count | 0/3 | 0/3 | 0/3 |

**Winner: INCONCLUSIVE**

---

### 20×960×384  —  `q_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,384,960) | (32,384,960) | (32,384,960) |
| Tile | (16,32,32) | (16,32,32) | (16,32,32) |
| Tile source | fixed | fallback_fixed | heuristic |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 199.2 | 197.3 | 199.3 |
| Trial 2 avg (µs) | 197.9 | 197.1 | 197.6 |
| Trial 3 avg (µs) | 197.8 | 197.8 | 221.6 |
| **Mean avg (µs)** | **FAIL** | **FAIL** | **FAIL** |
| Pass count | 0/3 | 0/3 | 0/3 |

**Winner: INCONCLUSIVE**

---

### 20×960×480  —  `q_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,480,960) | (32,480,960) | (32,480,960) |
| Tile | (16,32,32) | (16,32,32) | (16,32,32) |
| Tile source | fixed | fallback_fixed | heuristic |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 415.3 | 322.3 | 413.9 |
| Trial 2 avg (µs) | 303.7 | 356.4 | 435.6 |
| Trial 3 avg (µs) | 446.9 | 301.5 | 304.6 |
| **Mean avg (µs)** | **FAIL** | **FAIL** | **FAIL** |
| Pass count | 0/3 | 0/3 | 0/3 |

**Winner: INCONCLUSIVE**

---

### 20×960×576  —  `q_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,576,960) | (32,576,960) | (32,576,960) |
| Tile | (16,32,32) | (16,32,32) | (16,32,32) |
| Tile source | fixed | fallback_fixed | heuristic |
| Pad impl | numpy_pad | numpy_pad | numpy_pad |
| Trial 1 avg (µs) | 373.2 | 364.0 | 408.7 |
| Trial 2 avg (µs) | 393.2 | 369.5 | 363.3 |
| Trial 3 avg (µs) | 365.9 | 368.4 | 362.0 |
| **Mean avg (µs)** | **FAIL** | **FAIL** | **FAIL** |
| Pass count | 0/3 | 0/3 | 0/3 |

**Winner: INCONCLUSIVE**

---

### 20×960×672  —  `q_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,672,960) | (32,672,960) | (32,672,960) |
| Tile | (16,32,32) | (16,32,32) | (16,32,32) |
| Tile source | fixed | fallback_fixed | heuristic |
| Pad impl | numpy_pad | numpy_pad | numpy_pad |
| Trial 1 avg (µs) | 426.6 | 449.1 | 424.1 |
| Trial 2 avg (µs) | 426.3 | 422.3 | 484.6 |
| Trial 3 avg (µs) | 424.9 | 429.7 | 499.2 |
| **Mean avg (µs)** | **FAIL** | **FAIL** | **FAIL** |
| Pass count | 0/3 | 0/3 | 0/3 |

**Winner: INCONCLUSIVE**

---

### 20×960×720  —  `q_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,768,960) | (32,768,960) | (32,768,960) |
| Tile | (16,32,32) | (16,32,32) | (16,32,32) |
| Tile source | fixed | fallback_fixed | heuristic |
| Pad impl | numpy_pad | numpy_pad | numpy_pad |
| Trial 1 avg (µs) | 261.6 | 261.6 | 263.6 |
| Trial 2 avg (µs) | 262.6 | 263.7 | 351.3 |
| Trial 3 avg (µs) | 261.2 | 257.3 | 481.2 |
| **Mean avg (µs)** | **FAIL** | **FAIL** | **FAIL** |
| Pass count | 0/3 | 0/3 | 0/3 |

**Winner: INCONCLUSIVE**

---

### 30×32×96  —  `action_in_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,96,32) | (32,96,32) | (32,96,32) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | fallback_fixed | heuristic |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 130.5 | 133.9 | 130.7 |
| Trial 2 avg (µs) | 128.2 | 129.7 | 126.6 |
| Trial 3 avg (µs) | 130.9 | 129.3 | 129.7 |
| **Mean avg (µs)** | **129.9** | **131.0** | **129.0** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=+1.1µs (+0.8%)   Δ(AG vs B0)=-0.9µs (-0.7%)   Δ(AG vs B2)=-1.9µs (-1.5%)

---

### 30×32×192  —  `action_in_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,192,32) | (32,192,32) | (32,192,32) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | fallback_fixed | heuristic |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 131.1 | 126.5 | 121.9 |
| Trial 2 avg (µs) | 126.9 | 126.9 | 124.8 |
| Trial 3 avg (µs) | 127.1 | 127.6 | 126.1 |
| **Mean avg (µs)** | **128.4** | **127.0** | **124.3** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: AGENTIC**
Δ(B2 vs B0)=-1.4µs (-1.1%)   Δ(AG vs B0)=-4.1µs (-3.2%)   Δ(AG vs B2)=-2.8µs (-2.2%)

---

### 30×32×288  —  `action_in_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,288,32) | (32,288,32) | (32,288,32) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | fallback_fixed | heuristic |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 131.8 | 130.7 | 133.2 |
| Trial 2 avg (µs) | 130.9 | 128.4 | 132.4 |
| Trial 3 avg (µs) | 131.3 | 134.8 | 133.0 |
| **Mean avg (µs)** | **131.3** | **131.3** | **132.9** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-0.0µs (-0.0%)   Δ(AG vs B0)=+1.6µs (+1.2%)   Δ(AG vs B2)=+1.6µs (+1.2%)

---

### 30×32×384  —  `action_in_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,384,32) | (32,384,32) | (32,384,32) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | fallback_fixed | heuristic |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 153.2 | 155.4 | 157.0 |
| Trial 2 avg (µs) | 156.1 | 158.6 | 157.7 |
| Trial 3 avg (µs) | 156.0 | 159.1 | 156.0 |
| **Mean avg (µs)** | **155.1** | **157.7** | **156.9** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=+2.6µs (+1.7%)   Δ(AG vs B0)=+1.8µs (+1.1%)   Δ(AG vs B2)=-0.8µs (-0.5%)

---

### 30×32×480  —  `action_in_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,480,32) | (32,480,32) | (32,480,32) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | fallback_fixed | heuristic |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 133.4 | 135.6 | 137.9 |
| Trial 2 avg (µs) | 136.1 | 136.0 | 138.9 |
| Trial 3 avg (µs) | 136.4 | 136.6 | 134.1 |
| **Mean avg (µs)** | **135.3** | **136.0** | **136.9** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=+0.7µs (+0.5%)   Δ(AG vs B0)=+1.6µs (+1.2%)   Δ(AG vs B2)=+0.9µs (+0.7%)

---

### 30×32×576  —  `action_in_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,576,32) | (32,576,32) | (32,576,32) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | fallback_fixed | heuristic |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 139.0 | 143.1 | 139.5 |
| Trial 2 avg (µs) | 138.4 | 136.7 | 139.1 |
| Trial 3 avg (µs) | 140.5 | 140.5 | 141.8 |
| **Mean avg (µs)** | **139.3** | **140.1** | **140.2** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=+0.8µs (+0.6%)   Δ(AG vs B0)=+0.9µs (+0.6%)   Δ(AG vs B2)=+0.1µs (+0.1%)

---

### 30×32×672  —  `action_in_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,672,32) | (32,672,32) | (32,672,32) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | fallback_fixed | heuristic |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 146.3 | 145.3 | 148.2 |
| Trial 2 avg (µs) | 147.0 | 147.5 | 147.4 |
| Trial 3 avg (µs) | 145.3 | 148.4 | 147.1 |
| **Mean avg (µs)** | **146.2** | **147.1** | **147.5** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=+0.8µs (+0.6%)   Δ(AG vs B0)=+1.3µs (+0.9%)   Δ(AG vs B2)=+0.5µs (+0.3%)

---

### 30×32×720  —  `action_in_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,768,32) | (32,768,32) | (32,768,32) |
| Tile | (32,32,32) | (32,64,32) | (32,64,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 196.9 | 158.6 | 155.5 |
| Trial 2 avg (µs) | 193.2 | 152.9 | 158.6 |
| Trial 3 avg (µs) | 196.3 | 155.2 | 153.8 |
| **Mean avg (µs)** | **195.5** | **155.6** | **155.9** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-39.9µs (-20.4%)   Δ(AG vs B0)=-39.5µs (-20.2%)   Δ(AG vs B2)=+0.3µs (+0.2%)

---

### 30×96×32  —  `action_out_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,32,128) | (32,32,128) | (32,32,128) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 120.8 | 121.7 | 121.1 |
| Trial 2 avg (µs) | 121.6 | 124.3 | 117.4 |
| Trial 3 avg (µs) | 124.0 | 125.7 | 121.9 |
| **Mean avg (µs)** | **122.1** | **123.9** | **120.1** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=+1.8µs (+1.5%)   Δ(AG vs B0)=-2.0µs (-1.6%)   Δ(AG vs B2)=-3.8µs (-3.1%)

---

### 30×96×256  —  `gate_proj, up_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,256,128) | (32,256,128) | (32,256,128) |
| Tile | (32,32,32) | (32,64,32) | (32,64,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 143.6 | 134.5 | 135.2 |
| Trial 2 avg (µs) | 144.1 | 133.9 | 132.9 |
| Trial 3 avg (µs) | 140.3 | 135.7 | 133.9 |
| **Mean avg (µs)** | **142.7** | **134.7** | **134.0** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-8.0µs (-5.6%)   Δ(AG vs B0)=-8.7µs (-6.1%)   Δ(AG vs B2)=-0.7µs (-0.5%)

---

### 30×96×960  —  `o_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,960,128) | (32,960,128) | (32,960,128) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | fallback_fixed | heuristic |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 180.6 | 187.9 | 186.6 |
| Trial 2 avg (µs) | 187.8 | 186.2 | 184.4 |
| Trial 3 avg (µs) | 185.9 | 186.7 | 187.0 |
| **Mean avg (µs)** | **184.8** | **186.9** | **186.0** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=+2.2µs (+1.2%)   Δ(AG vs B0)=+1.2µs (+0.7%)   Δ(AG vs B2)=-0.9µs (-0.5%)

---

### 30×192×32  —  `action_out_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,32,256) | (32,32,256) | (32,32,256) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 123.2 | 125.5 | 127.8 |
| Trial 2 avg (µs) | 128.3 | 127.0 | 127.2 |
| Trial 3 avg (µs) | 127.6 | 128.0 | 125.8 |
| **Mean avg (µs)** | **126.4** | **126.8** | **126.9** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=+0.5µs (+0.4%)   Δ(AG vs B0)=+0.6µs (+0.4%)   Δ(AG vs B2)=+0.1µs (+0.1%)

---

### 30×192×256  —  `gate_proj, up_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,256,256) | (32,256,256) | (32,256,256) |
| Tile | (32,32,32) | (32,64,32) | (32,64,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 144.8 | 139.2 | 141.5 |
| Trial 2 avg (µs) | 146.2 | 140.6 | 142.0 |
| Trial 3 avg (µs) | 146.9 | 138.4 | 136.6 |
| **Mean avg (µs)** | **146.0** | **139.4** | **140.0** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-6.6µs (-4.5%)   Δ(AG vs B0)=-6.0µs (-4.1%)   Δ(AG vs B2)=+0.6µs (+0.4%)

---

### 30×192×960  —  `o_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,960,256) | (32,960,256) | (32,960,256) |
| Tile | (16,32,32) | (16,32,32) | (16,32,32) |
| Tile source | fixed | fallback_fixed | heuristic |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | FAIL | 243.9 | 244.5 |
| Trial 2 avg (µs) | FAIL | FAIL | FAIL |
| Trial 3 avg (µs) | FAIL | FAIL | FAIL |
| **Mean avg (µs)** | **FAIL** | **243.9** | **244.5** |
| Pass count | 0/3 | 1/3 | 1/3 |

**Winner: INCONCLUSIVE**
Δ(AG vs B2)=+0.6µs (+0.2%)

---

### 30×256×96  —  `down_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,96,256) | (32,96,256) | (32,96,256) |
| Tile | (32,32,32) | (32,32,32) | (32,32,64) |
| Tile source | fixed | fallback_fixed | heuristic |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 130.9 | 130.2 | 133.8 |
| Trial 2 avg (µs) | 130.1 | 132.7 | 135.1 |
| Trial 3 avg (µs) | 131.5 | 132.9 | 136.7 |
| **Mean avg (µs)** | **130.8** | **131.9** | **135.2** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=+1.1µs (+0.8%)   Δ(AG vs B0)=+4.4µs (+3.4%)   Δ(AG vs B2)=+3.3µs (+2.5%)

---

### 30×256×192  —  `down_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,192,256) | (32,192,256) | (32,192,256) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | fallback_fixed | heuristic |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 134.3 | 136.3 | 135.0 |
| Trial 2 avg (µs) | 134.1 | 134.1 | 137.7 |
| Trial 3 avg (µs) | 136.7 | 135.4 | 136.9 |
| **Mean avg (µs)** | **135.0** | **135.3** | **136.5** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=+0.2µs (+0.2%)   Δ(AG vs B0)=+1.5µs (+1.1%)   Δ(AG vs B2)=+1.3µs (+0.9%)

---

### 30×256×288  —  `down_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,288,256) | (32,288,256) | (32,288,256) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | fallback_fixed | heuristic |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 158.1 | 152.5 | 155.1 |
| Trial 2 avg (µs) | 163.9 | 155.9 | 154.1 |
| Trial 3 avg (µs) | 154.9 | 155.5 | 152.5 |
| **Mean avg (µs)** | **159.0** | **154.7** | **153.9** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-4.3µs (-2.7%)   Δ(AG vs B0)=-5.1µs (-3.2%)   Δ(AG vs B2)=-0.8µs (-0.5%)

---

### 30×256×384  —  `down_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,384,256) | (32,384,256) | (32,384,256) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | fallback_fixed | heuristic |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 156.5 | 158.1 | 156.7 |
| Trial 2 avg (µs) | 157.2 | 155.0 | 157.4 |
| Trial 3 avg (µs) | 159.5 | 156.5 | 157.8 |
| **Mean avg (µs)** | **157.8** | **156.6** | **157.3** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-1.2µs (-0.8%)   Δ(AG vs B0)=-0.5µs (-0.3%)   Δ(AG vs B2)=+0.8µs (+0.5%)

---

### 30×256×480  —  `down_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,480,256) | (32,480,256) | (32,480,256) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | fallback_fixed | heuristic |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 173.9 | 174.7 | 174.4 |
| Trial 2 avg (µs) | 174.0 | 173.9 | 176.8 |
| Trial 3 avg (µs) | 173.1 | 173.5 | 173.7 |
| **Mean avg (µs)** | **173.7** | **174.0** | **175.0** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=+0.3µs (+0.2%)   Δ(AG vs B0)=+1.3µs (+0.8%)   Δ(AG vs B2)=+1.0µs (+0.6%)

---

### 30×256×576  —  `down_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,576,256) | (32,576,256) | (32,576,256) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | fallback_fixed | heuristic |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 190.0 | 187.4 | 190.3 |
| Trial 2 avg (µs) | 196.2 | 187.8 | 196.1 |
| Trial 3 avg (µs) | 190.5 | 196.3 | 192.6 |
| **Mean avg (µs)** | **192.2** | **190.5** | **193.0** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-1.8µs (-0.9%)   Δ(AG vs B0)=+0.7µs (+0.4%)   Δ(AG vs B2)=+2.5µs (+1.3%)

---

### 30×256×672  —  `down_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,672,256) | (32,672,256) | (32,672,256) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | fallback_fixed | heuristic |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 206.0 | 206.9 | 205.2 |
| Trial 2 avg (µs) | 207.4 | 205.2 | 207.2 |
| Trial 3 avg (µs) | 209.0 | 208.0 | 207.0 |
| **Mean avg (µs)** | **207.5** | **206.7** | **206.5** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-0.8µs (-0.4%)   Δ(AG vs B0)=-1.0µs (-0.5%)   Δ(AG vs B2)=-0.3µs (-0.1%)

---

### 30×256×720  —  `down_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,768,256) | (32,768,256) | (32,768,256) |
| Tile | (32,32,32) | (32,64,32) | (32,64,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 197.6 | 165.1 | 163.1 |
| Trial 2 avg (µs) | 194.6 | 162.6 | 161.1 |
| Trial 3 avg (µs) | 196.4 | 161.2 | 162.9 |
| **Mean avg (µs)** | **196.2** | **163.0** | **162.4** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-33.2µs (-16.9%)   Δ(AG vs B0)=-33.8µs (-17.2%)   Δ(AG vs B2)=-0.6µs (-0.3%)

---

### 30×288×32  —  `action_out_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,32,512) | (32,32,512) | (32,32,512) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 130.5 | 132.3 | 135.0 |
| Trial 2 avg (µs) | 132.7 | 130.3 | 131.7 |
| Trial 3 avg (µs) | 132.2 | 135.7 | 127.9 |
| **Mean avg (µs)** | **131.8** | **132.8** | **131.5** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=+0.9µs (+0.7%)   Δ(AG vs B0)=-0.3µs (-0.2%)   Δ(AG vs B2)=-1.2µs (-0.9%)

---

### 30×288×256  —  `gate_proj, up_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,256,512) | (32,256,512) | (32,256,512) |
| Tile | (32,32,32) | (32,64,32) | (32,64,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 157.2 | 148.5 | 148.3 |
| Trial 2 avg (µs) | 156.1 | 147.0 | 148.5 |
| Trial 3 avg (µs) | 156.1 | 147.3 | 150.4 |
| **Mean avg (µs)** | **156.5** | **147.6** | **149.1** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-8.9µs (-5.7%)   Δ(AG vs B0)=-7.4µs (-4.7%)   Δ(AG vs B2)=+1.5µs (+1.0%)

---

### 30×288×960  —  `o_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,960,512) | (32,960,512) | (32,960,512) |
| Tile | (16,32,32) | (16,32,32) | (16,32,32) |
| Tile source | fixed | fallback_fixed | heuristic |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 354.1 | FAIL | 381.7 |
| Trial 2 avg (µs) | FAIL | FAIL | 355.3 |
| Trial 3 avg (µs) | FAIL | FAIL | FAIL |
| **Mean avg (µs)** | **354.1** | **FAIL** | **368.5** |
| Pass count | 1/3 | 0/3 | 2/3 |

**Winner: INCONCLUSIVE**
Δ(AG vs B0)=+14.4µs (+4.1%)

---

### 30×384×32  —  `action_out_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,32,512) | (32,32,512) | (32,32,512) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 133.0 | 128.2 | 131.1 |
| Trial 2 avg (µs) | 129.7 | 131.0 | 129.6 |
| Trial 3 avg (µs) | 133.4 | 132.1 | 132.2 |
| **Mean avg (µs)** | **132.0** | **130.4** | **130.9** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-1.6µs (-1.2%)   Δ(AG vs B0)=-1.1µs (-0.8%)   Δ(AG vs B2)=+0.5µs (+0.4%)

---

### 30×384×256  —  `gate_proj, up_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,256,512) | (32,256,512) | (32,256,512) |
| Tile | (32,32,32) | (32,64,32) | (32,64,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 156.1 | 150.7 | 150.3 |
| Trial 2 avg (µs) | 154.1 | 149.2 | 149.5 |
| Trial 3 avg (µs) | 155.8 | 149.2 | 149.0 |
| **Mean avg (µs)** | **155.3** | **149.7** | **149.6** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-5.7µs (-3.6%)   Δ(AG vs B0)=-5.8µs (-3.7%)   Δ(AG vs B2)=-0.1µs (-0.1%)

---

### 30×384×960  —  `o_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,960,512) | (32,960,512) | (32,960,512) |
| Tile | (16,32,32) | (16,32,32) | (16,32,32) |
| Tile source | fixed | fallback_fixed | heuristic |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 354.5 | 349.3 | FAIL |
| Trial 2 avg (µs) | 435.8 | 351.0 | FAIL |
| Trial 3 avg (µs) | 352.7 | FAIL | 478.0 |
| **Mean avg (µs)** | **381.0** | **350.1** | **478.0** |
| Pass count | 3/3 | 2/3 | 1/3 |

**Winner: BASELINE 2**
Δ(B2 vs B0)=-30.9µs (-8.1%)   Δ(AG vs B0)=+97.0µs (+25.5%)   Δ(AG vs B2)=+127.9µs (+36.5%)

---

### 30×480×32  —  `action_out_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,32,512) | (32,32,512) | (32,32,512) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 133.5 | 131.2 | 132.5 |
| Trial 2 avg (µs) | 131.6 | 132.3 | 137.4 |
| Trial 3 avg (µs) | 134.0 | 128.7 | 127.4 |
| **Mean avg (µs)** | **133.0** | **130.8** | **132.4** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-2.3µs (-1.7%)   Δ(AG vs B0)=-0.6µs (-0.4%)   Δ(AG vs B2)=+1.7µs (+1.3%)

---

### 30×480×256  —  `gate_proj, up_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,256,512) | (32,256,512) | (32,256,512) |
| Tile | (32,32,32) | (32,64,32) | (32,64,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 152.9 | 149.7 | 152.1 |
| Trial 2 avg (µs) | 156.3 | 151.9 | 147.9 |
| Trial 3 avg (µs) | 156.4 | 149.1 | 149.5 |
| **Mean avg (µs)** | **155.2** | **150.2** | **149.8** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-4.9µs (-3.2%)   Δ(AG vs B0)=-5.4µs (-3.5%)   Δ(AG vs B2)=-0.4µs (-0.3%)

---

### 30×480×960  —  `o_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,960,512) | (32,960,512) | (32,960,512) |
| Tile | (16,32,32) | (16,32,32) | (16,32,32) |
| Tile source | fixed | fallback_fixed | heuristic |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 353.3 | 353.5 | 353.4 |
| Trial 2 avg (µs) | 353.1 | FAIL | FAIL |
| Trial 3 avg (µs) | FAIL | FAIL | FAIL |
| **Mean avg (µs)** | **353.2** | **353.5** | **353.4** |
| Pass count | 2/3 | 1/3 | 1/3 |

**Winner: TIE**
Δ(B2 vs B0)=+0.3µs (+0.1%)   Δ(AG vs B0)=+0.3µs (+0.1%)   Δ(AG vs B2)=-0.0µs (-0.0%)

---

### 30×576×32  —  `action_out_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,32,768) | (32,32,768) | (32,32,768) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 136.4 | 136.4 | 137.6 |
| Trial 2 avg (µs) | 138.3 | 137.4 | 138.9 |
| Trial 3 avg (µs) | 136.6 | 136.7 | 133.9 |
| **Mean avg (µs)** | **137.1** | **136.8** | **136.8** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-0.2µs (-0.2%)   Δ(AG vs B0)=-0.3µs (-0.2%)   Δ(AG vs B2)=-0.1µs (-0.0%)

---

### 30×576×256  —  `gate_proj, up_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,256,768) | (32,256,768) | (32,256,768) |
| Tile | (32,32,32) | (32,64,32) | (32,64,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 166.5 | 156.4 | 159.6 |
| Trial 2 avg (µs) | 165.8 | 160.1 | 157.1 |
| Trial 3 avg (µs) | 165.6 | 158.7 | 157.1 |
| **Mean avg (µs)** | **166.0** | **158.4** | **158.0** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-7.6µs (-4.6%)   Δ(AG vs B0)=-8.0µs (-4.8%)   Δ(AG vs B2)=-0.4µs (-0.3%)

---

### 30×576×960  —  `o_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,960,768) | (32,960,768) | (32,960,768) |
| Tile | (16,32,32) | (16,32,32) | (16,32,32) |
| Tile source | fixed | fallback_fixed | heuristic |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 548.9 | 465.6 | FAIL |
| Trial 2 avg (µs) | FAIL | FAIL | FAIL |
| Trial 3 avg (µs) | FAIL | FAIL | FAIL |
| **Mean avg (µs)** | **548.9** | **465.6** | **FAIL** |
| Pass count | 1/3 | 1/3 | 0/3 |

**Winner: INCONCLUSIVE**
Δ(B2 vs B0)=-83.3µs (-15.2%)

---

### 30×672×32  —  `action_out_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,32,768) | (32,32,768) | (32,32,768) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 139.8 | 138.9 | 136.4 |
| Trial 2 avg (µs) | 140.0 | 137.3 | 136.6 |
| Trial 3 avg (µs) | 136.5 | 138.3 | 136.9 |
| **Mean avg (µs)** | **138.8** | **138.2** | **136.6** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-0.6µs (-0.4%)   Δ(AG vs B0)=-2.1µs (-1.5%)   Δ(AG vs B2)=-1.5µs (-1.1%)

---

### 30×672×256  —  `gate_proj, up_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,256,768) | (32,256,768) | (32,256,768) |
| Tile | (32,32,32) | (32,64,32) | (32,64,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 163.1 | 161.4 | 159.5 |
| Trial 2 avg (µs) | 164.4 | 160.9 | 160.9 |
| Trial 3 avg (µs) | 164.6 | 159.3 | 158.7 |
| **Mean avg (µs)** | **164.0** | **160.5** | **159.7** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-3.5µs (-2.1%)   Δ(AG vs B0)=-4.3µs (-2.6%)   Δ(AG vs B2)=-0.8µs (-0.5%)

---

### 30×672×960  —  `o_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,960,768) | (32,960,768) | (32,960,768) |
| Tile | (16,32,32) | (16,32,32) | (16,32,32) |
| Tile source | fixed | fallback_fixed | heuristic |
| Pad impl | numpy_pad | numpy_pad | numpy_pad |
| Trial 1 avg (µs) | FAIL | FAIL | FAIL |
| Trial 2 avg (µs) | 742.0 | FAIL | FAIL |
| Trial 3 avg (µs) | FAIL | FAIL | FAIL |
| **Mean avg (µs)** | **742.0** | **FAIL** | **FAIL** |
| Pass count | 1/3 | 0/3 | 0/3 |

**Winner: INCONCLUSIVE**

---

### 30×720×32  —  `action_out_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,32,768) | (32,32,768) | (32,32,768) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 135.5 | 136.2 | 138.5 |
| Trial 2 avg (µs) | 136.6 | 138.7 | 136.9 |
| Trial 3 avg (µs) | 139.3 | 138.7 | 137.4 |
| **Mean avg (µs)** | **137.2** | **137.9** | **137.6** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=+0.7µs (+0.5%)   Δ(AG vs B0)=+0.4µs (+0.3%)   Δ(AG vs B2)=-0.3µs (-0.2%)

---

### 30×720×256  —  `gate_proj, up_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,256,768) | (32,256,768) | (32,256,768) |
| Tile | (32,32,32) | (32,64,32) | (32,64,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 166.7 | 161.1 | 160.7 |
| Trial 2 avg (µs) | 163.9 | 158.1 | 161.4 |
| Trial 3 avg (µs) | 164.7 | 158.8 | 159.6 |
| **Mean avg (µs)** | **165.1** | **159.3** | **160.6** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-5.7µs (-3.5%)   Δ(AG vs B0)=-4.5µs (-2.7%)   Δ(AG vs B2)=+1.2µs (+0.8%)

---

### 30×720×960  —  `o_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,960,768) | (32,960,768) | (32,960,768) |
| Tile | (16,32,32) | (16,32,32) | (16,32,32) |
| Tile source | fixed | fallback_fixed | heuristic |
| Pad impl | numpy_pad | numpy_pad | numpy_pad |
| Trial 1 avg (µs) | FAIL | 465.8 | FAIL |
| Trial 2 avg (µs) | 768.0 | FAIL | FAIL |
| Trial 3 avg (µs) | 462.5 | 466.4 | FAIL |
| **Mean avg (µs)** | **615.2** | **466.1** | **FAIL** |
| Pass count | 2/3 | 2/3 | 0/3 |

**Winner: INCONCLUSIVE**
Δ(B2 vs B0)=-149.2µs (-24.2%)

---

### 30×960×96  —  `q_proj`

**Pre-screened skip**: bf16_accuracy_small_MN_large_K

---

### 30×960×192  —  `q_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,192,960) | (32,192,960) | (32,192,960) |
| Tile | (16,32,32) | (16,32,32) | (16,32,32) |
| Tile source | fixed | fallback_fixed | heuristic |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 176.8 | 172.4 | 174.6 |
| Trial 2 avg (µs) | 174.2 | 173.6 | 173.3 |
| Trial 3 avg (µs) | 176.2 | 168.8 | 173.1 |
| **Mean avg (µs)** | **FAIL** | **FAIL** | **FAIL** |
| Pass count | 0/3 | 0/3 | 0/3 |

**Winner: INCONCLUSIVE**

---

### 30×960×288  —  `q_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,288,960) | (32,288,960) | (32,288,960) |
| Tile | (16,32,32) | (16,32,32) | (16,32,32) |
| Tile source | fixed | fallback_fixed | heuristic |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 244.4 | 245.5 | 241.6 |
| Trial 2 avg (µs) | 242.0 | 239.9 | 246.9 |
| Trial 3 avg (µs) | 242.0 | 243.2 | 244.7 |
| **Mean avg (µs)** | **FAIL** | **FAIL** | **FAIL** |
| Pass count | 0/3 | 0/3 | 0/3 |

**Winner: INCONCLUSIVE**

---

### 30×960×384  —  `q_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,384,960) | (32,384,960) | (32,384,960) |
| Tile | (16,32,32) | (16,32,32) | (16,32,32) |
| Tile source | fixed | fallback_fixed | heuristic |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 200.4 | 199.8 | 199.8 |
| Trial 2 avg (µs) | 198.2 | 197.1 | 198.1 |
| Trial 3 avg (µs) | 199.7 | 198.7 | 251.2 |
| **Mean avg (µs)** | **FAIL** | **FAIL** | **FAIL** |
| Pass count | 0/3 | 0/3 | 0/3 |

**Winner: INCONCLUSIVE**

---

### 30×960×480  —  `q_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,480,960) | (32,480,960) | (32,480,960) |
| Tile | (16,32,32) | (16,32,32) | (16,32,32) |
| Tile source | fixed | fallback_fixed | heuristic |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 402.7 | 391.7 | 431.1 |
| Trial 2 avg (µs) | 401.8 | 430.0 | 443.3 |
| Trial 3 avg (µs) | 427.4 | 414.8 | 308.5 |
| **Mean avg (µs)** | **FAIL** | **FAIL** | **FAIL** |
| Pass count | 0/3 | 0/3 | 0/3 |

**Winner: INCONCLUSIVE**

---

### 30×960×576  —  `q_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,576,960) | (32,576,960) | (32,576,960) |
| Tile | (16,32,32) | (16,32,32) | (16,32,32) |
| Tile source | fixed | fallback_fixed | heuristic |
| Pad impl | numpy_pad | numpy_pad | numpy_pad |
| Trial 1 avg (µs) | 363.2 | 365.8 | 368.0 |
| Trial 2 avg (µs) | 363.7 | 366.3 | 369.9 |
| Trial 3 avg (µs) | 648.3 | 365.3 | 361.8 |
| **Mean avg (µs)** | **FAIL** | **FAIL** | **FAIL** |
| Pass count | 0/3 | 0/3 | 0/3 |

**Winner: INCONCLUSIVE**

---

### 30×960×672  —  `q_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,672,960) | (32,672,960) | (32,672,960) |
| Tile | (16,32,32) | (16,32,32) | (16,32,32) |
| Tile source | fixed | fallback_fixed | heuristic |
| Pad impl | numpy_pad | numpy_pad | numpy_pad |
| Trial 1 avg (µs) | 601.3 | 431.3 | 523.0 |
| Trial 2 avg (µs) | 706.1 | 657.0 | 503.6 |
| Trial 3 avg (µs) | 467.0 | 567.0 | 455.8 |
| **Mean avg (µs)** | **FAIL** | **FAIL** | **FAIL** |
| Pass count | 0/3 | 0/3 | 0/3 |

**Winner: INCONCLUSIVE**

---

### 30×960×720  —  `q_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,768,960) | (32,768,960) | (32,768,960) |
| Tile | (16,32,32) | (16,32,32) | (16,32,32) |
| Tile source | fixed | fallback_fixed | heuristic |
| Pad impl | numpy_pad | numpy_pad | numpy_pad |
| Trial 1 avg (µs) | 300.6 | 311.2 | 258.2 |
| Trial 2 avg (µs) | 257.8 | 317.3 | 260.7 |
| Trial 3 avg (µs) | 260.8 | 261.8 | 408.6 |
| **Mean avg (µs)** | **FAIL** | **FAIL** | **FAIL** |
| Pass count | 0/3 | 0/3 | 0/3 |

**Winner: INCONCLUSIVE**

---

### 32×32×96  —  `action_in_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,96,32) | (32,96,32) | (32,96,32) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | fallback_fixed | heuristic |
| Pad impl | none | none | none |
| Trial 1 avg (µs) | 131.1 | 130.3 | 130.3 |
| Trial 2 avg (µs) | 128.7 | 129.5 | 126.3 |
| Trial 3 avg (µs) | 130.9 | 129.9 | 129.8 |
| **Mean avg (µs)** | **130.2** | **129.9** | **128.8** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-0.3µs (-0.3%)   Δ(AG vs B0)=-1.4µs (-1.1%)   Δ(AG vs B2)=-1.1µs (-0.8%)

---

### 32×32×192  —  `action_in_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,192,32) | (32,192,32) | (32,192,32) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | fallback_fixed | heuristic |
| Pad impl | none | none | none |
| Trial 1 avg (µs) | 123.9 | 124.6 | 124.7 |
| Trial 2 avg (µs) | 127.1 | 127.0 | 127.9 |
| Trial 3 avg (µs) | 126.3 | 125.7 | 126.2 |
| **Mean avg (µs)** | **125.8** | **125.8** | **126.2** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=+0.0µs (+0.0%)   Δ(AG vs B0)=+0.5µs (+0.4%)   Δ(AG vs B2)=+0.5µs (+0.4%)

---

### 32×32×288  —  `action_in_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,288,32) | (32,288,32) | (32,288,32) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | fallback_fixed | heuristic |
| Pad impl | none | none | none |
| Trial 1 avg (µs) | 133.6 | 131.5 | 129.8 |
| Trial 2 avg (µs) | 132.2 | 129.2 | 131.8 |
| Trial 3 avg (µs) | 129.8 | 132.1 | 130.0 |
| **Mean avg (µs)** | **131.9** | **130.9** | **130.5** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-0.9µs (-0.7%)   Δ(AG vs B0)=-1.4µs (-1.0%)   Δ(AG vs B2)=-0.4µs (-0.3%)

---

### 32×32×384  —  `action_in_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,384,32) | (32,384,32) | (32,384,32) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | fallback_fixed | heuristic |
| Pad impl | none | none | none |
| Trial 1 avg (µs) | 155.0 | 157.8 | 155.1 |
| Trial 2 avg (µs) | 157.7 | 157.6 | 156.7 |
| Trial 3 avg (µs) | 155.7 | 159.5 | 157.3 |
| **Mean avg (µs)** | **156.1** | **158.3** | **156.4** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=+2.2µs (+1.4%)   Δ(AG vs B0)=+0.2µs (+0.2%)   Δ(AG vs B2)=-1.9µs (-1.2%)

---

### 32×32×480  —  `action_in_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,480,32) | (32,480,32) | (32,480,32) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | fallback_fixed | heuristic |
| Pad impl | none | none | none |
| Trial 1 avg (µs) | 133.8 | 134.4 | 132.2 |
| Trial 2 avg (µs) | 134.5 | 136.0 | 135.0 |
| Trial 3 avg (µs) | 136.6 | 135.5 | 136.2 |
| **Mean avg (µs)** | **135.0** | **135.3** | **134.5** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=+0.4µs (+0.3%)   Δ(AG vs B0)=-0.5µs (-0.4%)   Δ(AG vs B2)=-0.9µs (-0.6%)

---

### 32×32×576  —  `action_in_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,576,32) | (32,576,32) | (32,576,32) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | fallback_fixed | heuristic |
| Pad impl | none | none | none |
| Trial 1 avg (µs) | 138.0 | 137.5 | 138.2 |
| Trial 2 avg (µs) | 139.1 | 140.1 | 141.2 |
| Trial 3 avg (µs) | 139.9 | 142.5 | 143.9 |
| **Mean avg (µs)** | **139.0** | **140.0** | **141.1** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=+1.0µs (+0.7%)   Δ(AG vs B0)=+2.1µs (+1.5%)   Δ(AG vs B2)=+1.1µs (+0.8%)

---

### 32×32×672  —  `action_in_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,672,32) | (32,672,32) | (32,672,32) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | fallback_fixed | heuristic |
| Pad impl | none | none | none |
| Trial 1 avg (µs) | 145.7 | 148.8 | 144.9 |
| Trial 2 avg (µs) | 144.7 | 147.2 | 146.9 |
| Trial 3 avg (µs) | 151.2 | 149.2 | 147.2 |
| **Mean avg (µs)** | **147.2** | **148.4** | **146.3** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=+1.2µs (+0.8%)   Δ(AG vs B0)=-0.8µs (-0.6%)   Δ(AG vs B2)=-2.0µs (-1.4%)

---

### 32×32×720  —  `action_in_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,768,32) | (32,768,32) | (32,768,32) |
| Tile | (32,32,32) | (32,64,32) | (32,64,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 196.3 | 156.9 | 156.7 |
| Trial 2 avg (µs) | 196.1 | 153.7 | 154.8 |
| Trial 3 avg (µs) | 193.2 | 155.3 | 156.6 |
| **Mean avg (µs)** | **195.2** | **155.3** | **156.0** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-39.9µs (-20.4%)   Δ(AG vs B0)=-39.2µs (-20.1%)   Δ(AG vs B2)=+0.7µs (+0.5%)

---

### 32×96×32  —  `action_out_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,32,128) | (32,32,128) | (32,32,128) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 121.9 | 120.0 | 124.0 |
| Trial 2 avg (µs) | 122.3 | 120.6 | 122.1 |
| Trial 3 avg (µs) | 121.0 | 122.3 | 120.2 |
| **Mean avg (µs)** | **121.7** | **121.0** | **122.1** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-0.8µs (-0.6%)   Δ(AG vs B0)=+0.4µs (+0.3%)   Δ(AG vs B2)=+1.1µs (+0.9%)

---

### 32×96×256  —  `gate_proj, up_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,256,128) | (32,256,128) | (32,256,128) |
| Tile | (32,32,32) | (32,64,32) | (32,64,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 140.9 | 132.5 | 133.4 |
| Trial 2 avg (µs) | 143.5 | 128.9 | 129.1 |
| Trial 3 avg (µs) | 145.2 | 131.4 | 132.1 |
| **Mean avg (µs)** | **143.2** | **130.9** | **131.5** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-12.2µs (-8.6%)   Δ(AG vs B0)=-11.7µs (-8.2%)   Δ(AG vs B2)=+0.6µs (+0.4%)

---

### 32×96×960  —  `o_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,960,128) | (32,960,128) | (32,960,128) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | fallback_fixed | heuristic |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 186.6 | 183.5 | 186.9 |
| Trial 2 avg (µs) | 182.7 | 186.7 | 186.2 |
| Trial 3 avg (µs) | 186.4 | 187.4 | 191.0 |
| **Mean avg (µs)** | **185.2** | **185.9** | **188.0** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=+0.7µs (+0.4%)   Δ(AG vs B0)=+2.8µs (+1.5%)   Δ(AG vs B2)=+2.2µs (+1.2%)

---

### 32×192×32  —  `action_out_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,32,256) | (32,32,256) | (32,32,256) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 128.6 | 126.3 | 127.4 |
| Trial 2 avg (µs) | 125.9 | 125.4 | 126.0 |
| Trial 3 avg (µs) | 128.2 | 128.6 | 128.3 |
| **Mean avg (µs)** | **127.6** | **126.8** | **127.2** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-0.8µs (-0.6%)   Δ(AG vs B0)=-0.3µs (-0.3%)   Δ(AG vs B2)=+0.5µs (+0.4%)

---

### 32×192×256  —  `gate_proj, up_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,256,256) | (32,256,256) | (32,256,256) |
| Tile | (32,32,32) | (32,64,32) | (32,64,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 144.9 | 139.8 | 136.4 |
| Trial 2 avg (µs) | 140.7 | 140.5 | 138.4 |
| Trial 3 avg (µs) | 143.0 | 138.1 | 141.2 |
| **Mean avg (µs)** | **142.9** | **139.5** | **138.7** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-3.4µs (-2.4%)   Δ(AG vs B0)=-4.2µs (-2.9%)   Δ(AG vs B2)=-0.8µs (-0.6%)

---

### 32×192×960  —  `o_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,960,256) | (32,960,256) | (32,960,256) |
| Tile | (16,32,32) | (16,32,32) | (16,32,32) |
| Tile source | fixed | fallback_fixed | heuristic |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | FAIL | FAIL | 245.1 |
| Trial 2 avg (µs) | FAIL | FAIL | FAIL |
| Trial 3 avg (µs) | FAIL | FAIL | FAIL |
| **Mean avg (µs)** | **FAIL** | **FAIL** | **245.1** |
| Pass count | 0/3 | 0/3 | 1/3 |

**Winner: INCONCLUSIVE**

---

### 32×256×96  —  `down_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,96,256) | (32,96,256) | (32,96,256) |
| Tile | (32,32,32) | (32,32,32) | (32,32,64) |
| Tile source | fixed | fallback_fixed | heuristic |
| Pad impl | none | none | none |
| Trial 1 avg (µs) | 126.8 | 134.6 | 135.0 |
| Trial 2 avg (µs) | 133.6 | 128.8 | 136.1 |
| Trial 3 avg (µs) | 133.3 | 130.7 | 133.3 |
| **Mean avg (µs)** | **131.2** | **131.3** | **134.8** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=+0.1µs (+0.1%)   Δ(AG vs B0)=+3.6µs (+2.7%)   Δ(AG vs B2)=+3.5µs (+2.6%)

---

### 32×256×192  —  `down_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,192,256) | (32,192,256) | (32,192,256) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | fallback_fixed | heuristic |
| Pad impl | none | none | none |
| Trial 1 avg (µs) | 133.1 | 133.5 | 134.5 |
| Trial 2 avg (µs) | 135.1 | 136.4 | 133.3 |
| Trial 3 avg (µs) | 133.4 | 134.5 | 134.1 |
| **Mean avg (µs)** | **133.8** | **134.8** | **134.0** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=+1.0µs (+0.7%)   Δ(AG vs B0)=+0.1µs (+0.1%)   Δ(AG vs B2)=-0.8µs (-0.6%)

---

### 32×256×288  —  `down_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,288,256) | (32,288,256) | (32,288,256) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | fallback_fixed | heuristic |
| Pad impl | none | none | none |
| Trial 1 avg (µs) | 154.8 | 155.9 | 162.3 |
| Trial 2 avg (µs) | 153.6 | 158.9 | 153.6 |
| Trial 3 avg (µs) | 161.2 | 153.9 | 156.7 |
| **Mean avg (µs)** | **156.6** | **156.2** | **157.5** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-0.3µs (-0.2%)   Δ(AG vs B0)=+0.9µs (+0.6%)   Δ(AG vs B2)=+1.3µs (+0.8%)

---

### 32×256×384  —  `down_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,384,256) | (32,384,256) | (32,384,256) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | fallback_fixed | heuristic |
| Pad impl | none | none | none |
| Trial 1 avg (µs) | 156.2 | 159.8 | 156.4 |
| Trial 2 avg (µs) | 156.2 | 157.2 | 158.0 |
| Trial 3 avg (µs) | 159.1 | 158.5 | 155.5 |
| **Mean avg (µs)** | **157.2** | **158.5** | **156.6** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=+1.3µs (+0.8%)   Δ(AG vs B0)=-0.5µs (-0.3%)   Δ(AG vs B2)=-1.9µs (-1.2%)

---

### 32×256×480  —  `down_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,480,256) | (32,480,256) | (32,480,256) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | fallback_fixed | heuristic |
| Pad impl | none | none | none |
| Trial 1 avg (µs) | 174.3 | 175.5 | 172.1 |
| Trial 2 avg (µs) | 174.9 | 173.7 | 173.5 |
| Trial 3 avg (µs) | 175.4 | 171.7 | 172.3 |
| **Mean avg (µs)** | **174.9** | **173.7** | **172.6** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-1.2µs (-0.7%)   Δ(AG vs B0)=-2.2µs (-1.3%)   Δ(AG vs B2)=-1.0µs (-0.6%)

---

### 32×256×576  —  `down_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,576,256) | (32,576,256) | (32,576,256) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | fallback_fixed | heuristic |
| Pad impl | none | none | none |
| Trial 1 avg (µs) | 191.1 | 198.5 | 191.9 |
| Trial 2 avg (µs) | 187.4 | 191.3 | 189.5 |
| Trial 3 avg (µs) | 197.5 | 193.6 | 188.1 |
| **Mean avg (µs)** | **192.0** | **194.5** | **189.8** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=+2.5µs (+1.3%)   Δ(AG vs B0)=-2.1µs (-1.1%)   Δ(AG vs B2)=-4.6µs (-2.4%)

---

### 32×256×672  —  `down_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,672,256) | (32,672,256) | (32,672,256) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | fallback_fixed | heuristic |
| Pad impl | none | none | none |
| Trial 1 avg (µs) | 204.4 | 206.4 | 203.5 |
| Trial 2 avg (µs) | 216.0 | 204.9 | 214.1 |
| Trial 3 avg (µs) | 207.0 | 205.8 | 205.6 |
| **Mean avg (µs)** | **209.1** | **205.7** | **207.8** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-3.4µs (-1.6%)   Δ(AG vs B0)=-1.4µs (-0.7%)   Δ(AG vs B2)=+2.0µs (+1.0%)

---

### 32×256×720  —  `down_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,768,256) | (32,768,256) | (32,768,256) |
| Tile | (32,32,32) | (32,64,32) | (32,64,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 197.3 | 161.9 | 162.4 |
| Trial 2 avg (µs) | 195.6 | 159.0 | 158.8 |
| Trial 3 avg (µs) | 195.4 | 160.8 | 164.9 |
| **Mean avg (µs)** | **196.1** | **160.6** | **162.1** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-35.5µs (-18.1%)   Δ(AG vs B0)=-34.1µs (-17.4%)   Δ(AG vs B2)=+1.5µs (+0.9%)

---

### 32×288×32  —  `action_out_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,32,512) | (32,32,512) | (32,32,512) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 132.7 | 130.1 | 132.5 |
| Trial 2 avg (µs) | 133.2 | 132.0 | 130.3 |
| Trial 3 avg (µs) | 132.1 | 130.6 | 134.5 |
| **Mean avg (µs)** | **132.7** | **130.9** | **132.4** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-1.8µs (-1.3%)   Δ(AG vs B0)=-0.2µs (-0.2%)   Δ(AG vs B2)=+1.5µs (+1.2%)

---

### 32×288×256  —  `gate_proj, up_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,256,512) | (32,256,512) | (32,256,512) |
| Tile | (32,32,32) | (32,64,32) | (32,64,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 154.7 | 148.8 | 147.8 |
| Trial 2 avg (µs) | 156.4 | 149.0 | 150.3 |
| Trial 3 avg (µs) | 154.6 | 149.4 | 148.8 |
| **Mean avg (µs)** | **155.3** | **149.1** | **148.9** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-6.2µs (-4.0%)   Δ(AG vs B0)=-6.3µs (-4.1%)   Δ(AG vs B2)=-0.1µs (-0.1%)

---

### 32×288×960  —  `o_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,960,512) | (32,960,512) | (32,960,512) |
| Tile | (16,32,32) | (16,32,32) | (16,32,32) |
| Tile source | fixed | fallback_fixed | heuristic |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | FAIL | FAIL | 441.4 |
| Trial 2 avg (µs) | 352.6 | 354.0 | FAIL |
| Trial 3 avg (µs) | FAIL | FAIL | 353.5 |
| **Mean avg (µs)** | **352.6** | **354.0** | **397.4** |
| Pass count | 1/3 | 1/3 | 2/3 |

**Winner: TIE**
Δ(B2 vs B0)=+1.4µs (+0.4%)   Δ(AG vs B0)=+44.8µs (+12.7%)   Δ(AG vs B2)=+43.4µs (+12.3%)

---

### 32×384×32  —  `action_out_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,32,512) | (32,32,512) | (32,32,512) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 135.0 | 131.8 | 131.3 |
| Trial 2 avg (µs) | 135.2 | 132.3 | 134.2 |
| Trial 3 avg (µs) | 128.6 | 127.2 | 129.8 |
| **Mean avg (µs)** | **132.9** | **130.4** | **131.8** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-2.5µs (-1.9%)   Δ(AG vs B0)=-1.1µs (-0.9%)   Δ(AG vs B2)=+1.4µs (+1.0%)

---

### 32×384×256  —  `gate_proj, up_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,256,512) | (32,256,512) | (32,256,512) |
| Tile | (32,32,32) | (32,64,32) | (32,64,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 152.8 | 148.3 | 150.0 |
| Trial 2 avg (µs) | 154.9 | 150.1 | 147.3 |
| Trial 3 avg (µs) | 153.6 | 148.6 | 149.6 |
| **Mean avg (µs)** | **153.7** | **149.0** | **149.0** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-4.7µs (-3.1%)   Δ(AG vs B0)=-4.8µs (-3.1%)   Δ(AG vs B2)=-0.1µs (-0.0%)

---

### 32×384×960  —  `o_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,960,512) | (32,960,512) | (32,960,512) |
| Tile | (16,32,32) | (16,32,32) | (16,32,32) |
| Tile source | fixed | fallback_fixed | heuristic |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 351.5 | FAIL | 355.4 |
| Trial 2 avg (µs) | FAIL | FAIL | FAIL |
| Trial 3 avg (µs) | FAIL | 353.2 | FAIL |
| **Mean avg (µs)** | **351.4** | **353.2** | **355.4** |
| Pass count | 1/3 | 1/3 | 1/3 |

**Winner: TIE**
Δ(B2 vs B0)=+1.7µs (+0.5%)   Δ(AG vs B0)=+3.9µs (+1.1%)   Δ(AG vs B2)=+2.2µs (+0.6%)

---

### 32×480×32  —  `action_out_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,32,512) | (32,32,512) | (32,32,512) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 127.4 | 127.8 | 133.6 |
| Trial 2 avg (µs) | 127.4 | 131.6 | 133.7 |
| Trial 3 avg (µs) | 132.7 | 127.0 | 132.8 |
| **Mean avg (µs)** | **129.2** | **128.8** | **133.4** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-0.4µs (-0.3%)   Δ(AG vs B0)=+4.2µs (+3.2%)   Δ(AG vs B2)=+4.6µs (+3.6%)

---

### 32×480×256  —  `gate_proj, up_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,256,512) | (32,256,512) | (32,256,512) |
| Tile | (32,32,32) | (32,64,32) | (32,64,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 153.8 | 148.8 | 147.1 |
| Trial 2 avg (µs) | 157.2 | 151.0 | 155.2 |
| Trial 3 avg (µs) | 153.7 | 149.7 | 150.9 |
| **Mean avg (µs)** | **154.9** | **149.8** | **151.0** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-5.1µs (-3.3%)   Δ(AG vs B0)=-3.9µs (-2.5%)   Δ(AG vs B2)=+1.2µs (+0.8%)

---

### 32×480×960  —  `o_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,960,512) | (32,960,512) | (32,960,512) |
| Tile | (16,32,32) | (16,32,32) | (16,32,32) |
| Tile source | fixed | fallback_fixed | heuristic |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | FAIL | FAIL | FAIL |
| Trial 2 avg (µs) | FAIL | 352.1 | 370.4 |
| Trial 3 avg (µs) | FAIL | 603.2 | 352.0 |
| **Mean avg (µs)** | **FAIL** | **477.6** | **361.2** |
| Pass count | 0/3 | 2/3 | 2/3 |

**Winner: INCONCLUSIVE**
Δ(AG vs B2)=-116.4µs (-24.4%)

---

### 32×576×32  —  `action_out_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,32,768) | (32,32,768) | (32,32,768) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 138.3 | 137.0 | 135.7 |
| Trial 2 avg (µs) | 139.4 | 137.8 | 135.8 |
| Trial 3 avg (µs) | 133.8 | 138.9 | 137.1 |
| **Mean avg (µs)** | **137.2** | **137.9** | **136.2** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=+0.7µs (+0.5%)   Δ(AG vs B0)=-1.0µs (-0.7%)   Δ(AG vs B2)=-1.7µs (-1.2%)

---

### 32×576×256  —  `gate_proj, up_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,256,768) | (32,256,768) | (32,256,768) |
| Tile | (32,32,32) | (32,64,32) | (32,64,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 164.3 | 160.5 | 158.2 |
| Trial 2 avg (µs) | 162.7 | 156.4 | 157.3 |
| Trial 3 avg (µs) | 162.9 | 158.3 | 157.4 |
| **Mean avg (µs)** | **163.3** | **158.4** | **157.6** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-4.9µs (-3.0%)   Δ(AG vs B0)=-5.7µs (-3.5%)   Δ(AG vs B2)=-0.8µs (-0.5%)

---

### 32×576×960  —  `o_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,960,768) | (32,960,768) | (32,960,768) |
| Tile | (16,32,32) | (16,32,32) | (16,32,32) |
| Tile source | fixed | fallback_fixed | heuristic |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | FAIL | 467.8 | 528.4 |
| Trial 2 avg (µs) | FAIL | 554.4 | FAIL |
| Trial 3 avg (µs) | 459.8 | 463.5 | 459.8 |
| **Mean avg (µs)** | **459.8** | **495.2** | **494.1** |
| Pass count | 1/3 | 3/3 | 2/3 |

**Winner: BASELINE 0**
Δ(B2 vs B0)=+35.5µs (+7.7%)   Δ(AG vs B0)=+34.4µs (+7.5%)   Δ(AG vs B2)=-1.1µs (-0.2%)

---

### 32×672×32  —  `action_out_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,32,768) | (32,32,768) | (32,32,768) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 136.4 | 135.2 | 137.1 |
| Trial 2 avg (µs) | 138.3 | 137.1 | 136.6 |
| Trial 3 avg (µs) | 135.9 | 134.4 | 139.2 |
| **Mean avg (µs)** | **136.9** | **135.6** | **137.6** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-1.3µs (-0.9%)   Δ(AG vs B0)=+0.8µs (+0.6%)   Δ(AG vs B2)=+2.1µs (+1.5%)

---

### 32×672×256  —  `gate_proj, up_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,256,768) | (32,256,768) | (32,256,768) |
| Tile | (32,32,32) | (32,64,32) | (32,64,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 162.3 | 156.4 | 157.4 |
| Trial 2 avg (µs) | 164.5 | 156.6 | 159.0 |
| Trial 3 avg (µs) | 165.5 | 159.7 | 159.3 |
| **Mean avg (µs)** | **164.1** | **157.6** | **158.6** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-6.6µs (-4.0%)   Δ(AG vs B0)=-5.5µs (-3.4%)   Δ(AG vs B2)=+1.0µs (+0.6%)

---

### 32×672×960  —  `o_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,960,768) | (32,960,768) | (32,960,768) |
| Tile | (16,32,32) | (16,32,32) | (16,32,32) |
| Tile source | fixed | fallback_fixed | heuristic |
| Pad impl | numpy_pad | numpy_pad | numpy_pad |
| Trial 1 avg (µs) | FAIL | FAIL | FAIL |
| Trial 2 avg (µs) | 466.9 | FAIL | FAIL |
| Trial 3 avg (µs) | 571.3 | 468.8 | 466.2 |
| **Mean avg (µs)** | **519.1** | **468.8** | **466.2** |
| Pass count | 2/3 | 1/3 | 1/3 |

**Winner: TIE**
Δ(B2 vs B0)=-50.4µs (-9.7%)   Δ(AG vs B0)=-52.9µs (-10.2%)   Δ(AG vs B2)=-2.5µs (-0.5%)

---

### 32×720×32  —  `action_out_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,32,768) | (32,32,768) | (32,32,768) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 138.0 | 135.5 | 137.7 |
| Trial 2 avg (µs) | 135.9 | 137.0 | 138.4 |
| Trial 3 avg (µs) | 135.8 | 138.1 | 139.0 |
| **Mean avg (µs)** | **136.6** | **136.9** | **138.3** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=+0.3µs (+0.2%)   Δ(AG vs B0)=+1.8µs (+1.3%)   Δ(AG vs B2)=+1.5µs (+1.1%)

---

### 32×720×256  —  `gate_proj, up_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,256,768) | (32,256,768) | (32,256,768) |
| Tile | (32,32,32) | (32,64,32) | (32,64,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 166.3 | 156.7 | 161.0 |
| Trial 2 avg (µs) | 164.6 | 162.4 | 156.5 |
| Trial 3 avg (µs) | 161.8 | 158.8 | 161.6 |
| **Mean avg (µs)** | **164.2** | **159.3** | **159.7** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-5.0µs (-3.0%)   Δ(AG vs B0)=-4.5µs (-2.8%)   Δ(AG vs B2)=+0.4µs (+0.3%)

---

### 32×720×960  —  `o_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,960,768) | (32,960,768) | (32,960,768) |
| Tile | (16,32,32) | (16,32,32) | (16,32,32) |
| Tile source | fixed | fallback_fixed | heuristic |
| Pad impl | numpy_pad | numpy_pad | numpy_pad |
| Trial 1 avg (µs) | FAIL | FAIL | 459.2 |
| Trial 2 avg (µs) | FAIL | 464.5 | FAIL |
| Trial 3 avg (µs) | 472.7 | 475.9 | 473.5 |
| **Mean avg (µs)** | **472.7** | **470.2** | **466.3** |
| Pass count | 1/3 | 2/3 | 2/3 |

**Winner: TIE**
Δ(B2 vs B0)=-2.6µs (-0.5%)   Δ(AG vs B0)=-6.4µs (-1.4%)   Δ(AG vs B2)=-3.8µs (-0.8%)

---

### 32×960×96  —  `q_proj`

**Pre-screened skip**: bf16_accuracy_small_MN_large_K

---

### 32×960×192  —  `q_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,192,960) | (32,192,960) | (32,192,960) |
| Tile | (16,32,32) | (16,32,32) | (16,32,32) |
| Tile source | fixed | fallback_fixed | heuristic |
| Pad impl | none | none | none |
| Trial 1 avg (µs) | 173.4 | 173.6 | 176.7 |
| Trial 2 avg (µs) | 173.6 | 169.7 | 174.1 |
| Trial 3 avg (µs) | 172.1 | 174.1 | 174.2 |
| **Mean avg (µs)** | **FAIL** | **FAIL** | **FAIL** |
| Pass count | 0/3 | 0/3 | 0/3 |

**Winner: INCONCLUSIVE**

---

### 32×960×288  —  `q_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,288,960) | (32,288,960) | (32,288,960) |
| Tile | (16,32,32) | (16,32,32) | (16,32,32) |
| Tile source | fixed | fallback_fixed | heuristic |
| Pad impl | none | none | none |
| Trial 1 avg (µs) | 240.5 | 248.0 | 241.2 |
| Trial 2 avg (µs) | 244.2 | 241.3 | 250.0 |
| Trial 3 avg (µs) | 248.6 | 240.2 | 347.5 |
| **Mean avg (µs)** | **FAIL** | **FAIL** | **FAIL** |
| Pass count | 0/3 | 0/3 | 0/3 |

**Winner: INCONCLUSIVE**

---

### 32×960×384  —  `q_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,384,960) | (32,384,960) | (32,384,960) |
| Tile | (16,32,32) | (16,32,32) | (16,32,32) |
| Tile source | fixed | fallback_fixed | heuristic |
| Pad impl | none | none | none |
| Trial 1 avg (µs) | 198.8 | 197.2 | 198.4 |
| Trial 2 avg (µs) | 197.6 | 197.7 | 198.3 |
| Trial 3 avg (µs) | 199.1 | 197.8 | 199.2 |
| **Mean avg (µs)** | **FAIL** | **FAIL** | **FAIL** |
| Pass count | 0/3 | 0/3 | 0/3 |

**Winner: INCONCLUSIVE**

---

### 32×960×480  —  `q_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,480,960) | (32,480,960) | (32,480,960) |
| Tile | (16,32,32) | (16,32,32) | (16,32,32) |
| Tile source | fixed | fallback_fixed | heuristic |
| Pad impl | none | none | none |
| Trial 1 avg (µs) | 301.2 | 301.5 | 304.9 |
| Trial 2 avg (µs) | 301.5 | 306.1 | 304.9 |
| Trial 3 avg (µs) | 395.3 | 301.2 | 303.5 |
| **Mean avg (µs)** | **FAIL** | **FAIL** | **FAIL** |
| Pass count | 0/3 | 0/3 | 0/3 |

**Winner: INCONCLUSIVE**

---

### 32×960×576  —  `q_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,576,960) | (32,576,960) | (32,576,960) |
| Tile | (16,32,32) | (16,32,32) | (16,32,32) |
| Tile source | fixed | fallback_fixed | heuristic |
| Pad impl | none | none | none |
| Trial 1 avg (µs) | 371.9 | 366.9 | 491.0 |
| Trial 2 avg (µs) | 367.3 | 367.8 | 369.1 |
| Trial 3 avg (µs) | 370.6 | 367.9 | 368.5 |
| **Mean avg (µs)** | **FAIL** | **FAIL** | **FAIL** |
| Pass count | 0/3 | 0/3 | 0/3 |

**Winner: INCONCLUSIVE**

---

### 32×960×672  —  `q_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,672,960) | (32,672,960) | (32,672,960) |
| Tile | (16,32,32) | (16,32,32) | (16,32,32) |
| Tile source | fixed | fallback_fixed | heuristic |
| Pad impl | none | none | none |
| Trial 1 avg (µs) | 428.1 | 447.9 | 565.3 |
| Trial 2 avg (µs) | 427.9 | 427.3 | 560.5 |
| Trial 3 avg (µs) | 426.1 | 618.9 | 550.8 |
| **Mean avg (µs)** | **FAIL** | **FAIL** | **FAIL** |
| Pass count | 0/3 | 0/3 | 0/3 |

**Winner: INCONCLUSIVE**

---

### 32×960×720  —  `q_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (32,768,960) | (32,768,960) | (32,768,960) |
| Tile | (16,32,32) | (16,32,32) | (16,32,32) |
| Tile source | fixed | fallback_fixed | heuristic |
| Pad impl | numpy_pad | numpy_pad | numpy_pad |
| Trial 1 avg (µs) | 260.0 | 260.1 | 259.2 |
| Trial 2 avg (µs) | 259.9 | 261.8 | 262.2 |
| Trial 3 avg (µs) | 265.5 | 265.1 | 399.3 |
| **Mean avg (µs)** | **FAIL** | **FAIL** | **FAIL** |
| Pass count | 0/3 | 0/3 | 0/3 |

**Winner: INCONCLUSIVE**

---

### 50×32×192  —  `action_in_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,192,32) | (64,192,32) | (64,192,32) |
| Tile | (32,32,32) | (16,64,16) | (16,64,16) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 126.6 | 131.9 | 130.6 |
| Trial 2 avg (µs) | 128.6 | 132.3 | 132.9 |
| Trial 3 avg (µs) | 128.5 | 130.0 | 129.7 |
| **Mean avg (µs)** | **127.9** | **131.4** | **131.1** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: BASELINE 0**
Δ(B2 vs B0)=+3.5µs (+2.8%)   Δ(AG vs B0)=+3.2µs (+2.5%)   Δ(AG vs B2)=-0.3µs (-0.3%)

---

### 50×32×288  —  `action_in_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,288,32) | (64,288,32) | (64,288,32) |
| Tile | (32,32,32) | (16,32,16) | (16,32,16) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 135.8 | 157.7 | 155.5 |
| Trial 2 avg (µs) | 137.9 | 156.2 | 157.1 |
| Trial 3 avg (µs) | 135.4 | 156.8 | 154.4 |
| **Mean avg (µs)** | **136.3** | **156.9** | **155.6** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: BASELINE 0**
Δ(B2 vs B0)=+20.5µs (+15.1%)   Δ(AG vs B0)=+19.3µs (+14.2%)   Δ(AG vs B2)=-1.2µs (-0.8%)

---

### 50×32×384  —  `action_in_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,384,32) | (64,384,32) | (64,384,32) |
| Tile | (32,32,32) | (16,128,16) | (16,128,16) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 150.8 | 133.7 | 134.0 |
| Trial 2 avg (µs) | 150.5 | 135.5 | 133.1 |
| Trial 3 avg (µs) | 152.3 | 133.9 | 131.1 |
| **Mean avg (µs)** | **151.2** | **134.4** | **132.8** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-16.9µs (-11.1%)   Δ(AG vs B0)=-18.5µs (-12.2%)   Δ(AG vs B2)=-1.6µs (-1.2%)

---

### 50×32×480  —  `action_in_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,480,32) | (64,480,32) | (64,480,32) |
| Tile | (32,32,32) | (16,32,16) | (16,32,16) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 141.1 | 181.8 | 183.3 |
| Trial 2 avg (µs) | 144.4 | 181.7 | 185.3 |
| Trial 3 avg (µs) | 141.2 | 185.7 | 185.5 |
| **Mean avg (µs)** | **142.2** | **183.1** | **184.7** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: BASELINE 0**
Δ(B2 vs B0)=+40.8µs (+28.7%)   Δ(AG vs B0)=+42.5µs (+29.9%)   Δ(AG vs B2)=+1.7µs (+0.9%)

---

### 50×32×576  —  `action_in_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,576,32) | (64,576,32) | (64,576,32) |
| Tile | (32,32,32) | (16,64,16) | (16,64,16) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 147.7 | 156.7 | 154.0 |
| Trial 2 avg (µs) | 148.1 | 156.0 | 155.2 |
| Trial 3 avg (µs) | 146.0 | 158.4 | 155.4 |
| **Mean avg (µs)** | **147.3** | **157.0** | **154.9** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: BASELINE 0**
Δ(B2 vs B0)=+9.7µs (+6.6%)   Δ(AG vs B0)=+7.6µs (+5.1%)   Δ(AG vs B2)=-2.1µs (-1.4%)

---

### 50×32×672  —  `action_in_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,672,32) | (64,672,32) | (64,672,32) |
| Tile | (32,32,32) | (16,32,32) | (16,32,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 163.9 | 208.6 | 205.5 |
| Trial 2 avg (µs) | 162.7 | 209.6 | 210.5 |
| Trial 3 avg (µs) | 157.4 | 210.3 | 207.2 |
| **Mean avg (µs)** | **161.3** | **209.5** | **207.8** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: BASELINE 0**
Δ(B2 vs B0)=+48.2µs (+29.9%)   Δ(AG vs B0)=+46.4µs (+28.8%)   Δ(AG vs B2)=-1.8µs (-0.8%)

---

### 50×32×720  —  `action_in_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,768,32) | (64,768,32) | (64,768,32) |
| Tile | (32,32,32) | (16,256,16) | (16,256,16) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 182.8 | 150.1 | 145.8 |
| Trial 2 avg (µs) | 186.1 | 153.9 | 151.3 |
| Trial 3 avg (µs) | 184.8 | 151.3 | 153.8 |
| **Mean avg (µs)** | **184.6** | **151.8** | **150.3** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-32.8µs (-17.8%)   Δ(AG vs B0)=-34.3µs (-18.6%)   Δ(AG vs B2)=-1.5µs (-1.0%)

---

### 50×192×32  —  `action_out_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,32,256) | (64,32,256) | (64,32,256) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 126.3 | 123.8 | 125.6 |
| Trial 2 avg (µs) | 125.5 | 125.9 | 125.7 |
| Trial 3 avg (µs) | 126.0 | 123.4 | 124.3 |
| **Mean avg (µs)** | **126.0** | **124.4** | **125.2** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-1.6µs (-1.2%)   Δ(AG vs B0)=-0.7µs (-0.6%)   Δ(AG vs B2)=+0.8µs (+0.7%)

---

### 50×192×256  —  `gate_proj, up_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,256,256) | (64,256,256) | (64,256,256) |
| Tile | (64,64,64) | (16,64,32) | (16,64,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 149.5 | 134.2 | 138.3 |
| Trial 2 avg (µs) | 154.8 | 135.3 | 138.1 |
| Trial 3 avg (µs) | 153.9 | 139.2 | 139.7 |
| **Mean avg (µs)** | **152.7** | **136.2** | **138.7** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-16.5µs (-10.8%)   Δ(AG vs B0)=-14.0µs (-9.2%)   Δ(AG vs B2)=+2.5µs (+1.8%)

---

### 50×192×960  —  `o_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,960,256) | (64,960,256) | (64,960,256) |
| Tile | (64,64,64) | (16,32,16) | (16,32,16) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 251.4 | 260.2 | 382.3 |
| Trial 2 avg (µs) | 234.2 | 375.3 | 273.6 |
| Trial 3 avg (µs) | 250.6 | 260.6 | 350.3 |
| **Mean avg (µs)** | **245.4** | **298.7** | **335.4** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: BASELINE 0**
Δ(B2 vs B0)=+53.3µs (+21.7%)   Δ(AG vs B0)=+90.0µs (+36.7%)   Δ(AG vs B2)=+36.7µs (+12.3%)

---

### 50×256×192  —  `down_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,192,256) | (64,192,256) | (64,192,256) |
| Tile | (64,64,64) | (16,64,32) | (16,64,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 152.3 | 154.6 | 153.7 |
| Trial 2 avg (µs) | 147.4 | 153.6 | 153.3 |
| Trial 3 avg (µs) | 154.0 | 156.0 | 153.5 |
| **Mean avg (µs)** | **151.2** | **154.7** | **153.5** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=+3.5µs (+2.3%)   Δ(AG vs B0)=+2.3µs (+1.5%)   Δ(AG vs B2)=-1.2µs (-0.8%)

---

### 50×256×288  —  `down_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,288,256) | (64,288,256) | (64,288,256) |
| Tile | (32,32,32) | (16,32,32) | (16,32,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 155.7 | 172.6 | 170.6 |
| Trial 2 avg (µs) | 154.5 | 174.3 | 173.6 |
| Trial 3 avg (µs) | 156.6 | 174.3 | 170.5 |
| **Mean avg (µs)** | **155.6** | **173.7** | **171.6** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: BASELINE 0**
Δ(B2 vs B0)=+18.1µs (+11.7%)   Δ(AG vs B0)=+16.0µs (+10.3%)   Δ(AG vs B2)=-2.2µs (-1.2%)

---

### 50×256×384  —  `down_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,384,256) | (64,384,256) | (64,384,256) |
| Tile | (64,64,64) | (16,32,128) | (16,32,128) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 156.9 | 169.1 | 168.9 |
| Trial 2 avg (µs) | 158.0 | 167.9 | 168.9 |
| Trial 3 avg (µs) | 156.1 | 165.7 | 167.1 |
| **Mean avg (µs)** | **157.0** | **167.6** | **168.3** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: BASELINE 0**
Δ(B2 vs B0)=+10.6µs (+6.7%)   Δ(AG vs B0)=+11.3µs (+7.2%)   Δ(AG vs B2)=+0.8µs (+0.5%)

---

### 50×256×480  —  `down_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,480,256) | (64,480,256) | (64,480,256) |
| Tile | (32,32,32) | (16,32,32) | (16,32,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 171.3 | 202.8 | 200.0 |
| Trial 2 avg (µs) | 176.5 | 202.7 | 202.7 |
| Trial 3 avg (µs) | 175.5 | 203.8 | 202.2 |
| **Mean avg (µs)** | **174.4** | **203.1** | **201.6** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: BASELINE 0**
Δ(B2 vs B0)=+28.7µs (+16.4%)   Δ(AG vs B0)=+27.2µs (+15.6%)   Δ(AG vs B2)=-1.5µs (-0.7%)

---

### 50×256×576  —  `down_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,576,256) | (64,576,256) | (64,576,256) |
| Tile | (64,64,64) | (16,64,32) | (16,64,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 195.9 | 208.9 | 274.7 |
| Trial 2 avg (µs) | 201.1 | 214.3 | 332.0 |
| Trial 3 avg (µs) | 192.6 | 210.5 | 210.4 |
| **Mean avg (µs)** | **196.6** | **211.2** | **272.4** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: BASELINE 0**
Δ(B2 vs B0)=+14.7µs (+7.5%)   Δ(AG vs B0)=+75.8µs (+38.6%)   Δ(AG vs B2)=+61.1µs (+28.9%)

---

### 50×256×672  —  `down_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,672,256) | (64,672,256) | (64,672,256) |
| Tile | (32,32,32) | (16,32,32) | (16,32,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 208.7 | 227.2 | 227.1 |
| Trial 2 avg (µs) | 253.4 | 222.8 | 223.8 |
| Trial 3 avg (µs) | 204.3 | 259.6 | 224.3 |
| **Mean avg (µs)** | **222.1** | **236.5** | **225.1** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=+14.4µs (+6.5%)   Δ(AG vs B0)=+2.9µs (+1.3%)   Δ(AG vs B2)=-11.5µs (-4.9%)

---

### 50×256×720  —  `down_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,768,256) | (64,768,256) | (64,768,256) |
| Tile | (64,64,64) | (16,64,128) | (16,64,128) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 182.7 | 184.2 | 175.8 |
| Trial 2 avg (µs) | 179.8 | 181.3 | 177.3 |
| Trial 3 avg (µs) | 183.6 | 181.3 | 178.1 |
| **Mean avg (µs)** | **182.0** | **182.3** | **177.1** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: AGENTIC**
Δ(B2 vs B0)=+0.2µs (+0.1%)   Δ(AG vs B0)=-5.0µs (-2.7%)   Δ(AG vs B2)=-5.2µs (-2.9%)

---

### 50×288×32  —  `action_out_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,32,512) | (64,32,512) | (64,32,512) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 131.2 | 131.5 | 129.0 |
| Trial 2 avg (µs) | 131.7 | 130.5 | 131.5 |
| Trial 3 avg (µs) | 128.4 | 132.7 | 133.6 |
| **Mean avg (µs)** | **130.5** | **131.5** | **131.4** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=+1.1µs (+0.8%)   Δ(AG vs B0)=+0.9µs (+0.7%)   Δ(AG vs B2)=-0.2µs (-0.1%)

---

### 50×288×256  —  `gate_proj, up_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,256,512) | (64,256,512) | (64,256,512) |
| Tile | (64,64,64) | (16,64,32) | (16,64,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 159.2 | 148.8 | 149.6 |
| Trial 2 avg (µs) | 155.6 | 149.6 | 146.8 |
| Trial 3 avg (µs) | 159.6 | 151.0 | 147.6 |
| **Mean avg (µs)** | **158.1** | **149.8** | **148.0** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-8.3µs (-5.3%)   Δ(AG vs B0)=-10.1µs (-6.4%)   Δ(AG vs B2)=-1.8µs (-1.2%)

---

### 50×288×960  —  `o_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,960,512) | (64,960,512) | (64,960,512) |
| Tile | (64,64,64) | (16,16,128) | (16,16,128) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 324.4 | 377.2 | 375.7 |
| Trial 2 avg (µs) | 456.7 | 373.7 | 708.3 |
| Trial 3 avg (µs) | 329.3 | 612.9 | 374.1 |
| **Mean avg (µs)** | **370.1** | **454.6** | **486.0** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: BASELINE 0**
Δ(B2 vs B0)=+84.5µs (+22.8%)   Δ(AG vs B0)=+115.9µs (+31.3%)   Δ(AG vs B2)=+31.4µs (+6.9%)

---

### 50×384×32  —  `action_out_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,32,512) | (64,32,512) | (64,32,512) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 135.0 | 135.3 | 130.2 |
| Trial 2 avg (µs) | 131.3 | 137.4 | 134.2 |
| Trial 3 avg (µs) | 136.2 | 132.2 | 133.0 |
| **Mean avg (µs)** | **134.2** | **134.9** | **132.4** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=+0.8µs (+0.6%)   Δ(AG vs B0)=-1.7µs (-1.3%)   Δ(AG vs B2)=-2.5µs (-1.9%)

---

### 50×384×256  —  `gate_proj, up_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,256,512) | (64,256,512) | (64,256,512) |
| Tile | (64,64,64) | (16,64,32) | (16,64,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 160.2 | 150.7 | 150.8 |
| Trial 2 avg (µs) | 157.9 | 149.9 | 142.9 |
| Trial 3 avg (µs) | 163.1 | 150.7 | 149.1 |
| **Mean avg (µs)** | **160.4** | **150.4** | **147.6** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-9.9µs (-6.2%)   Δ(AG vs B0)=-12.8µs (-8.0%)   Δ(AG vs B2)=-2.8µs (-1.9%)

---

### 50×384×960  —  `o_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,960,512) | (64,960,512) | (64,960,512) |
| Tile | (64,64,64) | (16,16,128) | (16,16,128) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 324.5 | 654.5 | 758.6 |
| Trial 2 avg (µs) | 326.5 | 378.0 | 379.9 |
| Trial 3 avg (µs) | 324.1 | 620.3 | 754.9 |
| **Mean avg (µs)** | **325.1** | **550.9** | **631.1** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: BASELINE 0**
Δ(B2 vs B0)=+225.9µs (+69.5%)   Δ(AG vs B0)=+306.1µs (+94.2%)   Δ(AG vs B2)=+80.2µs (+14.6%)

---

### 50×480×32  —  `action_out_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,32,512) | (64,32,512) | (64,32,512) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 134.4 | 134.3 | 132.0 |
| Trial 2 avg (µs) | 130.5 | 130.5 | 132.4 |
| Trial 3 avg (µs) | 136.2 | 133.8 | 132.0 |
| **Mean avg (µs)** | **133.7** | **132.8** | **132.1** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-0.9µs (-0.6%)   Δ(AG vs B0)=-1.6µs (-1.2%)   Δ(AG vs B2)=-0.7µs (-0.5%)

---

### 50×480×256  —  `gate_proj, up_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,256,512) | (64,256,512) | (64,256,512) |
| Tile | (64,64,64) | (16,64,32) | (16,64,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 159.8 | 151.6 | 146.2 |
| Trial 2 avg (µs) | 160.8 | 153.7 | 150.8 |
| Trial 3 avg (µs) | 158.8 | 153.3 | 151.1 |
| **Mean avg (µs)** | **159.8** | **152.8** | **149.4** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: AGENTIC**
Δ(B2 vs B0)=-7.0µs (-4.4%)   Δ(AG vs B0)=-10.4µs (-6.5%)   Δ(AG vs B2)=-3.5µs (-2.3%)

---

### 50×480×960  —  `o_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,960,512) | (64,960,512) | (64,960,512) |
| Tile | (64,64,64) | (16,16,128) | (16,16,128) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 330.4 | 380.3 | 377.3 |
| Trial 2 avg (µs) | 326.7 | 377.2 | 522.5 |
| Trial 3 avg (µs) | 326.3 | 525.1 | 447.0 |
| **Mean avg (µs)** | **327.8** | **427.6** | **448.9** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: BASELINE 0**
Δ(B2 vs B0)=+99.8µs (+30.4%)   Δ(AG vs B0)=+121.1µs (+37.0%)   Δ(AG vs B2)=+21.4µs (+5.0%)

---

### 50×576×32  —  `action_out_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,32,768) | (64,32,768) | (64,32,768) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 141.5 | 139.4 | 140.9 |
| Trial 2 avg (µs) | 141.0 | 140.8 | 139.7 |
| Trial 3 avg (µs) | 143.3 | 141.3 | 138.7 |
| **Mean avg (µs)** | **141.9** | **140.5** | **139.8** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-1.4µs (-1.0%)   Δ(AG vs B0)=-2.1µs (-1.5%)   Δ(AG vs B2)=-0.7µs (-0.5%)

---

### 50×576×256  —  `gate_proj, up_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,256,768) | (64,256,768) | (64,256,768) |
| Tile | (64,64,64) | (16,64,32) | (16,64,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 167.3 | 158.6 | 159.6 |
| Trial 2 avg (µs) | 170.0 | 158.8 | 161.3 |
| Trial 3 avg (µs) | 170.4 | 158.9 | 161.2 |
| **Mean avg (µs)** | **169.2** | **158.8** | **160.7** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-10.5µs (-6.2%)   Δ(AG vs B0)=-8.5µs (-5.0%)   Δ(AG vs B2)=+1.9µs (+1.2%)

---

### 50×576×960  —  `o_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,960,768) | (64,960,768) | (64,960,768) |
| Tile | (64,64,64) | (16,16,128) | (16,16,128) |
| Tile source | fixed | profiling | profiling |
| Pad impl | numpy_pad | numpy_pad | numpy_pad |
| Trial 1 avg (µs) | 455.2 | 429.2 | 655.5 |
| Trial 2 avg (µs) | 421.3 | 585.4 | 378.3 |
| Trial 3 avg (µs) | 423.7 | 617.6 | 379.8 |
| **Mean avg (µs)** | **433.4** | **544.1** | **471.2** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: BASELINE 0**
Δ(B2 vs B0)=+110.7µs (+25.5%)   Δ(AG vs B0)=+37.8µs (+8.7%)   Δ(AG vs B2)=-72.9µs (-13.4%)

---

### 50×672×32  —  `action_out_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,32,768) | (64,32,768) | (64,32,768) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 139.5 | 139.9 | 142.2 |
| Trial 2 avg (µs) | 138.4 | 140.1 | 142.7 |
| Trial 3 avg (µs) | 139.2 | 139.4 | 142.6 |
| **Mean avg (µs)** | **139.1** | **139.8** | **142.5** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=+0.7µs (+0.5%)   Δ(AG vs B0)=+3.4µs (+2.5%)   Δ(AG vs B2)=+2.7µs (+1.9%)

---

### 50×672×256  —  `gate_proj, up_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,256,768) | (64,256,768) | (64,256,768) |
| Tile | (64,64,64) | (16,64,32) | (16,64,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 171.3 | 163.4 | 160.4 |
| Trial 2 avg (µs) | 171.1 | 159.6 | 162.8 |
| Trial 3 avg (µs) | 167.9 | 160.0 | 160.1 |
| **Mean avg (µs)** | **170.1** | **161.0** | **161.1** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-9.1µs (-5.4%)   Δ(AG vs B0)=-9.0µs (-5.3%)   Δ(AG vs B2)=+0.1µs (+0.1%)

---

### 50×672×960  —  `o_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,960,768) | (64,960,768) | (64,960,768) |
| Tile | (64,64,64) | (16,16,128) | (16,16,128) |
| Tile source | fixed | profiling | profiling |
| Pad impl | numpy_pad | numpy_pad | numpy_pad |
| Trial 1 avg (µs) | 420.9 | 431.1 | 662.9 |
| Trial 2 avg (µs) | 420.8 | 432.6 | 465.4 |
| Trial 3 avg (µs) | 424.4 | 688.8 | 379.5 |
| **Mean avg (µs)** | **422.1** | **517.5** | **502.6** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: BASELINE 0**
Δ(B2 vs B0)=+95.5µs (+22.6%)   Δ(AG vs B0)=+80.5µs (+19.1%)   Δ(AG vs B2)=-14.9µs (-2.9%)

---

### 50×720×32  —  `action_out_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,32,768) | (64,32,768) | (64,32,768) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 139.9 | 139.9 | 140.5 |
| Trial 2 avg (µs) | 140.3 | 138.2 | 143.2 |
| Trial 3 avg (µs) | 138.3 | 140.5 | 142.9 |
| **Mean avg (µs)** | **139.5** | **139.5** | **142.2** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=+0.0µs (+0.0%)   Δ(AG vs B0)=+2.7µs (+1.9%)   Δ(AG vs B2)=+2.6µs (+1.9%)

---

### 50×720×256  —  `gate_proj, up_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,256,768) | (64,256,768) | (64,256,768) |
| Tile | (64,64,64) | (16,64,32) | (16,64,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 171.5 | 160.2 | 161.9 |
| Trial 2 avg (µs) | 170.9 | 160.3 | 164.8 |
| Trial 3 avg (µs) | 170.8 | 162.4 | 161.9 |
| **Mean avg (µs)** | **171.1** | **161.0** | **162.8** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-10.1µs (-5.9%)   Δ(AG vs B0)=-8.2µs (-4.8%)   Δ(AG vs B2)=+1.9µs (+1.2%)

---

### 50×720×960  —  `o_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,960,768) | (64,960,768) | (64,960,768) |
| Tile | (64,64,64) | (16,16,128) | (16,16,128) |
| Tile source | fixed | profiling | profiling |
| Pad impl | numpy_pad | numpy_pad | numpy_pad |
| Trial 1 avg (µs) | 423.6 | 443.6 | 380.6 |
| Trial 2 avg (µs) | 421.6 | 378.0 | 496.5 |
| Trial 3 avg (µs) | 424.2 | 437.5 | 558.5 |
| **Mean avg (µs)** | **423.1** | **419.7** | **478.5** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-3.5µs (-0.8%)   Δ(AG vs B0)=+55.4µs (+13.1%)   Δ(AG vs B2)=+58.9µs (+14.0%)

---

### 50×960×192  —  `q_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,192,960) | (64,192,960) | (64,192,960) |
| Tile | (64,64,64) | (16,32,64) | (16,32,64) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 219.9 | 219.9 | 219.1 |
| Trial 2 avg (µs) | 221.9 | 219.2 | 222.7 |
| Trial 3 avg (µs) | 219.3 | 219.5 | 224.2 |
| **Mean avg (µs)** | **220.4** | **219.5** | **222.0** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-0.8µs (-0.4%)   Δ(AG vs B0)=+1.6µs (+0.7%)   Δ(AG vs B2)=+2.5µs (+1.1%)

---

### 50×960×288  —  `q_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,288,960) | (64,288,960) | (64,288,960) |
| Tile | (16,32,32) | (16,32,64) | (16,32,64) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 408.2 | 307.1 | 270.5 |
| Trial 2 avg (µs) | 274.4 | 273.7 | 270.9 |
| Trial 3 avg (µs) | 269.8 | 270.4 | 271.6 |
| **Mean avg (µs)** | **FAIL** | **283.7** | **271.0** |
| Pass count | 0/3 | 3/3 | 3/3 |

**Winner: INCONCLUSIVE**
Δ(AG vs B2)=-12.7µs (-4.5%)

---

### 50×960×384  —  `q_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,384,960) | (64,384,960) | (64,384,960) |
| Tile | (64,64,64) | (16,32,64) | (16,32,64) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 221.6 | 212.9 | 210.4 |
| Trial 2 avg (µs) | 222.3 | 213.1 | 210.7 |
| Trial 3 avg (µs) | 221.9 | 308.8 | 211.1 |
| **Mean avg (µs)** | **221.9** | **244.9** | **210.7** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: AGENTIC**
Δ(B2 vs B0)=+23.0µs (+10.4%)   Δ(AG vs B0)=-11.2µs (-5.0%)   Δ(AG vs B2)=-34.2µs (-14.0%)

---

### 50×960×480  —  `q_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,480,960) | (64,480,960) | (64,480,960) |
| Tile | (16,32,32) | (16,32,64) | (16,32,64) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 367.3 | 544.2 | 738.2 |
| Trial 2 avg (µs) | 362.3 | 644.9 | 365.2 |
| Trial 3 avg (µs) | 361.8 | 690.2 | 424.5 |
| **Mean avg (µs)** | **FAIL** | **626.4** | **509.3** |
| Pass count | 0/3 | 3/3 | 3/3 |

**Winner: INCONCLUSIVE**
Δ(AG vs B2)=-117.1µs (-18.7%)

---

### 50×960×576  —  `q_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,576,960) | (64,576,960) | (64,576,960) |
| Tile | (64,64,64) | (16,64,64) | (16,64,64) |
| Tile source | fixed | profiling | profiling |
| Pad impl | numpy_pad | numpy_pad | numpy_pad |
| Trial 1 avg (µs) | 431.9 | 560.8 | 555.4 |
| Trial 2 avg (µs) | 439.3 | 564.9 | 553.3 |
| Trial 3 avg (µs) | 447.2 | 577.1 | 541.9 |
| **Mean avg (µs)** | **439.5** | **567.6** | **550.2** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: BASELINE 0**
Δ(B2 vs B0)=+128.1µs (+29.1%)   Δ(AG vs B0)=+110.7µs (+25.2%)   Δ(AG vs B2)=-17.4µs (-3.1%)

---

### 50×960×672  —  `q_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,672,960) | (64,672,960) | (64,672,960) |
| Tile | (16,32,32) | (16,32,64) | (16,32,64) |
| Tile source | fixed | profiling | profiling |
| Pad impl | numpy_pad | numpy_pad | numpy_pad |
| Trial 1 avg (µs) | 688.5 | 531.6 | 453.8 |
| Trial 2 avg (µs) | 447.5 | 456.1 | 452.5 |
| Trial 3 avg (µs) | 454.5 | 472.2 | 454.8 |
| **Mean avg (µs)** | **FAIL** | **486.6** | **453.7** |
| Pass count | 0/3 | 3/3 | 3/3 |

**Winner: INCONCLUSIVE**
Δ(AG vs B2)=-32.9µs (-6.8%)

---

### 50×960×720  —  `q_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,768,960) | (64,768,960) | (64,768,960) |
| Tile | (64,64,64) | (16,64,64) | (16,64,64) |
| Tile source | fixed | profiling | profiling |
| Pad impl | numpy_pad | numpy_pad | numpy_pad |
| Trial 1 avg (µs) | 278.3 | 257.3 | 257.2 |
| Trial 2 avg (µs) | 276.6 | 272.6 | 256.8 |
| Trial 3 avg (µs) | 303.2 | 256.6 | 256.3 |
| **Mean avg (µs)** | **286.0** | **262.2** | **256.8** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: AGENTIC**
Δ(B2 vs B0)=-23.9µs (-8.3%)   Δ(AG vs B0)=-29.2µs (-10.2%)   Δ(AG vs B2)=-5.4µs (-2.1%)

---

### 60×32×96  —  `action_in_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,96,32) | (64,96,32) | (64,96,32) |
| Tile | (32,32,32) | (16,32,16) | (16,16,32) |
| Tile source | fixed | profiling | heuristic |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 130.4 | 131.4 | 141.2 |
| Trial 2 avg (µs) | 134.1 | 131.4 | 142.4 |
| Trial 3 avg (µs) | 134.4 | 134.6 | 142.1 |
| **Mean avg (µs)** | **133.0** | **132.5** | **141.9** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-0.5µs (-0.4%)   Δ(AG vs B0)=+8.9µs (+6.7%)   Δ(AG vs B2)=+9.4µs (+7.1%)

---

### 60×32×192  —  `action_in_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,192,32) | (64,192,32) | (64,192,32) |
| Tile | (32,32,32) | (16,64,16) | (16,64,16) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 128.9 | 130.7 | 132.6 |
| Trial 2 avg (µs) | 130.1 | 129.6 | 134.2 |
| Trial 3 avg (µs) | 129.1 | 133.6 | 135.9 |
| **Mean avg (µs)** | **129.4** | **131.3** | **134.2** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=+1.9µs (+1.5%)   Δ(AG vs B0)=+4.8µs (+3.8%)   Δ(AG vs B2)=+2.9µs (+2.2%)

---

### 60×32×288  —  `action_in_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,288,32) | (64,288,32) | (64,288,32) |
| Tile | (32,32,32) | (16,32,16) | (16,32,16) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 135.4 | 157.6 | 156.0 |
| Trial 2 avg (µs) | 139.6 | 155.5 | 156.3 |
| Trial 3 avg (µs) | 136.8 | 154.5 | 152.6 |
| **Mean avg (µs)** | **137.2** | **155.9** | **155.0** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: BASELINE 0**
Δ(B2 vs B0)=+18.6µs (+13.6%)   Δ(AG vs B0)=+17.7µs (+12.9%)   Δ(AG vs B2)=-0.9µs (-0.6%)

---

### 60×32×384  —  `action_in_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,384,32) | (64,384,32) | (64,384,32) |
| Tile | (32,32,32) | (16,128,16) | (16,128,16) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 151.2 | 133.0 | 132.9 |
| Trial 2 avg (µs) | 151.8 | 134.0 | 136.4 |
| Trial 3 avg (µs) | 157.7 | 135.3 | 132.9 |
| **Mean avg (µs)** | **153.6** | **134.1** | **134.1** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-19.5µs (-12.7%)   Δ(AG vs B0)=-19.5µs (-12.7%)   Δ(AG vs B2)=+0.0µs (+0.0%)

---

### 60×32×480  —  `action_in_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,480,32) | (64,480,32) | (64,480,32) |
| Tile | (32,32,32) | (16,32,16) | (16,32,16) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 140.3 | 186.1 | 184.2 |
| Trial 2 avg (µs) | 142.9 | 185.8 | 184.7 |
| Trial 3 avg (µs) | 141.6 | 182.1 | 183.6 |
| **Mean avg (µs)** | **141.6** | **184.7** | **184.2** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: BASELINE 0**
Δ(B2 vs B0)=+43.1µs (+30.4%)   Δ(AG vs B0)=+42.6µs (+30.1%)   Δ(AG vs B2)=-0.5µs (-0.3%)

---

### 60×32×576  —  `action_in_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,576,32) | (64,576,32) | (64,576,32) |
| Tile | (32,32,32) | (16,64,16) | (16,64,16) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 148.0 | 157.2 | 156.8 |
| Trial 2 avg (µs) | 148.5 | 156.6 | 157.1 |
| Trial 3 avg (µs) | 148.0 | 155.3 | 154.8 |
| **Mean avg (µs)** | **148.2** | **156.4** | **156.2** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: BASELINE 0**
Δ(B2 vs B0)=+8.2µs (+5.5%)   Δ(AG vs B0)=+8.0µs (+5.4%)   Δ(AG vs B2)=-0.1µs (-0.1%)

---

### 60×32×672  —  `action_in_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,672,32) | (64,672,32) | (64,672,32) |
| Tile | (32,32,32) | (16,32,32) | (16,32,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 154.7 | 214.9 | 209.6 |
| Trial 2 avg (µs) | 158.0 | 208.0 | 210.2 |
| Trial 3 avg (µs) | 159.4 | 207.6 | 208.5 |
| **Mean avg (µs)** | **157.4** | **210.2** | **209.4** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: BASELINE 0**
Δ(B2 vs B0)=+52.8µs (+33.5%)   Δ(AG vs B0)=+52.0µs (+33.1%)   Δ(AG vs B2)=-0.8µs (-0.4%)

---

### 60×32×720  —  `action_in_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,768,32) | (64,768,32) | (64,768,32) |
| Tile | (32,32,32) | (16,256,16) | (16,256,16) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 180.6 | 153.8 | 152.4 |
| Trial 2 avg (µs) | 182.2 | 153.5 | 152.6 |
| Trial 3 avg (µs) | 182.5 | 150.4 | 151.9 |
| **Mean avg (µs)** | **181.8** | **152.6** | **152.3** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-29.2µs (-16.1%)   Δ(AG vs B0)=-29.4µs (-16.2%)   Δ(AG vs B2)=-0.3µs (-0.2%)

---

### 60×96×32  —  `action_out_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,32,128) | (64,32,128) | (64,32,128) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 125.1 | 126.2 | 125.8 |
| Trial 2 avg (µs) | 125.7 | 126.2 | 122.2 |
| Trial 3 avg (µs) | 125.0 | 124.8 | 125.3 |
| **Mean avg (µs)** | **125.3** | **125.7** | **124.4** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=+0.5µs (+0.4%)   Δ(AG vs B0)=-0.8µs (-0.7%)   Δ(AG vs B2)=-1.3µs (-1.0%)

---

### 60×96×256  —  `gate_proj, up_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,256,128) | (64,256,128) | (64,256,128) |
| Tile | (64,64,64) | (16,64,64) | (16,64,64) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 142.4 | 136.3 | 140.0 |
| Trial 2 avg (µs) | 141.5 | 135.3 | 137.3 |
| Trial 3 avg (µs) | 142.4 | 135.7 | 134.0 |
| **Mean avg (µs)** | **142.1** | **135.8** | **137.1** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-6.3µs (-4.5%)   Δ(AG vs B0)=-5.0µs (-3.5%)   Δ(AG vs B2)=+1.3µs (+1.0%)

---

### 60×96×960  —  `o_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,960,128) | (64,960,128) | (64,960,128) |
| Tile | (64,64,64) | (16,64,32) | (16,64,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 185.5 | 203.1 | 202.8 |
| Trial 2 avg (µs) | 184.7 | 204.0 | 204.3 |
| Trial 3 avg (µs) | 194.6 | 200.6 | 203.3 |
| **Mean avg (µs)** | **188.2** | **202.6** | **203.5** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: BASELINE 0**
Δ(B2 vs B0)=+14.3µs (+7.6%)   Δ(AG vs B0)=+15.2µs (+8.1%)   Δ(AG vs B2)=+0.9µs (+0.5%)

---

### 60×192×32  —  `action_out_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,32,256) | (64,32,256) | (64,32,256) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 123.6 | 126.8 | 123.8 |
| Trial 2 avg (µs) | 122.1 | 125.9 | 126.9 |
| Trial 3 avg (µs) | 127.8 | 125.0 | 127.9 |
| **Mean avg (µs)** | **124.5** | **125.9** | **126.2** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=+1.4µs (+1.1%)   Δ(AG vs B0)=+1.7µs (+1.4%)   Δ(AG vs B2)=+0.3µs (+0.2%)

---

### 60×192×256  —  `gate_proj, up_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,256,256) | (64,256,256) | (64,256,256) |
| Tile | (64,64,64) | (16,64,32) | (16,64,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 149.1 | 135.6 | 135.2 |
| Trial 2 avg (µs) | 149.1 | 138.0 | 136.6 |
| Trial 3 avg (µs) | 148.3 | 138.1 | 139.0 |
| **Mean avg (µs)** | **148.8** | **137.2** | **137.0** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-11.6µs (-7.8%)   Δ(AG vs B0)=-11.9µs (-8.0%)   Δ(AG vs B2)=-0.3µs (-0.2%)

---

### 60×192×960  —  `o_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,960,256) | (64,960,256) | (64,960,256) |
| Tile | (64,64,64) | (16,32,16) | (16,32,16) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 233.0 | 258.9 | 257.2 |
| Trial 2 avg (µs) | 228.4 | 257.7 | 258.5 |
| Trial 3 avg (µs) | 231.1 | 261.4 | 256.6 |
| **Mean avg (µs)** | **230.8** | **259.4** | **257.4** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: BASELINE 0**
Δ(B2 vs B0)=+28.5µs (+12.4%)   Δ(AG vs B0)=+26.6µs (+11.5%)   Δ(AG vs B2)=-1.9µs (-0.7%)

---

### 60×256×96  —  `down_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,96,256) | (64,96,256) | (64,96,256) |
| Tile | (32,32,32) | (16,32,16) | (64,32,64) |
| Tile source | fixed | profiling | heuristic |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 137.3 | 138.1 | 138.4 |
| Trial 2 avg (µs) | 136.8 | 139.9 | 136.7 |
| Trial 3 avg (µs) | 139.8 | 139.1 | 138.6 |
| **Mean avg (µs)** | **137.9** | **139.1** | **137.9** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=+1.1µs (+0.8%)   Δ(AG vs B0)=-0.1µs (-0.0%)   Δ(AG vs B2)=-1.2µs (-0.8%)

---

### 60×256×192  —  `down_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,192,256) | (64,192,256) | (64,192,256) |
| Tile | (64,64,64) | (16,64,32) | (16,64,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 154.5 | 157.3 | 152.8 |
| Trial 2 avg (µs) | 154.8 | 151.3 | 154.9 |
| Trial 3 avg (µs) | 155.7 | 156.8 | 154.4 |
| **Mean avg (µs)** | **155.0** | **155.1** | **154.0** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=+0.1µs (+0.1%)   Δ(AG vs B0)=-1.0µs (-0.6%)   Δ(AG vs B2)=-1.1µs (-0.7%)

---

### 60×256×288  —  `down_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,288,256) | (64,288,256) | (64,288,256) |
| Tile | (32,32,32) | (16,32,32) | (16,32,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 158.2 | 169.9 | 169.4 |
| Trial 2 avg (µs) | 159.5 | 173.2 | 175.6 |
| Trial 3 avg (µs) | 161.6 | 169.5 | 171.0 |
| **Mean avg (µs)** | **159.8** | **170.9** | **172.0** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: BASELINE 0**
Δ(B2 vs B0)=+11.1µs (+7.0%)   Δ(AG vs B0)=+12.2µs (+7.7%)   Δ(AG vs B2)=+1.1µs (+0.7%)

---

### 60×256×384  —  `down_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,384,256) | (64,384,256) | (64,384,256) |
| Tile | (64,64,64) | (16,32,128) | (16,32,128) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 155.7 | 166.7 | 169.3 |
| Trial 2 avg (µs) | 154.4 | 167.3 | 166.8 |
| Trial 3 avg (µs) | 158.5 | 167.5 | 170.1 |
| **Mean avg (µs)** | **156.2** | **167.1** | **168.8** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: BASELINE 0**
Δ(B2 vs B0)=+11.0µs (+7.0%)   Δ(AG vs B0)=+12.6µs (+8.1%)   Δ(AG vs B2)=+1.6µs (+1.0%)

---

### 60×256×480  —  `down_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,480,256) | (64,480,256) | (64,480,256) |
| Tile | (32,32,32) | (16,32,32) | (16,32,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 173.2 | 258.7 | 204.6 |
| Trial 2 avg (µs) | 176.2 | 202.4 | 426.4 |
| Trial 3 avg (µs) | 174.6 | 204.9 | 202.5 |
| **Mean avg (µs)** | **174.7** | **222.0** | **277.9** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: BASELINE 0**
Δ(B2 vs B0)=+47.3µs (+27.1%)   Δ(AG vs B0)=+103.2µs (+59.0%)   Δ(AG vs B2)=+55.9µs (+25.2%)

---

### 60×256×576  —  `down_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,576,256) | (64,576,256) | (64,576,256) |
| Tile | (64,64,64) | (16,64,32) | (16,64,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 197.0 | 386.0 | 209.8 |
| Trial 2 avg (µs) | 193.4 | 212.2 | 208.8 |
| Trial 3 avg (µs) | 263.3 | 210.6 | 210.5 |
| **Mean avg (µs)** | **217.9** | **269.6** | **209.7** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: AGENTIC**
Δ(B2 vs B0)=+51.7µs (+23.7%)   Δ(AG vs B0)=-8.2µs (-3.8%)   Δ(AG vs B2)=-59.9µs (-22.2%)

---

### 60×256×672  —  `down_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,672,256) | (64,672,256) | (64,672,256) |
| Tile | (32,32,32) | (16,32,32) | (16,32,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 206.0 | 225.6 | 223.4 |
| Trial 2 avg (µs) | 211.9 | 224.2 | 226.2 |
| Trial 3 avg (µs) | 207.4 | 224.4 | 228.5 |
| **Mean avg (µs)** | **208.4** | **224.8** | **226.1** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: BASELINE 0**
Δ(B2 vs B0)=+16.4µs (+7.8%)   Δ(AG vs B0)=+17.6µs (+8.5%)   Δ(AG vs B2)=+1.3µs (+0.6%)

---

### 60×256×720  —  `down_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,768,256) | (64,768,256) | (64,768,256) |
| Tile | (64,64,64) | (16,64,128) | (16,64,128) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 181.4 | 179.7 | 179.9 |
| Trial 2 avg (µs) | 183.9 | 181.4 | 181.0 |
| Trial 3 avg (µs) | 181.9 | 181.5 | 183.9 |
| **Mean avg (µs)** | **182.4** | **180.9** | **181.6** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-1.5µs (-0.8%)   Δ(AG vs B0)=-0.8µs (-0.4%)   Δ(AG vs B2)=+0.7µs (+0.4%)

---

### 60×288×32  —  `action_out_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,32,512) | (64,32,512) | (64,32,512) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 132.2 | 127.0 | 133.6 |
| Trial 2 avg (µs) | 132.5 | 134.3 | 129.7 |
| Trial 3 avg (µs) | 132.4 | 133.2 | 133.1 |
| **Mean avg (µs)** | **132.4** | **131.5** | **132.1** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-0.9µs (-0.7%)   Δ(AG vs B0)=-0.3µs (-0.2%)   Δ(AG vs B2)=+0.6µs (+0.5%)

---

### 60×288×256  —  `gate_proj, up_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,256,512) | (64,256,512) | (64,256,512) |
| Tile | (64,64,64) | (16,64,32) | (16,64,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 157.3 | 151.3 | 146.8 |
| Trial 2 avg (µs) | 158.5 | 150.5 | 151.3 |
| Trial 3 avg (µs) | 159.6 | 150.0 | 148.9 |
| **Mean avg (µs)** | **158.5** | **150.6** | **149.0** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-7.9µs (-5.0%)   Δ(AG vs B0)=-9.5µs (-6.0%)   Δ(AG vs B2)=-1.6µs (-1.1%)

---

### 60×288×960  —  `o_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,960,512) | (64,960,512) | (64,960,512) |
| Tile | (64,64,64) | (16,16,128) | (16,16,128) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 329.5 | 374.1 | 373.7 |
| Trial 2 avg (µs) | 324.6 | 683.3 | 408.0 |
| Trial 3 avg (µs) | 384.8 | 374.3 | 380.9 |
| **Mean avg (µs)** | **346.3** | **477.3** | **387.6** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: BASELINE 0**
Δ(B2 vs B0)=+131.0µs (+37.8%)   Δ(AG vs B0)=+41.2µs (+11.9%)   Δ(AG vs B2)=-89.7µs (-18.8%)

---

### 60×384×32  —  `action_out_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,32,512) | (64,32,512) | (64,32,512) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 132.6 | 129.7 | 133.4 |
| Trial 2 avg (µs) | 136.8 | 134.1 | 133.5 |
| Trial 3 avg (µs) | 135.1 | 134.0 | 130.9 |
| **Mean avg (µs)** | **134.8** | **132.6** | **132.6** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-2.2µs (-1.7%)   Δ(AG vs B0)=-2.2µs (-1.7%)   Δ(AG vs B2)=-0.0µs (-0.0%)

---

### 60×384×256  —  `gate_proj, up_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,256,512) | (64,256,512) | (64,256,512) |
| Tile | (64,64,64) | (16,64,32) | (16,64,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 156.2 | 152.3 | 151.3 |
| Trial 2 avg (µs) | 159.0 | 152.4 | 146.9 |
| Trial 3 avg (µs) | 160.1 | 149.0 | 145.6 |
| **Mean avg (µs)** | **158.4** | **151.2** | **147.9** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: AGENTIC**
Δ(B2 vs B0)=-7.2µs (-4.5%)   Δ(AG vs B0)=-10.5µs (-6.6%)   Δ(AG vs B2)=-3.3µs (-2.2%)

---

### 60×384×960  —  `o_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,960,512) | (64,960,512) | (64,960,512) |
| Tile | (64,64,64) | (16,16,128) | (16,16,128) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 412.8 | 378.0 | 379.1 |
| Trial 2 avg (µs) | 545.4 | 376.1 | 522.0 |
| Trial 3 avg (µs) | 471.4 | 379.5 | 374.0 |
| **Mean avg (µs)** | **476.5** | **377.9** | **425.1** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: BASELINE 2**
Δ(B2 vs B0)=-98.7µs (-20.7%)   Δ(AG vs B0)=-51.5µs (-10.8%)   Δ(AG vs B2)=+47.2µs (+12.5%)

---

### 60×480×32  —  `action_out_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,32,512) | (64,32,512) | (64,32,512) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 136.0 | 129.7 | 130.0 |
| Trial 2 avg (µs) | 129.7 | 131.4 | 135.0 |
| Trial 3 avg (µs) | 132.4 | 131.2 | 130.6 |
| **Mean avg (µs)** | **132.7** | **130.7** | **131.9** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-2.0µs (-1.5%)   Δ(AG vs B0)=-0.8µs (-0.6%)   Δ(AG vs B2)=+1.2µs (+0.9%)

---

### 60×480×256  —  `gate_proj, up_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,256,512) | (64,256,512) | (64,256,512) |
| Tile | (64,64,64) | (16,64,32) | (16,64,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 161.1 | 148.6 | 148.5 |
| Trial 2 avg (µs) | 161.4 | 150.0 | 153.2 |
| Trial 3 avg (µs) | 154.8 | 148.9 | 148.0 |
| **Mean avg (µs)** | **159.1** | **149.2** | **149.9** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-10.0µs (-6.3%)   Δ(AG vs B0)=-9.2µs (-5.8%)   Δ(AG vs B2)=+0.7µs (+0.5%)

---

### 60×480×960  —  `o_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,960,512) | (64,960,512) | (64,960,512) |
| Tile | (64,64,64) | (16,16,128) | (16,16,128) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 321.5 | 373.7 | 375.7 |
| Trial 2 avg (µs) | 326.9 | 481.0 | 497.8 |
| Trial 3 avg (µs) | 359.9 | 376.1 | 373.3 |
| **Mean avg (µs)** | **336.1** | **410.2** | **415.6** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: BASELINE 0**
Δ(B2 vs B0)=+74.1µs (+22.1%)   Δ(AG vs B0)=+79.5µs (+23.6%)   Δ(AG vs B2)=+5.3µs (+1.3%)

---

### 60×576×32  —  `action_out_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,32,768) | (64,32,768) | (64,32,768) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 139.2 | 142.1 | 145.1 |
| Trial 2 avg (µs) | 137.8 | 139.6 | 141.8 |
| Trial 3 avg (µs) | 138.8 | 139.5 | 140.1 |
| **Mean avg (µs)** | **138.6** | **140.4** | **142.3** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=+1.8µs (+1.3%)   Δ(AG vs B0)=+3.8µs (+2.7%)   Δ(AG vs B2)=+2.0µs (+1.4%)

---

### 60×576×256  —  `gate_proj, up_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,256,768) | (64,256,768) | (64,256,768) |
| Tile | (64,64,64) | (16,64,32) | (16,64,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 173.1 | 158.9 | 162.5 |
| Trial 2 avg (µs) | 171.7 | 158.7 | 156.4 |
| Trial 3 avg (µs) | 169.7 | 158.7 | 160.3 |
| **Mean avg (µs)** | **171.5** | **158.8** | **159.7** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-12.7µs (-7.4%)   Δ(AG vs B0)=-11.8µs (-6.9%)   Δ(AG vs B2)=+0.9µs (+0.6%)

---

### 60×576×960  —  `o_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,960,768) | (64,960,768) | (64,960,768) |
| Tile | (64,64,64) | (16,16,128) | (16,16,128) |
| Tile source | fixed | profiling | profiling |
| Pad impl | numpy_pad | numpy_pad | numpy_pad |
| Trial 1 avg (µs) | 423.5 | 430.8 | 379.6 |
| Trial 2 avg (µs) | 420.7 | 541.6 | 546.9 |
| Trial 3 avg (µs) | 426.1 | 377.6 | 380.2 |
| **Mean avg (µs)** | **423.4** | **450.0** | **435.6** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: BASELINE 0**
Δ(B2 vs B0)=+26.6µs (+6.3%)   Δ(AG vs B0)=+12.1µs (+2.9%)   Δ(AG vs B2)=-14.4µs (-3.2%)

---

### 60×672×32  —  `action_out_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,32,768) | (64,32,768) | (64,32,768) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 143.2 | 139.9 | 136.5 |
| Trial 2 avg (µs) | 140.9 | 142.3 | 138.6 |
| Trial 3 avg (µs) | 139.1 | 143.1 | 139.4 |
| **Mean avg (µs)** | **141.1** | **141.8** | **138.2** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: AGENTIC**
Δ(B2 vs B0)=+0.7µs (+0.5%)   Δ(AG vs B0)=-2.9µs (-2.1%)   Δ(AG vs B2)=-3.6µs (-2.5%)

---

### 60×672×256  —  `gate_proj, up_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,256,768) | (64,256,768) | (64,256,768) |
| Tile | (64,64,64) | (16,64,32) | (16,64,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 169.3 | 165.6 | 160.0 |
| Trial 2 avg (µs) | 169.6 | 160.0 | 160.2 |
| Trial 3 avg (µs) | 174.2 | 162.1 | 156.6 |
| **Mean avg (µs)** | **171.1** | **162.6** | **158.9** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: AGENTIC**
Δ(B2 vs B0)=-8.5µs (-5.0%)   Δ(AG vs B0)=-12.1µs (-7.1%)   Δ(AG vs B2)=-3.6µs (-2.2%)

---

### 60×672×960  —  `o_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,960,768) | (64,960,768) | (64,960,768) |
| Tile | (64,64,64) | (16,16,128) | (16,16,128) |
| Tile source | fixed | profiling | profiling |
| Pad impl | numpy_pad | numpy_pad | numpy_pad |
| Trial 1 avg (µs) | 515.5 | 496.2 | 602.1 |
| Trial 2 avg (µs) | 536.0 | 529.5 | 378.1 |
| Trial 3 avg (µs) | 425.1 | 570.5 | 379.5 |
| **Mean avg (µs)** | **492.2** | **532.0** | **453.2** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: AGENTIC**
Δ(B2 vs B0)=+39.8µs (+8.1%)   Δ(AG vs B0)=-39.0µs (-7.9%)   Δ(AG vs B2)=-78.8µs (-14.8%)

---

### 60×720×32  —  `action_out_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,32,768) | (64,32,768) | (64,32,768) |
| Tile | (32,32,32) | (32,32,32) | (32,32,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 142.3 | 139.6 | 139.5 |
| Trial 2 avg (µs) | 140.6 | 142.1 | 141.1 |
| Trial 3 avg (µs) | 138.9 | 142.1 | 141.4 |
| **Mean avg (µs)** | **140.6** | **141.3** | **140.7** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=+0.7µs (+0.5%)   Δ(AG vs B0)=+0.1µs (+0.0%)   Δ(AG vs B2)=-0.6µs (-0.4%)

---

### 60×720×256  —  `gate_proj, up_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,256,768) | (64,256,768) | (64,256,768) |
| Tile | (64,64,64) | (16,64,32) | (16,64,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 173.1 | 153.6 | 159.1 |
| Trial 2 avg (µs) | 172.5 | 159.8 | 161.0 |
| Trial 3 avg (µs) | 173.5 | 161.8 | 155.5 |
| **Mean avg (µs)** | **173.1** | **158.4** | **158.6** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-14.6µs (-8.4%)   Δ(AG vs B0)=-14.5µs (-8.4%)   Δ(AG vs B2)=+0.1µs (+0.1%)

---

### 60×720×960  —  `o_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,960,768) | (64,960,768) | (64,960,768) |
| Tile | (64,64,64) | (16,16,128) | (16,16,128) |
| Tile source | fixed | profiling | profiling |
| Pad impl | numpy_pad | numpy_pad | numpy_pad |
| Trial 1 avg (µs) | 422.0 | 383.8 | 381.4 |
| Trial 2 avg (µs) | 427.1 | 560.4 | 383.2 |
| Trial 3 avg (µs) | 421.3 | 377.4 | 445.1 |
| **Mean avg (µs)** | **423.5** | **440.5** | **403.2** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: AGENTIC**
Δ(B2 vs B0)=+17.0µs (+4.0%)   Δ(AG vs B0)=-20.2µs (-4.8%)   Δ(AG vs B2)=-37.3µs (-8.5%)

---

### 60×960×96  —  `q_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,96,960) | (64,96,960) | (64,96,960) |
| Tile | (16,32,32) | (16,32,64) | (16,32,64) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 170.4 | 174.4 | 171.0 |
| Trial 2 avg (µs) | 169.7 | 174.4 | 174.0 |
| Trial 3 avg (µs) | 170.6 | 170.8 | 169.6 |
| **Mean avg (µs)** | **FAIL** | **173.2** | **171.5** |
| Pass count | 0/3 | 3/3 | 3/3 |

**Winner: INCONCLUSIVE**
Δ(AG vs B2)=-1.7µs (-1.0%)

---

### 60×960×192  —  `q_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,192,960) | (64,192,960) | (64,192,960) |
| Tile | (64,64,64) | (16,32,64) | (16,32,64) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 250.9 | 218.9 | 219.9 |
| Trial 2 avg (µs) | 219.9 | 221.3 | 221.2 |
| Trial 3 avg (µs) | 221.6 | 223.5 | 222.1 |
| **Mean avg (µs)** | **230.8** | **221.3** | **221.0** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-9.5µs (-4.1%)   Δ(AG vs B0)=-9.8µs (-4.2%)   Δ(AG vs B2)=-0.2µs (-0.1%)

---

### 60×960×288  —  `q_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,288,960) | (64,288,960) | (64,288,960) |
| Tile | (16,32,32) | (16,32,64) | (16,32,64) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 266.8 | 373.5 | 272.4 |
| Trial 2 avg (µs) | 591.5 | 269.2 | 272.2 |
| Trial 3 avg (µs) | 269.0 | 269.8 | 273.1 |
| **Mean avg (µs)** | **FAIL** | **304.1** | **272.6** |
| Pass count | 0/3 | 3/3 | 3/3 |

**Winner: INCONCLUSIVE**
Δ(AG vs B2)=-31.6µs (-10.4%)

---

### 60×960×384  —  `q_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,384,960) | (64,384,960) | (64,384,960) |
| Tile | (64,64,64) | (16,32,64) | (16,32,64) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 224.1 | 214.3 | 324.0 |
| Trial 2 avg (µs) | 219.3 | 213.9 | 209.8 |
| Trial 3 avg (µs) | 220.2 | 211.9 | 247.1 |
| **Mean avg (µs)** | **221.2** | **213.4** | **260.3** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: BASELINE 2**
Δ(B2 vs B0)=-7.8µs (-3.5%)   Δ(AG vs B0)=+39.1µs (+17.7%)   Δ(AG vs B2)=+47.0µs (+22.0%)

---

### 60×960×480  —  `q_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,480,960) | (64,480,960) | (64,480,960) |
| Tile | (16,32,32) | (16,32,64) | (16,32,64) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 363.8 | 647.7 | 372.9 |
| Trial 2 avg (µs) | 366.4 | 637.4 | 532.6 |
| Trial 3 avg (µs) | 426.4 | 364.9 | 694.4 |
| **Mean avg (µs)** | **FAIL** | **550.0** | **533.3** |
| Pass count | 0/3 | 3/3 | 3/3 |

**Winner: INCONCLUSIVE**
Δ(AG vs B2)=-16.7µs (-3.0%)

---

### 60×960×576  —  `q_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,576,960) | (64,576,960) | (64,576,960) |
| Tile | (64,64,64) | (16,64,64) | (16,64,64) |
| Tile source | fixed | profiling | profiling |
| Pad impl | numpy_pad | numpy_pad | numpy_pad |
| Trial 1 avg (µs) | 510.1 | 569.1 | 658.8 |
| Trial 2 avg (µs) | 375.7 | 540.8 | 616.8 |
| Trial 3 avg (µs) | 508.3 | 401.9 | 589.1 |
| **Mean avg (µs)** | **464.7** | **503.9** | **621.6** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: BASELINE 0**
Δ(B2 vs B0)=+39.2µs (+8.4%)   Δ(AG vs B0)=+156.9µs (+33.8%)   Δ(AG vs B2)=+117.6µs (+23.3%)

---

### 60×960×672  —  `q_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,672,960) | (64,672,960) | (64,672,960) |
| Tile | (16,32,32) | (16,32,64) | (16,32,64) |
| Tile source | fixed | profiling | profiling |
| Pad impl | numpy_pad | numpy_pad | numpy_pad |
| Trial 1 avg (µs) | 452.0 | 455.0 | 455.5 |
| Trial 2 avg (µs) | 521.5 | 453.2 | 450.9 |
| Trial 3 avg (µs) | 660.1 | 746.8 | 455.7 |
| **Mean avg (µs)** | **FAIL** | **551.7** | **454.0** |
| Pass count | 0/3 | 3/3 | 3/3 |

**Winner: INCONCLUSIVE**
Δ(AG vs B2)=-97.7µs (-17.7%)

---

### 60×960×720  —  `q_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (64,768,960) | (64,768,960) | (64,768,960) |
| Tile | (64,64,64) | (16,64,64) | (16,64,64) |
| Tile source | fixed | profiling | profiling |
| Pad impl | numpy_pad | numpy_pad | numpy_pad |
| Trial 1 avg (µs) | 274.7 | 259.6 | 260.7 |
| Trial 2 avg (µs) | 279.6 | 261.5 | 259.4 |
| Trial 3 avg (µs) | 276.2 | 257.0 | 255.8 |
| **Mean avg (µs)** | **276.9** | **259.4** | **258.6** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-17.5µs (-6.3%)   Δ(AG vs B0)=-18.2µs (-6.6%)   Δ(AG vs B2)=-0.8µs (-0.3%)

---

### 115×320×96  —  `k_proj, v_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (128,96,512) | (128,96,512) | (128,96,512) |
| Tile | (32,32,32) | (64,32,64) | (64,32,64) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 147.9 | 153.8 | 154.6 |
| Trial 2 avg (µs) | 149.4 | 150.5 | 153.9 |
| Trial 3 avg (µs) | 153.3 | 153.7 | 154.1 |
| **Mean avg (µs)** | **150.2** | **152.7** | **154.2** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=+2.5µs (+1.6%)   Δ(AG vs B0)=+4.0µs (+2.7%)   Δ(AG vs B2)=+1.5µs (+1.0%)

---

### 115×320×192  —  `k_proj, v_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (128,192,512) | (128,192,512) | (128,192,512) |
| Tile | (64,64,64) | (32,64,64) | (32,64,64) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 180.1 | 177.5 | 178.7 |
| Trial 2 avg (µs) | 180.6 | 178.0 | 176.5 |
| Trial 3 avg (µs) | 180.3 | 178.1 | 178.0 |
| **Mean avg (µs)** | **180.3** | **177.9** | **177.7** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-2.4µs (-1.4%)   Δ(AG vs B0)=-2.6µs (-1.4%)   Δ(AG vs B2)=-0.2µs (-0.1%)

---

### 115×320×288  —  `k_proj, v_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (128,288,512) | (128,288,512) | (128,288,512) |
| Tile | (32,32,32) | (64,32,64) | (64,32,64) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 206.7 | 270.8 | 194.0 |
| Trial 2 avg (µs) | 291.6 | 190.7 | 341.9 |
| Trial 3 avg (µs) | 207.7 | 190.3 | 191.1 |
| **Mean avg (µs)** | **235.4** | **217.3** | **242.4** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: BASELINE 2**
Δ(B2 vs B0)=-18.1µs (-7.7%)   Δ(AG vs B0)=+7.0µs (+3.0%)   Δ(AG vs B2)=+25.1µs (+11.5%)

---

### 115×320×384  —  `k_proj, v_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (128,384,512) | (128,384,512) | (128,384,512) |
| Tile | (64,64,64) | (32,32,128) | (32,32,128) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 183.8 | 227.0 | 197.1 |
| Trial 2 avg (µs) | 180.4 | 209.5 | 190.9 |
| Trial 3 avg (µs) | 199.7 | 191.7 | 190.2 |
| **Mean avg (µs)** | **188.0** | **209.4** | **192.7** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: BASELINE 0**
Δ(B2 vs B0)=+21.4µs (+11.4%)   Δ(AG vs B0)=+4.7µs (+2.5%)   Δ(AG vs B2)=-16.7µs (-8.0%)

---

### 115×320×480  —  `k_proj, v_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (128,480,512) | (128,480,512) | (128,480,512) |
| Tile | (32,32,32) | (64,32,64) | (64,32,64) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 261.5 | 384.3 | 239.9 |
| Trial 2 avg (µs) | 261.2 | 358.9 | 229.0 |
| Trial 3 avg (µs) | 354.0 | 227.4 | 225.9 |
| **Mean avg (µs)** | **292.2** | **323.5** | **231.6** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: AGENTIC**
Δ(B2 vs B0)=+31.3µs (+10.7%)   Δ(AG vs B0)=-60.6µs (-20.8%)   Δ(AG vs B2)=-91.9µs (-28.4%)

---

### 115×320×576  —  `k_proj, v_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (128,576,512) | (128,576,512) | (128,576,512) |
| Tile | (64,64,64) | (32,64,64) | (32,64,64) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 257.9 | 423.0 | 279.2 |
| Trial 2 avg (µs) | 262.3 | 430.7 | 695.3 |
| Trial 3 avg (µs) | 262.7 | 434.0 | 281.5 |
| **Mean avg (µs)** | **261.0** | **429.3** | **418.7** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: BASELINE 0**
Δ(B2 vs B0)=+168.3µs (+64.5%)   Δ(AG vs B0)=+157.7µs (+60.4%)   Δ(AG vs B2)=-10.6µs (-2.5%)

---

### 115×320×672  —  `k_proj, v_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (128,672,512) | (128,672,512) | (128,672,512) |
| Tile | (32,32,32) | (64,32,64) | (64,32,64) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 309.8 | 292.7 | 290.4 |
| Trial 2 avg (µs) | 528.8 | 370.4 | 297.0 |
| Trial 3 avg (µs) | 309.6 | 303.9 | 294.2 |
| **Mean avg (µs)** | **382.7** | **322.3** | **293.9** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: AGENTIC**
Δ(B2 vs B0)=-60.4µs (-15.8%)   Δ(AG vs B0)=-88.8µs (-23.2%)   Δ(AG vs B2)=-28.4µs (-8.8%)

---

### 115×320×720  —  `k_proj, v_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (128,768,512) | (128,768,512) | (128,768,512) |
| Tile | (64,64,64) | (64,64,64) | (64,64,64) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 221.5 | 360.0 | 222.0 |
| Trial 2 avg (µs) | 518.4 | 222.5 | 225.2 |
| Trial 3 avg (µs) | 224.4 | 223.2 | 220.1 |
| **Mean avg (µs)** | **321.4** | **268.6** | **222.5** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: AGENTIC**
Δ(B2 vs B0)=-52.9µs (-16.4%)   Δ(AG vs B0)=-99.0µs (-30.8%)   Δ(AG vs B2)=-46.1µs (-17.2%)

---

### 115×960×320  —  `k_proj, v_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (128,320,960) | (128,320,960) | (128,320,960) |
| Tile | (64,64,64) | (64,64,64) | (64,64,64) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 287.0 | 284.9 | 283.0 |
| Trial 2 avg (µs) | 281.8 | 281.8 | 423.0 |
| Trial 3 avg (µs) | 283.6 | 441.8 | 283.7 |
| **Mean avg (µs)** | **284.1** | **336.2** | **329.9** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: BASELINE 0**
Δ(B2 vs B0)=+52.1µs (+18.3%)   Δ(AG vs B0)=+45.8µs (+16.1%)   Δ(AG vs B2)=-6.3µs (-1.9%)

---

### 115×960×960  —  `q_proj, o_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (128,960,960) | (128,960,960) | (128,960,960) |
| Tile | (64,64,64) | (64,64,64) | (64,64,64) |
| Tile source | fixed | profiling | profiling |
| Pad impl | numpy_pad | numpy_pad | numpy_pad |
| Trial 1 avg (µs) | 501.3 | 496.2 | 500.4 |
| Trial 2 avg (µs) | 499.3 | 498.9 | 498.2 |
| Trial 3 avg (µs) | 525.8 | 495.2 | 498.3 |
| **Mean avg (µs)** | **508.8** | **495.7** | **498.3** |
| Pass count | 3/3 | 2/3 | 2/3 |

**Winner: TIE**
Δ(B2 vs B0)=-13.1µs (-2.6%)   Δ(AG vs B0)=-10.5µs (-2.1%)   Δ(AG vs B2)=+2.6µs (+0.5%)

---

### 115×960×2560  —  `gate_proj, up_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (128,2560,960) | (128,2560,960) | (128,2560,960) |
| Tile | (64,64,64) | (64,64,64) | (64,32,64) |
| Tile source | fixed | profiling | heuristic |
| Pad impl | numpy_pad | numpy_pad | numpy_pad |
| Trial 1 avg (µs) | 674.9 | 737.5 | 1273.5 |
| Trial 2 avg (µs) | 808.9 | 755.8 | 1187.9 |
| Trial 3 avg (µs) | 784.4 | 707.0 | 1323.0 |
| **Mean avg (µs)** | **741.9** | **737.5** | **1230.7** |
| Pass count | 2/3 | 1/3 | 2/3 |

**Winner: TIE**
Δ(B2 vs B0)=-4.4µs (-0.6%)   Δ(AG vs B0)=+488.8µs (+65.9%)   Δ(AG vs B2)=+493.2µs (+66.9%)

---

### 115×2560×960  —  `down_proj`

**Pre-screened skip**: bf16_K_limit

---

### 128×320×192  —  `k_proj, v_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (128,192,512) | (128,192,512) | (128,192,512) |
| Tile | (64,64,64) | (32,64,64) | (32,64,64) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 180.6 | 178.0 | 178.3 |
| Trial 2 avg (µs) | 177.5 | 179.0 | 179.4 |
| Trial 3 avg (µs) | 183.1 | 179.9 | 178.6 |
| **Mean avg (µs)** | **180.4** | **179.0** | **178.8** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-1.5µs (-0.8%)   Δ(AG vs B0)=-1.7µs (-0.9%)   Δ(AG vs B2)=-0.2µs (-0.1%)

---

### 128×320×288  —  `k_proj, v_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (128,288,512) | (128,288,512) | (128,288,512) |
| Tile | (32,32,32) | (64,32,64) | (64,32,64) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 208.7 | 193.5 | 192.2 |
| Trial 2 avg (µs) | 211.0 | 192.3 | 241.6 |
| Trial 3 avg (µs) | 323.2 | 287.1 | 189.2 |
| **Mean avg (µs)** | **247.6** | **224.3** | **207.7** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: AGENTIC**
Δ(B2 vs B0)=-23.3µs (-9.4%)   Δ(AG vs B0)=-40.0µs (-16.1%)   Δ(AG vs B2)=-16.6µs (-7.4%)

---

### 128×320×384  —  `k_proj, v_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (128,384,512) | (128,384,512) | (128,384,512) |
| Tile | (64,64,64) | (32,32,128) | (32,32,128) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 182.5 | 191.4 | 190.2 |
| Trial 2 avg (µs) | 179.7 | 187.4 | 188.8 |
| Trial 3 avg (µs) | 182.6 | 248.2 | 187.6 |
| **Mean avg (µs)** | **181.6** | **209.0** | **188.9** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: BASELINE 0**
Δ(B2 vs B0)=+27.4µs (+15.1%)   Δ(AG vs B0)=+7.3µs (+4.0%)   Δ(AG vs B2)=-20.1µs (-9.6%)

---

### 128×320×480  —  `k_proj, v_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (128,480,512) | (128,480,512) | (128,480,512) |
| Tile | (32,32,32) | (64,32,64) | (64,32,64) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 264.3 | 230.2 | 286.6 |
| Trial 2 avg (µs) | 264.0 | 229.0 | 295.6 |
| Trial 3 avg (µs) | 261.6 | 227.7 | 228.4 |
| **Mean avg (µs)** | **263.3** | **229.0** | **270.2** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: BASELINE 2**
Δ(B2 vs B0)=-34.3µs (-13.0%)   Δ(AG vs B0)=+6.9µs (+2.6%)   Δ(AG vs B2)=+41.2µs (+18.0%)

---

### 128×320×576  —  `k_proj, v_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (128,576,512) | (128,576,512) | (128,576,512) |
| Tile | (64,64,64) | (32,64,64) | (32,64,64) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 261.7 | 284.2 | 473.8 |
| Trial 2 avg (µs) | 259.6 | 376.8 | 283.2 |
| Trial 3 avg (µs) | 263.3 | 450.2 | 550.8 |
| **Mean avg (µs)** | **261.6** | **370.4** | **436.0** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: BASELINE 0**
Δ(B2 vs B0)=+108.8µs (+41.6%)   Δ(AG vs B0)=+174.4µs (+66.7%)   Δ(AG vs B2)=+65.6µs (+17.7%)

---

### 128×320×672  —  `k_proj, v_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (128,672,512) | (128,672,512) | (128,672,512) |
| Tile | (32,32,32) | (64,32,64) | (64,32,64) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 725.3 | 581.0 | 651.6 |
| Trial 2 avg (µs) | 305.5 | 292.6 | 290.9 |
| Trial 3 avg (µs) | 310.5 | 479.8 | 290.7 |
| **Mean avg (µs)** | **447.1** | **451.1** | **411.1** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: AGENTIC**
Δ(B2 vs B0)=+4.0µs (+0.9%)   Δ(AG vs B0)=-36.0µs (-8.1%)   Δ(AG vs B2)=-40.1µs (-8.9%)

---

### 128×320×720  —  `k_proj, v_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (128,768,512) | (128,768,512) | (128,768,512) |
| Tile | (64,64,64) | (64,64,64) | (64,64,64) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 220.7 | 224.6 | 259.1 |
| Trial 2 avg (µs) | 224.5 | 223.6 | 221.4 |
| Trial 3 avg (µs) | 222.8 | 221.5 | 223.8 |
| **Mean avg (µs)** | **222.7** | **223.2** | **234.7** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=+0.6µs (+0.3%)   Δ(AG vs B0)=+12.1µs (+5.4%)   Δ(AG vs B2)=+11.5µs (+5.2%)

---

### 179×320×96  —  `k_proj, v_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (192,96,512) | (192,96,512) | (192,96,512) |
| Tile | (32,32,32) | (32,32,32) | (64,32,64) |
| Tile source | fixed | fallback_fixed | heuristic |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 171.7 | 172.0 | 176.9 |
| Trial 2 avg (µs) | 173.0 | 172.7 | 175.1 |
| Trial 3 avg (µs) | 172.3 | 172.4 | 177.3 |
| **Mean avg (µs)** | **172.3** | **172.4** | **176.5** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=+0.0µs (+0.0%)   Δ(AG vs B0)=+4.1µs (+2.4%)   Δ(AG vs B2)=+4.1µs (+2.4%)

---

### 179×320×192  —  `k_proj, v_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (192,192,512) | (192,192,512) | (192,192,512) |
| Tile | (64,64,64) | (64,64,64) | (64,32,64) |
| Tile source | fixed | fallback_fixed | heuristic |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 180.8 | 186.5 | 222.9 |
| Trial 2 avg (µs) | 184.3 | 184.5 | 214.5 |
| Trial 3 avg (µs) | 183.5 | 184.1 | 222.7 |
| **Mean avg (µs)** | **182.9** | **185.1** | **220.0** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=+2.2µs (+1.2%)   Δ(AG vs B0)=+37.1µs (+20.3%)   Δ(AG vs B2)=+35.0µs (+18.9%)

---

### 179×320×288  —  `k_proj, v_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (192,288,512) | (192,288,512) | (192,288,512) |
| Tile | (32,32,32) | (32,32,32) | (64,32,64) |
| Tile source | fixed | fallback_fixed | heuristic |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 334.1 | 397.9 | 269.8 |
| Trial 2 avg (µs) | 269.9 | 339.8 | 303.2 |
| Trial 3 avg (µs) | 274.1 | 276.2 | 419.0 |
| **Mean avg (µs)** | **292.7** | **338.0** | **330.6** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: BASELINE 0**
Δ(B2 vs B0)=+45.2µs (+15.5%)   Δ(AG vs B0)=+37.9µs (+13.0%)   Δ(AG vs B2)=-7.3µs (-2.2%)

---

### 179×320×384  —  `k_proj, v_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (192,384,512) | (192,384,512) | (192,384,512) |
| Tile | (64,64,64) | (64,64,64) | (64,64,64) |
| Tile source | fixed | fallback_fixed | heuristic |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 230.0 | 229.7 | 230.8 |
| Trial 2 avg (µs) | 229.9 | 231.9 | 231.8 |
| Trial 3 avg (µs) | 231.0 | 228.7 | 340.8 |
| **Mean avg (µs)** | **230.3** | **230.1** | **267.8** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: TIE**
Δ(B2 vs B0)=-0.2µs (-0.1%)   Δ(AG vs B0)=+37.5µs (+16.3%)   Δ(AG vs B2)=+37.7µs (+16.4%)

---

### 179×320×480  —  `k_proj, v_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (192,480,512) | (192,480,512) | (192,480,512) |
| Tile | (16,32,32) | (16,32,32) | (64,32,64) |
| Tile source | fixed | fallback_fixed | heuristic |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 691.7 | 498.6 | 363.3 |
| Trial 2 avg (µs) | 563.9 | 500.2 | 535.5 |
| Trial 3 avg (µs) | 518.9 | 497.8 | 365.8 |
| **Mean avg (µs)** | **591.5** | **498.9** | **421.5** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: AGENTIC**
Δ(B2 vs B0)=-92.6µs (-15.7%)   Δ(AG vs B0)=-170.0µs (-28.7%)   Δ(AG vs B2)=-77.3µs (-15.5%)

---

### 179×320×576  —  `k_proj, v_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (192,576,512) | (192,576,512) | (192,576,512) |
| Tile | (64,64,64) | (64,64,64) | (64,64,64) |
| Tile source | fixed | fallback_fixed | heuristic |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 281.9 | 278.2 | 283.1 |
| Trial 2 avg (µs) | 300.9 | 280.6 | 279.6 |
| Trial 3 avg (µs) | 278.5 | 504.7 | 279.8 |
| **Mean avg (µs)** | **287.1** | **354.5** | **280.8** |
| Pass count | 3/3 | 3/3 | 3/3 |

**Winner: AGENTIC**
Δ(B2 vs B0)=+67.4µs (+23.5%)   Δ(AG vs B0)=-6.3µs (-2.2%)   Δ(AG vs B2)=-73.7µs (-20.8%)

---

### 179×320×672  —  `k_proj, v_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (192,672,512) | (192,672,512) | (192,672,512) |
| Tile | (16,32,32) | (16,32,32) | (16,32,32) |
| Tile source | fixed | fallback_fixed | heuristic |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | 668.4 | FAIL | FAIL |
| Trial 2 avg (µs) | 875.3 | FAIL | FAIL |
| Trial 3 avg (µs) | 655.2 | FAIL | FAIL |
| **Mean avg (µs)** | **733.0** | **FAIL** | **FAIL** |
| Pass count | 3/3 | 0/3 | 0/3 |

**Winner: INCONCLUSIVE**

---

### 179×320×720  —  `k_proj, v_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (192,768,512) | (192,768,512) | (192,768,512) |
| Tile | (16,32,32) | (16,32,32) | (16,32,32) |
| Tile source | fixed | fallback_fixed | heuristic |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | FAIL | FAIL | FAIL |
| Trial 2 avg (µs) | FAIL | FAIL | FAIL |
| Trial 3 avg (µs) | FAIL | FAIL | FAIL |
| **Mean avg (µs)** | **FAIL** | **FAIL** | **FAIL** |
| Pass count | 0/3 | 0/3 | 0/3 |

**Winner: INCONCLUSIVE**

---

### 179×960×320  —  `k_proj, v_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (192,320,960) | (192,320,960) | (192,320,960) |
| Tile | (16,32,32) | (16,32,32) | (16,32,32) |
| Tile source | fixed | fallback_fixed | heuristic |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | FAIL | FAIL | FAIL |
| Trial 2 avg (µs) | FAIL | FAIL | FAIL |
| Trial 3 avg (µs) | FAIL | FAIL | FAIL |
| **Mean avg (µs)** | **FAIL** | **FAIL** | **FAIL** |
| Pass count | 0/3 | 0/3 | 0/3 |

**Winner: INCONCLUSIVE**

---

### 179×960×960  —  `q_proj, o_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (192,960,960) | (192,960,960) | (192,960,960) |
| Tile | (16,32,32) | (16,32,32) | (16,32,32) |
| Tile source | fixed | fallback_fixed | heuristic |
| Pad impl | numpy_pad | numpy_pad | numpy_pad |
| Trial 1 avg (µs) | FAIL | FAIL | FAIL |
| Trial 2 avg (µs) | FAIL | FAIL | FAIL |
| Trial 3 avg (µs) | FAIL | FAIL | FAIL |
| **Mean avg (µs)** | **FAIL** | **FAIL** | **FAIL** |
| Pass count | 0/3 | 0/3 | 0/3 |

**Winner: INCONCLUSIVE**

---

### 179×960×2560  —  `gate_proj, up_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (192,2560,960) | (192,2560,960) | (192,2560,960) |
| Tile | (16,32,32) | (16,32,32) | (16,32,32) |
| Tile source | fixed | fallback_fixed | heuristic |
| Pad impl | numpy_pad | numpy_pad | numpy_pad |
| Trial 1 avg (µs) | FAIL | FAIL | FAIL |
| Trial 2 avg (µs) | FAIL | FAIL | FAIL |
| Trial 3 avg (µs) | FAIL | FAIL | FAIL |
| **Mean avg (µs)** | **FAIL** | **FAIL** | **FAIL** |
| Pass count | 0/3 | 0/3 | 0/3 |

**Winner: INCONCLUSIVE**

---

### 179×2560×960  —  `down_proj`

**Pre-screened skip**: bf16_K_limit

---

### 235×320×96  —  `k_proj, v_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (256,96,512) | (256,96,512) | (256,96,512) |
| Tile | (16,32,32) | (16,32,32) | (16,32,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | FAIL | FAIL | FAIL |
| Trial 2 avg (µs) | FAIL | FAIL | FAIL |
| Trial 3 avg (µs) | FAIL | FAIL | FAIL |
| **Mean avg (µs)** | **FAIL** | **FAIL** | **FAIL** |
| Pass count | 0/3 | 0/3 | 0/3 |

**Winner: INCONCLUSIVE**

---

### 235×320×192  —  `k_proj, v_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (256,192,512) | (256,192,512) | (256,192,512) |
| Tile | (16,32,32) | (16,32,32) | (16,32,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | FAIL | FAIL | FAIL |
| Trial 2 avg (µs) | FAIL | FAIL | FAIL |
| Trial 3 avg (µs) | FAIL | FAIL | FAIL |
| **Mean avg (µs)** | **FAIL** | **FAIL** | **FAIL** |
| Pass count | 0/3 | 0/3 | 0/3 |

**Winner: INCONCLUSIVE**

---

### 235×320×288  —  `k_proj, v_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (256,288,512) | (256,288,512) | (256,288,512) |
| Tile | (16,32,32) | (16,32,32) | (16,32,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | FAIL | FAIL | FAIL |
| Trial 2 avg (µs) | FAIL | FAIL | FAIL |
| Trial 3 avg (µs) | FAIL | FAIL | FAIL |
| **Mean avg (µs)** | **FAIL** | **FAIL** | **FAIL** |
| Pass count | 0/3 | 0/3 | 0/3 |

**Winner: INCONCLUSIVE**

---

### 235×320×384  —  `k_proj, v_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (256,384,512) | (256,384,512) | (256,384,512) |
| Tile | (16,32,32) | (16,32,32) | (16,32,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | FAIL | FAIL | FAIL |
| Trial 2 avg (µs) | FAIL | FAIL | FAIL |
| Trial 3 avg (µs) | FAIL | FAIL | FAIL |
| **Mean avg (µs)** | **FAIL** | **FAIL** | **FAIL** |
| Pass count | 0/3 | 0/3 | 0/3 |

**Winner: INCONCLUSIVE**

---

### 235×320×480  —  `k_proj, v_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (256,480,512) | (256,480,512) | (256,480,512) |
| Tile | (16,32,32) | (16,32,32) | (16,32,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | FAIL | FAIL | FAIL |
| Trial 2 avg (µs) | FAIL | FAIL | FAIL |
| Trial 3 avg (µs) | FAIL | FAIL | FAIL |
| **Mean avg (µs)** | **FAIL** | **FAIL** | **FAIL** |
| Pass count | 0/3 | 0/3 | 0/3 |

**Winner: INCONCLUSIVE**

---

### 235×320×576  —  `k_proj, v_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (256,576,512) | (256,576,512) | (256,576,512) |
| Tile | (16,32,32) | (16,32,32) | (16,32,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | FAIL | FAIL | FAIL |
| Trial 2 avg (µs) | FAIL | FAIL | FAIL |
| Trial 3 avg (µs) | FAIL | FAIL | FAIL |
| **Mean avg (µs)** | **FAIL** | **FAIL** | **FAIL** |
| Pass count | 0/3 | 0/3 | 0/3 |

**Winner: INCONCLUSIVE**

---

### 235×320×672  —  `k_proj, v_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (256,672,512) | (256,672,512) | (256,672,512) |
| Tile | (16,32,32) | (16,32,32) | (16,32,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | FAIL | FAIL | FAIL |
| Trial 2 avg (µs) | FAIL | FAIL | FAIL |
| Trial 3 avg (µs) | FAIL | FAIL | FAIL |
| **Mean avg (µs)** | **FAIL** | **FAIL** | **FAIL** |
| Pass count | 0/3 | 0/3 | 0/3 |

**Winner: INCONCLUSIVE**

---

### 235×320×720  —  `k_proj, v_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (256,768,512) | (256,768,512) | (256,768,512) |
| Tile | (16,32,32) | (16,32,32) | (16,32,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | FAIL | FAIL | FAIL |
| Trial 2 avg (µs) | FAIL | FAIL | FAIL |
| Trial 3 avg (µs) | FAIL | FAIL | FAIL |
| **Mean avg (µs)** | **FAIL** | **FAIL** | **FAIL** |
| Pass count | 0/3 | 0/3 | 0/3 |

**Winner: INCONCLUSIVE**

---

### 235×960×320  —  `k_proj, v_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (256,320,960) | (256,320,960) | (256,384,960) |
| Tile | (16,32,32) | (16,32,32) | (16,32,32) |
| Tile source | fixed | fallback_fixed | heuristic |
| Pad impl | manual_copy | manual_copy | manual_copy |
| Trial 1 avg (µs) | FAIL | FAIL | FAIL |
| Trial 2 avg (µs) | FAIL | FAIL | FAIL |
| Trial 3 avg (µs) | FAIL | FAIL | FAIL |
| **Mean avg (µs)** | **FAIL** | **FAIL** | **FAIL** |
| Pass count | 0/3 | 0/3 | 0/3 |

**Winner: INCONCLUSIVE**

---

### 235×960×960  —  `q_proj, o_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (256,960,960) | (256,960,960) | (256,960,960) |
| Tile | (16,32,32) | (16,32,32) | (16,32,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | numpy_pad | numpy_pad | numpy_pad |
| Trial 1 avg (µs) | FAIL | FAIL | FAIL |
| Trial 2 avg (µs) | FAIL | FAIL | FAIL |
| Trial 3 avg (µs) | FAIL | FAIL | FAIL |
| **Mean avg (µs)** | **FAIL** | **FAIL** | **FAIL** |
| Pass count | 0/3 | 0/3 | 0/3 |

**Winner: INCONCLUSIVE**

---

### 235×960×2560  —  `gate_proj, up_proj`

| | Baseline 0 | Baseline 2 | Agentic |
|-|------------|------------|---------|
| Padded shape | (256,2560,960) | (256,2560,960) | (256,2560,960) |
| Tile | (16,32,32) | (16,32,32) | (16,32,32) |
| Tile source | fixed | profiling | profiling |
| Pad impl | numpy_pad | numpy_pad | numpy_pad |
| Trial 1 avg (µs) | FAIL | FAIL | FAIL |
| Trial 2 avg (µs) | FAIL | FAIL | FAIL |
| Trial 3 avg (µs) | FAIL | FAIL | FAIL |
| **Mean avg (µs)** | **FAIL** | **FAIL** | **FAIL** |
| Pass count | 0/3 | 0/3 | 0/3 |

**Winner: INCONCLUSIVE**

---

### 235×2560×960  —  `down_proj`

**Pre-screened skip**: bf16_K_limit

---

## New Best Tiles Found (Agentic)

| Shape | Old best tile | Old avg (µs) | New best tile | New avg (µs) | Δ |
|-------|--------------|-------------|--------------|-------------|---|
| (64,96,256) bf16 | (16,32,16) | 139.1 | (64,32,64) | 137.9 | -0.8% |
| (192,288,512) bf16 | (32,32,32) | 338.0 | (64,32,64) | 330.6 | -2.2% |
| (192,480,512) bf16 | (16,32,32) | 498.9 | (64,32,64) | 421.5 | -15.5% |

## Memory Updates

- `npu_execution_profiling.json`: updated with agentic trial results
- `strategy_insights.md`: updated with new insights (if any)
- `error-logs/errors.md`: updated if new error patterns found
