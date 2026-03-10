# Agentic NPU Workflow — Master Prompt

This document is the detailed step-by-step instruction set for the agentic NPU
GEMM optimization agent. It mirrors the structure in the `agentic-npu` skill
but is written as a standalone runnable prompt you can paste directly into a
Claude Code session.

---

## How to use this prompt

1. Copy the entire content of this file into a Claude Code session.
2. Replace `<M>`, `<N>`, `<K>`, `<dtype>` with your desired values.
3. The agent will run through all five phases and produce a session log.

---

## Input (fill in before running)

```
M     = <M>
N     = <N>
K     = <K>
dtype = <dtype>   # one of: i8, i16, bf16
```

---

## Phase 0 — Load memory and knowledge

Read (in parallel where possible):
- All files in `/home/ec935/agent-gemm/memory/knowledge-base/`
- All files in `/home/ec935/agent-gemm/memory/gemm-data/`
- All files in `/home/ec935/agent-gemm/memory/error-logs/`
- All files in `/home/ec935/agent-gemm/memory/rules/`
- All files in `/home/ec935/agent-gemm/documentation/`
- First 300 lines of `/home/ec935/vla-to-npu/gemm/trials/vla_profile_combined.json`

Write a 2–4 bullet summary of what is most relevant to this specific M, N, K,
dtype before proceeding.

---

## Phase 1 — Propose a strategy

1. Determine whether padding is needed (is M in ALLOWED_M, N in ALLOWED_N,
   K in ALLOWED_K?).
2. Load the profile DB from `vla_profile_combined.json` and run all three
   selection strategies (hashmap_27, sorted_first_fit, preprocessed_map).
3. Pick the strategy with the lowest estimated total latency.
4. Select tile (m, n, k) from the profile or use safe defaults.
5. Select padding implementation based on knowledge base thresholds.
6. Output the strategy JSON block (see skill SKILL.md for format).
7. Confirm with user (unless `--auto` was passed).

---

## Phase 2 — Run on NPU

Run `v2_test_mapping_large_gemm.py` with the chosen strategy.
Log: execution time, correctness, any errors.
On failure: try up to 2 fallbacks, log each attempt.

---

## Phase 3 — Analyze

- Compute estimation error (%).
- Was there a better shape/tile available in the profile?
- Any new error patterns?
- Which knowledge base rules need updating?

---

## Phase 4 — Update memory

- Append to `memory/gemm-data/runs.json`
- Update `memory/knowledge-base/*.md` with new insights (dated bullets)
- Append to `memory/error-logs/errors.md` if new error was seen
- Create `memory/sessions/<timestamp>.md` with session summary

---

## Allowed shapes (hard constraints — never violate)

```
ALLOWED_M = [32, 64, 128, 256, 512, 1024, 2048]
ALLOWED_N = [32, 64, 128, 256, 512, 768, 1024, 2048]
ALLOWED_K = [32, 64, 128, 256, 512, 768, 1024, 2048]
```

Tile (m, n, k) must evenly divide padded (Mp, Np, Kp).
