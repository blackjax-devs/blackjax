# Relocated design notes

Collected verbatim from `blackjax/adaptation/meta/*.py` during the PR #1001
readability pass.  The TL folds these into the design doc / WIP paper.

Each block records: **file:lines** · **enclosing function** · **verbatim text**.

---

## Rule-2 — Multi-line WHY-essays removed from code

### Block 1 — `builders.py:117-120`, `build_meta_adaptation_core / init`

```
Buffer sized to half the budget so the growing-window schedule's largest
window (≈ max_budget_steps * 0.4–0.5) does not overflow the buffer.
With the cap n=min(buffer_idx, B) in final(), overflow is still safe
(RESET policy keeps the most-recent B draws = the more-stationary tail).
```

In-code replacement: `# half-budget ceiling; overflow is safe — RESET keeps the most-recent B draws`

---

### Block 2 — `builders.py:269-273`, `build_meta_adaptation_core / final`

```
Controller-signal scalars are always stored as float32: they are gate
comparisons and AIRM velocities that don't benefit from float64 precision.
Explicit casts here are required so that both branches of the
jax.lax.cond in staged_adaptation produce identical types under x64
(the false branch returns the init-time float32 carry unchanged).
```

In-code replacement: `# cast to float32 so both lax.cond branches match under x64`

---

### Block 3 — `builders.py:406-408`, `build_multi_chain_meta_core / init`

```
buf_size: sized to hold the largest single-chain window.  With the
budget split across n_chains, each chain runs max_budget_steps_per_chain
steps, so buf_size is half that (growing-window largest window ≈ 40–50%).
```

In-code replacement: `# half-budget ceiling per chain; same logic as single-chain init`

---

### Block 4 — `builders.py:543-546`, `build_multi_chain_meta_core / final`

```
NOTE: new_flag_count is computed AFTER BLOCKER-3 (mode-consistency + contraction)
because the primary multimodality signal is any_mode_flag (mode-consistency),
not the gap-stat alone.  The flag counter uses multimodality_signal (either signal)
to count consecutive windows for the 2-window confirmation gate.
```

In-code replacement: `# flag counter: multimodality_signal (any_mode_flag | ~is_unimodal) for 2-window confirmation`

---

### Block 5 — `builders.py:554-555`, `build_multi_chain_meta_core / final`

```
Fixes the v2 padding bug (chain-major zeros contaminated R² / Fisher paths)
and removes between-chain transient inflation from the curvature gate.
```

In-code replacement: removed (rationale already documented in `_router._build_pc_centered_time_major_pool` docstring)

---

## Rule-3 — Internal-process labels rewritten

All hits are neutralized in-place; rewrites recorded here for traceability.

| File | Line | Original label | Rewrite |
|------|------|----------------|---------|
| `_calibration.py` | 193 | `# BLOCKER-2 GAIN+abstain: ...` | `# Projected-tier GAIN+abstain: ...` |
| `builders.py` | 440 | `# v2.1 new fields — NaN / 0 until first window` | `# NaN / 0 until first window` |
| `builders.py` | 538 | `# ---- T-branch Gate 5: unimodality (gap-stat, corroborator after BLOCKER-3) ----` | `# ---- T-branch Gate 5: unimodality guard (gap-stat corroborator) ----` |
| `builders.py` | 544 | `# NOTE: new_flag_count is computed AFTER BLOCKER-3 ...` (4 lines) | see Block 4 above |
| `builders.py` | 553 | `# Fixes the v2 padding bug ...` (2 lines) | see Block 5 above |
| `builders.py` | 601 | `# BLOCKER-2: projected-tier GAIN+abstain router ...` | `# Projected-tier GAIN+abstain router ...` |
| `builders.py` | 677 | `# ---- BLOCKER-3: mode-consistency flag + contraction stat (T-branch guard) ----` | `# ---- Mode-consistency flag + contraction stat (T-branch guard) ----` |
| `builders.py` | 678 | `# Mode-consistency (BLOCKER-3): per direction e_j, flag iff` | `# Mode-consistency: per direction e_j, flag iff` |
| `builders.py` | 694 | `# Contraction stat (BLOCKER-3): per-chain split-half drift t.` | `# Contraction stat: per-chain split-half drift t.` |
| `builders.py` | 702 | `# ---- T-branch three-way (BLOCKER-3 fix, replaces binary gap-stat gate) ----` | `# ---- T-branch three-way rule (replaces binary gap-stat gate) ----` |
| `builders.py` | 863 | `# v2.1 new fields` | removed |
| `_state.py` | 131 | `# v2.1 additions — W-branch diagnostics + T-branch guard state` | `# W-branch diagnostics + T-branch guard state` |
| `_detection.py` | 331 | `BLOCKER-3 fix.  For each admitted T-spike direction …` (docstring) | `For each admitted T-spike direction …` |
| `_detection.py` | 490 | `BLOCKER-3 fix.  For each chain m, computes …` (docstring) | `For each chain m, computes …` |
| `test_meta_builders_e2e.py` | 1254 | `"""All 5 new v2.1 state fields are finite …"""` | `"""The five W-branch/T-branch diagnostic state fields are finite …"""` |
| `test_meta_builders_e2e.py` | 1555 | `This is the HEADLINE v2.1 acceptance cell.  In v2 the runtime was …` (class docstring) | removed (archaeology); behavioral description kept |
| `test_meta_builders_e2e.py` | 1559 | `The BLOCKER-1 fix (pooled-aware schedule) produces windows …` | `The pooled-aware schedule produces windows …` |
| `test_meta_builders_e2e.py` | 1595 | `With the BLOCKER-1 pooled-aware schedule, …` | `The pooled-aware schedule produces …` |
| `test_meta_builders_e2e.py` | 1631 | `f"BLOCKER-1 acceptance: staged_adaptation(n_chains=8) …"` | assertion message without label |
| `test_meta_detection.py` | 785 | `This is the CORRECT regression test for the v2 padding bug: …` | behavioral description only |
| `test_staged_adaptation.py` | 215 | `"""All three slice-1 recipes produce …"""` | `"""The core metric recipes (welford_diag, welford_dense, fisher_diag) produce …"""` |
