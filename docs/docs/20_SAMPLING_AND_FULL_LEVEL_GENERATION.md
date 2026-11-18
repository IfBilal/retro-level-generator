# 20 — Sampling and Full-Level Generation

> **Modules:** `diffusion/sampler.py` (guided DDIM with a known-region hook), `generate/full_level.py`, `generate/framing.py`, `generate/sanitizer.py` (doc 05). **Ledger:** M13, G1-G5, H10. **Math:** doc 13 §7-9. **Gate:** G-P5 (Day 11). **Consumers:** doc 28 (app), doc 22 (eval).

## 1. Single-window sampling (`sampler.sample`)
Inputs: model (EMA), scheduler, context (canonical caption(s) -> text path, or null), `w`, `steps` (50; fast 25), `eta = 0`, `generator` (CPU, seeded), optional `known (grid, mask)`, optional `callback(step, x0_hat)`.
Per step: guided prediction `eps = eps_u + w (eps_c - eps_u)` (w = 1: skip the unconditional pass), `x0_hat` computed in fp32 and **clipped to [-1, 1]**, DDIM update with `eta = 0`. Final tile grid = `argmax` over channels of `x0_hat`. The `callback` gives the denoising frames (seam S-2, doc 31; used by the optional filmstrip, FR-A-10 MAY).

## 2. Known-region replacement (RePaint, G1)
For a window overlapping previously generated columns: at every step replace the known region of `x_t` by the forward-noised known tiles `q(x_t | x0_known)` and keep the unknown region from the model. Add **r = 2 resampling jumps** during the first 60 % of steps: `renoise(x_s -> x_t)` then denoise again (doc 13 §9). Cost per window about `S + r * ceil(0.6 S)` = 110 guided passes at S = 50 (H10). Fast mode: S = 25, r = 0.

## 3. Sliding-window outpainting (`generate/full_level.generate`)
1. Normalise the prompt (doc 12) -> list of scenes; scenes are spread over content windows in order (doc 12 §7); extra scenes merge into the last (`TOO_MANY_SCENES`).
2. Content width `W_gen` in {16, 48, 80, 112}; window 16 x 16, **stride 8** (overlap 8). Number of windows = `(W_gen - 16)/8 + 1` (width 112 -> 13).
3. Window 0: unconditional start from noise with the scene caption. Window k: known region = the right 8 columns of window k-1, new region = the right 8 columns.
4. Concatenate content columns; apply **framing** (G3): prepend 3 full-floor spawn columns, append the 6-column end template (floor + staircase + flag). Framing cells are flagged `templated` in the level JSON (doc 29).
5. **Sanitizer** (G4): detect broken pipes (`<>` / `[]`) and cannons (`B`/`b`) on the **raw** grid; record counts; optional display-time repair produces `display_grid`. Metrics always use `raw_grid`.
6. **Verify loop** (G5): run the fast checker (doc 21) on the framed raw level. On failure take the failing window (from `CheckResult.fail_col`), re-sample it with a new derived seed `seed' = hash(seed, window, attempt)`; **max 5 attempts per window, 20 per level**. Result `verified` / `not_verified` plus the attempt log. This function is the seam for the future ghost-bot feature (doc 31 S-1).
7. Return the level dict (doc 29).

## 4. Determinism
Initial noise per window from a CPU `torch.Generator` seeded by `(seed, window_index, attempt)` so the result does not depend on the machine's GPU RNG (NFR-REP-04). Same prompt + seed + w + width + steps + fast_mode + checkpoint -> same level on the same machine; cross-machine agreement is measured, not guaranteed.

## 5. Seam metric (G-P5)
Column-transition model fitted on train levels (smoothed column bigram; the corpus has 653 distinct column patterns, so add-k smoothing is required; doc 22 §3.11). `seam_ratio = mean NLL(transitions at window seams) / mean NLL(interior transitions)`; real dev levels with pseudo-seams give the reference (about 1.0, to be measured Day 11). Gate: ratio <= 1.3.

## 6. Escalation ladder when seams are bad (doc 25 R-5.1..R-5.4)
stride 8 -> 4 -> r = 2 -> 5 -> direct wide sampling of the full width (last resort, X12).

## 7. Fast mode and the cloud profile
Fast mode (25 steps, r = 0) is forced without CUDA. Cloud profile: CPU, batch of one window, `time_budget_s` guard checked between windows; on exceeding, abort and serve the nearest showcase level (ledger CL6). The app may use r = 1 and 40 steps only if stated on the About page (doc 28).

## 8. Tests (`tests/test_full_level.py`, `tests/test_sanitizer.py`)
window count and stride; known-region columns unchanged in the output within the replacement logic; framing columns exact and flagged; sanitizer detects each broken pattern and never alters `raw_grid`; verify loop respects 5/20 budgets (mock checker); determinism for equal seeds.
