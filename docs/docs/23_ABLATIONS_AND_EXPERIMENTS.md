# 23 — Ablations & Experiment Matrix (X1–X12)

> **18-day scope note (2026-11-18, doc 02 §14):** the experiment set is **reduced to X1 (auxiliary CE) plus the linear-vs-cosine schedule comparison (U-lin/U-cos, Day 5)**. X11 and X12 (free, inference-only) and X7 (free analysis) stay SHOULD. X2, X3, X4, X9 are MAY and run only if time remains or their trigger fires. X5, X6, X8, X10 are **cut** (listed as future work in the report). The matrix below is kept as reference; the tag in the "Priority" column of this note overrides it.

> **Purpose:** decide **config v1** (Day 10) and the **final config** (Day 14 freeze) with controlled, cheap, dev-only experiments, and produce the ablation table for the report. Ledger rows touched: M2, M9, M11, M13, D8, D9, G1, H10, E3, E4.
> **Primary decision metric:** the dev composite **S** (E4, doc 22 §3.13), compared with a **paired, level-clustered bootstrap under common random numbers** (doc 22 §7.1-7.2; 5 dev levels, so intervals are wide). Everything else is secondary or a tie-breaker.
> **Configs:** `configs/ablations/X*.yaml` (diffs over a base config). **Runner:** `scripts/launch.py`. **Analysis:** `scripts/analyze_ablation.py`. **Registry:** `reports/ablations/REGISTRY.csv`.

---

## 0. Rules of the game

1. **One change per arm** relative to its baseline (the only exception is the explicit combination check in §6).
2. **Same everything else:** same base config, `train_seed`, data order (where the architecture permits), step budget, evaluation cadence, prompt sets and **sampling seeds** (CRN, doc 22 §2.3), and the same checkpoint-selection rule (argmax S, patience 5).
3. **Dev only.** P-dev (+ P-rand-dev for robustness). No ablation ever touches a test-side prompt file. `eval_dev.py` hard-refuses them.
4. **Batch 1 (Day 10) can change the model; batch 2 (cut in the 18-day plan; only X7 may run, on Day 15) cannot.** The freeze happens on Day 14 (`FROZEN.json`), so any batch-2 result is **report-only**. If a batch-2 arm is much better, it goes into "future work" and the frozen model stays as it is.
5. **Training-free experiments** (X2 if dual-EMA, X7, X11, X12) reuse a trained checkpoint and only change inference or evaluation. Run them first: they're nearly free.
6. **Everything is logged:** each arm gets a run directory (doc 30), one registry row, and a `summary.md` with the paired-bootstrap table and a figure.

---

## 1. Baselines

| Name | Definition | Produced |
|---|---|---|
| **U-lin / U-cos** | unconditional models, linear vs cosine schedule, 15k steps (`uncond_linear.yaml`, `uncond_cosine.yaml`) | Day 5 → winner = Phase-3 schedule (M2) |
| **C-main** | conditional baseline (`cond_main.yaml`): ledger M1–M13 defaults with the Phase-3 schedule; ≤ 60k steps, early-stopped | Day 8 |
| **C-main@w\*** | C-main at the dev-selected guidance w\* (Day 9) — **the comparison point for batch 1** | Day 9 |
| **config v1** | C-main + the adopted batch-1 changes (§6) | Day 10 |
| **final** | config v1 + inference choices from X11/X12 (Day 11–14) → `configs/final.yaml`, frozen | Day 14 |

Batch-1 arms train for **40k steps** (timeline Day 10). The baseline used for paired comparison is **C-main's 40k-step-capped checkpoint selection**: C-main's dev evaluations up to step 40k, argmax S. Capping at 40k keeps budgets equal, even if C-main itself continued to 60k.

---

## 2. Experiment matrix (summary)

| ID | Name | Type | Priority | Day | Machine | Budget | Baseline | Changes the model? |
|---|---|---|---|---|---|---|---|---|
| **X1** | Auxiliary CE loss (λ = 0.001, ᾱ_t-weighted) | train | **P0** | D10 | DT | 40k | C-main | yes (batch 1) |
| **X2** | EMA decay 0.995 vs 0.999 | free* / train | MAY | D10 | — / LT | 0 / 40k | C-main | yes (batch 1) |
| **X3** | Pooled text → time embedding | train | MAY | D10 | LT | 40k | C-main | yes (batch 1) |
| **X4** | Concept-balanced sampling (inverse-sqrt) | train, **conditional** | MAY (trigger) | D10 | DT | 40k | C-main | yes (batch 1) |
| **X5** | Horizontal flip + tile/caption remap | train | CUT | D15 | DT/LT | final steps | final | **no** (report-only) |
| **X6** | Schedule confirmation at the final config | train | CUT (Day 5 comparison replaces it) | D15 | DT/LT | final steps | final | **no** (report-only) |
| **X7** | Per-concept guidance-scale analysis | free | SHOULD | D15 | ANY | 0 | final | **no** |
| **X8** | Base width 48 / 96 (capacity) | train | CUT | D15 (if time) | DT | 40k ×2 | config v1 | **no** |
| **X9** | ResBlock dropout 0 / 0.2 | train, **conditional** | MAY (trigger) | D10 (if triggered) / D15 | LT | 40k ×1–2 | C-main | yes if D10 |
| **X10** | p_uncond 0.05 / 0.2 | train | CUT | D15 (if time) | ANY | 40k ×2 | config v1 | **no** |
| **X11** | DDIM steps 25 / 50 / 100 (+ CPU fast mode) | free | **P0** | D11 | ANY | 0 | config v1 | inference only |
| **X12** | Direct wide sampling vs sliding-window outpainting | free | P1 | D11 | ANY | 0 | config v1 | inference only |

\*X2 is free if the trainer keeps a second EMA shadow (`ema.extra_decays: [0.995]`, +73 MB of VRAM, well inside the cap). If doc 16's trainer doesn't support that, X2 becomes a separate 40k-step run.

**Conditional triggers.**
- **X4** runs only if the rare-concept exact match (doc 22 §3.3; rare set R = {rectangular, ceiling, cannon, upside down pipe, irregular, descending staircase}) on P-dev at C-main@w\* is **< 50 %** (D9).
- **X9** runs on Day 10 only if doc 19's diagnostics show memorisation: copy rate above the (calibrated) reference, or a dev-vs-train per-bucket loss gap growing for ≥ 3 consecutive evaluations. Run **0.2** then. Run **0.0** only if underfitting is diagnosed instead. Otherwise X9 waits for Day 15 (report-only).

---

## 3. Experiment cards

Every card gives the hypothesis, the exact change (YAML keys are indicative; match doc 17's schema, because the *semantic* change is what's binding), the metrics, the decision rule and the risks.

### X1 — Auxiliary cross-entropy on x̂₀ (M9 option; paper-faithful)
- **Hypothesis:** an ᾱ_t-weighted CE term on the x̂₀ logits sharpens per-cell tile decisions. That cuts broken structures and stray tiles without hurting adherence.
- **Change:** `loss.aux_ce_lambda: 0.001`, `loss.aux_ce_weighting: alphabar` (logit construction exactly as doc 13/16 define it).
- **Metrics:** S (primary); broken_rate, floor_valid, JS, Cscore (secondary).
- **Decision:** adopt if ΔS ≥ 0.01 with the CI above 0, **or** (structure override) broken_rate falls ≥ 2 pp while the ΔCscore CI stays above −0.02.
- **Risks:** unstable gradients at large t (division by √ᾱ_t in x̂₀) — the ᾱ_t weighting suppresses this. Watch for NaN (doc 25 R-3.3).

### X2 — EMA decay 0.995 (PDF) vs 0.999 (ledger default)
- **Hypothesis:** 0.999 (horizon ~1,000 steps) gives smoother, better samples than 0.995 (~200 steps) at 40k–60k-step budgets (doc 01 #16).
- **Change:** evaluate the 0.995 shadow (`ema.extra_decays: [0.995]`) at the same checkpoints as the 0.999 shadow. Otherwise `ema.decay: 0.995` in a separate run.
- **Metrics:** S; broken_rate (EMA mainly affects structure), the S-vs-step curve (lag).
- **Decision:** switch to 0.995 only if ΔS ≥ 0.01 with the CI above 0 at the selected checkpoint (0.999 stays default otherwise).
- **Report extra:** the "raw vs EMA" broken-rate table requested by the PDF (Phase 6.2) comes free from this run (evaluate the raw weights too at 3 checkpoints).

### X3 — Pooled text summary into the time embedding
- **Hypothesis:** a global caption summary added to the time embedding (as in Imagen's pooled conditioning) helps **count** adherence (bucket MAE ↓), which cross-attention alone represents weakly.
- **Change:** `model.pooled_text_to_temb: true`. Masked mean of the adapter outputs (B, 128) → `Linear(128, 256)` with **zero-initialised weights and bias** → added to the 256-d time embedding before the per-block FiLM projections. +33,024 parameters. Under CFG null dropout, the pooled vector comes from the null caption, consistent with the cross-attention context.
- **Metrics:** S; per-concept bucket MAE/bias and strict EM (secondary, decisive for interpretation).
- **Decision:** adopt if ΔS ≥ 0.01 with the CI above 0. If ΔS is inconclusive but mean bucket MAE falls ≥ 0.1 with no ΔCscore loss, record it as "promising" and re-test in the combination check (§6).

### X4 — Concept-balanced sampling (D9; conditional)
- **Hypothesis:** oversampling windows that contain rare concepts lifts rare-concept EM without hurting common concepts.
- **Change:** `data.sampling: concept_balanced`. Window weight `ω_i = (min_{k ∈ K(i)} f_k)^(−1/2)`, where `K(i)` = concepts in window i's caption and `f_k` = prevalence of concept k **in the train split**. Normalise to mean 1, **cap at 5.0**, then draw batches with `torch.multinomial(ω, 64, replacement=True, generator=g)` on the GPU (seeded, resumable state). Captions always have ≥ 1 phrase (F14), so `K(i)` is never empty.
- **Metrics:** S; rare-concept EM (doc 22 §3.3); common-concept EM; **copy rate on prompts that request rare concepts** (oversampled rare windows are the first to be memorised).
- **Decision:** adopt if ΔS CI ≥ 0, **and** rare-concept EM rises ≥ 10 pp, **and** rare-prompt copy rate rises ≤ 5 pp.
- **Note:** concepts with < 30 train windows (check `descending staircase`) may stay unlearnable. Document that instead of raising the cap.

### X5 — Horizontal flip with remap (report-only)
- **Hypothesis:** flipping doubles the data and helps the very rare *descending staircase* (104 windows in the whole corpus) by mirroring ascending ones (861). It may hurt rightward design semantics and therefore playability.
- **Change:** `data.flip_p: 0.5`. Flip the columns; remap tiles `1↔2` (`<`↔`>`) and `9↔10` (`[`↔`]`); in the caption swap `ascending`↔`descending`. **Test first:** recompute captions with the detector on 2,000 flipped windows and assert equality with the swapped caption for ≥ 99.5 % (`tests/data/test_flip.py`). If that fails, recompute captions on the fly instead of swapping words.
- **Metrics:** S; descending-staircase recall and EM; playable; per-game breakdown.
- **Reading:** report it in the ablation table. Never applied to the frozen model.

### X6 — Schedule confirmation at the final config (report-only)
- **Hypothesis:** the schedule chosen on unconditional models (M2) is also better for the conditional model.
- **Change:** `diffusion.schedule: <the other one>`; everything else = final; step budget = final's selected step count (cap 60k).
- **Decision:** none (post-freeze). If the other schedule wins by ΔS ≥ 0.02 (CI above 0), the report states it as a limitation and future work.
- **Pitfall:** compare **S and samples, never raw loss values**: ε-MSE scales differ by schedule (doc 25 §0.2: cosine's uniform-t average loss is ~1.5× linear's for the same quality).

### X7 — Per-concept guidance analysis (training-free, report-only)
- **Question:** is a single global w adequate, or do some concepts need much stronger or weaker guidance?
- **Procedure:** w ∈ {1, 2, 3, 5, 7.5, 10} × P-dev × 3 seeds (reuse the Day-14 sweep samples, no new sampling) → per-concept strict EM and recall vs w → `w*_k = argmax_w EM_k` (support ≥ 5).
- **Output:** a heat-map (concept × w) and a table of `w*_k`. If the `w*_k` span ≥ 3 grid steps, report "concept-dependent guidance" as future work (e.g., phrase-wise guidance). The frozen model is unchanged.

### X8 — Capacity: base width 48 / 96 (P2)
- **Hypothesis:** base 64 (18,312,013 params) is near the sweet spot. Base 48 (**10,725,117** params) underfits fine structure. Base 96 (**39,553,005** params) memorises more (copy rate ↑) for little S gain. Counts come from the reference build.
- **Change:** `model.base_channels: 48 | 96` (GroupNorm 8 groups still divides 48/96/192/384; heads of 4 → head dims 24/48).
- **Metrics:** S; train-vs-dev per-bucket loss gap; copy rate; sampling latency (affects G-P7).
- **Run only if** doc 19 diagnoses clear underfitting (→ 96) or overfitting (→ 48) on config v1. Otherwise skip and say so.

### X9 — Dropout 0 / 0.2 (conditional)
- **Hypothesis:** dropout 0.2 lowers memorisation (copy rate ↓, smaller dev-train gap) at a small adherence cost. Dropout 0 raises the dev C-score through memorisation, which is visible as a copy-rate rise.
- **Change:** `model.dropout: 0.0 | 0.2`.
- **Decision (if run on Day 10):** adopt 0.2 if the copy rate falls ≥ 3 pp and the ΔS CI ≥ −0.005 (protecting against memorisation is worth a negligible S cost).

### X10 — CFG null-dropout rate 0.05 / 0.2 (P2)
- **Hypothesis:** p = 0.2 trains a better unconditional branch, so guidance works better at high w. p = 0.05 leaves more capacity for the conditional branch.
- **Change:** `cond.p_uncond: 0.05 | 0.2`.
- **Fairness:** p_uncond shifts the optimal w, so compare each arm at **its own** best w ∈ {3, 5, 7.5} (as well as at the baseline w\*).

### X11 — DDIM steps 25 / 50 / 100 (+ CPU fast mode) (P0, training-free)
- **Question:** what does the CPU fast mode (H10: 25 steps, r = 0) cost in quality, and does 100 steps add anything?
- **Procedure (scene):** P-dev × 3 seeds at steps ∈ {25, 50, 100} (CRN: same x_T) → S, broken, playable, novelty, ms/sample (GPU and CPU).
- **Procedure (level):** L-single-dev at (50 steps, r = 2) vs (25 steps, r = 0) → seam ratio, verified rate, playable (level mode), time per 112-wide level on the laptop GPU and on CPU.
- **Decision:** fast mode is acceptable if ΔS ≥ −0.03, Δbroken ≤ +2 pp and seam ratio ≤ 1.3. Otherwise report it and propose a ledger change (e.g., 35 steps) only if the CPU time budget (< 45 s, G-P7) still holds.

### X12 — Direct wide sampling vs outpainting (training-free)
- **Hypothesis:** sampling 16 × W in one pass (the U-Net is fully convolutional, and attention takes any token count) produces globally consistent levels with no seams. But the model never saw W > 16, so structure quality may degrade (attention spread over more tokens, repetition).
- **Procedure:** L-single-dev (single-scene prompts only: direct sampling can't assign different prompts to regions) at W_gen ∈ {48, 80, 112}. Metrics: level C-score, broken, floor_valid, playable (level mode), seam ratio (pseudo-seams at outpainting positions), novelty, time.
- **Decision:** outpainting stays default (G1) unless direct wins on S_level by ≥ 0.02 with the CI above 0 **and** has ≤ the broken rate. Direct sampling is the last step of the seam rollback ladder either way (doc 25 R-5.1).

---

## 4. Seeds policy

| Seed | Value | Used for |
|---|---|---|
| `train_seed` | **20261117** | model init, data order, augmentation and dropout RNG, for **every** baseline and ablation arm and every CV fold (same seed per fold; the data differ) |
| replicate train seeds | 20261118, 20261119 | (a) the **noise-floor run**: C-main re-trained with seed 20261118 (P1, Day 10 if a GPU slot is free) to measure seed-to-seed ΔS; (b) replicating the top-2 batch-1 candidates when their ΔS lies within ±0.02 |
| sampling seeds | doc 22 §2.3 formula | identical x_T across arms (CRN) |
| bootstrap RNG | 12345 | every CI and paired test |

**Noise-floor rule:** if the measured seed-to-seed |ΔS| of the baseline is larger than an arm's ΔS, that arm's "win" is noise. The decision then defaults to the simpler option, whatever the bootstrap CI says (the bootstrap only captures sampling noise, not training noise).

---

## 5. Compute budget

Planning throughput (**replace with Day-4 `hw_report.json` values**): DT 15 it/s, LT 9 it/s at batch 64 (bf16), the single planning assumption of ledger H11, plus ≈ 30 s per dev evaluation (≈ 350 samples incl. checker) every 2k steps.
`hours(steps) = steps / (it_s × 3600) + (steps / 2000) × 0.5 / 60`

| Job | Steps | DT h | LT h | Notes |
|---|---|---|---|---|
| batch-1 arm (X1, X3, X4, X9, X2-separate) | 40k | ≈ 0.9 | ≈ 1.1 | each |
| noise-floor replicate of C-main | 40k (cap) | ≈ 0.9 | ≈ 1.1 | P1 |
| combination check (config v1 candidate) | 40k | ≈ 0.9 | ≈ 1.1 | §6 step 3 |
| X11 / X12 / X7 | 0 | ≈ 0.3–0.6 | ≈ 0.4–0.8 | sampling only (X11 includes a CPU timing pass) |
| CV fold (final config) | ≤ 60k | ≈ 1.4 | ≈ 1.7 | ×5 folds |
| X5 / X6 (final steps) | ≤ 60k | ≈ 1.4 | ≈ 1.7 | each, report-only |
| X8 (48 / 96) | 40k | ≈ 0.7 / 1.6 | ≈ 0.9 / 2.0 | P2 |

**Day plans:**
- **Day 10 🌙:** DT: X1 → X4 (if triggered) → noise-floor replicate. LT: X3 → X9 (if triggered). X2 is free from C-main's dual EMA. ≈ 3–4 h per GPU. **Day 10 morning (or Day 11):** combination check if ≥ 2 arms were adopted.
- **Day 11:** X11 + X12 (sampling only, ≈ 1 h total on either GPU).
- **Day 15 🌙:** CV folds 1–3 on DT (≈ 4.2 h), folds 4–5 on LT (≈ 3.4 h). Then X6 (DT) and X5 (LT), ≈ 1.4–1.7 h each. X7 runs on whichever GPU frees first. X8 and X10 only if more than 3 GPU-hours remain.

---

## 6. Decision procedure

**Config v1 (Day 10, after batch 1):**
1. For each arm: paired ΔS vs C-main (40k-capped) at **C-main's w\*** (primary) and at the arm's own best w ∈ {3, 5, 7.5} (secondary), with the 95 % cluster-bootstrap CI.
2. **Adopt** an arm if ΔS ≥ 0.01 and the CI is above 0, and its ΔS beats the noise floor (§4). Card-specific overrides apply (X1 structure override, X9 memorisation override).
3. **Two or more adopted →** train the **combination** for 40k steps and compare it with the best single arm. Keep the combination only if it beats that arm with the CI above 0; otherwise keep the best single arm (guards against interactions).
4. **No arm adopted →** config v1 = C-main. That's a legitimate, reportable outcome.
5. Record the decision, ΔS values and run IDs in `REGISTRY.csv` and the ledger change log (§14).

**Final (Day 14):** config v1 + the X11/X12 inference choices (DDIM steps stay 50 unless the ledger is changed; outpainting vs direct per X12), re-swept w\* (doc 22 §6.2) → `configs/final.yaml` → `FROZEN.json`.

---

## 7. How to read the results (and the traps)

1. **Health before metrics.** Check per-t-bucket loss curves, NaN events and `events.jsonl` (fallbacks such as gradient accumulation) before trusting an arm's metrics. A crashed-and-resumed arm must show a continuous loss curve (doc 25 R-0.2).
2. **Matched selection.** Compare arms at their selected checkpoints (same rule), **and** look at the S-vs-step curves. An arm that peaks later than the 40k cap is under-budgeted, not worse: note it.
3. **C-score saturates** (absent/absent credit, doc 22 §3.1 Q1). Always read the per-concept strict EM and the ΔC-vs-null next to it.
4. **Gains from memorisation aren't gains.** If an arm raises Cscore while the copy rate rises or novelty_med falls (typical for dropout 0, larger width, oversampling), treat the gain as suspect. The capped novelty term in S only partly protects against this.
5. **w interactions.** X3, X10 and to some extent X1 move the optimal w. Use the secondary own-best-w comparison before rejecting an arm.
6. **Rare concepts have tiny support.** A +20 pp EM on a concept with 6 dev prompts is about one prompt. Report support, and aggregate rare concepts (doc 22 §3.3).
7. **Per-game split.** SMB2 is harder (more gaps, upside-down pipes) and has a different natural copy rate (14 % vs 21 % LOLO). An arm that helps only SMB1 shows that in the per-game table.
8. **Latency is a metric for the app.** X8-96 and X11-100 can win on S but break G-P7 (< 10 s per 112-wide level). The app config may then differ from the best-S config only if the ledger records it. Default: one config for both (timeline: "same frozen checkpoint").
9. **No test peeking.** Questions like "what would X do on test?" are answered with CV (Day 15), never with test prompts.

---

## 8. Execution

```yaml
# configs/ablations/X1_aux_ce.yaml        (key names must match doc 17; semantics are binding)
experiment: X1
base: configs/cond_main.yaml
overrides:
  loss.aux_ce_lambda: 0.001
  loss.aux_ce_weighting: alphabar
  train.max_steps: 40000
  run.tags: [ablation, batch1]
```
```
python scripts/launch.py --config configs/ablations/X1_aux_ce.yaml --machine desktop_3060ti
python scripts/analyze_ablation.py --baseline runs/<C-main-id> --cap-steps 40000 \
       --arms runs/<X1-id>,runs/<X3-id> --set p_dev --w <w*> --secondary-w 3,5,7.5 --out reports/ablations/batch1/
```
Files: `configs/ablations/X1_aux_ce.yaml`, `X2_ema995.yaml`, `X3_pooled_temb.yaml`, `X4_concept_balanced.yaml`, `X5_flip.yaml`, `X6_schedule_swap.yaml`, `X7_percept_w.yaml` (eval-only), `X8_base48.yaml`, `X8_base96.yaml`, `X9_dropout0.yaml`, `X9_dropout02.yaml`, `X10_puncond005.yaml`, `X10_puncond02.yaml`, `X11_ddim_steps.yaml` (eval-only), `X12_direct_wide.yaml` (eval-only).

`REGISTRY.csv` columns: `exp_id, arm, run_id, config_hash, train_seed, machine, steps_at_best, w, S, S_ci_lo, S_ci_hi, cscore, broken, playable, novelty_med, copy_rate, dS, dS_ci_lo, dS_ci_hi, decision, decided_on, notes`.

---

## 9. Reporting template (report §Ablations)

| Exp | Change | ΔS [95 % CI] | ΔCscore | Δbroken | Δplayable | Δnovelty med | Δcopy | Decision |
|---|---|---|---|---|---|---|---|---|
| X1 | aux CE λ = 1e-3 | | | | | | | adopt / reject |
| X2 | EMA 0.995 | | | | | | | |
| X3 | pooled text→temb | | | | | | | |
| X4 | concept-balanced | | | | | | | (if run) |
| X9 | dropout 0.2 | | | | | | | (if run) |
| X11 | DDIM 25 / 100 | | | | | | | inference |
| X12 | direct wide | | | | | | | inference |
| X5 / X6 / X7 / X8 / X10 | (post-freeze) | | | | | | | report-only |

## 10. Deliberately not ablated
T = 1000, the 16×16 window, AdamW and the LR schedule shape, batch 64 (fixed by H4), MiniLM vs other encoders (studied by the paper; out of scope), tokenizer length 48, CFG convention, x̂₀ clipping. Changing any of them needs a ledger change and a reason that comes from a diagnosed failure (doc 25), not from curiosity.
