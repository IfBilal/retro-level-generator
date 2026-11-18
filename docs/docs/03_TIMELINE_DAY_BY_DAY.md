# 03 — Day-by-Day Timeline (Day 0 = Wed 18 Nov 2026 → Day 18 = Sun 6 Dec 2026, submission)

> **18-day plan (rewritten 2026-11-18).** The earlier 21-day plan (Day 0–20) ended on 8 Dec, after the 6 Dec GCR deadline. Calendar: Day N = 18 Nov + N days. Presentations are 7–9 Dec (after Day 18). Never cut: level-grouped split, playability checker (and its validation), one-shot test, the MVP game, Streamlit Cloud deployment, report/slides/evidence/zip. Cuts and demotions: doc 02 §14.
>
> Backbone of the project. Each day lists: **Track** (M = Model/ML, G = Game/App, B = Both), **Machine hint** (DT = desktop RTX 3060 Ti, LT = laptop RTX 4050, ANY = either; by the min-VRAM contract H2–H4 every job can run on either machine), **Tasks** (checkbox list Claude Code executes in order), **Artifacts** (must exist at end of day), **Exit criteria** (gate or check), and **If blocked** (pre-agreed fallback). Hours are focused-work estimates; GPU wall-clock runs unattended (overnight where marked 🌙).
>
> Daily ritual (every day, both machines): `git pull` → `python scripts/verify_env.py --quick` → read `HANDOFF.md` → work → update `PROGRESS.md` (date, done, blocked, next) → `git push` → if a run is in flight, follow `07_DUAL_GPU_HANDOFF_PROTOCOL.md` before switching machines.

---

## Phase map (PDF phases → days)

| Phase | Name | Days | Gate |
|---|---|---|---|
| 0 | Environment + repo + assets fetch | D0 | `verify_env` green on both machines |
| 1 | Data acquisition, audit, splits, loader, renderer | D1–D2 | **G-P1** |
| 2 | Scheduler + forward/reverse maths | D3 | **G-P2** |
| 3 | Unconditional U-Net baseline | D4–D6 | **G-P3** |
| 4 | Text conditioning + CFG | D7–D10 | **G-P4** |
| 5 | Full-level generation (outpainting, framing, sanitizer) | D11 | **G-P5** |
| 5b | Playability checker (built on the G track D5–D6) **and validated on D7, before the `playable` term is used in any selection** | D5–D7 | **G-P5b** |
| 7a/7b | Streamlit app + browser game ("Type it, play it" MVP) | D3, D5, D9 (G track) + D12–D13 | **G-P7** |
| 6 | Evaluation, freeze (D14), one-shot test (D15), optional CV | D14–D15 | **G-P6** |
| 8 | **Streamlit Cloud deployment** (dry run D12, final D14) | D12, D14 | **G-P8** |
| — | Report, slides, evidence, hardening, viva, zip, upload | D16–D18 | Doc 35 submission checklist |

The report is written section by section on the dated days marked **[W]** (doc 32 §1.2), not at the end. Phase 7 runs **before** Phase 6 on purpose. The app has to exist before the final numbers are frozen, so the demo and the reported numbers come from the **same frozen checkpoint**.

---

## Day 0 (Wed 18 Nov) — Environment, repo scaffold, autonomous asset fetch  (B · DT + LT · ~6 h)
**Docs:** 06, 05, 07, 26, 34
- [ ] On **both** machines: install Python 3.12, create the venv, install torch per ledger V2 (check `nvidia-smi` driver first), install `requirements.txt`.
- [ ] Create repo `text2mario` with the exact tree in doc 05; copy this pack into `docs/`, `CLAUDE.md` to root, `reference/ref_unet.py`.
- [ ] Write `scripts/verify_env.py`: prints torch/CUDA/driver/GPU name/VRAM, runs a 5-second bf16 matmul + SDPA test, applies the 4.5 GiB cap (H2–H3) and checks it with a deliberate 5 GiB allocation that **must fail** with OOM.
- [ ] Write `configs/machine/desktop_3060ti.yaml` and `configs/machine/laptop_4050.yaml`.
- [ ] `scripts/fetch_assets.py` → Phaser 3.90.0 + Kenney Pixel Platformer (CC0) into `app/static/`, with sha256 in `assets/ASSET_MANIFEST.json`; fallback `make_placeholder_sprites.py` (doc 26).
- [ ] **Verify versions (ledger V1, V2, V4):** `pip index versions torch`, `pip index versions streamlit`, Python availability on Streamlit Community Cloud (deployment dialog, advanced settings); record the chosen pins in the ledger. Do not rely on any version claim in earlier drafts.
- [ ] **Cloud account check:** log in to Streamlit Community Cloud with the group GitHub account, confirm it can read the repo (or create the public deployment mirror, ledger CL8), note the platform's stated memory limit and sleep behaviour in `env/cloud_notes.md`.
- [ ] **[W]** Report skeleton `report/` (sections of doc 32 §1.1) and `evidence/` folder; paste doc 02 §2 into the report as the problem/justification/objectives draft.
- [ ] **Team:** fill doc 36 §1 (ownership table) with names; start the daily 20-minute walk-through (doc 36 §3) from Day 1.
- [ ] Check Java: `java -version`. Record in `env/capabilities.json` (decides doc 21 tier 2).
- [ ] Initialise `PROGRESS.md`, `HANDOFF.md`, `runs/` (git-ignored), the sync folder (doc 07).
- **Artifacts:** `env/lock_desktop.txt`, `env/lock_laptop.txt` (pip freeze), `env/capabilities.json`, the asset manifest.
- **Exit:** `verify_env.py` green on **both** machines with **the same torch minor version**.
- **If blocked:** driver too old for CUDA 13 and can't be updated → pin both to torch 2.14 + cu126 (V2). Kenney download fails → placeholder sprites (A2) and retry on Day 1.

## Day 1 (Thu 19 Nov) — Phase 1a: acquisition + audit + EDA  (M · ANY · ~6 h)
**Docs:** 08, 09
- [ ] `scripts/fetch_data.py`: clone MarioDiffusion at a pinned commit into `data/raw/MarioDiffusion/` (isolated; never import its code, read files only); copy the 1 regular JSON + `smb.json`; write `data/MANIFEST.json` (commit SHA, sha256).
- [ ] `scripts/audit_dataset.py`: assert **all F1–F17 facts** of the ledger; reconstruct chains → `data/processed/index.parquet` with columns `idx, level_id, game, col_offset, caption, n_phrases, scene_hash, is_dup`.
- [ ] `src/text2mario/data/tiles.py`: the single tile table (F6) + char/ID/colour maps; test against `smb.json`.
- [ ] EDA report `reports/eda/EDA.md` + figures: tile frequencies (global, per game, per row), caption-length histogram, phrase frequencies, concept co-occurrence matrix, per-level concept profile heatmap (42 × 16 concepts), duplicate list.
- [ ] Write the caption detector **from the caption-grammar description** (doc 12 §2 and the paper's published phrase list) in `src/text2mario/captions/detector.py`. **Do not port or transcribe MarioDiffusion's `assign_caption` code** (ledger O3, CLAUDE.md §2.10). Then **check it against the stored captions**: target 100 % of the 7,687 scenes (set-equality of phrases), ≥ 99.5 % with a documented exceptions file. Mismatches are listed and explained, never silently patched toward their code. Keep the MIT attribution for the *dataset* in `DATA_CARD.md`.
- [ ] **[W]** One paragraph in the report's originality box on how the detector was built and verified.
- [ ] (G track, LT, 1 h) Tile-mapping contact sheet for the Kenney atlas → `assets/tile_map.json` (doc 26).
- **Exit:** audit script passes; detector agrees with the stored captions on 100 % (or ≥ 99.5 % + exceptions file).
- **If blocked:** detector < 100 % → list mismatches, fix, and do not proceed to evaluation work until it is ≥ 99.5 % with documented exceptions.

## Day 2 (Fri 20 Nov) — Phase 1b: splits, leakage tests, loader, renderer  (M · ANY · ~7 h)
**Docs:** 10, 11, 12 (prompt sets only)
- [ ] `scripts/make_splits.py`: level-grouped selection (doc 10 algorithm, seed 20261117) → `data/splits/split_v1.json` (level_ids for train/dev/test + the 5 CV folds over the 36 non-test levels).
- [ ] Leakage tests (`tests/test_splits.py`): no level in two splits; every concept present in every split (≥ 5 windows each, except documented rare ones); NN-Hamming between test and train windows reported; reused-segment exceptions documented.
- [ ] Write `data/splits/LOCK` = sha256(split_v1.json). CI test fails if the hash changes.
- [ ] Write the **P-casual** set: 60 handwritten casual prompts with expected attribute dicts → split 30/30 dev/test; split the **paraphrase lexicon** into train/test variants (doc 12).
- [ ] `src/text2mario/data/dataset.py`: GPU-resident uint8 tensors, on-the-fly one-hot → {−1, +1}, seeded batch sampler with a resumable state, caption augmentation pipeline (D8).
- [ ] `src/text2mario/render/renderer.py`: `tensor_to_grid`, `grid_to_image(tile_px, skin)`, `grid_to_vglc_text`, `grid_to_json`.
- [ ] **[W]** Report dataset section (source, features, target, preprocessing, limitations, split + leakage) from the audit/EDA outputs.
- [ ] **Gate G-P1:** 10 random DataLoader items → render → saved to `reports/gates/P1/`. Check orientation (floor at the bottom, pipes upright, `<` left of `>`), and that the round trip `grid → onehot → argmax` is the identity.
- **Exit:** G-P1 passes (all checks scripted in `scripts/gate_p1.py`).

## Day 3 (Sat 21 Nov) — Phase 2: scheduler  (M · ANY · ~5 h) ‖ G track: game skeleton (LT · ~3 h)
**Docs:** 13, 24 · (G) 27
- [ ] `src/text2mario/diffusion/scheduler.py`: linear + cosine; buffers for β, α, ᾱ, √ᾱ, √(1−ᾱ), posterior variance (both β̃ and β choices), posterior mean coefficients, DDIM timesteps; all in fp64, cast at use.
- [ ] `q_sample`, `predict_x0_from_eps`, `p_mean_variance`, `ddpm_step`, `ddim_step(eta)`, `renoise(x_prev, t_prev → t)`.
- [ ] Property tests (hypothesis): shapes and dtypes; monotonic ᾱ; oracle-ε DDIM exact reconstruction (fp64 MSE < 1e-6, argmax 100 %); empirical mean/std of x_t vs theory for t ∈ {0, 250, 500, 750, 999}.
- [ ] **Gate G-P2:** render corruption strips at t = 0, 250, 500, 750, 999 for both schedules → `reports/gates/P2/`; numeric checks per ledger §9.
- [ ] (G) Phaser skeleton: loads a **real training level** (export of level 0 via renderer → JSON schema doc 29), arrow keys + jump, tile collisions, camera follow. `app/static/game/physics.json` = the shared physics constants (also read by the playability checker).
- **Exit:** G-P2 passes; the game skeleton is playable for one real level.

## Day 4 (Sun 22 Nov) — Phase 3a: U-Net + trainer core + cross-machine resume drill  (M · LT first, then DT · ~7 h)
**Docs:** 14, 16, 17, 07
- [ ] `src/text2mario/models/unet.py` exactly per doc 14 (use `reference/ref_unet.py` as the oracle; parameter count must equal **18,312,013** for the default config with the cross-attention ON, or the documented unconditional variant count).
- [ ] Unit tests: output shape for W ∈ {16, 24, 112}; gradient reaches every parameter; time-embedding sensitivity; mask handling.
- [ ] `src/text2mario/train/train.py`: bf16 autocast, AdamW param groups (no decay on norm/bias/emb), warm-up + cosine LR, grad-clip, EMA, deterministic per-t-bucket validation loss, atomic checkpoints, resume (full state incl. RNG + sampler), JSONL/CSV/TensorBoard logging, NVML temperature watchdog (H9), VRAM cap (H3).
- [ ] **Overfit-one-batch test:** 8 fixed scenes, fixed t, 1,500 steps → loss < 0.01. (Proves the optimiser path works.)
- [ ] **Min-VRAM measurement on the LAPTOP:** batch 64, 200 steps → `hw_report.json` (peak allocated/reserved, it/s). Must be ≤ 4.5 GiB. **Also record, on both GPUs, the per-pass forward time (batch 2, 16×16 and 16×112) in `hw_report.json`**: the first GPU latency measurement, input to the G-P7 pin on Day 12 (ledger measure-first rule). Replace the it/s planning assumptions (ledger H11: DT 15, LT 9) with the measured values.
- [ ] **[W]** Report algorithm-description draft (forward process, ε-loss, U-Net, CFG) and the first walk-through of the model code by its owner to the other members (doc 36).
- [ ] **Resume drill (simplified, AT-08 SHOULD; doc 07 minimal form):** train 1,000 steps on DT → handoff → resume on LT for 500 steps → loss curve continuous (no spike > 2× the running mean), step counter, LR and EMA continuous.
- **Exit:** all tests green, drill passes, hw_report recorded for both machines.

## Day 5 (Mon 23 Nov) — Phase 3b: unconditional training  (M · DT + LT in parallel 🌙 · ~4 h hands-on) ‖ G: playability checker v0 (~4 h)
**Docs:** 18, 19, 21
- [ ] Run **U-lin** (linear) on DT and **U-cos** (cosine) on LT, 15k steps each (configs `uncond_linear.yaml`, `uncond_cosine.yaml`).
- [ ] `src/text2mario/eval/structure.py`: floor validity, broken pipes/cannons, tile-distribution JS divergence, NN-Hamming novelty + copy rate (vectorised on GPU against the train set).
- [ ] Eval at 5k / 10k / 15k with 256 EMA samples (DDIM 50). **Selection uses `S_P3` without the `playable` term (`S_noplay`)**, because the checker is not validated yet (ledger E4). This linear-vs-cosine comparison replaces ablation X6.
- [ ] (G) `src/text2mario/playability/checker.py` v0: tile-graph A* with jump arcs computed from `physics.json` (doc 21). Run it on **all 42 real levels** (full width) and on 16-wide windows → baseline solvable rate (expect high-80s to low-90s % for windows, cf. F18; an expectation to be measured, not a result).
- [ ] Start the **50-level hand-label set** for G-P5b: levels only from train/dev real levels and the Day-5 unconditional samples (never test levels); each labelled "beatable / not" by two members independently (doc 21 §8).
- **Exit:** **Gate G-P3** on the better schedule. Record the schedule choice in the ledger change log (M2).
- **If blocked:** follow doc 25 rows R-3.x (sky dominance, noise, broken pipes). Day 6 is the rollback buffer.

## Day 6 (Tue 24 Nov) — Phase 3 buffer / Playability v1  (B · ANY · ~6 h)
- [ ] If G-P3 failed → execute the rollback (doc 25), retrain the winning schedule only, re-gate.
- [ ] If G-P3 passed → playability checker v1: enemies as hazards, cannons as solid, gap death, MarioEval.jar bridge (`grid_to_vglc_text` → `java -jar MarioEval.jar astar ...`) if Java is available (doc 21 tier 2); agreement study scaffold.
- [ ] Produce the "unconditional sample sheet" figure (64 samples) for the report. Finish the hand-labels (disagreements settled by playing the level in the game skeleton).
- **Exit:** G-P3 passed (hard requirement before Day 7).

## Day 7 (Wed 25 Nov) — Phase 4a: text conditioning wiring  (M · LT · ~6 h) ‖ G: **checker validation G-P5b** (CPU)
**Docs:** 15, 12, 21
- [ ] `src/text2mario/text/encoder.py`: frozen MiniLM (bf16, eval, `requires_grad_(False)`), tokeniser max_len 48, LRU cache keyed by caption string, null embedding of "", adapter MLP + LayerNorm.
- [ ] Caption augmentation in the dataset (shuffle, phrase dropout, paraphrase substitution from the train lexicon, CFG null dropout).
- [ ] **Test:** `assert adapter params ⊂ optimizer.param_groups` and MiniLM params ∉ the optimiser (PDF rollback #3 turned into a test).
- [ ] **Test:** null-dropout empirical rate 0.10 ± 0.01 over 10k samples.
- [ ] Smoke conditional run, 3k steps: loss decreases; cross-attention maps aren't uniform (entropy below the uniform maximum).
- [ ] **(G) Gate G-P5b, checker validation, moved here from Day 11 (ledger E4/G-P5b):** run the fast checker on the 300 levels (windows + real train/dev levels + Phase-3 samples) and compare with MarioEval.jar (agreement ≥ 85 %), or, if Java is unavailable, with the 50 hand-labelled levels (agreement ≥ 90 %) → `reports/gates/P5b/RESULT.json`. **Only after `pass: true` may the `playable` term enter S or any selection.** If it fails: doc 25 R-5.x and keep `S_noplay`.
- [ ] **[W]** Report testing-section skeleton (gates, pytest inventory) and the checker-validity paragraph.
- **Exit:** tests green; smoke run healthy; G-P5b recorded (pass, or documented fallback to `S_noplay`).

## Day 8 (Thu 26 Nov) — Phase 4b: main conditional run  (M · DT 🌙 · ~5 h hands-on) ‖ G: prompt normalizer v1 (LT)
**Docs:** 16, 18, 22, 12
- [ ] Launch **C-main** (`cond_main.yaml`), up to 60k steps, dev evaluation every 2k steps (C-score on P-dev at w = 5 with 3 seeds, broken rate, fast-checker playability, novelty), early stopping on composite score E4 (patience 5). The `playable` term is included **only because G-P5b passed on Day 7**; otherwise run with `S_noplay`.
- [ ] `src/text2mario/eval/adherence.py`: C-score (paper formula) + per-concept exact match, using the Day 1 detector.
- [ ] (G) `src/text2mario/text/normalizer.py` v1: synonym → canonical, digits/number words → buckets, multi-scene splitter; tested on P-casual-dev (doc 12 targets).
- [ ] **[W]** Report methodology (training protocol, text path) and pipeline diagram v1 (Fig 1).
- **Exit:** run in flight and healthy at the evening check (loss per t-bucket falling, dev composite improving).

## Day 9 (Fri 27 Nov) — Phase 4c: CFG sweep + gate  (M · ANY · ~5 h) ‖ G: game mechanics (LT · ~4 h)
- [ ] Select the best checkpoint by dev composite; sweep w ∈ {1, 2, 3, 5, 7.5, 10} on P-dev (3 seeds) → curves: C-score, broken rate, playability, novelty vs w.
- [ ] **PDF contrast test:** "full floor with three pipes and no enemies" vs "staircase with gap and several enemies" (normalised to canonical) × 8 seeds → figure.
- [ ] **Gate G-P4.** Record default w in the ledger.
- [ ] (G) Game: enemies (walker AI, stomp/kill), question-block bump → coin, breakable bricks (bump), coins, pipes, cannons (static solid in the MVP), pits = death, respawn, flag = win screen, HUD (time, coins).
- **If blocked:** doc 25 rows R-4.x (ignores prompt, counts wrong, rare concepts).

## Day 10 (Sat 28 Nov) — Phase 4d: ablation batch 1  (M · DT + LT in parallel 🌙 · ~4 h)
**Docs:** 23
- [ ] Run **X1 (aux CE λ = 0.001)** on DT (40k steps, same seeds, dev eval only). X2–X4 (X4 only if rare-concept exact match < 50 %) are MAY; X5–X10 are cut (doc 23 scope note). Meanwhile on LT/CPU: start `generate/full_level.py`.
- [ ] Choose **config v1** = the best dev composite; tie-break by playability, then simplicity. Record in the ledger.
- **Exit:** config v1 chosen and its checkpoint retrained or continued to completion.

## Day 11 (Sun 29 Nov) — Phase 5: full-level generation + playability gate  (M · ANY · ~7 h)
**Docs:** 20, 21
- [ ] `src/text2mario/generate/full_level.py`: scene-prompt splitter → sliding window (stride 8) with RePaint known-region replacement + r = 2 resampling (first 60 % of steps) → framing (G3) → sanitizer detect (+ optional repair) → checker → window resampling loop (G5).
- [ ] Seam metric (doc 20 §seam): column-transition NLL at seams vs interiors, using a column bigram model fitted on train levels.
- [ ] **Gate G-P5** (seams, sanitizer). (G-P5b was passed on Day 7; re-run the agreement on 30 generated full levels as a non-gating monitor.)
- [ ] **[W]** Report generation and playability-checker subsections.
- **If blocked:** escalation ladder in doc 25 R-5.x (stride 4, r = 5, direct wide sampling).

## Day 12 (Mon 30 Nov) — Phase 7a: Streamlit studio skeleton  (G · LT · ~7 h)
**Docs:** 28, 29
- [ ] `app/streamlit_app.py`: prompt box, w slider, width select (16/48/80/112), seed, "fast mode" toggle (auto-on without CUDA), Generate → denoising animation (frames every 5 DDIM steps), final PNG, normalised-prompt display.
- [ ] `st.cache_resource` for model + encoder; warm-up generation at start-up.
- [ ] Exports: PNG, JSON (schema v1), CSV, VGLC .txt.
- [ ] **Measure first, then pin (ledger G-P7):** time the full pipeline (cold start; generate + verify at widths 48 and 112, GPU full quality and CPU fast mode; p50/p95 over 20 prompts) → `reports/app/latency.json`; pin the G-P7 thresholds in the ledger (measured median × 1.25) with a §14 row.
- [ ] **Cloud dry run (G-P8, ledger §3b):** pinned `requirements.txt` (CPU torch, no dev packages), `.streamlit/config.toml`; export the candidate EMA checkpoint to `app/model/t2m_ema.safetensors`; bundle MiniLM fp16 + tokenizer in `app/model/minilm/`; add `T2M_PROFILE=cloud`, the time guard and the showcase fallback (stub with 3 levels); deploy to Streamlit Community Cloud; record build log, memory, cold start and width-48 generation time in `reports/app/cloud_dryrun.json`; pin the cloud numbers (CL5, CL6) in the ledger.
- **Exit:** thresholds pinned; generation works on GPU and CPU; the app is live on a cloud URL (even if slow), or the blocker is logged with a decided workaround (smaller checkpoint, cached-only mode).

## Day 13 (Tue 1 Dec) — Phase 7b: "Type it, play it" MVP  (G · LT · ~7 h)
**Docs:** 27, 28, 29
- [ ] Game component: level JSON + atlas embedded into the Phaser HTML (inline/base64; doc 27 §embedding), rendered through the embed shim (doc 28: `st.iframe` or `st.components.v1.html`), with a "click to focus" overlay; **test it on the cloud URL too** (doc 27 cloud iframe case).
- [ ] "Verified beatable ✓ / not verified" badge from G5; attempt counter.
- [ ] "Download playable HTML" (single self-contained file).
- [ ] End-to-end test script: 20 prompts → generate → checker → the HTML loads in headless Chromium (Playwright smoke test: the canvas exists and the player sprite moves on simulated key presses).
- [ ] **[W]** Report implementation, app and testing sections (evidence folder skeleton filled by `scripts/collect_evidence.py`).
- **Exit:** **MVP done.** Gate G-P7 timing checks (against the thresholds pinned on Day 12) pass on the laptop.

## Day 14 (Wed 2 Dec) — Phase 6a: full dev evaluation + freeze + final cloud deploy  (M+G · ANY · ~7 h)
**Docs:** 22, 19, 28
- [ ] Full metric suite on P-dev, P-rand, P-casual-dev, 3 seeds, **level-clustered bootstrap CIs** (ledger E3), per-level table, per-game breakdown (SMB1 vs SMB2), per-concept table; error analysis (20 worst prompts, with renders). State in every table that 5 dev levels give wide intervals.
- [ ] Overfit/underfit dashboard (doc 19): train vs dev loss per t-bucket, copy rate vs steps, novelty vs the real-dev reference.
- [ ] **Freeze:** `configs/final.yaml` + checkpoint sha256 → `FROZEN.json`; commit `reports/final/eval_plan.json` (pre-registration, doc 22 §6.5). No tuning after this point.
- [ ] **Export and deploy (G-P8 final):** `scripts/export_cloud_ckpt.py` writes `app/model/t2m_ema.safetensors` + sidecar from the frozen EMA weights; generate the 12 showcase levels with the frozen model and **human-play each to the flag**; push; redeploy on Streamlit Cloud; run `scripts/gate_p8.py` against the public URL; save URL, commit SHA and screenshot in `evidence/deployment/`.
- [ ] **[W]** Slides draft (12, doc 32 §9.1) from the report skeleton.
- [ ] **(MAY) overnight 🌙:** 5-fold grouped CV on both GPUs if free (read on Day 15).
- **Exit:** freeze committed and tagged `v0.9-frozen`; **G-P8 passes** (or the CL6 cached-only fallback is the documented state).

## Day 15 (Thu 3 Dec) — Phase 6b/6c: ONE-SHOT test + optional CV read-out  (M · ANY · ~4 h)
- [ ] Verify `LOCK`, `FROZEN.json` and `eval_plan.json` hashes → run `scripts/eval_test.py` **once** on P-test, P-rand (fresh seed), P-casual-test; MarioEval.jar on a 300-level subset.
- [ ] Write `reports/final/RESULTS.md` (numbers + level-clustered CIs + per-level table + references). The test run's log is committed as-is, with no re-runs. (If a bug is found in the *metric code*, fix it, re-run, and document both runs.)
- [ ] (MAY) aggregate CV: mean ± std; if CV was not run, say so in the report and rely on the per-level table.
- [ ] **[W]** Report results prose and the final limitations list; update the results slides.
- [ ] **Viva-drill deadline:** every member has typed and run every drill of doc 36 §5 by the end of today.
- **Exit:** **Gate G-P6.**

## Day 16 (Fri 4 Dec) — Report, slides, evidence, model/data cards, demo hardening  (B · ~8 h, two parallel tracks)
**Docs:** 32, 35
- [ ] (Writing track) `scripts/make_figures.py` → all figures/tables; `MODEL_CARD.md`, `DATA_CARD.md`; assemble the full report (every statement-mandated section, doc 32 §1.1); **12-slide deck final** (doc 32 §9.1) with provenance in speaker notes; `check_provenance.py` green.
- [ ] (Writing track) **Evidence-of-evaluation folder** via `scripts/collect_evidence.py` (doc 35 §3).
- [ ] (App track) Offline run (Wi-Fi off) from a cold boot; CPU-only rehearsal (`CUDA_VISIBLE_DEVICES=""`); browser matrix Chrome + Edge, 1080p and 768p; re-check the **cloud URL** (cold start after sleep, showcase fallback).
- [ ] (App track) Showcase levels verified and human-played; standalone HTML copies on USB.
- **Exit:** report draft complete, slides final, evidence folder complete, app hardened.

## Day 17 (Sat 5 Dec) — Buffer (morning), viva rehearsal, zip build and check  (B · ~7 h)
- [ ] **Buffer first (morning):** absorb slips; nothing new is started.
- [ ] Proof-read the report; build the PDF; slides PDF.
- [ ] Two mock vivas with doc 32 §7 and doc 36 §4; time the live demo (< 3 min). **Each member sets up the project from the zip on their own laptop and runs the app** (rubric: otherwise −20 %).
- [ ] **Build the single submission zip** (doc 35 §4), unzip it in a clean directory, install, run the fast tests, start the app, play one level; check size and contents against the checklist. Tag `v1.0`.
- **Exit:** doc 35 checklist all ticked except the upload.

## Day 18 (Sun 6 Dec) — SUBMISSION DAY
- [ ] One member uploads the **single zip** to GCR (statement: GCR only, one submission per group). Target: morning; the statement gives the date only, so confirm the cut-off hour on GCR.
- [ ] Verify the upload (file opens, size correct); keep a copy and its sha256 in `evidence/`.
- [ ] Re-check the cloud URL loads. Pack laptops, chargers and USB for the 7–9 Dec presentations.

---

## What changed from the 21-day plan (2026-11-18)
Cut or merged to fit Day 18 = 6 Dec: the separate buffer day (old Day 19); old Days 17 + 18 merged into Day 16; old viva day now Day 17 with the zip build; ablation batch 2 and CV demoted to optional (old Day 15); one-shot test moved from Day 16 to Day 15. Added: Streamlit Cloud deployment (Days 12, 14), report/slide/evidence/zip tasks, checker validation on Day 7, detector-from-grammar rule, measure-first rule for latency. Details: doc 02 §14, ledger §14.

## If the schedule slips (pre-agreed cuts, in order)
1. Optional CV and X7. 2. X1 (keep C-main as config v1). 3. Hand-labelled set replaces MarioEval.jar (G-P5b). 4. Showcase 12 → 6 levels. **Never cut:** level-grouped split, playability checker and its validation, one-shot test, the MVP game, Streamlit Cloud deployment (fallback: cached-only mode, CL6), report, slides, evidence, zip. The Day 18 upload date cannot move.

## Team mapping (2–3 members)
Ownership by component is in `36_TEAM_OWNERSHIP_AND_VIVA.md` §1: **M-model** (Days 3–11, 14–15), **G-app** (assets, game, checker, normaliser, app, cloud), **E-data/eval** (Days 1–2, report, evidence, slides). With two members, E is merged into the other two as listed there. Ownership is by component, not by machine. Every member must still be able to explain and modify every component (doc 36 §3).
