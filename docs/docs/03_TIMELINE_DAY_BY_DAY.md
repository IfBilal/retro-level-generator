# 03 — Day-by-Day Timeline (Day 0 → Day 20)

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
| 5b | Playability checker (built in parallel on the G track D5–D6) | D5–D6, D11 | **G-P5b** |
| 7a/7b | Streamlit app + browser game ("Type it, play it" MVP) | D3, D5, D9 (G track) + D12–D13 | **G-P7** |
| 6 | Evaluation, CV, one-shot test | D14–D16 | **G-P6** |
| — | Report, hardening, buffer, viva | D17–D20 | Release checklist |

Phase 7 runs **before** Phase 6 on purpose. The app has to exist before the final numbers are frozen, so the demo and the reported numbers come from the **same frozen checkpoint**.

---

## Day 0 — Environment, repo scaffold, autonomous asset fetch  (B · DT + LT · ~6 h)
**Docs:** 06, 05, 07, 26, 34
- [ ] On **both** machines: install Python 3.12, create the venv, install torch per ledger V2 (check `nvidia-smi` driver first), install `requirements.txt`.
- [ ] Create repo `text2mario` with the exact tree in doc 05; copy this pack into `docs/`, `CLAUDE.md` to root, `reference/ref_unet.py`.
- [ ] Write `scripts/verify_env.py`: prints torch/CUDA/driver/GPU name/VRAM, runs a 5-second bf16 matmul + SDPA test, applies the 4.5 GiB cap (H2–H3) and checks it with a deliberate 5 GiB allocation that **must fail** with OOM.
- [ ] Write `configs/machine/desktop_3060ti.yaml` and `configs/machine/laptop_4050.yaml`.
- [ ] `scripts/fetch_assets.py` → Phaser 3.90.0 + Kenney Pixel Platformer (CC0) into `app/static/`, with sha256 in `assets/ASSET_MANIFEST.json`; fallback `make_placeholder_sprites.py` (doc 26).
- [ ] Check Java: `java -version`. Record in `env/capabilities.json` (decides doc 21 tier 2).
- [ ] Initialise `PROGRESS.md`, `HANDOFF.md`, `runs/` (git-ignored), the sync folder (doc 07).
- **Artifacts:** `env/lock_desktop.txt`, `env/lock_laptop.txt` (pip freeze), `env/capabilities.json`, the asset manifest.
- **Exit:** `verify_env.py` green on **both** machines with **the same torch minor version**.
- **If blocked:** driver too old for CUDA 13 and can't be updated → pin both to torch 2.14 + cu126 (V2). Kenney download fails → placeholder sprites (A2) and retry on Day 1.

## Day 1 — Phase 1a: acquisition + audit + EDA  (M · ANY · ~6 h)
**Docs:** 08, 09
- [ ] `scripts/fetch_data.py`: clone MarioDiffusion at a pinned commit into `data/raw/MarioDiffusion/` (isolated; never import its code, read files only); copy the 1 regular JSON + `smb.json`; write `data/MANIFEST.json` (commit SHA, sha256).
- [ ] `scripts/audit_dataset.py`: assert **all F1–F17 facts** of the ledger; reconstruct chains → `data/processed/index.parquet` with columns `idx, level_id, game, col_offset, caption, n_phrases, scene_hash, is_dup`.
- [ ] `src/text2mario/data/tiles.py`: the single tile table (F6) + char/ID/colour maps; test against `smb.json`.
- [ ] EDA report `reports/eda/EDA.md` + figures: tile frequencies (global, per game, per row), caption-length histogram, phrase frequencies, concept co-occurrence matrix, per-level concept profile heatmap (42 × 16 concepts), duplicate list.
- [ ] Port the caption detector: re-implement `assign_caption` logic in `src/text2mario/captions/detector.py` (read their code for the rules, rewrite cleanly, keep MIT attribution), and check that it **reproduces the stored caption for 100 % of the 7,687 scenes** (modulo phrase order).
- [ ] (G track, LT, 1 h) Tile-mapping contact sheet for the Kenney atlas → `assets/tile_map.json` (doc 26).
- **Exit:** audit script passes; detector reproduces 100 % of captions (set-equality of phrases).
- **If blocked:** detector < 100 % → list mismatches, fix, and do not proceed to evaluation work until it is ≥ 99.5 % with documented exceptions.

## Day 2 — Phase 1b: splits, leakage tests, loader, renderer  (M · ANY · ~7 h)
**Docs:** 10, 11, 12 (prompt sets only)
- [ ] `scripts/make_splits.py`: level-grouped selection (doc 10 algorithm, seed 20261117) → `data/splits/split_v1.json` (level_ids for train/dev/test + the 5 CV folds over the 36 non-test levels).
- [ ] Leakage tests (`tests/test_splits.py`): no level in two splits; every concept present in every split (≥ 5 windows each, except documented rare ones); NN-Hamming between test and train windows reported; reused-segment exceptions documented.
- [ ] Write `data/splits/LOCK` = sha256(split_v1.json). CI test fails if the hash changes.
- [ ] Write the **P-casual** set: 60 handwritten casual prompts with expected attribute dicts → split 30/30 dev/test; split the **paraphrase lexicon** into train/test variants (doc 12).
- [ ] `src/text2mario/data/dataset.py`: GPU-resident uint8 tensors, on-the-fly one-hot → {−1, +1}, seeded batch sampler with a resumable state, caption augmentation pipeline (D8).
- [ ] `src/text2mario/render/renderer.py`: `tensor_to_grid`, `grid_to_image(tile_px, skin)`, `grid_to_vglc_text`, `grid_to_json`.
- [ ] **Gate G-P1:** 10 random DataLoader items → render → saved to `reports/gates/P1/`. Check orientation (floor at the bottom, pipes upright, `<` left of `>`), and that the round trip `grid → onehot → argmax` is the identity.
- **Exit:** G-P1 passes (all checks scripted in `scripts/gate_p1.py`).

## Day 3 — Phase 2: scheduler  (M · ANY · ~5 h) ‖ G track: game skeleton (LT · ~3 h)
**Docs:** 13, 24 · (G) 27
- [ ] `src/text2mario/diffusion/scheduler.py`: linear + cosine; buffers for β, α, ᾱ, √ᾱ, √(1−ᾱ), posterior variance (both β̃ and β choices), posterior mean coefficients, DDIM timesteps; all in fp64, cast at use.
- [ ] `q_sample`, `predict_x0_from_eps`, `p_mean_variance`, `ddpm_step`, `ddim_step(eta)`, `renoise(x_prev, t_prev → t)`.
- [ ] Property tests (hypothesis): shapes and dtypes; monotonic ᾱ; oracle-ε DDIM exact reconstruction (fp64 MSE < 1e-6, argmax 100 %); empirical mean/std of x_t vs theory for t ∈ {0, 250, 500, 750, 999}.
- [ ] **Gate G-P2:** render corruption strips at t = 0, 250, 500, 750, 999 for both schedules → `reports/gates/P2/`; numeric checks per ledger §9.
- [ ] (G) Phaser skeleton: loads a **real training level** (export of level 0 via renderer → JSON schema doc 29), arrow keys + jump, tile collisions, camera follow. `app/static/game/physics.json` = the shared physics constants (also read by the playability checker).
- **Exit:** G-P2 passes; the game skeleton is playable for one real level.

## Day 4 — Phase 3a: U-Net + trainer core + cross-machine resume drill  (M · LT first, then DT · ~7 h)
**Docs:** 14, 16, 17, 07
- [ ] `src/text2mario/models/unet.py` exactly per doc 14 (use `reference/ref_unet.py` as the oracle; parameter count must equal **18,312,013** for the default config with the cross-attention ON, or the documented unconditional variant count).
- [ ] Unit tests: output shape for W ∈ {16, 24, 112}; gradient reaches every parameter; time-embedding sensitivity; mask handling.
- [ ] `src/text2mario/train/train.py`: bf16 autocast, AdamW param groups (no decay on norm/bias/emb), warm-up + cosine LR, grad-clip, EMA, deterministic per-t-bucket validation loss, atomic checkpoints, resume (full state incl. RNG + sampler), JSONL/CSV/TensorBoard logging, NVML temperature watchdog (H9), VRAM cap (H3).
- [ ] **Overfit-one-batch test:** 8 fixed scenes, fixed t, 1,500 steps → loss < 0.01. (Proves the optimiser path works.)
- [ ] **Min-VRAM measurement on the LAPTOP:** batch 64, 200 steps → `hw_report.json` (peak allocated/reserved, it/s). Must be ≤ 4.5 GiB.
- [ ] **Resume drill:** train 1,000 steps on DT → handoff → resume on LT for 500 steps → loss curve continuous (no spike > 2× the running mean), step counter, LR and EMA continuous.
- **Exit:** all tests green, drill passes, hw_report recorded for both machines.

## Day 5 — Phase 3b: unconditional training  (M · DT + LT in parallel 🌙 · ~4 h hands-on) ‖ G: playability checker v0 (~4 h)
**Docs:** 18, 19, 21
- [ ] Run **U-lin** (linear) on DT and **U-cos** (cosine) on LT, 15k steps each (configs `uncond_linear.yaml`, `uncond_cosine.yaml`).
- [ ] `src/text2mario/eval/structure.py`: floor validity, broken pipes/cannons, tile-distribution JS divergence, NN-Hamming novelty + copy rate (vectorised on GPU against the train set).
- [ ] Eval at 5k / 10k / 15k with 256 EMA samples (DDIM 50).
- [ ] (G) `src/text2mario/playability/checker.py` v0: tile-graph A* with jump arcs computed from `physics.json` (doc 21). Run it on **all 42 real levels** (full width) and on 16-wide windows → baseline solvable rate (expect high-80s to low-90s % for windows, cf. F18).
- **Exit:** **Gate G-P3** on the better schedule. Record the schedule choice in the ledger change log (M2).
- **If blocked:** follow doc 25 rows R-3.x (sky dominance, noise, broken pipes). Day 6 is the rollback buffer.

## Day 6 — Phase 3 buffer / Playability v1  (B · ANY · ~6 h)
- [ ] If G-P3 failed → execute the rollback (doc 25), retrain the winning schedule only, re-gate.
- [ ] If G-P3 passed → playability checker v1: enemies as hazards, cannons as solid, gap death, MarioEval.jar bridge (`grid_to_vglc_text` → `java -jar MarioEval.jar astar ...`) if Java is available (doc 21 tier 2); agreement study scaffold.
- [ ] Produce the "unconditional sample sheet" figure (64 samples) for the report.
- **Exit:** G-P3 passed (hard requirement before Day 7).

## Day 7 — Phase 4a: text conditioning wiring  (M · LT · ~6 h)
**Docs:** 15, 12
- [ ] `src/text2mario/text/encoder.py`: frozen MiniLM (bf16, eval, `requires_grad_(False)`), tokeniser max_len 48, LRU cache keyed by caption string, null embedding of "", adapter MLP + LayerNorm.
- [ ] Caption augmentation in the dataset (shuffle, phrase dropout, paraphrase substitution from the train lexicon, CFG null dropout).
- [ ] **Test:** `assert adapter params ⊂ optimizer.param_groups` and MiniLM params ∉ the optimiser (PDF rollback #3 turned into a test).
- [ ] **Test:** null-dropout empirical rate 0.10 ± 0.01 over 10k samples.
- [ ] Smoke conditional run, 3k steps: loss decreases; cross-attention maps aren't uniform (entropy below the uniform maximum).
- **Exit:** tests green; smoke run healthy.

## Day 8 — Phase 4b: main conditional run  (M · DT 🌙 · ~5 h hands-on) ‖ G: prompt normalizer v1 (LT)
**Docs:** 16, 18, 22, 12
- [ ] Launch **C-main** (`cond_main.yaml`), up to 60k steps, dev evaluation every 2k steps (C-score on P-dev at w = 5 with 3 seeds, broken rate, fast-checker playability, novelty), early stopping on composite score E4 (patience 5).
- [ ] `src/text2mario/eval/adherence.py`: C-score (paper formula) + per-concept exact match, using the Day 1 detector.
- [ ] (G) `src/text2mario/text/normalizer.py` v1: synonym → canonical, digits/number words → buckets, multi-scene splitter; tested on P-casual-dev (doc 12 targets).
- **Exit:** run in flight and healthy at the evening check (loss per t-bucket falling, dev composite improving).

## Day 9 — Phase 4c: CFG sweep + gate  (M · ANY · ~5 h) ‖ G: game mechanics (LT · ~4 h)
- [ ] Select the best checkpoint by dev composite; sweep w ∈ {1, 2, 3, 5, 7.5, 10} on P-dev (3 seeds) → curves: C-score, broken rate, playability, novelty vs w.
- [ ] **PDF contrast test:** "full floor with three pipes and no enemies" vs "staircase with gap and several enemies" (normalised to canonical) × 8 seeds → figure.
- [ ] **Gate G-P4.** Record default w in the ledger.
- [ ] (G) Game: enemies (walker AI, stomp/kill), question-block bump → coin, breakable bricks (bump), coins, pipes, cannons (static solid in the MVP), pits = death, respawn, flag = win screen, HUD (time, coins).
- **If blocked:** doc 25 rows R-4.x (ignores prompt, counts wrong, rare concepts).

## Day 10 — Phase 4d: ablation batch 1  (M · DT + LT in parallel 🌙 · ~4 h)
**Docs:** 23
- [ ] Run X1 (aux CE λ = 0.001), X2 (EMA 0.995), X3 (pooled-text-to-time-embedding), X4 (concept-balanced sampling, only if rare-concept exact match < 50 %). Same seeds; 40k-step budget each; dev eval only.
- [ ] Choose **config v1** = the best dev composite; tie-break by playability, then simplicity. Record in the ledger.
- **Exit:** config v1 chosen and its checkpoint retrained or continued to completion.

## Day 11 — Phase 5: full-level generation + playability gate  (M · ANY · ~7 h)
**Docs:** 20, 21
- [ ] `src/text2mario/generate/full_level.py`: scene-prompt splitter → sliding window (stride 8) with RePaint known-region replacement + r = 2 resampling (first 60 % of steps) → framing (G3) → sanitizer detect (+ optional repair) → checker → window resampling loop (G5).
- [ ] Seam metric (doc 20 §seam): column-transition NLL at seams vs interiors, using a column bigram model fitted on train levels.
- [ ] **Gate G-P5** (seams, sanitizer) and **Gate G-P5b** (checker vs MarioEval.jar agreement on 300 levels, or vs 50 hand-labelled levels).
- **If blocked:** escalation ladder in doc 25 R-5.x (stride 4, r = 5, direct wide sampling).

## Day 12 — Phase 7a: Streamlit studio skeleton  (G · LT · ~7 h)
**Docs:** 28, 29
- [ ] `app/streamlit_app.py`: prompt box, w slider, width select (16/48/80/112), seed, "fast mode" toggle (auto-on without CUDA), Generate → denoising animation (frames every 5 DDIM steps), final PNG, normalised-prompt display.
- [ ] `st.cache_resource` for model + encoder; warm-up generation at start-up.
- [ ] Exports: PNG, JSON (schema v1), CSV, VGLC .txt.
- **Exit:** cold start < 20 s on the laptop; generation works on GPU and CPU.

## Day 13 — Phase 7b: "Type it, play it" MVP  (G · LT · ~7 h)
**Docs:** 27, 28, 29
- [ ] Game component: level JSON + atlas embedded into the Phaser HTML (inline/base64; doc 27 §embedding), rendered via `st.components.v1.html`, with a "click to focus" overlay.
- [ ] "Verified beatable ✓ / not verified" badge from G5; attempt counter.
- [ ] "Download playable HTML" (single self-contained file).
- [ ] End-to-end test script: 20 prompts → generate → checker → the HTML loads in headless Chromium (Playwright smoke test: the canvas exists and the player sprite moves on simulated key presses).
- **Exit:** **MVP done.** Gate G-P7 timing checks pass on the laptop.

## Day 14 — Phase 6a: full dev evaluation + freeze  (M · ANY · ~6 h)
**Docs:** 22, 19
- [ ] Full metric suite on P-dev, P-rand, P-casual-dev, 3 seeds, bootstrap CIs; per-game breakdown (SMB1 vs SMB2); per-concept table; error analysis (20 worst prompts, with renders).
- [ ] Overfit/underfit dashboard (doc 19): train vs dev loss per t-bucket, copy rate vs steps, novelty vs the real-dev reference.
- [ ] **Freeze:** `configs/final.yaml` + checkpoint sha256 → `FROZEN.json`. No further tuning after this point.
- **Exit:** freeze committed and tagged `v0.9-frozen`.

## Day 15 — Phase 6b: 5-fold level-grouped CV + ablation batch 2  (M · DT + LT 🌙 · ~3 h hands-on)
**Docs:** 10 §CV, 23
- [ ] 5 folds × final config (the fold-k model trains on 4 folds, evaluates on fold k's levels): distribute as DT folds 1–3, LT folds 4–5 (each fold is an independent run directory; doc 07 avoids collisions).
- [ ] Ablation batch 2 (if time): X5 flip-augmentation, X6 linear-vs-cosine confirmation at the final config, X7 w per-concept.
- **Exit:** CV table mean ± std for every metric.

## Day 16 — Phase 6c: ONE-SHOT test evaluation  (M · ANY · ~4 h)
- [ ] Verify `LOCK` and `FROZEN.json` hashes → run `scripts/eval_test.py` **once** on P-test, P-rand (fresh seed), P-casual-test; MarioEval.jar on a 300-level subset.
- [ ] Write `reports/final/RESULTS.md` (numbers + CIs). The test run's log is committed as-is, with no re-runs. (If a bug is found in the *metric code*, fix it, re-run, and document both runs.)
- **Exit:** **Gate G-P6.**

## Day 17 — Report figures, model card, data card  (B · ANY · ~7 h)
**Docs:** 32
- [ ] All figures/tables in doc 32's list, generated by `scripts/make_figures.py` (no manual plots).
- [ ] `MODEL_CARD.md`, `DATA_CARD.md` (licences, split, known biases: SMB2 harder, rare concepts).

## Day 18 — Demo hardening  (G · LT · ~6 h)
- [ ] Offline run (Wi-Fi off) from a cold boot: app starts, assets load, model loads.
- [ ] CPU-only rehearsal (`CUDA_VISIBLE_DEVICES=""`).
- [ ] Pre-generate 12 showcase levels (cached JSON + PNG) as the instant fallback.
- [ ] Browser matrix: Chrome + Edge (+ Firefox best-effort); keyboard focus; 1080p and 768p layouts.

## Day 19 — Buffer  (B)
Absorbs slips. If unused: a "report card lite" panel (requested vs detected concepts table, beatable, NN-copy check), which is a direct seam toward future feature F6 (doc 31).

## Day 20 — Viva rehearsal + release  (B)
- [ ] Two full mock vivas with doc 32's Q&A list; time the live demo (< 3 min).
- [ ] Tag `v1.0`; archive `runs/final/`, `FROZEN.json`, reports; zip release.

---

## Compressed 16-day variant (only if the deadline forces it)
Day 0+1 merged (setup + acquisition), D5+D6 merged (single schedule: cosine, rollback buffer cut), D10 ablations cut to X1 only, D15 CV cut (report seed variance only, and say so in the report), D17+D18 merged, D19 buffer cut. **Never cut:** level-grouped split, playability checker, one-shot test discipline, the MVP game.

## Two-person mapping (if two people work in parallel)
Person A owns the **M track** (Days 1–2, 3-sched, 4–5, 7–11, 14–16); Person B owns the **G track** (assets, game, checker, normalizer, app: D0 assets, D3/D5/D6/D8/D9 game+checker+normalizer, D12–13, D18). Ownership is by track, not by machine; either track can use either GPU.
