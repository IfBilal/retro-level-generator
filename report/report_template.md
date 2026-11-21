# text2mario: type a sentence, play a verified level

*AI2002 Software Category project, Path A (software application with a trained model, deployed with Streamlit).*
*Every number in this report is filled in by `scripts/build_report.py` from the saved result files; figures come from `scripts/make_figures.py`.*

## 1. Problem, justification, path and objectives

**Problem.** A 2D platformer level is a grid of tiles, and a designer's first draft is slow to build by hand. Procedural generators are
either random (hard to steer) or learned without any control from words, and almost none check that the result can be finished.
We build a system that turns **one typed sentence** (for example *"flat ground, one pipe, then several coins and a staircase"*) into a
full Super-Mario-style level that has been **checked for beatability** and can be **played immediately in the browser**.

**Who needs it.** (U1) an indie/hobbyist designer who wants several steerable first drafts and an export to an editable tile format;
(U2) a student or teacher of procedural content generation who wants an inspectable, honest end-to-end example; (U3) anyone given the app
link. *No interviews or surveys were done; these are stated needs, not measured demand. The only claims made below are measured ones.*

**Selected path.** Path A: a trained machine-learning model is the core of a Streamlit application (deployed on Streamlit Community Cloud).

**Input and output.** Input: one sentence (1-400 characters), a width in {16, 48, 80, 112} generated columns, a guidance strength *w*, a seed.
Output: a framed level of 16 rows of 13 tile types, the canonical caption(s) the model actually received, a verification badge, a PNG, a playable game,
and exports (level JSON, CSV, VGLC text, standalone HTML).

**Objectives.** O-1 content follows the prompt (caption adherence); O-2 structures stay valid (no broken pipes or cannons); O-3 verify beatability and
*measure how far that verification can be trusted*; O-4 do not copy training windows; O-5 evaluate honestly on **unseen levels**
(level-grouped split, one-shot test); O-6 a working, deployed Streamlit app.

**Why a neural network.** Rules and templates can place pipes and gaps but cannot learn the joint structure of real level segments (which floors, platforms,
enemies and pipes occur together) from 42 levels. A tile **diffusion** model generates a whole 16x16 window by iterative refinement, so a pipe's cap and body
appear together; it supports classifier-free guidance (a controllable strength of prompt adherence) and overlap-consistent outpainting (RePaint) for long levels.
The main alternative, an autoregressive tile model, generates left to right and cannot revise a column once written.

> **Originality and attribution.** The dataset, the caption grammar and the idea of text-conditioned diffusion for Mario levels come from MarioDiffusion
> (Schrum et al., arXiv 2507.00184, MIT licence). Our own work: the level-grouped locked split and leakage tests, the caption detector (written from the
> grammar description and *verified* against all 7,687 stored captions), the diffusion maths and U-Net implementation, the trainer, the prompt normalizer,
> the C-score re-implementation (checked on 12 golden vectors), the playability checker, the Phaser game, the generation pipeline, the app and the evaluation.
> No third-party code is imported; the reference solver `MarioEval.jar` is only *run* as a separate process.

## 2. Dataset and preprocessing

**Source and relevance.** The MarioDiffusion corpus: 7,687 windows of 16x16 tiles from 42 levels (20 of *Super Mario Bros*, 22 of *The Lost Levels*), each with
a machine-written caption (e.g. `full floor. a few enemies. two pipes.`). Levels are reconstructed from overlapping windows (window *i+1* is window *i* shifted one column),
which recovers all 42 level chains byte-for-byte. Relevance: it is a public, level-structured corpus with captions in a fixed grammar of 16 concepts and 5 quantity buckets.

**Features and target.** A window is a 16x16 grid over 13 tile classes (sky, four pipe parts, question block (full/empty), breakable block, ground, cannon top and
support, enemy, coin). It is represented as a {-1, +1} one-hot tensor of shape 13x16x16, which is the diffusion variable *x0*. The conditioning input is the caption
text; the target of the denoiser is the added noise. Preprocessing is done once and identically for training and inference: the same one-hot encoder, the same
MiniLM encoder and adapter, and the same normalizer (the app calls the same code as training).

**Splitting and leakage prevention.** The 42 levels are split **by level** (31 train, 5 dev, 6 test; 5,709 / 905 / 1,073 windows), never by window, because windows
overlap by 15 of 16 columns. Near-duplicate level families (the two games share many levels) are kept in one split. The split is chosen by an exhaustive
deterministic search (seed 20261117), hashed into `data/splits/LOCK`, and any edit raises an error. Tests assert that no level is in two splits, that the test windows can
not be loaded before `FROZEN.json` exists, and that dev tooling refuses the test prompts. Hyper-parameters, checkpoints, the CFG scale and every threshold were
chosen on **dev only**. The provided train/validate/test files of the original repository are never used.

**Caption detector.** The detector that reads a caption off any grid (used for metrics and for the normalizer's inverse) was written from the grammar description
and reproduces **7,687 / 7,687** stored captions exactly (a strong check that our understanding of the data is right).

## 3. Algorithm

**Diffusion in one paragraph.** Forward process: *x_t = sqrt(abar_t) x_0 + sqrt(1 - abar_t) eps* with *eps ~ N(0, I)* over *T = 1000* steps; a network
*eps_theta(x_t, t, c)* is trained to predict *eps* (mean-squared error), so *x_0* can be recovered as *(x_t - sqrt(1-abar_t) eps_hat) / sqrt(abar_t)*.
Two noise schedules were compared: linear and cosine (both computed in float64; the oracle test "perfect eps gives exact reconstruction" passes for both).
Sampling uses **DDIM** with 50 steps (eta = 0, deterministic), the predicted *x_0* is **clipped to [-1, 1]** at every step, and the final tile is the arg-max over the 13
channels. The error in eps-space equals SNR-weighted error in x0-space, so the loss is tiny where the picture is easy; we therefore monitor the loss per noise bucket
against a computed "context-free floor" (the best a denoiser could do looking at each pixel alone).

**Architecture (18,312,013 parameters).** A U-Net over 16x16 inputs with channel multipliers (1, 2, 4) from 64 base channels: residual blocks with GroupNorm and a
sinusoidal time embedding, self-attention at the 8x8 and 4x4 levels, and **cross-attention to the text tokens** at the same levels. Input: 13x16x16 noisy tiles,
timestep, text context; output: 13x16x16 predicted noise. The last layers are zero-initialised, which gives an initial loss of about 1.0 (we verify it).

**Text conditioning (adds 66,048 trainable parameters).** The caption is encoded by a **frozen** MiniLM (all-MiniLM-L6-v2) into token states (no pooling), mapped by a small
trainable adapter (MLP + LayerNorm) to the U-Net's 128-d context. A test asserts that the optimiser holds exactly the U-Net and adapter parameters (18,378,061) and no MiniLM
tensor. During training 10 % of captions are replaced by the empty caption, which is what makes **classifier-free guidance** possible:
*eps = eps_u + w (eps_c - eps_u)*. Captions are augmented (phrase dropout, shuffling, synonym substitution using only the *train half* of a lexicon).

**Long levels by outpainting.** A level is built from 16x16 windows with a stride of 8: window *k+1* keeps the right 8 columns of window *k* as a known region
(RePaint: at every step the known region of *x_t* is replaced by its forward-noised known tiles, with 2 resampling jumps in the first 60 % of steps). The result is framed
by 3 floor columns in front and a 6-column end template (stairs and flag).

**Why this is appropriate.** The model must generate structured grids that are globally coherent (a pipe is a 2-wide object with a cap), controllable by text, and *diverse*;
diffusion with cross-attention and guidance gives all three, trains stably on 5.7k windows with a small network (peak 0.77 GiB of GPU memory), and its sampling interface
(known-region replacement) gives long levels for free.

**Key settings** (all in `configs/*.yaml`): AdamW, lr 2e-4 with 1,000 warm-up steps and cosine decay to 1e-5, weight decay 0.01 (not on norms/biases), gradient clip 1.0,
batch 64, bf16 autocast, EMA 0.999 (evaluation always uses the EMA weights), 15,000 steps unconditional / 60,000 conditional, VRAM cap 4.5 GiB.

## 4. Implementation

![Pipeline](figures/fig01_pipeline.png)
*Figure 1. From a typed sentence to a playable, checked level.*

**Pipeline.** `text/normalizer.py` (rules + lexicon: synonyms, number words, vague quantities, negation, multi-scene splitting, clamping to the buckets seen in training)
→ `text/conditioner.py` → `models/unet.py` + `diffusion/` → `generate/full_level.py` (windows, outpainting, framing) → `generate/sanitizer.py` (detect broken pipes/cannons on the
*raw* grid; optional display repair) → `playability/checker.py` → resample the window where the search stalled (at most 5 attempts per window, 20 per level) →
`render/` (PNG, JSON, CSV, VGLC, one-file offline HTML) → `app/streamlit_app.py`.

**Prompt normalizer.** 30/30 exact matches on the 30 development prompts (topic F1 1.00) and 100 % passthrough of the 2,668 distinct stored captions.

**Playability checker.** A* over standing cells with jump arcs simulated with the **same integrator as the game** (gravity, wall blocking, head bumps, landing), then limited by
conservative human margins. The physics numbers live in one file (`app/static/game/physics.json`, shared by checker and game); `scripts/derive_physics.py` derives the jump
envelopes from it. Enemies are treated as killable (stomp). "Verified" means *the checker found a route under the game's physics*, not a proof for every player.

**Game.** Phaser 3.90.0 (MIT) with Kenney Pixel Platformer sprites (CC0): tile collisions, run/jump with the shared physics, walking enemies with stomp, question blocks, coins, pits and
respawn, a flag and a HUD. A level becomes **one self-contained HTML file** (the engine, level and sprites are inlined), so it works offline and as a download.

**Application.** Streamlit: prompt box with examples, a preview of what the model will understand (canonical scenes, clamps, ignored words), guidance and width controls, seed,
fast mode (25 steps; automatic without a GPU), generation with a verification badge, the embedded game, and downloads. Input validation rejects empty, over-long or
control-character prompts and out-of-range settings with a readable message instead of a stack trace.

**Measured latency** (this machine; `reports/app/latency.json`; thresholds were pinned *after* measuring, not before). Fast mode (25 steps, no resampling, checker budget 1,000 nodes) is the app default:
48 columns {{lat_cuda_48_fast_p50}} s median (p95 {{lat_cuda_48_fast_p95}} s), 112 columns {{lat_cuda_112_fast_p50}} s (p95 {{lat_cuda_112_fast_p95}} s) on the RTX 3060 Ti; CPU fast mode {{lat_cpu_48_p50}} s and {{lat_cpu_112_p50}} s; cold start {{lat_cold_start_s}} s.
Full quality (50 steps, resampling) takes {{lat_cuda_48_p50}} s and {{lat_cuda_112_p50}} s median, dominated by the diffusion passes and by the checker exhausting unsolvable levels, so it is an opt-in toggle. The provisional target of 10 s for 112 columns is **not met**;
the long p95 comes from levels the checker cannot verify, which use the whole retry budget.

**Engineering.** {{tests_pass}} automated tests (fast suite) cover data, split leakage, detector, schedules, U-Net, sampler, trainer, text path, normalizer, C-score golden vectors,
checker micro-levels, level schema/export, full-level generation with a stub sampler, and input validation; end-to-end tests drive the real game in headless Chromium.
Bugs found by this process and fixed (each with a regression test): an LRU-cache eviction that crashed long training runs, a CPU/CUDA generator mismatch that crashed resampling,
a pluralisation bug that corrupted caption augmentation ("bad guies"), and a detector crash on generated rows without floor.

## 5. Experiments and results on development data

### 5.1 Phase 3: unconditional baseline and the schedule comparison

Training the unconditional U-Net for 15,000 steps takes about 28 minutes on the RTX 3060 Ti (0.111 s/step, 0.77 GiB peak). The per-noise-bucket dev loss ends below the
context-free floor for both schedules (mean 0.0062 against 0.0076 linear; 0.0101 against 0.0116 cosine), i.e. the network uses spatial context.
Samples look like levels (Fig. 7). **The gate G-P3 FAILED**: no checkpoint of any variant has both a broken-structure rate <= 0.05 *and* a copy rate <= 0.154.

{{p3_table}}

![Trade-off](figures/fig03_p3_tradeoff.png)
*Figure 3. Broken structures fall with training while copying rises; auxiliary cross-entropy (lambda 0.001) removes broken structures faster but memorises much more.*

The rollback matrix (R-3.4, R-3.6) prescribes auxiliary cross-entropy for broken structures and an earlier stop for copying. We tried both with dense checkpoints: the closest
points are linear 14k (broken 0.066 / copy 0.148) and cosine 14k (0.070 / 0.137); at equal steps cosine is better on both axes, so **cosine was chosen** (ties go to cosine in the
design). The gate is therefore recorded as **failed, with a documented waiver** (`reports/gates/P3/WAIVER.md`): the conditional model, which is the product, is judged by
its own, stricter-in-practice metrics below. We did not train longer to hide the problem.

### 5.2 Phase 4: text-conditioned model (C-main) and guidance

C-main (cosine, 60,000 steps) is evaluated on the 115 dev prompts x 3 seeds. Selection among milestones uses
*S_noplay = 0.5 C + 0.25 (1 - broken) + 0.25 min(1, novelty/reference)* because the playability term may only be used after the checker is validated (it was not, section 5.4).
The frozen model is step **{{frozen_step}}** (cosine) with guidance **w = {{frozen_w}}**.

| Gate G-P4 check | Result |
|---|---|
| P4.01 dev C-score at w* (target >= 0.75) | {{p4_01}} |
| P4.02 guidance helps (paired, level-clustered CI above 0) | {{p4_02}} |
| P4.03 optimiser holds exactly U-Net + adapter, no MiniLM | {{p4_03}} |
| P4.04 PDF contrast test (pipes vs enemies/gaps/stairs, 8 seeds) | {{p4_04}} |

C-score null (a realistic level that ignores the prompt): {{p4_null}}; improvement over null: {{p4_delta}}. Gate G-P4: **{{p4_pass}}**. The failing check is P4.04: the omitted-enemies prompt still contains enemies (about 3.9 enemy tiles against 5.5 for the enemy prompt; the pipes and the gap/stairs contrasts pass 8/8). The training captions never express *absence*, so leaving a topic out is only a weak signal; the generator therefore keeps a soft *forbid* list from the normalizer ("no enemies") and re-samples windows that contain a forbidden concept within the same attempt budget. We report this honestly as a limitation of caption-conditioned generation.
Dev metrics at w*: C-score {{dev_c_score}}, exact match {{dev_exact_match}}, broken rate {{dev_broken_rate}}, floor validity {{dev_floor_valid}},
median nearest-train Hamming {{dev_novelty_med}} (real dev windows: 13), copy rate {{dev_copy_rate}}.

{{cfg_table}}

![CFG sweep](figures/fig06_cfg.png)
*Figure 6. Guidance strength: adherence rises with w, structure quality and copying move the other way.*

**Ablation X1** (auxiliary cross-entropy, lambda 0.001, on the conditional model; planned for 40k steps, **stopped at 17,640** to relieve memory pressure; dev set, same seeds): {{x1_table}}

*Process note, stated plainly:* the X1 run was evaluated **after** C-main had been frozen and tested. At the matched step (17,640 against 18,000) X1 has a higher dev C-score and a lower broken rate with similar novelty and copy rate; under the plan's rule
"config v1 = the best dev composite" it would probably have been chosen. Because the one-shot test had already been run on C-main, the test cannot be repeated without breaking its protocol, so **C-main remains the reported model** and X1 is reported as a dev-only ablation
(single checkpoint, 3 seeds, 5 dev levels: the difference is suggestive, not established). Selecting X1 would be the first follow-up.

### 5.3 Phase 5: full levels

100 dev levels (50 prompts x 2 seeds, 112 columns, 50 steps, r = 2): seam ratio {{p5_01}} (limit 1.3; real dev levels with pseudo-seams: {{p5_real_dev_pseudo_seam_ratio_mean}}),
broken structures after the display repair {{p5_02}}, verify-loop limits {{p5_03}}. Pre-repair broken rate {{p5_pre_repair_broken_rate}}, verified rate {{p5_verified_rate}},
{{p5_seconds_per_level_mean}} s per level on average (p95 {{p5_seconds_per_level_p95}} s). Gate G-P5: **{{p5_pass}}**: the verify-loop limits hold and, after the display repair (which deletes every component the detector classifies as broken), no broken structure remains; the seam check fails. The first escalation step of the plan (more RePaint resampling jumps, r = 5) lowered the seam ratio only from 1.47 to 1.38 on 12 paired levels (still above 1.3, within noise, 1.8x the cost; `reports/gates/P5/seam_ladder_r.json`); stride 4 and direct wide sampling were not tried. The real dev levels with pseudo-seams scatter between 0.75 and 1.6 under the same estimator, so the metric itself is noisy with 31 training levels.

### 5.4 The playability checker: how far can "verified" be trusted?

On the **real** train+dev levels the checker solves {{cal_levels}} of the full-width levels and {{cal_windows}} of framed 16-wide windows (design target 0.90, **not reached**). Physics
was tuned only within the permitted range (jump 23.0, run 7.0) and then frozen. The remaining failures are *Lost Levels* sky-bridge sections that need a 4-tile rise over a
4-tile gap, beyond the conservative human margins; treating every enemy as an impassable hazard solved only 22 % (enemies can be stomped, so they are modelled as killable).

**Gate G-P5b (checker agreement with the reference solver MarioEval.jar) FAILED**: agreement {{p5b_agree}} (threshold 0.85), kappa {{p5b_kappa}}. The jar fails even a one-tile wall in synthetic
levels, so it is not a usable oracle in this setting. Following the documented fallback, **`S_noplay` is used for every selection** and "verified" is never used to rank models.
Independent evidence for the badge instead: (i) a *physics-contract* test in the real Phaser engine ({{game_cases}} micro-levels): gaps of 1-6 tiles and walls/steps of 1-4 tiles that
the checker accepts are cleared by the game, gap 7 and height 5 are not accepted ({{game_unsound}} unsound cases); (ii) replaying the checker's own path with a closed-loop bot in the
game on {{replay_n}} real windows reaches the flag in {{replay_rate}} of cases (a lower bound: the bot, not the level, fails the rest). (iii) The same replay on **generated** full levels (enemies removed, because the bot cannot stomp while the checker assumes stomps; the bot re-plans from where it stands
after a slip or a death): {{rl_showcase_rate}} of {{rl_showcase_n}} verified showcase levels and {{rl_p5_rate}} of {{rl_p5_n}} verified levels sampled from the G-P5 set are finished. Debugging this exposed two real limits: the bot first stalled at wall
feet and overshot narrow landings (it did not brake in the air the way the game allows), both fixed in the bot; what remains is a bot weaker than a human, plus the checker's assumption that a player can stop where they want in the air.
The app and the game therefore say "route found by the checker (not a guarantee)" instead of "verified beatable". **So "verified" is a statement about the checker under optimistic-but-documented assumptions, and its end-to-end reliability is the number above, not 100 %.**

![Checker](figures/fig05_checker.png)
*Figure 5. Left: share of real levels the checker solves. Right: the checker's accept/reject decisions against what the real engine can do.*

## 6. Final evaluation on unseen test levels (one shot)

The model, guidance, thresholds and evaluation plan were frozen before the test set was touched (`FROZEN.json`, checkpoint sha256 `{{frozen_sha}}...`); `scripts/eval_test.py` refuses to
run twice. Intervals resample the **{{t_n_levels}} held-out levels**; with so few levels they are wide, and per-level results are given as well.

| Metric | P-test ({{t_n_prompts}} prompts) | P-rand ({{r_n_prompts}} prompts) |
|---|---|---|
| C-score | {{t_c_score}} | {{r_c_score}} |
| exact match | {{t_exact_match}} | {{r_exact_match}} |
| broken-structure rate (raw samples) | {{t_broken_rate}} | {{r_broken_rate}} |
| floor validity | {{t_floor_valid}} | {{r_floor_valid}} |
| checker-verified rate (framed windows; diagnostic) | {{t_checker_verified_rate}} | {{r_checker_verified_rate}} |
| median nearest-train Hamming / copy rate | {{t_novelty_med}} / {{t_copy_rate}} | {{r_novelty_med}} / {{r_copy_rate}} |
| rare-concept exact match | {{t_rare_exact_match}} | {{r_rare_exact_match}} |

Reference rows (the same prompts, a generator that ignores them): null {{t_ref_null_other_level}}, always-empty {{t_ref_always_empty}}, always "full floor" {{t_ref_always_full_floor}},
always the modal caption {{t_ref_always_modal}}. A C-score must always be read against these (an *empty* level already scores about 0.55 because absent topics earn credit).
P-casual-test (30 held-out casual prompts through the normalizer): scene-count accuracy {{casual_n1}}, C-score of generated scenes against the expected captions {{casual_c}}.

{{per_level_table}}

{{concept_table}}

![Failure gallery](../reports/final/failure_gallery.png)
*Figure 21. The 20 lowest-scoring test samples (prompt and C-score above each).*

**Error analysis.** Held-out levels score well on prompts built from real windows (C-score far above the prompt-ignoring null and above every constant generator) but a second set of *random seven-topic* prompts (P-rand) is a stress test the model fails: its C-score is below an always-empty generator, because unseen topic combinations produce broken structures (each costs 2/18 of the C-score) and over-generated phrases. P-rand has no source levels, so it has no level-clustered interval and no cross-level null. The per-concept table shows rare and countable concepts are weakest; the failure gallery shows the 20 worst samples; `reports/final/NOTES.md` records the one crashed-then-repeated run of the test script.

## 7. Testing and reliability

| Evidence | Where |
|---|---|
| {{tests_pass}} fast automated tests passing | `tests/`, `reports/testing/pytest_summary.json` |
| representative inputs: normalizer dev set, 12 C-score golden vectors, checker micro-levels, 4 example prompts end to end | `tests/`, `reports/gates/*` |
| invalid and boundary inputs: empty / whitespace / None / over-long / control-character prompts, out-of-range seed / w / width, malformed levels (9 schema violations) | `tests/test_level_schema.py` |
| consistency: same prompt + seed + settings gives the same level; checker and normalizer deterministic | `tests/test_full_level.py`, `tests/test_checker.py` |
| end-to-end: the real game in headless Chromium (canvas, key presses, ghost-bot wins a level) | `tests/e2e/` |
| leakage and one-shot guards | `tests/test_one_shot_guards.py`, `tests/test_splits.py` |
| gates | `reports/gates/*/RESULT.json` (pass/fail recorded honestly) |

## 8. Limitations (with numbers)

- **Small data:** 31 training levels. The unconditional baseline cannot satisfy both gate bounds (broken <= 0.05 and copy <= 0.154); the conditional model's copy rate and novelty are reported against the real reference (13).
- **Random combinations fail:** on P-rand (random seven-topic prompts) the C-score is below an always-empty generator; the model follows prompts that look like real level segments, not arbitrary specifications.
- **Absence cannot be expressed:** captions never say "no X"; the forbid list is a soft re-sampling filter (contrast test P4.04 failed).
- **Seams:** the seam ratio gate (1.3) was not met (G-P5.01); the first remedy (r = 5) was not enough.
- **Conditioning is coarse:** captions have no positions, so "a pipe near the end" cannot be expressed; quantities are 5 buckets; rare concepts (descending stairs, ceilings with gaps) have few training examples and are clamped.
- **Checker:** solves {{cal_levels}} of real full levels (target 0.90) and disagrees with the reference jar (agreement {{p5b_agree}}); "verified" is a conservative physics claim, with killable enemies assumed.
- **Statistics:** only 5 dev and 6 test levels, so confidence intervals are wide; no human study of playability or enjoyment was run.
- **Platform:** one game family (SMB1/SMB2 tiles), 16-row levels, no moving platforms, springs or bullets (cannons are static).
- **Deployment:** the Streamlit Community Cloud deployment needs the team's GitHub/Streamlit login and could not be performed by the automated build; the repository is prepared for it (`DEPLOY.md`).

## 9. Conclusion

We built and evaluated a complete text-to-level system: a diffusion model with text conditioning and classifier-free guidance, a normalizer that makes free text usable, outpainting for long levels,
a checker that backs a "verified beatable" badge, a game that runs the same physics, and a Streamlit app around all of it, with a level-grouped, one-shot evaluation. Where gates failed (G-P3, G-P5b, the
checker's real-level target) the report says so, shows the evidence, and applies the documented fallback instead of weakening the claim.

## Appendix A. Reproduction

```
uv venv --python 3.12 .venv && uv pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cu132 && uv pip install -r requirements-dev.txt
python scripts/fetch_data.py && python scripts/build_index.py && python scripts/fetch_models.py && python scripts/fetch_assets.py && python scripts/make_atlas.py
python -m text2mario.train.train --config configs/cond_main.yaml --run-dir runs/C-main      # PYTHONPATH=src
python scripts/gate_p4.py --ckpt runs/C-main/step_XXXXXXX.pt && python scripts/gate_p5.py --ckpt runs/C-main/step_XXXXXXX.pt
python scripts/freeze.py --ckpt ... --w ... && python scripts/eval_test.py
python scripts/export_cloud_ckpt.py runs/C-main/step_XXXXXXX.pt && streamlit run app/streamlit_app.py
```
