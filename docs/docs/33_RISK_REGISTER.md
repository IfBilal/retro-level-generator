# 33 — Risk Register

> **Owner:** B (both tracks). **Reviewed:** daily trigger scan, at every gate, and in full on Days 6, 13 and 20 (§6).
> **ID convention:** risks are **R1–R38** (no hyphen). Rows of the rollback matrix in doc 25 use the hyphenated form **R-<phase>.<n>** (e.g. `R-3.x` = Phase 3 rows). Don't confuse the two. A risk's "Contingency" often points to an R-x.y rollback row.
> **Ledger links:** IDs in brackets (F*, D*, M*, G*, E*, H*, V*, A*, G-P*) point to `00_DECISIONS_LEDGER.md`.

---

## 1. Scoring

| Score | Likelihood (L) — chance it happens during the project | Impact (I) — if it happens |
|---|---|---|
| 1 | < 5 % | Cosmetic; < 2 h lost |
| 2 | 5–20 % | ≤ ½ day lost, or a minor metric caveat |
| 3 | 20–50 % | 1–2 days lost, or a gate re-run, or a weakened report claim |
| 4 | 50–80 % | 3+ days lost, or a demo feature degraded, or a key claim must be withdrawn |
| 5 | > 80 % | Critical-path failure, invalid results, or a failed viva demo |

**Risk score = L × I** (1–25). Bands: **≥ 15 critical** (active mitigation now, daily watch) · **10–14 high** (mitigation scheduled, watch at its phase) · **6–9 medium** (mitigation built into the plan) · **≤ 5 low** (accept and log).

**Owner track:** M = Model/ML, G = Game/App, B = both (timeline doc 03).

---

## 2. Heat map (count of risks per cell, current scores)

| L \ I | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| **5** | | | | | |
| **4** | | 2 (R13, R16) | 2 (R4, R28) | | |
| **3** | | 4 (R18, R22, R25, R36) | 11 (R3, R8, R14, R17, R20, R23, R24, R32, R35, R37, R38) | 5 (R7, R10, R15, R26, R31) | 4 (R5, R9, R21, R30) |
| **2** | | | 5 (R11, R12, R19, R27, R29) | 2 (R1, R34) | 3 (R2, R6, R33) |
| **1** | | | | | |

38 risks in total. Each is counted once, at its register score (§3). Recompute this map at every full review (§6).

---

## 3. Summary register (sorted by ID)

| ID | Category | Risk (short) | L | I | Score | Owner | Linked docs / rollback rows |
|---|---|---|---|---|---|---|---|
| R1 | Data | Upstream data differs from ledger facts / commit not pinned | 2 | 4 | 8 | M | 08, 09; ledger §1 |
| R2 | Data | Level-chain reconstruction wrong → wrong level IDs → leaky split | 2 | 5 | 10 | M | 09, 10; F7–F11 |
| R3 | Data | Cross-level segment reuse leaks content across the split | 3 | 3 | 9 | M | 10; G-P1 |
| R4 | Data | Rare concepts too sparse in dev/test → unstable per-concept metrics | 4 | 3 | 12 | M | 10, 22; D6, D9 |
| R5 | Data/Eval | Caption-detector port doesn't reproduce stored captions → adherence metric invalid | 3 | 5 | **15** | M | 09, 22; F19 |
| R6 | Process | Test-set contamination (tuning on test levels/prompts) | 2 | 5 | 10 | B | 10, 22, 30; D4 |
| R7 | Model | Unconditional baseline fails G-P3 (sky-dominant / noisy samples) | 3 | 4 | 12 | M | 18, 19; doc 25 R-3.x |
| R8 | Model | Broken multi-tile structures persist (half pipes, orphan cannons) | 3 | 3 | 9 | M | 14, 20; doc 25 R-3.x; G4 |
| R9 | Model | Model ignores the text / dev C-score < 0.75 | 3 | 5 | **15** | M | 15, 18; doc 25 R-4.x; G-P4 |
| R10 | Model | Over-fitting / memorisation (copy rate > 10 %, novelty < reference) | 3 | 4 | 12 | M | 19, 22; E1, E4 |
| R11 | Model | Under-fitting: step budget too small | 2 | 3 | 6 | M | 17, 19; M12 |
| R12 | Model | Training instability (NaN/Inf in bf16, loss spikes) | 2 | 3 | 6 | M | 16; H5 |
| R13 | Model | High CFG w breaks structures / playability | 4 | 2 | 8 | M | 15, 20; M13 |
| R14 | Eval | Detector circularity (Goodhart): the model learns detector quirks | 3 | 3 | 9 | M | 22, 32 |
| R15 | Infra | GPU switching corrupts a run (two machines resume the same run, stale checkpoint) | 3 | 4 | 12 | B | 07, 30 |
| R16 | Infra | Laptop thermal throttling / shutdown mid-run | 4 | 2 | 8 | M | 07; H9 |
| R17 | Infra | Torch/CUDA/driver mismatch between machines | 3 | 3 | 9 | B | 06; V2 |
| R18 | Infra | Windows-specific breakage (multiprocessing, paths, symlinks, compile) | 3 | 2 | 6 | B | 06; H6, H7 |
| R19 | Infra | Disk exhaustion from checkpoints | 2 | 3 | 6 | B | 30 §12 |
| R20 | Infra | Sync-folder conflicts / partial uploads corrupt checkpoints | 3 | 3 | 9 | B | 07, 30 |
| R21 | Eval/App | Playability checker disagrees with game physics or MarioEval.jar → false "verified beatable" | 3 | 5 | **15** | G | 21, 27; G-P5b |
| R22 | Eval | Java / MarioEval.jar unavailable or format-incompatible | 3 | 2 | 6 | G | 21; V6 |
| R23 | Eval | Evaluation sampling dominates wall-clock | 3 | 3 | 9 | M | 16, 22 |
| R24 | Eval | Statistical noise picks the wrong ablation winner | 3 | 3 | 9 | M | 23, 22; E3 |
| R25 | App | Asset download fails / licence unclear | 3 | 2 | 6 | G | 26; A1–A4 |
| R26 | App | Phaser-in-Streamlit iframe: keyboard focus, sandbox, offline asset loading | 3 | 4 | 12 | G | 27, 28 |
| R27 | App | Generation latency > 10 s on the laptop GPU | 2 | 3 | 6 | G | 20, 28; G-P7, H10 |
| R28 | App | Prompt normalizer fails on the examiner's casual phrasing | 4 | 3 | 12 | G | 12, 28 |
| R29 | Legal | Nintendo IP / licensing objection (sprites or layouts) | 2 | 3 | 6 | B | 26, 32; A3 |
| R30 | Schedule | Critical-path slip (gates fail repeatedly) | 3 | 5 | **15** | B | 03, 04 |
| R31 | Process | Scope creep: future features started before MVP | 3 | 4 | 12 | B | 02, 31; S3 |
| R32 | Process | Claude Code drifts from the spec (magic numbers, silent assumptions) | 3 | 3 | 9 | B | 34, 30 |
| R33 | Demo | Demo-day failure (hardware, OS update, projector, no internet) | 2 | 5 | 10 | G | 28, 32 |
| R34 | Process | Single point of knowledge / illness | 2 | 4 | 8 | B | 30 §13 |
| R35 | Model/App | Negation & absence ("no enemies") not enforceable; phrase dropout weakens "omitted = absent" | 3 | 3 | 9 | B | 12, 15, 22; D8 |
| R36 | App | Game feel / physics not Mario-like enough for the examiner | 3 | 2 | 6 | G | 27 |
| R37 | Model | Seams / incoherence in long levels (G-P5 fails) | 3 | 3 | 9 | M | 20; doc 25 R-5.x |
| R38 | Data | Missing mechanics in the tile set (springs, vines, moving platforms) make some real levels look unbeatable, which skews playability baselines | 3 | 3 | 9 | G | 21, 09 |

---

## 4. Detailed entries

Each entry: **Description · Early-warning trigger · Mitigation (preventive, already in the plan) · Contingency (if it happens) · Owner · Links.**

### Data

**R1 — Upstream data differs from the ledger** (L2 × I4 = 8)
- *Description:* MarioDiffusion's JSON changes between the plan date and Day 1 (new commit), or the clone is partial.
- *Trigger:* `audit_dataset.py` fails any F1–F17 assertion; sha256 differs from a previous machine's MANIFEST.
- *Mitigation:* pin the commit SHA in `data/MANIFEST.json` (D1); verify sha256 on every `verify_env --quick`; keep a zipped copy of the pinned JSON in the sync folder (doc 07).
- *Contingency:* check out the pinned commit (`git checkout <sha>`); if the history was rewritten, use the sync-folder copy. If facts genuinely differ, re-measure, update the ledger §1 + change log, and re-run Day 1–2.
- *Owner:* M · *Links:* 08, 09.

**R2 — Chain reconstruction wrong** (L2 × I5 = 10)
- *Description:* exact-duplicate windows (F4: 67) or a level whose end matches the next level's start could merge or split chains, giving wrong `level_id`s and therefore a leaky or mis-stratified split.
- *Trigger:* chain count ≠ 42, SMB1 chain count ≠ 20, chain lengths ≠ the F11 profile, or chains don't align with `SMB1_Levels.json` / `SMB2_Levels.json` boundaries (F9).
- *Mitigation:* the audit asserts count, per-game counts, min/max lengths and total columns (8,317), and cross-checks against the per-game source files; a unit test re-stitches each chain into a full level and checks width = windows + 15.
- *Contingency:* reconstruct level identity by matching against TheVGLC level files (D2 optional path); otherwise reconstruct per game from `SMB1_Levels.json`/`SMB2_Levels.json` separately.
- *Owner:* M · *Links:* 09, 10.

**R3 — Cross-level segment reuse** (L3 × I3 = 9)
- *Description:* SMB levels reuse patterns, and SMB2 (Lost Levels) reuses SMB1 assets and layouts. A test level may share long segments with a train level even under a level-grouped split.
- *Trigger:* G-P1 leakage test finds test windows within Hamming ≤ 2 of a train window beyond the documented exceptions; a high share of test windows with very small NN distances.
- *Mitigation:* doc 10's split algorithm penalises cross-split near-duplicates; the leakage report lists every test window with NN ≤ 2 and its train partner.
- *Contingency:* **before** `LOCK` is written (Day 2), swap the offending test level for the next candidate. After the lock, document the overlap and report metrics with and without the affected windows.
- *Owner:* M · *Links:* 10; G-P1.

**R4 — Rare concepts too sparse in dev/test** (L4 × I3 = 12)
- *Description:* descending staircases (~104 windows in the whole corpus), upside-down pipes, cannons and coin lines are rare. Stride-8 sub-sampling (D6) of 5–6 held-out levels may leave a handful of prompts → huge CIs or missing concepts.
- *Trigger:* `constraints_report` in `split_v1.json` shows any concept with < 5 windows in dev or test; per-concept CI width > 0.4.
- *Mitigation:* split constraints require concept coverage (doc 10); generative eval picks the stride-8 offset per level that maximises rare-concept coverage (fixed by project seed, D6, doc 30 §5.2); per-concept results are reported with n.
- *Contingency:* pool rare concepts into a "rare" group for headline metrics; mark per-concept cells with n < 5 as "insufficient n"; use P-rand (which samples concepts uniformly) for rare-concept adherence.
- *Owner:* M · *Links:* 10, 22.

**R5 — Caption-detector port mismatch** (L3 × I5 = **15, critical**)
- *Description:* C-score, exact match, the report card and future self-training all depend on `captions/detector.py`. If the re-implementation differs from MarioDiffusion's `assign_caption`, every adherence number is silently wrong.
- *Trigger:* Day 1 reproduction check < 100 % phrase-set equality on the 7,687 scenes; any mismatch cluster (e.g. staircases, ceilings).
- *Mitigation:* port rule by rule with the original's thresholds, keeping its structure (floor/ceiling rows, staircase verticality, `already_accounted` sets); property tests (rendered fixtures for each concept); the reproduction check is a Day 1 exit criterion and a CI test.
- *Contingency:* diff mismatching scenes against the original (read-only, in `data/raw/`); fix until ≥ 99.5 % with documented exceptions (timeline Day 1). **Block all evaluation work until fixed.**
- *Owner:* M · *Links:* 09, 22; F19.

### Process / test discipline

**R6 — Test-set contamination** (L2 × I5 = 10)
- *Description:* any decision (checkpoint, w, threshold, normalizer lexicon, prompt set) influenced by test levels or P-casual-test inflates results (audit #29).
- *Trigger:* `tests/test_no_test_access.py` (static scan) finds a test `level_id` or the P-casual-test file referenced outside `scripts/eval_test.py`; the dataset loader's guard raises; the runs index shows an eval on `P-test` before `FROZEN.json` exists.
- *Mitigation:* `LOCK` (D4); the dataset loader refuses test levels unless `allow_test=True` **and** `FROZEN.json` exists; pre-registered `eval_plan.json` (doc 30 §8.4); the normalizer is tuned only on P-casual-dev and the paraphrase lexicon's train half (D8c).
- *Contingency:* disclose in the report exactly what leaked; promote the 5-fold CV results (D5) to primary evidence; never re-split after Day 2.
- *Owner:* B · *Links:* 10, 22, 30.

### Model / training

**R7 — Unconditional baseline fails G-P3** (L3 × I4 = 12)
- *Description:* samples are ≥ 95 % sky, scattered noise, or lack floors (the PDF's "sky overwhelms everything" symptom). Sky is 87.1 % of tiles (F13).
- *Trigger:* at the 5k-step eval: JS divergence > 0.15, floor validity < 50 %, or per-t-bucket val loss flat in the mid buckets.
- *Mitigation:* G-P2 oracle test (scheduler correct), the Day 4 overfit-one-batch test (optimiser path correct), verified {−1, +1} scaling (D7), both schedules trained (M2).
- *Contingency:* doc 25 R-3.x rows in order: check scaling/one-hot order (F6) → aux CE (M9 option) → channel-weighted MSE (rollback) → capacity checks. Day 6 is the buffer. Never "train longer" (golden rule).
- *Owner:* M · *Links:* 18, 19, 25.

**R8 — Broken multi-tile structures persist** (L3 × I3 = 9)
- *Description:* `<` without `>`, pipe bodies without tops, `b` without `B`.
- *Trigger:* broken-structure rate > 5 % at G-P3, or rising with w during the CFG sweep.
- *Mitigation:* self-attention at 8×8 and 4×4 from the start (M5); EMA eval (M11); sanitizer detects and reports **before** repair (G4).
- *Contingency:* doc 25 R-3.x (aux CE λ = 0.001; check EMA warm-up); select w with the broken-rate term of the composite (E4); display-only repair.
- *Owner:* M · *Links:* 14, 20, 25.

**R9 — Model ignores the text** (L3 × I5 = **15, critical**)
- *Description:* valid levels that don't follow the prompt; counts wrong; no gain from CFG.
- *Trigger:* Day 8 evening check: dev C-score not rising after 10k steps; Day 9 sweep: best C-score < 0.75 or no improvement over w = 1; cross-attention entropy ≈ maximum (uniform).
- *Mitigation:* Day 7 tests (adapter ∈ optimiser, MiniLM ∉ optimiser, null-dropout rate 0.10 ± 0.01); key-padding mask test; cross-attention entropy monitor; online encoding avoids the pre-cache/shuffle conflict (M8).
- *Contingency:* doc 25 R-4.x: verify the mask/null embedding → X3 pooled text → X4 concept-balanced sampling → larger adapter. Unfreezing MiniLM's last layer is a **ledger-level change** (M7), allowed only as a last resort with a change-log row.
- *Owner:* M · *Links:* 15, 18, 25; G-P4.

**R10 — Over-fitting / memorisation** (L3 × I4 = 12)
- *Description:* ~31 training levels and stride-1 windows make memorisation easy (audit #11). The model can then reproduce training windows: high C-score, but zero novelty.
- *Trigger:* copy rate rising across evals; `val/loss_t*` rising while `train/loss_t*` falls (deterministic val noise, doc 30 §5.4); novelty_med below the real-dev reference (E4).
- *Mitigation:* early stopping on the composite, which includes novelty (E4); dropout 0.1 (M3); weight decay 0.01 (M10); augmentation (D8); step budgets (M12).
- *Contingency:* pick an earlier checkpoint; stronger phrase dropout/paraphrase; smaller base width (48) as an experiment; report the memorisation analysis honestly.
- *Owner:* M · *Links:* 19, 22.

**R11 — Under-fitting** (L2 × I3 = 6)
- *Trigger:* dev composite still improving at the 60k cap; train loss per bucket still falling steeply.
- *Mitigation:* LR schedule with warm-up + cosine (M10); per-bucket diagnostics (doc 19).
- *Contingency:* extend the budget by 20k steps (ledger change-log row); check LR isn't decayed too early.
- *Owner:* M · *Links:* 17, 19.

**R12 — Training instability** (L2 × I3 = 6)
- *Trigger:* non-finite loss; grad-norm spikes > 10× the median; `nonfinite_skip` events in `events.jsonl` (doc 30 §4.2).
- *Mitigation:* grad-clip 1.0 (M10); attention softmax in fp32 (H5); NaN guard skips the step and logs it; abort after 3 consecutive skips (`nan_abort`).
- *Contingency:* resume from the last good checkpoint with LR 1e-4; one fp32 debug run to localise the op.
- *Owner:* M · *Links:* 16.

**R13 — High CFG w breaks structures** (L4 × I2 = 8)
- *Trigger:* broken rate or playability worsens monotonically with w in the Day 9 sweep.
- *Mitigation:* x̂₀ clipping (M13); w chosen by the composite, not by C-score alone (E4).
- *Contingency:* lower default w; per-concept w analysis (X7); report the trade-off curve.
- *Owner:* M · *Links:* 15, 20, 23.

**R35 — Negation / absence not enforceable** (L3 × I3 = 9)
- *Description:* regular captions list only **present** concepts, so in raw data "omitted ⇒ absent". Phrase dropout (D8b) deliberately teaches that omitted concepts *may* appear. Users type "no enemies" (the PDF's own contrast prompt does), but there is no canonical "no enemies" phrase in the regular grammar (F14).
- *Trigger:* in the P-casual-dev subset with negations, the forbidden concept appears in > 20 % of samples.
- *Mitigation:* the normalizer marks negated concepts (`negated: true`, doc 12 / doc 31 seam S-7) and removes them from the canonical prompt; absence adherence is reported as its own metric (doc 22).
- *Contingency:* enforce absence by detector-based rejection (best-of-N, N ≤ 4, within the G5 attempt budget); optional negative guidance as an experiment (ε̂ = ε_u + w(ε_c − ε_u) − w_neg(ε_neg − ε_u)), recorded as an X-row in doc 23 if run.
- *Owner:* B · *Links:* 12, 15, 22.

**R37 — Seams / incoherence in long levels** (L3 × I3 = 9)
- *Trigger:* G-P5 seam ratio > 1.3; visible floor cut-offs every 8–16 columns.
- *Mitigation:* overlap 8 + RePaint r = 2 (G1); seam metric (doc 20).
- *Contingency:* doc 25 R-5.x ladder: stride 4 → r = 5 → direct wide sampling.
- *Owner:* M · *Links:* 20, 25.

### Evaluation

**R14 — Detector circularity (Goodhart)** (L3 × I3 = 9)
- *Description:* the same rule-based detector produced the training captions and scores adherence. The model can satisfy rule quirks that a human would not call "three pipes".
- *Trigger:* human spot-check disagrees with the detector on > 15 % of 50 samples.
- *Mitigation:* a 50-sample human spot-check on Day 14 (blind to w/config), reported as detector–human agreement; the limitation is stated in the report (doc 32 §3).
- *Contingency:* report human-judged adherence for the subset next to the detector numbers; soften the claims.
- *Owner:* M · *Links:* 22, 32.

**R21 — Playability checker gives a false "verified beatable"** (L3 × I5 = **15, critical**)
- *Description:* the badge is the product promise. If the checker's jump model is more generous than the Phaser game, a "verified" level can be impossible to play, possibly in front of the examiner.
- *Trigger:* G-P5b agreement < 85 % (or < 90 % on hand labels); any showcase level a human can't finish on Day 18; the checker marks unbeatable a real full level that has no missing-mechanic explanation (see R38).
- *Mitigation:* one `physics.json` read by both the game and the checker (doc 21/27); the checker's jump arcs are **conservative** (a subset of what the game allows), so checker-beatable implies game-beatable; agreement study vs MarioEval.jar; real-level baseline on all 42 full levels; Day 18 human playthrough of every cached showcase level.
- *Contingency:* tighten the arcs (more conservative) and re-verify; the badge text becomes "verified by automated checker"; showcase only levels that have been hand-verified.
- *Owner:* G · *Links:* 21, 27; G-P5b.

**R22 — Java / MarioEval.jar unavailable** (L3 × I2 = 6)
- *Trigger:* `env/capabilities.json → java: false`; the jar crashes on our VGLC text export.
- *Mitigation:* Day 0 capability check; a converter test that round-trips a real level through `grid_to_vglc_text` and the jar.
- *Contingency:* G-P5b alternative: ≥ 90 % agreement with 50 hand-labelled levels (ledger §9); report "paper-parity check not run".
- *Owner:* G · *Links:* 21.

**R23 — Evaluation dominates wall-clock** (L3 × I3 = 9)
- *Trigger:* `sys/eval_wall_s` > 20 % of run wall-clock; dev eval > 10 min.
- *Mitigation:* batched sampling (256 at once), stride-8 prompt sub-sampling (D6), CPU-cheap metrics on uint8 grids, eval every 2k steps (M12).
- *Contingency:* during training, 1 eval seed (3 seeds only for selection and final), eval every 4k steps; playability on a fixed 64-prompt subset during training.
- *Owner:* M · *Links:* 16, 22.

**R24 — Statistical noise picks the wrong winner** (L3 × I3 = 9)
- *Trigger:* ablation Δ within the paired-bootstrap CI; ranking flips between eval seeds.
- *Mitigation:* pre-registered decision rules (doc 30 §14), paired bootstrap (E3), "simpler wins ties".
- *Contingency:* second training seed for the top-2 configs; if still a tie, keep the simpler one.
- *Owner:* M · *Links:* 23, 22.

**R38 — Missing mechanics distort playability baselines** (L3 × I3 = 9)
- *Description:* the 13-tile set has no springboards, vines, moving lifts, Lakitu or wind (Lost Levels). Some real levels may require mechanics that aren't represented, so the checker (correctly, for our tile semantics) calls them unbeatable.
- *Trigger:* the real-level checker baseline (Day 5) marks some of the 42 full levels unbeatable.
- *Mitigation:* classify each failing real level as "missing mechanic" vs "checker bug" by inspecting the stuck column (the `stuck_col` output, doc 21); exclude only documented missing-mechanic cases from the baseline, with the list published.
- *Contingency:* report playability against the window-level baseline (≈ 93 % for real windows per F18) instead of full levels.
- *Owner:* G · *Links:* 21, 09.

### Infrastructure

**R15 — GPU switching corrupts a run** (L3 × I4 = 12)
- *Trigger:* `HANDOFF.md` lock held by the other machine at resume; two `run.json` segments overlap in steps; `ckpt_index.json` sha mismatch.
- *Mitigation:* doc 07 lock protocol; atomic checkpoints and copy-not-symlink (doc 30 §3.1); resume refuses if the lock is held or the hash mismatches; Day 4 resume drill.
- *Contingency:* keep the branch with the highest verified step; mark the other `superseded` in `run.json`; never merge histories.
- *Owner:* B · *Links:* 07, 30.

**R16 — Laptop thermal throttling** (L4 × I2 = 8)
- *Trigger:* GPU temperature ≥ 85 °C; it/s drops > 30 % vs `hw_report.json`.
- *Mitigation:* H9 rules (AC power, performance mode, cooling); watchdog stops at 87 °C sustained for 60 s; checkpoints every 2k steps **and** every 15 min of wall-clock.
- *Contingency:* resume on the desktop (doc 07).
- *Owner:* M · *Links:* 07; H9.

**R17 — Version mismatch between machines** (L3 × I3 = 9)
- *Trigger:* `verify_env` reports different torch minor versions; a checkpoint load refuses on torch version (doc 30 §3.3).
- *Mitigation:* V2 rule (same minor on both); lock files on Day 0.
- *Contingency:* align both machines to the lower common version (2.14 + cu126 if a driver can't be updated).
- *Owner:* B · *Links:* 06; V2.

**R18 — Windows-specific breakage** (L3 × I2 = 6)
- *Mitigation:* no DataLoader workers (H7), `torch.compile` off (H6), `pathlib` everywhere, no symlinks, Windows `MAX_PATH`-safe run IDs.
- *Contingency:* run the affected job under WSL2 on that machine (same lock file).
- *Owner:* B · *Links:* 06.

**R19 — Disk exhaustion** (L2 × I3 = 6)
- *Trigger:* free disk < 15 GB.
- *Mitigation:* retention policy and `launch.py` disk guard (doc 30 §12).
- *Contingency:* `prune_ckpts.py`; move old ablation runs to external storage.
- *Owner:* B · *Links:* 30.

**R20 — Sync-folder conflicts / partial uploads** (L3 × I3 = 9)
- *Trigger:* files named "… (conflicted copy)"; checkpoint sha mismatch at resume; a sync client still uploading.
- *Mitigation:* sync only closed, renamed files; verify sha256 at resume; never let two machines write the same run directory at the same time (doc 07).
- *Contingency:* transfer by USB/LAN copy; re-verify hashes.
- *Owner:* B · *Links:* 07, 30.

### App / game / assets

**R25 — Asset download fails / licence unclear** (L3 × I2 = 6)
- *Mitigation:* multiple source URLs, sha256 manifest, licence files stored (doc 26); placeholder sprites never block (A2).
- *Contingency:* ship the placeholder skin; log the fallback (CLAUDE.md rule 12).
- *Owner:* G · *Links:* 26.

**R26 — Phaser in a Streamlit iframe misbehaves** (L3 × I4 = 12)
- *Description:* `st.components.v1.html` renders in an iframe. Keyboard focus, relative asset URLs and offline loading are the typical failure points.
- *Trigger:* the Day 13 Playwright smoke test fails (no canvas, player doesn't move); arrow keys scroll the page instead of moving the player.
- *Mitigation:* inline/base64 embedding of the atlas and level, vendored Phaser (A4), a click-to-focus overlay, `preventDefault` on game keys, and a standalone HTML export (doc 27/28).
- *Contingency:* "Open in new tab" with the standalone HTML; for the demo, cached showcase levels as standalone files.
- *Owner:* G · *Links:* 27, 28.

**R27 — Latency > 10 s on the laptop** (L2 × I3 = 6)
- *Trigger:* G-P7 timing check fails.
- *Mitigation:* batched CFG (cond + uncond in one pass), bf16 inference, model warm-up at start-up, `st.cache_resource`.
- *Contingency:* DDIM 25 steps / r = 1 for the interactive path (still EMA; reported metrics stay at the frozen settings); CPU fast mode as last resort (H10).
- *Owner:* G · *Links:* 20, 28.

**R28 — Normalizer fails on the examiner's phrasing** (L4 × I3 = 12)
- *Trigger:* P-casual-dev concept accuracy < 80 %; unrecognised-word rate > 20 % on casual prompts.
- *Mitigation:* large synonym lexicon, digit and number-word → bucket mapping (F15), multi-scene splitter; the app shows the normalised prompt and highlights unrecognised words.
- *Contingency:* prompt chips (example prompts) in the UI; MiniLM still embeds the canonical text, so fall back to the closest canonical phrase by embedding similarity (doc 12).
- *Owner:* G · *Links:* 12, 28.

**R36 — Game feel not Mario-like** (L3 × I2 = 6)
- *Mitigation:* SMB-like constants (≈ 4-tile jump height, run speed) in `physics.json`, tuned on real levels in the Day 3 skeleton.
- *Contingency:* present it as a "Mario-like platformer"; the claim is about level layout, not a physics clone.
- *Owner:* G · *Links:* 27.

### Legal

**R29 — IP / licensing objection** (L2 × I3 = 6)
- *Mitigation:* CC0 Kenney sprites (A1); Nintendo-derived sprites never committed (A3); level data used for non-commercial academic research with attribution (MarioDiffusion MIT; VGLC cited); a licence section in the report and the data card (doc 32).
- *Contingency:* remove any classic skin; present layouts with CC0 sprites only.
- *Owner:* B · *Links:* 26, 32.

### Schedule / process / demo

**R30 — Critical-path slip** (L3 × I5 = **15, critical**)
- *Description:* Phases 1 → 2 → 3 → 4 → 5 → 7b → 6 are serial (doc 04). Two failed gates can consume both buffers.
- *Trigger:* any gate more than 1 day late vs doc 03; Day 6 buffer consumed by Phase 3; G-P4 not passed by end of Day 10.
- *Mitigation:* the G track runs in parallel (game, checker, normalizer); buffers on Days 6 and 19; failing gates go straight to doc 25 rollbacks (no "train longer").
- *Contingency:* switch to the compressed variant (doc 03 §Compressed): cut CV, cut ablations to X1, merge D17 and D18. **Never cut** the level-grouped split, the checker, the one-shot test or the MVP.
- *Owner:* B · *Links:* 03, 04.

**R31 — Scope creep** (L3 × I4 = 12)
- *Trigger:* commits under `src/text2mario/features/` or feature branches before the Day 13 MVP gate.
- *Mitigation:* S3; doc 31 lists seams only; CLAUDE.md workflow rules; FF6-lite is the only feature allowed, and only in the Day 19 buffer.
- *Contingency:* park the work on a branch; revert from `main`.
- *Owner:* B · *Links:* 02, 31.

**R32 — Claude Code drifts from the spec** (L3 × I3 = 9)
- *Trigger:* numeric literals in `src/` outside `configs/` (lint check); PROGRESS "Doc conflicts" growing without ledger updates; gate scripts edited to pass.
- *Mitigation:* ledger-first rule; all numbers in YAML (CLAUDE.md rule 4); `scripts/check_ledger_sync.py` compares key YAML values with ledger rows; gate thresholds are read from the ledger-backed config, never hard-coded in the gate.
- *Contingency:* weekly spec audit at the full reviews (Days 6, 13, 20); revert drifting commits.
- *Owner:* B · *Links:* 34, 30.

**R33 — Demo-day failure** (L2 × I5 = 10)
- *Trigger:* any failure in the Day 18 hardening checklist; a pending OS update on the laptop.
- *Mitigation:* offline cold-boot rehearsal, CPU fast mode, 12 cached showcase levels, standalone HTML on USB, OS updates paused, charger and adapters packed (doc 32 §8).
- *Contingency:* use the desktop or another laptop with the standalone HTML files; a recorded screen capture of the demo as the last resort.
- *Owner:* G · *Links:* 28, 32.

**R34 — Single point of knowledge / illness** (L2 × I4 = 8)
- *Mitigation:* everything scripted; `PROGRESS.md` and `HANDOFF.md` updated every session; the ledger is current.
- *Contingency:* compressed timeline; the other person (or a fresh Claude Code session) resumes from `PROGRESS.md`.
- *Owner:* B · *Links:* 30 §13.

---

## 5. Top-5 summary (current)

Ranking rule: score → impact → on the critical path → earliest trigger day.

| Rank | ID | Why it's top-5 | First trigger day | Single most important action |
|---|---|---|---|---|
| 1 | **R5** Detector mismatch | Silently invalidates every adherence number and the report card | Day 1 | 100 % caption reproduction as a hard exit criterion + CI test |
| 2 | **R9** Text ignored | Fails G-P4, the core claim of "text-to-level" | Day 8–9 | Day 7 optimiser/mask/null tests before the long run |
| 3 | **R30** Critical-path slip | Two failed gates eat both buffers | Day 5 onward | Pre-agreed rollbacks + the compressed variant |
| 4 | **R21** False "verified beatable" | Demo credibility; the product promise | Day 11, Day 18 | Shared `physics.json` + conservative arcs + human playthrough of showcase levels |
| 5 | **R7** Unconditional baseline fails | First model gate; blocks Phase 4 | Day 5 | G-P2 oracle test + overfit-one-batch before any long run |

**Watchlist (score 12):** R4, R10, R15, R26, R28, R31.

---

## 6. Review cadence and process

| When | Who | What | Output |
|---|---|---|---|
| **Every session start** (2 min) | whoever works | Scan the day's watch list (§7) and check the triggers against yesterday's `events.jsonl` / gate results | "Risks triggered" section in `PROGRESS.md` (doc 30 §13) |
| **Every gate** (G-P1 … G-P7) | track owner | Re-score the risks linked to that phase; close risks that can no longer occur (e.g. R2 after G-P1 passes) | Row appended to `reports/risks/risk_log.csv` |
| **Full review: Days 6, 13, 20** | both | Re-score all open risks; update §2–§5; add new risks; run the R32 spec audit | Updated doc 33 + `risk_log.csv` |
| **Event-driven** | automatic | Any `fallback`, `oom_fallback`, `watchdog_thermal`, `nan_abort`, `crash`, or a gate `pass: false` → look up the linked risk and act on its contingency | `PROGRESS.md` entry |

**`reports/risks/risk_log.csv` columns:** `date, day, risk_id, event (rescored|triggered|realised|closed|added), L, I, score, note, link`.
`scripts/make_figures.py` plots the **risk burn-down** (sum of open scores per day) for the report appendix. Adding a risk: next free ID (R39…), all fields filled, linked doc named, and a log row with `event=added`.

**Closing rules:** a risk closes when its trigger window has passed and its gate is green (e.g. R1/R2/R3 close at G-P1; R7 at G-P3; R9 at G-P4; R21 stays open until Day 20 because the demo depends on it).

---

## 7. Day → risks to watch

| Day | Watch | Day | Watch |
|---|---|---|---|
| D0 | R17, R18, R25, R19 | D11 | R21, R22, R37, R38 |
| D1 | R1, R2, R5 | D12 | R27, R28 |
| D2 | R3, R4, R6 | D13 | R26, R21 |
| D3 | R36, R26 (skeleton) | D14 | R6, R10, R14, R24 |
| D4 | R15, R16, R12, R19 | D15 | R15, R16, R20 (parallel CV runs) |
| D5 | R7, R8, R16, R38 | D16 | R6 (one-shot discipline) |
| D6 | R7 (rollback), R22 | D17 | R29, R32 |
| D7 | R9 (tests), R35 | D18 | R33, R21, R26 |
| D8 | R9, R23, R10 | D19 | R31 (only FF6-lite allowed) |
| D9 | R9, R13, R14 | D20 | R33, R34 |
| D10 | R24, R10 | **Always** | R30, R31, R32, R20 |

---

## 8. Realised-risk log template (`reports/risks/realised.md`)

```markdown
### R<id> — <title> — realised on Day <n> (<date>)
- Trigger observed: …
- Impact (time lost, metric affected): …
- Contingency applied (incl. doc 25 row): …
- Residual risk / re-score: L? × I? = ?
- Lessons → doc/ledger changes: … (change-log row id)
```
