# 24 — Verification Gates & Test Inventory

> **Scope:** the executable form of ledger §9. One script per gate (`scripts/gate_p1.py` … `scripts/gate_p7.py`, plus `scripts/gate_p5b.py`), one shared result format, and the full `pytest` inventory with markers, commands and property tests.
> **Rule (CLAUDE.md §3):** a phase is finished only when its gate writes `reports/gates/PN/RESULT.json` with `"pass": true`. The next phase's scripts check that file. A failed gate sends you to doc 25 (rollback matrix), never to "train longer".
> Metric definitions live in doc 22. This doc fixes *what is checked, against which threshold, with which inputs, producing which artifacts.*

---

## 0. Gate mechanics (shared by all gate scripts)

### 0.1 Runner contract
- Location: `scripts/gate_p{1,2,3,4,5,5b,6,7}.py`; shared helpers in `src/text2mario/gates/` (`result.py`, `checks.py`, `prereq.py`).
- Exit codes: **0** pass · **1** fail (one or more checks failed) · **2** error (crash, missing input, prerequisite not met).
- Every run writes `reports/gates/PN/RESULT_<UTC-timestamp>.json` and copies it to `reports/gates/PN/RESULT.json` (the latest). Older results are kept for history.
- **Prerequisites** are enforced: the script reads the prerequisite gates' `RESULT.json` and aborts with exit 2 unless each is `pass: true` **and** was produced at a git SHA that is an ancestor of HEAD.

| Gate | Prerequisites |
|---|---|
| G-P1 | — |
| G-P2 | G-P1 |
| G-P3 | G-P2 |
| G-P4 | G-P3 |
| G-P5 | G-P4 |
| G-P5b | G-P3 (it uses Phase-3 unconditional samples and real train/dev levels; **must pass before the playable term is used in any selection, Day 8 onward**) |
| G-P7 | G-P5, G-P5b |
| G-P8 | G-P7 (dry run on a candidate checkpoint Day 12; final with the frozen checkpoint Day 14) |
| G-P6 | G-P7, G-P8 (the demo, the deployment and the reported numbers come from the same frozen checkpoint) |

- **Thresholds are read from `configs/gates.yaml`**, whose values mirror ledger §9 (each key has a comment naming the ledger row). Gate scripts never hard-code thresholds.
- **Borderline rule** (doc 22 §7.6): if a proportion or mean threshold lies inside the check's 95 % CI, the script draws **one** extension of equal size with fresh seeds, decides on the pooled estimate, and records `"extended": true`.
- **Waivers:** `--waive <check_id> --reason "<text>"` marks a check as waived in RESULT.json. That is allowed only for checks tagged `waivable: true` in `configs/gates.yaml`, and it is logged to `PROGRESS.md`. **Never waivable:** anything about test discipline (P1.16, P6.*), tile order (P1.04) or leakage (P1.15).

### 0.2 `RESULT.json` schema (v1)
```json
{
  "schema_version": 1,
  "gate": "G-P3",
  "pass": true,
  "status": "pass",                         // pass | fail | error
  "started_at": "2026-11-22T21:04:10+05:00",
  "finished_at": "2026-11-22T21:09:55+05:00",
  "git_sha": "3f2c1ab", "git_dirty": false,
  "machine": "desktop_3060ti", "gpu": "NVIDIA GeForce RTX 3060 Ti", "torch": "2.15.0+cu130", "python": "3.12.x",
  "config_hash": "sha256:…",
  "inputs": {"checkpoint": {"path": "runs/U-cos/best_ema.pt", "sha256": "…"}, "split_lock": "sha256:…"},
  "prerequisites": {"G-P2": {"pass": true, "result_sha256": "…", "git_sha": "…"}},
  "checks": [
    {"id": "P3.01", "name": "floor_valid", "value": 0.914, "op": ">=", "threshold": 0.85, "ci95": [0.879, 0.945],
     "n": 256, "pass": true, "waivable": false, "ledger_ref": "§9 G-P3", "extended": false}
  ],
  "diagnostics": {"S_P3": {"linear": 0.81, "cosine": 0.84}, "schedule_choice": "cosine"},
  "warnings": ["P3.03 JS 0.031 > drift warning 0.025"],
  "artifacts": ["reports/gates/P3/sample_sheet_cosine.png"],
  "waivers": [],
  "notes": ""
}
```
`tests/gates/test_result_schema.py` validates every RESULT.json in the repo against `src/text2mario/gates/result_schema.json`.

### 0.3 Status board
`python scripts/gate_status.py` prints, for each gate: status, date, git SHA, failed check IDs, and whether HEAD has moved since then (stale ⚠). It is run as part of `scripts/check_all.py` (§3).

---

## 1. Gate specifications

### G-P1 — Data (Day 1–2) · `scripts/gate_p1.py`
**Inputs:** `data/raw/MarioDiffusion/datasets/Mar1and2_LevelsAndCaptions-regular.json` (read-only), `smb.json`, `data/MANIFEST.json`, `data/processed/index.parquet`, `data/splits/split_v1.json`, `data/splits/LOCK`, `data/prompts/*`, `src/text2mario/data/{tiles,dataset}.py`, `src/text2mario/render/renderer.py`, `src/text2mario/captions/detector.py`.

| ID | Check | Threshold / expected | Ledger |
|---|---|---|---|
| P1.01 | total records | = 7,687 | F3 |
| P1.02 | unique scenes (byte hash) | = 7,620 | F4 |
| P1.03 | encoding | every value an int in 0..12; shape 16×16 | F5 |
| P1.04 | tile order | `tiles.ID_TO_CHAR == sorted(smb.json["tiles"].keys())` = `- < > ? B E Q S X [ ] b o` | F6 |
| P1.05 | chains / games | 42 chains; first 3,606 records = 20 SMB1 chains; next 4,081 = 22 SMB2 chains | F7, F9, F10 |
| P1.06 | chain lengths | min 130, max 358; Σ(len + 15) = 8,317 columns | F11 |
| P1.07 | padding rows | non-sky cells in rows 0–1 = 64 | F12 |
| P1.08 | tile frequencies | each within ±0.01 percentage points of the F13 values (87.13, 8.66, 1.70, …) | F13 |
| P1.09 | caption statistics | words 2–28 (mean 11.39 ± 0.01); phrases 1–10 (mean 4.03 ± 0.01); 62 distinct phrases; 44-word vocabulary | F14 |
| P1.10 | official-split leak reproduces | test covers 42 levels; median NN 8; 83.9 % ≤ 16; 2.1 % exact (± 0.1 pp) | F16–F17 |
| P1.11 | detector parity | stored caption reproduced (phrase-set equality) for **100 %** of 7,687; ≥ 99.5 % allowed only with `data/processed/detector_exceptions.json` documenting each mismatch | timeline D1 |
| P1.12 | split structure | 31 / 5 / 6 levels; test = 3 SMB1 + 3 SMB2; dev = 2+3 or 3+2; pairwise disjoint; union = 0..41 | D3 |
| P1.13 | CV folds | 5 folds partition the 36 non-test levels; each fold contains both games; sizes 7–8 | D5 |
| P1.14 | concept coverage | every T16 concept has ≥ 5 windows in train, dev and test, except concepts listed in `split_v1.json["coverage_exceptions"]` with counts | timeline D2 |
| P1.15 | **leakage (operational)** | see box below | §9 G-P1 |
| P1.16 | locks | `sha256(split_v1.json) == LOCK`; `PROMPTS_LOCK` matches the test-side prompt files | D4 |
| P1.17 | tensor contract | batch values ∈ {−1, +1}; exactly one +1 per cell; shape (B, 13, 16, 16); storage uint8 on the GPU (≈ 2 MB) when CUDA is present | D7, H7 |
| P1.18 | round trip | `argmax(onehot(G)) == G` for all 7,687 scenes | D7 |
| P1.19 | render orientation | (a) corpus row-15 solid share (≈ 65.6 %) > row-2 solid share; (b) renderer pixel blocks equal the atlas sprite of `G[r, c]` for 10 random windows (exact pixel equality, row 0 at the top); (c) for every `<` at column c < 15 the block right of it is the `>` sprite | timeline D2 |
| P1.20 | renders saved | 10 random DataLoader items → `reports/gates/P1/sample_XX.png` with the caption printed underneath | timeline D2 |
| P1.21 | prompt assets | P-dev and P-test counts = number of dev/test windows with `col_offset % 8 == 0`; P-rand-dev/test = 100 each; P-casual-dev/test = 30/30; paraphrase lexicon train/test variants disjoint (doc 12) | E2, D8 |

> **P1.15 — leakage check, operational form.** The ledger's text is "no test window within Hamming ≤ 2 of any train window, *except documented reused segments*". Measured reality: **7.5 % of all windows** have a window from *another* level within ≤ 2 tiles (LOLO, stride 1). About half are simple generic segments (complexity < 12 non-sky tiles in rows 0–14), and the rest concentrate in two **near-duplicate level families** (chain-order IDs): **{5, 7, 16}** and **{19, 20, 23, 28, 29, 31, 33, 36}**. The family membership comes from union-find over level pairs where > 10 % of complex windows have an NN ≤ 2 in the other level. The literal "zero windows" rule is therefore unsatisfiable for any split. The check is implemented as:
> 1. For each dev and test level ℓ: `s_ℓ` = share of its stride-1 windows with NN ≤ 2 to **train** windows → **s_ℓ ≤ 0.10** (background median 3.1 %).
> 2. Among windows of ℓ with complexity ≥ 12: share with NN ≤ 2 to train **≤ 0.05**.
> 3. **Family rule:** no near-duplicate family (recomputed by `make_splits.py`, written to `split_v1.json["families"]`) is split across train and dev/test.
> 4. Every window with complexity ≥ 12 and NN ≤ 2 to train is listed in `data/splits/reused_segments.json` (level, offset, NN level, NN offset, distance). This auto-generated file **is** the "documented reused segments" of the ledger. Windows with complexity < 12 are exempt as generic segments.
> This interpretation is flagged in the conflicts report; doc 10's selection algorithm should treat families as atomic units.

**Artifacts:** `reports/gates/P1/{sample_00..09.png, audit.json, leakage.json, coverage.csv, families.json}`. **Runtime:** ~1–2 min (CPU is fine; NN computation is chunked).

### G-P2 — Scheduler (Day 3) · `scripts/gate_p2.py`
**Inputs:** `src/text2mario/diffusion/scheduler.py`, 1,024 real **train** windows (seeded draw), both schedules, all buffers in fp64.

| ID | Check | Threshold | Ledger |
|---|---|---|---|
| P2.01 | β ranges | linear: β₁ = 1e-4, β_T = 0.02 (±1e-12), non-decreasing; cosine: 0 < β ≤ 0.999 | M1, M2 |
| P2.02 | ᾱ | strictly decreasing; **ᾱ_T < 1e-4 (linear; pack 4.036e-5)**, **< 1e-8 (cosine; pack 2.43e-9)** | §9 G-P2 |
| P2.03 | forward moments | for t ∈ {0, 250, 500, 750, 999}: \|mean(x_t) − √ᾱ_t·mean(x₀)\| ≤ 0.01 and \|std(x_t) − √(ᾱ_t·var(x₀) + 1 − ᾱ_t)\| ≤ 0.01 | timeline D3 |
| P2.04 | endpoint | x₉₉₉: mean within ±0.02, std within 1 ± 0.02, both schedules (pack: −0.0046/1.0013 linear, −0.0021/0.9988 cosine) | §9 G-P2 |
| P2.05 | inverse | `predict_x0_from_eps(q_sample(x₀, t, ε), t, ε) == x₀` (fp64 max abs error < 1e-10) | doc 13 |
| P2.06 | **oracle-ε DDIM** | 50 steps, η = 0, ε supplied by the oracle `(x_t − √ᾱ_t x₀)/√(1−ᾱ_t)` → **MSE < 1e-6 (fp64), argmax agreement 100 %** (pack: 0.0 / 1.0, both schedules) | §9 G-P2 |
| P2.07 | DDPM posterior | posterior mean coefficients match the closed form; β̃_t = β_t(1−ᾱ_{t−1})/(1−ᾱ_t) ≥ 0; the t = 0 variance edge case is handled (clipped log-variance) | doc 13 |
| P2.08 | DDIM η = 1 | with stride 1, σ_t² equals β̃_t (rel. error < 1e-9) | doc 13 |
| P2.09 | timesteps | length = n_steps (50); strictly decreasing; unique; max ≤ 999; min = 0; spacing as defined in doc 13 | M13 |
| P2.10 | renoise | x_t produced by `renoise(x_{t_prev} → t)` from x_{t_prev} ~ q(·\|x₀) has q(x_t\|x₀) moments (\|Δ\| ≤ 0.01) | G1, doc 20 |
| P2.11 | confetti / intact | argmax agreement with x₀ at t = 0 = 100 %; JS(argmax histogram at t = 999 ‖ uniform-13) < 0.01 | §9 G-P2 |
| P2.12 | strips saved | t ∈ {0, 250, 500, 750, 999} × {linear, cosine} rendered for 2 windows → `reports/gates/P2/strip_<sched>.png` | timeline D3 |

**Runtime:** < 1 min (CPU).

### G-P3 — Unconditional baseline (Day 5–6) · `scripts/gate_p3.py`
**Inputs:** `runs/U-lin/best_ema.pt`, `runs/U-cos/best_ema.pt` (15k steps), `REFERENCES.json`, train-split windows. **Samples:** 256 per schedule from P-uncond seeds (doc 22 §2.3), DDIM 50, EMA.

| ID | Check | Threshold | Ledger |
|---|---|---|---|
| P3.01 | floor_valid (doc 22 §3.5) | **≥ 0.85** (real reference 0.9991) | §9 G-P3 |
| P3.02 | broken_rate (doc 22 §3.4) | **≤ 0.05** (real 0) | §9 G-P3 |
| P3.03 | tile JS, base 2 (doc 22 §3.10) | **≤ 0.05**; warning if > 0.025 | §9 G-P3 |
| P3.04 | copy_rate (nn ≤ 4) | **≤ 0.10**, or the calibrated `copy_rate_ref + 0.05` once the ledger amendment is logged (doc 22 §3.8 box). The script prints `copy_rate_ref` beside it either way | §9 G-P3 |
| P3.05 | schedule choice (diag.) | `S_P3` per schedule (doc 22 §3.13); choose the max among **passing** schedules; tie (\|ΔS\| < 0.01 or the paired CI includes 0) → cosine (M2). Written to `diagnostics.schedule_choice` | M2 |
| P3.06 | loss health (diag.) | per-bucket dev loss at 15k vs the context-free Bayes reference (doc 25 §0.2): warning if any mid-SNR bucket is > 0.7× the reference | doc 25 |
| P3.07 | samples saved | 64-sample sheet per schedule; the 8 lowest-nn samples beside their nearest train windows; tile-frequency table; per-row JS chart; loss curves | timeline D6 |

**Pass** = at least one schedule passes P3.01–P3.04, and the chosen schedule is a passing one. **Runtime:** ~2–5 min per schedule on the GPU (sampling + checker if available).

### G-P4 — Conditional + CFG (Day 9) · `scripts/gate_p4.py`
**Inputs:** C-main `best_ema.pt`, P-dev, the CFG-sweep outputs (doc 22 §6.2), the optimiser state in the checkpoint, the normalizer (doc 12).

| ID | Check | Threshold | Ledger |
|---|---|---|---|
| P4.01 | dev C-score at w\* (3 seeds, CRN) | **≥ 0.75** | §9 G-P4 |
| P4.02 | CFG helps | paired ΔCscore(w\* − w=1) > 0 with the 95 % CI above 0 | §9 G-P4 |
| P4.03 | optimiser membership | numel of params in the optimiser state = U-Net + adapter (+ optional X3 linear), **exactly**; no MiniLM tensor is in any param group; every adapter tensor is present | §9 G-P4, doc 01 #10 |
| P4.04 | PDF contrast test | A = normalize("full floor with three pipes and no enemies"), B = normalize("staircase with gap and several enemies"), 8 seeds each (CRN). Pass if **all three** hold and each holds in ≥ 6/8 seed pairs: mean pipe count A > B; enemy presence A < B; (gap count + staircase presence) B > A. The 2×8 figure is saved. The normalised strings are logged | §9 G-P4 |
| P4.05 | diagnostics | ΔC vs null (`cscore_null_dev`), per-concept table, broken and playable at w\*, S-vs-w curve | doc 22 |

**Non-blocking warning:** if `C(w*) − cscore_null_dev < 0.15`, the gate prints "conditioning adds little beyond realism" even when P4.01 passes (see conflicts report: a candidate ledger tightening). **Runtime:** sweep ≈ 6 × 350 samples (~3–6 min GPU) + analysis.

### G-P5 — Full-level generation (Day 11) · `scripts/gate_p5.py`
**Inputs:** config-v1 checkpoint, L-single-dev (50 prompts × 2 seeds = **100 levels**, W_gen = 112, full quality: 50 steps, r = 2), the doc-20 seam estimator fitted on train levels, `REFERENCES.json["seam_ratio_real"]`.

| ID | Check | Threshold | Ledger |
|---|---|---|---|
| P5.01 | seam ratio (doc 22 §3.11) | **≤ 1.3** (cluster-bootstrap CI over levels reported) | §9 G-P5 |
| P5.02 | broken after sanitizer repair | **= 0** across all 100 levels | §9 G-P5, G4 |
| P5.03 | G5 loop limits | max 5 attempts per window and 20 per level, verified from the generator's event log | G5 |
| P5.04 | diagnostics | pre-repair broken rate, level C-score, floor_valid per window, verified rate, attempts histogram, GPU seconds per level | doc 22 §3.14 |

**Artifacts:** 6 full-level renders (best/median/worst by S_level), a seam close-up figure, `levels/*.json` (schema doc 29).

### G-P5b — Playability checker agreement / validity (**Day 7**, before any use of the playable term) · `scripts/gate_p5b.py`
**Why Day 7:** the `playable` term of the dev composite S (E4) is used for checkpoint/CFG selection from Day 8. A checker used for selection must be validated first (ledger E4 circularity rule). Until `reports/gates/P5b/RESULT.json` has `pass: true`, all selections use `S_noplay`.
**Inputs:** `src/text2mario/playability/checker.py`, `physics.json` (frozen end of Day 6), `env/capabilities.json` (Java available?), MarioEval.jar bridge (doc 21).
**Agreement set (fixed, seeded, saved as `reports/gates/P5b/agreement_set.json`):** 100 real windows from **train + dev** levels (stratified to ~50/50 by fast-checker verdict where possible), 100 **unconditional Phase-3 samples** (no conditional model exists on Day 7), 100 framed real train/dev level segments of width 48 (keeps jar runtime modest). On Day 11 the agreement is re-run as a non-gating monitor on 30 generated full levels. Test levels are never used. **Never test levels.**

| ID | Check | Threshold | Ledger |
|---|---|---|---|
| P5b.01 | agreement (fast vs jar), 300 items | **≥ 0.85** | §9 G-P5b |
| P5b.01-alt | if Java is unavailable: agreement with 50 human-labelled levels (played in the browser game; labels in `reports/gates/P5b/human_labels.csv`) | **≥ 0.90** | §9 G-P5b |
| P5b.02 | diagnostics | confusion matrix, Cohen's κ, **false-"verified" rate** (fast = solved, jar = failed; warning if > 10 %), jar timeouts, 20 disagreements rendered for inspection | doc 21 |

Jar failures on levels a human can beat do happen (agent imperfections). Disagreements are inspected and categorised (checker too lenient / too strict / jar artefact) in `disagreements.md` before any fix (doc 25 R-5.5).

### G-P6 — Evaluation discipline (Day 14–15) · `scripts/gate_p6.py`

| ID | Check | Threshold | Ledger |
|---|---|---|---|
| P6.01 | dev suite complete | `metrics.json` for P-dev, P-rand-dev, P-casual-dev, P-div and L-single-dev contains every E1 metric with a non-null `ci95` | §9 G-P6, E1 |
| P6.02 | freeze integrity | `FROZEN.json` exists; checkpoint sha256 and `configs/final.yaml` hash match | timeline D14 |
| P6.03 | pre-registration | `reports/final/eval_plan.json` committed **before** the first test-run directory was created (git commit time < run start); its hashes match | doc 22 §6.5 |
| P6.04 | one-shot | exactly one directory in `reports/final/test_runs/`, **or** (k + 1) directories with k documented metric-bug reruns in `TEST_LEDGER.md` | D4 |
| P6.05 | locks untouched | `LOCK` and `PROMPTS_LOCK` each have exactly one commit in `git log` (Day 2) and still match | D4 |
| P6.06 | results | `RESULTS.md` exists, names the test-run ID, and contains H1–H4 verdicts | doc 22 §6.5 |

### G-P7 — App: "Type it, play it" (Day 13; re-run Day 16) · `scripts/gate_p7.py`
**Measure first, then pin (ledger G-P7):** the thresholds below (20 s, 10 s, 45 s) are **provisional**; no GPU measurement exists yet. The Day-12 run writes `reports/app/latency.json`; the pinned values replace them in the ledger and in `configs/gates.yaml` (with a ledger §14 row). Gate scripts read `configs/gates.yaml`, never constants in code.
**Must run on the laptop profile** (`laptop_4050`) for P7.01–P7.03. Otherwise those checks are marked `not_on_target`, which fails the gate.

| ID | Check | Threshold | Ledger |
|---|---|---|---|
| P7.01 | cold start | spawn `streamlit run app/streamlit_app.py --server.headless true` with `HF_HUB_OFFLINE=1` → time until the app writes `runs/app_ready.json` (model + encoder loaded, warm-up generation done); median of 3 cold runs **< 20 s** | §9 G-P7 |
| P7.02 | GPU latency | 112-wide generate + verify, full quality (50 steps, r = 2, G5 loop on), 10 prompts after warm-up: median **< 10 s** (p90 reported) | §9 G-P7 |
| P7.03 | CPU fast mode | `CUDA_VISIBLE_DEVICES=""`, 25 steps, r = 0 (H10), 3 prompts: median **< 45 s** | §9 G-P7, H10 |
| P7.04 | keyboard play, Chrome + Edge | Playwright, channels `chrome` and `msedge`: open the exported standalone HTML of 3 levels; click to focus; hold ArrowRight 1 s → player x increases ≥ 2 tiles; press jump → player y rises then returns; no uncaught console errors | §9 G-P7 |
| P7.05 | offline | the standalone HTML contains no `http(s)://` URLs; the Playwright run blocks all network requests except `file://` and localhost, and the game still loads | doc 27 |
| P7.06 | badge truth (diag.) | for 20 prompts, the badge state equals the checker verdict stored in the level JSON | G5 |

### G-P8 — Streamlit Community Cloud deployment (dry run Day 12, final Day 14; re-check Days 16-18) · `scripts/gate_p8.py`
**Inputs:** the public app URL, the deployed commit SHA, the checkpoint sidecar (sha256 must equal `FROZEN.json` on the final run).
| ID | Check | Threshold | Ledger |
|---|---|---|---|
| P8.01 | URL reachable | HTTP 200 and the "Ready" marker within the cold-start time measured on Day 12 (pinned) | §9 G-P8 |
| P8.02 | profile | About panel shows `profile: cloud` and checkpoint sha256 (first 12) equal to `FROZEN.json` (final run) | CL3 |
| P8.03 | generation | 3 prompts, width 48, fast mode: level, canonical caption, badge and PNG appear; median time within `cloud.time_budget_s` (provisional 60 s, pinned Day 12) | CL2, CL6 |
| P8.04 | fallback | with `time_budget_s` forced to 1 s the app serves a labelled cached showcase level and logs the fallback | CL6 |
| P8.05 | game | Playwright on the public URL: canvas present, ArrowRight moves the player, no console errors, inside the Streamlit iframe | doc 27 cloud iframe |
| P8.06 | no external requests | no requests to hosts other than the Streamlit platform | CL7 |
| P8.07 | memory (diag.) | resident memory vs the pinned cap | CL5 |
Evidence is copied to `evidence/deployment/`. If the build fails, the documented state is the cached-only fallback (doc 25 R-6.6).

---

## 2. Test inventory (`tests/`, pytest)

**Markers** (registered in `pyproject.toml`): *(none)* = fast unit test · `slow` (> 5 s) · `gpu` (needs CUDA) · `data` (needs Day-1 artifacts; auto-skips with a reason if missing) · `java` · `e2e` (browser/Playwright) · `app` (Streamlit `AppTest`).

### 2.1 Data — `tests/data/`
| Test | Asserts | Marker |
|---|---|---|
| `test_tiles.py::test_order_matches_smbjson` | ID→char equals the sorted `smb.json` keys (F6) | — |
| `test_tiles.py::test_char_id_roundtrip` | char↔ID maps are inverse bijections over 13 tiles | — |
| `test_tiles.py::test_pdf_order_is_rejected` | constructing the table in the PDF's order raises (guards doc 01 #2) | — |
| `test_audit_facts.py::test_counts` | 7,687 records, 7,620 unique, values 0..12 (F3–F5) | data |
| `test_audit_facts.py::test_chains_and_games` | 42 chains; 3,606 / 4,081 split; lengths 130–358; 8,317 columns (F7–F11) | data |
| `test_audit_facts.py::test_tile_frequencies` | F13 within ±0.01 pp | data |
| `test_audit_facts.py::test_caption_stats` | F14 values | data |
| `test_audit_facts.py::test_official_split_leak` | F17 statistics reproduce | data, slow |
| `test_splits.py::test_level_disjoint_and_complete` | train/dev/test disjoint; union = 0..41; 31/5/6 | data |
| `test_splits.py::test_game_stratification` | test 3+3; dev 2+3 or 3+2 | data |
| `test_splits.py::test_cv_partition` | 5 folds partition the 36 non-test levels, each fold has both games | data |
| `test_splits.py::test_concept_coverage` | ≥ 5 windows per concept per split, or a documented exception | data |
| `test_splits.py::test_families_not_split` | no near-duplicate family crosses train ↔ dev/test | data |
| `test_splits.py::test_leakage_operational` | P1.15 rules 1–2; `reused_segments.json` complete | data, slow |
| `test_splits.py::test_lock` | `sha256(split_v1.json) == LOCK` | data |
| `test_splits.py::test_prompts_lock` | PROMPTS_LOCK matches the test-side prompt files | data |
| `test_dataset.py::test_onehot_contract` | values ∈ {−1, +1}; one +1 per cell; shape and dtype | data |
| `test_dataset.py::test_argmax_roundtrip` | identity on all scenes | data |
| `test_dataset.py::test_gpu_resident_uint8` | storage dtype uint8; on CUDA when available | data, gpu |
| `test_dataset.py::test_sampler_deterministic` | the same seed gives the same batch sequence | data |
| `test_dataset.py::test_sampler_resume` | state saved at step k and restored gives the same batches k+1… as an uninterrupted run | data |
| `test_dataset.py::test_phrase_shuffle_preserves_set` | the shuffled caption has the same phrase multiset | — |
| `test_dataset.py::test_phrase_dropout_rate` | per-phrase drop rate 0.10 ± 0.01 over 10k draws; never 0 phrases | — |
| `test_dataset.py::test_paraphrase_train_lexicon_only` | substitutions only ever use train-half variants (D8c) | — |
| `test_dataset.py::test_null_dropout_rate` | null caption rate 0.10 ± 0.01 over 10k draws, applied last | — |
| `test_dataset.py::test_split_isolation` | a train-mode dataset never yields dev or test level IDs | data |
| `test_flip.py::test_flip_caption_equivalence` | (X5) detector(flip(G)) == swap_asc_desc(caption) for ≥ 99.5 % of 2,000 windows | data, slow |

### 2.2 Captions & evaluation — `tests/captions/`, `tests/eval/`
| Test | Asserts | Marker |
|---|---|---|
| `test_detector.py::test_reproduces_stored_captions` | 100 % phrase-set equality (or documented exceptions) | data, slow |
| `test_detector.py::test_details_in_bounds` | per-phrase contributing coordinates lie inside the grid | — |
| `test_detector.py::test_quantity_buckets` | 1→one, 2→two, 3–4→a few, 5–9→several, ≥ 10→many | — |
| `test_detector.py::test_floor_phrases` | hand grids: full / one gap / a few gaps / giant gap with chunks / none; enemies in row 15 count as gaps | — |
| `test_detector.py::test_broken_pipe_cases` | hand grids: orphan `<`, `[` without `]`, valid upright, valid upside-down, edge-cropped caps (scene mode) | — |
| `test_detector.py::test_broken_cannon_cases` | `b` with no `B`/`b` above → broken; `b` under `b` → valid; `b` in row 0 → broken | — |
| `test_cscore.py::test_golden_vectors` | all 12 rows of doc 22 §3.1 exactly (abs error < 1e-9) | — |
| `test_cscore.py::test_identity_and_range` | C(a, a) = 1; C ∈ [−1, 1] | — |
| `test_cscore.py::test_quirk_floor_conflation` | `floor with one gap` vs `giant gap with one chunk of floor` → 1.0 (replicated quirk Q2) | — |
| `test_cscore.py::test_broken_topic_penalty` | adding one broken phrase lowers C by exactly 2/18 | — |
| `test_structure.py::test_floor_valid_rule` | hand grids + real pass rate ≥ 0.998 | data |
| `test_structure.py::test_real_broken_zero` | 0 broken pipes / cannons in the real corpus | data, slow |
| `test_structure.py::test_js_properties` | JS(p, p) = 0; symmetric; ≤ 1 (base 2); JS(all-sky ‖ corpus) = 0.0676 ± 0.002 | data |
| `test_novelty.py::test_matmul_equals_bruteforce` | the one-hot-matmul Hamming distance equals the brute-force count on random grids | — |
| `test_novelty.py::test_copy_threshold_inclusive` | nn = 4 counts as a copy; nn = 5 does not | — |
| `test_novelty.py::test_gpu_cpu_equal` | identical nn on CPU and CUDA | gpu |
| `test_stats.py::test_bootstrap_coverage` | ≥ 93 % of nominal 95 % CIs cover the truth over 200 simulations | slow |
| `test_stats.py::test_cluster_unit` | resampling draws **levels** (clusters; all of a level's prompts and seeds travel together), not prompts or samples; prompt-only sets are flagged as prompt-clustered | — |
| `test_stats.py::test_paired_sign_and_seed` | Δ sign is correct; results are deterministic with seed 12345 | — |
| `test_stats.py::test_holm` | known textbook example | — |

### 2.3 Diffusion core — `tests/diffusion/`
| Test | Asserts | Marker |
|---|---|---|
| `test_scheduler.py::test_beta_ranges` | P2.01 | — |
| `test_scheduler.py::test_alphabar_monotone_and_endpoints` | P2.02 | — |
| `test_scheduler.py::test_q_sample_moments` | P2.03 (1M-sample tolerance) | slow |
| `test_scheduler.py::test_predict_x0_inverse` | P2.05 | — |
| `test_scheduler.py::test_oracle_ddim_exact` | P2.06, both schedules | — |
| `test_scheduler.py::test_posterior_closed_form` | P2.07 | — |
| `test_scheduler.py::test_ddim_eta1_matches_ddpm_variance` | P2.08 | — |
| `test_scheduler.py::test_timesteps` | P2.09 | — |
| `test_scheduler.py::test_renoise_moments` | P2.10 | slow |
| `test_scheduler.py::test_dtype_device` | fp64 buffers; outputs cast to the input dtype and device | — |
| `test_cfg.py::test_guidance_convention` | w = 1 returns ε_c exactly; w = 0 returns ε_u; linear in w | — |
| `test_cfg.py::test_x0_clipping` | x̂₀ is clipped to [−1, 1] before the DDIM update when enabled | — |

### 2.4 Model & text — `tests/models/`, `tests/text/`
| Test | Asserts | Marker |
|---|---|---|
| `test_unet.py::test_param_count` | 18,312,013 for the default conditional config (H8); the documented count for the unconditional variant | — |
| `test_unet.py::test_matches_reference` | with weights copied from `reference/ref_unet.py` (name map), outputs equal within 1e-5 (fp32, eval) | — |
| `test_unet.py::test_shapes_variable_width` | W ∈ {16, 24, 112} → (B, 13, 16, W) | — |
| `test_unet.py::test_width_not_multiple_of_4_raises` | a clear error for W % 4 ≠ 0 | — |
| `test_unet.py::test_all_params_get_grad` | non-zero grad norm for every parameter after one backward pass | — |
| `test_unet.py::test_time_sensitivity` | outputs differ for t = 10 vs t = 900 | — |
| `test_unet.py::test_context_sensitivity_and_mask` | outputs change with the context; changing **masked** tokens doesn't change outputs | — |
| `test_unet.py::test_eval_deterministic` | dropout off in eval mode → identical outputs | — |
| `test_unet.py::test_bf16_forward_finite` | autocast bf16 forward has no NaN/Inf | gpu |
| `test_encoder.py::test_minilm_frozen` | every MiniLM param has `requires_grad = False`; the module is in eval mode | — |
| `test_encoder.py::test_optimizer_membership` | adapter params are in the optimiser; MiniLM params are not (P4.03) | — |
| `test_encoder.py::test_null_embedding` | null = encoding of "" ([CLS][SEP]) → adapter; shape (1, 2, 128) | — |
| `test_encoder.py::test_cache_and_truncation` | a cache hit returns the identical tensor; max_len 48; padding mask correct | — |
| `test_normalizer.py::test_synonyms` | the doc-12 table (e.g., Goombas → enemies, tubes → pipes, pits → gaps) | — |
| `test_normalizer.py::test_numbers_to_buckets` | digits and number words → one / two / a few / several / many; "no X" handled per doc 12 | — |
| `test_normalizer.py::test_scene_split` | multi-scene splitting on "then", ";", "followed by", "after that" | — |
| `test_normalizer.py::test_idempotent_and_canonical_passthrough` | normalize(normalize(x)) == normalize(x); canonical captions unchanged | — |
| `test_normalizer.py::test_p_casual_dev_accuracy` | `norm_exact` ≥ the doc-12 target on P-casual-dev | data |

### 2.5 Training — `tests/train/`
| Test | Asserts | Marker |
|---|---|---|
| `test_train.py::test_overfit_one_batch` | 8 scenes, fixed t, 1,500 steps → loss < 0.01 | slow, gpu |
| `test_train.py::test_param_groups_no_decay` | norms, biases and embeddings have wd = 0; others 0.01 | — |
| `test_train.py::test_lr_schedule` | warm-up and cosine values at steps 0, 500, 1,000, mid, end (1e-5 floor) | — |
| `test_train.py::test_ema_update_and_warmup` | decay = min(0.999, (1+n)/(10+n)); EMA tensors on the right device | — |
| `test_train.py::test_grad_clip_applied` | total norm ≤ 1.0 after clipping | — |
| `test_train.py::test_no_gradscaler_bf16` | no GradScaler with bf16 autocast | — |
| `test_train.py::test_atomic_checkpoint` | a simulated crash mid-save leaves the previous checkpoint intact; no partial file loads | — |
| `test_train.py::test_resume_equivalence` | 50 + 50 steps resumed == 100 straight (CPU, fp32, deterministic mode, tolerance 1e-6) | slow |
| `test_train.py::test_machine_yaml_guard` | a machine YAML overriding any non-performance key → error (CLAUDE.md §2.5) | — |
| `test_train.py::test_vram_cap_applied` | the per-process memory fraction equals 4.5 GiB / total | gpu |
| `test_train.py::test_val_loss_deterministic` | the per-bucket validation loss is identical on two calls (fixed t and noise) | data |
| `test_train.py::test_temperature_watchdog` | a stubbed NVML reading > 87 °C for 60 s triggers checkpoint + stop (H9) | — |

### 2.6 Generation, playability, render, app — `tests/generate/`, `tests/playability/`, `tests/render/`, `tests/app/`
| Test | Asserts | Marker |
|---|---|---|
| `test_full_level.py::test_width_and_framing` | output width = 3 + W_gen + 6; framing columns equal the templates (G3) | — |
| `test_full_level.py::test_window_count` | windows = (W_gen − 16)/8 + 1 at stride 8 | — |
| `test_full_level.py::test_known_region_locked` | after the final step, overlap columns equal the previously generated tiles | gpu |
| `test_full_level.py::test_seed_determinism` | the same seed gives an identical level (same machine) | gpu |
| `test_full_level.py::test_resample_limits` | ≤ 5 attempts per window, ≤ 20 per level; badge state correct | — |
| `test_full_level.py::test_fast_mode_params` | CPU fast mode = 25 steps, r = 0 (H10) | — |
| `test_full_level.py::test_sanitizer_detect_vs_repair` | detection counts are logged; repair yields 0 broken; metrics use pre-repair grids | — |
| `test_checker.py::test_micro_levels` | flat floor → solved; gap > max jump → unsolved; gap ≤ max jump → solved; wall > max jump height → unsolved; climbable staircase → solved; pipe ≤ jump height → solved | — |
| `test_checker.py::test_physics_single_source` | the checker loads `app/static/game/physics.json` (the same file as the game) | — |
| `test_checker.py::test_real_levels_band` | real windows' solvable rate lies in the band recorded in REFERENCES.json; all real full levels solvable or listed | data, slow |
| `test_checker.py::test_jar_bridge_format` | `grid_to_vglc_text` → jar-readable text (row count, chars) round-trips | — |
| `test_checker.py::test_jar_runs` | the jar solves a trivial flat level | java |
| `test_renderer.py::test_image_size_and_orientation` | image = (16·tile_px, W·tile_px); row 0 at the top; block equals the sprite | — |
| `test_renderer.py::test_vglc_json_roundtrip` | grid → VGLC text / JSON (doc 29 schema) → grid is the identity | — |
| `test_renderer.py::test_placeholder_skin` | works with no downloaded assets (A2) | — |
| `test_app.py::test_app_loads_and_generates` | `AppTest` with a stub model: prompt → image + level JSON + badge | app |
| `test_app.py::test_exports_valid` | PNG/JSON/CSV/VGLC/standalone HTML are produced and valid | app |
| `test_app.py::test_html_offline` | no external URLs in the generated HTML; size < 8 MB | app |
| `test_e2e.py::test_play_in_chrome_and_edge` | P7.04 + P7.05 via Playwright | e2e |
| `test_repo_hygiene.py::test_no_imports_from_data_raw` | no module under `src/`, `scripts/` or `app/` imports from `data/raw` (CLAUDE.md rule 10) | — |
| `test_repo_hygiene.py::test_no_nintendo_assets_committed` | asset manifest licences ∈ {CC0, MIT}; no `mapsheet.png` tracked (A3) | — |

### 2.7 Gates — `tests/gates/`
| Test | Asserts | Marker |
|---|---|---|
| `test_result_schema.py::test_all_results_valid` | every RESULT.json validates against the schema | — |
| `test_prereq.py::test_prerequisite_enforcement` | a gate exits 2 when a prerequisite is missing or failed | — |
| `test_thresholds.py::test_gates_yaml_matches_ledger` | `configs/gates.yaml` values equal the ledger §9 numbers (a parsed table in the test) | — |

---

## 3. Commands (local "CI")

```bash
# fast suite (< 60 s) — required before every commit (CLAUDE.md §3)
pytest -q -m "not slow and not gpu and not e2e and not java and not app"
# everything this machine can run
pytest -q -m "not e2e" && pytest -q -m e2e        # e2e needs: pip install playwright && playwright install chrome msedge
# one-shot local CI
python scripts/check_all.py            # compileall + ruff (if installed) + validate_configs + fast tests + lock check + gate_status
python scripts/check_all.py --full     # + slow, + gpu (if CUDA), + java (if available), + app, + e2e (if Playwright)
# install the git pre-commit hook (fast suite + lock check)
python scripts/install_hooks.py
```
`pyproject.toml`:
```toml
[tool.pytest.ini_options]
markers = ["slow: >5s", "gpu: needs CUDA", "data: needs Day-1 artifacts", "java: needs a JRE",
           "e2e: browser via Playwright", "app: Streamlit AppTest"]
addopts = "-ra"
```
`scripts/validate_configs.py`: every YAML parses, has the required keys, has no machine-YAML overrides of non-performance keys, and has thresholds equal to ledger §9 (shares its logic with `test_thresholds.py`).

---

## 4. Property-based tests (hypothesis)

| Property | Strategy | Where |
|---|---|---|
| `predict_x0(q_sample(x₀, t, ε), t, ε) == x₀` | random one-hot grids, t ∈ [0, 999], ε ~ N(0, I) | scheduler |
| oracle-ε DDIM reconstructs x₀ exactly | random schedules (linear endpoints in [1e-5, 1e-3] × [0.01, 0.05]; cosine s ∈ [0.001, 0.02]), n_steps ∈ {10, 25, 50, 100} | scheduler |
| ᾱ strictly decreasing, β ∈ (0, 0.999] | the same schedule strategy | scheduler |
| `argmax(onehot(G)) == G` | random grids, W ∈ {16, 24, 112} | data |
| renderer round trip (grid → VGLC → grid, grid → JSON → grid) | random grids | render |
| C(a, b) = C(b, a); C(a, a) = 1; C ∈ [−1, 1] | random canonical captions from the 62-phrase grammar (1–10 phrases) | eval |
| matmul Hamming == brute force | random grid batches | eval |
| normalizer idempotent; canonical passthrough | random canonical captions + random casual templates from the train lexicon | text |
| sampler resume equivalence | random stop step k ∈ [1, 500] | data |
| sanitizer repair → 0 broken; repair never touches non-pipe/cannon tiles | random grids with injected broken pipes/cannons | generate |
| floor_valid invariant to changes in rows 0–1 | random grids | eval |

Run with `--hypothesis-seed=0` in CI for reproducibility; add `@settings(max_examples=200, deadline=None)` to slow properties.

---

## 5. What "done" looks like for this doc
- [ ] `src/text2mario/gates/` (result, checks, prereq) and `configs/gates.yaml` (ledger-mirrored thresholds).
- [ ] Eight gate scripts with the checks above, each writing RESULT.json and artifacts.
- [ ] The test files above, the markers registered, the fast suite under 60 s, and `check_all.py` green.
- [ ] `gate_status.py` shows P1 → P7 → P6 all passing at the release tag.
