# 02 — Scope and Requirements

> **Authority:** `00_DECISIONS_LEDGER.md` (cited as S*, H*, D*, M*, G*, E*, A*, G-P*). This document turns the ledger's scope decisions into numbered, testable requirements. Every requirement has an ID, a priority (**MUST** = MVP blocker, **SHOULD** = expected but cuttable per `04_CRITICAL_PATH_AND_DEPENDENCIES.md` §7, **MAY** = only if time allows), a verification method, and a gate or acceptance test.
>
> Requirement IDs are stable. Never renumber; deprecate with ~~strikethrough~~ and a change-log row (§12).

---

## 1. Product statement

**text2mario** is a Streamlit application (local-GPU for development and the live demo, CPU profile on Streamlit Community Cloud for the mandatory deployment, ledger S6) plus a training pipeline. A user types an English sentence describing a Super Mario Bros.-style level (for example *"lots of goombas, three pipes and a big gap near the end"*). A from-scratch, text-conditioned discrete-tile diffusion model trained on SMB1 and SMB2 (Lost Levels) scenes generates a full-width level. The level is checked for beatability by a physics-consistent planner and opens **as a real, playable game in the browser** inside a Streamlit studio (ledger S1, S2, S5, S6).

The scientific deliverable is equally important. The reported metrics must be **honest**: a level-grouped split (D3), a one-shot locked test set (D4), playability measured rather than assumed (E1), and novelty measured against the training set rather than inferred from diversity (E1).

## 2. Problem definition, users and stakeholders (rubric row 1, ledger §2b P1-P4)

### 2.1 The problem
2D platformer levels are tile grids, and a designer's first draft is slow to build by hand in a tile editor. Existing procedural generators are either random (hard to steer) or learned without any control from words, and almost none **check that the result can be finished**. The problem this project addresses: **give an indie/hobbyist level designer a fast, text-controllable first draft of a platformer level that has already been checked for beatability, and that can be played immediately.**

### 2.2 Who has the problem (identified users and the practical need)
| Persona | Who | Practical need | What the app gives them |
|---|---|---|---|
| **U1 Indie / hobbyist game developer or level designer** | Someone building a Mario-like platformer in a tile editor, who needs several candidate layouts to react to ("a short level with three pipes and a gap near the end") | Many quick, steerable draft levels; knowing early whether a draft is finishable; an export to a tile format they can edit | Typed sentence -> full level in seconds, canonical caption shown, "Verified beatable" badge, PNG/JSON/CSV/VGLC exports, playable HTML |
| **U2 Educator / student of procedural content generation** | A teacher or student who wants a small, runnable example of text-conditioned generation plus a verifier | An end-to-end, inspectable pipeline with honest evaluation | The report, the open pipeline, the live app with "About this model" panel |
| **U3 Player / demo user** | Anyone given the Streamlit link | Type a sentence, play the level | Same app, game embedded in the page |
| **E1 Evaluator (not a target user)** | Course instructor, viva panel | Defensible evaluation and a working deployed app | Split protocol, CIs, one-shot test, cloud URL, submission checklist (doc 35) |
| **T1 Developer-operator** | The student team, working through Claude Code | Build and train without guessing; every member able to explain and modify the work | Specs, gates, fallbacks, ownership and viva drills (doc 36) |
| **T2 Future extender** | Whoever adds the next 9 features | Clean seams | Interfaces and data fields (doc 31) |

**Honesty note.** No interviews or surveys were done. U1 and U2 are *stated, plausible needs* taken from the project's purpose; the project does not claim measured demand, user counts or time savings. The only user-facing claims the report may make are the measured ones (latency, playability agreement, adherence, novelty).

### 2.3 Input, output, objectives
- **Input:** one sentence (1-300 characters, free English), level width in {16, 48, 80, 112} generated columns, guidance scale w, seed.
- **Output:** a framed level (16 rows x (3 + width + 6) columns of 13 tile types), the canonical caption(s) the model received, a beatability badge with the resampling attempts used, a PNG render, a playable Phaser game, and exports (doc 29).
- **Objectives (O-1..O-6):** O-1 produce levels whose content follows the prompt (adherence, C-score); O-2 keep structures valid (pipes/cannons not broken); O-3 verify beatability and report the checker's validity; O-4 avoid copying training windows (novelty/copy rate against real references); O-5 evaluate honestly on **unseen levels** (level-grouped split, one-shot test); O-6 deliver a working Streamlit app, **deployed on Streamlit Community Cloud**.
- **Measurable success criteria:** the gates G-P1..G-P8 of the ledger §9 (with E1 metrics, level-clustered CIs and per-level tables). Numeric targets that are already pre-registered: G-P3 broken rate <= 5 %, G-P4 dev C-score >= max(0.75, null + 0.20), G-P5 seam ratio <= 1.3, G-P5b checker agreement >= 85 % / 90 %. Latency targets are **provisional** until measured (G-P7, G-P8).
- **Expected impact (stated modestly):** a working, honest prototype of controllable, verified level drafting for one game family (SMB1/SMB2 tiles). It is not a production level-design tool.

### 2.4 Why a neural network (algorithm justification, one paragraph for the report)
Rules and templates can place pipes and gaps but cannot learn the joint structure of real level segments (floors, platforms, enemies and pipes that co-occur) from 42 levels, and a text-conditioned model lets one sentence steer that structure. A discrete-tile diffusion model is chosen because it generates a whole 16x16 window by iterative refinement (so global constraints such as a pipe's top and body appear together), supports classifier-free guidance for controllable adherence, and supports overlap-consistent outpainting (RePaint) for long levels. A plain classifier or regressor would not produce a level; an autoregressive tile model is the main alternative (MarioGPT), and is discussed in the related work as a baseline we did not train. Its output feeds a concrete decision: the designer keeps, edits or rejects a draft, helped by the beatability badge.

## 3. Scope summary

| Category | Items | Ledger |
|---|---|---|
| **In MVP (MUST ship)** | "Type it, play it": prompt → normalised canonical caption(s) → full-width level (16/48/80/112 generated columns + templated framing) → playability check with bounded resampling → "verified beatable" badge → playable Phaser game in the Streamlit page → exports | S2, G1–G5, V4, V5 |
| **In pipeline (MUST)** | Data audit, level-grouped splits, unconditional baseline, conditional model with CFG, full-level generator, playability checker, evaluation suite with CIs, CV, one-shot test, report figures | S4, D1–D9, M1–M13, E1–E4 |
| **Also MUST (delivery)** | Report with every statement-mandated section, 12-slide deck, evidence-of-evaluation folder, Streamlit Community Cloud deployment (G-P8), single submission zip for GCR (doc 35) | S6, S7 |
| **Seams only (no MVP UI)** | Ghost-bot repair, difficulty dial, erase-and-re-ask, whole-level pacing, word highlighting, report card, blind test, overnight self-training, endless mode | S3, doc 31 |
| **Out of scope** | See §9 Non-goals | — |

## 4. User stories

Each story has acceptance criteria (AC) that map to the acceptance tests in §7.

| ID | Story | Acceptance criteria |
|---|---|---|
| **US-01** | As a player, I type a sentence and press Generate so that I get a level that matches my words. | AC1: a level appears for any prompt of 1–300 characters. AC2: the canonical caption the model actually received is shown. AC3: unmapped words are listed ("ignored: …"). |
| **US-02** | As a player, I press Play and control the character with the keyboard in the same page. | AC1: arrow keys/A-D move, Space/W/Up jump, R restarts. AC2: the game captures focus after one click on a "Click to play" overlay. AC3: falling into a pit or touching an enemy from the side ends the attempt; reaching the flag shows a win screen. |
| **US-03** | As a player, I want to know whether the level can be finished before I try it. | AC1: the badge shows "Verified beatable ✓" or "Not verified". AC2: the number of resampling attempts used is shown (G5). |
| **US-04** | As a player, I choose how long the level is. | AC1: width options 16 / 48 / 80 / 112 generated columns (G2). AC2: the framed level adds 3 spawn columns and a 6-column end template (G3). |
| **US-05** | As a player, I can get the same level again or a different one. | AC1: a seed field and a "new seed" button. AC2: same prompt + seed + w + width + checkpoint on the same machine → identical level (NFR-REP-04). |
| **US-06** (MAY) | As a curious user, I watch the level emerge from noise. | AC1: a denoising animation with at least one frame per 5 DDIM steps (timeline D12). |
| **US-07** | As a user, I download what I made. | AC1: PNG, level JSON (schema v1, doc 29), CSV, VGLC `.txt`, and a single-file playable HTML. |
| **US-08** | As a demo presenter, the app works on my laptop with no internet and, if needed, no GPU, **and** a public Streamlit Community Cloud URL exists as a backup (US-13). | AC1: offline cold start works (NFR-OFF-01). AC2: CPU fast mode generates a 112-wide level in < 45 s (G-P7, H10). AC3: 12 cached showcase levels load instantly (timeline D16). |
| **US-09** | As an examiner, I see what model produced a level. | AC1: an "About this model" panel shows the checkpoint sha256 (first 12 chars), config hash, training steps, the frozen dev metrics, and the test metrics once they exist. |
| **US-10** | As an examiner, I trust the reported numbers. | AC1: the test split is locked (D4) and evaluated once (G-P6). AC2: every figure regenerates from `scripts/make_figures.py`. AC3: CIs and CV spreads are reported (E3). |
| **US-11** (SHOULD) | As an operator, I move a training run from the desktop to the laptop and back without losing progress. | AC1: the handoff protocol (doc 07) resumes from the last checkpoint. AC2: resume verification passes (doc 07 §8). |
| **US-12** (MAY) | As a future extender, I add a feature without rewriting the generator. | AC1: the seam interfaces in §10 exist, are typed, and are unit-tested (doc 31). |

| **US-13** | As a user with only a browser, I open the public Streamlit Cloud URL and generate and play a level. | AC1: the URL loads without installing anything. AC2: a prompt yields a level, or a clearly labelled cached showcase level if live generation exceeds the time budget (CL6). AC3: the game is playable by keyboard. |
| **US-14** | As a group member, I can explain and modify any part of the system live. | AC1: every member has typed and run every drill in doc 36 §5 at least once before 3 Dec. AC2: each member can walk through the model, diffusion code, preprocessing and app (doc 36 §3). |

## 5. Functional requirements

### 5.1 Application ("Type it, play it")

| ID | Pri. | Requirement | Verification |
|---|---|---|---|
| **FR-A-01** | MUST | The studio page provides a free-text prompt box, limited to **300 characters** (enforced in the UI and again server-side). | AT-12 |
| **FR-A-02** | MUST | Prompts pass through the rule-based normaliser (doc 12): synonyms → canonical nouns, numbers and number words → quantity buckets (F15: one, two, a few, several, many), multi-scene splitting. The canonical caption(s) are displayed. Unmapped tokens are listed and never cause an error. | AT-12, `tests/test_normalizer.py` |
| **FR-A-03** | MUST | Controls: guidance scale *w* (slider, range 1–10, default = the dev-selected value recorded in the ledger on Day 9, M13); width ∈ {16, 48, 80, 112} (G2); integer seed in [0, 2³¹−1] plus a "new seed" button; a "fast mode" toggle that is **forced on when CUDA is unavailable** (H10). | AT-12 |
| **FR-A-04** | MUST | Generation uses the full-level generator (doc 20): 16×16 windows, stride 8, RePaint known-region replacement + r = 2 resampling for the first 60 % of steps (G1); DDIM 50 steps, η = 0, x̂₀ clipping, CFG convention `ε_u + w(ε_c − ε_u)` (M13). Fast mode: DDIM 25 steps, r = 0 (H10). | AT-13, AT-14 |
| **FR-A-05** | MUST | Deterministic framing (G3): prepend 3 full-floor spawn columns; append the 6-column end template (floor + staircase + flag). Framing columns are flagged as `templated` in the level JSON (doc 29). | `tests/test_full_level.py` |
| **FR-A-06** | MUST | The multi-tile sanitizer **detects** broken pipes and cannons on every level (G4). Optional display-time repair is a UI toggle. The level JSON stores both `raw_grid` and `display_grid`; metrics and the report-card seam use `raw_grid`. | `tests/test_sanitizer.py` |
| **FR-A-07** | MUST | Verification (G5): the fast playability checker (doc 21) runs on the framed level. On failure, the failing window is re-sampled with a new derived seed, at most **5 attempts per window and 20 in total**. Result: badge "Verified beatable ✓" or "Not verified", plus the attempt count. | AT-13 |
| **FR-A-08** | MUST | The final level is rendered as a PNG using the active sprite skin (A1; placeholder skin A2 if assets are missing, logged per CLAUDE.md §2.12). | AT-12 |
| **FR-A-09** | MUST | **Play in browser**: the Phaser 3.90.0 game (V5, doc 27) is embedded in the Streamlit page through the embed shim of doc 28 (`st.iframe` if the pinned Streamlit has it, else `st.components.v1.html`; ledger V4), with the level JSON and sprite atlas inlined (no network). It uses the same `physics.json` as the checker, has a click-to-focus overlay and keyboard controls (US-02), a HUD with time and coins, restart on death, and a win screen at the flag. | AT-15, AT-16 |
| **FR-A-10** | MAY (demoted, §14) | Denoising animation: intermediate argmax(x̂₀) frames at least every 5 DDIM steps, shown as an image sequence or slider after generation completes. | AT-12 (presence check) |
| **FR-A-11** | MUST | Exports: PNG, level JSON (schema v1), CSV (one row per tile row), VGLC `.txt` (characters per F6), and a single-file playable HTML (doc 27 §standalone). | AT-17 |
| **FR-A-12** | MUST | "About this model" panel (US-09) reads `FROZEN.json` and the checkpoint sidecar JSON. | AT-12 |
| **FR-A-13** | **MUST** (promoted: it is the cloud fallback, CL6) | Showcase mode: 12 pre-generated levels (cached JSON + PNG, human-played to the flag) are selectable without running the model and are the automatic fallback when live generation exceeds the cloud time budget (timeline D16). | AT-18, AT-21 |
| **FR-A-14** | MUST | Errors never crash the app. Empty prompt → message; generator exception → message plus a log entry, and the last good level stays visible; checker timeout → "Not verified" with reason. | AT-19 (fuzz) |
| **FR-A-16** | MUST | **Cloud profile** (ledger §3b): `T2M_PROFILE=cloud` selects CPU fast mode, loads the bundled `.safetensors` checkpoint and bundled MiniLM (no downloads), applies the time guard and the showcase fallback, and shows which profile is active. | AT-21 |
| **FR-A-17** | MUST | **Deployment:** the app is deployed on Streamlit Community Cloud from the GitHub repo, with a pinned `requirements.txt` and the frozen checkpoint; the public URL is recorded in `evidence/deployment/` and on the report's first page (G-P8). | AT-21 |
| **FR-A-15** | MUST | Each generated level records full provenance in its JSON: prompt, canonical caption(s), seed, w, width, steps, fast_mode, checkpoint sha256, config hash, git SHA, machine, timestamps, verification result and attempt log (doc 29). | `tests/test_level_schema.py` |

### 5.2 Pipeline (data → model → evaluation)

| ID | Pri. | Requirement | Verification / Gate |
|---|---|---|---|
| **FR-P-01** | MUST | `scripts/fetch_data.py` acquires the dataset from a pinned MarioDiffusion commit and writes `data/MANIFEST.json` with the commit SHA and sha256 values (D1). | G-P1 |
| **FR-P-02** | MUST | `scripts/audit_dataset.py` asserts ledger facts F1–F17 and reconstructs level IDs by chain detection (D2, F10). | G-P1 |
| **FR-P-03** | MUST | `scripts/make_splits.py` produces the level-grouped split (31/5/6 levels) and 5 grouped CV folds over the 36 non-test levels (D3, D5); `data/splits/LOCK` holds the sha256 of the split (D4). | G-P1, `tests/test_splits.py` |
| **FR-P-04** | MUST | The caption detector (`src/text2mario/captions/detector.py`) is **implemented from the caption-grammar description (doc 12), not ported from MarioDiffusion's code (ledger O3)**, and is then **checked against** the stored captions: it reproduces the stored phrase set for 100 % of the 7,687 scenes, or ≥ 99.5 % with documented exceptions (timeline D1). | `tests/test_detector.py` |
| **FR-P-05** | MUST | The dataset is GPU-resident uint8 with an on-the-fly {−1, +1} one-hot (D7, H7) and the D8 augmentations; the sampler state is resumable. | G-P1, `tests/test_dataset.py` |
| **FR-P-06** | MUST | The scheduler implements linear and cosine schedules, q_sample, posterior, DDPM step, DDIM step and renoise (M1, M2). | G-P2 |
| **FR-P-07** | MUST | The U-Net matches doc 14 / `reference/ref_unet.py`: **18,312,013** parameters for the default conditional configuration (H8, M3–M6). | `tests/test_unet.py` |
| **FR-P-08** | MUST | The trainer: bf16 autocast (H5), AdamW groups (M10), EMA (M11), budgets in steps (M12), deterministic per-t-bucket validation loss, atomic checkpoints, full-state resume, VRAM cap (H3), thermal watchdog (H9). | Day-4 drill, `tests/test_checkpoint.py` |
| **FR-P-09** | MUST | Unconditional baseline passes **G-P3**; the conditional model passes **G-P4**. | G-P3, G-P4 |
| **FR-P-10** | MUST | The full-level generator passes **G-P5**; the playability checker passes **G-P5b**. | G-P5, G-P5b |
| **FR-P-11** | MUST | The evaluation suite computes all E1 metrics on the E2 prompt sets with E3 statistics; the dev composite score E4 drives checkpoint and CFG selection. | G-P6 |
| **FR-P-12** | MUST | `scripts/eval_test.py` refuses to run unless `data/splits/LOCK` and `FROZEN.json` hashes verify, and it records each invocation in `reports/final/test_invocations.jsonl` (D4). | G-P6 |
| **FR-P-13** | MAY (demoted, §14) | 5-fold level-grouped CV of the frozen config is reported as mean ± std (D5). | Day 15 |
| **FR-P-14** | MUST | `scripts/make_figures.py` regenerates every report figure from saved artifacts (CLAUDE.md §3). | Day 16 |

### 5.3 Delivery requirements (statement "Mandatory Deliverables", rubric row 10 and deduction rules)

| ID | Pri. | Requirement | Produced by | Day |
|---|---|---|---|---|
| **FR-D-01** | MUST | **Source code**: complete, organised, runnable, commented, meaningful names, clear structure; every module has an explain-me note (CLAUDE.md §6). | repo | continuous; checked D17 |
| **FR-D-02** | MUST | **Report** with every statement-mandated section: problem statement, real-world justification, selected path (Path A), objectives, methodology, algorithm description, system architecture/workflow (a proper pipeline diagram), dataset description, implementation details, testing, results, limitations, conclusion (+ originality and attribution). | doc 32 §1 (section map), prose tasks on dated days (doc 03) | D0 first draft sections; D16 complete; D17 PDF |
| **FR-D-03** | MUST | **Demonstration**: live Streamlit app on each member's laptop (rubric: every member brings a laptop with the project running, otherwise -20 %), plus the cloud URL. | doc 36 §6, doc 32 §6 | D17 rehearsal |
| **FR-D-04** | MUST | **Presentation slides (12)** covering problem, motivation, proposed solution, selected AI/ML technique, implementation details, testing details, results (qualitative and quantitative), limitations, conclusion. | doc 32 §9 slide map | draft D14, final D16 |
| **FR-D-05** | MUST | **Evidence of evaluation** folder: metrics tables, CIs, per-level tables, figures, gate RESULT.json files, test log, checker-validation study. | `scripts/collect_evidence.py` | D16 |
| **FR-D-06** | MUST | **One submission zip** on GCR by one member; zip built and checked on a clean directory (unzip, install, tests, app start). Submission-guidelines compliance (-2 if not). | doc 35 §4 | D17 build, D18 upload |

## 6. Non-functional requirements

### 6.1 Performance and latency
| ID | Pri. | Requirement | Measurement |
|---|---|---|---|
| **NFR-PERF-01** | MUST | **Cold start < 20 s on the laptop (provisional, pinned after Day 12 measurement; ledger G-P7)**: from `streamlit run` process start to the "Ready" status, with model and encoder loaded and the warm-up generation done. | `scripts/gate_p7.py` parses the `app_start` and `ready` timestamps in the app log |
| **NFR-PERF-02** | MUST | **Generate + verify, width 112, full quality: median < 10 s (provisional, never measured; pinned after Day 12)** on the laptop GPU over the 20-prompt AT set (G-P7). The 95th percentile is reported. The worst case is bounded by the G5 resampling budget. | `scripts/gate_p7.py` |
| **NFR-PERF-03** | MUST | **CPU fast mode, width 112: median < 45 s** (G-P7, H10), with `CUDA_VISIBLE_DEVICES=""`. | `scripts/gate_p7.py --cpu` |
| **NFR-PERF-04** | MAY (demoted, §14) | Game frame rate: median ≥ 55 FPS, 5th percentile ≥ 45 FPS in Chrome on the laptop at 1920×1080 for a 121-column level. | Playwright FPS probe (`window.__t2m_fps`) |
| **NFR-PERF-05** | SHOULD | The checker verifies a 121-column framed level in < 1.0 s on the laptop CPU (median). | `tests/test_checker.py::test_speed` (slow) |

| **NFR-PERF-06** | MUST | **Cloud profile**: median generation time (width 48, fast mode) and peak resident memory are **measured on Streamlit Cloud on Day 12** and pinned in the ledger; until then the provisional caps of ledger CL5/CL6 apply (RSS <= 1 GB, time budget 60 s then showcase fallback). | `scripts/gate_p8.py` + manual cloud run |

### 6.2 Resources
| ID | Pri. | Requirement |
|---|---|---|
| **NFR-RES-01** | MUST | Every training, evaluation and app process applies the **4.5 GiB VRAM cap** at start-up and stays within it (H2, H3). Peak allocated memory is recorded in `hw_report.json`. |
| **NFR-RES-02** | MUST | Hyperparameters are identical on both machines; only allow-listed performance keys differ (H4, CLAUDE.md §2.5, doc 07 §3). |
| **NFR-RES-03** | SHOULD | Host RAM: ≤ 6 GB per training/eval process, ≤ 3 GB for the app process. |
| **NFR-RES-04** | MAY | Disk: ≤ 15 GB under `runs/` per machine with the retention policy of doc 07 §6; ≥ 20 GB free before starting Day 5 (checked by `verify_env.py --full`). |

### 6.3 Offline operation and portability
| ID | Pri. | Requirement |
|---|---|---|
| **NFR-OFF-01** | MUST | **No network is needed at run time in either profile.** Local profile: MiniLM is loaded from the local HF cache with `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` (doc 06 §8). Cloud profile: MiniLM and the checkpoint are bundled in the repo (ledger CL3, CL4) and loaded from local paths. In both, Phaser and sprites are vendored and no CDN URLs appear in served HTML. (Deployment itself, GitHub to Streamlit Cloud, naturally needs the network; the *running app* does not.) |
| **NFR-OFF-02** | MUST | Supported OS: Windows 11 (laptop, likely desktop) and Ubuntu 22.04/24.04 (desktop option). Python 3.12 (V1). Same torch minor version on both machines (V2). |
| **NFR-OFF-03** | MUST | Supported browsers: Chrome and Edge (current stable) for play; Firefox best-effort (G-P7). |
| **NFR-OFF-04** | MUST | All paths are handled with `pathlib`; all text I/O uses `encoding="utf-8"`; no shell-specific code in `src/`. |

### 6.4 Reproducibility
| ID | Pri. | Requirement |
|---|---|---|
| **NFR-REP-01** | MUST | Every run directory contains the resolved config, config hash, git SHA (+ dirty flag), data-manifest hash, split LOCK hash, env lock reference, seeds, and lineage (doc 07 §5, doc 30). |
| **NFR-REP-02** | MUST | Data artifacts are derived deterministically and identified by **content hashes** (canonical arrays, not file bytes), so both machines agree (doc 07 §11e). |
| **NFR-REP-03** | MUST | Re-running an evaluation on the same machine with the same checkpoint, seeds and `deterministic_eval: true` reproduces every metric to 4 decimal places. |
| **NFR-REP-04** | MUST | App generation is deterministic per (prompt, seed, w, width, steps, fast_mode, checkpoint, machine). Initial noise is drawn from a CPU `torch.Generator` seeded from these inputs, so it is machine-independent. Cross-machine tile agreement is measured and reported but not guaranteed bitwise (doc 07 §9). |
| **NFR-REP-05** | MUST | No claim of bitwise cross-GPU reproducibility is made anywhere; results are reported with seeds and CIs (E3). |

### 6.5 Scientific validity
| ID | Pri. | Requirement |
|---|---|---|
| **NFR-SCI-01** | MUST | Splits are by level only (D3). The repo's own leaky splits are never used (F16, F17, CLAUDE.md §2.1). |
| **NFR-SCI-02** | MUST | The test set is evaluated once, after the freeze (D4, G-P6). All tuning uses dev (CLAUDE.md §2.2). |
| **NFR-SCI-03** | MUST | Metrics are computed on raw samples, before sanitizer repair (G4, CLAUDE.md §2.8). |
| **NFR-SCI-04** | MUST | Novelty is measured as NN distance to the train split and calibrated against real dev windows (E1, E4). Diversity is never presented as novelty. |

### 6.6 Licensing and IP
| ID | Pri. | Requirement |
|---|---|---|
| **NFR-LIC-01** | MUST | Only CC0/MIT/Apache/BSD-licensed third-party files are committed: Kenney Pixel Platformer (CC0, A1), Phaser 3.90.0 (MIT, A4). Licence texts are stored next to each asset (`app/static/vendor/LICENSE-phaser.txt`, `assets/LICENSES/`). |
| **NFR-LIC-02** | MUST | Nintendo-derived sprites (e.g., the repo's `mapsheet.png`) are never committed or shipped (A3). An optional local "classic skin" is git-ignored and off by default. |
| **NFR-LIC-03** | MUST | MarioDiffusion (MIT) is credited in `DATA_CARD.md` and in every module that re-implements its logic (`detector.py`). Level layouts originate from Nintendo games via the VGLC and are used for non-commercial academic research; `DATA_CARD.md` documents provenance. |
| **NFR-LIC-04** | MUST | Model licences recorded in `MODEL_CARD.md`: all-MiniLM-L6-v2 (Apache-2.0 per its model card, **re-check on Day 0**; frozen). It **is** bundled in the repo for the cloud profile (fp16 `.safetensors` + tokenizer files + licence text, ledger CL4); Apache-2.0 permits redistribution with the licence notice. |

### 6.7 Security, privacy, robustness
| ID | Pri. | Requirement |
|---|---|---|
| **NFR-SEC-01** | MUST | Checkpoints load with `torch.load(..., weights_only=True)`; RNG states are stored as tensors or primitives so this works (doc 07 §5). Inference weights ship as `.safetensors`. |
| **NFR-SEC-02** | MUST | Local profile: the app binds to `localhost` by default (no LAN exposure unless explicitly requested). Cloud profile: the app is public by design, so it holds **no secrets or private data**, accepts only the 300-character prompt and numeric controls, and writes no user data to disk beyond the app log. Streamlit usage statistics are disabled in both (`browser.gatherUsageStats = false`). |
| **NFR-SEC-03** | MUST | Embedded HTML is built only from local templates plus JSON-escaped level data; user prompt text is HTML-escaped wherever it is displayed. |
| **NFR-SEC-04** | MUST | Third-party code in `data/raw/` is never imported (CLAUDE.md §2.10); `MarioEval.jar` runs only in a subprocess with a timeout (doc 21). |
| **NFR-ROB-01** | MUST | Fuzz test: 200 random prompts (random Unicode, empty, 300-character, numbers only, emoji, SQL/HTML fragments) → no exceptions escape; every prompt yields a level or a friendly message. |

### 6.8 Usability and accessibility
| ID | Pri. | Requirement |
|---|---|---|
| **NFR-UX-01** | MUST | Keyboard-only play; controls legend visible under the game. |
| **NFR-UX-02** | MAY | The placeholder skin (A2) uses tile colours that stay distinguishable under deuteranopia simulation; every tile also has a glyph. |
| **NFR-UX-03** | SHOULD | Layout usable at 1366×768 and 1920×1080 (timeline D16). |
| **NFR-UX-04** | MUST | Every non-obvious state has text: "Generating… (window k/N)", "Verifying…", "Not verified because: …". |

### 6.9 Maintainability, testability, observability
| ID | Pri. | Requirement |
|---|---|---|
| **NFR-MNT-01** | MUST | The fast test suite (`pytest -q -m "not slow"`) runs in < 60 s and passes before every commit (CLAUDE.md §3). |
| **NFR-MNT-02** | MUST | Lint with `ruff` (check + format) via pre-commit (doc 05 §8). |
| **NFR-MNT-03** | MUST | No magic numbers in `src/`; all values come from `configs/*.yaml` (CLAUDE.md §2.4). |
| **NFR-OBS-01** | MUST | Training writes `metrics.csv`, `events.jsonl` and TensorBoard logs; the app writes `runs/app_logs/<date>.jsonl` with timings per stage (normalise, sample, sanitize, verify, render). |
| **NFR-OBS-02** | MUST | Every fallback (placeholder sprites, CPU mode, gradient accumulation, gate waiver, `--allow-*` flag) emits an `events.jsonl` entry and a `PROGRESS.md` note (CLAUDE.md §2.12). |

## 7. Acceptance test catalogue

| AT | What | Tool / script | Pass criterion | Gate / Day |
|---|---|---|---|---|
| AT-01 | Environment on both machines | `scripts/verify_env.py --full` | exit 0 on DT and LT; same torch minor | Phase 0 / D0 |
| AT-02 | Data facts | `scripts/audit_dataset.py` | all F1–F17 assertions true | G-P1 / D1 |
| AT-03 | Caption detector fidelity | `pytest tests/test_detector.py` | 100 % (or ≥ 99.5 % with exceptions file) | D1 |
| AT-04 | Split integrity and leakage | `pytest tests/test_splits.py`, `scripts/gate_p1.py` | no shared levels; concept coverage; no test window within Hamming ≤ 2 of train except documented | G-P1 / D2 |
| AT-05 | Render orientation and round trip | `scripts/gate_p1.py` | 10 renders saved; `grid → onehot → argmax` identity | G-P1 / D2 |
| AT-06 | Scheduler maths | `scripts/gate_p2.py` | oracle-DDIM exact; ᾱ_T bounds; x_999 moments | G-P2 / D3 |
| AT-07 | Model and trainer | `pytest tests/test_unet.py tests/test_checkpoint.py`; overfit-one-batch | params = 18,312,013; loss < 0.01 on 8 scenes | D4 |
| AT-08 (SHOULD, simplified) | Cross-machine resume | one resume drill (doc 07 §8, minimal form) | resume verification passes; no loss spike > 2× | D4 |
| AT-09 | Unconditional quality | `scripts/gate_p3.py` | G-P3 thresholds | G-P3 / D5–D6 |
| AT-10 | Conditional adherence | `scripts/gate_p4.py` | dev C-score ≥ 0.75; CFG beats w = 1; adapter in optimiser | G-P4 / D9 |
| AT-11a | Checker validity (**before any use of the playable term**) | `scripts/gate_p5b.py` | checker agreement ≥ 85 % of 300 levels (Java) or ≥ 90 % of 50 hand-labelled levels | G-P5b / **D7** |
| AT-11b | Full level | `scripts/gate_p5.py` | seam ratio ≤ 1.3; 0 broken after sanitizer | G-P5 / D11 |
| AT-12 | Studio UI functions | `pytest tests/test_app_smoke.py` (Streamlit `AppTest`) | prompt, controls, canonical caption, PNG, About panel present | D12 |
| AT-13 | Verified-beatable loop | `pytest tests/test_full_level.py::test_verify_loop` | badge state consistent with checker; budgets respected (5/window, 20 total) | D13 |
| AT-14 | Latency (measure first, then pin) | `scripts/gate_p7.py` | NFR-PERF-01..03 against thresholds pinned after the Day 12 measurement | G-P7 / D12-D13 |
| AT-15 | Game playable | `scripts/smoke_e2e.py` (Playwright, headless Chromium) | 20 prompts: canvas present, player x increases after ArrowRight is held for 1 s, no console errors | G-P7 / D13 |
| AT-16 | Browser matrix | manual checklist + Playwright on Chrome and Edge channels | keyboard works after the click-to-focus overlay | G-P7 / D16 |
| AT-17 | Exports | `pytest tests/test_exports.py` | 5 formats written; JSON validates against schema v1; VGLC round trip identical | D13 |
| AT-18 | Offline and CPU | `scripts/gate_p7.py --offline --cpu` + Wi-Fi-off manual run | app reaches Ready offline; CPU fast mode < 45 s; showcase loads | D16 |
| AT-19 | Robustness fuzz | `pytest tests/test_app_fuzz.py` (slow) | NFR-ROB-01 | D16 |
| AT-21 | Cloud deployment | `scripts/gate_p8.py` (HTTP check of the public URL + Playwright against it) + manual run | URL loads; a prompt yields a level or a labelled showcase fallback; game canvas present; no console errors | **G-P8 / D12 dry run, D14 final** |
| AT-20 | One-shot test discipline | `scripts/eval_test.py` + `reports/final/test_invocations.jsonl` | exactly one successful invocation (or documented metric-bug rerun) | G-P6 / D15 |

## 8. Traceability (requirement → design doc → test → gate)

| Requirement group | Design docs | Tests | Gate |
|---|---|---|---|
| FR-P-01..05 | 08, 09, 10, 11, 12 | AT-02..05 | G-P1 |
| FR-P-06 | 13 | AT-06 | G-P2 |
| FR-P-07..09 | 14, 15, 16, 17, 18, 19 | AT-07..10 | G-P3, G-P4 |
| FR-P-10, FR-A-04..07 | 20, 21 | AT-11, AT-13 | G-P5, G-P5b |
| FR-P-11..14 | 22, 23, 30, 32 | AT-20 | G-P6 |
| FR-A-01..03, 08..15 | 12, 26, 27, 28, 29 | AT-12, AT-14..19 | G-P7 |
| NFR-RES, NFR-REP | 06, 07, 30 | AT-01, AT-08 | Phase 0, D4 drill |

## 9. Non-goals (explicitly out of scope)

1. **No other games** or tile sets (S1). No Lode Runner or Mega Man, even though the MarioDiffusion repo supports them.
2. **No pixel-space diffusion** and no image generation of sprites. Sprites are fixed assets (A1–A3).
3. **No LLM-based prompt understanding.** The normaliser is rule-based (doc 12). A frozen MiniLM is the only language model (V3).
4. **No fine-tuning of the text encoder.** MiniLM stays frozen (M7).
5. **None of the 9 future features in the MVP UI** (S3). Only their seams (§10).
6. **No accounts, databases or multi-user features.** (The earlier non-goal "no cloud deployment / localhost only" is **withdrawn**: Streamlit Community Cloud deployment is mandatory, ledger S6.) The cloud app is stateless and public.
7. **No faithful SMB physics clone.** The game is "SMB-like"; its physics are defined by `physics.json` and tuned so that "checker says beatable" ⇒ "humanly beatable" (doc 21, doc 27).
8. **No moving cannons, Bullet Bills, power-ups, lives system, multiple enemy types or audio** in the MVP (cannons are static solids, timeline D9; one walker enemy type, F6 'E').
9. **No mobile or touch support** (best-effort only).
10. **No bitwise cross-GPU reproducibility** (NFR-REP-05).
11. **No hyperparameter search** beyond the listed ablations X1–X12 (doc 23).
12. **No use of the official leaky splits**, not even "for comparison with the paper", unless clearly labelled as a leaky reference number in an appendix table.
13. **No shipping of Nintendo-derived sprites** (A3).

## 10. MVP vs future seams

The MVP must leave these seams in place, so that each future feature (doc 31) is additive. "Seam" = an interface, data field or artifact that exists and is tested in the MVP, without the user-facing feature.

| # | Future feature (user's list) | Seam delivered in the MVP | Where |
|---|---|---|---|
| F1 | **Watch the proof** (ghost bot runs first; AI redraws stuck spots) | Checker returns the full path plus the failure column; per-window resample loop (G5); `physics.json` shared with the game | `playability/checker.py::check_level → CheckResult(path, fail_col)`; `generate/full_level.py::resample_window` |
| F2 | **Difficulty dial** (crowd of clumsy bots) | Checker exposes a `noise_profile` parameter (jump-timing jitter), currently fixed at 0; level JSON has a `metrics.difficulty` field reserved (null) | doc 21, doc 29 |
| F3 | **Erase and re-ask** (mouse-erase a region, regenerate it) | Generator API `inpaint(grid, mask, caption, seed)`, the same RePaint code path used for outpainting | `generate/full_level.py::inpaint` |
| F4 | **Whole levels, not screens** (pacing arc) | Multi-scene prompt splitter + per-window captions + templated start/flag framing (G3) | `text/normalizer.py::split_scenes`, `generate/framing.py` |
| F5 | **See what it heard** (hover a word → highlight tiles) | Detector `return_details=True` (F19) gives tiles per phrase; the normaliser keeps a word→phrase alignment; optional cross-attention map capture hook | `captions/detector.py`, `text/normalizer.py::NormalizedPrompt.alignment`, `models/unet.py` attention hook |
| F6 | **Report card** (asked for 3 pipes, got 3; beatable; not a copy) | Detector, checker and NN-copy check are callable per level; "report card lite" is **cut** (MAY, only if everything else is done) | `eval/adherence.py`, `eval/novelty.py` |
| F7 | **Blind test** (Nintendo vs AI) | Real dev/test levels and generated levels share level JSON schema v1 and the same renderer | doc 29, `render/renderer.py` |
| F8 | **It practises overnight** (generate → filter → retrain) | `generate_batch` + filters (adherence, playability, novelty) + data manifest versioning (`split_v1` is immutable; a future `selfplay_v1` set would be additive) | `scripts/` (future), doc 30 |
| F9 | **Endless mode** (stretch) | The stride-8 outpainting loop is written as a generator `iter_windows(...)` that can stream windows indefinitely | `generate/full_level.py::iter_windows` |

## 11. Assumptions and constraints

| ID | Assumption | If false |
|---|---|---|
| AS-01 | Both NVIDIA drivers can run the same torch minor (V2) | Pin 2.14 + cu126 on both (doc 06 §3) |
| AS-02 | The laptop runs Windows 11; the desktop runs Windows 11 or Ubuntu | Commands are given for both in doc 06 |
| AS-03 | A cloud-synced folder (OneDrive or Google Drive) is available on both machines | Use the USB / Syncthing alternatives (doc 07 §7) |
| AS-04 | Java may or may not be installable | G-P5b falls back to 50 hand-labelled levels (ledger §9) |
| AS-05 | The deadline allows **18 days: Day 0 = 18 Nov to Day 18 = 6 Dec 2026** (GCR submission; presentations 7-9 Dec) | Cut further per §14 (never cut: level-grouped split, playability checker, one-shot test, MVP game, Streamlit Cloud deployment) |
| AS-07 | Streamlit Community Cloud can run the CPU profile within its memory and build limits | Reduce width cap, shrink the model to fp16, or serve cached showcase levels only (CL6); measured Day 12 |
| AS-08 | GitHub repo can be connected to Streamlit Cloud (private repo access or public mirror) | Public deployment mirror repo (CL8) |
| AS-06 | The instructor accepts CC0 sprites in the report | Optional local classic skin for figures only, never committed (A3) |

Hard constraints: 6 GB minimum VRAM card for training (H2); no paid services required; the Python version is fixed on Day 0 (V1); the app must run both locally and on Streamlit Community Cloud.

## 12. Requirement change control

1. A requirement changes only through a pull request (or commit) that edits this file **and** adds a row below.
2. If the change touches a ledger decision, the ledger is edited first (ledger §14).
3. MUST → SHOULD demotions require a note in `PROGRESS.md` explaining the buffer state (doc 04 §6).

| Date | Requirement | Change | Reason |
|---|---|---|---|
| 2026-11-17 | all | v1.0 created | initial pack |
| 2026-11-18 | §1, §2, §3, US-13/14 | Problem definition rewritten with real users (U1-U3), evaluator separated from users, input/output/objectives and success criteria; no invented research | rubric row 1 |
| 2026-11-18 | FR-A-16/17, NFR-PERF-06, AT-21, §13 | Streamlit Community Cloud deployment and cloud profile added; non-goal 6 and NFR-SEC-02 / NFR-OFF-01 / NFR-LIC-04 amended | statement + rubric row 6; ledger S6, §3b |
| 2026-11-18 | FR-D-01..06 (new §5.3) | Mandatory deliverables become dated requirements | statement; doc 35 |
| 2026-11-18 | FR-A-13 SHOULD -> MUST; FR-A-10, FR-P-13, NFR-PERF-04, NFR-RES-04, NFR-UX-02, US-06, US-12 -> MAY; US-11, AT-08 -> SHOULD (simplified) | fit 18 days (§14) | ledger S7 |
| 2026-11-18 | AT-11 split into AT-11a (D7) and AT-11b (D11); AT-14 measure-first | checker circularity; unmeasured latency | ledger E4, G-P5b, G-P7 |
| 2026-11-18 | AS-05 rewritten (18 days); AS-07, AS-08 added | 21-day plan overran 6 Dec | ledger §10 |
| 2026-11-18 | FR-P-04 detector from grammar, not a port | originality (§15) | ledger O3 |
| 2026-11-18 | §14 effort vs marks, §15 originality (new) | scope and plagiarism rules | ledger S7, S8 |

## 13. Cloud deployment requirements (summary of ledger §3b)

1. Two profiles, one codebase: `local` (GPU, offline-capable, development, live demo) and `cloud` (CPU fast mode, bundled weights).
2. Cloud artifact set: `app/model/t2m_ema.safetensors` (73.5 MB by arithmetic), `app/model/minilm/` (fp16, about 45 MB), pinned `requirements.txt`, `.streamlit/config.toml`, 12 showcase levels, licence files.
3. Cold start: weights load once via `st.cache_resource`; the first visit after a platform sleep is slower; the UI says so and offers showcase levels immediately.
4. Time guard and fallback: generation over the time budget is aborted and replaced by the nearest cached showcase level, labelled as such. The fallback is logged.
5. Gate G-P8 with dated days: dry run Day 12, final Day 14 (frozen checkpoint), re-check Days 16 and 17.
6. Local GPU mode stays the development path; all training and evaluation happen locally.

## 14. Scope vs marks: effort mapping and what is cut or demoted

Total rubric = **135 marks**. Effort is allocated roughly in proportion to marks, with the mandatory-deliverable deductions (-20 % each for a missing deliverable, a non-working live app, use-case mismatch, unjustified scope) treated as the largest risks.

| Rubric row | Marks | Where the effort goes | Main artifacts |
|---|---|---|---|
| 1 Problem definition | 6 | §2 of this doc, report section 1, slide 2 (small, done Day 0) | doc 02 §2, report |
| 2 Dataset and preprocessing | 8 | Day 1-2: audit, level-grouped split, leakage tests | docs 08-12, G-P1 |
| 3 Algorithm understanding | 20 | Report algorithm section, docs 13-15, **viva drills** (doc 36) | report, doc 36 |
| 4 Model implementation | 15 | Days 3-11: from-scratch U-Net, trainer, sampler, same preprocessing at inference | docs 13-21, gates G-P2..G-P5 |
| 5 Evaluation | 12 | Days 5-15: metrics, level-clustered CIs, one-shot test, error analysis | doc 22, evidence folder |
| 6 Streamlit application | 20 (5 for deployment) | Days 12-14, 16: studio, game, **cloud deployment** | docs 27, 28, G-P7, G-P8 |
| 7 Testing and reliability | 12 | Continuous pytest, fuzz test, E2E smoke, evidence | doc 24 |
| 8 Individual viva | 10 | Daily walk-through ritual from Day 4 | doc 36 |
| 9 Live modification | 10 | 15+ drills, every member runs each before 3 Dec | doc 36 §5 |
| 10 Report, code, presentation | 12 | Prose written as sections are finished; slides Days 14-16; zip Day 17 | docs 32, 35 |
| 11 Overall technical demonstration | 10 | Rehearsals Days 16-17; each member's laptop runs the app | doc 32 §6, doc 36 §6 |

**Cut or demoted to fit 18 days** (everything below is reflected in the requirement tables above, the timeline and the risk register):

| Item | Was | Now | Where |
|---|---|---|---|
| 21-day plan, buffer day, separate hardening day | Day 0-20 | 18-day plan; one buffer day (Day 6) plus Day 17 | doc 03 |
| Dual-GPU handoff protocol (doc 07) | full protocol, every machine switch | **simplified**: single owner per run in `HANDOFF.md`, one resume drill (AT-08, SHOULD); the rest of doc 07 is reference, not a gate | doc 07 header, AT-08 |
| Ablations (doc 23) | X1-X12 in two batches | **X1 (aux CE) plus the linear-vs-cosine schedule comparison (Day 5)** only; X2, X3, X4 run only if time and a trigger fires (MAY); X5, X6, X8, X10 cut; X7 is a free analysis (SHOULD); X9 only if the memorisation trigger fires | doc 23 |
| 5-fold grouped CV | SHOULD | **MAY** (overnight Day 14 to 15 if GPUs are free; otherwise report seed variance and the per-level table, and say so) | FR-P-13 |
| Risk register (doc 33) | 38 risks | **Keep short**: only the top-10 by score are tracked daily (doc 33 §0); the rest stays as reference | doc 33 |
| Report card lite, Tiled export, denoising animation | optional | MAY / cut | FR-A-10, doc 28 |
| Future-feature seams (doc 31) | tested seams | MAY, only seams that cost < 30 min (F1 path return, F3 inpaint signature) | doc 31 |
| FPS probe, deuteranopia check, disk policy | SHOULD | MAY | NFR-PERF-04, UX-02, RES-04 |
| Firefox | best effort | out of scope | NFR-OFF-03 |

**Never cut:** level-grouped split, playability checker (with its validation), one-shot test discipline, the MVP game, Streamlit Cloud deployment, report, slides, evidence folder, zip.

## 15. Originality and attribution (ledger §11b)

This project **builds on** the published *MarioDiffusion* work (Schrum et al., arXiv 2507.00184; MIT-licensed repository): its dataset of 16x16 Mario windows with rule-based captions, and the idea of text-conditioned diffusion over tiles. The statement forbids copying an existing project, so the report says plainly what is external and what is ours.

| External (used and credited) | Group's own work |
|---|---|
| Mar1and2 windows + captions (MarioDiffusion dataset), VGLC levels beneath it | **Level reconstruction and level-grouped split with near-duplicate families, leakage audit of the official split (F17, F21)** |
| The idea of caption-conditioned tile diffusion | **From-scratch conditional U-Net, noise schedules, DDPM/DDIM/CFG sampler and trainer** (no diffusion library) |
| Caption grammar as published (62 phrases) | **Detector implemented from the grammar description and verified against the stored captions; casual-text normaliser; prompt sets** |
| all-MiniLM-L6-v2 (frozen) | Trainable adapter and cross-attention wiring |
| Phaser 3 (MIT), Kenney sprites (CC0) | **Browser game with shared physics, Streamlit studio, cloud deployment** |
| MarioEval.jar (optional) | **Physics-consistent playability checker and its validation (G-P5b)** |
| Paper's C-score definition | **Evaluation protocol: null ladder, level-clustered CIs, per-level tables, one-shot pre-registered test, novelty/copy references** |

Viva answer: doc 32 §7 question C7. Third-party code in `data/raw/` is never imported or transcribed (CLAUDE.md §2.10, ledger O3).

