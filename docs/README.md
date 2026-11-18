# Text-to-Mario Level Diffusion — Implementation Harness (v1.0)

A complete, multi-document plan that Claude Code follows to build, train, evaluate and ship a **text-conditioned discrete-tile diffusion model for Super Mario Bros. levels**, plus a **Streamlit app where a typed sentence becomes a real, playable browser game** ("Type it, play it").

It starts from the 6-page PDF blueprint, audits it hard (`docs/01_PDF_AUDIT.md`), fixes what it gets wrong, and adds what it lacks: an honest level-grouped data split, playability verification, a two-GPU workflow, a browser game engine, licensed assets, and a test discipline you can defend at a viva.

## How to use this pack with Claude Code

1. Create an empty repo folder `text2mario/`. Copy this pack in: `docs/` → `text2mario/docs/`, `CLAUDE.md` → repo root, `reference/` → `text2mario/reference/`.
2. Open Claude Code in `text2mario/` and say: **"Read CLAUDE.md, then docs/00_DECISIONS_LEDGER.md and docs/03_TIMELINE_DAY_BY_DAY.md. Start Day 0."**
3. Each session: Claude Code reads `PROGRESS.md` + `HANDOFF.md`, does the current day's checklist, runs the day's gate script, and updates `PROGRESS.md`.
4. Switching GPUs: follow `docs/07_DUAL_GPU_HANDOFF_PROTOCOL.md` (a two-minute checklist).

## Reading order

| Order | Doc | Purpose |
|---|---|---|
| 1 | `CLAUDE.md` | Operating rules for Claude Code (copy of doc 34) |
| 2 | `00_DECISIONS_LEDGER.md` | **Single source of truth**: every number, verified facts |
| 3 | `01_PDF_AUDIT.md` | 30 findings against the PDF and how each is fixed |
| 4 | `02_SCOPE_AND_REQUIREMENTS.md` | MVP definition, acceptance criteria, non-goals |
| 5 | `03_TIMELINE_DAY_BY_DAY.md` | Day 0–20 plan, gates, fallbacks |
| 6 | `04_CRITICAL_PATH_AND_DEPENDENCIES.md` | Dependency graph, slack, what can run in parallel |

## Full index

**Planning & infrastructure**
- `02_SCOPE_AND_REQUIREMENTS.md` — functional and non-functional requirements, acceptance tests
- `03_TIMELINE_DAY_BY_DAY.md` — the itinerary
- `04_CRITICAL_PATH_AND_DEPENDENCIES.md` — DAG, critical path, buffers
- `05_REPO_STRUCTURE.md` — repo tree, module contracts, coding conventions
- `06_ENVIRONMENT_SETUP.md` — Windows/Linux setup for both machines, version pins, verification
- `07_DUAL_GPU_HANDOFF_PROTOCOL.md` — min-VRAM contract, machine profiles, checkpoint sync, resume

**Data engineering**
- `08_DATA_ACQUISITION.md` — pinned sources, checksums, licences
- `09_DATA_AUDIT_AND_EDA.md` — assertions, exploratory analysis, expected values
- `10_DATASET_SPLITS_AND_CV.md` — level-grouped train/dev/test, 5-fold grouped CV, leakage tests
- `11_DATA_PIPELINE_AND_REPRESENTATION.md` — tensors, GPU-resident loader, augmentation
- `12_CAPTION_GRAMMAR_AND_PROMPT_NORMALIZER.md` — canonical grammar, casual→canonical normalizer

**Deep learning**
- `13_DIFFUSION_MATHEMATICS.md` — DDPM/DDIM/CFG/RePaint derivations, the loss, schedules
- `14_MODEL_ARCHITECTURE_UNET.md` — exact layer-by-layer U-Net, shapes, parameter counts
- `15_TEXT_CONDITIONING_AND_CFG.md` — MiniLM + adapter + cross-attention + guidance
- `16_TRAINING_PIPELINE.md` — trainer specification (AMP, EMA, checkpointing, validation)
- `17_HYPERPARAMETERS_AND_CONFIGS.md` — every YAML value with its rationale
- `18_TRAINING_RUNBOOK.md` — exact run sequence and commands, what to watch
- `19_OVERFITTING_UNDERFITTING_BIAS_VARIANCE.md` — diagnostics, thresholds, remedies

**Generation, verification, evaluation**
- `20_SAMPLING_AND_FULL_LEVEL_GENERATION.md` — DDIM + CFG, outpainting, framing, sanitizer
- `21_PLAYABILITY_CHECKER.md` — physics-consistent A* checker + MarioEval.jar bridge
- `22_EVALUATION_PROTOCOL_AND_METRICS.md` — metric formulas, prompt sets, statistics
- `23_ABLATIONS_AND_EXPERIMENTS.md` — experiment matrix X1–X12
- `24_VERIFICATION_GATES_AND_TESTS.md` — gate scripts + pytest inventory
- `25_ROLLBACK_DIAGNOSTIC_MATRIX.md` — symptom → cause → rollback target → action

**Application**
- `26_ASSETS_PIPELINE.md` — autonomous CC0 asset fetch, tile mapping, fallbacks
- `27_BROWSER_GAME_ENGINE.md` — Phaser 3 game spec ("type it, play it")
- `28_STREAMLIT_APP.md` — studio app spec
- `29_LEVEL_SCHEMA_AND_EXPORT_FORMATS.md` — level JSON schema v1 and exports

**Process & delivery**
- `30_EXPERIMENT_TRACKING_AND_REPRODUCIBILITY.md` — run IDs, seeds, manifests, hashes
- `31_FUTURE_FEATURES_HOOKS.md` — seams reserved for the other 9 features
- `32_REPORT_AND_VIVA_PREP.md` — figure list, tables, viva Q&A bank
- `33_RISK_REGISTER.md` — risks, likelihood/impact, mitigations, triggers
- `34_CLAUDE_CODE_OPERATING_RULES.md` — the rules (also at root as `CLAUDE.md`)

**Reference code**
- `reference/ref_unet.py` — reference U-Net + scheduler checks used to verify this pack's numbers (18,312,013 params; oracle-DDIM exact reconstruction)

## The project in five lines
- **Data:** 7,687 16×16 windows from 42 SMB1/SMB2 levels with rule-based captions. Split **by level**: 31 train / 5 dev / 6 locked test.
- **Model:** from-scratch 18.3 M-parameter conditional U-Net (16→8→4), ε-prediction DDPM (T = 1000), frozen MiniLM + trainable adapter, cross-attention at 8×8 and 4×4, CFG.
- **Generation:** DDIM 50 steps, sliding-window outpainting with RePaint harmonisation, templated start/flag framing, playability-verified resampling.
- **Evaluation:** C-score, exact match, broken-structure rate, playability, novelty vs train, copy rate, diversity; bootstrap CIs; 5-fold grouped CV; one-shot test.
- **App:** Streamlit studio + Phaser 3 game. Type a sentence and play the level in the browser.
