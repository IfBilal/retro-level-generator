# CLAUDE.md — Operating Rules for Claude Code (text2mario)

> This file is copied to the repo root as `CLAUDE.md`. It is the first thing to read in every session.

## 0. Session start (always, in this order)
1. `git pull`; read `PROGRESS.md` (last entry) and `HANDOFF.md` (who has the GPU run, where the latest checkpoint is).
2. `python scripts/verify_env.py --quick` → note the machine profile (desktop_3060ti / laptop_4050).
3. Open `docs/03_TIMELINE_DAY_BY_DAY.md` at the current day; work its checklist **top to bottom**.
4. Before writing code for a module, read its spec doc (index in `README.md`) and `docs/00_DECISIONS_LEDGER.md`.

## 1. Source-of-truth hierarchy
`00_DECISIONS_LEDGER.md` > topic doc (08–33) > timeline > your own judgement. If two docs disagree, follow the ledger and log the discrepancy in `PROGRESS.md` under "Doc conflicts". Never silently pick one.

## 2. Hard rules (never break)
1. **Never use the MarioDiffusion repo's own train/validate/test files.** Only `data/splits/split_v1.json` (level-grouped). The test split's levels are touched only by `scripts/eval_test.py`, once, on Day 15 (3 Dec), after `FROZEN.json` exists.
2. **Never tune anything on test numbers.** Model selection, CFG scale, thresholds: dev only.
3. **Tile ID order is ledger F6.** Import it from `text2mario.data.tiles`; never retype it.
4. **All numbers live in `configs/*.yaml`.** No magic constants in code (except mathematical ones).
5. **Same hyperparameters on both GPUs.** Machine YAMLs may change only `num_workers`, `cudnn_benchmark`, `compile`, `log_every`. The trainer refuses to start if a machine YAML overrides any other key.
6. **VRAM cap 4.5 GiB** is applied at process start on both machines (H2–H3). Do not raise it to "make something fit"; reduce the micro-batch and use accumulation (logged).
7. **Evaluate with EMA weights**, DDIM 50, x̂₀ clipping, the CFG convention `ε_u + w(ε_c − ε_u)`.
8. **Measure before repair.** Sanitizer repairs are for display; metrics use raw samples.
9. **Atomic checkpoints** (write `*.tmp`, fsync, rename). Never delete the last two good checkpoints.
10. **Do not import code from `data/raw/`** (third-party clone). Read data files only; re-implement logic in `src/` with attribution.
11. **Assets:** only CC0/MIT assets are committed (Kenney, Phaser). Nintendo-derived sprites are never committed.
12. **No silent fallbacks.** Any fallback (placeholder sprites, CPU mode, grad accumulation, gate waiver) is logged to the run's `events.jsonl` and `PROGRESS.md`.

13. **Original work:** write the caption detector from the grammar description (doc 12) and check it against the stored captions; never port or transcribe MarioDiffusion's code (ledger O3). Credit the dataset and idea (doc 02 §15).
14. **Measure before pinning:** latency, memory, throughput and version numbers are *to be measured/verified* (ledger V1/V2/V4, H11, G-P7, G-P8). Do not state them as facts until measured.
15. **Playable term only after the checker is validated:** use `S_noplay` until `reports/gates/P5b/RESULT.json` says `pass: true` (ledger E4).
16. **Confidence intervals cluster by held-out level**, never by prompt alone; always report per-level results too (ledger E3).

## 3. Workflow rules
- **Gate before advancing.** Each phase ends with `python scripts/gate_pN.py`, which writes `reports/gates/PN/RESULT.json` (`pass: bool`, metrics, timestamp, git SHA). The next phase's scripts check that file exists with `pass: true`.
- **Failed gate →** open `docs/25_ROLLBACK_DIAGNOSTIC_MATRIX.md`, find the symptom row, apply the action, re-gate. Do **not** train longer to fix a data/scaling/capacity bug.
- **Tests:** `pytest -q` must pass before every commit (fast suite < 60 s; slow suite marked `@pytest.mark.slow`).
- **Commits:** small, conventional (`feat(data): ...`, `fix(sched): ...`), one logical change each. Never commit `runs/`, `data/raw/`, checkpoints, or `.env`.
- **Long GPU runs:** launch with `scripts/launch.py` (writes run dir, config snapshot, git SHA, machine profile). Update `HANDOFF.md` when you start or stop a run.
- **Every figure is produced by a script** in `scripts/` from saved artifacts; never hand-made.
- **End of session:** update `PROGRESS.md` (Done / In-flight runs / Blocked / Next), push.

## 4. When uncertain
- Spec unclear → check the ledger → the topic doc's "Decisions" section → if still unclear, choose the **simplest option that keeps all gates measurable**, implement it behind a config flag, and log it under "Assumptions" in `PROGRESS.md`.
- Never invent dataset facts. Measure them (the audit script) and add them to the ledger.

## 5. Definition of Done (project)
- G-P1 … G-P8 all `pass: true` (or the documented fallback); `FROZEN.json` exists; the one-shot test results are in `reports/final/RESULTS.md`.
- `streamlit run app/streamlit_app.py` on every member's laptop: type a sentence → a full level is generated, verified and playable in the browser within the latency pinned after the Day-12 measurement (target < 10 s on GPU, provisional), and works offline; the same app is **deployed on Streamlit Community Cloud** (cloud profile, ledger §3b).
- The submission zip is built and clean-room checked on Day 17 and uploaded on Day 18 (6 Dec) per `docs/35_SUBMISSION_CHECKLIST.md`.
- Report figures regenerate with `python scripts/make_figures.py`.

## 6. Code must be explainable (every member must be able to explain and modify it live)
The group must own the work (project statement; rubric rows 8, 9, 11). So:
1. **Comment generously and plainly.** Every function gets a one-line purpose; every non-obvious step (shapes, why a clamp, why this order) gets a why-comment. Meaningful names.
2. **Explain-me note per module:** a short docstring at the top of each module: what it does in two sentences, its inputs and outputs, which module comes before and after it, and one setting to change to see an effect.
3. **Keep plans and code small enough for 2-3 students:** prefer the simplest implementation that keeps the gates measurable; no clever abstractions; no framework layers nobody asked for.
4. **Support the drills:** keep the knobs of `docs/36_TEAM_OWNERSHIP_AND_VIVA.md` §5 easy to find (one config key or one small function each) so a student can change them live.
5. **Never write a change the humans cannot explain:** after a non-trivial change, add 3-5 lines to `PROGRESS.md` ("what changed, why, how to see it"), and expect the daily 20-minute walk-through (doc 36 §3) to cover it.
