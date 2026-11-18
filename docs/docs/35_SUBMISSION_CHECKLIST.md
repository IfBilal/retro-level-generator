# 35 — Submission Checklist

> Maps every mandatory deliverable (project statement), every rubric row (`Rubrics_SoftwareCategory.pdf`, total **135** marks) and every deduction rule to the artifact that satisfies it and the day that produces it. Dates: Day 0 = Wed 18 Nov 2026 ... Day 18 = **Sun 6 Dec 2026 (GCR submission)**; presentations 7-9 Dec. Tick boxes on Day 17; upload on Day 18. (This file replaces the earlier plan for a red-team decision-record doc 35.)

## 1. Mandatory deliverables (statement)
| # | Deliverable | Artifact (path in the zip) | Produced | Done when |
|---|---|---|---|---|
| 1 | Source code: complete, organised, runnable, commented, meaningful names, clear structure | `src/`, `app/`, `scripts/`, `configs/`, `tests/`, `README.md`, `requirements.txt` | continuous; checked D17 | fresh unzip installs and `pytest -q -m "not slow"` passes; each module has an explain-me note (CLAUDE.md §6) |
| 2 | Project report (problem, real-world justification, selected path, objectives, methodology, algorithm description, pipeline diagram, dataset, implementation, testing, results, limitations, conclusion) | `report/report.pdf` (+ source) | prose D0-D15 (doc 32 §1.2); assembled D16; PDF D17 | each section of doc 32 §1.1 present |
| 3 | Live demonstration (working Streamlit app) | app in zip + public cloud URL | cloud D12/D14; rehearsal D16-17 | runs on **every member's laptop** (doc 36 §6); cloud URL loads |
| 4 | Presentation slides (problem, motivation, solution, technique, implementation, testing, results qualitative + quantitative, limitations, conclusion) | `slides/slides.pdf` (+ pptx) | draft D14; final D16 | 12 slides (doc 32 §9.1) |
| 5 | Evidence of evaluation (ML metrics and results; Path A) | `evidence/` | collected D15-D16 | contents below |
| - | Video | **Path B only; not required** | - | - |

## 2. Rubric (135 marks) -> artifact -> day
| Row | Marks | Checkpoints | Artifact | Day |
|---|---|---|---|---|
| 1 Problem definition | 6 | specific real-world problem (2); identified user and need (2); input, output, objectives (2) | doc 02 §2 pasted into report section 1 and slide 2 | D0 (draft), D16 |
| 2 Dataset and preprocessing | 8 | source/relevance (2); feature/target (2); preprocessing implementation (2); correct split, leakage prevention (2) | report section 3, `DATA_CARD.md`, `reports/eda/`, `data/splits/LOCK`, leakage tests, G-P1 | D1-D2 |
| 3 Algorithm understanding | 20 | working (5); justification (5); parameters/architecture and I/O (10) | report sections 4 + App. A, doc 13-15, Fig 1/7, viva drills | prose D4-D8; drills D4-D15 |
| 4 Model implementation | 15 | complete pipeline (5); correct implementation and execution (5); consistent preprocessing at inference (2); real predictions, not hard-coded (3) | `src/`, gates G-P2..G-P5, same normaliser/encoder path in app and training, `FROZEN.json`, sample sheets | D3-D14 |
| 5 Model evaluation | 12 | metrics (3); unseen test data (3); quantitative results/visuals (3); error analysis (3) | `reports/final/RESULTS.md`, one-shot test log, level-clustered CIs, per-level table, Fig 21 failure gallery | D14-D15 |
| 6 Streamlit application | 20 | real model integration (10); input interface (3); correct result display (5); validation/error handling (2); end-to-end workflow (5); **deployment on Streamlit Community Cloud (5)** | `app/`, G-P7, **G-P8**, `evidence/deployment/` (URL, commit SHA, screenshot) | D12-D14, re-check D16-D17 |
| 7 Testing and reliability | 12 | multiple representative inputs (3); invalid/boundary (2); evidence (5); consistent outputs (2) | `tests/`, `test_app_fuzz.py`, E2E smoke, `reports/gates/*/RESULT.json`, determinism test (same seed same level), `evidence/testing/` | D13, D16 |
| 8 Individual viva | 10 | explains algorithm (3); traces input through preprocessing and prediction (3); technical questions (4) | doc 36 ritual, doc 32 §7 | daily from D1; mock vivas D17 |
| 9 Live modification | 10 | independent coding, runtime changes | doc 36 §5 drills, each member by end of D15 | D4-D15 |
| 10 Report, code, presentation | 12 | report and methodology (3); organised runnable code (3); technical presentation and results (5); complete deliverables (1) | report, repo, slides, this checklist | D16-D17 |
| 11 Overall technical demonstration | 10 | working solution (5); explains key decisions (5) | live app + doc 32 §6 script, doc 36 | D16-D17 rehearsals |

## 3. Evidence-of-evaluation folder (`evidence/`)
`metrics/` (RESULTS.md, `metrics.json`, per-level and per-concept tables, CIs with the level-clustered method), `figures/` (from `make_figures.py`), `gates/` (all `RESULT.json`), `test_run/` (one-shot log, `test_invocations.jsonl`, `eval_plan.json`, `FROZEN.json`), `checker_validation/` (agreement, confusion matrix, `human_labels.csv`), `testing/` (pytest output, fuzz and E2E logs, latency.json), `deployment/` (URL, commit SHA, screenshot, `cloud_dryrun.json`), `samples/` (showcase and failure levels), `ablations/` (X1 and schedule comparison, plus "not run" list). Built by `scripts/collect_evidence.py` on D16.

## 4. Building and checking the single zip (D17), uploading (D18)
1. `git status` clean; tag `v1.0`; `scripts/build_submission_zip.py` includes: repo source (no `data/raw/`, `runs/`, training checkpoints, `.env`, secrets), `app/model/` (EMA safetensors + MiniLM fp16), `report/`, `slides/`, `evidence/`, `LICENSES/`, `README.md` with run instructions and the **cloud URL**, `FROZEN.json`.
2. Size check (target well below any GCR limit; the GCR limit is **to be checked on D0**). Zip name: `<group>_text2mario_AI2002.zip` (adapt to any naming rule on the GCR post).
3. Clean-room test: unzip into an empty directory on a different machine/user, create the venv from `requirements.txt`, run `pytest -q -m "not slow"`, start `streamlit run app/streamlit_app.py`, generate and play one level.
4. Every member repeats step 3 on their own laptop (rubric deduction rule).
5. Only **one** member uploads on D18; record the upload time, size and sha256 in `evidence/`.

## 5. Deduction rules -> prevention
| Rule (from obtained marks) | Prevention | Check |
|---|---|---|
| Submission guidelines not followed (-2) | single zip on GCR only; one uploader; all deliverables inside | section 4 |
| Missing any mandatory deliverable (-20 %) | section 1 table all ticked | D17 |
| Application not working live (-20 %) | local run on every laptop + cloud URL + offline/CPU rehearsal + cached showcase fallback | D16 rehearsals |
| Use case not per instructions (-30 %) | Path A, ML at the core (diffusion model generates the level), Streamlit deployment, documented dataset, metrics reported and interpreted | doc 02 §1-2, report |
| Scope unjustified (-30 %) | problem definition (doc 02 §2), originality statement (§15), effort-to-marks table (§14) | report section 1 |
| A member without a laptop running the project (-20 %) | doc 36 §6 | D17 |

## 6. Final sign-off (D17 evening)
- [ ] Section 1 all done; [ ] section 2 every row has its artifact; [ ] G-P1..G-P8 `pass: true` (or documented fallback); [ ] zip clean-room test on all laptops; [ ] cloud URL loads; [ ] no secrets or tokens in the zip; [ ] licences present (Kenney CC0, Phaser MIT, MiniLM Apache-2.0, MarioDiffusion MIT credited).
