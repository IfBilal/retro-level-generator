# Pack status

Updated 2026-11-20 (v1.1 + implementation of the pre-training phase). Plan: **18 days, Day 0 = Wed 18 Nov 2026 to Day 18 = Sun 6 Dec 2026 (GCR submission)**.

| State | Docs |
|---|---|
| Complete (first draft, amended 2026-11-18) | 00-36, CLAUDE.md, README, `reference/ref_unet.py` |
| Written 2026-11-18 (previously missing) | 16 trainer, 17 hyperparameters/configs, 18 runbook, 19 over/underfitting diagnostics, 20 sampling and full-level generation, 21 playability checker, 25 rollback matrix, 29 level schema, 31 future-feature seams (demoted to MAY) |
| New 2026-11-18 | 35 submission checklist, 36 team ownership and viva |
| Not planned | A red-team decision-record doc and a red-team review doc (earlier drafts called them 35 and 36; **35 is now the submission checklist and 36 the ownership/viva doc**; no red-team review is scheduled in the 18-day plan) |

## Implemented and run, 2026-11-20/21 (details: `PROGRESS.md` top section, `reports/gates/*/RESULT.json`, `report/report.pdf`)
Everything in docs 02-36 that can be built without a login has been built, run and evaluated: data, split, detector, schedules, U-Net, trainer, text path, normalizer, C-score, sampler, full-level generator,
sanitizer, playability checker, Phaser game, level schema/exports, Streamlit app, evaluation, freeze and the one-shot test, report, slides, evidence folder, model/data cards, cloud-ready export.
Trained: unconditional baselines (15k steps, linear and cosine, plus dense and aux-CE variants), conditional C-main (early-stopped at 36,295; **frozen step 18,000, CFG w = 7.5**), X1 ablation (stopped at 17,640).

## Gate outcomes (honest)
G-P1, G-P2 pass. **G-P3 fail** (waiver), **G-P4 fail on P4.04**, **G-P5 fail on P5.01** (seam ratio), **G-P5b fail** (jar agreement 0.663; `S_noplay` fallback). G-P7 latency measured (target not met at 112 columns in fast mode:
12.9 s median). G-P8 (Streamlit Cloud) not done: needs the team's login (`DEPLOY.md`).

## Still open (honest list)
- Streamlit Community Cloud deployment, viva and live-modification drills, laptops, GCR upload, human playthrough of showcase levels (`human_played: false`), the hand-labelled checker set (needs two people).
- Checker: 69 % of real full levels solved (target 90 %); end-to-end replay of 'verified' levels by a re-planning bot finishes 40-70 % (lower bound). UI wording: 'route found (not a guarantee)'.
- Model: seam ratio 1.54 (limit 1.3); P-rand stress prompts score below an always-empty generator; absence ('no enemies') cannot be conditioned.
- X1 (aux CE) looked better on dev but was evaluated after the freeze; C-main remains the reported model.

Change history: `docs/00_DECISIONS_LEDGER.md` §14 and `docs/02_SCOPE_AND_REQUIREMENTS.md` §12.
