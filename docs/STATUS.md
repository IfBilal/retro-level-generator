# Pack status

Updated 2026-11-20 (v1.1 + implementation of the pre-training phase). Plan: **18 days, Day 0 = Wed 18 Nov 2026 to Day 18 = Sun 6 Dec 2026 (GCR submission)**.

| State | Docs |
|---|---|
| Complete (first draft, amended 2026-11-18) | 00-36, CLAUDE.md, README, `reference/ref_unet.py` |
| Written 2026-11-18 (previously missing) | 16 trainer, 17 hyperparameters/configs, 18 runbook, 19 over/underfitting diagnostics, 20 sampling and full-level generation, 21 playability checker, 25 rollback matrix, 29 level schema, 31 future-feature seams (demoted to MAY) |
| New 2026-11-18 | 35 submission checklist, 36 team ownership and viva |
| Not planned | A red-team decision-record doc and a red-team review doc (earlier drafts called them 35 and 36; **35 is now the submission checklist and 36 the ownership/viva doc**; no red-team review is scheduled in the 18-day plan) |

## Implemented 2026-11-20 (see `PROGRESS.md`)
Data fetch, audit (39/39), detector (7,687/7,687), locked level-grouped split, dataset/augmentation, scheduler, U-Net (18,312,013 params), text conditioner, CFG sampler, trainer and 139 fast tests; gates G-P1 and G-P2 pass. **No model has been trained.**

## Still open (honest list)
- No model, GPU measurement or app exists yet. Every remaining "to be measured on Day N" and "(provisional)" value is an assumption: GPU latency (G-P7), cloud limits and timings (G-P8), it/s (H11), CPU cost on the cloud, hand-label agreement, level-clustered CI widths.
- Version pins (torch, Streamlit, Python) are **to verify on Day 0** (ledger V1, V2, V4).
- Docs 16-21, 25, 29, 31 are new and have not been cross-reviewed against code; the first implementation days (4, 7, 11) should confirm their interfaces.
- Scope cuts (doc 02 §14): CV optional, ablations X1 + schedule comparison only, dual-GPU protocol simplified, risk register kept to the top 10, report card cut.

Change history: `docs/00_DECISIONS_LEDGER.md` §14 and `docs/02_SCOPE_AND_REQUIREMENTS.md` §12.
