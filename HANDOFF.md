# HANDOFF

Run ownership (edit only when you start or stop a GPU run; doc 07, simplified in the 18-day plan).

| Field | Value |
|---|---|
| Current owner of the GPU run | nobody: all training has finished or been stopped on purpose |
| Run in flight | none |
| Frozen model | `runs/C-main/step_0018000.pt` (sha256 in `FROZEN.json`), CFG w = 7.5, cosine schedule; exported inference bundle in `app/model/` |
| Stopped / finished | `runs/C-main` stopped at step 36,295 by `STOP` file (early stop: dev composite peaked at 18k). `runs/C-x01` (X1 ablation, aux CE) was paused at 17,640 for memory pressure, resumed after fixing a resume bug and finished at 40,000. |
| Last updated | 2026-11-22 (audit/prompt-adherence merged to main) |

Checkpoints live in `runs/` (git-ignored) and are not part of the submission zip; the zip ships the exported bundle instead (`scripts/export_cloud_ckpt.py`).
