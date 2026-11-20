# HANDOFF

Run ownership (edit only when you start or stop a GPU run; doc 07, simplified in the 18-day plan).
**Full instructions for the current handover: top of `PROGRESS.md`.**

| Field | Value |
|---|---|
| Current owner of the GPU run | Taimoor (Phase 3 training and monitoring) |
| Run in flight | none yet (to be filled in when U-lin / U-cos start) |
| Latest checkpoint | none (no model has been trained yet) |
| Last updated | 2026-11-20 |

Runs: `runs/U-lin` (`configs/uncond_linear.yaml`) and `runs/U-cos` (`configs/uncond_cosine.yaml`), 15,000 steps each.
When you start or stop one, edit this table (owner, run, last checkpoint step, date) and push only this file.
Milestone checkpoints (step 5000/10000/15000) are kept automatically. After both runs finish: `python scripts/gate_p3.py --runs runs/U-lin runs/U-cos`.
Checkpoints live in `runs/` (git-ignored); share them through a drive/USB, never through git.
