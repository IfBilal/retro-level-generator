# HANDOFF

Run ownership (edit only when you start or stop a GPU run; doc 07, simplified in the 18-day plan).

| Field | Value |
|---|---|
| Current owner of the GPU run | nobody |
| Run in flight | none |
| Latest checkpoint | none (no model has been trained yet) |
| Last updated | 2026-11-20 |

Training has deliberately NOT been started. To start Phase 3 on a GPU machine: `python scripts/verify_env.py`, then
`python -m text2mario.train.train --config configs/uncond_linear.yaml` (and `uncond_cosine.yaml` on the other machine).
