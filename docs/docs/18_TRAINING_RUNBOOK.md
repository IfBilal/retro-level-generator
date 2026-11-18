# 18 — Training Runbook

> **Use:** exact run sequence and what to watch. Specs: doc 16 (trainer), doc 17 (configs). Failure handling: doc 25. Days follow doc 03 (Day 0 = 18 Nov). Times are *planning assumptions* (ledger H11: DT 15 it/s, LT 9 it/s) until `hw_report.json` replaces them on Day 4.

## 0. Before any run (every time)
```
git pull
python scripts/verify_env.py --quick          # machine profile, torch minor, VRAM cap check
python scripts/status.py                      # critical task, gate states, run ETAs (doc 04 §8)
cat HANDOFF.md                                # who owns the GPU run (doc 07, simplified: one owner per run)
```
Required gate files: Phase 3 needs `reports/gates/P2/RESULT.json` pass; Phase 4 needs P3 pass. The trainer refuses otherwise.

## 1. Day 4 checks (no real training yet)
```
pytest -q tests/test_unet.py tests/test_checkpoint.py tests/test_optim.py
python -m text2mario.train.train --config configs/smoke_overfit.yaml     # 8 scenes, fixed t, 1500 steps -> loss < 0.01
python scripts/bench.py --config configs/base.yaml --steps 200            # hw_report.json on the LAPTOP (<= 4.5 GiB), plus per-pass forward time
# resume drill (minimal): 1000 steps on DT, handoff, 500 on LT (doc 07)
```

## 2. Day 5, Phase 3 (unconditional)
```
python scripts/launch.py --config configs/uncond_linear.yaml   --machine desktop_3060ti   # 15k steps
python scripts/launch.py --config configs/uncond_cosine.yaml   --machine laptop_4050      # 15k steps
python scripts/eval_dev.py --run <run_id> --suite uncond --watch
python scripts/gate_p3.py --linear <run> --cosine <run>
```
Selection uses `S_noplay` (checker not validated yet; ledger E4). Watch: `train/loss_t*` falling in all buckets, `skipped_nonfinite` = 0, floor validity and broken rate at 5k/10k/15k, tile JS. Record the schedule choice in the ledger change log (M2).
**If G-P3 fails:** doc 25 R-3.x rows in order; Day 6 absorbs one retrain. Never "train longer" as the first move.

## 3. Day 7 smoke (conditional)
```
python scripts/launch.py --config configs/cond_smoke.yaml    # 3k steps
```
Checks: adapter parameters in the optimiser, MiniLM not; null rate 0.10 +/- 0.01; loss decreases; conditioning gain `G_b` > 0 in mid-t buckets; cross-attention entropy below the uniform maximum. Failing -> doc 25 R-4.x.

## 4. Day 8, C-main
```
python scripts/launch.py --config configs/cond_main.yaml --machine desktop_3060ti     # <= 60k steps, early stop
python scripts/eval_dev.py --run <run_id> --suite dev --watch
```
`playable` enters S only because G-P5b passed on Day 7 (else `S_noplay`). Evening check: loss per bucket falling; dev composite improving; cross-attention not uniform. Queue the next job so no GPU idles overnight (`scripts/queue.py`).

## 5. Day 9 CFG sweep, Day 10 X1
```
python scripts/sweep_cfg.py --run <best> --w 1 2 3 5 7.5 10 --shard A|B ; python scripts/sweep_cfg.py --merge
python scripts/gate_p4.py
python scripts/launch.py --config configs/ablations/x01_aux_ce.yaml    # 40k steps, same seeds, DT
```
Decision rule for X1 (doc 23): adopt only if the paired level-clustered CI of delta S is above 0 and delta S >= 0.01 and delta copy rate <= +0.02; otherwise keep C-main (the simpler option).

## 6. Day 14 freeze
`configs/final.yaml` from config v1 + the dev-selected `default_w`; evaluate on dev (doc 22); write `FROZEN.json` (doc 30 §9.6), tag `v0.9-frozen`; export the EMA weights to `.safetensors` for the cloud profile (`scripts/export_cloud_ckpt.py`). No training after the freeze except the optional CV folds (report-only).

## 7. Optional CV (MAY): `python scripts/launch.py --config configs/cv_fold.yaml --fold k` for k = 1..5 over the 36 non-test levels, DT 1-3, LT 4-5; aggregate with `scripts/aggregate_cv.py`.

## 8. What to watch (every run)
| Signal | Healthy | If not |
|---|---|---|
| `train/loss`, `train/loss_t0..t9` | falling, no NaN | doc 25 R-3.3 |
| `val/loss_t*` vs `train/loss_t*` | gap small and stable | doc 19 |
| `sys/vram_peak_gib` | <= 4.5 | reduce micro-batch (logged) |
| `sys/gpu_temp_c` | < 87 for 60 s | watchdog stops the run |
| `dev/copy_rate`, `dev/novelty_med` | copy <= reference + 0.05; novelty near the real-dev reference | doc 19 §3 |
| loss curve after a resume | continuous | doc 25 R-0.2 |

## 9. End-of-session ritual
Update `HANDOFF.md` (run, step, ETA, heartbeat) and `PROGRESS.md`; push; queue the next job. A run is never left without an owner entry.
