# 16 — Training Pipeline (trainer specification)

> **Owner:** M-model (doc 36). **Days:** 4 (trainer core, resume drill), 5 (uncond runs), 7-8 (conditional), 10 (X1). **Ledger rows:** H2-H5, M1, M9-M12, D7-D8. **Modules (doc 05 §4.6):** `train/train.py`, `loop.py`, `optim.py`, `checkpoint.py`, `validate.py`, `early_stop.py`, `models/ema.py`. **Values:** doc 17. **Runbook:** doc 18. **Diagnostics:** doc 19. **Failures:** doc 25.
> Every number here comes from `configs/*.yaml` (doc 17) and traces to the ledger. This document defines behaviour; it does not restate measurements. Anything not yet measured is marked "to be measured on Day N".

## 1. Entry point and run flow
`python -m text2mario.train.train --config configs/<overlay>.yaml [--resume <ckpt>] [--fold k]`, normally started through `scripts/launch.py` (creates the run directory, snapshots the resolved config, writes git SHA, machine profile and `meta.json`; doc 30 §3).
1. `bootstrap_env()` first (doc 05); load `base.yaml` + overlay + machine YAML; **refuse to start** if the machine YAML overrides any key outside `{num_workers, cudnn_benchmark, compile, log_every}` (CLAUDE.md §2.5).
2. Apply the VRAM cap `set_per_process_memory_fraction(4.5 GiB / total)` (H3).
3. Verify `data/splits/LOCK`; load the GPU-resident dataset (H7, doc 11); build model, EMA, optimiser, scheduler (doc 13), text path if conditional (doc 15).
4. Resume if requested (§7); otherwise step 0.
5. Loop (§3) until `train.max_steps` or early stop (§11); graceful stop on `SIGINT` / `STOP` file writes a checkpoint first.
6. Write `finished` event, final EMA `.safetensors`, `best.json`.

## 2. Precision and memory
bf16 autocast on both GPUs, no GradScaler (H5); attention softmax in fp32 (SDPA); the scheduler's buffers are fp64 and cast at use (doc 13). Peak VRAM is logged as `sys/vram_peak_gib` and must stay <= 4.5 GiB (H2). Never raise the cap; reduce the micro-batch and accumulate (§17).

## 3. Step function
```
x0 = dataset.batch(64)                       # {-1,+1} one-hot (D7), caption strings (D8 augmentation done by the dataset)
t  ~ Uniform{0..T-1} (per sample), eps ~ N(0,I)   # CPU generator seeded from run seed (machine independent)
xt = q_sample(x0, t, eps)
ctx = text_path(captions)  if conditional else learned_null_context     # doc 15
eps_hat = unet(xt, t, ctx)                    # autocast bf16
loss = mse(eps, eps_hat)  [+ lambda * alpha_bar_t-weighted CE(x0_hat logits, y)]   # M9; lambda = 0 in the baseline
```
Gradient clip 1.0 (M10). A non-finite loss or gradient skips the step and logs `skipped_nonfinite`; 5 consecutive skips abort with `nan_abort` (doc 25 R-3.3).

## 4. Loss variants
Baseline = plain ε-MSE (M9). X1 adds `aux_ce_lambda = 0.001` (doc 23). Channel-weighted MSE is a rollback only (doc 25 R-3.x), never a default.

## 5. Optimiser, schedule, EMA
- AdamW, lr 2e-4, betas (0.9, 0.999), batch 64, clip 1.0 (M10).
- **Parameter groups:** *decay* group (weight decay 0.01) = Conv/Linear weights incl. the adapter weights; *no-decay* group = norm scales/shifts, all biases, embeddings, learned null context. The text encoder (MiniLM) is **not** in the optimiser (test, doc 03 Day 7).
- LR: linear warm-up 1,000 steps, then cosine decay to 1e-5 over `train.max_steps`.
- EMA decay 0.999 with warm-up `min(0.999, (1+n)/(10+n))` (M11); the adapter is included. **Evaluation and export use EMA weights only.**

## 6. Checkpoints
Atomic: write `*.tmp`, fsync, rename (CLAUDE.md §2.9); verify by reloading the header and checking the sha256 sidecar. Contents: model, EMA, optimiser, scheduler state, step, RNG states (as tensors/primitives so `weights_only=True` loads), sampler state (doc 11), config hash, data and split hashes, git SHA. Cadence `checkpoint.every_steps = 2000`, keep last 2 plus the `best.json` one; milestone every 10,000 steps for `cond_main` and `final` (doc 07 §6). EMA weights are also saved as `.safetensors` (inference, the cloud profile, CL3).

## 7. Resume
Full-state resume: step counter, LR position, EMA, optimiser moments, RNG streams, sampler order. A resumed run must show a continuous loss curve (no spike > 2x the running mean; doc 25 R-0.2). Changing the split (`n`) or config hash invalidates a resume; the trainer refuses. Cross-machine resume follows the minimal drill of doc 07 (AT-08, SHOULD).

## 8. RNG portability
Initial noise, dropout masks of the dataset and sampler order use CPU `torch.Generator` objects seeded from `(run_seed, step, stream_id)`; GPU RNG is used only for in-network dropout. This makes data order and noise machine-independent (doc 07 §5.3). No claim of bitwise GPU reproducibility (NFR-REP-05).

## 9. Data feed and caption dropout
The dataset (doc 11) yields `(x0, caption_string)`. Augmentations (a) shuffle, (b) phrase dropout 0.10, (c) casual paraphrase 0.30 (train lexicon half only), then (d) **null caption `""` with probability 0.10** (D8, doc 15 §6), using the dataset's CPU generator. Empirical rates are tested (0.10 +/- 0.01 over 10k draws).

## 10. Validation loss (deterministic, per t-bucket)
`validate.py` evaluates the dev windows (stride-8 subset) on a **fixed set of (window, t, eps) triples** generated once from a fixed seed, so the curve is noise-free across evaluations (doc 30 §6.5). It logs `val/loss`, `val/loss_t0..t9` (10 equal t-buckets, bucket k = t in [100k, 100k+99]) and, for conditional runs, the same triples with the null caption to give the **conditioning gain** `G_b = L_b(null) - L_b(cond)` (doc 15 §8; must turn positive in the mid-t buckets in the Day-7 smoke run). It also logs `simplex_dev` = mean |sum_c x0_hat_c + 11| per bucket (doc 13 §1 diagnostic). Per-bucket losses are compared only within one schedule (doc 13).

## 11. Dev evaluation inside training and early stopping
Every 2,000 steps (M12): sample from the **EMA** model on P-dev (conditional) or 256 unconditional samples (Phase 3), DDIM 50, **fixed w = 5.0** (so checkpoint selection is decoupled from the w search; doc 15 §7), 3 seeds, compute the metrics of doc 22 and the composite. **Composite:** `S` (E4) only if `reports/gates/P5b/RESULT.json` says `pass: true`; otherwise `S_noplay` (ledger E4 circularity rule) and the run is tagged accordingly. Also log the **final-step margin** (top-1 minus top-2 of x0_hat at the last step; healthy median > 1.0, doc 13 §11). `early_stop.py`: patience 5 evaluations, tie rules per doc 22 §3.13; `new_best` writes `best.json`.

## 12. Evaluation cadence and cost
Sampling dominates wall-clock (audit #18). Cost per dev evaluation is to be measured on Day 5 and recorded in `hw_report.json`; the planning assumption of doc 04 §5.1 (1-2 minutes) is replaced by the measurement. Phase 3: evaluations at 5k/10k/15k steps. Phase 4: every 2k steps up to `max_steps` 60,000 (X1: 40,000).

## 13. Logging
`metrics.csv`, `events.jsonl` (closed vocabulary, doc 30 §4.2), TensorBoard tags `<split>/<name>[_t<k>]` (doc 30 §4.1), `hw_report.json` after step 200 and at the end. Log every `log_every` steps (machine YAML may change it).

## 14. Watchdogs
Thermal: NVML temperature > 87 C for 60 s writes `watchdog_thermal`, checkpoints and stops (H9). Disk: refuse to start with < 20 GB free (doc 06). Heartbeat file for `HANDOFF.md` (doc 07).

## 15. Text-encoder cost
MiniLM runs online, frozen, `no_grad`, bf16 with a string-keyed LRU cache (M8). The cache hit rate and the added step time are measured on Day 7 (smoke run) and recorded in `hw_report.json`; doc 15 §5 gives the FLOP estimate (an estimate, not a measurement). WordPiece length of every caption including augmented variants is measured on Day 1 to confirm `max_len = 48` (M7).

## 16. Failure handling
OOM -> enable accumulation automatically (§17), log `oom_fallback`. NaN -> §3. Crash -> resume from the last good checkpoint. Any fallback is logged to `events.jsonl` and `PROGRESS.md` (CLAUDE.md §2.12). Symptom to action table: doc 25.

## 17. Gradient accumulation fallback
Micro-batch `m` with `64/m` accumulation steps is mathematically equivalent to batch 64 because GroupNorm is batch-independent (doc 13 §12) and the loss is averaged over the full batch. It is used only after an OOM and is logged.

## 18. Tests and acceptance (Day 4)
`tests/test_unet.py` (18,312,013 parameters, shapes, gradient reaches every parameter), `tests/test_checkpoint.py` (atomic save, round trip, resume equality), `tests/test_optim.py` (groups; adapter in optimiser, MiniLM not), `tests/test_dataset.py` (null rate). **Overfit-one-batch:** 8 fixed scenes, fixed t, 1,500 steps -> loss < 0.01. **Resume drill** (simplified): 1,000 steps then 500 after resume, continuous curve. **Min-VRAM:** batch 64, 200 steps -> `hw_report.json` on the laptop, <= 4.5 GiB; per-pass forward time recorded for the G-P7 measure-first rule.

*Explain-me note (for viva, doc 36):* the trainer does one thing per step: corrupt a real window with noise at a random strength, ask the network to predict the noise, and nudge the weights; the EMA copy is a smoothed version used for sampling.
