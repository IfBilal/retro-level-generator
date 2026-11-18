# 17 — Hyperparameters and Configs

> **Rule:** this table must equal the ledger (`00_DECISIONS_LEDGER.md`). If they differ, the ledger wins; fix this file in the same commit. Every YAML value traces to a ledger row. Values marked *(provisional)* or "to be measured on Day N" are not facts yet.

## 1. Config system
1.1 `configs/base.yaml` holds every default. Experiment overlays (`uncond_linear.yaml`, `uncond_cosine.yaml`, `cond_smoke.yaml`, `cond_main.yaml`, `ablations/x01_aux_ce.yaml`, `final.yaml`) override only what they change. Machine YAMLs (`configs/machine/*.yaml`) may override only `num_workers`, `cudnn_benchmark`, `compile`, `log_every` (H4).
1.2 Resolution order: base -> overlay -> machine -> CLI `--set key=value` (logged).
1.3 Unknown keys are an error. No magic numbers in `src/` (NFR-MNT-03).
1.4 Overlays used on the app side: `configs/generate/app.yaml` (local) and `configs/app/cloud.yaml` (cloud profile, ledger §3b).
1.5 **Config hash:** sha256 of the canonical JSON (sorted keys) of the resolved config **excluding** the machine allow-list keys and `log_every`; stored in `meta.json`, checkpoints and level JSON. Any model or training key change changes the hash and the run identity.

## 2. Table (key, value, ledger row, rationale)
| Key | Value | Ledger | Rationale |
|---|---|---|---|
| `data.window` | 16 x 16 | F3, D7 | dataset geometry |
| `data.tile_classes` | 13 | F6 | tile vocabulary |
| `data.representation` | one-hot scaled to {-1,+1} | D7 | PDF-faithful |
| `data.split_path` | `data/splits/split_v1.json` | D3 | level-grouped |
| `data.gpu_resident` | true | H7 | 2 MB, no workers |
| `aug.phrase_shuffle` | always | D8a | order is not a cue |
| `aug.phrase_dropout` | 0.10 (keep >= 1 phrase) | D8b | regularise |
| `aug.paraphrase_p` | 0.30, train lexicon half | D8c | casual robustness |
| `aug.null_caption_p` | 0.10 (applied last) | D8d, M7 | CFG |
| `aug.hflip` | false | D8 | breaks level semantics |
| `diffusion.T` | 1000 | M1 | |
| `diffusion.schedule` | `linear` (beta 1e-4 -> 0.02) or `cosine` (s = 0.008, beta clip 0.999); winner chosen on Day 5 | M2 | tie -> cosine |
| `diffusion.prediction` | `eps` | M1 | |
| `model.base_channels` | 64 | M3 | |
| `model.channel_mult` | [1, 2, 4] | M3 | 64/128/256 |
| `model.res_blocks_enc / dec` | 2 / 3 | M3 | |
| `model.groups` | 8 | M3 | GroupNorm |
| `model.dropout` | 0.1 | M3 | |
| `model.self_attn_levels` | [8x8, 4x4], 4 heads | M5 | |
| `model.cross_attn_levels` | [8x8, 4x4], 4 heads, d_ctx 128 | M6 | |
| `model.time_embed` | sinusoidal 128 -> MLP 256 -> FiLM per ResBlock | M4 | |
| `text.encoder` | `all-MiniLM-L6-v2` frozen | V3, M7 | |
| `text.max_len` | 48 (provisional; measure Day 1, set 64 if exceeded) | M7 | |
| `text.adapter` | Linear(384,128) -> SiLU -> Linear(128,128) -> LayerNorm | M7 | 66,048 params |
| `text.cache` | LRU, string keyed | M8 | |
| `loss.type` | `mse` | M9 | |
| `loss.aux_ce_lambda` | 0.0 (X1: 0.001) | M9 | |
| `optim.name` | AdamW | M10 | |
| `optim.lr` | 2e-4 | M10 | |
| `optim.betas` | [0.9, 0.999] | M10 | |
| `optim.weight_decay` | 0.01 (not norms/biases/embeddings) | M10 | |
| `optim.warmup_steps` | 1000 | M10 | |
| `optim.lr_final` | 1e-5 (cosine) | M10 | |
| `optim.grad_clip` | 1.0 | M10 | |
| `train.batch_size` | 64 (global, both machines) | H4 | |
| `train.precision` | bf16 autocast | H5 | |
| `train.max_steps` | 15,000 (Phase 3) / 60,000 (cond_main) / 40,000 (X1) | M12 | budgets in steps |
| `train.vram_cap_gib` | 4.5 | H2, H3 | |
| `ema.decay` | 0.999, warm-up `min(0.999,(1+n)/(10+n))` | M11 | |
| `ema.extra_decays` | [] (X2: [0.995]) | M11 | |
| `eval.every_steps` | 2000 | M12 | |
| `eval.patience` | 5 | M12 | |
| `eval.train_w` | 5.0 | M13, doc 15 §7 | selection decoupled from w search |
| `eval.seeds` | 3 | E3 | |
| `eval.composite` | `S` if G-P5b passed else `S_noplay` | E4 | circularity rule |
| `eval.bootstrap` | B 10,000, seed 12345, unit = **level** | E3 | |
| `checkpoint.every_steps` | 2000 | doc 07 | |
| `checkpoint.keep_last` | 2 | CLAUDE.md 2.9 | never < 2 |
| `checkpoint.milestone_every_steps` | 0 (cond_main, final: 10000) | doc 07 | |
| `sample.ddim_steps` | 50 | M13 | |
| `sample.eta` | 0 | M13 | |
| `sample.clip_x0` | true | M13 | |
| `sample.cfg_convention` | `eps_u + w (eps_c - eps_u)` | M13 | w = 1 is no guidance |
| `sample.w_grid` | [1, 2, 3, 5, 7.5, 10] | M13 | default chosen on dev (Day 9) |
| `generate.stride` | 8 | G1 | |
| `generate.resample_r` | 2 for the first 60 % of steps | G1 | |
| `generate.widths` | [16, 48, 80, 112] | G2 | |
| `generate.frame_start / frame_end` | 3 / 6 columns | G3 | templated |
| `generate.max_attempts_window / total` | 5 / 20 | G5 | |
| `generate.fast_mode` | 25 steps, r = 0 | H10 | forced on without CUDA |
| `app.cloud.profile` | `cloud` via `T2M_PROFILE` | CL1 | |
| `app.cloud.default_width` | 48 | CL2 | |
| `app.cloud.time_budget_s` | 60 *(provisional; pin Day 12)* | CL6 | then showcase fallback |
| `app.cloud.rss_cap_gb` | 1.0 *(provisional; pin Day 12)* | CL5 | |
| `app.cloud.ckpt` | `app/model/t2m_ema.safetensors` | CL3 | |
| `splits.seed` | 20261117 | D3 | |

## 3. Overlay specifics
`uncond_linear` / `uncond_cosine`: `text.enabled: false`, learned null context, `max_steps: 15000`, `diffusion.schedule` set, `eval.composite: S_noplay`. `cond_main`: text on, `max_steps: 60000`, schedule = Day-5 winner, milestones 10000. `x01_aux_ce`: `loss.aux_ce_lambda: 0.001`, `max_steps: 40000`. `final`: config v1 plus the chosen `default_w`; its hash goes into `FROZEN.json`.

## 4. Things that are NOT hyperparameters yet
Wall-clock numbers, it/s (assumption DT 15 / LT 9, ledger H11), latency thresholds (G-P7 provisional), cloud memory/time caps: all pinned only after the Day 4 / Day 12 measurements.
