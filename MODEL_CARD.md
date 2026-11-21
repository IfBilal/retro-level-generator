# Model card: text2mario level generator

## Model details
- **What it is:** a text-conditioned denoising-diffusion model over 16x16 tile windows (13 tile classes, {-1, +1} one-hot), used by a sliding-window outpainter to build levels up to 112 columns wide.
- **Architecture:** U-Net, 18,312,013 parameters (channel multipliers 1-2-4 from 64, GroupNorm residual blocks, self-attention at 8x8 and 4x4, cross-attention to text tokens). Text: frozen `all-MiniLM-L6-v2` + trainable adapter (66,048 parameters).
- **Training:** AdamW, lr 2e-4 (1,000 warm-up, cosine to 1e-5), batch 64, bf16, EMA 0.999, 60,000 steps (cosine noise schedule, T = 1000, eps-prediction MSE, 10 % caption dropout for classifier-free guidance). One NVIDIA RTX 3060 Ti; peak memory below 1 GiB.
- **Inference:** EMA weights, DDIM 50 steps (25 in CPU fast mode), CFG `eps_u + w (eps_c - eps_u)`, x0 clipping, stride-8 RePaint outpainting with 2 resampling jumps.
- **Frozen release:** see `FROZEN.json` (checkpoint sha256, step, guidance w, config hash, physics hash, split lock hash). Exported inference bundle: `app/model/` (`EXPORT.json`).
- **Authors / licence:** the project group; code under the repository licence; see `THIRD_PARTY_NOTICES.md` for the dataset, encoder, Phaser and Kenney assets.

## Intended use
Fast, text-steerable first drafts of Mario-like platformer levels for hobbyists, students of procedural content generation and demonstrations; each level is checked by a conservative playability checker and can be played in the browser.

## Out-of-scope use
Not a production level-design tool; not a general text-to-image or text-to-game model; not suitable for other tile sets, other game physics or for claims about player enjoyment or difficulty. "Verified beatable" is a statement about *our checker under our game's physics*, not about every player.

## Training data
MarioDiffusion corpus (7,687 windows, 42 levels of Super Mario Bros. and The Lost Levels) with rule-based captions; **only the 31 training levels (5,709 windows)** are used for training. See `DATA_CARD.md`.

## Evaluation data
5 dev levels (905 windows; 115 dev prompts) for all selection; 6 held-out test levels (1,073 windows; 137 test prompts) used **once** after the freeze (`reports/final/RESULTS.md`).

## Metrics
C-score (caption adherence, with the prompt-ignoring null shown beside it), exact caption match, per-concept precision/recall/exact match, broken-structure rate, floor validity, nearest-training-window Hamming distance and copy rate, tile-distribution JS divergence, seam ratio, checker-verified rate. 95 % intervals resample held-out **levels**; with 6 test levels they are wide.

## Quantitative analyses
`reports/final/RESULTS.md` (test), `reports/gates/*/RESULT.json` (every gate, pass or fail), `reports/dev_eval/` (dev checkpoint and guidance sweeps), per-level and per-concept tables, failure gallery.

## Ethical considerations and caveats
- Level designs derive from Nintendo games; the raw data and any Nintendo sprites are not redistributed; the app uses CC0 Kenney sprites. Academic, non-commercial use.
- Small training set (31 levels): the model can reproduce training structures (copy rate is reported against the real-data reference) and cannot express positions or exact counts beyond five buckets.
- Known failed gates, kept visible: G-P3 (unconditional baseline could not satisfy both the broken-structure and copy-rate bounds), G-P5b (reference-solver agreement). Fallbacks applied: `S_noplay` for selection; physics-contract and replay evidence for the verification badge.
- No user study was performed.

## How to reproduce
See `README.md` and Appendix A of the report: `scripts/fetch_data.py`, `build_index.py`, `fetch_models.py`, `fetch_assets.py`, `make_atlas.py`, then `python -m text2mario.train.train --config configs/cond_main.yaml`, the gate scripts, `freeze.py`, `eval_test.py`.
