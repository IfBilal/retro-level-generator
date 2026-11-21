# Viva and live-demo guide (rubric rows 8, 9, 11)

Every member should be able to explain every part. This guide is a map, not a script: run the commands yourself and say what you see.

## 1. The 60-second explanation
We turn a typed sentence into a playable Mario-style level. (1) A rule-based **normalizer** rewrites the sentence into the exact caption grammar the model was trained on.
(2) A frozen **MiniLM** turns the caption into token vectors, a small trainable **adapter** maps them to the U-Net's context. (3) A **diffusion U-Net** (18.3 M parameters) starts from random noise and removes it step by step into a 16x16 tile grid,
steered by the text through cross-attention and **classifier-free guidance**. (4) Long levels are built from windows that advance 8 columns at a time, each keeping the overlap of the last (**RePaint**). (5) A **checker** searches for a route from start to flag under the game's own physics; if it fails,
the window where it got stuck is re-sampled. (6) The same grid becomes a **Phaser** game in one offline HTML file, shown by the **Streamlit** app.

## 2. Trace one input end to end (run it)
```
python scripts/trace_example.py "full floor with two pipes and a few enemies" --seed 3
```
It prints: the sentence -> normalizer scenes -> canonical caption -> MiniLM word-pieces -> context tensor (1, 12, 128) -> start noise (1, 13, 16, 16) -> 50 DDIM steps with the guidance formula -> arg-max tile grid -> the caption the detector reads back ->
C-score -> framing -> checker verdict. Be ready to say what each shape is: **13** = tile classes (one-hot, values in {-1, +1}); **16 x 16** = rows x columns of one window; **12 tokens** = MiniLM word-pieces of that caption; **128** = the U-Net context width.

## 3. Numbers to know (all from saved files)
| Fact | Value | Where |
|---|---|---|
| Data | 7,687 windows, 42 levels; split by level 31 / 5 / 6 (5,709 / 905 / 1,073 windows) | `data/splits/`, report 2 |
| Detector check | reproduces 7,687 / 7,687 stored captions | `scripts/check_detector.py` |
| U-Net / adapter | 18,312,013 / 66,048 parameters; optimiser holds exactly 18,378,061 (test P4.03) | `scripts/verify_env.py`, `gate_p4` |
| Training speed / memory | 0.111 s per step, 0.77 GiB peak | `runs/hw_probe/hw_report.json` |
| Frozen model | C-main step 18,000, guidance w = 7.5, cosine schedule | `FROZEN.json` |
| Test result (unseen levels) | C-score 0.819 [0.767, 0.861] vs prompt-ignoring null 0.463 | `reports/final/RESULTS.md` |
| Latency (fast mode) | 4.8 s for 48 columns, 12.9 s for 112 (median, RTX 3060 Ti) | `reports/app/latency.json` |

## 4. Be ready for the awkward questions (the honest answers)
- *"Did everything pass?"* No. G-P3 (unconditional baseline: broken structures and copying move in opposite directions with training; waiver written), G-P4's contrast check (omitting "enemies" does not remove them: captions never say absence), G-P5's seam ratio (1.54 > 1.3), and G-P5b (the reference solver MarioEval.jar disagrees with our checker, agreement 0.663, and fails a 1-tile wall in synthetic levels). Each is in `reports/gates/*/RESULT.json`.
- *"Is 'verified' true?"* It means our checker found a route under the game's physics with enemies assumed stompable. A re-planning bot that cannot stomp finishes 40-70 % of such levels, so the app says "route found (not a guarantee)".
- *"Why is the test number good but the random-prompt number bad?"* P-test prompts are captions of real held-out windows; P-rand is seven random topics, mostly unseen combinations. On P-rand the C-score (0.348) is below an always-empty level. Say so; it is in the report.
- *"Did you tune on the test set?"* No. Selection used dev only; `FROZEN.json` was written before `scripts/eval_test.py` ran once (the first attempt crashed on a 49-token prompt, changed nothing, and is logged).
- *"What would you do next?"* Choose the auxiliary-cross-entropy variant (looked better on dev, evaluated after the freeze), widen the data, add position words to the grammar, improve seam handling (stride 4).

## 5. Live-modification drills (doc 36 §5) mapped to this repository
All generation and app numbers are in **`configs/generate/app.yaml`**; model and training numbers are in `configs/base.yaml` and the run overlays.
| # | Drill | Edit | Check |
|---|---|---|---|
| 1 | guidance scale | `cfg.default_w` in `configs/generate/app.yaml` (slider default), or pass `--w` | `python scripts/trace_example.py --w 1` vs `--w 10`: read-back caption and C-score |
| 2 | merge two tile classes | `src/text2mario/data/tiles.py` | `pytest -q tests/test_data.py tests/test_detector.py`: which tests fail and why |
| 3 | linear vs cosine | `diffusion.schedule` in a run overlay (needs a matching trained model); strips: `python scripts/gate_p2.py` | corruption strips in `reports/gates/P2/` |
| 4 | new synonym | `data/prompts/lexicon.yaml` (e.g. add `tubes` under pipe) | `python scripts/eval_normalizer.py`; `normalize("two tubes")` |
| 5 | stride 8 -> 4 | `stride: 4` in `configs/generate/app.yaml` | windows (48-16)/4+1 = 9; `pytest -q tests/test_full_level.py -k stride`; seam positions change |
| 6 | add a metric | `src/text2mario/eval/structure.py` + add it in `eval/devloop.py` (`out[...]`) | appears in `reports/dev_eval/*.json` after `python scripts/eval_dev.py CKPT` |
| 7 | DDIM steps 50 -> 20 | `modes.full.steps` (or `modes.fast.steps`) | latency vs C-score |
| 8 | resampling jumps 2 -> 0 | `modes.full.resample_r` | seam appearance, `scripts/gate_p5.py --n 10` |
| 9 | add width 64 | `widths: [16, 48, 64, 80, 112]` (must satisfy (w-16) % stride == 0) | the slider shows it; 7 windows |
| 10 | checker jump rise | `jump_velocity` in `app/static/game/physics.json`, then `python scripts/derive_physics.py` | `checker_max_rise` changes; physics hash changes; `pytest -q tests/test_checker.py` |
| 11 | dropout | `model.dropout` in `configs/base.yaml` | `python scripts/verify_env.py`: parameter count unchanged |
| 12 | null-caption probability | `aug.null_p` in `configs/base.yaml` | `pytest -q tests/test_text.py -k null_rate` fails/passes |
| 13 | aux-CE weight | `python -m text2mario.train.train --config configs/uncond_linear.yaml --run-dir runs/tmp --set train.max_steps=50 --set loss.aux_ce_lambda=0.001` | printed loss values |
| 14 | new example chip | `examples:` in `configs/generate/app.yaml` | a new button in the app sidebar |
| 15 | new export format | `src/text2mario/render/schema.py` (`export_all`) | round-trip test in `tests/test_level_schema.py` |
| 16 | show the seed of window k | `window_seed` in `src/text2mario/generate/full_level.py` | determinism: same prompt + seed gives the same level |
| 17 | showcase size / filter | `PROMPTS` and the `verified` condition in `scripts/make_showcase.py` | `app/showcase/manifest.json` |
| 18 | cloud time budget | `cloud.time_budget_s` (with `T2M_PROFILE=cloud`) | "time budget" in the verification reason |
| 19 | how many objects per window / why one per window | `adherence.max_new_per_window` in `configs/generate/app.yaml` (raise `pipe` to 2) | `python scripts/audit_adherence.py --label try --ids cnt_pipe_2 --n-seeds 5` (pass rate drops: the model hits "two pipes in one window" only ~10-30 %) |
| 20 | plan "several" at its low end instead of the middle | `adherence.range_target: low` | `tests/test_adherence_properties.py::test_range_target_low_vs_mid`; the per-window captions in the app's Details tab |
| 21 | turn the count verification off (see the old bug) | `adherence.enabled: false` | `one pipe` gives one pipe per window again: `scripts/audit_adherence.py --ids cnt_pipe_1` |
| 22 | candidates per window on the GPU | `adherence.batch_tries` (1 = off) | seconds per level in `reports/audit/*/RESULTS.md`; CPU always uses `max_window_tries` |
| 23 | the largest count a level can hold | `adherence.capacity_per_16_cols` | typing "five pipes" with width 16 gives the message "will not fit" |
| 24 | what the model is told per window | open the **Details** tab after generating | "Caption each 16-column window was actually given" |

## 5b. Trace of the count path (audit P2) - say this in the viva
1. `normalize("flat ground, one pipe, then several coins and a staircase")` -> 2 scenes; each scene keeps `want` (pipe [1,1]; coin [5,9], staircase [1,1]) besides the coarse caption.
2. `generate()` -> `adherence.check_feasible` (contradiction / too many for the width -> a clear message) -> `plan_caption` per window: only what that window must ADD ("full floor. one pipe." for the one window that carries the pipe, "full floor." for the others) -> the window is redrawn until its NEW columns match (CUDA: 16 candidates in one pass).
3. The RAW level is counted per scene region (`adherence.evaluate`, the caption detector's definition); an unmet count resamples the owning window (<= 16 whole-level resamples, <= 40 extra draws). The result is stored in `verification.adherence` and the app says "Counts match your sentence" or which counts were not met. Nothing is ever edited.
4. Why it was broken: the scene caption went to EVERY window (one pipe per 16 columns), the caption only has 5 buckets, and the model is loose at exact counts (`reports/audit/window_exactness.json`). Evidence: `reports/audit/baseline` (113/628) vs `reports/audit/<final round>`.

## 6. Demo checklist (each member's laptop)
`streamlit run app/streamlit_app.py` (works offline; CPU fast mode is automatic without a GPU) -> click an example -> Generate & Play -> click the game -> arrows/WASD, Space, Shift. Fallback: the **Gallery** page shows pre-made levels.
Before the viva run `pytest -q -m "not slow"` (about 35 s) and have `reports/final/RESULTS.md`, `FROZEN.json` and `reports/gates/` open.
