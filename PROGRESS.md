# PROGRESS

## >>> HANDOVER (2026-11-20): Phase 3 model training, for the friend who has the GPU <<<

**Who does what.** The pre-training code is finished and pushed. From here the friend with the GPU **runs and monitors the
Phase 3 unconditional training and reports back**; he is not expected to finish the project. The original developer (with
Claude Code) continues from the reported results: sampling evaluation, `gate_p3.py`, choosing the schedule, then Phase 4.
This laptop (RTX 4050) cannot train right now: no NVIDIA driver is installed (only the open-source `nouveau`; no
`nvidia-smi`; torch sees no CUDA).

**Do not change** the code, configs, split files (`data/splits/*`, locked) or hyperparameters without telling the group first.
Never use or look at the test split (`MarioWindows("test")` refuses anyway). Machine YAMLs may only change
`num_workers`, `cudnn_benchmark`, `compile`, `log_every`. Read `CLAUDE.md` once (5 minutes).

### Step 1: set up (about 15 min)
```bash
git clone https://github.com/T361/retro-level-generator.git && cd retro-level-generator
uv venv --python 3.12 .venv && source .venv/bin/activate          # or any Python 3.12 venv
# CUDA wheel of torch 2.15.x that matches YOUR driver (check `nvidia-smi` first), e.g. for CUDA 12.6:
uv pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cu126
uv pip install -r requirements-dev.txt
python scripts/fetch_data.py && python scripts/build_index.py && python scripts/fetch_models.py
python scripts/audit_dataset.py && python scripts/gate_p1.py && python scripts/gate_p2.py     # all must PASS
python scripts/verify_env.py        # must be green, including "VRAM cap: a 5 GiB allocation must fail" (needs the GPU)
pytest -q -m "not slow"             # 139 tests should pass
```
If torch has no wheel for your driver, tell the group: both machines must use the same torch minor version (ledger V2).
Copy the output of `verify_env.py` and `nvidia-smi` into your report (machine name, GPU, VRAM, driver).

### Step 2: 200-step hardware probe (about 5 min)
```bash
python -m text2mario.train.train --config configs/uncond_linear.yaml --set train.max_steps=200 --run-dir runs/hw_probe
```
Send back: `runs/hw_probe/hw_report.json` (peak VRAM; must be <= 4.5 GiB) and the median of the `step_s` column in
`runs/hw_probe/metrics.csv` for steps 50-200 (= seconds per step; steps/s = 1 / that). These replace the planning assumptions
of ledger H11. If it runs out of memory the trainer logs `oom_fallback` in `events.jsonl` and halves the micro-batch; report it.

### Step 3: the real runs (15,000 steps each, same hyperparameters on both)
```bash
python -m text2mario.train.train --config configs/uncond_linear.yaml --run-dir runs/U-lin
python -m text2mario.train.train --config configs/uncond_cosine.yaml --run-dir runs/U-cos
```
- With ONE GPU, run them one after the other (or both at once if VRAM allows; the speed of that is unmeasured, so note what you did).
  Use `tmux`/`nohup` so a closed terminal does not kill them.
- Plug in the laptop / keep the GPU cool. **The automatic thermal watchdog is NOT implemented**: watch `nvidia-smi`; if the GPU
  stays above 87 C for a minute, stop the run safely with `touch runs/<run>/STOP` (it saves a checkpoint and exits).
- **Resume** after any interruption: `python -m text2mario.train.train --config <same overlay> --run-dir runs/<run> --resume runs/<run>/step_00XXXXX.pt`
  (it refuses if the config differs; that is intentional). Checkpoints are written every 2,000 steps; the last two are kept.

### Step 4: monitor (this is the main job)
Files in each run directory: `metrics.csv` (step, loss, lr, grad_norm, step_s), `events.jsonl` (started, checkpoint,
skipped_nonfinite, oom_fallback, finished), `meta.json`, `config.yaml`, `step_*.pt`, `ema.safetensors` at the end.

What healthy looks like:
- Step 0 loss is about **1.0** (0.95-1.05). If not, stop: something is wrong with the data or model wiring.
- Loss falls quickly during the first ~1,000 warm-up steps, then keeps slowly improving. No `skipped_nonfinite` events
  (5 in a row aborts the run with `nan_abort`).
- Every checkpoint, run the dev check (uses EMA weights, fixed noise, so numbers are comparable):
  ```bash
  python scripts/validate_ckpt.py runs/U-lin/step_0004000.pt     # also at 6000, 10000, 14000 and the final one
  ```
  It prints the dev loss per noise-level bucket next to the "context-free floor" of that schedule (mean floor: linear 0.0076,
  cosine 0.0116). **Rule (doc 13 §5): at about 5k steps the mean dev loss must be BELOW the floor.** If it is not, or any
  mid bucket stays above its floor, stop and debug (data scaling, tile order, wiring: see `docs/docs/25_ROLLBACK_DIAGNOSTIC_MATRIX.md`
  and `19_...`); **do not just train longer.** Do not compare buckets across the two schedules, only within one.
- A dev loss that keeps falling while the training loss falls is fine. Dev loss rising while training loss falls = overfitting
  (doc 19); report it, do not fix it yourself.

### Step 5: report back (send to the group chat / drive; runs/ is git-ignored on purpose)
Per run: `metrics.csv`, `events.jsonl`, `hw_report.json`, all `*.val.json`, the **last checkpoint `step_0015000.pt`** (+ its
`.sha256`) and `ema.safetensors`; plus the numbers from Step 2, wall-clock time, and anything odd. Add a short dated entry to
this file under "Handover log" below and push only that (and nothing else) to GitHub.

### Handover log (friend fills in; newest first)
| Date | Who | Machine / GPU / driver / torch | What happened (steps reached, problems, numbers) |
|---|---|---|---|
| | | | |

### What is NOT done (so nobody waits on it)
The sampling evaluation (256 EMA samples scored with the structure metrics), `gate_p3.py` and the schedule decision (tie goes to
cosine) are written by the original developer **while your runs go**; the metrics they use already exist in `eval/structure.py`.
After your 15k-step checkpoints arrive, the group picks the winning schedule and starts Phase 4 (text conditioning). Not built
yet at all: asset fetch, game, playability checker, prompt normalizer, Streamlit app, report.

---

## 2026-11-20 — pre-training phase built (Days 0-4 of the plan, ML side), no model trained

### Done
- **Day 0 (partly):** repo scaffold per doc 05; Python 3.12.16 + torch 2.15.0 (CPU) environment; `scripts/verify_env.py`;
  pinned `requirements*.txt`; `env/lock_cpu_dev.txt`; machine/base/overlay configs with the allow-list guard; ledger V1/V2/V4
  updated with what could be verified.
- **Day 1:** pinned, hash-verified data fetch (`scripts/fetch_data.py`); tile table; level-chain reconstruction (42 levels,
  names from VGLC, byte-for-byte match); audit (`reports/eda/AUDIT.json`, **39/39** facts pass); caption detector written from the
  grammar, **reproduces 7,687/7,687 stored captions exactly**; token lengths measured (max 40 <= 48).
- **Day 2:** level-grouped split with near-duplicate families, regenerated and identical to the doc 10 prototype; `LOCK`; reused
  segments; P-dev 115 / P-test 137 / P-rand 100 prompt sets; train-only clamp table; GPU-resident dataset with the test guard;
  stateless sampler; caption augmentation (train-half lexicon only); renderer; **gate G-P1 passes**; real-data references
  (`reports/references/REFERENCES.json`) measured before any model exists.
- **Day 3 (ML side):** float64 linear/cosine schedules, DDPM/DDIM/renoise, three losses; **gate G-P2 passes** (oracle DDIM exact).
- **Day 4 (code only):** U-Net (18,312,013 params; oracle parity), EMA, optimiser groups + LR schedule, atomic checkpoints,
  deterministic validation, trainer + entry point (refuses to start unless gates P1/P2 passed and LOCK verifies), text
  conditioner (frozen MiniLM + adapter), CFG sampler with RePaint hook, structure/novelty/JS metrics.
- Tests: `pytest -q -m "not slow"` -> 139 pass; slow: overfit-one-batch sanity (tiny U-Net, 8 windows, loss < 0.01 in 1,500
  steps) passes.

### In-flight runs
None. **Model training has not been started** (by instruction).

### Blocked / needs another machine
- This machine is the RTX 4050 laptop but has **no NVIDIA driver installed** (nouveau only), so torch sees no CUDA. Not yet possible here: `verify_env` on the GPU machines, min-VRAM measurement,
  `hw_report.json` (it/s, forward timings for the G-P7 pin), the cross-machine resume drill, the CUDA torch wheel check.
- Streamlit Community Cloud checks (Python options, memory limit) need a logged-in browser session.

### Not done yet (still in the plan)
Asset fetch (Phaser, Kenney), game skeleton, playability checker (needed for G-P5b), prompt normalizer + P-casual set (the
paraphrase-invariant test T-A4 waits for it), C-score/adherence, composite score, bootstrap statistics, dev-eval loop inside
the trainer, early stopping, `gate_p3.py`, full-level generator, Streamlit app, report. Floor validity, broken rate, novelty,
copy rate, diversity and tile-JS exist (`eval/structure.py`).

### Next
See the HANDOVER section at the top: the friend runs Phase 3; the original developer writes the sampling evaluation and `gate_p3.py` meanwhile.

### Assumptions
- The CPU development machine stands in for Day 0-4 work; nothing reported as a speed or memory number comes from it.
- The overfit sanity uses `base_channels=16` for speed; the default model's overfit/VRAM check is for the GPU machine.

### Doc conflicts
- Ledger M5 says "32-d heads"; doc 14 §2.2 notes it holds only at 8x8 (4 heads: 32-d at 8x8, 64-d at 4x4). Code follows doc 14.
- Ledger H8's per-stage list omits `conv_in` (7,552); the U-Net total 18,312,013 is correct (doc 14 §3).
- Docs say `docs/03_...`; in this repo they live in `docs/docs/`. `reference/` moved to the repo root (doc 05).
- Detector rule ambiguities in doc 09 §6.2 (platform candidates, pipe edge cases) were resolved by implementing the literal text;
  the result reproduces all stored captions, so no documented exceptions are needed.

### Walk-through
(To be filled by the group: area, presenter, a question nobody could answer, and its answer.)
