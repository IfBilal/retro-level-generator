# PROGRESS

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
- This machine has **no NVIDIA GPU**. Not yet possible here: `verify_env` on the GPU machines, min-VRAM measurement,
  `hw_report.json` (it/s, forward timings for the G-P7 pin), the cross-machine resume drill, the CUDA torch wheel check.
- Streamlit Community Cloud checks (Python options, memory limit) need a logged-in browser session.

### Not done yet (still in the plan)
Asset fetch (Phaser, Kenney), game skeleton, playability checker (needed for G-P5b), prompt normalizer + P-casual set (the
paraphrase-invariant test T-A4 waits for it), C-score/adherence, composite score, bootstrap statistics, dev-eval loop inside
the trainer, early stopping, `gate_p3.py`, full-level generator, Streamlit app, report. Floor validity, broken rate, novelty,
copy rate, diversity and tile-JS exist (`eval/structure.py`).

### Next
1. On a GPU machine: `python scripts/verify_env.py`, then Phase 3 (`uncond_linear` / `uncond_cosine`, 15k steps each).
2. Write `eval/adherence.py`, the normalizer and `gate_p3.py` while those runs go.

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
