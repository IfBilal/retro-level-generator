# 31 — Future Features: Reserved Seams

> **Status: MAY (demoted 2026-11-18, doc 02 §14).** None of the nine future features is built in the 18-day plan. Only seams that cost < 30 minutes and fall out of required code are kept; each seam is an interface, a data field or an artifact. Nothing here is a deliverable or a gate. "Report card lite" is cut.

| Seam | Future feature (doc 02 §10) | Where it lives | Cost now | State |
|---|---|---|---|---|
| **S-1** | F1 Watch the proof (ghost bot, redraw stuck spots) | `check_level -> CheckResult(path, fail_col)`; `full_level.resample_window`; `physics.json` shared with the game; `T2M_DEBUG.replay(actions)` in the game (doc 27) | free (needed by G5) | kept |
| **S-2** | Denoising filmstrip / progress | `sampler.sample(..., callback)` returns x0_hat per step (doc 20 §1) | free | kept |
| **S-3** | F3 Erase and re-ask | `generate/full_level.py::inpaint(grid, mask, caption, seed)` reusing the RePaint path | signature only | MAY |
| **S-4** | F4 Whole-level pacing | multi-scene splitter + per-window captions + templated framing (doc 12 §7, doc 20) | free | kept |
| **S-5** | F5 See what it heard (word -> tiles) | detector `return_details`-style output (tiles per phrase), normaliser `NormalizedPrompt.alignment`, `unet` attention-capture hook (off in training) | detector detail output only | MAY |
| **S-6** | F6 Report card | detector, checker and NN-copy check callable per level (`eval/adherence.py`, `eval/novelty.py`) | free | kept; UI cut |
| **S-7** | Negation ("no enemies") | normaliser marks `negated: true` and removes the concept; absence adherence reported separately (doc 22, R35) | rule in normaliser | kept |
| **S-8** | F8 Overnight self-training | `generate_batch` + filters; `split_v1` immutable, a future `selfplay_v1` set is additive (doc 30) | none | MAY |
| **S-9** | F9 Endless mode | `iter_windows(...)` generator in `full_level.py` | free (loop already a generator) | kept |
| **S-10** | F2 Difficulty dial | checker `noise_profile` (fixed 0); level JSON `metrics.difficulty` (null) | field only | kept |
| **S-11** | F7 Blind test (real vs AI) | real and generated levels share schema v1 (doc 29) and the renderer | free | kept |

**Rules.** (1) A seam must not change model outputs or any evaluated metric. (2) Seams are unit-tested only if the test is trivial (signature/field presence). (3) No feature UI before every MUST item of doc 02 is done. (4) Any new feature is a ledger row first (CLAUDE.md §1).
