# 36 — Team Ownership and Viva Preparation

> The statement requires that **every group member understands the submitted work equally and can explain it**; rubric rows 8 (viva, 10), 9 (live modification, 10) and 11 (demonstration, 10) test this directly, so 30 of 135 marks depend on it. Claude Code writes much of the code; this document makes sure the humans own it. Names are filled in on Day 0.

## 1. Ownership (primary owner explains first; **everyone must still be able to explain and modify everything**)
| Component | Files | Primary owner (A / B / C) | Backup |
|---|---|---|---|
| Data: audit, split, leakage, preprocessing, detector | `data/`, `captions/`, doc 08-12 | C (data/eval) | A |
| Diffusion maths and sampler | `diffusion/`, doc 13, 20 | A (model) | B |
| U-Net, text path, CFG | `models/`, `text/`, doc 14-15 | A | C |
| Trainer and experiments | `train/`, doc 16-19, 23 | A | C |
| Playability checker and game | `playability/`, `app/static/game`, doc 21, 27 | B (app) | A |
| Streamlit app, exports, cloud deployment | `app/`, doc 28-29 | B | C |
| Evaluation, statistics, report, slides, evidence | `eval/`, doc 22, 32, 35 | C | B |
With **two** members: A takes model + checker; B takes data/eval + app; the third column merges accordingly and the daily ritual still covers all components.

## 2. How Claude Code must support this (see CLAUDE.md §6)
Every module carries explanatory comments and a short "explain-me" docstring at the top (what it does, its inputs/outputs, how it connects to the previous and next module, one thing to change to see an effect). Plans stay short enough for 2-3 students to read in a day.

## 3. Daily 20-minute walk-through (from Day 1; one member presents, the others ask)
Rotating schedule so that each member presents each area at least twice before Day 15:
1. Day-of-the-week rotation: **preprocessing -> diffusion code -> model -> app** (5 minutes each, the owner presents on their day but another member is picked at random to present the area the day before).
2. Format: open the actual file; trace one input end to end (a caption string and one 16x16 window): tokenise -> MiniLM -> adapter -> cross-attention; add noise -> predict noise -> loss; sample with CFG -> grid -> checker -> game.
3. Log in `PROGRESS.md` under "Walk-through": area, presenter, one question the others could not answer, and its answer.

## 4. Member-specific viva questions (add to doc 32 §7)
Each member prepares the answers to the doc 32 §7 bank and to: why a level-grouped split (leakage numbers F17)? Why ε-prediction and what does CFG do to the two predictions? What exactly goes into the U-Net (shapes)? How does a typed sentence become the model's input? What does "verified beatable" mean and how was the checker validated? What did you not measure? What is external vs yours (doc 02 §15, C7)? Why are the confidence intervals wide?

## 5. Live-modification drills (rubric row 9)
**Rule: every member has typed and run every drill below at least once before the end of Day 15 (3 Dec)**, on their own laptop, without Claude Code writing the change. Each drill is timed; target < 10 minutes; the member explains the effect. Record in `PROGRESS.md` (drill number, member, time, observation).
| # | Drill | Where | What to observe |
|---|---|---|---|
| 1 | Change the guidance scale w (slider default and the sampler value) | `configs/generate/app.yaml`, `sampler.py` | w = 1 vs 10: adherence vs broken structures |
| 2 | Change the tile vocabulary: merge two tile classes (e.g. both pipe halves) in the tile table and re-render | `data/tiles.py`, renderer | which tests fail and why (F6 is shared) |
| 3 | Swap the noise schedule linear <-> cosine for sampling | `diffusion/scheduler.py`, config | corruption strips; why a trained model must match |
| 4 | Add a prompt rule to the normaliser (a new synonym, e.g. "tubes" -> pipes) | `data/prompts/lexicon.yaml`, `normalizer.py` | canonical caption changes |
| 5 | Change the window width of generation (stride 8 -> 4) | `generate` config | seam metric, time |
| 6 | Add a metric (e.g. share of ground tiles) to the eval suite | `eval/structure.py`, `suites.py` | appears in `metrics.json` |
| 7 | Change the number of DDIM steps (50 -> 20) | config | speed vs quality |
| 8 | Change the number of resampling jumps r (2 -> 0) | config | seam appearance |
| 9 | Change the width options of the app (add 64) | `app/streamlit_app.py`, config | UI and window count |
| 10 | Change the checker's maximum jump rise via `physics.json` and re-derive | `scripts/derive_physics.py` | verdicts change, hash changes |
| 11 | Change dropout in the model config and print the parameter count | `configs`, `models/unet.py` | param count unchanged; behaviour in train mode |
| 12 | Change null-caption probability 0.10 -> 0.3 and run the null-rate test | `aug` config, `test_dataset.py` | test fails/passes |
| 13 | Change a loss weight (aux CE lambda) and run 50 steps | config, `train/loop.py` | loss values |
| 14 | Add a new example prompt chip in the app | `app/` | UI |
| 15 | Add a new export format (e.g. ASCII text with different characters) | `render/renderer.py`, doc 29 | round-trip test |
| 16 | Change the seed handling: show the seed used for window k | `full_level.py` | determinism |
| 17 | Change the number of generated showcase levels or filter on verified | `scripts/make_showcase.py` | manifest |
| 18 | Change the time budget of the cloud fallback and trigger it | `configs/app/cloud.yaml` | fallback label shown |
Additional drills may be added by the group; the minimum is 15 distinct drills completed by each member.

## 6. Demo rule (rubric deduction rule: -20 % without it)
**Each member brings a laptop with the project running in parallel on presentation day (7-9 Dec).** On Day 17 each member sets up the project from the zip on their own laptop, runs the app (local profile, CPU fast mode if no GPU), and opens the cloud URL. A USB copy of the zip and standalone HTML levels is carried as a backup.

## 7. Mock viva protocol (Day 17)
Two rounds; 15 random questions from doc 32 §7 plus 3 member-specific ones; 60 s per answer; one live modification drill chosen at random per person; scorer notes unclear answers; weak areas get the next daily walk-through.
