# Model root cause (H1, H3, H4, H6, H8), measured 2026-11-27 with the adherence planner OFF, frozen model step 18000, fast mode, seeds 1-20
Reproduce: `python scripts/audit2_model_rca.py` (raw numbers: `model_rca.json`); curves from `reports/dev_eval/C-*_p_dev_w5.json`.

**H1 scope mismatch: TRUE (main cause).** The caption describes ONE 16-column window and is repeated on every window. Mean objects for the caption "full floor. one pipe.":
width 16 (1 window) 1.35, width 48 (5 windows) 5.15, width 112 (13 windows) 12.0. "two enemies": 2.85 / 9.4 / 21.5. Counts scale with the number of windows.

**H3 the model cannot count exactly: PARTLY TRUE (loose, not blind).** Width 16, 20 seeds, got-count histograms:
"one pipe" 1:13, 2:7 (65% exact); "two pipes" 2:5, 3:11, 4:4 (25%); "one enemy" 0:8, 1:11, 3:1 (55%); "two enemies" 1:1, 2:7, 3:6, 4:6 (35%);
"a few enemies" (training meaning 3-4) 6:3, 7:8, 8:8, 9:1 (0% in range: the model over-draws). It knows "more or fewer" but not "exactly n".

**H6 CFG: FALSE as a lever.** P(exactly one pipe) at w = 1 / 3 / 5 / 7.5 / 10: 0.65 / 0.75 / 0.75 / 0.65 / 0.70, mean pipes 1.15-1.35. No scale fixes it (n=20 per cell, so differences are noise).

**H4 high bias or high variance: leaning HIGH VARIANCE, not high bias.** Dev, w=5, same prompts, by step (C-main): c_score 0.669 / 0.757 / 0.796 / 0.805 / 0.823 / 0.811 (6k..36k), exact match 0.009 -> 0.099 -> 0.081, copy rate 0.026 -> 0.052 -> 0.096 -> 0.157 -> 0.238 -> 0.267, median novelty 23 -> 12 -> 8 tiles. The C-x01 run keeps going: exact 0.142 and copy rate 0.351 at 40k. Quality gains flatten while copying the 31 training levels keeps growing, so more training buys memorisation, not counting; step 18000 was chosen before the copy rate climbed. Exact-count ability stays low at every step (bias-like), and caption buckets cannot express "exactly 3" at all (a few = 3-4), so a structured count conditioning (H5c) would be the principled model change.

**H5 four-way encoder comparison and any retraining: NOT RUN, deliberately.** The mission's own rule is "model work only if the constraint layer cannot reach the targets". It does (630/630 same seeds, 630/630 fresh seeds, 100/100 held-out prompts), a retrain needs a new FROZEN entry and a post-hoc test, and the dev set is 5 levels. Cost of the layer, measured: levels are sparser and CPU/cloud matching is weaker.

**H7 verification optimises the wrong thing: WAS TRUE, fixed in audit 1** (the loop now checks counts on the raw level; the route checker is still a heuristic, G-P5b fails).

**H8 post-processing changes what is shown: TRUE for 1 level in 15 (before audit2 fix).** "three pipes and 5 enemies" seed 4: raw 3 pipes, displayed 2. The display repair deleted a pipe cut in half by the end-of-level template (6 cells at the last content columns). Fixed: `adherence.evaluate_both` counts the raw AND the displayed content and the worse decides; test `test_counts_are_also_judged_on_what_the_player_sees`. After the fix the same seed shows 3 and raw == displayed on 15/15 levels.
