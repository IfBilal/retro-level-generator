# 19 — Overfitting, Underfitting, Bias and Variance Diagnostics

> **Purpose:** decide, from logged evidence, whether a run is underfitting, overfitting/memorising, or healthy, and which remedy applies (doc 25). Metric names follow doc 22 and doc 30 §4.1. **No threshold below is a measured value;** thresholds marked "calibrate Day N" are fixed after the named measurement and then written in the ledger or `reports/references/`.

## 1. Evidence available
| Evidence | Source | Day |
|---|---|---|
| train vs val loss per t-bucket (10 buckets) | `train/loss_t*`, `val/loss_t*` (deterministic noise, doc 16 §10) | every 2k steps |
| conditioning gain `G_b` | doc 16 §10 | conditional runs |
| copy rate and novelty vs the **real held-out reference** | `dev/copy_rate`, `dev/novelty_med`; reference from doc 22 §3.7-3.10 (F22: real copy rate about 0.175 by leave-one-level-out; novelty median 13 in the example split; to be re-measured on the real split on Day 2) | every eval |
| diversity ratio | doc 22 §3.9 | every eval |
| capacity probe | half-width model, 5,160,621 parameters (doc 14 §7) | only if triggered |

## 2. Underfitting (high bias)
Signs (all on the **same schedule**): `train/loss_t*` and `val/loss_t*` both high and still falling at the end of the budget, or both flat in the mid-SNR buckets while `val` is close to the context-free reference (doc 25 §0.2); floor validity and C-score low; samples look noisy or all sky.
Not underfitting: a high loss at high-t buckets alone (low SNR errors count little; doc 13 §5).
Remedies in order (doc 25 R-3.x): verify scaling and tile order (F6) -> check EMA warm-up -> aux CE (M9) -> channel-weighted MSE (rollback) -> more capacity (X8 is cut; use the base-width knob only if the ledger is amended) . **Never first: train longer.**

## 3. Overfitting and memorisation (variance)
Signs: `train/loss_t*` falls while `val/loss_t*` rises or stalls, **gap growing for >= 3 consecutive evaluations**; `copy_rate` above `copy_rate_ref + 0.05` (G-P3 rule); `novelty_med` below the real-dev reference; high C-score with low novelty. The **retrieval baseline** (doc 22 §2.3: copy rate about 1) is the upper reference.
Remedies: early stopping on S (already, patience 5; S includes novelty); dropout 0.1 -> 0.2 (X9, MAY-trigger); stronger augmentation; fewer steps; a smaller budget. Do not lower the copy threshold to hide it.
Important caveat: the training set is 31 levels, stride-1 windows, so adjacent windows are near-duplicates; a small val gap is expected and does not prove generalisation. Only **held-out levels** (dev) count.

## 4. Mode collapse and noise
Diversity ratio < 0.5 -> mode collapse (R-3.8). Ratio > 2 with a rising broken-structure rate -> noise, not creativity (doc 22 §3.9).

## 5. Bias-variance reading for the report
Report, per final config: the per-bucket train/val curves, the copy-rate/novelty plot vs steps with the real reference lines, and the per-level table (variance across held-out levels; doc 22 §7.3). State plainly that with 5 dev levels the variance estimate is itself noisy.

## 6. Decision table
| Observation | Diagnosis | Action |
|---|---|---|
| train and val high, falling | underfit / budget | continue within M12 budget; if flat -> R-3.x |
| train low, val up for >= 3 evals, copy high | overfit | early stop at best S; X9 if time |
| train and val low, copy within reference, C-score at target | healthy | proceed |
| `G_b` approx 0 | text ignored | R-4.x |
| val noisy between evals | non-deterministic validation | check fixed (window, t, eps) triples |
