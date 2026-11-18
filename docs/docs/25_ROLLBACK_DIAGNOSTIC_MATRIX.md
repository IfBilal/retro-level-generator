# 25 — Rollback Diagnostic Matrix

> **Use:** a gate failed or a metric looks wrong. Find the phase block, match the symptom, check the cause with the stated test, apply the **first** action, re-gate. **Golden rule: never train longer to fix a data, scaling or capacity bug** (CLAUDE.md §3). IDs are `R-<phase>.<n>`. Risk register entries (`R<n>`, doc 33) link here. Thresholds come from the ledger §9 and doc 24; none is invented here. "Reference loss" values are recorded in `reports/references/` after Day 4/5 (to be measured).

## 0. General
| ID | Symptom | Likely cause | Check | Action |
|---|---|---|---|---|
| R-0.1 | Gate script cannot find a predecessor `RESULT.json` | Previous gate not run or failed | `ls reports/gates/*/RESULT.json` | Run/fix the previous gate; never edit the file |
| R-0.2 | Loss spike or discontinuity after resume | State not fully restored (EMA, LR, RNG, sampler) or different config hash | `verify_ckpt.py`; compare step, LR, EMA in the log | Restore from the last good checkpoint; fix the missing state; add to `test_checkpoint.py` |
| R-0.3 | Different results on the two machines | Different torch minor, or non-allow-listed machine override | `compare_env.py`, trainer start log | Align versions (ledger V2); remove the override. Cross-GPU bitwise equality is **not** expected (NFR-REP-05) |
| R-0.4 | OOM | Cap 4.5 GiB exceeded | `sys/vram_peak_gib` | Micro-batch + accumulation (logged); never raise the cap |
| R-0.5 | Gate waiver tempted | Deadline pressure | - | Not allowed silently; log `--allow-*` use in `events.jsonl` and `PROGRESS.md`; the one-shot test and split lock are never waived |
Reference note (§0.2 of earlier drafts): per-bucket ε-MSE values are not comparable across schedules (cosine's uniform-t average is larger for the same quality; derivation doc 13). Compare samples and S, not raw loss.

## 3. Phase 3 (unconditional; G-P3)
| ID | Symptom | Likely cause | Check | Action |
|---|---|---|---|---|
| R-3.1 | Samples all sky; floor validity low; tile JS high | Sky dominance; loss ignores rare classes; wrong scaling | tile histogram vs train; confirm {-1,+1} scaling and F6 channel order | fix scaling/order -> aux CE lambda 0.001 (M9) -> channel-weighted MSE (rollback) |
| R-3.2 | Samples look like noise / salt-and-pepper | Under-trained, bad LR, EMA warm-up wrong | EMA vs raw samples; per-bucket loss | check EMA warm-up (M11) and LR schedule; evaluate EMA not raw |
| R-3.3 | NaN / `skipped_nonfinite` > 0 | Division by sqrt(alpha_bar) at large t, bf16 overflow, aux CE weighting | log; fp32 x0_hat clip | ensure fp32 for x0_hat and clipping (doc 13 §14); aux CE weighted by alpha_bar_t; lower LR once; 5 consecutive skips abort |
| R-3.4 | Broken pipes/cannons > 5 % | Weak local structure | broken rate vs step | self-attention present (M5)? EMA eval? aux CE; display-only repair is not a fix for the metric |
| R-3.5 | Floor validator fails on real dev windows | Validator mis-calibrated | run on real dev windows (must pass >= 95 %) | recalibrate the validator, not the model (F22) |
| R-3.6 | Copy rate > reference + 0.05 | Memorisation | doc 19 §3 | early stop earlier; dropout; do not move the threshold |
| R-3.7 | Linear vs cosine disagree | Schedule effect | compare S_noplay, samples (not loss) | tie -> cosine (M2); log in ledger |
| R-3.8 | Diversity ratio < 0.5 | Mode collapse | doc 22 §3.9 | lower w is irrelevant here; check EMA, dropout, dataset sampling; aux CE |

## 4. Phase 4 (conditional; G-P4)
| ID | Symptom | Likely cause | Check | Action |
|---|---|---|---|---|
| R-4.1 | Model ignores the prompt (C-score at null level, `G_b` approx 0) | adapter not in optimiser / null embedding wrong / mask bug | tests T3, T6, T7 (doc 15 §10); attention entropy | fix wiring first; then X3 pooled text (MAY); X4 concept balance (MAY); larger adapter |
| R-4.2 | CFG does not beat w = 1 | text path weak, null dropout rate wrong | null rate 0.10 +/- 0.01; paired CI | verify drop rate and null embedding; then R-4.1 |
| R-4.3 | Counts wrong (quantities) | quantity buckets weak | per-concept exact match | check augmentation did not corrupt counts; concept-balanced sampling (X4, MAY) |
| R-4.4 | Rare concepts fail | imbalance | rare-concept exact match < 50 % | X4 trigger (doc 23) |
| R-4.5 | High C-score but copy rate high | memorisation | doc 19 | early stop; compare with the retrieval baseline |
| R-4.6 | w high -> broken structures rise | over-guidance | broken vs w curve | pick w with the broken term in S (E4); optional x0 simplex projection (doc 13 §7) |
| R-4.7 | Normaliser output misses casual phrases | lexicon gap | P-casual-dev | add synonyms **before** the freeze only |
Unfreezing MiniLM is a ledger-level change (M7); last resort only.

## 5. Phase 5 and playability (G-P5, G-P5b)
| ID | Symptom | Likely cause | Check | Action |
|---|---|---|---|---|
| R-5.1 | Visible seams (ratio > 1.3) | too little overlap/context | seam metric per seam | ladder: stride 4 -> r = 5 -> direct wide sampling (X12) |
| R-5.2 | Pipes broken across seams | multi-tile structure split | sanitizer log | resample the window; raise r |
| R-5.3 | Start/flag area incompatible with generated content | framing mismatch | render joins | adjust blend columns in framing config (templated) |
| R-5.4 | Verified rate low | checker strict or model weak | per-window failure columns | check real-level baseline first; then more attempts (budget 5/20 is fixed by G5) |
| R-5.5 | Checker disagrees with jar/human (< threshold) or timeouts > 1 % | physics too strict/lenient, search budget | `disagreements.md`, confusion matrix, false-"verified" rate | classify each disagreement; tune physics only within doc 27 §4.5 ranges; raise node budget; keep `S_noplay` until the gate passes |
| R-5.6 | A "verified" showcase level is not humanly beatable | checker too lenient or mechanic mismatch | human playthrough | make the arc envelope more conservative; re-run G-P5b; remove the level from the showcase |

## 6. Phase 6, app and deployment (G-P6, G-P7, G-P8)
| ID | Symptom | Likely cause | Check | Action |
|---|---|---|---|---|
| R-6.1 | Dev metrics unstable between runs | seeds/CRN not shared | seed log | fix seeds (doc 22 §2.3) |
| R-6.2 | Level-clustered CI very wide | only 5/6 levels | per-level table | expected; report per-level results and say so (ledger E3) |
| R-6.3 | Tempted to re-run the test | unhappy number | `test_invocations.jsonl` | forbidden; only a documented metric-code bug allows a re-run, reporting both |
| R-6.4 | `eval_test.py` guard fails (hash/lock mismatch, prompts lock opened early) | files changed after the lock | guard message | restore the locked files; never edit the lock |
| R-6.5 | G-P7 latency over target | unmeasured threshold was wrong, or cache/compile issue | `latency.json` | re-pin the threshold per the measure-first rule (ledger G-P7) if the measurement is sound; else profile |
| R-6.6 | Cloud build fails (dependency/size/Python version) | heavy package, CUDA wheel, wrong version | build log | CPU torch index, remove dev deps, match Python version (ledger V1); keep the cached-only fallback ready |
| R-6.7 | Cloud app too slow or out of memory | CPU cores shared; model + encoder RSS | measured RSS/time | cap width, time guard to showcase fallback (CL6), fp16 weights |
| R-6.8 | Game keyboard dead in the cloud iframe | focus/sandbox | doc 27 §cloud iframe | click-to-focus overlay; `st.iframe`/components shim; test on the public URL |
| R-6.9 | Cloud URL down at presentation time | platform sleep/outage | G-P8 re-check | run locally on every member's laptop (primary demo); cloud is the backup |
