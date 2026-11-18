# 10 — Dataset Splits, Cross-Validation and Leakage Control

> **Day:** 2 (Phase 1b). **Gate:** G-P1 (part 2: leakage tests). **Ledger rows:** D3, D4, D5, D6, E2, E4, F17, G-P1.
> **Script:** `scripts/make_splits.py` → `data/splits/split_v1.json`, `data/splits/LOCK`, `data/splits/reused_segments.json`, `data/processed/families.parquet`, `data/prompts/p_dev.json`, `p_test.json`, `p_rand.json`, `reports/splits/SPLIT_REPORT.md`.
> **Inputs:** `data/processed/index.parquet`, `scenes_u8.npy` (doc 09).
> **Principle:** the unit of generalisation is a **level** (and a family of near-duplicate levels), never a window.

---

## 1. Why a random window split is invalid here

Windows are cut at stride 1 (F8), so a window and its neighbour share 15 of 16 columns. Their Hamming distance has median **14 tiles** out of 256 (A13). A random 90/5/5 window split therefore puts, for almost every test window, its immediate neighbours in train. Measured on the upstream split (F17): the 385 test windows span **all 42 levels**, the **median distance to the nearest training window is 8 tiles**, 83.9 % are within 16 tiles, and 2.1 % are exact copies.

Consequences if we used it:
1. **Denoising loss** on "test" equals training loss on near-copies, so overfitting is invisible.
2. **Caption adherence** is inflated: the model can reproduce a memorised neighbour whose caption is nearly identical.
3. **Novelty** is meaningless: a perfect memoriser scores a small NN distance on train and on "test" alike.
4. Any claim of "generates new levels" fails at the viva the moment an examiner asks how the split was made.

So: **split by level**, keep **near-duplicate level families together**, **lock the test set** (D4), and evaluate it **once** (Day 15).

---

## 2. Units of assignment

| Unit | Definition | Count |
|---|---|---|
| Level | One reconstructed chain (doc 09 §2) | 42 (20 SMB1, 22 SMB2) |
| Family | Connected component (size ≥ 2) of the level near-duplicate graph (§3) | 2 in the measured data |
| Assignment unit | A family, or a level that is in no family | 27 + 2 = 29 units |

---

## 3. Near-duplicate families (cross-level leakage)

Different levels reuse layouts (SMB2 reuses SMB1 8-3 segments; SMB1 3-2 / 5-1 / 8-1 share long stretches). Measured: **7.49 %** of all windows have a window **in another level** within Hamming ≤ 2 (A12).

### 3.1 Definition
- `link(a, b)` = share of level a's windows whose nearest window in level b has Hamming ≤ 2.
- Levels a and b are **linked** if `max(link(a, b), link(b, a)) ≥ τ`, with **τ = 0.05** (configurable in `configs/data.yaml: family_link_threshold`).
- **Families** = connected components of the link graph with ≥ 2 levels (transitive closure).

### 3.2 Computation
Hamming for all pairs = `256 − matches`, with `matches = A·Aᵀ` where A is the (7687, 13·256) one-hot matrix in float32. That's 7,687² entries ≈ 236 MB, computed in ~4 s on CPU with PyTorch; use chunks of 1,024 rows on GPU if memory is tight. Store the int16 cross-level NN distance per window (`nn_cross_level` in the index) and the 42 × 42 link matrix in `families.parquet`.

### 3.3 Measured families (τ = 0.05)
| Family | Levels | Windows | Strongest links |
|---|---|---|---|
| **F1** | SMB1 1-1, 3-2, 4-1, 5-1, 6-1, 7-1, 8-1 (7 levels, all SMB1) | 1,459 | 3-2↔5-1 42.8 % / 27.9 %; 3-2↔8-1 31.4 %; 5-1↔8-1 24.6 %; 3-2↔4-1 21.1 %; 1-1↔3-2 17.0 % |
| **F2** | SMB1 8-3, SMB2 1-1, 2-1, 4-1, 4-2, 5-1, 6-1, A-1 (8 levels, cross-game) | 1,662 | 8-3↔SMB2 5-1 27.6 %; 8-3↔SMB2 4-1 24.1 %; 8-3↔SMB2 A-1 22.6 %; 8-3↔SMB2 6-1 22.1 %; 8-3↔SMB2 4-2 19.6 % |

### 3.4 Rule
- **Family members are assigned to train only** (never dev or test). They hold 15 levels / 3,121 windows. This removes the worst leakage at its source and costs nothing in coverage (see §5.3).
- In cross-validation (§8) a family is one indivisible unit.
- **Sensitivity check (measured; report appendix):**
  - τ = 0.10 → families of 5 + 8 levels (F1 loses SMB1 6-1 and 7-1; F2 unchanged).
  - τ = 0.03 → families of **10 + 9 + 2** levels (F1 gains SMB1 3-1, 5-2, 6-3; F2 gains SMB2 2-2; a new family {SMB2 2-3, C-2}). That leaves only one eligible descending-staircase level (SMB1 4-2), so **C6 becomes infeasible** for test and dev together.
  - τ = 0.05 is therefore the strictest threshold that keeps the constraints satisfiable. It is the ledger default; the other values are only reported.

---

## 4. Selection constraints (hard)

Let E = levels in no family (eligible for dev/test): **12 SMB1 + 15 SMB2** in the measured data.

| # | Constraint | Value |
|---|---|---|
| C1 | Test composition | exactly **3 SMB1 + 3 SMB2** levels, all from E |
| C2 | Dev composition | exactly **5** levels from E \ test, either 2 SMB1 + 3 SMB2 or 3 SMB1 + 2 SMB2 |
| C3 | Train | all remaining levels (**31**), including every family member |
| C4 | Test window share | 0.14 ± 0.03 of 7,687 → [0.11, 0.17] |
| C5 | Dev window share | 0.12 ± 0.03 → [0.09, 0.15] |
| C6 | Concept coverage (test, dev) | for **each of the 16 concepts**, ≥ **10 windows** in the split |
| C7 | Concept coverage (train) | each of the 16 concepts present in ≥ **2 train levels** (so no concept is learned from a single level) |
| C8 | Determinism | the same inputs always produce the same split (no wall-clock randomness) |

Rare-concept bottlenecks that make C6/C7 bind (doc 09 §3):
- **Ceilings:** 4 levels (SMB1 1-2, 4-2; SMB2 1-2, 5-2), all eligible.
- **Descending staircases:** 5 levels. SMB1 1-1 (F1), SMB1 8-3 (F2) and SMB2 6-1 (F2) are family members, so the only eligible ones are **SMB1 4-2 and SMB2 2-2**. Test and dev therefore each receive exactly one of them, and train keeps the descending staircases of SMB1 1-1, 8-3 and SMB2 6-1 (3 levels, 60 windows; C7 satisfied).
- **Upside-down pipes:** 6 SMB2 levels. Eligible: SMB2 5-2, 8-1, D-1 (5-1, 6-1, A-1 are in F2).

---

## 5. Objective and search

### 5.1 Distributions compared to the full corpus
- **Concept mass** `c_split ∈ ℝ¹⁶`: number of windows in the split containing each concept, normalised to sum 1.
- **Tile distribution** `t_split ∈ ℝ¹³`: tile counts over the split's **distinct columns** (reconstructed levels, so overlap doesn't bias it), normalised.
- Reference `c_all`, `t_all`: the same over all 42 levels.

### 5.2 Score (lower is better)
```
J(split, target) = JS₂(c_split ‖ c_all) + JS₂(t_split ‖ t_all) + 2 · (share_split − target)²
JS₂(p‖q) = ½ KL₂(p‖m) + ½ KL₂(q‖m),  m = (p+q)/2,  with 0·log 0 := 0 (mask zeros; never add eps to p)
```
`target = 0.14` for test, `0.12` for dev.

### 5.3 Procedure (exhaustive, two-stage, deterministic)
1. **Test:** enumerate all C(12,3)·C(15,3) = 220 × 455 = **100,100** combinations of eligible levels (vectorised: per-level count vectors summed via index arrays). Keep those satisfying C1, C4, C6. Pick the minimum J; break ties by the lexicographically smallest sorted `level_id` tuple.
2. **Dev:** enumerate all 2+3 and 3+2 combinations of the remaining eligible levels (**13,464** in the measured data). Keep those satisfying C2, C5, C6, and C7 (computed on the implied train set). Minimum J; same tie-break.
3. **Train** = everything else.
4. Exhaustive enumeration takes < 1 s and needs no randomness. **Seed 20261117** (D3) is recorded in the split file and used only by the random-search fallback (500,000 sampled combinations, if the family rule is disabled for an ablation and enumeration grows to C(20,3)·C(22,3) ≈ 1.76 M; that is still enumerable but slower), by P-rand generation (§9.3), and by CV tie-breaks.
5. If no feasible combination exists: relax C6 to ≥ 5 windows, then C4/C5 to ± 0.05, **logging each relaxation** in `SPLIT_REPORT.md` and `PROGRESS.md`. Never relax C1–C3, C7 or the family rule.

### 5.4 Example output (prototype run on 2026-11-17) — **Claude Code must regenerate; do not hard-code**
Feasible test combinations: 13,525 of 100,100. Feasible dev combinations: 540 of 13,464.

| Split | Levels | Windows | Share |
|---|---|---|---|
| **Test** | SMB1 1-2, SMB1 3-1, SMB1 6-3, SMB2 2-2, SMB2 B-1, SMB2 D-1 | 1,073 | 13.96 % |
| **Dev** | SMB1 2-1, SMB1 4-2, SMB1 5-2, SMB2 8-1, SMB2 C-2 | 905 | 11.77 % |
| **Train** | the other 31 levels (incl. families F1, F2) | 5,709 | 74.27 % |

Scores: test J = 0.00296, dev J = 0.00805.

Concept window counts per split (example):

| Concept | Train | Dev | Test |
|---|---|---|---|
| floor | 4,602 | 773 | 1,033 |
| ceiling | 274 | 152 | 130 |
| pipe | 1,661 | 322 | 404 |
| upside down pipe | 184 | 85 | 43 |
| coin line | 1,080 | 188 | 266 |
| coin | 1,386 | 218 | 295 |
| cannon | 409 | 74 | 71 |
| question block | 1,037 | 193 | 255 |
| enemy | 4,008 | 716 | 878 |
| platform | 3,300 | 577 | 622 |
| tower | 727 | 158 | 198 |
| ascending staircase | 620 | 123 | 118 |
| descending staircase | 60 | 14 | 30 |
| rectangular block cluster | 540 | 107 | 117 |
| irregular block cluster | 169 | 61 | 57 |
| loose block | 1,823 | 377 | 451 |

Leakage of the example split (NN Hamming to the train split, all windows at stride 1):

| Split | median | ≤ 0 | ≤ 2 | ≤ 4 | ≤ 16 |
|---|---|---|---|---|---|
| Dev | 13 | 0.00 % | 1.66 % | 9.06 % | 65.4 % |
| Test | 14 | 0.19 % | 1.86 % | 5.78 % | 63.7 % |
| *Upstream random test (F17)* | *8* | *2.1 %* | *—* | *—* | *83.9 %* |

Stride-8 prompt counts (example): **P-dev 115**, **P-test 137**. `novelty_ref` (E4) for the example = median NN distance of dev windows to train = **13**.

---

## 6. Leakage tests (`tests/test_splits.py`, part of G-P1)

All tests read `split_v1.json` and the index; all are **hard** unless marked.

| Test | Check |
|---|---|
| T1 disjoint | No `level_id` appears in two splits; union = all 42 |
| T2 families intact | Every family's members are all in train |
| T3 composition | C1, C2, C3 |
| T4 shares | C4, C5 |
| T5 coverage | C6 (test, dev), C7 (train) |
| T6 NN leakage | For dev and test separately, against train: share of windows with NN ≤ 2 must be **≤ 3 %**; exact duplicates **≤ 0.5 %**; median NN **≥ 12** |
| T7 documented reuse | Every dev/test window with NN ≤ 2 to train is listed in `reused_segments.json` (`idx, split, level, nn_idx, nn_level, distance, caption`). The file is regenerated and compared, so a stale list fails the test. This is the "documented reused segments" exception of G-P1. |
| T8 lock | `sha256(canonical_json(split_v1.json) + corpus_content_hash) == LOCK` |
| T9 determinism | Re-running `make_splits.py` in a temp dir reproduces a byte-identical `split_v1.json` |
| T10 no-peek guard | `MarioWindows(split="test")` raises unless `allow_test=True` *and* `FROZEN.json` exists (doc 11 §3) |
| T11 (W) per-game balance | Report per-game window counts per split (no threshold) |

Reused-segment windows **stay** in dev/test (removing them would bias the evaluation toward unusual windows). They are flagged in every metric table, and copy-rate is also reported excluding them.

---

## 7. The LOCK

- After `make_splits.py` passes T1–T9 on Day 2: write `data/splits/LOCK` containing one line, `sha256(canonical_json(split_v1.json) + "\n" + corpus_content_hash)`, and commit both files in one commit `chore(data): lock split_v1`.
- Any code that reads splits verifies the LOCK first (`text2mario.data.splits.load(verify=True)`).
- **Changing the split after Day 2** requires `split_v2.json`, a new LOCK, a ledger change-log row with the reason, and **recomputing every reported number**. "The test numbers looked bad" is never a valid reason (D4).

---

## 8. Level-grouped 5-fold cross-validation (D5)

### 8.1 Purpose
(a) Error bars for the final configuration (mean ± std across folds). (b) Robustness check: does the dev-selected configuration generalise to other held-out level groups? CV runs **never** influence model selection; they run after the freeze (Day 15).

### 8.2 Fold construction (`make_splits.py --cv`)
1. Population: the **36 non-test levels** (train + dev).
2. Units: families (restricted to non-test members) + singleton levels → in the example, F1 (7 levels), F2 (8 levels), plus 21 singletons.
3. Sort units by window count, descending (ties: smallest `level_id`).
4. Greedy longest-processing-time assignment: put each unit in the fold minimising `windows_in_fold + 50 · Σ_game |games_in_fold + games_in_unit − games_total/5|` (window balance plus a game-balance penalty).
5. Record the folds in `split_v1.json["cv_folds"]` (covered by the LOCK).

Example result (prototype; regenerate):

| Fold | Levels | Windows | SMB1 / SMB2 |
|---|---|---|---|
| 0 | F2: SMB1 8-3, SMB2 1-1, 2-1, 4-1, 4-2, 5-1, 6-1, A-1 | 1,662 | 1 / 7 |
| 1 | F1: SMB1 1-1, 3-2, 4-1, 5-1, 6-1, 7-1, 8-1 | 1,459 | 7 / 0 |
| 2 | SMB1 2-1, 3-3, 5-2, 8-2; SMB2 1-2, 1-3, 3-1 | 1,173 | 4 / 3 |
| 3 | SMB1 4-3, 6-2; SMB2 2-3, 3-3, A-3, B-3, C-2 | 1,162 | 2 / 5 |
| 4 | SMB1 1-3, 4-2, 5-3; SMB2 4-3, 5-2, 6-3, 8-1 | 1,158 | 3 / 4 |

**Honest caveat:** folds 0 and 1 are families, so they are game-skewed ("leave-family-out"). That makes CV *harder* than the dev/test protocol. The report shows per-fold results, not only the mean, and states this.

### 8.3 Fold protocol
- Fold k model: trains on the levels of the other four folds with the **frozen final config** (`FROZEN.json`), for a **fixed step budget = the final run's selected step**, and samples with the **frozen w**. No early stopping on fold k (that would select on the evaluation set) and no per-fold tuning.
- Evaluation on fold k's levels: stride-8 prompts, 3 seeds, the full metric suite (doc 22).
- Each fold is an independent run directory `runs/cv_f{k}_<runid>/`, so folds can run on either GPU in parallel (doc 07).
- Report: mean ± std over folds for every metric, the per-fold table, and per-game breakdown.

---

## 9. Evaluation prompt sets (E2)

All prompt files are committed under `data/prompts/` (they contain captions, not level data). Schema per record:
```json
{"pid": "pdev-0042", "set": "P-dev", "prompt": "full floor. two enemies. one pipe.",
 "source": {"idx": 3311, "level": "SMB1 2-1", "col_offset": 64},
 "expected_caption": "full floor. two enemies. one pipe.",
 "flags": {"reused_segment": false, "oov_phrases": []}}
```

### 9.1 P-dev and P-test
- Windows of dev (test) levels with `col_offset % 8 == 0` (D6), per level, in level order.
- `prompt = expected_caption = stored caption` (generator order; the model was trained with shuffled phrase order, so order is irrelevant).
- Example sizes: 115 (dev), 137 (test).
- The matching real windows are the per-prompt references for novelty (`novelty_ref`, E4) and the reference distribution for tile JS divergence.

### 9.2 Full-level references
The 5 dev and 6 test **full-width levels** (reconstructed, doc 09 §2.2) are the references for whole-level playability (doc 21) and seam statistics (doc 20).

### 9.3 P-rand (100 random captions, paper protocol)
Re-implementation of the paper's random-caption grammar (read from `captions/caption_generator.py`, MIT; re-implemented, not imported):
1. Topic inventory: 16 topics (floor, ceiling, upside down pipe, pipe, coin line, coin, platform, tower, cannon, ascending staircase, descending staircase, rectangular, irregular, question block, loose block, enemy), each with the paper's phrase list.
2. k ~ Uniform{1, …, 10}; draw k distinct topics uniformly; for each, one phrase uniformly from its list.
3. If "coin line" was drawn and "coin" was not, append a random coin phrase (so k can reach 11).
4. Shuffle the phrases; join with ". " and a final ".".
5. RNG: `numpy.random.default_rng(20261117)`. The captions differ from the paper's file, but the protocol is identical.

**OOV flag:** the paper's phrase inventory contains three variants **never seen in training**: "ceiling with a few gaps", "several upside down pipes", "many loose blocks". In the paper's own file `Mar1and2_RandomTest-regular.json`, **23 of 100 captions** contain at least one of them (24 phrase occurrences). Every P-rand record lists its OOV phrases; metrics are reported on **all**, **in-vocabulary only**, and **OOV-containing** subsets.

**P-rand-paper (optional, comparison only):** the paper's exact 100 captions (read from `data/source/Mar1and2_RandomTest-regular.json`) are evaluated with the same metrics for the paper-comparison table in the report.

### 9.4 P-casual (60 handwritten prompts)
Defined in full in `12_CAPTION_GRAMMAR_AND_PROMPT_NORMALIZER.md` §8 (30 dev / 30 test, with expected attributes). The test half is evaluated once, on Day 15, together with P-test.

---

## 10. `data/splits/split_v1.json` schema

```json
{
  "schema_version": 1,
  "name": "split_v1",
  "created_utc": "2026-11-19T10:02:11Z",
  "seed": 20261117,
  "corpus_content_hash": "6a15264a…de2c",
  "algorithm": {
    "family_link_threshold": 0.05, "family_link_hamming": 2,
    "test": {"smb1": 3, "smb2": 3, "share_target": 0.14, "share_tol": 0.03},
    "dev": {"n": 5, "share_target": 0.12, "share_tol": 0.03},
    "min_concept_windows": 10, "min_train_levels_per_concept": 2,
    "objective": "JS2(concept)+JS2(tile)+2*(share-target)^2", "tie_break": "lexicographic level_id",
    "relaxations_applied": []
  },
  "families": [{"id": "F1", "levels": [0, 5, 7, 10, 13, 16, 17]}, {"id": "F2", "levels": [19, 20, 23, 28, 29, 31, 33, 36]}],
  "levels": [{"level_id": 0, "name": "SMB1 1-1", "game": "SMB1", "windows": 187, "split": "train", "family": "F1"}],
  "splits": {"train": [0, 2, 5, 6, 7, 9, 10, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 36, 37, 39],
             "dev": [3, 8, 11, 35, 40], "test": [1, 4, 15, 24, 38, 41]},
  "window_counts": {"train": 5709, "dev": 905, "test": 1073},
  "scores": {"test_J": 0.00296, "dev_J": 0.00805},
  "leakage": {"dev": {"nn_median": 13, "le2": 0.0166, "eq0": 0.0}, "test": {"nn_median": 14, "le2": 0.0186, "eq0": 0.0019}},
  "cv_folds": [[19, 20, 23, 28, 29, 31, 33, 36], [0, 5, 7, 10, 13, 16, 17], ["…"]],
  "prompt_sets": {"P-dev": 115, "P-test": 137, "P-rand": 100, "P-casual-dev": 30, "P-casual-test": 30}
}
```
(The level ids shown match the example split of §5.4: dev = SMB1 2-1 (3), 4-2 (8), 5-2 (11), SMB2 8-1 (35), C-2 (40); test = SMB1 1-2 (1), 3-1 (4), 6-3 (15), SMB2 2-2 (24), B-1 (38), D-1 (41). Illustrative only; regenerate.)

---

## 11. `reports/splits/SPLIT_REPORT.md` (auto-generated)
Contents: the constraint table with observed values; chosen levels with per-level stats; the concept table (§5.4 format); per-game counts; NN-leakage table plus CDF figure overlaid on the upstream split's curve (EDA-12); the families graph (EDA-13); the CV fold table; prompt-set sizes and OOV statistics; relaxations applied (expected: none).

## 12. Day-2 exit checklist (this doc's share of G-P1)
- [ ] `make_splits.py` passes T1–T9; `SPLIT_REPORT.md` generated.
- [ ] `split_v1.json`, `LOCK`, `reused_segments.json`, `p_dev.json`, `p_test.json`, `p_rand.json` committed in one commit.
- [ ] Clamp table (doc 12 §5.4) recomputed on the **train split** and committed to `data/prompts/clamp_table.json`.
- [ ] Ledger change log updated if any example value above changed materially (expected: the regenerated split equals the example, since the procedure is deterministic and the data is pinned).
