# 22 — Evaluation Protocol & Metrics

> **Scope:** every number that appears in a gate, a model-selection decision, the report or the viva. Ledger rows implemented here: **E1–E4**, D4, D6, G4, M11, M13, §9 thresholds.
> **Code:** `src/text2mario/eval/{adherence.py, structure.py, stats.py, references.py, report.py}` · scripts `scripts/eval_dev.py`, `scripts/eval_test.py`, `scripts/compute_references.py`, `scripts/make_prompt_sets.py`, `scripts/make_results.py`.
> **Calibration:** all "reference" values below were **measured on the real corpus on 2026-11-17** (7,687 windows, 42 levels, with the chain-reconstructed level IDs of ledger F10) using a clean re-implementation of the paper's rules. Where the final split is needed, the pack value uses a **leave-one-level-out (LOLO)** approximation, and Claude Code recomputes it with `split_v1` on Day 2 (`compute_references.py`).

---

## 0. Non-negotiable principles

1. **Dev decides, test reports.** Checkpoints, CFG scale, thresholds and ablations are chosen on dev prompt sets only (CLAUDE.md §2.2). Test prompts are generated for and scored **once** (§6.5).
2. **EMA weights, DDIM 50, η = 0, x̂₀ clipped to [−1, 1], CFG `ε̂ = ε_u + w(ε_c − ε_u)`** (M11, M13) for every evaluation, unless an experiment varies exactly one of these (doc 23).
3. **Measure raw samples.** Metrics are computed on the argmax-decoded sample **before** the sanitizer repairs anything (G4). Repaired variants may be reported only as an extra, clearly labelled column.
4. **Same detector everywhere.** Captions of generated content come from `src/text2mario/captions/detector.py`, the Day-1 port that reproduces 100 % of stored captions (timeline Day 1). A metric is never computed with a different captioner.
5. **Every metric has a real-data reference** (§4). A number without its reference line is not reported.
6. **Common random numbers.** Two arms compared on the same prompt set use **identical initial noise** x_T per (prompt, replicate) (§2.3). Differences then reflect the treatment, not the noise.
7. **Uncertainty always.** Every headline number carries a 95 % CI (§7). CV numbers carry mean ± std over folds.
8. **Test levels are invisible to models until Day 15.** Data-construction scripts (`make_splits.py`, leakage tests, gate P1) may read test-level *tiles* to validate the split; no model generates for, or is scored against, a test-derived prompt before `scripts/eval_test.py` runs.

---

## 1. Objects, notation, decoding

| Symbol | Meaning |
|---|---|
| `G ∈ {0..12}^{16×W}` | decoded grid: `G = argmax_c x̂₀[c, :, :]` after the last DDIM step (tile IDs per ledger F6) |
| scene sample | `W = 16`, one window, prompt = one canonical caption |
| level sample | framed full level (`W = 3 + W_gen + 6`, G2–G3); framing columns are **excluded** from every content metric |
| `cap(G)` | detector caption of a 16-wide window (regular captions, no locations, no absence phrases) |
| `T` | the **18 C-score topics**, in this order: `floor, ceiling, broken pipe, upside down pipe, pipe, coin line, coin, platform, tower, broken cannon, cannon, ascending staircase, descending staircase, rectangular, irregular, question block, loose block, enem` (16 concepts + 2 "broken" topics) |
| `T16` | the 16 concepts (T minus the two broken topics) |
| `Q` | quantity order `[one, two, a few, several, many]`, index ι ∈ {0,…,4} (F15) |
| `Train` | all train-split windows at stride 1 (~5.5k incl. duplicates, D3) |
| `H(a, b)` | Hamming distance between two 16×16 grids = number of differing cells (0…256) |

---

## 2. Prompt sets, sample counts, seeds (E2, E3)

### 2.1 Inventory

| Set | Source | Prompts (expected) | Samples/prompt | Used for | Built |
|---|---|---|---|---|---|
| **P-dev** | captions of dev-level windows with `col_offset % 8 == 0` (D6) | ≈ 117 (5 levels × ~23.3; range 85–150 depending on the chosen levels) | 3 | checkpoint selection, CFG sweep, ablations, gates P4/P6 | Day 2 |
| **P-test** | same rule on test levels | ≈ 140 (6 × ~23.3) | 3 | one-shot test | Day 2 (sealed) |
| **P-rand-dev** | 100 random captions, seed **20261117** | 100 | 3 | dev robustness to unusual combinations | Day 2 |
| **P-rand-test** | 100 random captions, seed **20261118** | 100 | 3 | one-shot test | Day 2 (sealed) |
| **P-casual-dev / -test** | handwritten (doc 12) | 30 / 30 | 3 | normalizer + end-to-end casual prompting | Day 2 |
| **P-uncond** | none (null context) | — | 256 samples | gate P3, unconditional baseline B1 | — |
| **P-div** | 20 P-dev prompts, chosen by seeded draw (seed 7) | 20 | 10 | diversity probe (the PDF's "10 seeds, same prompt") | Day 2 |
| **L-single-dev / -test** | 50 P-dev (resp. P-test) captions as whole-level prompts | 50 | 2 levels at W_gen = 112 | full-level metrics (§3.14), gate P5 | Day 11 / Day 15 |
| **L-multi-dev** | 25 multi-scene prompts: 2–4 consecutive P-dev captions of the same dev level joined with `" then "` | 25 | 2 | secondary full-level metric (needs `scene_spans`, §3.14) | Day 11 |

Sample counts per full evaluation: P-dev ≈ 350 scenes, P-test ≈ 420, P-rand 300, P-casual 90. These give a 95 % CI half-width of about ±0.03 on C-score and ±3 pp on proportions near 0.9 (§7.6).

### 2.2 Construction rules

- **P-dev / P-test:** one prompt per selected window. Record schema (`data/prompts/p_dev.jsonl`):
  `{"prompt_id": "dev-L12-o0040", "caption": "<stored caption>", "level_id": 12, "game": "SMB2", "col_offset": 40, "source_idx": 4567}`.
  The caption is used **verbatim** (canonical grammar). Phrase order is shuffled with `random.Random(hash(prompt_id))` so that order is not an accidental cue (training also shuffles, D8).
- **P-rand (paper protocol, our seed):** for each of 100 captions: draw `k ~ Uniform{1,…,10}` topics without replacement from **T16**. For each topic, draw a phrase uniformly from the canonical phrases observed for that topic **in train-split captions** (so every phrase is grammatical and in-vocabulary). Shuffle the phrase order, join with `". "`, end with `"."`. Never emit "broken" phrases. Physically contradictory combinations are allowed (as in the paper): this set measures composition, not realism. Expect low scores (the paper's best random-caption C-score was < 0.5).
- **P-casual:** item format, gold canonical caption and expected-attribute dicts are defined in doc 12. Scoring is in §3.12.
- **Locks:** `scripts/make_prompt_sets.py` writes `data/prompts/PROMPTS_LOCK` = sha256 over (`p_test.jsonl`, `p_rand_test.jsonl`, `p_casual_test.jsonl`) on Day 2. `tests/data/test_splits.py::test_prompts_lock` fails if the files change. Test-side prompt files may be *created* on Day 2 but are **not opened by any model-facing code** before Day 15.

### 2.3 Seeds and common random numbers

```
sample_seed(set, prompt_index, replicate) = SET_BASE[set] + 1000 * prompt_index + replicate
SET_BASE = {P-dev: 1_000_000, P-rand-dev: 2_000_000, P-casual-dev: 3_000_000, P-div: 4_000_000,
            P-uncond: 5_000_000, L-single-dev: 6_000_000, L-multi-dev: 7_000_000,
            P-test: 9_100_000, P-rand-test: 9_200_000, P-casual-test: 9_300_000, L-single-test: 9_400_000}
replicate ∈ {0, 1, 2}   (P-div: 0..9)
```
`x_T = torch.randn(shape, generator=torch.Generator(device).manual_seed(sample_seed))`, drawn in **fp32 on the generating device**. Every stochastic step (e.g., RePaint resampling noise) draws from the same per-sample generator, in a fixed order. **Batching must not change the noise:** draw each sample's x_T separately, then stack. Cross-machine note: the same seed gives the same x_T on the same device type and torch version, but different GPUs may differ in the last bits of the network outputs. Comparisons are always made within one machine/run (doc 07).

---

## 3. Metric catalogue

Each entry gives the definition, formula, range and direction (↑ = higher is better), unit of analysis, implementation notes, reference value, and pitfalls.

### 3.1 C-score — caption adherence (paper replica) ↑

**Definition (bit-exact replica of the paper's `compare_captions`; our code is a clean rewrite, MIT-attributed):**

1. **Phrase extraction** φ(c): split the caption on `"."`, strip, drop empty phrases. For each topic τ in the order of **T**, pick the **first** phrase (in caption order) that contains τ as a **substring** and has not already been taken by an earlier topic. If that phrase starts with `"no "`, the topic is absent (only relevant to absence captions). Topic order matters: `broken pipe` and `upside down pipe` come before `pipe`, and `coin line` before `coin`.
2. **Plural normalisation** ν(p): replace `"enemies"` → `"enemy"`, then remove a trailing `"s"` from every word that ends in `s` but not `ss`.
3. **Quantity** qu(p): the first term of Q, in Q's order, that is a substring of ν(p); ⊥ if none.
4. **Per-topic match** for prompt phrase `a = φ(prompt)[τ]` and generated phrase `b = φ(cap(G))[τ]`:

| Case | Score |
|---|---|
| a = ∅ and b = ∅ | **+1** |
| exactly one of a, b is ∅ | **−1** |
| ν(a) = ν(b) | **+1** |
| qu(ν(a)) ≠ ⊥ and qu(ν(b)) ≠ ⊥ | **1 − \|ι(qu a) − ι(qu b)\| / (\|Q\| − 1)** = 1 − \|Δι\|/4 |
| otherwise (phrases differ, at most one has a quantity) | **0.1** |

5. **C(prompt, G) = (1/18) · Σ_{τ∈T} match_τ**, range [−1, 1]. The set-level C-score is the mean over samples.

**Golden vectors** (must be reproduced exactly by `tests/eval/test_cscore.py`; computed with the replica on 2026-11-17):

| Prompt | Generated caption | C |
|---|---|---|
| `full floor. one enemy.` | `full floor. one enemy.` | 1.000000 |
| `full floor. one enemy.` | `one enemy. full floor.` | 1.000000 |
| `full floor. one enemy.` | `full floor. two enemies.` | 0.986111 |
| `full floor. a few enemies. one pipe.` | `floor with one gap. a few enemies.` | 0.838889 |
| `floor with one gap. two enemies. one platform. one tower.` | `giant gap with one chunk of floor. two enemies. one platform. one tower.` | **1.000000** (quirk Q2) |
| `full floor. several coins.` | `full floor. many coins. one broken pipe.` | 0.875000 |
| `a few platforms.` | *(empty caption)* | 0.888889 |
| `full floor. one coin line. a few coins.` | `full floor. two coin lines.` | 0.875000 |
| `full floor. many enemies.` | `full floor. one enemy.` | 0.944444 |
| `one upside down pipe. full floor.` | `full floor. one pipe.` | 0.777778 |
| `full floor. one ascending staircase.` | `full floor. one descending staircase.` | 0.777778 |
| `full floor. two pipes.` | `full floor. one broken cannon. two pipes.` | 0.888889 |

Symmetry: C(a, b) = C(b, a) held for 20,000 random real caption pairs. This is a property test.

**Known quirks (replicated on purpose for paper parity; reported, not fixed):**
- **Q1 – absent/absent credit.** Topics missing from both captions earn +1, and a typical caption mentions only ~4 of 18 topics. Scores therefore saturate: an *empty* scene scores 0.889 against `a few platforms.` → always read C-score next to its null ladder (below) and the per-concept metrics (§3.3).
- **Q2 – floor quantity conflation.** `floor with one gap` vs `giant gap with one chunk of floor` gets full quantity credit (both contain "one"). The strict per-concept exact match (§3.3) separates them.
- **Q3 – broken topics.** Prompts never contain broken phrases, so each broken topic present in the output costs 2/18 = 0.111. C-score therefore already penalises broken structures.

**Calibration ladder (real data, measured):**

| Generator / pairing | Mean C-score |
|---|---|
| Realistic scene that **ignores the prompt** (real caption vs a random real caption from another level) — **the null** | **0.506** (median 0.514; p5 0.172, p95 0.825) |
| Always-empty scene | 0.552 |
| Always "full floor." | 0.602 |
| Always the modal caption (`full floor. a few enemies.`) | 0.642 |
| Same level, window shifted by 8 columns | 0.785 |
| Gate G-P4 threshold (dev, best w) | **0.75** |
| Paper's real-caption models (leaky split, F17) | > 0.9 |
| Perfect adherence | 1.000 |

Reading: G-P4's 0.75 sits about +0.24 above the null and +0.11 above the strongest constant generator. **Diagnostic reported with every C-score:** `ΔC_null = C − C_null(set)`, where `C_null(set)` is recomputed for that prompt set by `compute_references.py` (P-rand has a lower null because its captions mention more topics).

### 3.2 Whole-caption exact match ↑
`EM(prompt, G) = 1[{ν(p) : p ∈ phrases(prompt)} = {ν(p) : p ∈ phrases(cap(G))}]` (set equality after normalisation; order-free). It is the paper's "perfect match". Null (random cross-level real pairs): **0.11 %**. This is harsh and is reported as a secondary metric.

### 3.3 Per-concept metrics (T16; the broken topics are reported in §3.4)
For concept k, over the samples of a prompt set:
- **Presence:** `req_k` = the prompt has a phrase for k; `gen_k` = cap(G) has a phrase for k. TP/FP/FN/TN over samples → **precision_k, recall_k, F1_k**.
- **Strict exact match** `EM_k` = share of samples with `req_k` whose generated phrase satisfies `ν(gen) == ν(req)` (no Q2 conflation).
- **Absence respect** `AR_k` = share of samples with ¬`req_k` and ¬`gen_k`. This is meaningful for P-dev/P-test, whose captions are exhaustive, but **not** for P-casual (users omit what they don't care about).
- **Bucket error** for countable concepts, when both phrases have a quantity: `MAE_k = mean |ι(qu gen) − ι(qu req)|` and the signed **bias_k = mean (ι(qu gen) − ι(qu req))` (> 0 means over-generation).
- **Support** = number of prompts with `req_k`. Concepts with support < 5 in a prompt set are reported as "n/a (support s)" and are excluded from averages.
- **Rare-concept exact match** (trigger for X4, D9) = support-weighted mean of `EM_k` over the **rare set** R = {rectangular, ceiling, cannon, upside down pipe, irregular, descending staircase} (prevalence < 10 %, table below).

**Reference table (real data):** prevalence = share of the 7,687 windows containing the concept. The **null EM** is the probability that a realistic window from another level has exactly the requested phrase, given the prompt requests the concept. The **null presence** is the probability that it has the concept at all.

| Concept | Prevalence | Null EM | Null presence | | Concept | Prevalence | Null EM | Null presence |
|---|---|---|---|---|---|---|---|---|
| floor | 83.4 % | 0.233 | 0.830 | | question block | 19.3 % | 0.052 | 0.191 |
| enemy | 72.9 % | 0.239 | 0.734 | | tower | 14.1 % | 0.097 | 0.144 |
| platform | 58.5 % | 0.184 | 0.579 | | ascending staircase | 11.2 % | 0.094 | 0.105 |
| loose block | 34.5 % | 0.146 | 0.349 | | rectangular cluster | 9.9 % | 0.055 | 0.098 |
| pipe | 31.1 % | 0.157 | 0.311 | | ceiling | 7.2 % | 0.026 | 0.046 |
| coin | 24.7 % | 0.073 | 0.236 | | cannon | 7.2 % | 0.037 | 0.076 |
| coin line | 20.0 % | 0.137 | 0.193 | | upside down pipe | 4.1 % | 0.026 | 0.035 |
| | | | | | irregular cluster | 3.7 % | 0.022 | 0.029 |
| | | | | | descending staircase | **1.35 %** (104 windows) | 0.013 | 0.013 |

`descending staircase` may have < 5 dev prompts. If so it is "n/a" on dev, and on test it is reported with its tiny support and a warning.

### 3.4 Broken-structure rate ↓ (G-P3, G-P5, G4)
**Validators (exact replica of the paper's detector):**
- **Pipe structures** = 4-connected components of pipe tiles {`<`, `>`, `[`, `]`}, where the flood fill **never moves right from `>`/`]` and never moves left from `<`/`[`** (so side-by-side pipes stay separate). A component is **valid** if `valid_pipe(min_row, min_col)` (upright: cap `<>` on top, then any number of `[]` or `<>` rows, ending at a non-pipe tile or the bottom edge) **or** `valid_upside_down_pipe(max_row, min_col)` (mirror image; sky directly above the body counts as invalid). **Window-edge exemption** (scene mode only): a lone `>` cap at column 0, or a lone `<` cap at column W−1, with a consistent body below it, is valid, because the window cropped a real pipe. In level mode the exemption applies only at the true level boundary, which framing makes pipe-free.
- **Broken cannon** = any `b` tile (cannon support) whose cell directly above is neither `B` nor `b`, or which sits in row 0.
- **Metrics:** `broken_rate` = share of samples with ≥ 1 broken pipe component **or** ≥ 1 broken cannon (primary; the gates use this). `broken_pipe_struct_rate` = broken pipe components / all pipe components (secondary; the paper's "% of feature"). `broken_cannon_rate` = share of samples with ≥ 1 broken cannon.
- **Real reference:** **0 broken pipes in 3,852 pipe components and 0 broken cannons across all 7,687 windows.** Any broken structure is therefore a model defect.

### 3.5 Floor-pattern validity ↑ (G-P3)
The ledger's G-P3 text is "bottom row ≥ 50 % solid **or** a legal gap pattern". Operational definition, calibrated so that real windows pass:

```
FLOOR_ALPHABET = {sky '-', breakable 'S', ground 'X', pipe body '[', pipe body ']'}   # the only tiles ever seen in row 15
SOLID = {<, >, ?, B, Q, S, X, [, ], b}
valid_floor(G) =  all(G[15, c] ∈ FLOOR_ALPHABET for c)                       # (i) no coins/enemies/caps/cannons in the floor row
              and ( count(G[15, :] ∈ SOLID) ≥ 8                              # (ii-a) ≥ 50 % solid
                    or any(G[r, c] ∈ SOLID for r in 2..14, c) )             # (ii-b) legal gap pattern: the window is not a void
```
- **Real pass rate: 99.91 %** (void windows 0.09 %). Floorless windows (row 15 entirely passable) make up **16.64 %** of real data, and giant-gap windows 14.6 %. Both are legal SMB design, which is why "≥ 50 % solid" alone (71.0 % of real windows) cannot be the rule.
- **Diagnostic (non-blocking):** `floorless_share` vs the real 16.6 %. A share > 25 % or < 8 % is flagged as distribution drift, in the same report as JS (§3.10).
- Full levels: the rule is applied per 16-wide window at stride 8, excluding the framing columns.

### 3.6 Playability rate ↑
- **Tier 1 (fast checker, doc 21):** `playable = share of samples the checker solves`. **Scene mode** (16-wide): start and goal are defined in doc 21 (spawn on the left, reach the right edge). **Level mode** (framed): spawn in the 3-column start zone, goal = the flag column. Verdicts are `solved | unsolved | timeout`. **Timeouts count as unsolved** and are also reported separately (`timeout_rate`; it must stay < 1 %, otherwise raise the search budget, doc 25 R-5.5).
- **Tier 2 (MarioEval.jar, Baumgarten A\*):** used on the 300-level agreement subset (G-P5b, §6.5) and for paper-parity numbers on test. Never inside the training loop.
- **References:** paper's A\* on real windows = **93.1 %** (7,160 / 7,687, F18). Our checker on real windows (scene mode) is measured on Day 5 and stored in `REFERENCES.json` (expected high-80s to low-90s %). Real **full** levels should be ~100 % solvable; any real level the checker fails is a checker false negative and must be investigated before G-P5b.
- **Level-mode extras (G5 loop):** `verified_rate` (badge = verified after ≤ 20 attempts), `attempts_mean`, `attempts_p90`.

### 3.7 Novelty — nearest-neighbour Hamming distance to Train ↑ (capped)
`nn(G) = min_{X ∈ Train} H(G, X)`. Reported: **median** (primary, `novelty_med`), mean (= the paper's AMED_real), p10, p90, and the share of exact copies (`nn = 0`; the paper's "perfect matches").

**GPU implementation (exact integer arithmetic):**
```python
# Gq: (n,16,16) uint8 generated, Xt: (m,16,16) uint8 train — on the GPU
A = F.one_hot(Gq.long().flatten(1), 13).flatten(1).float()   # (n, 3328)
B = F.one_hot(Xt.long().flatten(1), 13).flatten(1).float()   # (m, 3328), cached once per split
eq = A @ B.T                                                  # equal-cell counts, exact in fp32 (≤ 256)
nn, arg = (256 - eq).min(dim=1)                               # chunk n by 1024 if needed
```
Cost: m ≈ 5.5k → `B` = 73 MB, which is within the VRAM cap. Keep `arg` to render "closest training window" side-by-side figures. For level samples, compute `nn` for every 16-wide window at stride 8 (excluding framing). Report the level's median and its minimum (`level_max_copy = min over windows`).

**Natural reference** `novelty_ref` = median nn of **real dev windows (stride 8)** to Train. LOLO pack estimate: **median 10, mean 12.85, p10 3, p90 24** (978 queries). It varies strongly by level (per-level medians 2 to 56.5), which is why §7.3 reports per-level numbers.

### 3.8 Copy rate ↓
`copy_rate = share of samples with nn ≤ 4` (ledger E1; the threshold is inclusive). Also reported: `copy_rate_strict` (nn ≤ 2) and `exact_copy_rate` (nn = 0). Complexity-stratified copy rates use complexity = number of non-sky tiles in rows 0–14.

**Natural reference (real held-out windows, LOLO, stride 8):**

| Subset | Share | copy_rate (≤ 4) | ≤ 2 | exact | median nn |
|---|---|---|---|---|---|
| all | 100 % | **17.5 %** | 8.9 % | 1.6 % | 10 |
| complexity ≥ 12 | 74.8 % | 12.3 % | — | — | 13 |
| complexity ≥ 24 | 35.4 % | 7.2 % | — | — | 18 |
| SMB1 / SMB2 | — | 21.4 % / 14.0 % | — | — | 10 / 10 |

> ⚠ **Ledger calibration issue (reported to the ledger owner, not silently changed).** G-P3 requires copy rate ≤ 10 %. But *real, never-seen* windows already have a natural copy rate of ~17.5 % (LOLO). Generic segments such as "full floor + a few enemies" recur across levels, and an empty-ish window is within 4 tiles of many training windows. A perfect generator of the real distribution would therefore **fail** G-P3. **Pre-registered procedure:** on Day 2, `compute_references.py` measures `copy_rate_ref` (real dev windows at stride 8 vs Train). If `copy_rate_ref > 0.08`, Claude Code amends ledger §9 G-P3 to **`copy_rate ≤ copy_rate_ref + 0.05`** and logs the change in ledger §14 **before Day 5**, i.e. before any model sample exists. That is a calibration, not post-hoc tuning. Until the amendment is logged, `gate_p3.py` applies the literal 10 % and prints the reference beside it.

### 3.9 Diversity ↑ (within prompt; collapse indicator)
- `div(p) = mean_{i<j} H(G_i, G_j)` over the samples of prompt p (3 samples → 3 pairs; P-div: 10 → 45 pairs). `diversity` = mean over prompts.
- **Real reference:** real windows sharing an identical caption (stride 8, groups ≥ 3) have a mean pairwise Hamming of **19.9** (weighted). It depends on the caption: `full floor. a few enemies.` 9.9; `full floor.` 9.1; `a few platforms.` 17.8; `full floor. a few enemies. one pipe.` 22.2.
- **Diversity ratio** (P-div only) `= div_model(p) / div_real(cap p)`, using prompts whose exact caption has a real group of ≥ 3. A ratio < 0.5 means mode collapse (doc 25 R-3.8). A ratio > 2 together with a rising broken rate means noise, not creativity.
- **AMED_self** (paper-comparable) = mean over samples of the min Hamming distance to any *other* sample in the same set. It depends strongly on set size, so always report n. ("Edit distance" in the paper's code is this tile-wise Hamming distance, not Levenshtein.)
- Diversity must be read **together with** novelty and copy rate (doc 01 #4): high diversity + low novelty = memorising many different training windows.

### 3.10 Tile-distribution Jensen–Shannon divergence ↓ (G-P3)
`p` = tile histogram over **all cells** of the sample set (13 bins); `q` = histogram over all cells of Train. `M = (p + q)/2`; `JS(p‖q) = ½ KL(p‖M) + ½ KL(q‖M)` with **log base 2** (range [0, 1]). Zero bins are skipped in the KL sums (0·log 0 = 0).
**References:** 5-level hold-out vs rest: mean **0.0045**, p95 0.0122, max 0.0234. Single level vs rest: median 0.011, max 0.077. SMB1 vs SMB2: 0.0010. **All-sky generator 0.0676.** Uniform-noise generator 0.619. Reading: the G-P3 threshold 0.05 rejects sky collapse and noise, but values above **0.025** (beyond any natural 5-level variation) are flagged as a drift warning in the gate report.
**Diagnostic:** per-row JS (row-conditional tile histograms, rows 2–15), shown as a 14-bar chart. It catches floors in the wrong row and ceiling or sky-row artefacts.

### 3.11 Seam metric ↓ (G-P5; estimator defined in doc 20 §seam)
`seam_ratio = mean NLL(column transitions crossing window seams) / mean NLL(interior transitions)`, using the column-transition model fitted on train levels (doc 20). The corpus has **653 distinct columns** in 8,317, so the estimator must be smoothed (doc 20). **Reference:** the same estimator on real dev levels with pseudo-seams at the generator's seam positions (expected ≈ 1.0). Gate: model ratio ≤ 1.3 (ledger §9). Reported with a level-clustered bootstrap CI.

### 3.12 Casual-prompt metrics (P-casual; normalizer = doc 12)
- **Normalizer accuracy** (text only, no model): `norm_exact` = share of items whose normalised canonical phrase set equals the gold canonical caption; `norm_attr_acc` = share of gold attributes recovered with the correct bucket.
- **End-to-end attribute satisfaction** `attr_sat` = share of the *specified* attributes satisfied by `cap(G)` (bucket equality for counts; absence for explicit negations such as "no enemies"). Unspecified attributes are ignored.
- **End-to-end C-score** against the gold canonical caption (§3.1), reported beside the C-score against the normalizer's own output, so the two error sources stay separate.
- Discipline: dictionary fixes are driven by P-casual-dev failures only (doc 25 R-6.4). P-casual-test is opened only by `eval_test.py`.

### 3.13 Dev composite score S (E4) — checkpoint & CFG selection
```
S = 0.4·Cscore + 0.2·(1 − broken_rate) + 0.2·playable + 0.2·min(1, novelty_med / novelty_ref)
```
- All terms are computed on **P-dev** samples at the evaluated (checkpoint, w): Cscore per §3.1 (range [−1, 1]), broken_rate per §3.4, playable per §3.6 (scene mode, tier 1), novelty_med per §3.7, novelty_ref = the real dev reference (§3.7, frozen on Day 2).
- Range: [−0.4, 1.0]. S is a **ranking device only**, never reported as a quality claim.
- The novelty term is capped: more novelty than real held-out levels earns nothing, so S cannot be gamed by noise. Noise is punished through Cscore, broken_rate and playable.
- **Ties:** |ΔS| < 0.01, or a paired-bootstrap CI that includes 0 (§7.2) → prefer, in order, higher playable, then the simpler config, then the earlier checkpoint.
- **Phase-3 unconditional variant** (needed for M2's schedule choice, where no prompts exist): `S_P3 = 0.4·floor_valid + 0.2·(1 − broken_rate) + 0.2·playable + 0.2·min(1, novelty_med/novelty_ref)` on the 256 P-uncond samples. If the checker isn't ready on Day 5, drop the playable term and renormalise the weights (0.5 / 0.25 / 0.25), and log it. *(This variant is not spelled out in the ledger; it is listed in the conflicts report.)*

### 3.14 Full-level metrics (level mode)
- **L-single:** C-score of every stride-8 window of the generated content against the single prompt → level C-score = mean over windows. Broken, floor validity, novelty and copy rate per window; playability in level mode on the framed level; seam ratio (§3.11); generation time (wall-clock, GPU synchronised).
- **L-multi:** requires the generator to emit `scene_spans` (column span per scene prompt) in the level JSON (doc 29 metadata). Each window that lies **fully inside** a scene span is scored against that scene's caption. Windows straddling two scenes are excluded (and counted).

---

## 4. Real-data references (`reports/references/REFERENCES.json`)

`scripts/compute_references.py` (Day 2; it re-runs automatically whenever `LOCK` changes) writes every reference with its provenance. Pack values (LOLO, 2026-11-17) are listed so that Claude Code can sanity-check the recomputation. Large deviations (> 25 % relative) mean a bug or an unusual split.

| Key | Definition | Pack value |
|---|---|---|
| `cscore_null_dev` | real dev captions vs random real captions from other levels | ≈ 0.51 |
| `cscore_mode_dev` | modal-caption constant generator | ≈ 0.64 |
| `em_null` | whole-caption exact match null | 0.0011 |
| `novelty_ref` | median nn of real dev windows (stride 8) to Train | ≈ 10 (LOLO) |
| `copy_rate_ref` | share of real dev windows with nn ≤ 4 | ≈ 0.175 (LOLO) |
| `broken_rate_real` | broken structures in real windows | 0 (exact) |
| `floor_valid_real` | §3.5 on real windows | 0.9991 |
| `floorless_real` | row 15 entirely passable | 0.1664 |
| `js_holdout_p95` | JS of a random 5-level hold-out vs rest | 0.0122 |
| `div_real_same_caption` | mean pairwise Hamming, identical-caption real groups | 19.9 |
| `playable_real_scene` | fast checker on real dev windows (stride 8) | measured Day 5 |
| `playable_real_level` | fast checker on all real train+dev levels (full) | measured Day 5 (≈ 1.0 expected) |
| `seam_ratio_real` | doc 20 estimator on real dev levels at pseudo-seams | measured Day 11 (≈ 1.0) |

Test-level references (natural novelty and solvability of test windows) are computed **inside** `eval_test.py` on Day 15 only.

---

## 5. Comparators (rows of every results table)

| ID | System | Purpose |
|---|---|---|
| **B0** | Real data: (a) real windows of the evaluated levels for structure/novelty/diversity/JS/playability; (b) the **shuffled-caption null** for C-score/EM | ceiling and floor of what is possible |
| **B1** | Unconditional model (Phase-3 winner), its samples scored against each prompt | what realism alone achieves without text |
| **B2** | Final conditional model at **w = 1** (no guidance) | what conditioning achieves without CFG |
| **B3** | Final conditional model at **w\*** (dev-selected) | **the system** |
| **B4†** (optional) | Authors' released `schrum2/MarioDiffusion-MiniLM-single-regular0` on our prompts | external reference only |

**B4 caveats (print them under the table):** it was trained on the leaky random split (F17), which includes windows of **our dev and test levels**, so its numbers on our prompts are contaminated (optimistic) and its "novelty vs our Train" is not its true novelty. It uses a different text encoder (`multi-qa-MiniLM-L6-cos-v1`) and its own sampler defaults (guidance 7.5, 30 steps). Run it only in an **isolated folder/venv** (`tools/external_baseline/`, its own requirements; never imported into `src/`, CLAUDE.md rule 10). It writes `scenes.json`, and our evaluation code scores that file. Mark every B4 cell with †. Stretch item (Day 15/17, only if time).

---

## 6. Procedures

### 6.1 In-training dev evaluation (every 2,000 steps; M12)
- EMA weights; **all P-dev prompts × 3 seeds** at the current default w (5.0 until Day 9, then w\*). Metrics: Cscore, broken_rate, playable (tier 1), novelty_med, copy_rate, S, floor_valid, JS. Wall-clock budget ≈ 15–30 s on either GPU (≈ 350 samples).
- Written to `runs/<id>/dev_eval.jsonl`, one line per evaluation: `{"step", "w", "n", "cscore", "cscore_ci", "broken", "playable", "novelty_med", "copy_rate", "S", "floor_valid", "js", "time_s"}`.
- Early stopping (M12): patience 5 evaluations on S; `best_ema.pt` = argmax S (ties per §3.13).

### 6.2 CFG sweep (Day 9; repeated on Day 14 for the frozen model)
w ∈ {1, 2, 3, 5, 7.5, 10} × P-dev × 3 seeds (CRN across w) plus P-rand-dev. Plot S, Cscore, broken_rate, playable, novelty_med and diversity against w. **w\* = argmax S** (ties → the smaller w). Write w\* to the ledger change log (M13).

### 6.3 Full dev evaluation (Day 14): `scripts/eval_dev.py`
```
python scripts/eval_dev.py --ckpt runs/<id>/best_ema.pt --sets p_dev,p_rand_dev,p_casual_dev,p_div,l_single_dev,l_multi_dev \
       --w <w*> --seeds 3 --out reports/dev/<run_id>/ [--compare runs/<other>/best_ema.pt] [--per-level] [--jar-subset 0]
```
Outputs (§8): `samples.npz`, `per_sample.parquet`, `metrics.json`, `tables.md`, figures. `--compare` produces paired-bootstrap tables (§7.2). It refuses any test-side prompt file (hard-coded allow-list).

### 6.4 Cross-validation evaluation (Day 15; D5)
For fold k: train on the other 4 folds (final config), select the checkpoint with S computed on **fold-k stride-8 prompts**, then report all metrics on fold-k prompts + P-rand-dev. Report **mean ± std (ddof = 1), min and max over 5 folds**, plus the per-fold table. Prompts differ between folds, so CV measures variance from *both* the training-level sample *and* the evaluated levels. This is the honest generalisation spread.

### 6.5 One-shot test (Day 15): pre-registration and guards
**(a) Pre-registration.** Before Day 15, commit `reports/final/eval_plan.json` (it must be in `git log` before the test run starts):
```json
{
  "schema_version": 1,
  "frozen": {"frozen_json_sha256": "...", "checkpoint_sha256": "...", "config_hash": "..."},
  "split_lock_sha256": "...", "prompts_lock_sha256": "...",
  "prompt_sets": {"p_test": {"path": "data/prompts/p_test.jsonl", "sha256": "...", "n": 140},
                  "p_rand_test": {"...": "..."}, "p_casual_test": {"...": "..."}, "l_single_test": {"n_levels": 100, "width": 112}},
  "sampling": {"w": 5.0, "ddim_steps": 50, "eta": 0.0, "clip_x0": true, "seeds_per_prompt": 3, "set_base": {"p_test": 9100000}},
  "comparators": ["B0", "B1", "B2", "B3"],
  "metrics": ["cscore", "em", "per_concept", "broken_rate", "floor_valid", "playable_fast", "playable_jar_subset",
              "novelty_med", "copy_rate", "diversity", "js", "seam_ratio", "attr_sat"],
  "statistics": {"ci": "percentile cluster bootstrap by level", "cluster_unit": "level_id", "prompt_level_secondary": true, "B": 10000, "rng_seed": 12345, "multiplicity": "Holm"},
  "hypotheses": [
    {"id": "H1", "claim": "CFG improves adherence on unseen levels", "test": "paired ΔCscore(B3 − B2) > 0, 95% CI excludes 0"},
    {"id": "H2", "claim": "Text conditioning beats realism-only", "test": "paired ΔCscore(B3 − B1) > 0"},
    {"id": "H3", "claim": "No memorisation beyond natural level", "test": "copy_rate(B3) ≤ copy_rate_ref_test + 0.05"},
    {"id": "H4", "claim": "Structures valid", "test": "broken_rate(B3) ≤ 0.05"}
  ],
  "jar_subset": {"n": 300, "selection": "first 300 samples of B3 by sha256(prompt_id|replicate) order"},
  "code": {"git_sha": "...", "eval_files_sha256": {"src/text2mario/eval/adherence.py": "...", "...": "..."}},
  "anomaly_policy": "Only metric-code bugs justify a re-run: fix, re-run, report BOTH runs. Model, config, w and prompt sets are never changed."
}
```
**(b) Guards in `scripts/eval_test.py`** (each failure aborts with a non-zero exit before any sampling):
1. Working tree clean; `eval_plan.json` is committed and its commit time precedes now; the sha256 values listed in it match the files on disk.
2. `sha256(split_v1.json) == LOCK`; `PROMPTS_LOCK` matches; `FROZEN.json` matches the checkpoint sha256 and config hash.
3. `reports/final/test_runs/` is empty. Otherwise `--rerun-reason "<metric-bug id + description>"` is required, and the reason is appended to `reports/final/TEST_LEDGER.md` (append-only).
4. The w and sampling settings passed in equal those in `eval_plan.json` (the script reads them from there; no CLI override).

**(c) Run** → `reports/final/test_runs/<UTC-timestamp>_<gitsha>/` (samples, per_sample, metrics) → `scripts/make_results.py` → `reports/final/RESULTS.md` (main table, per-concept, per-game, casual, CIs, hypotheses H1–H4 verdicts). Commit as-is.

---

## 7. Statistics

### 7.1 Unit of analysis and cluster bootstrap (E3, amended 2026-11-18)
Samples of one **level** are not independent: adjacent windows share 15 of 16 columns, prompts of the same level describe the same content, and the 3 seeds of a prompt share it too. **The resampling unit is therefore the held-out LEVEL**, carrying all of its prompts and all their replicates. (The earlier draft clustered by prompt; that treats 117 prompts from 5 levels as 117 independent draws and gives intervals that are too narrow.)
```python
def level_cluster_bootstrap(per_level, stat, B=10_000, seed=12345):
    """per_level: list with one entry per held-out level; each entry holds ALL samples (prompts x seeds)
    generated for prompts that originate from that level. stat: list-of-entries -> float (recomputed per replicate)."""
    rng = np.random.default_rng(seed); L = len(per_level)      # L = 5 (dev) or 6 (test)
    boots = np.array([stat([per_level[i] for i in rng.integers(0, L, L)]) for _ in range(B)])
    return stat(per_level), np.percentile(boots, [2.5, 97.5])
```
- **Few clusters, wide intervals.** With L = 5 (dev) or 6 (test) there are only 5^5 = 3,125 or 6^6 = 46,656 distinct level resamples, so the percentile interval is coarse and **wide**; it is still the honest one. The report says so in words next to every headline table: "intervals are wide because only 5 (dev) / 6 (test) levels are held out".
- **Always report per-level results too** (§7.3 table): one row per held-out level with n, the metric and its seed spread. A reader can then see which level drives a number.
- **Prompts without a level of origin** (P-rand, P-casual) have no cluster: they are bootstrapped **by prompt** and every such interval is labelled "prompt-clustered; conditional on the trained model, not on levels". They are never mixed into a level-clustered headline number.
- **Secondary, labelled interval:** the old prompt-clustered interval may be printed in a second column titled "conditional on these levels".
- Mean-type metrics use the vectorised form (multinomial counts over levels x per-level sums / per-level n). Quantile-type metrics (novelty_med) and composites (S) are recomputed per replicate. The bootstrap RNG seed is fixed (12345) for reproducibility. CIs use the percentile method; switch to BCa only if a metric is highly skewed (report which). Results JSON key: `statistics.ci = "percentile cluster bootstrap by LEVEL"` (§6.5 `eval_plan.json`).

### 7.2 Paired comparisons (A/B, CFG, ablations)
With CRN (§2.3), for each prompt p: `d_p = mean_r [m_A(p, r) − m_B(p, r)]`. Compute `d_p` per prompt, average within each level, then bootstrap the **levels** → CI of the mean over levels (for composites, recompute S_A − S_B per replicate). **Decision rule:** "A beats B" if the 95 % CI of Δ lies above 0 **and** Δ exceeds the minimum effect of interest: ΔS ≥ 0.01, ΔCscore ≥ 0.02, Δplayable or Δbroken ≥ 2 pp. Otherwise "no detectable difference" → the simpler option wins (doc 23 §6).

### 7.3 Level-level uncertainty (be honest)
Prompt-level CIs are **conditional on the particular held-out levels**: 5 dev and 6 test levels are few, and per-level difficulty varies (per-level novelty medians range 2–56.5). Always add: (a) a **per-level table** (each held-out level's Cscore, playable, broken, novelty_med); (b) a **leave-one-level-out jackknife range** for headline metrics (min/max over dropping each level); (c) the 5-fold CV spread (§6.4) as the primary estimate of level-sampling variance.

### 7.4 Multiple comparisons
Each experiment pre-declares **one primary metric: S** (dev). Secondary metrics are descriptive. When the report claims several "significant" differences from one family (e.g., X1–X12), apply **Holm–Bonferroni** at family-wise α = 0.05 to the paired-bootstrap p-values (p = 2·min(P(Δ* ≤ 0), P(Δ* ≥ 0)) from the bootstrap distribution). The test-set hypotheses H1–H4 are one family (Holm).

### 7.5 Cross-validation statistics
Report mean ± sample std (ddof = 1), min and max over 5 folds. Don't compute p-values from folds: the training sets overlap, so fold results are dependent and the variance of the CV mean has no unbiased estimator (Bengio & Grandvalet, 2004). Present the std as "spread", not as a standard error.

### 7.6 Precision expectations and the borderline rule for gates
- A proportion near 0.85 with n = 256 independent samples would have a 95 % half-width of ±4.4 pp (an *optimistic* binomial figure for gates; the borderline rule below uses it only as a guide). Near 0.90 with n ≈ 350: ±3.1 pp. C-score (per-sample SD ≈ 0.15–0.20): the prompt-level half-width with ≈ 117 prompts would be ±0.03, **but that treats prompts as independent; the level-clustered interval from 5-6 levels will be considerably wider** (to be measured on Day 9/15; no number is claimed here). Proportions are likewise binomial approximations only; the reported CI is the level-clustered one.
- Gates compare the **point estimate** with the ledger threshold. **Borderline rule (pre-declared):** if the threshold lies inside the 95 % CI, draw **one** extension of equal size with fresh seeds (base + 500,000) and decide on the pooled estimate. Only one extension is allowed. It is logged in RESULT.json (doc 24).

---

## 8. Output artifacts and schemas
- `samples.npz`: `grids uint8 (n,16,W)`, `prompt_id str (n)`, `replicate int8 (n)`, `seed int64 (n)`, `w float32 (n)`, `ckpt_sha256`, `config_hash`.
- `per_sample.parquet` columns: `prompt_id, set, replicate, seed, w, caption_prompt, caption_gen, cscore, em, broken_pipe_n, broken_cannon_n, floor_valid, playable, play_status, nn, nn_train_idx, complexity, level_id(dev/test origin, nullable), game, time_ms`, plus per-concept `req_k, gen_k, exact_k`.
- `metrics.json`: `{metric: {"value", "ci95": [lo, hi], "n", "n_prompts", "reference", "delta_vs_reference"}}` per set, plus a `meta` block (git SHA, machine, torch, timestamps).
- All figures are made by scripts reading these files (CLAUDE.md §3).

---

## 9. Reporting templates (copy into `RESULTS.md` / the report)

**Table 1 — Main results (P-test, 3 seeds, 95 % CI)**
| System | C-score ↑ | ΔC vs null | EM ↑ | Broken ↓ | Floor-valid ↑ | Playable ↑ | Novelty med (ref) | Copy ↓ (ref) | Diversity | JS ↓ |
|---|---|---|---|---|---|---|---|---|---|---|
| B0 real / null | 0.51 (null) | 0 | 0.1 % | 0 % | 99.9 % | ref | ref | ref | ref | ref |
| B1 uncond | | | | | | | | | | |
| B2 cond w = 1 | | | | | | | | | | |
| **B3 cond w\*** | | | | | | | | | | |
| B4† external | | | | | | | | | | |

**Table 2 — Per concept (B3, P-test):** concept · support · precision · recall · F1 · strict EM (null EM) · bucket MAE · bias.
**Table 3 — Per game:** SMB1 vs SMB2 rows × main metrics (+ per-game references: copy_rate_ref 21.4 % / 14.0 % LOLO).
**Table 4 — CFG sweep (dev):** w · S · C-score · broken · playable · novelty med · diversity.
**Table 5 — Cross-validation:** metric · mean ± std · min · max · fold 1…5.
**Table 6 — Casual prompts:** norm_exact · norm_attr_acc · attr_sat · C-score vs gold · C-score vs normalised.
**Table 7 — Full levels (L-single-test):** level C-score · broken · seam ratio (ref) · playable (level mode) · verified rate · attempts · time (GPU/CPU).
**Table 8 — Playability agreement:** fast checker vs jar confusion matrix, agreement %, Cohen's κ, false-"verified" rate.

---

## 10. Implementation map and checklist
- [ ] `adherence.py`: `extract_phrases`, `normalize_plural`, `quantity`, `cscore`, `caption_exact`, `per_concept_table`, golden-vector tests.
- [ ] `structure.py`: `pipe_components`, `valid_pipe`, `valid_upside_down_pipe`, `broken_counts`, `floor_valid`, `tile_js`, `row_js`, `nn_hamming` (GPU), `copy_rates`, `diversity`, `amed_self`.
- [ ] `stats.py`: `cluster_bootstrap`, `paired_bootstrap`, `holm`, `jackknife_levels`.
- [ ] `references.py` + `scripts/compute_references.py` → `REFERENCES.json` (pack values in §4 as sanity bounds).
- [ ] `scripts/make_prompt_sets.py` → `data/prompts/*.jsonl` + `PROMPTS_LOCK`.
- [ ] `scripts/eval_dev.py`, `scripts/eval_test.py` (guards §6.5b), `scripts/make_results.py`.
- [ ] Report the null ladder (§3.1) and the references (§4) next to every headline number.
