# 01 — Brutal Audit of the PDF Blueprint

**Audited document:** *Text-to-Mario 2D Level Diffusion Pipeline — End-to-End Architecture, Iterative Execution Stages, Verification Gates, and Feedback Rollback Loops* (6 pages, all read).
**Method:** every factual claim was checked against (a) a fresh clone of `schrum2/MarioDiffusion`, by measuring the data directly, (b) the AIIDE 2025 paper *Text-to-Level Diffusion Models With Various Text Encoders for Super Mario Bros* (Schrum et al., arXiv 2507.00184) and its appendix, and (c) first-principles diffusion maths, verified with a reference implementation (`reference/ref_unet.py`).
**Standard applied:** "almost unfair". A claim passes only if it is correct, specific enough to implement without guessing, and testable.

Severity scale: **S0 = invalidates results**, **S1 = will cause a failed gate or wasted days**, **S2 = ambiguity Claude Code would have to guess at**, **S3 = cosmetic / minor**.

---

## A. Verdict in one paragraph

The PDF has the right skeleton: a small discrete-tile DDPM, a frozen sentence encoder with a trainable adapter, cross-attention, CFG, sliding-window outpainting, phase gates, and a rollback table. It is a good student blueprint. But as written it **would produce scientifically invalid evaluation numbers**, because it inherits a data split in which test windows are near-copies of training windows (S0). It **never measures playability** (S0 for a product whose headline is "type it, play it"). Its "novelty" metric measures diversity, not novelty (S0 for the "not a copy of a Nintendo level" claim). Several numbers are wrong: dataset size, air fraction, and a rollback that contradicts the main plan. The gates are partly unmeasurable ("visually confirm"). Everything below is fixed in this pack.

---

## B. Findings table

| # | PDF location | PDF claim | Reality (evidence) | Sev. | Fix in this pack |
|---|---|---|---|---|---|
| 1 | §1 table, "Primary Dataset" | "~8,000+ sliding-window 16×16 ASCII level chunks" | **7,687** samples (7,620 unique), stored as **integers 0–12, not ASCII** (F3–F5). | S2 | Ledger F3–F5; loader reads integers. A char view is derived only for export. |
| 2 | Phase 1.1 | Lists 13 tiles in the order `- X S ? Q o E < > [ ] B b` | That order is **not the ID order** in the data. The real order is the sorted-character order: `- < > ? B E Q S X [ ] b o` = 0..12 (F6). Building the one-hot from the PDF's order silently scrambles every tile. | **S1** | Ledger F6 is authoritative; `tiles.py` defines it once; a unit test asserts it against `smb.json`. |
| 3 | Phase 1 / Phase 6 (implicit) | Uses the dataset as-is; Phase 6 "novelty" relies on it | The repo's split is a **random split over stride-1 windows**. The 385 test windows cover all 42 levels; median distance to the nearest train window is **8 of 256 tiles**; 84 % are within 16 tiles; 2.1 % are exact copies (F17). Any test metric on it measures memorisation, not generalisation. | **S0** | **Level-grouped split** (31 train / 5 dev / 6 test levels) plus level-grouped 5-fold CV (D3–D5, doc 10). |
| 4 | Phase 6.3 | "Pairwise Hamming distance between 10 seeds of the same prompt proves the model synthesises novel layouts rather than memorising" | Pairwise distance **between samples** measures **diversity**. A model that memorised 10 different training windows would score well. Novelty needs the distance **to the training set**. | **S0** | Separate metrics: diversity (within-prompt), **novelty = NN distance to train**, **copy rate** (NN ≤ 4), all calibrated against real held-out levels (E1, E4). |
| 5 | Whole PDF | No playability measurement anywhere | The paper measures solvability with an A* agent; even real data is only ~93 % solvable (F18). The product promise ("type it, play it") requires it. | **S0** | Phase 5b: fast Python checker that shares physics constants with the browser game, cross-validated against `MarioEval.jar` (doc 21). |
| 6 | Rollback row "Sky overwhelms" | "Air (tile 0) makes up ~75 % of every Mario grid" | Air is **87.1 %** because of the 2 padded top rows (F12–F13). Channel weights tuned to 75 % are mis-calibrated. | S2 | Weights derived from measured frequencies; channel weighting becomes a rollback option (M9). |
| 7 | Rollback row "Vertical seams" vs Phase 5.2 | Rollback: "increase overlap from 4 columns to 8" | Phase 5.2 **already** specifies an 8-column shift (overlap 8). The rollback is a no-op, so the real fallback is undefined. | S1 | Overlap 8 is the default; rollbacks escalate to overlap 12 (stride 4), more resampling (r = 2 → 5), and finally direct wide sampling (G1, doc 25). |
| 8 | Phase 2 gate | "At t = 999 … uniform random tile confetti with mean ≈ 0 and unit variance" | Correct in spirit but untestable as worded ("≈"). It also misses the strongest scheduler test there is. | S2 | Numeric tolerances plus an **oracle-ε DDIM exact-reconstruction test** (G-P2). Verified: MSE = 0.0, argmax agreement 100 % for both schedules. |
| 9 | Phase 3 gate | "≥ 85 % have coherent floors and complete pipes" | "Coherent" is undefined; 16 samples gives a ±18 % confidence interval. Real levels contain legal gaps, so "floor at row 15" is not always correct. | S1 | 256 samples; programmatic floor and structure validators; JS divergence and copy rate added (G-P3). |
| 10 | Phase 4 gate | "Visually confirm that the generated tiles obey the prompt" | Not a measurable gate. | S1 | Dev C-score ≥ 0.75 at the best w, and CFG must beat w = 1 (G-P4). The visual contrast check is kept as a secondary check. |
| 11 | Phase 3.3 / Milestones | "50 epochs", "100–150 epochs" | With stride-1 windows, one epoch shows each column ~16 times, so epochs are a misleading budget unit. 150 epochs over ~5.5k windows ≈ 13k steps, while the paper trained 500 epochs. Overfitting risk is high and unmeasured. | S1 | Budgets in **steps** (M12) with early stopping on a held-out-level dev score, plus memorisation monitors (doc 19). |
| 12 | §1 "Text Conditioner" | MiniLM + "2-Layer MLP Projection Head (128-dim)" "prevents gradient collapse" | "Gradient collapse" is not the reason. A frozen encoder avoids overfitting a 22 M-parameter LM on 62 distinct phrases, and saves memory. The claim is also unspecified: token states or a pooled vector? Mask handling? Null embedding? | S2 | Token states (B, L, 384) + key-padding mask; null = embedding of "" (M7); the rationale is corrected in doc 15. |
| 13 | Phase 1.4 | "Pre-cache frozen embeddings (optional speed boost)" | Pre-caching per caption string **conflicts with phrase-shuffle augmentation**, which the paper relies on (it randomises phrase order). | S2 | Online frozen encoding with an LRU cache (M8). |
| 14 | Phase 4.4 | CFG formula `ε_u + w(ε_c − ε_u)` with w = 5.0 | Formula is fine, but the PDF doesn't say w = 1 means "no guidance" (other papers use 1 + w). High w with continuous one-hot data pushes x̂₀ outside [−1, 1]; no clipping is specified. | S2 | Convention fixed (M13); x̂₀ clipping; w chosen on dev from a sweep. |
| 15 | Phase 3.2 | U-Net channel spec, but no number of ResBlocks per level, no attention at 8×8, GroupNorm group count unspecified | Under-specified. Self-attention at 8×8 appears only as a rollback, although it is cheap and the paper uses attention throughout. | S2 | Exact layer table (doc 14); self-attention from the start (M5); parameter count verified at 18.3 M (H8). |
| 16 | Phase 3.3 | EMA 0.995 | With 15k–60k steps, 0.995 averages over only ~200 steps, which is barely smoothing. 0.999 (~1,000 steps) suits this budget better. | S3 | 0.999 with warm-up; 0.995 ablated (M11). |
| 17 | §1 Compute | "RTX 3060 Ti handles full training; RTX 4050 handles prototyping" | The user will switch machines often. The PDF has no handoff, resume or VRAM-cap protocol, and doesn't note that this model fits easily in 6 GB. | S1 | Min-VRAM contract (H2–H4), identical hyperparameters, resumable checkpoints (doc 07). |
| 18 | Phase 1.2 / §1 | One-hot scaled to [−1, 1] "cutting training time to minutes" | True for the training step. **False for evaluation**: sampling for C-score/playability every few k steps dominates wall-clock (the paper's runs took ~12 h because of this). | S3 | Evaluation budget planned explicitly (doc 16 §eval cadence). |
| 19 | Phase 5.1 | Synonym normalizer: "Goombas → enemies, tubes → pipes" | Necessary but not sufficient. Counts must map onto the dataset's **bucket grammar** (one / two / a few / several / many) and phrase templates ("floor with a few gaps"). Without a held-out paraphrase set, the normalizer's own success can't be measured. | S1 | Full canonical grammar (62 phrases, F14), digit→bucket mapping, train/test split of the paraphrase lexicon (doc 12). |
| 20 | Phase 5.2 | "Lock columns 8–15 noised to timestep t" | Correct idea (RePaint known-region replacement), but the renoising maths for **strided DDIM** steps isn't given, and "noised to t" is ambiguous (fresh noise vs a consistent trajectory). | S2 | Exact equations in doc 20 (fresh noise per step via q(x_t \| x₀), and the DDIM-compatible renoise step for resampling). |
| 21 | Phase 5.3 | "Ensure 100 % sprite perfection" via a sanitizer | Repairing then measuring hides model defects. | S1 | Detect always; measure **before** repair; repair only for display (G4). |
| 22 | Phase 6.1 | "Pass 200 generated levels through the dataset's rule-based feature detector … exact match %" | 200 is thin and has no CIs. "Exact match" on bucketed counts is harsh; the paper's C-score gives partial credit. Which prompts? | S2 | P-dev/P-test/P-rand/P-casual prompt sets, 3 seeds, bootstrap CIs, both exact-match and C-score (E1–E3). |
| 23 | Phase 6.2 | Broken-tile rate "before and after EMA" | Fine, but incomplete: also needed per CFG scale (high w is known to break structures) and per schedule. | S3 | Added to the ablation matrix (doc 23). |
| 24 | Phase 7 | "running on your RTX 4050 laptop" | The viva needs a fallback if CUDA fails on the day. | S2 | CPU fast mode (H10) plus pre-generated demo levels cached on disk (doc 28). |
| 25 | §4 Workload split | Two engineers, each tied to one machine | The user describes one workflow that moves between machines. Ownership by machine conflicts with frequent switching. | S2 | Work split by **track** (Model track / Game-App track), not by machine; either track can run on either GPU (doc 03). |
| 26 | Golden Rule | "If MSE plateaus above 0.15 … stop" | ε-MSE level depends on the schedule and the t-distribution; one global number is fragile. For one-hot data the low-t loss is near 0 and the high-t loss near 1. | S2 | Loss logged per t-bucket (10 buckets) with deterministic validation noise; plateau rules defined per bucket (doc 19). |
| 27 | Whole PDF | No assets, game engine or level framing (start/flag) | "Real game in the browser" needs sprites with clear licences, a physics engine and an end goal. | S1 | Assets pipeline (CC0, doc 26), Phaser game (doc 27), deterministic framing (G3). |
| 28 | Whole PDF | No reproducibility protocol (seeds, manifests, versions, config hashes) | Two machines + many runs → irreproducible results. | S1 | Doc 30. |
| 29 | Whole PDF | No test-set discipline (how many times the test set may be looked at) | Repeated test peeking inflates results. | S1 | Test set locked on Day 2, evaluated once on Day 15 (D4). |
| 30 | Phase 2.1 | "linear … or cosine" (unresolved) | A choice left open is a guess for Claude Code. | S3 | Train both in Phase 3, select by dev score, tie → cosine (M2). Verified ᾱ_T: linear 4.04e-5, cosine 2.43e-9. |

---

## C. What the PDF gets right (kept as-is)

1. Compact (13, H, W) one-hot tile representation instead of pixel diffusion.
2. Unconditional baseline before text conditioning (the most important process rule; kept as Phase 3).
3. Frozen pretrained text encoder + small trainable adapter + cross-attention.
4. CFG with 10 % null-caption dropout.
5. DDIM 50-step inference.
6. Sliding-window outpainting for long levels.
7. A rollback matrix, and the principle "don't fix data or capacity bugs with more epochs."

## D. Consequences for the schedule

Fixing #3, #4, #5 and #27 adds roughly **4 working days** to the PDF's 16. The first draft of this pack ran 21 days (Day 0–20), which **overran the 6 Dec 2026 deadline**; the plan is now **18 days (Day 0 = 18 Nov to Day 18 = 6 Dec)** (`03_TIMELINE_DAY_BY_DAY.md`, ledger §10). Optional CV, ablation batch 2 and the extra buffer day were cut first; the level-grouped split, the playability checker, the one-shot test, the MVP game and the Streamlit Cloud deployment are never cut.

## E. Open items Claude Code must confirm on Day 0–1 (not guessable)

1. Both machines' NVIDIA driver versions (decides V2: torch 2.15 + CUDA 13 vs torch 2.14 + cu126).
2. Whether Java is installable on the desktop (decides whether the MarioEval.jar agreement study runs; doc 21).
3. Whether the instructor requires the original Nintendo look for the report figures (A3). Default: CC0 Kenney sprites.
4. The real deadline date (decides full vs compressed timeline).
