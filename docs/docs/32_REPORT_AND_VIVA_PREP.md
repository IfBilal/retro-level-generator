# 32 — Report & Viva Preparation

> **Owner:** B. **Used on:** Day 14 (freeze inputs), Day 16 (results), Day 17 (figures, model/data cards), Day 18 (demo hardening), Day 20 (rehearsal).
> **Hard rules:** every figure and table is produced by a script from saved artifacts (CLAUDE.md §3; doc 30 §11). Every number carries a provenance tag (doc 30 §11.1). Test numbers come only from `runs/final/test_eval/` (D4).
> **Ledger links:** F1–F19 (data facts), D3–D6 (splits), M1–M13 (model), G1–G5 (generation), E1–E4 (evaluation), H8/H10 (size, cost), A1–A4 (assets).

---

## 1. Report structure

Target: 20–25 pages of main text plus appendices. The section order follows the argument: *the data was harder to use honestly than it looked → we built a careful pipeline → here is what it achieves, with error bars → here is the product → here is what it can't do.*

| § | Section | Target length | Must contain |
|---|---|---|---|
| — | **Abstract** | 150–200 words | Task; method in one sentence; the leakage finding; the headline test numbers with CIs; the app |
| 1 | **Introduction** | 1.5 p | PCG motivation; text-to-level; why levels must be *playable*, not just plausible; **contributions list** (below) |
| 2 | **Related work** | 1.5 p | PCGML (Summerville et al. 2018 survey); VGLC (Summerville et al. 2016); MarioGAN (Volz et al. 2018); MarioGPT (Sudhakaran et al. 2023); DDPM (Ho et al. 2020); improved DDPM / cosine schedule (Nichol & Dhariwal 2021); DDIM (Song et al. 2021); classifier-free guidance (Ho & Salimans 2022); RePaint (Lugmayr et al. 2022); discrete-tile diffusion for Mario (Lee & Simo-Serra 2023); text-to-level diffusion for Mario (Schrum et al., AIIDE 2025); discrete diffusion alternative D3PM (Austin et al. 2021); Sentence-BERT (Reimers & Gurevych 2019), MiniLM (Wang et al. 2020); A* Mario agent (Togelius, Karakovskiy & Baumgarten 2010) |
| 3 | **Data** | 2.5 p | Source and licence; audit facts (Table T1); level reconstruction; **leakage analysis of the official split** (Fig 4: the F17 numbers); our level-grouped split + CV (Table T2, Fig 5); EDA (Figs 2–3) |
| 4 | **Method** | 4 p | Representation (D7); diffusion maths (compact: forward, ε-loss, DDIM, CFG; full derivations in Appendix A); architecture (Table T3, Fig 7); text path (M7–M8); training (Table T4); full-level generation (G1–G5, Fig 16); playability checker (doc 21) |
| 5 | **Experimental setup** | 1.5 p | Prompt sets (E2); metrics (E1, definitions in Appendix B); statistics (E3); dev composite (E4); hardware + budgets (H1, M12); the pre-registered test plan (doc 30 §8.4) |
| 6 | **Results** | 5 p | 6.1 Unconditional baseline (Table T5, Figs 8–9) · 6.2 Conditional, dev (Fig 10, Table T6) · 6.3 CFG trade-off (Fig 11, Fig 12) · 6.4 Ablations (Fig 19, Table T7) · 6.5 5-fold CV (Fig 20, Table T8) · 6.6 **One-shot test** (Table T9 with reference rows) · 6.7 Full levels & seams (Figs 16–17) · 6.8 Playability validity (Fig 18) · 6.9 Casual prompts (Table T10) |
| 7 | **Analysis** | 2.5 p | Bias–variance & memorisation evidence (Figs 8, 14, 15); per-game (SMB1 vs SMB2); per-concept and rare concepts (Fig 13); failure gallery (Fig 21); detector–human agreement (R14) |
| 8 | **The application** | 1.5 p | Architecture diagram (Fig 1, app half); "Type it, play it" flow; screenshots (Fig 22); denoising filmstrip (Fig 23); latency (Table T11) |
| 9 | **Limitations & threats to validity** | 1 p | See §3.3, as a list, honestly |
| 10 | **Ethics, licensing, reproducibility** | 0.5 p | Licences (§3.4); reproduction chain (doc 30 §15); code/data availability |
| 11 | **Conclusion & future work** | 0.5 p | The 9 reserved features (doc 31) in one paragraph; D3PM; learned captioner |
| App. A | Diffusion derivations | — | From doc 13 |
| App. B | Metric definitions | — | From doc 22 |
| App. C | Full hyperparameters | — | `configs/final.yaml` rendered as a table |
| App. D | Gate results | — | Table T12 |
| App. E | PDF audit summary | — | Table T13 (top 10 of the 30 findings) |
| App. F | Risk outcomes | — | Table T14 + risk burn-down |
| App. G | Reproduction commands | — | doc 30 §15 |

**Contributions list (draft wording; adjust to the actual results):**
1. A leakage audit of the public MarioDiffusion split. Test windows are near-copies of training windows (median 8/256 tiles from the nearest training window); we release a level-grouped split with a 5-fold grouped CV protocol.
2. A from-scratch 18.3 M-parameter text-conditional tile diffusion model (frozen MiniLM + adapter, cross-attention, CFG), evaluated on held-out *levels* with CIs.
3. A playability checker that shares its physics with the browser game, validated against the Baumgarten A* agent, and used to verify every generated level before play.
4. An end-to-end application: a typed sentence becomes a full, verified, playable level in the browser.

---

## 2. Figures and tables

All figure scripts live in `scripts/figures/`, are orchestrated by `scripts/make_figures.py`, and write `reports/figures/figNN_<slug>.{png,pdf}` plus the exact data they plot to `reports/figures/data/figNN_<slug>.csv`. Paths refer to the run layout in doc 30 §3.

### 2.1 Figures

| ID | Title | Script | Data source | Ready by |
|---|---|---|---|---|
| Fig 1 | System overview (data → training → sampling → checker → app) | `fig01_system_overview.py` (renders `reports/figures/src/system.dot` with graphviz) | static source file | D12 |
| Fig 2 | Tile-frequency bars (global, per game, per row; log scale) | `fig02_tile_freq.py` | `data/processed/index.parquet` + raw grids | D1 |
| Fig 3 | Per-level concept heatmap (42 levels × 16 concepts), split membership annotated | `fig03_level_concepts.py` | `index.parquet`, `split_v1.json` | D2 |
| Fig 4 | **Leakage: NN-Hamming distance of held-out windows to training windows — official random split vs our level-grouped split** (histograms + medians) | `fig04_leakage.py` | `reports/eda/leakage_official.csv`, `reports/eda/leakage_ours.csv` (from `audit_dataset.py`/`make_splits.py`) | D2 |
| Fig 5 | Split composition: levels/windows per split per game; concept coverage per split | `fig05_split_composition.py` | `split_v1.json` (`constraints_report`) | D2 |
| Fig 6 | Forward-corruption strips at t = 0/250/500/750/999, linear vs cosine | `fig06_corruption.py` | `reports/gates/P2/` | D3 |
| Fig 7 | U-Net architecture diagram with shapes and per-stage parameter counts (H8) | `fig07_unet.py` | `src/text2mario/models/unet.py` introspection | D4 |
| Fig 8 | Per-t-bucket loss curves: train vs val (fixed noise), U-lin vs U-cos | `fig08_bucket_losses.py` | `runs/<U-lin>/metrics.jsonl`, `runs/<U-cos>/metrics.jsonl` | D5 |
| Fig 9 | Unconditional sample sheet (64 samples, EMA) | `fig09_uncond_sheet.py` | `runs/<winner>/samples/step_015000/samples.npz` | D6 |
| Fig 10 | Conditional training: dev composite + components vs step; early-stop marker; train/val bucket losses | `fig10_cond_training.py` | `runs/<C-main>/metrics.jsonl` | D9 |
| Fig 11 | **CFG trade-off:** C-score, exact match, broken rate, playability, novelty vs w ∈ {1, 2, 3, 5, 7.5, 10} with 95 % CIs | `fig11_cfg_sweep.py` | `runs/<C-main>/evals/*_P-dev_*_w*/metrics.json` | D9 |
| Fig 12 | PDF contrast test: "full floor with three pipes and no enemies" vs "staircase with gap and several enemies" × 8 seeds | `fig12_contrast.py` | `runs/<C-main>/evals/<contrast_eval>/samples.npz` | D9 |
| Fig 13 | Per-concept exact match (dev and test) with CIs and n per concept | `fig13_per_concept.py` | dev eval + `runs/final/test_eval/per_prompt.parquet` | D16 |
| Fig 14 | **Memorisation check:** distribution of NN distance to train — generated (P-dev, P-test) vs real dev/test windows; copy-rate threshold line at 4 | `fig14_novelty.py` | eval `per_prompt.parquet` (`nn_dist`) + `reports/eda/real_nn_ref.csv` | D16 |
| Fig 15 | Diversity vs novelty per prompt (scatter), coloured by w | `fig15_div_vs_nov.py` | eval `per_prompt.parquet` | D14 |
| Fig 16 | Full-level gallery: 3 prompts × width 112, seams marked, scene-prompt spans annotated | `fig16_full_levels.py` | `reports/final/showcase/*.json` (level schema v1, doc 29) | D11 |
| Fig 17 | Seam score: seam/interior NLL ratio for {stride 8 r = 0, stride 8 r = 2, stride 4 r = 2} | `fig17_seams.py` | `reports/gates/P5/seams.csv` | D11 |
| Fig 18 | Playability validity: confusion matrix fast checker vs MarioEval.jar (300 levels), plus real-level baseline bars | `fig18_playability_agreement.py` | `reports/gates/P5b/agreement.csv` | D11 |
| Fig 19 | Ablation forest plot: Δ dev composite (and Δ C-score, Δ copy rate) vs baseline with paired-bootstrap 95 % CIs (X1–X7) | `fig19_ablation_forest.py` | `reports/experiments/*.md` → `runs/<X*>/evals/*` | D15 |
| Fig 20 | 5-fold grouped CV: fold-wise points + mean ± std for each metric | `fig20_cv.py` | `runs/cv/fold*/` → evals | D15 |
| Fig 21 | Failure gallery (annotated): ignored counts, broken structure, unbeatable, near-copy, seam | `fig21_failures.py` | error-analysis list `reports/analysis/worst20.json` | D14 |
| Fig 22 | App screenshots: prompt → animation → game → badge | `fig22_app_screens.py` (Playwright screenshots) | live app at the frozen checkpoint | D18 |
| Fig 23 | Denoising filmstrip (every 5 DDIM steps) for one prompt | `fig23_filmstrip.py` | sampler callback frames (doc 31 seam S-2) | D12 |
| Fig A1 | Risk burn-down (sum of open risk scores per day) | `figA1_risk_burndown.py` | `reports/risks/risk_log.csv` | D20 |

### 2.2 Tables

| ID | Title | Script | Data source |
|---|---|---|---|
| T1 | Dataset facts (F1–F17), measured | `tab01_facts.py` | `data/MANIFEST.json → facts_verified` |
| T2 | Split: levels and windows per split × game; prompts after stride-8 | `tab02_split.py` | `split_v1.json`, `index.parquet` |
| T3 | Architecture: layer table, shapes, parameters (U-Net 18,312,013; adapter 66,048) | `tab03_arch.py` | model introspection |
| T4 | Hyperparameters (final) | `tab04_hparams.py` | `configs/final.yaml` + `FROZEN.json` |
| T5 | Phase 3: U-lin vs U-cos at 5k/10k/15k (floor validity, broken, JS, copy rate) | `tab05_uncond.py` | `reports/gates/P3/RESULT.json` + evals |
| T6 | Dev results at the chosen w (all E1 metrics, CIs) for P-dev, P-rand, P-casual-dev | `tab06_dev.py` | `runs/final/evals/*` |
| T7 | Ablations X1–X7: config delta, Δ metrics, decision | `tab07_ablations.py` | `reports/experiments/*.md` + evals |
| T8 | 5-fold CV mean ± std | `tab08_cv.py` | CV evals |
| T9 | **One-shot test results** (PE1–PE3 + secondary) **with reference rows** (§2.3) | `tab09_test.py` | `runs/final/test_eval/metrics.json` |
| T10 | Casual prompts: normalizer concept accuracy + adherence on P-casual-dev/test | `tab10_casual.py` | normalizer eval + test_eval |
| T11 | App latency: cold start, generate+verify by width, GPU vs CPU fast mode | `tab11_latency.py` | `reports/app/latency.json` (Day 13/18 benchmark) |
| T12 | Gate summary: G-P1 … G-P7, thresholds, measured values, date | `tab12_gates.py` | `reports/gates/*/RESULT.json` |
| T13 | PDF audit summary (top 10 findings) | `tab13_audit.py` | `docs/01_PDF_AUDIT.md` (parsed table) |
| T14 | Risk outcomes (realised / closed / open) | `tab14_risks.py` | `reports/risks/risk_log.csv` |

### 2.3 Reference rows for Table T9 (and T6)
Numbers mean little without anchors. Every results table includes these rows, computed with the same metric code:

| Row | How it is computed | What it anchors |
|---|---|---|
| **Real held-out windows** | The real dev/test windows (stride 8) scored against their own captions | Upper reference for C-score (≈ 1.0 by construction, which also sanity-checks the detector, R5); real playability rate (cf. F18 ≈ 93 %); **novelty_ref** = real windows' NN distance to train (E4); broken rate (expected 0) |
| **Retrieval baseline** | For each prompt, return a training window whose canonical phrase set equals the prompt's (uniform choice, project-seeded); if none, the training window with the highest C-score against the prompt | A "copy machine": high C-score, **copy rate ≈ 100 %, novelty ≈ 0**. Shows that our C-score is earned *with* novelty, not by retrieval |
| **Unconditional model** | The Phase 3 winner, sampled for the same number of samples | Lower reference for adherence (the effect of text) |
| **w = 1 (no guidance)** | Final model, w = 1 | The isolated effect of CFG |

The retrieval baseline must be implemented in `src/text2mario/eval/baselines.py` by Day 14 and listed in `eval_plan.json → reference_rows` (doc 30 §8.4).

---

## 3. Writing guidance: honest claims

### 3.1 Do / Don't

| Topic | Do say | Don't say |
|---|---|---|
| Split | "Evaluated on 6 held-out **levels** never seen in training; the public window-level split leaks (median 8/256 tiles from the nearest training window)." | "Our C-score beats Schrum et al." Their split is easier; the numbers are **not comparable**, so say so explicitly. |
| Novelty | "Median NN distance to training windows is X tiles (real held-out levels: Y); copy rate Z % at ≤ 4 tiles." | "The model never copies" or "the model is creative". |
| Diversity | "Within-prompt diversity (mean pairwise Hamming) is D." | Using diversity as evidence of novelty (audit #4). |
| Playability | "Beatable according to our checker, which agrees with the Baumgarten A* agent on A % of 300 levels." | "Beatable by humans" (only showcase levels were human-played; say how many). |
| Framing | "Start and flag segments are deterministic templates (3 + 6 columns); only the middle is generated." | Implying the model generates the flag or the safe start. |
| Sanitizer | "Metrics are computed on raw samples; repair is display-only." | Reporting post-repair broken rates as model quality. |
| Adherence | "Adherence measured by the rule-based detector that produced the training captions; on a 50-sample human check, the detector agreed with humans on H %." | "The model understands the prompt." Prefer "conditions on" / "follows". |
| Statistics | Report n, seeds, CI method; call a difference "significant" only if a test was run. | Bare point estimates; "clearly better" without a CI. |
| Test set | "Evaluated once, on Day 16, with a pre-registered plan." Disclose any metric-code rerun (doc 30 §8.4). | Silent re-runs. |
| Compute | "Trained on a 6 GB laptop GPU budget (≤ 4.5 GiB)." | Wall-clock claims without the machine named. |
| Casual text | "A rule-based normalizer maps casual words to the caption grammar before the frozen encoder." | Implying MiniLM alone understands "goombas". |

### 3.2 Phrasing templates
- Results sentence: "On P-test (n = N prompts × 3 seeds), the final model reaches C-score **a** [95 % CI a₁–a₂], exact match **b**, playability **c** (real windows: **c₀**), and copy rate **d** (retrieval baseline: ≈ 100 %)."
- Comparison sentence: "X3 changed the dev composite by **+δ** [paired 95 % CI δ₁–δ₂]; because the interval includes 0, we kept the simpler baseline."

### 3.3 Limitations (include all that apply, with numbers)
1. **Small corpus:** 42 levels, 8,317 distinct columns (F11); held-out sets are 5–6 levels → wide CIs; CV mitigates but doesn't remove this.
2. **Caption vocabulary:** 44 words, 62 phrases (F14). The model can only be asked for what the detector can describe. The normalizer bridges casual words, not new concepts.
3. **Detector circularity:** the same rules create the training captions and score adherence (R14).
4. **Single enemy type** and static cannons: all enemies share one tile, and the game renders them as a single walker type.
5. **Missing mechanics:** no springs, vines, lifts (R38). Some real levels can't be represented faithfully.
6. **Physics approximation:** the browser game is Mario-*like*; the checker is validated against the game's own physics and against A*, not against the NES.
7. **Locality:** 16-wide windows with overlap. No global plan (pacing is future work, FF4 in doc 31).
8. **Negation:** "no X" is enforced by omission + rejection, not learned (R35).
9. **Fast-mode training is not bitwise reproducible across GPUs.** Claims are statistical (doc 30 §1).

### 3.4 Licensing paragraph (fill and verify the exact licence text at the pinned commits)
- Level data and captions: MarioDiffusion repository (MIT), derived from the Video Game Level Corpus (VGLC). Cite both, and check the VGLC licence at the pinned commit and record it in `DATA_CARD.md`. Level designs are Nintendo's; used for non-commercial academic research.
- Sprites: Kenney "Pixel Platformer" (CC0). Phaser 3.90.0 (MIT). `sentence-transformers/all-MiniLM-L6-v2` (check the model card licence at the pinned revision; record it in `MODEL_CARD.md`).
- No Nintendo sprites are distributed (A3).

---

## 4. Model card template (`MODEL_CARD.md`)
*(after Mitchell et al., "Model Cards for Model Reporting", 2019)*

```markdown
# Model Card — text2mario v1.0 (frozen <date>)

## Model details
- Type: text-conditional ε-prediction DDPM (T = 1000) with DDIM sampling (50 steps, η = 0), CFG (w = <default_w>)
- Backbone: from-scratch 2D U-Net, 18,312,013 params; text adapter 66,048 params; frozen all-MiniLM-L6-v2 (revision <sha>)
- Input/Output: English prompt → (13, 16, W) one-hot tile grid (13 SMB tile types, ledger F6) → argmax tiles
- Checkpoint: <path>, sha256 <…> (FROZEN.json) · config_hash <…> · git tag v0.9-frozen
- Authors / contact / date / licence of weights: <…>

## Intended use
- Generating Super Mario Bros.-style tile levels from short descriptions for education, prototyping, research demos.
## Out-of-scope use
- Commercial distribution of Nintendo-derived content; claims of human-level design quality; any game other than SMB-style tile platformers.

## Training data
- 31 levels (SMB1 + SMB2/Lost Levels) from the MarioDiffusion corpus (7,687 windows total; train ≈ <n> windows); level-grouped split (DATA_CARD.md).

## Evaluation data
- Dev: 5 held-out levels (stride-8 prompts: <n>); Test: 6 held-out levels (<n> prompts), evaluated once; P-rand (100 random captions); P-casual (30 dev / 30 test).

## Metrics (test, one-shot; 95 % bootstrap CI grouped by prompt)
| Metric | Value | CI | Real-window reference | Retrieval baseline |
|---|---|---|---|---|
| C-score | | | | |
| Exact match | | | | |
| Playability (checker) | | | | |
| Broken-structure rate | | | | |
| Novelty (median NN tiles) | | | | |
| Copy rate (NN ≤ 4) | | | | |
| Diversity | | | | |

## Quantitative analyses
- Per game (SMB1 vs SMB2); per concept (Fig 13); CFG trade-off (Fig 11); 5-fold CV (Table T8).

## Ethical considerations & caveats
- Layout data derived from Nintendo games; academic use. Detector-defined adherence (circularity); single enemy type; templated start/flag; checker-verified (not human-verified) playability except for <k> showcase levels.

## How to reproduce
- docs/30 §15 command chain.
```

## 5. Data card template (`DATA_CARD.md`)
*(after Gebru et al., "Datasheets for Datasets")*

```markdown
# Data Card — text2mario split v1

## Motivation
Text-to-level training/evaluation with an honest, level-grouped split.

## Composition
- 7,687 windows (16×16) from 42 levels: 20 SMB1, 22 SMB2/Lost Levels (F3, F7); 7,620 unique (F4)
- Encoding: integers 0..12 (F6); rows 0–1 are padding (F12); sky 87.13 % (F13)
- Captions: rule-based, 1–10 phrases, 62 distinct phrases, 44-word vocabulary (F14); quantity buckets (F15)

## Collection & preprocessing
- Source: MarioDiffusion @ <commit> (MIT), derived from VGLC (<licence>); windows at stride 1 (F8)
- Level IDs reconstructed by overlap chains (F10–F11), validated against per-game files (F9)

## Splits (seed 20261117, LOCK <sha256>)
| Split | Levels (SMB1/SMB2) | Windows | Prompts @ stride 8 |
|---|---|---|---|
| Train | 31 (…/…) | | — |
| Dev | 5 (…/…) | | |
| Test | 6 (3/3) | | |
- 5-fold grouped CV over the 36 non-test levels (fold membership in split_v1.json)

## Known issues
- The official random split leaks (F17); do not use it for generalisation claims
- Rare concepts: <list with counts per split>
- Duplicates: 67 exact duplicates (mostly trivial windows), cross-split duplicates listed in reports/eda/leakage_ours.csv
- Missing mechanics (springs, vines, lifts); single enemy type

## Uses / distribution / maintenance
- Academic use; regenerate with scripts/fetch_data.py + make_splits.py --verify-lock
```

---

## 6. Live demo script (3 minutes, rehearsed to ≤ 2:50)

**Set-up before the examiner arrives:** app already running and warmed up (`streamlit run app/streamlit_app.py`); browser at 100 % zoom; an external keyboard plugged in for the examiner; second tab: the folder of 12 cached showcase levels as standalone HTML; slide deck open on the results slide; Wi-Fi **off** (proves it's offline).

| Time | Action | What you say (≈) |
|---|---|---|
| 0:00–0:15 | Studio screen | "You type a sentence; the model draws a Mario level tile by tile; a robot checks it can be finished; then you play it." |
| 0:15–0:55 | Type *"a few pipes, some coins and a gap, then a staircase"*, width 80 → **Generate**. Point at the denoising animation, then the normalised prompt line, then the **verified beatable** badge | "It starts from pure noise. The 50 steps you see are DDIM. These words were mapped to the training grammar. The checker found a path." |
| 0:55–1:20 | Click **Play**, play 15–20 s, reach a pipe and a gap | "This is the generated level as a real game, not a picture." |
| 1:20–1:45 | Second prompt, casual: *"lots of goombas and tubes, no gaps"* | "Casual words work through the normalizer. Note it now has many enemies and pipes." |
| 1:45–2:15 | Switch to the results slide (Table T9 + Fig 4) | "The public split leaks: test windows were 8 tiles from training ones. We split by level instead. On 6 unseen levels, evaluated once: C-score a, playability c, copy rate d." |
| 2:15–2:40 | **Hand the keyboard to the examiner** with a cached showcase level (or their own prompt) | "Please, try one yourself." |
| 2:40–2:50 | Close | "Limitations: small corpus, rule-based captions, templated flag. Next: ghost-bot repair, difficulty dial." |

**Contingency branches (practised on Day 18):**
- GPU/CUDA error → toggle **CPU fast mode** (≈ 25 s for 112 wide, H10). Say: "this is the CPU fallback."
- App won't start → open a standalone HTML level from the showcase folder (no Python needed).
- Laptop dead → desktop or USB: standalone HTML levels + report PDF + slides PDF.
- Examiner's prompt yields "not verified" → say so honestly, regenerate with another seed, and explain the G5 attempt cap.

---

## 7. Viva Q&A bank (52 questions)

Answers are short on purpose; expand only if asked. "Evidence" points to where the proof lives.

### A. Diffusion mathematics
**A1. Write the forward process.** q(x_t | x_{t−1}) = N(√(1−β_t) x_{t−1}, β_t I). Composing the steps gives the closed form q(x_t | x₀) = N(√ᾱ_t x₀, (1−ᾱ_t) I), with α_t = 1−β_t and ᾱ_t = ∏_{s≤t} α_s. So x_t = √ᾱ_t x₀ + √(1−ᾱ_t) ε, ε ~ N(0, I). *Evidence: doc 13; Fig 6.*

**A2. Why predict ε, and why MSE?** The ELBO's per-step terms are KLs between Gaussians. With the posterior mean reparameterised as μ_θ = (1/√α_t)(x_t − β_t/√(1−ᾱ_t) ε_θ), each KL reduces to a weighted ‖ε − ε_θ‖². Ho et al. found that dropping the weights (L_simple) gives better samples. ε_θ is also a scaled score estimate: ε_θ ≈ −√(1−ᾱ_t) ∇ log p_t(x_t).

**A3. Give the reverse posterior.** q(x_{t−1} | x_t, x₀) = N(μ̃_t, β̃_t I), with μ̃_t = [√ᾱ_{t−1} β_t/(1−ᾱ_t)] x₀ + [√α_t (1−ᾱ_{t−1})/(1−ᾱ_t)] x_t and β̃_t = β_t (1−ᾱ_{t−1})/(1−ᾱ_t). At sampling time, x₀ is replaced by x̂₀ from ε_θ.

**A4. How can DDIM use 50 steps when you trained with 1000?** DDIM defines a non-Markovian process with the **same marginals** q(x_t | x₀). Training only ever used those marginals, so the same network works. The update is x_{t'} = √ᾱ_{t'} x̂₀ + √(1−ᾱ_{t'} − σ²) ε̂ + σz with x̂₀ = (x_t − √(1−ᾱ_t) ε̂)/√ᾱ_t, valid for any decreasing subsequence of timesteps. With η = 0 (σ = 0) it is deterministic: the same x_T always gives the same level.

**A5. Derive classifier-free guidance.** Classifier guidance samples from p(x)·p(c|x)^w: score = ∇log p(x) + w ∇log p(c|x). By Bayes, ∇log p(c|x) = ∇log p(x|c) − ∇log p(x), so the guided score is ∇log p(x) + w(∇log p(x|c) − ∇log p(x)). Converting scores to ε (ε ∝ −score) gives ε̂ = ε_u + w(ε_c − ε_u). One network learns both by dropping the caption 10 % of the time (D8d). **w = 1 is plain conditional sampling** (M13).

**A6. What does w trade off?** Higher w sharpens towards prompt-typical layouts: adherence rises, diversity and novelty fall, and past some point structures break and playability drops (x̂₀ leaves [−1, 1]; we clip it). We chose w on dev by the composite, not C-score alone. *Evidence: Fig 11.*

**A7. Linear or cosine, and why?** Linear destroys information fast at this resolution: at t = 500, ᾱ = 0.079 (SNR 0.085) vs cosine 0.494 (SNR 0.98). Cosine spends more of the trajectory at informative noise levels (Nichol & Dhariwal). End points: ᾱ_T = 4.04e-5 (linear) vs 2.43e-9 (cosine). We trained both and selected on dev (M2). *Evidence: Table T5, Fig 8.*

**A8. How did you test the scheduler independently of the network?** The oracle-ε test: plug the *true* noise into DDIM, ε_oracle = (x_t − √ᾱ_t x₀)/√(1−ᾱ_t); the reverse pass must return x₀ exactly. It did: MSE 0.0, argmax agreement 100 % in fp64, for both schedules (G-P2).

**A9. Why clip x̂₀ to [−1, 1]?** The data lives in {−1, +1}^13 per tile. Under strong guidance x̂₀ overshoots, and clipping keeps the trajectory on the data's range (it is the "clip_sample" trick), which reduces broken structures at high w.

**A10. Explain your outpainting maths.** It is RePaint known-region replacement. At each step, the known columns are set to a fresh sample of q(x_t | x₀^known), and the unknown columns come from the model. Resampling (r = 2) re-noises from t' back to t with x_t = √(ᾱ_t/ᾱ_{t'}) x_{t'} + √(1 − ᾱ_t/ᾱ_{t'}) z and denoises again, letting the two regions harmonise. *Evidence: doc 20; Fig 17.*

**A11. Is this a likelihood model? What's your NLL?** In principle you could bound the likelihood with the ELBO, but the data is discrete and decoded by argmax, so a continuous-density NLL isn't meaningful for tiles. We report task metrics (adherence, playability, novelty, structure).

**A12. You treat discrete tiles as continuous. Isn't that wrong?** It is a continuous relaxation: one-hot in {−1, +1}, Gaussian diffusion, argmax decoding. It matches prior Mario diffusion work, and it keeps standard DDIM/CFG/RePaint tooling. Failure modes are argmax near-ties and structure breaks, which we measure (broken rate). A discrete diffusion (D3PM) is the principled alternative and is listed as future work. The optional aux-CE loss (M9) partly bridges the gap.

### B. Architecture & conditioning
**B1. Describe the U-Net.** Three resolutions 16 → 8 → 4 with 64/128/256 channels; 2 ResBlocks per level in the encoder, 3 in the decoder; GroupNorm(8), SiLU, dropout 0.1; FiLM time conditioning; self- and cross-attention at 8×8 and 4×4; output Conv to 13 channels. 18,312,013 parameters, 57 % of them in the decoder (10.5 M). *Evidence: Table T3, Fig 7, `reference/ref_unet.py`.*

**B2. Why self-attention from the start when the PDF only uses it as a rollback?** Pipes are 2-wide objects whose halves must agree, and floors are long-range. Attention at 8×8 lets distant columns coordinate. At this size it is cheap, so it fixes broken pipes up front instead of after a failed gate.

**B3. Why cross-attention only at 8×8 and 4×4?** Prompt semantics ("three pipes", "a gap") are layout decisions made at coarse scale. Local tile syntax is handled by convolutions at 16×16. It also halves the attention cost.

**B4. Why GroupNorm and not BatchNorm?** It doesn't depend on the batch. Training at batch 64 and sampling at any batch size behave identically, and a sample's output doesn't depend on its batch-mates (doc 30 §5.5).

**B5. Why a frozen MiniLM?** The captions use 44 words and 62 phrases. Fine-tuning a 22.7 M-parameter encoder on that would overfit and erase its general English knowledge, which we need for casual prompts. A 66,048-parameter adapter learns the mapping into the U-Net's space. (The PDF's "prevents gradient collapse" justification is wrong; audit #12.)

**B6. Token states or a pooled vector?** Token states (B, L ≤ 48, 384) with a key-padding mask, so attention can bind individual phrases to locations. A pooled vector added to the time embedding was tested as ablation X3. *Evidence: Table T7.*

**B7. What is the "null" condition?** The frozen encoder's embedding of the empty string "" (its [CLS][SEP] tokens), passed through the same adapter. That gives the unconditional branch ε_u for CFG.

**B8. Schrum et al. found a small from-scratch MLM encoder worked best. Why not use that?** Their prompts were in the caption grammar. Ours must accept casual English ("goombas", "tubes"). A pretrained encoder + normalizer is more robust off-grammar. It's a deliberate trade-off we name in the report. An MLM encoder is a natural ablation for future work.

### C. Data & splits
**C1. Describe the data.** 7,687 16×16 windows (7,620 unique) at stride 1 from 42 levels (20 SMB1, 22 Lost Levels); integer tiles 0..12; 87.1 % sky; rule-based captions with 62 phrases. *Evidence: Table T1.*

**C2. What's wrong with the official split?** It is random over stride-1 windows, and neighbouring windows share 15 of 16 columns. The 385 test windows cover all 42 levels; median distance to the nearest training window is 8 of 256 tiles; 84 % are within 16 tiles; 2.1 % are exact copies. Test scores there measure memorisation. *Evidence: Fig 4.*

**C3. The JSON has no level IDs. How did you split by level?** Consecutive windows of one level overlap by 15 columns, so we chained windows where `S[i−1][:,1:] == S[i][:,:-1]`. That gives exactly 42 chains (20 + 22), matching the per-game source files and the total of 8,317 columns. *Evidence: audit script, F7–F11.*

**C4. Why 31/5/6 levels?** Test ≈ 14 % of levels (3 SMB1 + 3 SMB2 for game balance), dev 5 for selection, the rest for training. Levels were chosen by a seeded algorithm that enforces concept coverage and avoids cross-split near-duplicates, then locked by hash on Day 2.

**C5. What does the 5-fold grouped CV add?** Each fold holds out different levels, so the spread estimates how results vary with *which* levels are unseen, plus training noise. It gives error bars that a single dev set can't. The test levels stay out of CV entirely.

**C6. Why sub-sample prompts at stride 8 for generative metrics?** Stride-1 windows are near-duplicates, which would inflate n and shrink CIs dishonestly. Stride 8 keeps half-overlapping windows at most.

**C7. Doesn't a level-grouped split still leak? Levels reuse patterns.** Some reuse is real game design (e.g. flat floor stretches). We measured test-to-train NN distances, listed every window within 2 tiles, and checked that real test windows are as far from train as the gate requires. The remaining overlap is documented, not hidden. *Evidence: `leakage_ours.csv`.*

**C8. Why no horizontal-flip augmentation?** Mario levels are designed for rightward play: staircases, run-ups before gaps, ascending vs descending semantics. Flipping (with remapped `<>`/`[]` and swapped staircase words) changes the design distribution. It was an ablation only (X5).

**C9. What are rows 0–1?** Padding: VGLC levels are 14 rows high, padded at the top to 16. That's why sky is 87 %, not the PDF's 75 %.

### D. Training & generalisation
**D1. Why measure budgets in steps, not epochs?** With stride-1 windows, one epoch shows each column about 16 times (~87 steps per epoch at batch 64). "150 epochs" sounds small but is ~13k steps of heavy repetition. Steps plus early stopping on held-out levels is the honest unit.

**D2. What's your evidence you didn't over-fit?** (1) Per-t-bucket validation loss on fixed noise doesn't rise while training loss falls before the chosen step. (2) Copy rate stays ≤ 10 % and novelty sits near the real held-out reference. (3) Test performance is close to dev and to the CV mean. *Evidence: Figs 8, 10, 14; Tables T8–T9.*

**D3. How do high bias and high variance show up in a diffusion model?** High bias: training and val loss both high in the mid t-buckets, blurry/sky-dominant samples, broken structures, low adherence on *training* prompts. High variance: training loss falling while val loss rises, high copy rate, a big dev–CV gap. *Evidence: doc 19.*

**D4. Why EMA 0.999 rather than the PDF's 0.995?** The averaging horizon is about 1/(1−decay): 0.995 averages ~200 steps, 0.999 ~1,000. With 15k–60k-step runs, 0.999 smooths meaningfully. We ablated 0.995 (X2).

**D5. Why bf16 without a GradScaler?** bf16 has fp32's exponent range, so gradients don't underflow the way they do in fp16. Both Ampere and Ada support it. Attention softmax stays in fp32.

**D6. You switched GPUs mid-run. How do you know results are comparable?** Every training input (data order, augmentation, CFG mask, t, ε, initial weights) is drawn from seeded CPU generators, so it is bit-identical on both machines. Checkpoints carry the full optimiser, EMA and RNG state. The resume drill showed continuous curves. Evaluating the same checkpoint on both GPUs agreed on ≥ 99 % of tiles. *Evidence: doc 30 §5–§7.*

**D7. Why exclude norms and biases from weight decay?** Decaying scale/shift parameters towards zero fights normalisation and doesn't regularise model capacity meaningfully. That's the standard practice.

### E. Evaluation
**E1. Define C-score.** Following Schrum et al.: for each concept t, match(p_t, c_t) between the prompt and the caption detected on the output is 1.0 if identical; 1 − |qu(p_t) − qu(c_t)|/(|Q| − 1) when both are countable (Q = the ordered quantity buckets, F15); 0.1 when only one has a quantity; −1.0 otherwise. Average over the concept set T, so the range is [−1, 1]. Doc 22 pins the exact case table and the definition of T.

**E2. Why report exact match as well?** C-score's partial credit can hide systematic off-by-one-bucket errors. Exact match per concept shows exactly where counts fail.

**E3. Novelty vs diversity: what's the difference?** Diversity is distance *between generated samples* for the same prompt. Novelty is distance *to the training set*. A model that memorised ten training windows has high diversity and zero novelty. The PDF conflated them (audit #4).

**E4. Why a copy threshold of 4 tiles?** That's 1.6 % of a 256-tile window, a few isolated tile flips. We calibrate against real held-out windows: their NN distance to train is the natural baseline (novelty_ref). Generated levels shouldn't be closer to train than real unseen levels are.

**E5. Why bootstrap grouped by prompt?** The 3 samples per prompt are correlated. Resampling prompts (with all their seeds) gives honest CIs. For ablations we use a paired bootstrap on the same prompts and seeds, so prompt difficulty cancels out.

**E6. How do you know "verified beatable" means anything?** (1) The checker and the game read the same `physics.json`. (2) The checker's jumps are deliberately more conservative than the game's. (3) It agrees with the Baumgarten A* agent on A % of 300 levels. (4) We played every showcase level by hand. *Evidence: Fig 18.*

**E7. Why are only ~93 % of real windows solvable?** Windows cut levels mid-structure (e.g. a gap at the right edge), some sections need mechanics the tile set lacks, and A* has search limits. That's why we compare against the real-window rate, not 100 %.

**E8. Isn't adherence circular, since the detector wrote the captions?** Partly, yes, and we say so. We mitigate with a blind 50-sample human check of detector judgements and report the agreement. *Evidence: §3.3 item 3.*

**E9. Why include a retrieval baseline?** It shows what "copying the right training window" achieves: high C-score with ~100 % copy rate. Our model must earn its C-score *with* novelty. *Evidence: Table T9.*

**E10. How many times did you look at the test set?** Once, on Day 16, with an analysis plan committed before the freeze and hashed into `FROZEN.json`. All tuning used dev.

**E11. Why that composite score for selection?** It balances adherence (0.4) with structure, playability and novelty (0.2 each). Selecting on C-score alone would pick memorising, high-w, structure-breaking checkpoints. We checked that the chosen checkpoint is stable under ±0.1 weight perturbations (report it).

### F. Generation & playability
**F1. How do long levels stay coherent?** Overlapping windows (stride 8) with known-region replacement and r = 2 resampling. We measured seams: the transition NLL at seams vs interiors, ratio ≤ 1.3 (G-P5). *Evidence: Fig 17.*

**F2. Is the flag generated?** No. The start (3 floor columns) and the end (6 columns with staircase and flag) are deterministic templates, and the report card says so (G3).

**F3. Your sanitizer repairs pipes. Doesn't that hide errors?** Metrics are computed **before** repair; repair is display-only (G4).

**F4. What happens if the checker says "not beatable"?** The failing window is resampled with a new seed, up to 5 times per window and 20 in total. Then the badge honestly says "not verified" (G5).

**F5. What if I type "no enemies"?** The regular captions have no "no X" phrase. The normalizer removes the concept and marks it negated, and samples containing it are rejected within the attempt budget. Absence adherence is reported separately (R35).

### G. App & engineering
**G1. How does a level become a game?** The tile grid is exported as level JSON (schema v1), embedded with the sprite atlas into a self-contained Phaser 3 page, and shown inside Streamlit. It works offline, and every level can be downloaded as a standalone HTML file.

**G2. How fast is it?** A 112-wide level generates and verifies in under 10 s on the laptop GPU (G-P7). On CPU, one guided forward pass is ~75 ms, so full quality would take ~107 s; the CPU fast mode (25 steps, no resampling) takes ~25 s (H10). *Evidence: Table T11.*

**G3. Why cap VRAM at 4.5 GiB when the model needs under 2 GiB?** So anything that runs on the 8 GB desktop is guaranteed to run on the 6 GB laptop, with room left for the display and browser (H2–H3).

**G4. What if CUDA fails during the viva?** CPU fast mode, then cached standalone levels. Both were rehearsed on Day 18.

### H. Ethics & licensing
**H1. Is using Nintendo levels legal or ethical?** It is non-commercial academic research on public research data (MarioDiffusion, MIT; derived from VGLC), with attribution. We ship no Nintendo sprites; the game uses CC0 Kenney art (A1, A3).

**H2. What licences apply?** Code: ours. Data: MIT (MarioDiffusion) + the VGLC licence at the pinned commit. Phaser: MIT. Sprites: CC0. MiniLM: per its model card at the pinned revision. All recorded in the model and data cards.

**H3. Does this replace level designers?** It is an ideation and prototyping tool. It works only within a tiny caption vocabulary, and the human chooses, edits and judges. The future "erase and re-ask" feature is explicitly human-in-the-loop.

### I. Critical & forward-looking
**I1. With 42 levels, what does "novel" even mean?** Exactly what we measure: distance to the training windows, compared to how far *real unseen* levels are. We don't claim novelty beyond that.

**I2. Why diffusion and not an LLM like MarioGPT?** Diffusion edits the whole 2D grid jointly, which suits inpainting/outpainting and CFG control. An autoregressive model generates column by column. Both are valid; diffusion gives us RePaint-style editing for free, which future features use.

**I3. What is the weakest part of your work?** Adherence is defined by the same rule-based detector that wrote the captions, and the held-out sets are small. Both are measured and disclosed (human check, CV error bars).

**I4. How do you know your improvements aren't seed luck?** Pre-registered decision rules, paired bootstrap on the same prompts and seeds, a second seed for close calls, and CV for the final config.

**I5. What would you do with another month?** The reserved features (doc 31): ghost-bot repair and the report card first, then erase-and-re-ask, difficulty measured by simulated players, and self-training with a strict judge. On the modelling side: D3PM-style discrete diffusion, and a learned captioner to escape the rule vocabulary.

**I6. The PDF was your instructor-approved plan. Why change it?** Measurement showed it was wrong in places: wrong tile order, wrong dataset size, a leaky split, diversity mistaken for novelty, no playability check. Every change traces to a finding in the audit with evidence. *Evidence: Table T13.*

---

## 8. Numbers card (memorise; fill the blanks on Day 16)

| Fact | Value | Source |
|---|---|---|
| Windows / unique / levels | 7,687 / 7,620 / 42 (20 + 22) | F3, F4, F7 |
| Distinct columns | 8,317 | F11 |
| Sky share | 87.13 % | F13 |
| Official-split leakage | median 8/256 tiles; 83.9 % ≤ 16; 2.1 % exact | F17 |
| Split | 31 / 5 / 6 levels; seed 20261117 | D3 |
| U-Net / adapter params | 18,312,013 / 66,048 | H8 |
| Schedules ᾱ_T | linear 4.04e-5; cosine 2.43e-9 | G-P2, audit #30 |
| ᾱ at t = 500 | linear 0.079; cosine 0.494 | ref build |
| CPU guided pass | ≈ 75 ms | H10 |
| Default w | ___ | FROZEN.json |
| Test C-score [CI] | ___ | T9 |
| Test playability / real ref | ___ / ___ | T9 |
| Copy rate (ours / retrieval) | ___ / ≈ 100 % | T9 |
| Checker–A* agreement | ___ % | Fig 18 |
| GPU latency 112-wide | ___ s | T11 |

---

## 9. Rehearsal checklist

**Content (Day 17–19)**
- [ ] Slides ≤ 12: problem · leakage (Fig 4) · method (Fig 1, Fig 7) · training evidence (Fig 8/10) · CFG trade-off (Fig 11) · test table (T9) · playability validity (Fig 18) · app (Fig 22) · limitations · future features.
- [ ] Every number on the slides has a provenance tag in the speaker notes (doc 30 §11).
- [ ] The report PDF builds from source; `check_provenance.py` green.

**Demo (Day 18–20)**
- [ ] 3 timed run-throughs of §6 (≤ 2:50 each); 1 run-through with Wi-Fi off; 1 with `CUDA_VISIBLE_DEVICES=""`.
- [ ] 12 cached showcase levels, all **human-played to the flag**; standalone HTML copies on USB.
- [ ] Examiner prompts drill: 15 random casual prompts typed by someone else; note any normalizer misses and add synonyms **before** the freeze only. After the freeze, use prompt chips; do not change the frozen lexicon (FROZEN.json hashes it).

**Q&A (Day 19–20)**
- [ ] Two mock vivas: 15 random questions from §7, 60 s per answer, the other person scoring clarity.
- [ ] Be able to write A1, A4 and A5 on a whiteboard from memory.
- [ ] Practise saying "I don't know, but here's how I'd find out" for questions outside scope.

**Logistics (Day 20, evening before)**
- [ ] Laptop: OS updates paused, notifications off, power plan "Performance", charger, display scaling 100 %, HDMI/USB-C adapter, mouse, **external keyboard for the examiner**.
- [ ] USB stick: standalone levels, report PDF, slides PDF, `FROZEN.json`, `RESULTS.md`, a short screen recording of the demo (last resort).
- [ ] If two presenters: roles fixed (one drives the demo, one answers maths), hand-off phrases agreed.
