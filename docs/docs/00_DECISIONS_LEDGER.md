# 00 — Decisions Ledger (Single Source of Truth)

> **Rule for Claude Code:** If any other document in this pack contradicts this ledger, **this ledger wins**. If you change a decision, change it here first, add a row to §14 (Change Log), then update downstream docs. Every number used in code must come from `configs/*.yaml`, and every YAML value must trace back to a row here.

Pack version: **v1.0** (2026-11-17). Source blueprint: `Text-to-Mario_Level_Diffusion_Engineering_Pipeline.pdf` (6 pages, audited in `01_PDF_AUDIT.md`).

---

## 1. Verified facts about the data (measured, not assumed)

These were measured directly from a clone of `github.com/schrum2/MarioDiffusion` (MIT licence) on 2026-11-17. Claude Code must **re-measure them on Day 1** and fail loudly if they differ (`scripts/audit_dataset.py` asserts them).

| ID | Fact | Value | How measured |
|---|---|---|---|
| F1 | File | `datasets/Mar1and2_LevelsAndCaptions-regular.json` | repo listing |
| F2 | Record schema | list of dicts `{"prompt": null, "scene": int[16][16], "caption": str}` | `json.load` |
| F3 | Total samples | **7,687** (the PDF's "~8,000+" is wrong) | `len()` |
| F4 | Unique scenes | **7,620** (67 exact duplicates) | hash of bytes |
| F5 | Scene encoding | **integers 0..12 already**, not characters | inspection |
| F6 | Tile ID order (authoritative) | `0 '-' sky, 1 '<' pipe-top-L, 2 '>' pipe-top-R, 3 '?' full question, 4 'B' cannon top, 5 'E' enemy, 6 'Q' empty question, 7 'S' breakable, 8 'X' ground/solid, 9 '[' pipe-body-L, 10 ']' pipe-body-R, 11 'b' cannon support, 12 'o' coin` | appendix Table 1 + `datasets/smb.json` sorted keys |
| F7 | Levels | **42** = 20 SMB1 + 22 SMB2 (The Lost Levels) | overlap-chain reconstruction |
| F8 | Window stride | **1 column** (adjacent samples share 15 of 16 columns) | paper + chain check |
| F9 | Ordering | first **3,606** records are SMB1 (20 chains), next **4,081** are SMB2 (22 chains) | `np.array_equal` vs `SMB1_Levels.json` |
| F10 | Level-ID field | **none in JSON** → must be reconstructed by chain detection (`S[i-1][:,1:] == S[i][:,:-1]`) | inspection |
| F11 | Chain lengths (windows) | min 130, max 358; level width = windows + 15; total distinct columns **8,317** | computed |
| F12 | Height | **Refined (doc 08 §1.1, doc 09 §9):** source levels are 14 rows (20 levels), 15 rows (21 levels: SMB1 8-2, 8-3 and 19 SMB2 levels) or 16 rows (1 level: SMB2 3-1). The builder top-pads with sky to 16 rows (padding 2, 1 or 0 rows) and extends pipe tiles in the top source row upward through the padding. That is the source of the 64 non-sky cells in rows 0–1 (all `[`/`]`, in 17 windows). | computed |
| F13 | Tile frequencies | sky **87.13 %**, ground 8.66 %, breakable 1.70 %, enemy 0.62 %, pipe bodies 0.43 % each, coin 0.34 %, empty-Q 0.19 %, pipe tops 0.18 % each, full-Q 0.05 %, cannon top 0.04 %, cannon support 0.03 % | counted (PDF's "~75 % air" is wrong) |
| F14 | Captions | 2–28 words (mean 11.4); 1–10 phrases (mean 4.0); 62 distinct phrases; 44-word vocabulary | computed |
| F15 | Quantity words | `one`, `two`, `a few` (3–4), `several` (5–9), `many` (10+) | paper + data |
| F16 | Official split | random 90/5/5 → 6,918 / 384 / 385 | repo files |
| F17 | **Official split leaks** | the 385 test scenes cover **all 42 levels**; median Hamming distance to the nearest training scene = **8 / 256 tiles**; 83.9 % within 16 tiles; 2.1 % exact duplicates | computed |
| F18 | Real-data solvability (paper) | 7,160 / 7,687 ≈ 93 % solvable by Baumgarten A* | arXiv 2507.00184 |
| F19 | Caption generator | `create_ascii_captions.py::assign_caption(..., return_details=True)` returns contributing tile coordinates per phrase | source inspection |
| F20 | Level names | Sorting the 20 SMB1 and then 22 SMB2 source file names lexicographically and slicing at stride 1 reproduces **all 42 chains byte-for-byte** → `level_id` ↔ level name is certain (table in doc 09 §3) | doc 08 §1.1 |
| F21 | Cross-level near-duplicates | 7.49 % of windows lie within Hamming ≤ 2 of a window from **another** level. Level link graph at τ = 0.05 → **2 families** covering **15 levels / 3,121 windows** (Lost Levels reuse SMB1 sections) | doc 10 §3.3, doc 09 Q10 |
| F22 | Natural references (real held-out data, measured 2026-11-17) | copy rate (NN ≤ 4) of real never-seen windows ≈ **17.5 %** (leave-one-level-out, stride 8; SMB1 21.4 %, SMB2 14.0 %); novelty median 10 (LOLO), **13** for the example dev split vs train; tile-JS for a 5-level hold-out mean 0.0045 (p95 0.0122); C-score of real dev captions vs random real captions (null) ≈ **0.51**; real bottom row ≥ 50 % solid in only **71 %** of windows (16.6 % have no floor); real within-caption diversity 19.9 | doc 22 §3.7–3.10, §3.1; doc 09 Q9 |
| F23 | C-score topic set | The paper's scorer averages over **18 topics** = 16 concepts + broken pipe + broken cannon | doc 09 §9 item 6 |

## 2. Project scope

| ID | Decision |
|---|---|
| S1 | Game domain: **Super Mario Bros. 1 + 2 (Lost Levels) tile levels**, as the PDF recommends. No other games. |
| S2 | **MVP feature (must ship): "Type it, play it."** User types a sentence → a full-width level is generated → it opens as a **real playable game in the browser** (inside the Streamlit app), with a "verified beatable" badge. |
| S3 | All other user-listed features (ghost-bot repair, difficulty dial, erase-and-re-ask, whole-level pacing, word highlighting, report card, blind test, overnight self-training, endless mode) are **out of MVP scope** but get **reserved seams** (interfaces, data fields, folders). See `31_FUTURE_FEATURES_HOOKS.md`. |
| S4 | The PDF's 7 phases are kept, but gates are tightened, and two phases are added: **Phase 5b Playability checker** and **Phase 7b Browser game**. |
| S5 | UI framework: **Streamlit** (user's choice; the PDF allowed Gradio/Streamlit/FastAPI). |

## 3. Hardware and the minimum-VRAM contract

| ID | Decision |
|---|---|
| H1 | Machines: **Desktop RTX 3060 Ti 8 GB (Ampere, sm_86)** and **Laptop Lenovo LOQ RTX 4050 6 GB (Ada, sm_89)**. Work moves between them often. |
| H2 | **Design target = the 6 GB card.** Hard VRAM budget for any training or eval process: **≤ 4.5 GiB peak allocated** (leaves ~1.5 GiB for the CUDA context, the Windows display compositor and a browser). |
| H3 | Both machines enforce the same cap: `torch.cuda.set_per_process_memory_fraction(4.5 GiB / total)` at start-up. **Anything that runs on the desktop therefore also runs on the laptop.** |
| H4 | **Hyperparameters are machine-independent.** Global batch 64 is identical on both. Gradient accumulation exists only as an automatic OOM fallback and is logged if used. Only performance knobs (`num_workers`, `cudnn.benchmark`, `compile`) vary, in `configs/machine/*.yaml`. |
| H5 | Precision: **bf16 autocast** on both (Ampere and Ada support bf16) → no GradScaler. Attention softmax runs in fp32 (SDPA handles this). |
| H6 | `torch.compile`: **off by default** (fragile on Windows); optional flag. |
| H7 | Dataset lives **entirely on the GPU** as uint8 tile indices (7,687 × 16 × 16 = 2 MB); one-hot is computed per batch on the GPU. **No DataLoader workers** → no Windows multiprocessing problems. |
| H8 | Model size (**verified** with a reference build of M3–M7 on 2026-11-17): U-Net **18,312,013** parameters + adapter **66,048** (+ frozen MiniLM ≈ 22.7 M, inference only). Train state (weights + EMA + grads + 2 AdamW moments, fp32) ≈ **0.37 GB**. Peak at batch 64 is expected < 2 GiB. Measured on Day 4 and recorded in `runs/<id>/hw_report.json`. Per-stage parameters: time-MLP 98,816 · encoder 4,608,192 · bottleneck 3,087,360 · decoder 10,502,464 · head 7,629. |
| H10 | Measured CPU cost (reference build, 4 threads): **≈ 75 ms per guided forward pass** (batch 2 = cond + uncond, 16×16). A 112-wide level at full quality (13 windows × 50 steps × 2.2 with resampling) ≈ 1,430 passes ≈ 107 s on CPU → the CPU fallback uses **"fast mode": 25 DDIM steps, r = 0** (≈ 325 passes ≈ 25 s). |
| H9 | Laptop rules: AC power, NVIDIA "Performance" mode, lid open or external cooling; training stops if GPU temperature exceeds 87 °C for 60 s (watchdog via `nvidia-smi`/NVML). |

## 4. Software stack (pin on Day 0, record exact versions in `env/lock_<machine>.txt`)

| ID | Item | Choice |
|---|---|---|
| V1 | Python | **3.12** (PyTorch 2.15 drops 3.10; 3.12 is safe on both OSes) |
| V2 | PyTorch | **2.15.x** (current stable as of 2026-11; PyPI default wheel is CUDA 13.x) if the NVIDIA driver supports CUDA 13 (driver ≥ 580 series). **If either machine's driver is older and cannot be updated, pin both machines to 2.14.x + cu126** (2.14 is the last release with CUDA 12.x wheels). **Both machines must run the same torch minor version** (checkpoint/state-dict compatibility and comparable numerics). sm_86 and sm_89 are supported by all of these. |
| V3 | Text encoder | `sentence-transformers/all-MiniLM-L6-v2` (384-d, frozen), loaded via `transformers` `AutoModel` + mean-free token outputs |
| V4 | UI | Streamlit **≥ 1.56, < 2** (latest is 1.66, Oct 2026). `st.components.v1.html` was deprecated in 1.56 in favour of `st.iframe`. The game embed goes through a small shim (doc 28) that prefers `st.iframe` and falls back to `components.v1.html`. |
| V5 | Game engine | **Phaser 3.90.0** (final v3 release; MIT; Arcade Physics), vendored as `app/static/vendor/phaser.min.js`. Phaser 4 exists but v3 is chosen for API stability and documentation depth. |
| V6 | Optional evaluator | Java 8+ JRE for `MarioEval.jar` (Baumgarten A*, paper-parity playability). Not needed by the app. |
| V7 | Tests | `pytest`, `hypothesis` (property tests for the scheduler) |
| V8 | Tracking | local CSV + JSONL + TensorBoard (no cloud account needed). Optional W&B. |

## 5. Data decisions

| ID | Decision |
|---|---|
| D1 | Source of truth: `Mar1and2_LevelsAndCaptions-regular.json` at a **pinned MarioDiffusion commit SHA** (recorded on Day 1 in `data/MANIFEST.json` with sha256). The repo's own train/validate/test files are **not used** (leaky, F17). |
| D2 | Level IDs reconstructed by chain detection → `level_id` 0..41, `game` ∈ {SMB1, SMB2}, `col_offset` (window start column within its level), `level_name` (certain, F20). |
| D3 | **Splits are by level (grouped), never by window.** Test = **6 levels** (3 SMB1 + 3 SMB2). Dev = **5 levels**. Train = **31 levels**. **Family rule:** all members of a near-duplicate family (F21) go to train only, so dev/test are drawn from the 27 family-free levels. Deterministic exhaustive search (doc 10 §5; seed **20261117** only for the random fallback). Example output (must be regenerated, not hard-coded): test 1,073 windows (13.96 %), dev 905 (11.77 %), train 5,709 (74.27 %). |
| D4 | **Test levels are locked** after Day 2: sha256 of the split file is written to `data/splits/LOCK`. The test set is evaluated **once**, on Day 16, with the frozen final config. |
| D5 | Cross-validation: **level-grouped 5-fold CV** (stratified by game) over the 36 non-test levels, used for (a) final-config variance estimates (error bars) and (b) the hyperparameter robustness check. Fold runs are independent units of work that can go to either GPU. |
| D6 | Window boundary rule: since windows never cross levels, no purge gap is needed between levels. Within the dev/test sets, windows are kept at stride 1 for loss metrics, and **sub-sampled at stride 8** for generative metrics (reduces correlated duplicates). |
| D7 | Representation: one-hot (13, 16, 16), scaled to {−1, +1} (`x = 2·onehot − 1`). Decoding = `argmax` over channels. |
| D8 | Augmentation (on the fly): (a) **caption phrase shuffle** (always), (b) **phrase dropout** p = 0.10 per phrase, keeping at least one phrase, (c) **casual-paraphrase substitution** p = 0.30 using only the *train half* of the paraphrase lexicon, (d) CFG null-caption dropout p = 0.10 (applied last, overrides). **No horizontal flip by default** (breaks rightward level-design semantics); flip-with-remap is an ablation only. |
| D9 | Concept-balanced sampling: off in the baseline; on as an ablation (inverse-sqrt concept frequency weights) if rare-concept adherence < 50 % on dev. |

## 6. Model decisions

| ID | Decision |
|---|---|
| M1 | ε-prediction DDPM, **T = 1000**. |
| M2 | Schedule: **both** linear (β 1e-4 → 0.02) and cosine (s = 0.008, β clipped at 0.999) are implemented. Phase 3 trains both; pick by the dev composite score; **tie → cosine**. |
| M3 | U-Net: base **64**, channel multipliers **(1, 2, 4)** → 64 / 128 / 256 at 16×16 / 8×8 / 4×4; **2 ResBlocks per level** (encoder), **3 per level** (decoder, including skip concat); GroupNorm with **8 groups**; SiLU; dropout **0.1** in ResBlocks; Conv2d(64, 13) output. |
| M4 | Time embedding: sinusoidal 128-d → MLP(128 → 256 → 256, SiLU) → per-ResBlock `Linear(256, 2C)` scale-shift (FiLM / AdaGN). |
| M5 | **Self-attention at 8×8 and 4×4 from the start** (4 heads, 32-d heads). (The PDF only adds it as a rollback. It is cheap at this size and fixes broken pipes up front.) |
| M6 | **Cross-attention at 8×8 and 4×4** (4 heads), Q from spatial features, K/V from the projected text tokens (d_ctx = 128), with a key-padding mask. |
| M7 | Text path (max_len 48 is **provisional**: on Day 1, measure the WordPiece length of every caption including augmented variants; if the maximum exceeds 48, set 64 and log it): frozen MiniLM token states (B, L ≤ 48, 384) → trainable adapter `Linear(384,128) → SiLU → Linear(128,128)` → LayerNorm. Null caption = the **embedding of ""** (the [CLS][SEP] tokens), computed by the same frozen encoder. Unconditional-phase models use a constant learned null context. |
| M8 | Text encoding runs **online** (frozen, `no_grad`, bf16) with a string-keyed LRU cache (captions repeat heavily). Pre-caching is optional. |
| M9 | Loss: **L = MSE(ε, ε̂)** (PDF-faithful baseline). Option **+ λ·CE(x̂₀ logits, y)** with λ = 0.001 (paper-faithful, Lee & Simo-Serra 2023), weighted by ᾱ_t. Channel-weighted MSE (air 0.5, rare 2.0) is a **rollback**, not a default. |
| M10 | Optimiser: AdamW, lr **2e-4**, betas (0.9, 0.999), weight decay **0.01** (not on norms, biases or embeddings), warm-up **1,000 steps**, cosine decay to **1e-5**, grad-clip **1.0**, batch **64**. |
| M11 | EMA: decay **0.999** with warm-up `min(0.999, (1+n)/(10+n))`; 0.995 (PDF) is ablated. **All evaluation uses EMA weights.** |
| M12 | Budgets are in **optimizer steps**, not epochs (with stride-1 windows, one "epoch" shows each column about 16 times). Phase 3: **15k steps**. Phase 4: up to **60k steps**, early-stopped on the dev composite score (evaluated every 2k steps, patience 5). One epoch on the train split ≈ **89 steps** (5,709 windows in the example split / 64; recompute after the real split). |
| M13 | Inference: DDIM, **50 steps**, η = 0, `clip x̂₀ to [−1, 1]`; CFG convention `ε̂ = ε_u + w(ε_c − ε_u)` (**w = 1 means no guidance**). Default w is chosen on dev from {1, 2, 3, 5, 7.5, 10}; the PDF's prior is 5.0, the paper used 7.5. |

## 7. Full-level generation

| ID | Decision |
|---|---|
| G1 | Window 16 × 16, **stride 8** (overlap 8), RePaint-style known-region replacement at every DDIM step plus **r = 2 resampling steps** for the first 60 % of steps. |
| G2 | Widths offered: 16, 48, 80, 112 tiles of generated content. |
| G3 | **Level framing (deterministic, not generated):** prepend 3 columns of full floor (spawn zone); append a 6-column end template (floor + staircase + flag). The report card states that framing is templated. |
| G4 | Multi-tile sanitizer (pipes `<>`/`[]`, cannons `B`/`b`): **detect always, repair optionally**. Metrics are reported **before repair**. |
| G5 | MVP verification: the fast playability checker runs on the framed level. If it fails, the failing window is re-sampled with a new seed (max **5** attempts per window, max **20** in total) → badge shows "verified beatable" or "not verified". This is the seam for the future ghost-bot feature. |

## 8. Evaluation decisions

| ID | Decision |
|---|---|
| E1 | Primary metrics: **C-score** (paper formula, averaged over the **18 topics** of F23 for paper parity; always reported beside its null ladder, because absent/absent topics earn credit and scores saturate), **exact-match per concept**, **broken-structure rate**, **playability rate** (fast checker; MarioEval.jar on a 300-level subset for agreement), **novelty** = NN Hamming distance to the train split, **copy rate** = share of samples with NN distance ≤ 4, **diversity** = mean pairwise Hamming within same-prompt groups, **tile-distribution JS divergence**. |
| E2 | Prompt sets: (P-dev) captions of dev windows at stride 8; (P-test) captions of test windows at stride 8; (P-rand) 100 random captions (paper protocol, regenerated with our seed); (P-casual) 60 handwritten casual prompts with expected attributes (30 dev / 30 test, split on Day 2). |
| E3 | Statistics: 3 sampling seeds × prompt set; **95 % bootstrap CIs** (10,000 resamples, grouped by prompt); paired comparisons use paired bootstrap; CV results reported as mean ± std over 5 folds. |
| E4 | Dev composite score used for checkpoint/CFG selection: `S = 0.4·Cscore + 0.2·(1 − broken) + 0.2·playable + 0.2·min(1, novelty_med / novelty_ref)`, where `novelty_ref` = median NN distance of real dev windows to train (the natural baseline; 13 in the example split). S is a ranking device only, never a reported quality claim. **Phase-3 variant** (no prompts): `S_P3 = 0.4·floor_valid + 0.2·(1 − broken) + 0.2·playable + 0.2·min(1, novelty_med / novelty_ref)`. If the checker isn't ready, drop `playable` and renormalise to 0.5 / 0.25 / 0.25 (logged). |

## 9. Gate thresholds (tightened from the PDF)

| Gate | Threshold |
|---|---|
| G-P1 data | All F-facts re-verified; 10 random renders correct; no level in two splits; family rule holds; **≤ 3 % of test windows within Hamming ≤ 2 of any train window** (these are auto-listed in `reused_segments.json`; the example split has 1.86 %, mostly generic flat-floor windows) |
| G-P2 scheduler | Oracle-ε DDIM reconstructs x₀ exactly (argmax agreement 100 %, MSE < 1e-6 in fp64); ᾱ_T < 1e-4 (linear) / < 1e-8 (cosine); x_999 mean within ±0.02 and std within 1 ± 0.02 |
| G-P3 unconditional | Floor validator **calibrated first on real dev windows** (must pass ≥ 95 % of them, F22/Q9); then ≥ **85 %** of 256 samples pass it; **broken-structure rate ≤ 5 %**; tile-JS to train ≤ 0.05 (warning above 0.025); **copy rate ≤ copy_rate_ref + 0.05**, where copy_rate_ref is measured on real dev windows vs train on Day 2 (≈ 0.175 by LOLO, F22). This replaces the original "≤ 10 %", which a perfect generator of the real distribution would fail. Pre-registered on 2026-11-18, before any model exists. |
| G-P4 conditional | Dev C-score ≥ **max(0.75, C_null_dev + 0.20)** at the best w (C_null_dev ≈ 0.51, F22); visible contrast in the PDF's two-prompt test; adapter parameters confirmed in the optimiser; CFG beats w = 1 (paired bootstrap, CI excludes 0) |
| G-P5 full level | Seam score (column-pair transition NLL at seams vs interior) ratio ≤ 1.3; 0 broken structures after the sanitizer |
| G-P5b playability | Fast checker agrees with MarioEval.jar on ≥ 85 % of 300 levels (or, if Java is unavailable, ≥ 90 % agreement with hand-labelled 50 levels) |
| G-P6 eval | All metrics computed on dev with CIs; test run executed exactly once |
| G-P7 app | Cold start < 20 s on the laptop, 112-wide generate + verify < 10 s on GPU (full quality), < 45 s in CPU fast mode (H10); game playable with keyboard in Chrome and Edge |

## 10. Timeline anchor

**21 days (Day 0 to Day 20).** Day 0 = environment. The PDF's 16-day milestone plan maps onto Days 1–16; Days 17–20 are report, hardening, buffer and viva rehearsal. Full plan: `03_TIMELINE_DAY_BY_DAY.md`; critical path: `04_CRITICAL_PATH_AND_DEPENDENCIES.md`.

## 11. Assets

| ID | Decision |
|---|---|
| A1 | Default sprite set: **Kenney "Pixel Platformer" (CC0, 18×18 tiles)**, fetched autonomously by `scripts/fetch_assets.py`, with sha256 recorded. Tile size is a parameter (`tile_px`). |
| A2 | Fallback that never blocks: `scripts/make_placeholder_sprites.py` draws 13 distinct procedural tiles with PIL. |
| A3 | Nintendo-derived sprites (e.g., the repo's `mapsheet.png`) are **not shipped by default**. A "classic skin" may be enabled locally at the user's discretion; it is never committed. |
| A4 | Phaser fetched from jsDelivr (`phaser@3.90.0/dist/phaser.min.js`) and vendored; licence file stored next to it. |

## 12. Repository

Name: **`text2mario`**. Layout in `05_REPO_STRUCTURE.md`. Root `CLAUDE.md` = `34_CLAUDE_CODE_OPERATING_RULES.md`.

## 13. Glossary of IDs used across docs

`F*` facts · `S*` scope · `H*` hardware · `V*` versions · `D*` data · `M*` model · `G*` generation · `E*` evaluation · `A*` assets · `G-P*` gates · `R*` risks (in `33_RISK_REGISTER.md`) · `X*` experiments (in `23_ABLATIONS_AND_EXPERIMENTS.md`).

## 14. Change log

| Date | Change | Reason | Docs touched |
|---|---|---|---|
| 2026-11-17 | v1.0 created | initial pack | all |
| 2026-11-18 | F12 refined; F20–F23 added | measured by the data/eval workers (docs 08, 09, 10, 22) | 08, 09, 10, 22 |
| 2026-11-18 | D2 names certain; D3 family rule + example split | F20, F21 | 10, 24 |
| 2026-11-18 | M7 max_len provisional; M12 ≈ 89 steps/epoch | unmeasured WordPiece length; example split = 5,709 train windows | 11, 15, 17 |
| 2026-11-18 | E1 C-score over 18 topics + null ladder; E4 adds S_P3 | paper parity; Phase 3 has no prompts | 22, 24 |
| 2026-11-18 | G-P1 leakage ≤ 3 %; G-P3 calibrated floor validator + relative copy-rate; G-P4 margin over null | literal thresholds were mis-calibrated against real data (F22). Pre-registered before any model exists | 22, 24, 25 |
| 2026-11-18 | V4 Streamlit ≥ 1.56 with `st.iframe` shim | `components.v1.html` deprecated in 1.56 | 27, 28 |
