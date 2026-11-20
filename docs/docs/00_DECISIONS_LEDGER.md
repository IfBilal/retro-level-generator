# 00 — Decisions Ledger (Single Source of Truth)

> **Rule for Claude Code:** If any other document in this pack contradicts this ledger, **this ledger wins**. If you change a decision, change it here first, add a row to §14 (Change Log), then update downstream docs. Every number used in code must come from `configs/*.yaml`, and every YAML value must trace back to a row here.

Pack version: **v1.1** (2026-11-18, schedule/deliverables/cloud amendment; see §14). Source blueprint: `Text-to-Mario_Level_Diffusion_Engineering_Pipeline.pdf` (6 pages, audited in `01_PDF_AUDIT.md`).

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
| F11 | Chain lengths (windows) | min 130, max 358; level width = windows + 15; **8,317 = Σ level widths = total column count (with repeats)**; the number of **distinct column patterns is 653** (doc 22 §3.11, doc 09 §9) | computed |
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
| S6 | **Deployment on Streamlit Community Cloud is MANDATORY** (course statement: "Streamlit deployment is MANDATORY"; rubric row 6 gives 5 of 135 marks to it). The old non-goal "no cloud deployment / localhost only" is **withdrawn**. Two run profiles exist: **local-GPU** (development path, offline-capable, used for training, evaluation and the live demo on each member's laptop) and **cloud-CPU** (§3b). |
| S7 | Scope is sized to the deadline: **6 Dec 2026 (Sunday), single zip on GCR; presentations 7-9 Dec.** Effort-to-marks mapping and the list of cut or demoted items are in doc 02 §14. MUST items: level-grouped split, playability checker, one-shot test, the MVP game, Streamlit Cloud deployment, the report, the 12-slide deck, the evidence folder, the submission zip. |
| S8 | Originality: the project **builds on** the published MarioDiffusion dataset and paper (§11b); what is external and what is the group's own work is stated in doc 02 §15 and in the report (doc 32). |

## 2b. Problem definition and users (P*)

| ID | Decision |
|---|---|
| P1 | **Problem.** Indie and hobbyist 2D platformer developers and level-design students need many *playable* level prototypes quickly. Hand-building a first-pass level in a tile editor is slow, and generic procedural generators are hard to steer in words and give no guarantee that the result can be finished. |
| P2 | **Users.** (a) Indie/hobbyist game developers and level designers who want a controllable first draft; (b) educators and students of procedural content generation who want to show how text-conditioned generation and a playability check fit together. The examiner is an **evaluator**, not a target user. **No user research was carried out; these are stated needs, not measured demand.** |
| P3 | **Input -> output.** Input: one English sentence (<= 300 characters) plus width, guidance scale and seed. Output: a full-width tile level (16 rows x (3 + width + 6) columns), a canonical caption showing what the model was actually asked, a "verified beatable" badge from the playability checker, a playable browser game, and exports (PNG, JSON, CSV, VGLC text, standalone HTML). |
| P4 | **Objectives and measurable success criteria** are the gates of §9 (G-P1..G-P8) plus the report metrics of E1; the headline criteria are: grouped-split test with CIs; dev C-score >= max(0.75, null + 0.20) (G-P4); broken-structure rate <= 5 % (G-P3); playability reported with its checker-validation agreement (G-P5b); app generates and verifies a level inside the (to-be-pinned) latency budget on both the local-GPU and cloud-CPU profiles (G-P7, G-P8). No number is a claim until measured. |


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
| H11 | **Planning throughput (assumptions, not measurements):** training speed at batch 64, bf16: **DT ≈ 15 it/s, LT ≈ 9 it/s** (one value, used by docs 04, 23, 07). Both are replaced by the measured `hw_report.json` values on Day 4 (`scripts/status.py` recomputes the schedule). Any wall-clock figure in the docs derived from these is likewise provisional. |
| H9 | Laptop rules: AC power, NVIDIA "Performance" mode, lid open or external cooling; training stops if GPU temperature exceeds 87 °C for 60 s (watchdog via `nvidia-smi`/NVML). |

## 3b. Cloud deployment profile (CL*)

All numbers marked *(provisional)* are assumptions to be measured on Day 12 (dry run) and then pinned here.

| ID | Decision |
|---|---|
| CL1 | **Profile switch:** `T2M_PROFILE=cloud` (set in the Streamlit Cloud app settings; also auto-selected when no CUDA is present and the repo is not on a development machine). Local-GPU remains the default development path. |
| CL2 | **CPU-only fast mode** is mandatory on cloud: DDIM 25 steps, r = 0 (H10), batch 1 window at a time, `torch.set_num_threads` from the config. Default width 48; width 112 is allowed but runs under the time guard (CL6). |
| CL3 | **Checkpoint:** EMA weights only, exported once at freeze (Day 14) as `app/model/t2m_ema.safetensors` (+ `t2m_ema.json` sidecar: sha256, config hash, step count). Size is arithmetic, not a measurement: (18,312,013 + 66,048) parameters x 4 bytes (fp32) = 73.5 MB, below GitHub's 100 MB per-file limit. If a larger variant is ever chosen, store fp16 (halves the size). No Git LFS. Loaded with `safetensors.torch.load_file` (no pickle). |
| CL4 | **Text encoder bundled:** MiniLM (22.7 M parameters) is committed as fp16 `.safetensors` (about 45 MB, 22.7 M x 2 bytes) with tokenizer files under `app/model/minilm/` and loaded with `local_files_only=True`; the app never downloads anything at start-up. Licence (Apache-2.0 per the model card, **re-check on Day 0**) text is stored beside it. A flaky Hub download can therefore not break the deployment. |
| CL5 | **Dependencies and memory:** `requirements.txt` pins every package to an exact version (CPU torch wheel via the PyTorch CPU index; no CUDA packages, no `transformers` extras beyond what MiniLM loading needs; no Playwright, pytest, hypothesis or TensorBoard in the cloud requirements, they live in `requirements-dev.txt`). Memory budget *(provisional)*: resident set <= 1 GB incl. model, encoder and Python; the real limit of the platform is **to be verified on Day 0** and the app measured against it on Day 12. |
| CL6 | **Cold start and fallback:** model and encoder load inside one `st.cache_resource` call; the first request after a platform wake-up may take longer than a warm one, so the UI shows a status line ("Loading model...") and the 12 cached showcase levels are selectable at once. **Time guard:** if a generation exceeds `cloud.time_budget_s` *(provisional, 60 s)* the run is aborted and the app serves the closest cached showcase level (matched by canonical phrase set), clearly labelled "cached showcase level (live generation exceeded the cloud time budget)". The fallback is logged (CLAUDE.md §2.12) and counted in the app log. |
| CL7 | **No secrets, no network at run time:** no API keys; Phaser, sprites, MiniLM, checkpoint and showcase levels are all inside the repo; the embedded game contains no external URLs. Usage statistics stay disabled. |
| CL8 | **Repository for deployment:** the deployed repo contains `app/`, `src/` (inference subset), `configs/`, `requirements.txt`, `.streamlit/config.toml`, the checkpoint, the showcase levels and licence files. It does not contain `data/raw/`, `runs/`, training checkpoints, optimizer state or the report. If the main repo is private, the Streamlit Cloud account must be authorised to read it; otherwise a public deployment mirror repo is used (decide on Day 0; the zip for GCR is unaffected). |
| CL9 | **Evidence:** the public URL, the commit SHA deployed, a screenshot of the running app and the G-P8 `RESULT.json` go into `evidence/deployment/`; the URL is also on the first page of the report and on slide 2. |

## 4. Software stack (pin on Day 0, record exact versions in `env/lock_<machine>.txt`)

| ID | Item | Choice |
|---|---|---|
| V1 | Python | **3.12 (verified locally 2026-11-20: CPython 3.12.16 via uv; a cp312 CPU torch wheel exists). Streamlit Community Cloud's Python options are STILL to be verified in the deployment dialog**: confirm that the chosen torch wheels exist for 3.12 on both OSes and that Streamlit Community Cloud offers the same Python version (advanced settings of the deployment dialog). If not, pick the newest version available in all three places and amend this row. |
| V2 | PyTorch | **2026-11-20: `pip index versions torch` lists 2.15.0 (latest) with a CPU wheel for cp312; development machine uses `torch==2.15.0+cpu`. The CUDA wheel of the same minor must still be verified against both GPU drivers.** Earlier text: **Version is NOT pinned in this pack: it is to be verified on Day 0 with `pip index versions torch`** (and `pip download --no-deps` of the intended wheel for both CUDA and CPU). Earlier drafts claimed "2.15.x current stable, 2.14.x + cu126 as fallback"; those claims could not be verified and are withdrawn. Rule that stays: pick one torch minor that has wheels for the driver of **both** local machines **and** a CPU wheel for the cloud profile (§3b); both machines run the same minor (state-dict compatibility, comparable numerics). Record the choice in `env/lock_<machine>.txt` and in this row. |
| V3 | Text encoder | `sentence-transformers/all-MiniLM-L6-v2` (384-d, frozen), loaded via `transformers` `AutoModel` + mean-free token outputs |
| V4 | UI | Streamlit **2026-11-20: `pip index versions streamlit` lists 1.67.0 as latest (pin on the app day, same version local and cloud)**; earlier text: to verify on Day 0 with `pip index versions streamlit`**; pin the exact version in `requirements.txt` (used by both the local and the cloud profile). Earlier drafts stated "latest is 1.66, `components.v1.html` deprecated in 1.56 in favour of `st.iframe`"; **unverified, treat as a hypothesis**. The game embed goes through a small shim (doc 28) that prefers `st.iframe` if the pinned version has it and otherwise uses `components.v1.html`. The cloud profile must use the same pinned version as the local one. |
| V5 | Game engine | **Phaser 3.90.0** (final v3 release; MIT; Arcade Physics), vendored as `app/static/vendor/phaser.min.js`. Phaser 4 exists but v3 is chosen for API stability and documentation depth. |
| V6 | Optional evaluator | Java 8+ JRE for `MarioEval.jar` (Baumgarten A*, paper-parity playability). Not needed by the app. |
| V7 | Tests | `pytest`, `hypothesis` (property tests for the scheduler) |
| V9 | Cloud runtime | Streamlit Community Cloud (profile §3b). Its Python versions, memory limit and sleep behaviour are **to be verified on Day 0 and Day 12** (not assumed). |
| V8 | Tracking | local CSV + JSONL + TensorBoard (no cloud account needed). Optional W&B. |

## 5. Data decisions

| ID | Decision |
|---|---|
| D1 | Source of truth: `Mar1and2_LevelsAndCaptions-regular.json` at a **pinned MarioDiffusion commit SHA** (recorded on Day 1 in `data/MANIFEST.json` with sha256). The repo's own train/validate/test files are **not used** (leaky, F17). |
| D2 | Level IDs reconstructed by chain detection → `level_id` 0..41, `game` ∈ {SMB1, SMB2}, `col_offset` (window start column within its level), `level_name` (certain, F20). |
| D3 | **Splits are by level (grouped), never by window.** Test = **6 levels** (3 SMB1 + 3 SMB2). Dev = **5 levels**. Train = **31 levels**. **Family rule:** all members of a near-duplicate family (F21) go to train only, so dev/test are drawn from the 27 family-free levels. Deterministic exhaustive search (doc 10 §5; seed **20261117** only for the random fallback). Example output (must be regenerated, not hard-coded): test 1,073 windows (13.96 %), dev 905 (11.77 %), train 5,709 (74.27 %). |
| D4 | **Test levels are locked** after Day 2: sha256 of the split file is written to `data/splits/LOCK`. The test set is evaluated **once**, on Day 15 (3 Dec 2026), with the frozen final config. |
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
| E3 | Statistics: 3 sampling seeds × prompt set; **95 % bootstrap CIs** (10,000 resamples) **clustered by held-out LEVEL** (the resampling unit is the level; all of its prompts and seeds travel together, because windows and prompts of one level are not independent and prompt-level resampling would understate the uncertainty); a prompt-level bootstrap is kept only as a secondary, clearly labelled "conditional on these levels" interval; paired comparisons use the paired level-clustered bootstrap. **With 5 dev and 6 test levels the intervals are wide; the report states this and shows per-level results next to every headline metric** (doc 22 §7.1, §7.3); CV results reported as mean ± std over 5 folds. |
| E4 | Dev composite score used for checkpoint/CFG selection: `S = 0.4·Cscore + 0.2·(1 − broken) + 0.2·playable + 0.2·min(1, novelty_med / novelty_ref)`, where `novelty_ref` = median NN distance of real dev windows to train (the natural baseline; 13 in the example split). S is a ranking device only, never a reported quality claim. **Phase-3 variant** (no prompts): `S_P3 = 0.4·floor_valid + 0.2·(1 − broken) + 0.2·playable + 0.2·min(1, novelty_med / novelty_ref)`. **Circularity rule:** the `playable` term may enter S (and so checkpoint/CFG selection) **only after the checker is validated**, i.e. `reports/gates/P5b/RESULT.json` has `pass: true` (agreement with MarioEval.jar or with 50 hand-labelled levels, G-P5b, **Day 7**). Until then (Phase 3 selection on Days 5-6, any earlier dev evaluation) `playable` is dropped and the weights renormalised to 0.5 / 0.25 / 0.25 and the run is tagged `S_noplay` (logged). First selection that uses `playable`: Day 8 onward, which is after G-P5b. |

## 9. Gate thresholds (tightened from the PDF)

| Gate | Threshold |
|---|---|
| G-P1 data | All F-facts re-verified; 10 random renders correct; no level in two splits; family rule holds; **≤ 3 % of test windows within Hamming ≤ 2 of any train window** (these are auto-listed in `reused_segments.json`; the example split has 1.86 %, mostly generic flat-floor windows) |
| G-P2 scheduler | Oracle-ε DDIM reconstructs x₀ exactly (argmax agreement 100 %, MSE < 1e-6 in fp64); ᾱ_T < 1e-4 (linear) / < 1e-8 (cosine); x_999 mean within ±0.02 and std within 1 ± 0.02 |
| G-P3 unconditional | Floor validator **calibrated first on real dev windows** (must pass ≥ 95 % of them, F22/Q9); then ≥ **85 %** of 256 samples pass it (real dev windows measured on 2026-11-20: 100 %); **broken-structure rate ≤ 5 %**; tile-JS to train ≤ 0.05 (warning above 0.025); **copy rate ≤ copy_rate_ref + 0.05**, where copy_rate_ref is measured on real dev windows vs train on Day 2 (≈ 0.175 by LOLO, F22; **measured on the locked split 2026-11-20: copy_rate_ref = 0.104, so the threshold is 0.154**, `reports/references/REFERENCES.json`, written before any model exists). This replaces the original "≤ 10 %", which a perfect generator of the real distribution would fail. Pre-registered on 2026-11-18, before any model exists. |
| G-P4 conditional | Dev C-score ≥ **max(0.75, C_null_dev + 0.20)** at the best w (C_null_dev ≈ 0.51, F22); visible contrast in the PDF's two-prompt test; adapter parameters confirmed in the optimiser; CFG beats w = 1 (paired bootstrap, CI excludes 0) |
| G-P5 full level | Seam score (column-pair transition NLL at seams vs interior) ratio ≤ 1.3; 0 broken structures after the sanitizer |
| G-P5b playability (checker validity) | Fast checker agrees with MarioEval.jar on ≥ 85 % of 300 levels (or, if Java is unavailable, ≥ 90 % agreement with 50 hand-labelled levels). **Day 7 (25 Nov), before the first use of the `playable` term in any selection (E4)**; the 300/50 levels come from train/dev real levels and Phase-3 unconditional samples only (never test levels). |
| G-P6 eval | All metrics computed on dev with CIs; test run executed exactly once |
| G-P7 app (local-GPU profile) | **Provisional** targets: cold start < 20 s on the laptop; 112-wide generate + verify < 10 s on GPU (full quality); < 45 s in CPU fast mode (H10); game playable with keyboard in Chrome and Edge. **Measure-first rule:** none of the GPU latency numbers has been measured. Per-pass GPU timing is recorded on Day 4 (`hw_report.json`), the full pipeline is timed on Day 12, and the thresholds are then **pinned** in this row (measured median x 1.25 headroom, never tighter than the measurement supports). A threshold changes only by editing this row and adding a §14 entry, never silently in `gate_p7.py`. |
| G-P8 cloud deployment | The app is deployed on Streamlit Community Cloud from the GitHub repo with the **frozen** checkpoint; the public URL loads, a typed prompt produces a level, the game plays, and the fallback to cached showcase levels triggers correctly when generation exceeds the cloud time budget (CL6). Dry run on **Day 12** (candidate checkpoint), final check **Day 14** (frozen checkpoint) and again on Days 16 and 17. Cloud latency and memory are **measured on Day 12 and then pinned** (same rule as G-P7); provisional cap CL5. |

## 10. Timeline anchor

**18-day plan, Day 0 = Wed 18 Nov 2026 to Day 18 = Sun 6 Dec 2026 (submission day).** The earlier 21-day plan (Day 0-20) overran the deadline (Day 20 would be 8 Dec) and was replaced on 2026-11-18. Day 0 = environment and data fetch; Days 1-14 = build, train, evaluate and freeze (Day 14 = 2 Dec); Day 15 = one-shot test (3 Dec); Day 16 = figures, report, slides, hardening; Day 17 = viva rehearsal, zip build and check; Day 18 = upload to GCR. Presentations: 7-9 Dec. Full plan: `03_TIMELINE_DAY_BY_DAY.md`; critical path: `04_CRITICAL_PATH_AND_DEPENDENCIES.md`; deliverable mapping: `35_SUBMISSION_CHECKLIST.md`.

## 11. Assets

| ID | Decision |
|---|---|
| A1 | Default sprite set: **Kenney "Pixel Platformer" (CC0, 18×18 tiles)**, fetched autonomously by `scripts/fetch_assets.py`, with sha256 recorded. Tile size is a parameter (`tile_px`). |
| A2 | Fallback that never blocks: `scripts/make_placeholder_sprites.py` draws 13 distinct procedural tiles with PIL. |
| A3 | Nintendo-derived sprites (e.g., the repo's `mapsheet.png`) are **not shipped by default**. A "classic skin" may be enabled locally at the user's discretion; it is never committed. |
| A4 | Phaser fetched from jsDelivr (`phaser@3.90.0/dist/phaser.min.js`) and vendored; licence file stored next to it. |

## 11b. Originality and attribution (O*)

| ID | Decision |
|---|---|
| O1 | **External (used, documented, justified):** the Mar1and2 level+caption dataset and the caption idea of **MarioDiffusion (Schrum et al., arXiv 2507.00184; MIT licence)**; the VGLC-derived levels underneath it; the frozen text encoder all-MiniLM-L6-v2; Phaser 3 (MIT); Kenney Pixel Platformer sprites (CC0); optional MarioEval.jar (Baumgarten A*). |
| O2 | **The group's own work:** the level-grouped split and leakage audit (F17, F21); the from-scratch conditional U-Net, noise schedules, DDPM/DDIM/CFG implementation and trainer; the caption-grammar normaliser for casual text; the physics-consistent playability checker and its validation; the sliding-window outpainting generator with framing and verified resampling; the browser game and the Streamlit studio/deployment; the evaluation protocol (null ladder, level-clustered CIs, one-shot test). |
| O3 | **Caption detector:** implemented **from the caption-grammar description** (doc 12 and the paper's published grammar), and then **checked against the stored captions** of the 7,687 scenes. MarioDiffusion's own detector source is **not ported or transcribed line by line**; the group may read it only to confirm what a phrase means, never to copy structure. Disagreements with the stored captions are reported, not hidden (timeline Day 1). |
| O4 | The report states the baseline openly: "we build on Schrum et al.'s published dataset and idea; our contributions are ..." (doc 32 §1, section 'Originality and attribution'). Viva answer: doc 32 §7 (C-series, question C7). |

## 12. Repository

Name: **`text2mario`**. Layout in `05_REPO_STRUCTURE.md`. Root `CLAUDE.md` = `34_CLAUDE_CODE_OPERATING_RULES.md`.

## 13. Glossary of IDs used across docs

`F*` facts · `S*` scope · `P*` problem definition · `CL*` cloud profile · `O*` originality · `H*` hardware · `V*` versions · `D*` data · `M*` model · `G*` generation · `E*` evaluation · `A*` assets · `G-P*` gates · `R*` risks (in `33_RISK_REGISTER.md`) · `X*` experiments (in `23_ABLATIONS_AND_EXPERIMENTS.md`).

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
| 2026-11-18 | §10 timeline: 21 days (Day 0-20) -> **18-day plan, Day 0 = 18 Nov ... Day 18 = 6 Dec**; D4 test day Day 16 -> Day 15 | the 21-day plan ended on 8 Dec, after the 6 Dec deadline | 02, 03, 04, 05, 07, 10, 12, 22-24, 26-28, 30, 32-34, README, CLAUDE.md |
| 2026-11-18 | S6 Streamlit Cloud mandatory; §3b cloud profile CL1-CL9; G-P8 added; G-P7 thresholds provisional with measure-first rule | statement + rubric row 6; GPU latency never measured | 02, 06, 24, 27, 28, 33, 35 |
| 2026-11-18 | §2b problem definition P1-P4; S7 scope sizing; S8 originality | rubric rows 1 and scope/originality deductions | 02, 32, README |
| 2026-11-18 | E3 bootstrap clustered by level; E4 playable term only after G-P5b; G-P5b moved to Day 7 | prompt-level CIs understate uncertainty; checker circularity | 22, 23, 24, 30, 32, 03 |
| 2026-11-18 | F11 clarified: 8,317 = total columns, 653 distinct patterns | the two numbers were conflated | 09, 32 |
| 2026-11-18 | V1, V2, V4 marked "to verify on Day 0" | torch 2.15 / Streamlit 1.66 / Python claims unverified | 06 |
| 2026-11-18 | H11 laptop 9 it/s and desktop 15 it/s as the single planning assumption | doc 04 said 9, doc 23 said 12 | 04, 23 |
| 2026-11-18 | §11b O1-O4 originality; detector implemented from grammar | academic-integrity requirement of the statement | 03, 12, 32 |
| 2026-11-18 | Docs 16-21, 25, 29, 31 written; 35 (submission checklist) and 36 (team ownership and viva) added; red-team docs withdrawn | STATUS.md listed them as missing | README, STATUS |
| 2026-11-20 | V1/V2/V4 verified as far as possible offline: Python 3.12.16, torch 2.15.0 (CPU wheel), Streamlit latest 1.67.0 | Day-0 verification; Streamlit Cloud Python options and CUDA wheels still open | 06 |
| 2026-11-20 | M7: max_len = 48 CONFIRMED with the real tokenizer: longest stored caption 40 tokens, longest of 10,000 augmented captions 39, `""` = 2 tokens; only 'staircases' splits into two pieces | `scripts/measure_tokens.py`, `reports/eda/tokens.json` | 11, 15 |
| 2026-11-20 | G-P3 copy-rate threshold pinned: copy_rate_ref = 0.104 (dev stride-8 windows vs train) -> **0.154**; floor_valid real = 100 %, real broken structures = 0, novelty_ref = 13, tile-JS real dev 0.0049 / all-sky 0.064 / noise 0.622 | pre-registered before any model exists (doc 22 §3.8) | 22, 24 |
| 2026-11-20 | F1-F19 and A-series re-measured on the pinned data: 39 / 39 checks pass; split regenerated and equals the doc 10 prototype exactly (test J 0.00296, dev J 0.00805, leakage 1.86 % / 1.66 %); 42 / 42 levels match VGLC | `scripts/audit_dataset.py`, `scripts/make_splits.py` | 09, 10 |
| 2026-11-20 | VGLC SMB1 8-3 has a stray trailing space on its last two rows; stripping it reproduces the chain (width 214) | build_index.py | 08, 09 |
| 2026-11-20 | Reference oracle moved `docs/reference/` -> `reference/` to match doc 05 | repo layout | 05 |
