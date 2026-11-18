# 05 — Repository Structure, Module Contracts and Conventions

> **Authority:** this document is authoritative for **where code and artifacts live** and for **public names** (modules, functions, scripts, config files). The topic doc listed per module is authoritative for **behaviour and internals**. If a topic doc uses a different public name for the same thing, implement the topic doc's behaviour under the name given here, and log it under "Doc conflicts" in `PROGRESS.md` (CLAUDE.md §1). Ledger IDs: H*, V*, D*, M*, G*, E*, A*.

Repository name: **`text2mario`** (ledger §12). Layout: `src/` package layout, Python 3.12 (V1).

---

## 1. Full tree

`[C]` = committed to git · `[I]` = git-ignored (generated or third-party) · `[Dn]` = the timeline day the item is first created.

```text
text2mario/
├── CLAUDE.md                         [C][D0]  operating rules (= docs/34)
├── README.md                         [C][D0]  how to install, run, reproduce
├── PROGRESS.md                       [C][D0]  daily log: Done / In-flight / Blocked / Next / Assumptions / Doc conflicts
├── HANDOFF.md                        [C][D0]  GPU run ownership; edited only by scripts/handoff.py (doc 07 §4)
├── FROZEN.json                       [C][D14] final config + checkpoint hashes (doc 30)
├── MODEL_CARD.md  DATA_CARD.md       [C][D16]
├── pyproject.toml                    [C][D0]  package metadata + ruff + pytest config (§7)
├── requirements.txt                  [C][D0]  exact runtime pins, torch excluded (§7.2)
├── requirements-dev.txt              [C][D0]  exact dev pins
├── .gitignore  .gitattributes        [C][D0]  (§6)
├── .pre-commit-config.yaml           [C][D0]  (§8)
├── Makefile  tasks.ps1               [C][D0]  task runners, Linux / Windows (§9)
├── .env.example                      [C][D0]  documented env vars (HF_HOME, T2M_MACHINE, ...)
├── .streamlit/config.toml            [C][D12] Streamlit server/browser settings (doc 28)
├── .cache/hf/                        [I][D0]  Hugging Face cache (HF_HOME) — MiniLM snapshot (doc 06 §8)
│
├── docs/                             [C][D0]  this pack (00–34)
├── reference/ref_unet.py             [C][D0]  numeric oracle for doc 14 (18,312,013 params)
│
├── src/text2mario/
│   ├── __init__.py                   version string only
│   ├── errors.py                     ConfigError, GateNotPassedError, LockMismatchError, CheckpointError, HandoffError
│   ├── data/
│   │   ├── tiles.py                  [D1]  THE tile table (F6); ids, chars, names, colours, solidity flags
│   │   ├── index.py                  [D1]  chain reconstruction (F10), level index dataframe
│   │   ├── manifest.py               [D1]  content hashing of data artifacts (canonical arrays)
│   │   ├── splits.py                 [D2]  level-grouped selection, CV folds, LOCK verification (doc 10)
│   │   ├── augment.py                [D2]  caption augmentation D8 (shuffle, phrase dropout, paraphrase, null)
│   │   └── dataset.py                [D2]  GPU-resident dataset + resumable batch sampler (H7, doc 11)
│   ├── captions/
│   │   ├── grammar.py                [D1]  canonical phrase grammar, parse/format, quantity buckets (F15, doc 12)
│   │   └── detector.py               [D1]  written from the caption grammar, checked against stored captions (O3, F19); no ported code
│   ├── text/
│   │   ├── encoder.py                [D7]  frozen MiniLM + adapter + LRU cache + null embedding (M7, M8, doc 15)
│   │   ├── normalizer.py             [D8]  casual → canonical, buckets, scene splitter (doc 12)
│   │   └── synonyms.yaml             [D8]  normaliser dictionary (content: doc 12)
│   ├── diffusion/
│   │   ├── scheduler.py              [D3]  schedules, q_sample, posterior, DDPM/DDIM steps, renoise (M1, M2, doc 13)
│   │   └── sampler.py                [D7]  guided DDIM sampling loop with CFG + known-region hook (M13, doc 20)
│   ├── models/
│   │   ├── embeddings.py             [D4]  sinusoidal time embedding + time MLP (M4)
│   │   ├── blocks.py                 [D4]  ResBlock (GroupNorm 8, SiLU, FiLM, dropout 0.1)
│   │   ├── attention.py              [D4]  SelfAttention2d, CrossAttention2d (SDPA, key-padding mask)
│   │   ├── unet.py                   [D4]  UNet (M3–M6), attention-map capture hook (seam F5)
│   │   └── ema.py                    [D4]  EMA with warm-up (M11)
│   ├── train/
│   │   ├── train.py                  [D4]  entry point: python -m text2mario.train.train --config ... (doc 16)
│   │   ├── loop.py                   [D4]  step function, accumulation fallback, graceful stop
│   │   ├── optim.py                  [D4]  AdamW param groups (no decay on norm/bias/emb), LR schedule (M10)
│   │   ├── checkpoint.py             [D4]  atomic save/load/verify, retention (doc 07 §5–6)
│   │   ├── validate.py               [D4]  deterministic per-t-bucket validation loss
│   │   └── early_stop.py             [D8]  E4-based early stopping state
│   ├── eval/
│   │   ├── structure.py              [D5]  floor validity, broken structures, tile JS divergence
│   │   ├── novelty.py                [D5]  NN Hamming to train (GPU), copy rate (E1)
│   │   ├── diversity.py              [D5]  within-prompt mean pairwise Hamming (E1)
│   │   ├── adherence.py              [D8]  C-score (paper formula) + per-concept exact match (E1)
│   │   ├── composite.py              [D8]  dev composite score (E4)
│   │   ├── stats.py                  [D8]  grouped bootstrap CIs, paired bootstrap (E3)
│   │   └── suites.py                 [D8]  run a named metric suite on a prompt set → JSON (doc 22)
│   ├── generate/
│   │   ├── full_level.py             [D11] generate_level, iter_windows (F9 seam), inpaint (F3 seam), resample loop (G1, G5)
│   │   ├── framing.py                [D11] deterministic start/end templates (G3)
│   │   └── sanitizer.py              [D11] multi-tile defect detection + optional repair (G4)
│   ├── playability/
│   │   ├── physics.py                [D5]  load/validate physics.json shared with the game
│   │   ├── checker.py                [D5]  A* over the tile graph with physics jump arcs (doc 21)
│   │   └── marioeval_bridge.py       [D6]  subprocess wrapper for MarioEval.jar (V6, optional)
│   ├── render/
│   │   ├── renderer.py               [D2]  tensor_to_grid, grid_to_image, grid_to_vglc_text, grid_to_json
│   │   ├── atlas.py                  [D2]  sprite atlas loading (kenney | placeholder | classic-local)
│   │   └── animate.py                [D12] denoising frames → GIF/strip
│   └── utils/
│       ├── env.py                    [D0]  bootstrap_env(): HF_HOME, offline flags, PYTHONUTF8 — call FIRST
│       ├── config.py                 [D0]  load/merge YAML, machine allow-list guard, config hash (doc 07 §3)
│       ├── hw.py                     [D0]  detect_machine, apply_vram_cap, hw_report (H2, H3, H8)
│       ├── watchdog.py               [D4]  thermal + AC-power watchdog (H9, doc 07 §10)
│       ├── seed.py                   [D0]  seeded generators, derive_seed (no global RNG in library code)
│       ├── io.py                     [D0]  atomic writes, sha256, canonical JSON
│       ├── runid.py                  [D4]  run ID construction (doc 07 §2)
│       ├── handoff_state.py          [D4]  read/write the HANDOFF.md state block (doc 07 §4)
│       ├── log.py                    [D4]  RunLogger: metrics.csv, events.jsonl, TensorBoard
│       └── git.py                    [D0]  git SHA, dirty flag, changed paths
│
├── app/
│   ├── streamlit_app.py              [D12] studio entry point (doc 28)
│   ├── services/model_bundle.py      [D12] cached loader (model, EMA weights, encoder, schedule)
│   ├── components/
│   │   ├── game.py                   [D13] build_game_html(level_json, atlas_b64, physics) -> str (doc 27)
│   │   ├── badges.py                 [D13] verified badge, attempt counter
│   │   └── about.py                  [D12] "About this model" panel (FR-A-12)
│   ├── static/
│   │   ├── vendor/phaser.min.js      [C][D0] Phaser 3.90.0 (A4)
│   │   ├── vendor/LICENSE-phaser.txt [C][D0]
│   │   ├── game/template.html        [C][D3] game page template (placeholders filled by game.py)
│   │   ├── game/game.js              [C][D3] Phaser scene code
│   │   ├── game/physics.json         [C][D3] shared physics constants (game + checker)
│   │   ├── sprites/kenney/           [C][D0] CC0 atlas + LICENSE (A1)
│   │   ├── sprites/placeholder/      [C][D0] generated by make_placeholder_sprites.py (A2)
│   │   └── sprites/classic_local/    [I]     optional, never committed (A3)
│   └── showcase/                     [C][D16] 12 cached levels: *.json + *.png
│
├── configs/
│   ├── base.yaml                     [D0]  every default value, traceable to the ledger (doc 17)
│   ├── machine/desktop_3060ti.yaml   [D0]  identity + paths + allow-listed overrides only (doc 07 §3)
│   ├── machine/laptop_4050.yaml      [D0]
│   ├── uncond_linear.yaml            [D5]  overlay on base
│   ├── uncond_cosine.yaml            [D5]
│   ├── cond_smoke.yaml               [D7]
│   ├── cond_main.yaml                [D8]
│   ├── ablations/x01_aux_ce.yaml … x12_*.yaml   [D10/D15] (ids per doc 23)
│   ├── cv_fold.yaml                  [D15] fold index passed with --fold k
│   ├── eval/dev.yaml  eval/test.yaml [D8/D15]
│   ├── generate/app.yaml             [D12] app generation defaults (w, steps, fast mode)
│   └── final.yaml                    [D14] frozen experiment config (hash in FROZEN.json)
│
├── scripts/                          (all are CLIs: python scripts/<name>.py --help)
│   ├── verify_env.py                 [D0]  doc 06 §9
│   ├── install_torch.ps1  install_torch.sh   [D0]  V2 decision procedure (doc 06 §5)
│   ├── fetch_models.py               [D0]  MiniLM snapshot at a pinned revision → HF_HOME
│   ├── fetch_assets.py               [D0]  Phaser + Kenney, sha256 manifest (doc 26)
│   ├── make_placeholder_sprites.py   [D0]  A2
│   ├── compare_env.py                [D0]  diff env/lock_*.txt between machines
│   ├── fetch_data.py                 [D1]  pinned MarioDiffusion commit → data/raw (doc 08)
│   ├── audit_dataset.py              [D1]  asserts F1–F17, writes data/processed/index.parquet
│   ├── eda.py                        [D1]  EDA figures + reports/eda/EDA.md
│   ├── make_splits.py                [D2]  split_v1.json + LOCK
│   ├── gate_p1.py … gate_p8.py, gate_p5b.py   [per phase; p8 = cloud deployment]  each writes reports/gates/PN/RESULT.json
│   ├── bench.py                      [D4]  hw_report.json (peak VRAM, it/s)
│   ├── launch.py                     [D4]  create run dir, snapshot config, register in HANDOFF, start trainer
│   ├── queue.py                      [D5]  sequential job queue per machine (exit-code aware)
│   ├── handoff.py                    [D4]  release | acquire | status | recover | publish (doc 07)
│   ├── verify_ckpt.py                [D4]  integrity + schema check of a checkpoint
│   ├── status.py  status_tasks.yaml  [D4]  daily critical-path report (doc 04 §8)
│   ├── eval_dev.py                   [D8]  dev suites; --watch mode; --merge shards
│   ├── sweep_cfg.py                  [D9]  CFG sweep (shardable)
│   ├── aggregate_cv.py               [D15]
│   ├── eval_test.py                  [D15] ONE-SHOT test evaluation (guarded by LOCK + FROZEN)
│   ├── export_level.py               [D12] CLI export of a level JSON to PNG/CSV/VGLC/HTML
│   ├── smoke_e2e.py                  [D13] Playwright end-to-end smoke (20 prompts)
│   ├── prune_runs.py                 [D5]  retention policy enforcement (doc 07 §6)
│   ├── export_cloud_ckpt.py          [D12/D14] EMA weights -> app/model/t2m_ema.safetensors + sidecar (ledger CL3)
│   ├── gate_p8.py                    [D12/D14] checks the public Streamlit Cloud URL (doc 24 G-P8)
│   ├── collect_evidence.py           [D16] builds evidence/ (doc 35 §3)
│   ├── build_submission_zip.py       [D17] single GCR zip (doc 35 §4)
│   ├── make_figures.py               [D16] every report figure from saved artifacts
│   └── hooks/forbid_artifacts.py     [D0]  pre-commit guard (§8)
│
├── tests/
│   ├── conftest.py                   fixtures: tiny synthetic dataset, cpu device, tmp run dir
│   ├── test_tiles.py  test_index.py  test_manifest.py  test_splits.py  test_dataset.py  test_augment.py
│   ├── test_renderer.py  test_detector.py  test_grammar.py
│   ├── test_scheduler.py             (hypothesis property tests; oracle DDIM)
│   ├── test_unet.py  test_ema.py  test_text_encoder.py  test_optimizer_groups.py
│   ├── test_checkpoint.py  test_config_guard.py  test_handoff_state.py  test_watchdog.py
│   ├── test_normalizer.py  test_sanitizer.py  test_framing.py  test_full_level.py
│   ├── test_checker.py  test_metrics.py  test_stats.py  test_level_schema.py  test_exports.py
│   ├── test_app_smoke.py             (Streamlit AppTest; slow)
│   ├── test_app_fuzz.py              (slow)
│   └── e2e/test_game_playwright.py   (slow, e2e)
│
├── data/
│   ├── MANIFEST.json                 [C][D1] commit SHA, file sha256, content hashes
│   ├── raw/                          [I][D1] third-party clone (read files only; never import)
│   ├── processed/                    [I][D1] index.parquet, scenes_u8.npy (content-hashed in MANIFEST)
│   ├── splits/split_v1.json          [C][D2]
│   ├── splits/LOCK                   [C][D2] sha256 of split_v1.json (D4)
│   └── prompts/                      [C][D2] p_casual.yaml (60, dev/test tagged), p_rand.json, lexicon_train.yaml, lexicon_test.yaml
│
├── runs/                             [I]     one directory per run (doc 07 §2, §5)
├── reports/
│   ├── eda/                          [C]     EDA.md + figures
│   ├── gates/P1 … P7, P5b/           [C]     RESULT.json + evidence images per gate
│   ├── dev/                          [C]     dev evaluation JSON + tables
│   ├── cv/                           [C]
│   ├── final/                        [C]     RESULTS.md, test_invocations.jsonl
│   └── figures/                      [C]     generated by make_figures.py
├── assets/
│   ├── ASSET_MANIFEST.json           [C][D0] url, sha256, licence for each fetched file
│   ├── tile_map.json                 [C][D1] tile id → atlas frame (doc 26)
│   ├── contact_sheets/               [C][D1]
│   └── LICENSES/                     [C][D0]
└── env/
    ├── lock_desktop.txt  lock_laptop.txt   [C][D0] pip freeze per machine
    ├── capabilities.json             [C][D0] keyed by machine profile (doc 06 §9.4)
    ├── models.json                   [C][D0] HF repo id, pinned revision, file sha256
    └── verify/                       [C]     verify_env JSON reports per machine and date
```

## 2. Package import rules

1. Library code lives only in `src/text2mario/`. Scripts in `scripts/` and the app in `app/` import from the package. The package never imports from `scripts/` or `app/`.
2. Dependency direction (no cycles): `utils` ← `data`, `captions` ← `text` ← `diffusion`, `models` ← `train`, `eval`, `generate` ← `playability`, `render` (`playability` and `render` depend only on `data.tiles` and `utils`).
3. `text2mario.utils.env.bootstrap_env()` is the **first call** in every entry point (scripts, the trainer, the app), before anything imports `transformers` or `huggingface_hub`. Those libraries read `HF_HOME` and the offline flags at import time.
4. Nothing under `data/raw/` is ever imported (CLAUDE.md §2.10).
5. `torch.cuda` is touched only through `utils.hw` (device selection, VRAM cap). Library functions take a `device` argument and never hard-code `"cuda"`.

## 3. Shape, dtype and naming conventions

| Object | Type | Shape | Values | Notes |
|---|---|---|---|---|
| Tile grid | `np.ndarray[np.uint8]` or `torch.uint8` | `(16, W)` or `(B, 16, W)` | 0..12 (F6) | Row 0 = top. Rows 0–1 are padding rows (F12). Column 0 = left. |
| Clean model input | `torch.float32` | `(B, 13, 16, W)` | {−1, +1} | `x = 2·onehot − 1` (D7) |
| Noisy input / ε | `torch.float32` (bf16 inside autocast) | `(B, 13, 16, W)` | ℝ | — |
| Timesteps | `torch.int64` | `(B,)` | 0..999 | T = 1000 (M1) |
| Raw text states | `torch.bfloat16`/`float32` | `(B, L, 384)` | — | L ≤ 48 (M7) |
| Text context | `torch.float32` | `(B, L, 128)` | — | After adapter + LayerNorm |
| Context mask | `torch.bool` | `(B, L)` | True = real token (attend) | Converted to an SDPA mask inside attention |
| Captions | `list[str]` | B | canonical grammar (doc 12) | `""` = null caption |
| Level JSON | `dict` | — | schema v1 (doc 29) | Always carries provenance (FR-A-15) |

- Docstrings state shapes in this notation: `x: (B, 13, 16, W) float32 in {-1,+1}`.
- Randomness: library functions take a `torch.Generator` or `np.random.Generator` argument. No `torch.manual_seed`/`np.random.seed` inside `src/` except `utils.seed.seed_everything` (called by entry points only).
- Names: modules `snake_case`, classes `PascalCase`, constants `UPPER_SNAKE`, config keys `snake_case` in dotted namespaces (`optim.lr`, `perf.cudnn_benchmark`).
- File names: checkpoints `ckpt_step{N:07d}.pt` (+ `.json` sidecar), EMA weights `ema_step{N:07d}.safetensors`, figures `fig_<section>_<name>.png`, gate evidence `reports/gates/P{n}/<check>.png`.
- Text I/O: always `encoding="utf-8"`; paths via `pathlib.Path`; JSON written with `utils.io.canonical_json` (sorted keys) when it will be hashed.

## 4. Module contracts (public API)

Signatures are Python 3.12 type-hinted. "→ doc NN" names the topic doc that governs behaviour.

### 4.1 `data/` → docs 08–11
```python
# tiles.py  (single source of truth for F6 — never retype the order elsewhere)
NUM_TILES: int = 13
TILE_CHARS: tuple[str, ...] = ("-", "<", ">", "?", "B", "E", "Q", "S", "X", "[", "]", "b", "o")
TILE_NAMES: tuple[str, ...]          # "sky", "pipe_top_left", ..., "coin"
CHAR_TO_ID: dict[str, int]; ID_TO_CHAR: dict[int, str]
SOLID_IDS: frozenset[int]            # solidity flags derived from smb.json descriptors
def grid_from_chars(lines: list[str]) -> np.ndarray: ...       # (16, W) uint8
def grid_to_chars(grid: np.ndarray) -> list[str]: ...

# index.py
def reconstruct_chains(scenes: np.ndarray) -> np.ndarray: ...  # (N,16,16) uint8 -> (N,) int64 level_id 0..41
def build_index(dataset_json: Path) -> pd.DataFrame: ...        # columns: idx, level_id, game, col_offset, caption, n_phrases, scene_hash, is_dup

# manifest.py
def content_hash(arrays: dict[str, np.ndarray]) -> str: ...     # sha256 over canonical bytes (name-sorted, C-order, dtype-tagged)

# splits.py
@dataclass(frozen=True)
class SplitSpec: train: tuple[int, ...]; dev: tuple[int, ...]; test: tuple[int, ...]; folds: tuple[tuple[int, ...], ...]; seed: int
def select_split(index: pd.DataFrame, seed: int = 20261117) -> SplitSpec: ...
def load_split(path: Path, verify_lock: bool = True) -> SplitSpec: ...   # raises LockMismatchError

# augment.py
def augment_caption(caption: str, gen: torch.Generator, cfg: Mapping) -> str: ...  # D8 order: shuffle, dropout, paraphrase, null

# dataset.py
class GpuTileDataset:
    def __init__(self, scenes_u8: torch.Tensor, captions: list[str], level_ids: torch.Tensor, device: torch.device): ...
    def batch(self, idx: torch.Tensor) -> tuple[torch.Tensor, list[str]]: ...  # x: (B,13,16,16) float32 {-1,+1}
class ResumableBatchSampler:
    def __init__(self, n: int, batch_size: int, gen: torch.Generator): ...
    def next_indices(self) -> torch.Tensor: ...          # (B,) int64, CPU
    def peek_indices(self, k: int = 8) -> list[int]: ... # for the resume fingerprint (doc 07 §8)
    def state_dict(self) -> dict: ...; def load_state_dict(self, s: dict) -> None: ...
```

### 4.2 `captions/` → docs 09, 12, 22
```python
# grammar.py
QUANTITY_BUCKETS: tuple[str, ...] = ("one", "two", "a few", "several", "many")   # F15
def bucket(n: int) -> str: ...                                   # 1→one, 2→two, 3–4→a few, 5–9→several, ≥10→many
def parse_caption(caption: str) -> dict[str, str]: ...          # concept -> quantity/descriptor
def format_caption(concepts: Mapping[str, str], order: Sequence[str] | None = None) -> str: ...
# detector.py
def assign_caption(grid: np.ndarray, return_details: bool = False) -> str | tuple[str, dict[str, list[tuple[int, int]]]]: ...
```

### 4.3 `text/` → docs 12, 15
```python
# encoder.py
class TextConditioner(torch.nn.Module):
    def __init__(self, model_dir_or_id: str, revision: str, d_ctx: int = 128, max_len: int = 48, cache_size: int = 8192): ...
    def forward(self, captions: list[str]) -> tuple[torch.Tensor, torch.Tensor]: ...   # (B,L,128), (B,L) bool
    def trainable_parameters(self) -> list[torch.nn.Parameter]: ...                    # adapter + LayerNorm only
# normalizer.py
@dataclass
class NormalizedPrompt: raw: str; scenes: list[str]; canonical: str; unmapped: list[str]; alignment: list[tuple[tuple[int, int], int]]
def normalize(prompt: str, n_scenes: int | None = None) -> NormalizedPrompt: ...
def split_scenes(prompt: str) -> list[str]: ...
```

### 4.4 `diffusion/` → docs 13, 20
```python
# scheduler.py
class NoiseSchedule:                                           # buffers stored in float64, cast at use
    @classmethod
    def from_config(cls, kind: Literal["linear", "cosine"], T: int = 1000, **kw) -> "NoiseSchedule": ...
    betas: torch.Tensor; alphas: torch.Tensor; alpha_bar: torch.Tensor
def q_sample(s: NoiseSchedule, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor: ...
def predict_x0_from_eps(s: NoiseSchedule, xt: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor: ...
def p_mean_variance(s: NoiseSchedule, xt, t, eps, clip_x0: bool = True) -> tuple[torch.Tensor, torch.Tensor]: ...
def ddpm_step(s, xt, t, eps, gen: torch.Generator) -> torch.Tensor: ...
def ddim_step(s, xt, t, t_prev, eps, eta: float = 0.0, gen: torch.Generator | None = None, clip_x0: bool = True) -> torch.Tensor: ...
def renoise(s, x_prev, t_prev, t, gen: torch.Generator) -> torch.Tensor: ...    # q(x_t | x_{t_prev}) for RePaint resampling
def ddim_timesteps(s, n_steps: int) -> torch.Tensor: ...                          # descending int64
# sampler.py
def sample(model, cond: tuple[torch.Tensor, torch.Tensor] | None, null: tuple[torch.Tensor, torch.Tensor],
           shape: tuple[int, int, int, int], s: NoiseSchedule, steps: int, w: float, gen: torch.Generator,
           known: torch.Tensor | None = None, known_mask: torch.Tensor | None = None,
           resample_r: int = 0, resample_frac: float = 0.6, callback=None) -> torch.Tensor: ...  # returns x0 (B,13,16,W)
```

### 4.5 `models/` → doc 14
```python
class UNet(torch.nn.Module):
    def __init__(self, in_ch: int = 13, base: int = 64, mult: tuple[int, ...] = (1, 2, 4), n_res_enc: int = 2, n_res_dec: int = 3,
                 groups: int = 8, dropout: float = 0.1, heads: int = 4, d_ctx: int | None = 128, attn_levels: tuple[int, ...] = (1, 2)): ...
    def forward(self, x: torch.Tensor, t: torch.Tensor, ctx: torch.Tensor | None, ctx_mask: torch.Tensor | None) -> torch.Tensor: ...
    def capture_attention(self, enabled: bool) -> None: ...   # seam F5; stores cross-attn maps when enabled
class EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999): ...
    def update(self, model: torch.nn.Module) -> None: ...      # decay_n = min(decay, (1+n)/(10+n))
    def state_dict(self) -> dict: ...; def load_state_dict(self, s: dict) -> None: ...
    def copy_to(self, model: torch.nn.Module) -> None: ...
```
Unit-test invariant: `sum(p.numel() for p in UNet().parameters()) == 18_312_013` (H8), matching `reference/ref_unet.py`.

### 4.6 `train/` → docs 16–18, 07
```python
def main(argv: list[str] | None = None) -> int: ...      # exit codes in doc 07 §10.4
def build_optimizer(params_decay, params_no_decay, cfg) -> torch.optim.AdamW: ...
def lr_at(step: int, cfg) -> float: ...                   # warm-up 1,000 → cosine to 1e-5 (M10)
def save_checkpoint(run_dir: Path, state: dict, step: int) -> Path: ...   # atomic; returns path
def load_checkpoint(path: Path, map_location: str = "cpu") -> dict: ...   # torch.load(weights_only=True)
def verify_checkpoint(path: Path) -> "CkptReport": ...
def deterministic_val_loss(model, data, s: NoiseSchedule, cfg, n_buckets: int = 10) -> dict[str, float]: ...
```

### 4.7 `eval/` → doc 22
```python
def floor_valid(grid: np.ndarray) -> bool: ...
def broken_structures(grid: np.ndarray) -> dict[str, int]: ...             # {"pipe": n, "cannon": m}
def tile_js_divergence(grids: np.ndarray, ref_hist: np.ndarray) -> float: ...
def nn_hamming(samples: torch.Tensor, train: torch.Tensor, chunk: int = 4096) -> torch.Tensor: ...  # (N,) int64
def copy_rate(dists: torch.Tensor, thr: int = 4) -> float: ...
def mean_pairwise_hamming(group: torch.Tensor) -> float: ...
def cscore(prompt_caption: str, sample_caption: str) -> float: ...          # in [-1, 1]
def exact_match(prompt_caption: str, sample_caption: str) -> dict[str, bool]: ...
def dev_composite(m: Mapping[str, float], novelty_ref: float) -> float: ... # E4
def bootstrap_ci(values: np.ndarray, groups: np.ndarray, n: int = 10_000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]: ...
```

### 4.8 `generate/`, `playability/`, `render/` → docs 20, 21, 27, 29
```python
# full_level.py
@dataclass
class LevelResult: raw_grid: np.ndarray; display_grid: np.ndarray; framed_grid: np.ndarray; templated_mask: np.ndarray
                   verified: bool; attempts: int; check: "CheckResult"; provenance: dict; frames: list[np.ndarray]
def generate_level(prompt: str | NormalizedPrompt, width: int, seed: int, w: float, steps: int, fast_mode: bool, bundle) -> LevelResult: ...
def iter_windows(captions: list[str], seed: int, w: float, steps: int, bundle): ...                # yields (col0, grid16) — seam F9
def inpaint(grid: np.ndarray, mask: np.ndarray, caption: str, seed: int, w: float, steps: int, bundle) -> np.ndarray: ...  # seam F3
# framing.py
def frame_level(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...    # (16, 3+W+6) grid, bool templated mask
# sanitizer.py
def detect(grid: np.ndarray) -> list["Defect"]: ...; def repair(grid: np.ndarray, defects: list["Defect"]) -> np.ndarray: ...
# physics.py / checker.py
def load_physics(path: Path = Path("app/static/game/physics.json")) -> "Physics": ...
@dataclass
class CheckResult: beatable: bool; path: list[tuple[int, int]]; fail_col: int | None; expanded: int; time_s: float
def check_level(grid: np.ndarray, physics: "Physics", noise_profile: Mapping | None = None, timeout_s: float = 2.0) -> CheckResult: ...
# marioeval_bridge.py
def run_marioeval(vglc_text: str, jar: Path, timeout_s: float = 30.0) -> bool | None: ...  # None = unavailable/timeout
# renderer.py
def tensor_to_grid(x: torch.Tensor) -> np.ndarray: ...                     # (13,16,W) or (B,13,16,W) -> uint8 via argmax
def grid_to_image(grid: np.ndarray, tile_px: int = 18, skin: str = "kenney") -> "PIL.Image.Image": ...
def grid_to_vglc_text(grid: np.ndarray) -> str: ...
def grid_to_json(grid: np.ndarray, meta: Mapping) -> dict: ...             # schema v1 (doc 29)
```

### 4.9 `utils/` → docs 06, 07, 30
```python
def bootstrap_env() -> None: ...                            # sets HF_HOME default, PYTHONUTF8, offline flags if T2M_OFFLINE=1
def load_config(experiment: Path, machine: Path | None = None, cli_overrides: list[str] = ()) -> "Config": ...
def config_hash(cfg: "Config") -> str: ...                  # sha256 of canonical JSON, machine sections excluded (doc 07 §3.3)
def detect_machine() -> str: ...                            # "desktop_3060ti" | "laptop_4050" | "unknown" (T2M_MACHINE overrides)
def apply_vram_cap(cap_gib: float = 4.5, device: int = 0) -> float: ...   # returns the applied fraction
def hw_report(run_dir: Path) -> dict: ...
def derive_seed(*parts: object) -> int: ...                  # stable 63-bit hash of the parts (sha256-based, never Python's hash())
def atomic_write_bytes(path: Path, data: bytes) -> None: ...; def sha256_file(path: Path) -> str: ...
def make_run_id(config_name: str, seed: int, machine_tag: str, now_utc: datetime | None = None) -> str: ...
class ThermalWatchdog: ...                                   # doc 07 §10
```

## 5. Config file layout

- `configs/base.yaml` holds **all** keys with defaults; every value cites a ledger ID in a trailing comment (`lr: 2.0e-4  # M10`). Doc 17 owns the full content.
- Experiment files are **overlays** that change only what differs (for example `uncond_cosine.yaml` sets `diffusion.schedule: cosine`, `text.enabled: false`, `train.max_steps: 15000`).
- Machine files (`configs/machine/*.yaml`) contain `machine`, `paths` and `overrides` sections only; `overrides` is restricted to the allow-list in doc 07 §3.2. The loader enforces this.
- Merge order: `base.yaml` ← experiment overlay ← machine `overrides` ← CLI `--set key=value` (CLI overrides are logged and included in the config hash unless they target an allow-listed key).

## 6. `.gitignore` and `.gitattributes`

```gitignore
# Python
__pycache__/
*.py[cod]
*.egg-info/
.venv/
.pytest_cache/
.ruff_cache/
.mypy_cache/
.hypothesis/
# Caches and environment
.cache/
.env
# Data and runs (regenerable or third-party)
data/raw/
data/processed/
runs/
# Model artifacts — never in git
*.pt
*.pth
*.ckpt
*.safetensors
*.onnx
# Local-only assets (A3)
app/static/sprites/classic_local/
# OS / editors
.DS_Store
Thumbs.db
.vscode/
.idea/
# Playwright
test-results/
playwright-report/
```

```gitattributes
* text=auto eol=lf
*.ps1 text eol=crlf
*.bat text eol=crlf
*.png binary
*.jpg binary
*.gif binary
*.npy binary
*.parquet binary
*.jar binary
app/static/vendor/*.js -diff linguist-vendored
```

## 7. `pyproject.toml` and dependency pins

### 7.1 `pyproject.toml`
```toml
[build-system]
requires = ["setuptools>=69", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "text2mario"
version = "0.1.0"
description = "Text-conditioned discrete-tile diffusion for Super Mario Bros. levels, with a playable browser studio"
requires-python = ">=3.12,<3.13"
# torch is intentionally NOT listed: it is installed by scripts/install_torch.* per ledger V2 (CUDA wheel index differs by driver).
dependencies = [
  "numpy", "pandas", "pyarrow", "pyyaml", "pillow", "matplotlib", "scipy", "tqdm",
  "transformers", "huggingface_hub", "safetensors", "streamlit>=1.52", "tensorboard",
  "nvidia-ml-py", "psutil", "imageio",
]

[project.optional-dependencies]
dev = ["pytest", "pytest-timeout", "hypothesis", "ruff", "pre-commit", "playwright", "pytest-playwright"]

[tool.setuptools.packages.find]
where = ["src"]

[tool.ruff]
target-version = "py312"
line-length = 100
src = ["src", "scripts", "app", "tests"]

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP", "SIM", "NPY", "PT", "RUF"]
ignore = ["E501"]           # line length handled by the formatter

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q --strict-markers --timeout=120"
markers = [
  "slow: takes > 5 s, or needs a GPU, a browser or Java",
  "gpu: requires CUDA",
  "e2e: browser end-to-end tests",
]
```

### 7.2 Pin policy (`requirements.txt`, `requirements-dev.txt`)
1. **Exact pins (`==`) for every top-level package**, torch excluded. Torch is pinned by `scripts/install_torch.*` and recorded in `env/lock_<machine>.txt`.
2. **Day 0 procedure.** On the first machine: `pip install -e ".[dev]"` (unpinned names) → run `verify_env.py --full` and `pytest -q` → `python scripts/compare_env.py --emit-pins > requirements.txt` (writes `name==version` for the top-level packages listed in `pyproject.toml`). On the second machine: install from the pinned files.
3. **Floors that must hold:** `streamlit>=1.52` (V4); Python 3.12 (V1); the torch minor version is identical on both machines (V2).
4. `env/lock_desktop.txt` and `env/lock_laptop.txt` = full `pip freeze`. `scripts/compare_env.py env/lock_desktop.txt env/lock_laptop.txt` must report **no differences** except platform-only packages (for example `pywin32`, `colorama`), and the torch local tag only if both machines are on the same CUDA major.
5. Upgrades happen only through a commit that changes `requirements*.txt` and both lock files, after `pytest -q` and `verify_env.py --full` pass on **both** machines. No upgrades between D14 (freeze) and D17.

`requirements.txt` (top-level names; versions filled on Day 0 by step 2):
```text
numpy==<pin>
pandas==<pin>
pyarrow==<pin>
pyyaml==<pin>
pillow==<pin>
matplotlib==<pin>
scipy==<pin>
tqdm==<pin>
transformers==<pin>
huggingface_hub==<pin>
safetensors==<pin>
streamlit==<pin>        # must be >= 1.52 (V4)
tensorboard==<pin>
nvidia-ml-py==<pin>     # NVML bindings for the watchdog and verify_env
psutil==<pin>           # AC-power detection (H9)
imageio==<pin>          # denoising animation
```
`requirements-dev.txt`: `pytest`, `pytest-timeout`, `hypothesis`, `ruff`, `pre-commit`, `playwright`, `pytest-playwright`, all `==<pin>`.

## 8. Pre-commit

```yaml
# .pre-commit-config.yaml — run `pre-commit install` once per clone (both machines)
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: <pin via `pre-commit autoupdate` on Day 0>
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: <pin via `pre-commit autoupdate` on Day 0>
    hooks:
      - id: check-yaml
      - id: check-json
      - id: check-merge-conflict
      - id: end-of-file-fixer
      - id: trailing-whitespace
      - id: mixed-line-ending
      - id: check-added-large-files
        args: [--maxkb=2048]          # phaser.min.js (~1.2 MB) passes; checkpoints are blocked
  - repo: local
    hooks:
      - id: forbid-artifacts
        name: forbid runs/, data/raw/, checkpoints, classic sprites
        entry: python scripts/hooks/forbid_artifacts.py
        language: system
        pass_filenames: true
      - id: split-lock
        name: verify data/splits/LOCK matches split_v1.json
        entry: python -c "from text2mario.data.splits import load_split; from pathlib import Path; load_split(Path('data/splits/split_v1.json'))"
        language: system
        files: ^data/splits/
        pass_filenames: false
      - id: pytest-fast
        name: fast test suite (< 60 s)
        entry: python -m pytest -m "not slow" -x -q
        language: system
        pass_filenames: false
        always_run: true
```
`scripts/hooks/forbid_artifacts.py` rejects any staged path matching `runs/`, `data/raw/`, `*.pt`, `*.pth`, `*.ckpt`, `*.safetensors`, or `app/static/sprites/classic_local/`, and any file > 2 MB outside `app/static/vendor/`.

## 9. Task runners

Both runners expose the same target names. Every target runs inside the activated venv.

### 9.1 `Makefile` (Linux)
```makefile
PY ?= python
EXP ?= configs/cond_main.yaml

.PHONY: setup verify fetch audit splits test test-slow lint gate-% train eval-dev sweep app figures status handoff-release handoff-acquire
setup:            ; $(PY) -m pip install -r requirements.txt -r requirements-dev.txt && $(PY) -m pip install -e . && pre-commit install
verify:           ; $(PY) scripts/verify_env.py --full
fetch:            ; $(PY) scripts/fetch_models.py && $(PY) scripts/fetch_assets.py && $(PY) scripts/fetch_data.py
audit:            ; $(PY) scripts/audit_dataset.py && $(PY) scripts/eda.py
splits:           ; $(PY) scripts/make_splits.py
test:             ; $(PY) -m pytest -m "not slow"
test-slow:        ; $(PY) -m pytest -m slow
lint:             ; ruff check . && ruff format --check .
gate-%:           ; $(PY) scripts/gate_$*.py            # e.g. make gate-p1, make gate-p5b
train:            ; $(PY) scripts/launch.py --config $(EXP)
eval-dev:         ; $(PY) scripts/eval_dev.py --run $(RUN)
sweep:            ; $(PY) scripts/sweep_cfg.py --run $(RUN)
app:              ; HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 streamlit run app/streamlit_app.py
figures:          ; $(PY) scripts/make_figures.py
status:           ; $(PY) scripts/status.py
handoff-release:  ; $(PY) scripts/handoff.py release --run $(RUN)
handoff-acquire:  ; $(PY) scripts/handoff.py acquire --run $(RUN)
```

### 9.2 `tasks.ps1` (Windows PowerShell)
```powershell
param([Parameter(Mandatory=$true)][string]$Task, [string]$Exp = "configs/cond_main.yaml", [string]$Run = "")
$ErrorActionPreference = "Stop"
switch -Regex ($Task) {
  '^setup$'    { python -m pip install -r requirements.txt -r requirements-dev.txt; python -m pip install -e .; pre-commit install }
  '^verify$'   { python scripts/verify_env.py --full }
  '^fetch$'    { python scripts/fetch_models.py; python scripts/fetch_assets.py; python scripts/fetch_data.py }
  '^audit$'    { python scripts/audit_dataset.py; python scripts/eda.py }
  '^splits$'   { python scripts/make_splits.py }
  '^test$'     { python -m pytest -m "not slow" }
  '^test-slow$'{ python -m pytest -m slow }
  '^lint$'     { ruff check .; ruff format --check . }
  '^gate-(.+)$'{ python ("scripts/gate_{0}.py" -f $Matches[1]) }
  '^train$'    { python scripts/launch.py --config $Exp }
  '^eval-dev$' { python scripts/eval_dev.py --run $Run }
  '^sweep$'    { python scripts/sweep_cfg.py --run $Run }
  '^app$'      { $env:HF_HUB_OFFLINE = "1"; $env:TRANSFORMERS_OFFLINE = "1"; streamlit run app/streamlit_app.py }
  '^figures$'  { python scripts/make_figures.py }
  '^status$'   { python scripts/status.py }
  '^handoff-release$' { python scripts/handoff.py release --run $Run }
  '^handoff-acquire$' { python scripts/handoff.py acquire --run $Run }
  default      { throw "Unknown task '$Task'" }
}
```
Usage: `.\tasks.ps1 verify`, `.\tasks.ps1 gate-p1`, `.\tasks.ps1 train -Exp configs/uncond_cosine.yaml`, `.\tasks.ps1 handoff-release -Run <run_id>`.

## 10. Run-directory layout (produced by `scripts/launch.py`)

```text
runs/<run_id>/                       # run_id format: doc 07 §2
├── config.yaml                      resolved experiment config (machine sections removed)
├── config_hash.txt                  sha256 (doc 07 §3.3)
├── meta.json                        git SHA, dirty flag, data manifest hash, split LOCK hash, env lock ref, seeds, created_utc
├── lineage.json                     segments: machine, start_step, end_step, torch, driver, started/ended_utc (doc 07 §5)
├── hw_report.json                   peak VRAM (allocated/reserved), it/s, machine (H8)
├── metrics.csv                      step, loss, loss_t{0..9}, lr, grad_norm, it_s, vram_gib
├── events.jsonl                     start/stop/resume/fallback/watchdog/eval events
├── tb/                              TensorBoard
├── checkpoints/
│   ├── ckpt_step0002000.pt  ckpt_step0002000.json
│   ├── ema_step0002000.safetensors
│   ├── latest.json  best.json       pointers {file, step, sha256}
├── eval/dev_step0002000.json …      dev suite outputs (E1, E4)
├── samples/                         PNG grids at eval steps
├── STOP                             (control file; create it to request a graceful stop)
└── HANDOFF_READY.json               written on graceful stop for handoff (doc 07 §4)
```

## 11. Definition of "scaffold done" (Day 0)
- [ ] Tree above exists (empty modules with docstrings citing their topic doc).
- [ ] `pip install -e .` works; `python -c "import text2mario"` succeeds on both machines.
- [ ] `pre-commit run --all-files` passes; `pytest -q` passes (placeholder tests: `test_tiles.py` asserts F6 against `TILE_CHARS`).
- [ ] `tasks.ps1` (Windows) and `make` (Linux) both run `verify`.
- [ ] `PROGRESS.md` and `HANDOFF.md` are initialised from the templates in doc 07 §4.
