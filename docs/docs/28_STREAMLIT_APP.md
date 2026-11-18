# 28 — Streamlit Level Studio (app specification)

> **Owner track:** G. **Days:** D12 (studio skeleton), D13 ("Type it, play it" MVP), D16 (hardening), D17 (optional report-card lite).
> **Ledger rows:** S2 (MVP), S3 (seams), S5 (Streamlit), V4 (Streamlit version), H2–H3 (VRAM cap), H10 (CPU fast mode), G1–G5 (generation, framing, verification), G-P7 (app gate), A1–A3 (skins).
> **Depends on:** doc 12 (normalizer), doc 20 (`generate/full_level.py`, including the G5 verification loop), doc 21 (checker), doc 27 (game HTML builder), doc 29 (level JSON schema v1).

---

## 1. Purpose and acceptance

The app is the viva demo and the product. **Acceptance (G-P7 plus MVP):**
1. Type a sentence → press **Generate & Play** → within **10 s on the laptop GPU** (112-wide, full quality) you see what the model understood, a verification badge, and a **playable game** in the page.
2. Cold start (process launch → model ready) **< 20 s** on the laptop.
3. Without CUDA, **CPU fast mode** (25 DDIM steps, r = 0; H10) produces a 112-wide level in **< 45 s**.
4. Works **offline** (Wi-Fi off). Keyboard play works in Chrome and Edge.
5. Exports: PNG, level JSON (schema v1), CSV, VGLC text, and a standalone playable HTML file.

## 2. Version note (Streamlit API change; **unverified, to verify on Day 0**)

> The statements below (deprecation in 1.56, `st.iframe`, latest 1.66) came from an earlier draft and could not be confirmed. On Day 0 run `pip index versions streamlit`, read the changelog of the pinned version and choose; the shim works either way. The same pinned version is used locally and on Streamlit Community Cloud (ledger V4).

- `st.components.v1.html` is **deprecated since Streamlit 1.56.0**; the documented replacement is **`st.iframe(src, *, width="stretch", height="content", tab_index=None, alt=None)`**. It accepts a raw HTML string (or a URL, or a `pathlib.Path` to an `.html` file) and runs the HTML with JavaScript enabled and same-origin access.
- Ledger V4: pin the **exact** version chosen on Day 0 in `requirements.txt`. The shim below uses `st.iframe` when it exists and otherwise `components.v1.html`.

```python
# app/components/game_embed.py
import streamlit as st

def embed_game(html: str, *, height: int = 600) -> None:
    """Render the self-contained game HTML (doc 27 §10) in an iframe that can take keyboard focus."""
    if hasattr(st, "iframe"):                                  # Streamlit >= 1.56
        st.iframe(html, height=height, tab_index=0, alt="Playable generated level")
    else:                                                      # 1.52-1.55 fallback
        import streamlit.components.v1 as components
        try:
            components.html(html, height=height, scrolling=False, tab_index=0)
        except TypeError:                                      # signature without tab_index
            components.html(html, height=height, scrolling=False)
```
Use a **fixed integer height** (default 600 px). `height="content"` auto-sizing fights Phaser's FIT scaling.

## 3. File layout

```
app/
  streamlit_app.py          # entry point: env setup, page config, st.navigation
  pages/studio.py           # main page: sidebar form + game + tabs
  pages/gallery.py          # showcase levels (instant demo / fallback)
  pages/about.py            # model card, data card, licences, frozen hashes, results
  components/game_embed.py  # §2 shim
  components/badges.py      # verification badge, timing caption
  state.py                  # session_state schema + helpers (§5)
  showcase/                 # 12 cached levels: *.level.json + *.png + manifest.json (generated Day 14 with the frozen model, checked Day 16)
  static/                   # doc 26/27: vendor/, game/, assets/ (served only in optional static mode)
src/text2mario/app/
  service.py                # LevelService: normalize → generate (+verify) → render → artifacts (§7)
  game_html.py              # build_game_html (doc 27 §10)
.streamlit/config.toml      # §12
scripts/run_app.ps1, scripts/run_app.sh, scripts/bench_app.py, scripts/make_showcase.py
```
If doc 05 differs on locations, doc 05 wins; the interfaces here stay the same.

## 4. Layout (wireframe)

```
┌─ text2mario · Level Studio ───────────────────────────────────────────────────────────────────┐
│ SIDEBAR (st.form "controls")        │ MAIN                                                    │
│ Describe your level                 │ You asked for:  [floor with one gap] [one pipe] [...]   │
│ ┌─────────────────────────────────┐ │ ⚠ Didn't understand: "lava"   (scene 1 of 3 ▸ chips)    │
│ │ a chill level with a few pipes, │ │ ┌──────────────────── GAME (iframe, 600 px) ──────────┐ │
│ │ then a staircase and coins      │ │ │ 🪙 00  ⏱ 00:00  💀 0            ✓ verified beatable │ │
│ └─────────────────────────────────┘ │ │                                                     │ │
│ Examples ▾ (fills the box)          │ │                ▶ Click to play                      │ │
│ Guidance w    1 ───●──── 10   5.0   │ │      ←/→ move · Space jump · Shift run · P pause    │ │
│ Width         16  48 [80] 112       │ └─────────────────────────────────────────────────────┘ │
│ Seed          [ 1234567 ]  🎲       │ ✓ Verified beatable · 1 attempt · 6.8 s on cuda · w=5.0 │
│ ☐ Fast mode   (forced on CPU)       │ [🖼 Level image] [🎞 Denoising] [📦 Export] [🔎 Details] │
│ ▸ Advanced                          │ [📋 Report card]                                        │
│   DDIM steps 50 · resample r 2      │  ……… tab content ………                                    │
│   Device auto · Repair display ☑    │                                                         │
│ [      ▶ Generate & Play      ]     │                                                         │
│ ── Recent (last 10) ──              │                                                         │
│ • "chill pipes…" seed 1234567  ↺    │                                                         │
└─────────────────────────────────────┴─────────────────────────────────────────────────────────┘
```
- The **game sits above the tabs, outside them.** An iframe inside a hidden tab gets zero size, and Phaser's FIT scaling misbehaves when it is revealed.
- `st.set_page_config(page_title="text2mario", page_icon="🎮", layout="wide")`.
- Navigation: `st.navigation([st.Page("pages/studio.py", title="Studio", icon="🎮", default=True), st.Page("pages/gallery.py", title="Gallery", icon="🖼️"), st.Page("pages/about.py", title="About", icon="ℹ️")]).run()`.

## 5. Controls and session state

### 5.1 Sidebar controls (inside one `st.form("controls")`)
A form means moving sliders doesn't trigger reruns. That matters because a rerun **interrupts a running generation**. Only the submit button runs the pipeline.

| Key | Widget | Range / default | Notes |
|---|---|---|---|
| `prompt` | `st.text_area` "Describe your level" | ≤ 300 chars (FR-A-01); placeholder example | Trimmed; empty → submit shows a warning |
| `example` | `st.selectbox` | 8 curated prompts (2 per difficulty mood) | Outside the form: selecting fills `prompt` via callback |
| `w` | `st.slider` | 1.0–10.0, step 0.5; default = `FROZEN.json → default_w` (M13) | w = 1 means no guidance |
| `width` | `st.segmented_control` / `st.radio` | 16 / 48 / 80 / 112; default 80 (G2) | Generated columns; framing adds 3 + 6 (G3) |
| `seed` | `st.number_input` | 0 … 2^53 − 1; default random | 🎲 button (outside the form) re-rolls. Seeds stay JS-safe integers (doc 29) |
| `fast_mode` | `st.toggle` | default off on GPU; **forced on and disabled on CPU** | 25 steps, r = 0 (H10) |
| `ddim_steps` | `st.slider` (Advanced) | 10–100; 50 (fast: 25) | |
| `resample_r` | `st.slider` (Advanced) | 0–5; 2 (fast: 0) (G1) | |
| `device` | `st.selectbox` (Advanced) | auto / cuda / cpu | `T2M_DEVICE` env var overrides |
| `repair_display` | `st.toggle` (Advanced) | on | Repairs broken pipes/cannons **for display and play only**; metrics stay raw (G4) |
| `generate` | `st.form_submit_button` "▶ Generate & Play" | — | `type="primary"` |

### 5.2 `st.session_state` schema (`app/state.py`, all keys namespaced `t2m_*`)
| Key | Type | Default | Meaning |
|---|---|---|---|
| `t2m_state_version` | `str` | `"1"` | Bump when the schema changes; mismatch → reset state |
| `t2m_params` | `dict` | from controls | Last submitted parameters (§5.1) |
| `t2m_level` | `dict \| None` | `None` | Current level, **schema v1** (doc 29), validated |
| `t2m_artifacts` | `dict` | `{}` | `png: bytes`, `gif: bytes \| None`, `game_html: str`, `csv: str`, `vglc: str`, `standalone_html: str` (built lazily on the Export tab) |
| `t2m_frames` | `list[np.ndarray]` | `[]` | Denoising snapshots (uint8 grids), ≤ 130, freed after the GIF is built |
| `t2m_history` | `list[dict]` | `[]` | Last 10: `{level_id, prompt, seed, w, width, thumb_png}`; "↺" restores |
| `t2m_status` | `str` | `"idle"` | `idle \| generating \| verifying \| rendering \| done \| error` |
| `t2m_error` | `dict \| None` | `None` | `{kind, message, traceback_path}` |
| `t2m_timings` | `dict` | `{}` | ms per stage (mirrors `meta.timings_ms`) |
| `t2m_mode` | `str` | `"model"` | `"model"`, or `"showcase"` when the model failed to load |

Initialise with `st.session_state.setdefault(...)` in `state.init()`, called at the top of every page.

## 6. Main flow (Generate & Play)

```
submit ─▶ validate prompt
       ─▶ service.normalize(prompt)            → normalized caption + scene prompts + unparsed terms (doc 12)
       ─▶ show "You asked for" chips immediately (before diffusion starts)
       ─▶ service.generate(request, on_progress) → doc 20 full_level: windows → framing (G3) → sanitizer
                                                  detect (+ display repair) → checker + window resampling (G5)
          inside st.status("Generating…", expanded=True):
             "Window 3/13 · step 25/50", "Verifying (attempt 2/20)", "Rendering"
       ─▶ level dict (schema v1) validated by text2mario.io.level_schema.validate()
       ─▶ artifacts: render_png (cached), build_game_html (cached), GIF from frames
       ─▶ session_state updated; history push; st.toast("Level ready — click the game to play")
       ─▶ rerun-safe display: embed_game(t2m_artifacts["game_html"]) + badge + tabs
```

- **Iframe stability:** `build_game_html` is deterministic (doc 27 §10.1). Store the HTML string in session state and pass the **same string** on every rerun, so the browser keeps the game (and the player's progress) when other widgets change.
- **Denoising animation (tab 🎞):** the generator calls `on_progress(stage, window_idx, n_windows, step, x0_hat_argmax)` every **5 DDIM steps** (Day 12 spec). Each snapshot is the full-width canvas so far (finished windows + the current window's x̂₀ prediction). The GIF uses the **flat-colour minimap renderer** (numpy palette lookup, 6 px per tile; no sprite blitting, so it's fast), subsampled to ≤ 60 frames at 80 ms each, with the last frame held for 1.5 s. Shown with `st.image(gif_bytes)`; the browser animates it natively. A toggle shows noisy-x_t argmax frames instead (the "static → level" look).
- **Tabs:**
  - 🖼 *Level image:* sprite PNG (skin `kenney`, scale 2) with flag/checkpoint overlays; a caption explains that framing columns are templated (G3).
  - 🎞 *Denoising:* the GIF plus a per-window frame slider.
  - 📦 *Export:* §9.
  - 🔎 *Details:* normalized prompt, scene prompts, per-window table (cols, scene, seed, attempt, resampled?), timings, device, checkpoint sha256 / config hash, physics hash. This tab carries the honesty notes: display repairs applied (count) and fast-mode quality warning.
  - 📋 *Report card:* §10.
- **Badge text** (`components/badges.py`), from `level.verification`:
  - `verified` → "✓ Verified beatable · k attempt(s)"
  - `not_verified` → "⚠ Not verified after k attempts. You can still play; it may be impossible. [Try another seed]"
  - `skipped` → "— Verification skipped"

## 7. Service layer (UI-free, unit-testable)

```python
@dataclass(frozen=True)
class GenerationRequest:
    prompt: str
    width: int = 80                 # G2
    w: float = 5.0                  # M13 default from FROZEN.json
    seed: int = 0
    ddim_steps: int = 50
    resample_r: int = 2
    fast_mode: bool = False
    repair_display: bool = True
    device: str = "auto"            # auto | cuda | cpu

class LevelService:
    def __init__(self, frozen_path: Path, device: str): ...   # loads EMA weights, encoder, normalizer, checker
    def normalize(self, prompt: str) -> NormalizedPrompt: ...
    def generate(self, req: GenerationRequest, on_progress: Callable | None = None) -> dict:  # schema-v1 level
        ...   # wraps doc 20 full_level.generate(...) — includes the G5 verify/resample loop — then fills
              # meta/framing/verification/metrics/provenance per doc 29; holds self._lock (threading.Lock)
    def warm_up(self) -> float: ...                            # one 16-wide, 5-step generation; returns seconds
```
- One `LevelService` per `(frozen_path, device)` via `@st.cache_resource`. It is shared across browser sessions, so `generate` holds a `threading.Lock` (one generation at a time; a local single-user app).
- `T2M_FAKE_GENERATOR=1` → `FakeLevelService` returns fixture levels instantly (a real dataset level, framed and checked) **without importing torch**. Used by AppTest and UI work before the model exists (Day 12 can start before C-main finishes).

## 8. Caching, device, warm-up, latency

### 8.1 Caching
| Function | Decorator | Key | Returns |
|---|---|---|---|
| `load_service(frozen_path, device)` | `st.cache_resource(show_spinner="Loading model…")` | path + device | `LevelService` (warm-up run inside) |
| `load_showcase()` | `st.cache_resource` | — | list of showcase levels |
| `render_png(grid_bytes, cols, framing_json, skin, scale)` | `st.cache_data(max_entries=64)` | bytes/str args | PNG bytes |
| `build_game_html_cached(level_json, skin, sound)` | `st.cache_data(max_entries=32)` | canonical level JSON string | HTML str |
| `render_gif(frames_npz_bytes)` | `st.cache_data(max_entries=8)` | npz bytes | GIF bytes |

Nothing heavy runs at module top level. A rerun after an input change should complete in < 300 ms.

### 8.2 Device selection
1. `T2M_DEVICE` env var (`cuda` / `cpu`) wins; else the Advanced selectbox; else auto = `cuda` if `torch.cuda.is_available()`.
2. On CUDA: apply the VRAM cap at service load, `torch.cuda.set_per_process_memory_fraction(4.5 GiB / total)` (H2–H3). Inference needs < 1 GiB, which leaves room for the browser.
3. On CPU: force fast mode (H10); `torch.set_num_threads(max(1, physical_cores))`; show the banner "CPU mode: fast settings (25 steps, no resampling)".
4. If CUDA OOM or a CUDA runtime error happens mid-generation: retry once on CPU fast mode, tell the user with a toast, and log an event (no silent fallbacks).

### 8.3 Cold start and warm-up (< 20 s target)
Budget on the laptop: Streamlit boot ~2 s · `import torch` 2–4 s · EMA checkpoint load (~73 MB fp32) ~0.5 s · MiniLM from local disk ~1.5 s · CUDA context + warm-up generation 1–3 s. Expected total 8–13 s. The About page and the Studio caption show "Model ready in X s". `scripts/bench_app.py --cold` measures it in a fresh process (5 runs, report the median).

### 8.4 Generation latency (112-wide, GPU, full quality < 10 s)
Work: 13 windows × 50 steps × ≈ 2.2 (r = 2 resampling for the first 60 % of steps) ≈ **1,430 guided forward passes** (batch 2 = cond + uncond), plus the checker and rendering. For an 18.3 M-parameter U-Net on 16 × 16 inputs, the pass time is dominated by kernel launches (expect roughly 4–6 ms on the RTX 4050 → 6–9 s). **Measure on Day 12** with `scripts/bench_app.py` (10 prompts, p50/p95 per stage).

If p95 > 10 s, apply in this order (stop as soon as it's under budget; log the choice in `PROGRESS.md`):
1. Capture the U-Net forward in a **CUDA Graph** (static shapes: batch 2, 13 × 16 × 16): removes launch overhead, numerically identical.
2. Generate **2 candidates per window in one batch** (batch 4) so the G5 verifier rarely needs a sequential retry.
3. r = 2 → 1 for the app only (and say so in Details).
4. DDIM 50 → 40 for the app only.
Changes 3–4 alter outputs relative to the evaluated configuration, so they must be stated on the About page.

## 9. Exports (📦 tab) — formats per doc 29
| Button | File | MIME |
|---|---|---|
| Level JSON (schema v1) | `t2m_<level_id>.level.json` | `application/json` |
| PNG (sprites, scale 2, metadata in tEXt chunks) | `t2m_<level_id>.png` | `image/png` |
| CSV (tile IDs) | `t2m_<level_id>.ids.csv` | `text/csv` |
| VGLC text (16 rows) | `t2m_<level_id>.vglc16.txt` | `text/plain` |
| Playable HTML (standalone) | `t2m_<level_id>.html` | `text/html` |
| Tiled map (MAY, cut) | `t2m_<level_id>.tmj` | `application/json` |

Use `st.download_button(label, data, file_name, mime, key=...)`. Where the installed version supports `on_click="ignore"`, use it so a download doesn't trigger a rerun.

## 10. Report card (📋 tab) — placeholder in the MVP; "lite" version cut (MAY)

- **MVP:** the tab shows "Coming soon", plus the raw `metrics` block if the generator filled it (doc 29: per-scene requested vs detected concepts, C-score, broken structures).
- **Lite version (MAY, cut; seam for future feature F6):**

| Row | Source | Display |
|---|---|---|
| Per scene: requested vs detected concepts | `metrics.per_scene[*]` (detector, doc 22) | Table with ✓ / ✗ / ~ (bucket off by one), C-score |
| Beatable? | `verification` | Yes/No + attempts + checker version |
| Copy of a Nintendo level? | NN Hamming of every 16-wide generated window (stride 8) against **all 7,687 corpus windows** (the full corpus, since this is a product-level question asked after the model is frozen; the evaluation metric E1 uses the train split instead) | "No: the closest original window differs by N tiles (level L, game G)"; flagged if N ≤ 4 |
| Broken structures | `metrics.broken_structures_raw`, `repairs` | "k detected, repaired for display" |

## 11. Showcase, errors, About page

**Showcase (`app/showcase/`, generated Day 14, checked Day 16):** `scripts/make_showcase.py` generates 12 levels from a curated prompt list with the frozen checkpoint, keeps only `verified` ones (re-seeding up to 5 times), and writes `*.level.json`, `*.png` and `manifest.json` (`level_id, prompt, seed, w, width, verified, png`). Committed. The Gallery page shows a 3-column grid of thumbnails; "▶ Play" loads the level into session state and calls `st.switch_page("pages/studio.py")`. If the model can't load, the Studio switches to `t2m_mode = "showcase"`: a banner explains why, Generate is disabled, and a showcase picker appears inline.

**Error handling**
| Failure | Detection | What the user sees | Logged |
|---|---|---|---|
| Empty prompt | validation | Inline warning; nothing runs | — |
| Prompt > 300 chars | validation | Truncated + caption | — |
| Normalizer understood nothing | empty normalized caption | Info: "Using your words directly"; the raw prompt goes to the encoder | event |
| CUDA OOM / CUDA error | exception type | Toast: "GPU problem, retrying on CPU (fast mode)" | event |
| Checkpoint missing or hash ≠ `FROZEN.json` | at `load_service` | Showcase mode banner with the reason | event |
| Verification exhausted (20 attempts, G5) | `verification.status` | Amber badge + "Try another seed" button | — |
| Any other generation exception | catch-all | `st.error` + an expander with the traceback (dev config) + "Load a showcase level" | event + traceback file |
| Game HTML over the size budget | builder check | Rebuilt with `sound="off"` | warning |

Events go to `logs/app_events.jsonl` (git-ignored): timestamp, level_id, stage timings, device, status, error kind.

**About page:** renders `MODEL_CARD.md` and `DATA_CARD.md` if present. Licences table: MarioDiffusion code/data **MIT**; levels derived from the VGLC; Kenney Pixel Platformer **CC0**; Phaser **MIT**; `all-MiniLM-L6-v2` (confirm the licence from the model card in the local snapshot; Apache-2.0 expected). It shows `FROZEN.json` (checkpoint sha256, config hash, default w), the physics hash, and the results table from `reports/final/RESULTS.md` ("not evaluated yet" before Day 15). Honesty notes: framing is templated (G3); repairs are display-only (G4); "verified" means our checker with physics hash X (doc 27 §4.4); any latency compromises from §8.4.

## 12. Configuration, offline mode, launch

`.streamlit/config.toml`
```toml
[server]
headless = true
port = 8501
runOnSave = false
enableStaticServing = true      # only needed for the optional static game mode (doc 27 §10.4)

[browser]
gatherUsageStats = false        # no telemetry, works offline (both profiles)

[client]
toolbarMode = "minimal"
showErrorDetails = true         # dev; set to false (or the version's equivalent) for the viva

[theme]
base = "dark"
```
Check the keys with `streamlit config show` for the installed version; option names occasionally change.

Offline: at the very top of `app/streamlit_app.py`, before anything imports `transformers`:
```python
import os
for k, v in {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
             "TOKENIZERS_PARALLELISM": "false"}.items():
    os.environ.setdefault(k, v)
```
MiniLM is loaded from the local snapshot path in the config (`text_encoder.local_path`, doc 15), never by hub name at runtime.

**Launch**
- Windows (PowerShell): `scripts\run_app.ps1` (activates `.venv`, sets the env vars above, runs `streamlit run app/streamlit_app.py`, opens the browser).
- Linux: `scripts/run_app.sh`.
- Force CPU (rehearsal, Day 16): `$env:T2M_DEVICE="cpu"` (PowerShell) or `T2M_DEVICE=cpu scripts/run_app.sh`. Also try `CUDA_VISIBLE_DEVICES=""`.

### 12.1 Local profile vs cloud profile (ledger §3b; FR-A-16, FR-A-17; G-P8)
| Aspect | Local (development, live demo) | Cloud (Streamlit Community Cloud) |
|---|---|---|
| Switch | default | `T2M_PROFILE=cloud` in the app settings |
| Binding | `localhost` by default | managed by the platform (public URL) |
| Device | CUDA if present, else CPU fast mode | CPU only, fast mode (25 steps, r = 0), width default 48 |
| Weights | run directory / `app/model/` | `app/model/t2m_ema.safetensors` (committed, below GitHub's 100 MB file limit) + `app/model/minilm/` (fp16, bundled), `local_files_only=True` |
| Dependencies | `requirements.txt` + `requirements-dev.txt` | `requirements.txt` only, exact pins, CPU torch wheel |
| Time guard | none | `cloud.time_budget_s` (provisional 60 s): abort and serve the closest cached showcase level, labelled; logged |
| Cold start | measured on the laptop (G-P7, pinned after Day 12) | first request after sleep is slower; status line "Loading model..."; showcase levels selectable immediately; measured Day 12 |
| Showcase | 12 levels | same 12 levels; also the fallback |
| Logs | `runs/app_logs/<date>.jsonl` | platform log + in-app counters (fallbacks, timings) |
`app/components/profile.py` reads the profile once; every profile-dependent branch is logged (CLAUDE.md §2.12). The cloud app stores no user data and holds no secrets (NFR-SEC-02). Memory: one model + encoder + Python must fit the platform limit (verified and pinned on Day 12).

## 13. Testing

**Unit (fast, CI):** `state.init` idempotent; `embed_game` takes the right branch (monkeypatch `hasattr(st, "iframe")`); `GenerationRequest` validation; `FakeLevelService` output validates against schema v1.

**AppTest (fast, CI; no torch, via `T2M_FAKE_GENERATOR=1`):**
```python
from streamlit.testing.v1 import AppTest

def test_generate_and_play(monkeypatch):
    monkeypatch.setenv("T2M_FAKE_GENERATOR", "1")
    at = AppTest.from_file("app/streamlit_app.py", default_timeout=30).run()
    assert not at.exception
    at.text_area(key="prompt").input("full floor with three pipes and no enemies")
    at.button(key="generate").click().run()        # form submit button
    assert not at.exception
    lvl = at.session_state["t2m_level"]
    assert lvl["schema_version"].startswith("1.")
    assert lvl["verification"]["status"] in {"verified", "not_verified", "skipped"}
    assert at.session_state["t2m_artifacts"]["game_html"].startswith("<!doctype html>")
```
Also: empty prompt → warning and no level; showcase mode when the checkpoint path is missing; history restore. (AppTest does not run iframe JavaScript; the game itself is covered by doc 27 §12. If the AppTest element API differs in the installed version, adapt to the Streamlit testing docs.)

**End-to-end (slow; Day 13 and Day 16):** `tests/e2e/test_app_e2e.py` starts `streamlit run` on port 8599 in a subprocess (real model), opens the page with Playwright, fills the prompt, clicks "Generate & Play", waits for the iframe titled "Playable generated level", clicks into it, holds → for 1 s, and asserts via `frame.evaluate("T2M_DEBUG.getState()")` that the player moved. It times submit → `ready` (must be < 10 s on the GPU).

**Bench:** `scripts/bench_app.py [--cold] [--cpu]` → `reports/app/bench_<machine>.json` (p50/p95 per stage), the evidence for G-P7.

## 14. Day checklist mapping
| Day | Must exist | Gate evidence |
|---|---|---|
| D12 | Pages skeleton, sidebar form, service + fake service, normalized-prompt chips, PNG, GIF, exports (JSON/CSV/VGLC/PNG), caching, CPU fast mode | Cold start measured; AppTest green |
| D13 | `embed_game` + game HTML, badge, standalone HTML export, e2e test | G-P7 timings on the laptop; e2e green |
| D16 | Showcase (12), offline run, CPU rehearsal, browser matrix, 768p layout check | Day 16 checklist (doc 03) |
| D12, D14 | Cloud profile (§12.1): dry run D12, final deploy D14 (G-P8) | `reports/app/cloud_dryrun.json`, `evidence/deployment/` |
| (cut) | Report-card lite (§10) | MAY, only if everything else is done |
