# 06 — Environment Setup (both machines)

> **Authority:** ledger H1–H10 (hardware), V1–V8 (software). Goal of Day 0: both machines pass `python scripts/verify_env.py --full` with the **same Python minor, the same torch version string, the same pinned packages**, and a working 4.5 GiB VRAM cap. Commands are given for **Windows 11 (PowerShell)** and **Linux (bash, Ubuntu 22.04/24.04)**. Assume the laptop runs Windows 11; the desktop may run either.

---

## 1. Target state

| Item | Desktop (DT) | Laptop (LT) | Ledger |
|---|---|---|---|
| GPU | RTX 3060 Ti 8 GB, Ampere sm_86 | RTX 4050 Laptop 6 GB, Ada sm_89 | H1 |
| Machine profile name / run tag | `desktop_3060ti` / `DT` | `laptop_4050` / `LT` | doc 07 §2 |
| OS | Windows 11 or Ubuntu 22.04/24.04 | Windows 11 | — |
| Python | 3.12.x (same minor on both) | 3.12.x | V1 |
| PyTorch | **to verify on Day 0 (`pip index versions torch`)**; earlier drafts proposed 2.15.x CUDA 13 or 2.14.x + cu126, unverified | identical to DT | V2 |
| NVIDIA driver | ≥ 580 for CUDA 13 wheels; ≥ 525 for cu126 wheels | same rule | V2 + NVIDIA minor-version compatibility table |
| VRAM cap | 4.5 GiB (fraction ≈ 0.56 of 8 GB) | 4.5 GiB (fraction ≈ 0.75 of 6 GB) | H2, H3 |
| Precision | bf16 autocast | bf16 autocast | H5 |
| `torch.compile` | off | off | H6 |
| Java (optional) | Temurin JRE 21 | optional | V6 |
| Playwright Chromium (optional, Day 13) | optional | **required for AT-15** | doc 02 §7 |
| Free disk on the repo drive | ≥ 20 GB | ≥ 20 GB | NFR-RES-04 |

## 2. Prerequisites

| Tool | Windows 11 | Ubuntu 22.04 / 24.04 |
|---|---|---|
| Git ≥ 2.40 | `winget install -e --id Git.Git` | `sudo apt install -y git` |
| Python 3.12 | `winget install -e --id Python.Python.3.12` | 24.04: `sudo apt install -y python3.12 python3.12-venv` · 22.04: `sudo add-apt-repository ppa:deadsnakes/ppa && sudo apt install -y python3.12 python3.12-venv` |
| GitHub CLI (optional, for auth) | `winget install -e --id GitHub.cli` | `sudo apt install -y gh` |
| Build tools | not needed (all wheels are binary) | `sudo apt install -y build-essential` (safety net) |
| NVIDIA driver | NVIDIA App or nvidia.com driver download | `ubuntu-drivers` (§3.3) |

Optional alternative for both OSes: `uv` (`pip install uv`, then `uv venv --python 3.12`). `pip` remains the canonical path in all commands below.

## 3. NVIDIA driver check and the torch channel decision (V2)

### 3.1 Read the driver and GPU (both OSes)
```bash
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader
# If your nvidia-smi rejects compute_cap, drop that field:
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
```
Expected names: `NVIDIA GeForce RTX 3060 Ti` and `NVIDIA GeForce RTX 4050 Laptop GPU`. Record the driver on both machines in `PROGRESS.md`.

### 3.2 Decision procedure (run it before installing torch)

NVIDIA's minor-version-compatibility table: **CUDA 13.x runtimes need driver ≥ 580**; **CUDA 12.x runtimes need driver ≥ 525**. Newer drivers also run older CUDA runtimes.

| Step | Condition | Action |
|---|---|---|
| 1 | Both drivers ≥ 580 | Channel **`cu13`**: torch **2.15.x** with the CUDA 13 wheel on **both** machines |
| 2 | Either driver < 580 but updatable today | Update it (§3.3), reboot, re-check, then step 1 |
| 3 | Either driver < 580 and **cannot** be updated (managed machine, OEM lock) | Channel **`cu126`**: torch **2.14.x** + cu126 on **both** machines (2.14 is the last release with CUDA 12.x wheels) |
| 4 | Either driver < 525 | Must update; no supported channel |
| 5 | Always | Record `torch_channel` in `env/capabilities.json` for both machines. **Never mix channels or torch minors across machines** (V2). |

> **Version claims in this section (torch 2.15/2.14, CUDA 13/12 channels, driver numbers) are unverified (ledger V2).** Treat every version below as a hypothesis; Day 0 task: `pip index versions torch`, `pip download --no-deps` of the intended wheels (CUDA and CPU), read NVIDIA's compatibility table, then record the real pins in the ledger.

Earlier note (unverified): a plain `pip install torch` on Linux resolved to `2.15.0+cu132`, so the 2.15 stable CUDA wheel is CUDA 13.2. On Windows the CUDA wheels come from the PyTorch index. **Always confirm the `--index-url` for your OS on pytorch.org/get-started/locally** (Stable → Pip → Python → CUDA 13.x) and pass it to the install script if it differs from the default.

### 3.3 Updating drivers
- **Windows (both machines):** install the latest driver from the NVIDIA App or nvidia.com (Studio or Game Ready both work; Studio is the steadier choice). The RTX 4050 Laptop GPU is supported by NVIDIA's generic notebook driver; Lenovo Vantage may offer an OEM build, and either is fine if it is ≥ 580. Reboot, then re-run §3.1.
- **Ubuntu:** `sudo ubuntu-drivers list` → install the newest `nvidia-driver-5xx` (≥ 580) shown, preferring the `-open` variant for RTX 30/40 (`sudo apt install -y nvidia-driver-5xx-open`) → reboot (enrol the MOK key if Secure Boot asks) → re-run §3.1.

## 4. Windows 11 machine preparation (once per Windows machine)

Run in an **elevated** PowerShell where marked (Admin).

| # | Setting | How | Why |
|---|---|---|---|
| W1 | Script execution for venv activation | `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` | `Activate.ps1` is blocked by default |
| W2 | UTF-8 for Python everywhere | `setx PYTHONUTF8 1` (new shells inherit it) | Avoids `cp1252` decode errors; code also passes `encoding="utf-8"` |
| W3 | Git long paths | `git config --global core.longpaths true` | Deep cache paths |
| W4 | Git line endings | `git config --global core.autocrlf false` | `.gitattributes` controls EOL (doc 05 §6) |
| W5 | No sleep on AC during runs | `powercfg /change standby-timeout-ac 0` and `powercfg /change hibernate-timeout-ac 0` | Overnight runs (🌙) must not be suspended |
| W6 | Lid close = do nothing on AC (laptop) | `powercfg /setacvalueindex SCHEME_CURRENT SUB_BUTTONS LIDACTION 0` then `powercfg /setactive SCHEME_CURRENT` | Prevents accidental sleep. Still keep the lid **open** for cooling (H9) |
| W7 | Power mode | Settings → System → Power & battery → Power mode: **Best performance** (on AC) | Avoids clock caps |
| W8 | Lenovo LOQ thermal mode (laptop) | Lenovo Vantage → Power/Thermal: **Performance** (Fn+Q cycles modes on Lenovo gaming laptops, if your model supports it). Optional: turn **Hybrid Mode off** (dGPU-only) for training sessions; this needs a reboot | H9 |
| W9 | **CUDA Sysmem Fallback Policy** | NVIDIA Control Panel → Manage 3D settings → Program Settings → Add both the venv `python.exe` **and** the base interpreter (`python -c "import sys; print(sys._base_executable)"`) → *CUDA – Sysmem Fallback Policy* = **Prefer No Sysmem Fallback** | Otherwise Windows silently spills VRAM into system RAM: training gets 5–20× slower instead of raising OOM (doc 07 §1.4) |
| W10 | Preferred GPU (Optimus laptops) | Same NVIDIA panel entries → *Preferred graphics processor* = High-performance NVIDIA processor; plus Settings → System → Display → Graphics → add both `python.exe` files → High performance | Keeps the dGPU awake and at full clocks |
| W11 | Windows Update during training weeks | Settings → Windows Update → **Pause updates** (up to 5 weeks) or set Active hours | Forced reboots kill overnight runs |
| W12 | (Optional, Admin) Defender exclusion for the repo | `Add-MpPreference -ExclusionPath "C:\path\to\text2mario"` | Faster `pip`/pytest; only if you accept the trade-off |
| W13 | (Optional) Developer Mode | Settings → System → For developers → Developer Mode **On** | Lets the HF cache use symlinks (otherwise it copies; works either way) |

Linux desktop equivalents (optional): prevent suspend during runs with `sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target` (undo with `unmask`); enable persistence mode with `sudo nvidia-smi -pm 1`.

## 5. Clone, virtual environment, torch

### 5.1 Clone and create the venv
**Windows (PowerShell):**
```powershell
git clone https://github.com/<you>/text2mario.git
cd text2mario
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python --version          # must print 3.12.x
```
**Linux (bash):**
```bash
git clone https://github.com/<you>/text2mario.git
cd text2mario
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python --version          # must print 3.12.x
```

### 5.2 `scripts/install_torch.ps1`
```powershell
param(
  [ValidateSet("cu13", "cu126")][string]$Channel = "cu13",
  [string]$Version = "",
  [string]$IndexUrl = ""
)
$ErrorActionPreference = "Stop"
if ($Channel -eq "cu13") {
  if (-not $Version)  { $Version  = "2.15.*" }
  if (-not $IndexUrl) { $IndexUrl = "https://download.pytorch.org/whl/cu132" }   # confirm on pytorch.org (§3.2)
} else {
  if (-not $Version)  { $Version  = "2.14.*" }
  if (-not $IndexUrl) { $IndexUrl = "https://download.pytorch.org/whl/cu126" }
}
python -m pip install "torch==$Version" --index-url $IndexUrl
python -c "import torch, sys; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); sys.exit(0 if (torch.version.cuda and torch.cuda.is_available()) else 1)"
if ($LASTEXITCODE -ne 0) { throw "CUDA build of torch not usable - see doc 06 section 13" }
```

### 5.3 `scripts/install_torch.sh`
```bash
#!/usr/bin/env bash
set -euo pipefail
CHANNEL="${1:-cu13}"; VERSION="${2:-}"; INDEX_URL="${3:-}"
if [[ "$CHANNEL" == "cu13" ]]; then
  VERSION="${VERSION:-2.15.*}"; INDEX_URL="${INDEX_URL:-https://download.pytorch.org/whl/cu132}"
else
  VERSION="${VERSION:-2.14.*}"; INDEX_URL="${INDEX_URL:-https://download.pytorch.org/whl/cu126}"
fi
python -m pip install "torch==${VERSION}" --index-url "${INDEX_URL}"
python - <<'EOF'
import sys, torch
print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
sys.exit(0 if (torch.version.cuda and torch.cuda.is_available()) else 1)
EOF
```

**Rule:** after installing on the first machine, read the exact `torch.__version__` (for example `2.15.0+cu132`) and install **exactly that version** on the second machine: `.\scripts\install_torch.ps1 -Version 2.15.0` / `bash scripts/install_torch.sh cu13 2.15.0`. torchvision and torchaudio are **not** needed.

## 6. Project install, pins and locks

```powershell
# Windows (venv active)
python -m pip install -r requirements.txt -r requirements-dev.txt
python -m pip install -e .
pre-commit install
python -m pip freeze > env\lock_laptop.txt          # or env\lock_desktop.txt on the desktop
```
```bash
# Linux (venv active)
python -m pip install -r requirements.txt -r requirements-dev.txt
python -m pip install -e .
pre-commit install
python -m pip freeze > env/lock_desktop.txt
```
- **First machine on Day 0** (before pins exist): `pip install -e ".[dev]"` → tests pass → `python scripts/compare_env.py --emit-pins > requirements.txt` (and the dev file) → commit → reinstall from the pins to confirm (doc 05 §7.2).
- **Second machine:** install from the committed pins only.
- **Check:** `python scripts/compare_env.py env/lock_desktop.txt env/lock_laptop.txt` → no differences except platform-only packages (doc 05 §7.2).

## 7. Git identity and authentication

```bash
git config --global user.name  "<Your Name>"
git config --global user.email "<you@example.com>"
gh auth login            # or add an SSH key to GitHub
git config --global pull.rebase true     # linear history; HANDOFF.md lock pushes stay simple (doc 07 §4)
```
The repo is **private**. Never commit `.env`, checkpoints, `runs/`, `data/raw/` (enforced by the pre-commit hook, doc 05 §8).

## 8. Hugging Face model prefetch and offline cache (MiniLM, V3)

### 8.1 Environment variables (`.env.example` documents them; `utils.env.bootstrap_env()` sets the defaults)

| Variable | Value | When | Purpose |
|---|---|---|---|
| `HF_HOME` | `<repo>/.cache/hf` | always | Per-repo, git-ignored cache on each machine |
| `HF_HUB_OFFLINE` | `1` | app, viva, training after Day 7 | Never touch the network |
| `TRANSFORMERS_OFFLINE` | `1` | same | Legacy transformers flag; set both |
| `HF_HUB_DISABLE_TELEMETRY` | `1` | always | Privacy (NFR-SEC-02) |
| `HF_HUB_DISABLE_SYMLINKS_WARNING` | `1` | Windows without Developer Mode | Silences the warning; the cache still works |
| `TOKENIZERS_PARALLELISM` | `false` | always | Avoids fork warnings and thread oversubscription |
| `PYTHONUTF8` | `1` | Windows | §4 W2 |
| `T2M_MACHINE` | `desktop_3060ti` / `laptop_4050` | only if auto-detect fails | Overrides GPU-name detection |
| `T2M_OFFLINE` | `1` | app and viva launchers | `bootstrap_env()` then sets both HF offline flags |
| `CUBLAS_WORKSPACE_CONFIG` | `:4096:8` | **only** in deterministic debug mode | Needed by `torch.use_deterministic_algorithms` (doc 07 §9) |

PowerShell session: `$env:HF_HOME = "$PWD\.cache\hf"`. Bash: `export HF_HOME="$PWD/.cache/hf"`. `bootstrap_env()` uses `os.environ.setdefault`, so explicit settings win. It must run **before** `import transformers` (doc 05 §2.3).

### 8.2 `scripts/fetch_models.py` (Day 0)
```text
usage: python scripts/fetch_models.py [--revision SHA] [--verify-only]
1. bootstrap_env(); repo_id = "sentence-transformers/all-MiniLM-L6-v2".
2. revision = --revision, else env/models.json["revision"] if it exists, else HfApi().model_info(repo_id).sha
   (the first machine resolves it once; the second machine reuses the committed value).
3. snapshot_download(repo_id, revision=revision,
       allow_patterns=["config.json", "tokenizer.json", "tokenizer_config.json",
                       "special_tokens_map.json", "vocab.txt", "model.safetensors"])
4. sha256 of every downloaded file → env/models.json {repo_id, revision, files{name: sha256}, fetched_utc}.
5. Offline self-test in a subprocess with HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1:
   AutoTokenizer/AutoModel.from_pretrained(repo_id, revision=revision); encode "full floor. one pipe.";
   assert last_hidden_state.shape == (1, L, 384) with L <= 48; print L.
exit 0 = cached and offline-loadable; 1 = failure.
```

## 9. `scripts/verify_env.py` — full specification

### 9.1 Interface
```text
usage: python scripts/verify_env.py [--quick | --full] [--json PATH] [--strict] [--machine NAME] [--allow-cpu]
  --quick    daily (< 15 s): checks marked Q
  --full     Day 0, before every long run, before the viva: all checks
  --strict   WARN results also fail
  --allow-cpu  for app-only rehearsal (CPU fast mode); GPU checks become WARN
outputs: console table; env/verify/<machine>_<YYYYMMDD-HHMM>.json; merges this machine's entry into env/capabilities.json
exit:    0 = every CRIT check passed; 1 = any CRIT failed (or any WARN with --strict)
```

### 9.2 Checks

| ID | Q | Severity | Check | Pass condition |
|---|---|---|---|---|
| V01 | Q | CRIT | Python version | `sys.version_info[:2] == (3, 12)` |
| V02 | Q | WARN | venv active | `sys.prefix != sys.base_prefix` |
| V03 | Q | CRIT | Package import | `import text2mario` resolves inside this repo (editable install) |
| V04 | Q | CRIT | torch build | `torch.version.cuda is not None` (not a CPU wheel) |
| V05 | Q | CRIT | CUDA usable | `torch.cuda.is_available()` and device count ≥ 1 |
| V06 | Q | CRIT | Machine identity | GPU name → profile (`"3060 Ti"` → desktop_3060ti, `"4050"` → laptop_4050) or `T2M_MACHINE`; compute capability ∈ {(8,6), (8,9)} |
| V07 | Q | CRIT | Driver vs channel | NVML driver major ≥ 580 if the torch CUDA major is 13; ≥ 525 if 12 |
| V08 | Q | CRIT | Same torch on both machines | If the other machine's `env/lock_*.txt` exists, its `torch==` line equals this machine's `torch.__version__` |
| V09 | Q | CRIT | **VRAM cap enforced** | `apply_vram_cap(4.5)` succeeds; then `torch.empty(5 * 2**30, dtype=torch.uint8, device="cuda")` **must raise** `torch.OutOfMemoryError`; then `torch.cuda.empty_cache()` |
| V10 | — | WARN | Cap not too tight | Under the cap, a 4.0 GiB allocation succeeds (else another process is holding VRAM: inspect `nvidia-smi`, close browsers/games before training) |
| V11 | Q | CRIT | bf16 | `torch.cuda.is_bf16_supported()`; 256×256 bf16 vs fp32 matmul relative error < 1e-2; throughput loop (1 s quick / 5 s full) of 4096² bf16 matmuls under autocast → TFLOPS recorded |
| V12 | Q | CRIT | SDPA self- and cross-attention | bf16 q,k,v `(64, 4, 64, 32)`; cross: q `(64, 4, 64, 32)`, k/v `(64, 4, 48, 32)`, boolean mask `(64, 1, 1, 48)` with the last 10 keys masked → max abs diff vs an fp32 manual softmax reference < 2e-2, no NaN; record which SDPA backends are enabled |
| V13 | — | WARN | cuDNN | available; version recorded |
| V14 | Q | CRIT (LT) / WARN (DT) | NVML telemetry | GPU temperature, power draw and clock-event reasons readable (needed by the watchdog, H9) |
| V15 | Q | WARN | AC power (laptop) | `psutil.sensors_battery()` is None (desktop) or `power_plugged` is True |
| V16 | — | WARN | Disk | ≥ 20 GB free on the repo drive (NFR-RES-04) |
| V17 | — | CRIT | Packages | every `requirements*.txt` pin importable at exactly the pinned version |
| V18 | — | CRIT after Day 0 fetch, else WARN | MiniLM offline | subprocess offline load per §8.2 step 5 |
| V19 | — | WARN | Assets | files in `assets/ASSET_MANIFEST.json` exist with matching sha256 (placeholder fallback A2 exists) |
| V20 | — | INFO | Java | `java -version` parsed → `capabilities.java` (decides doc 21 tier 2) |
| V21 | — | INFO (WARN on LT after D12) | Playwright | Chromium installed for Python Playwright |
| V22 | — | INFO | Git | branch, dirty flag, remote reachable |
| V23 | — | WARN | Sync root | `paths.sync_root` from the machine YAML exists and is writable (doc 07 §7) |
| V24 | Q | WARN | HF_HOME | set and pointing at the intended cache |
| V25 | — | INFO | Windows reminders | prints W9/W10/W11 reminders (they cannot be read reliably from code) |

### 9.3 VRAM-cap test (reference implementation of V09/V10)
```python
import torch
from text2mario.utils.hw import apply_vram_cap

GiB = 2**30
frac = apply_vram_cap(4.5)            # = 4.5 GiB / props.total_memory, set before ANY CUDA allocation
free, total = torch.cuda.mem_get_info()
try:
    x = torch.empty(5 * GiB, dtype=torch.uint8, device="cuda")   # must fail
    del x
    raise SystemExit("V09 FAIL: cap not enforced")
except torch.OutOfMemoryError:
    pass
torch.cuda.empty_cache()
if FULL:
    y = torch.empty(4 * GiB, dtype=torch.uint8, device="cuda")   # V10: should succeed under the cap
    del y
    torch.cuda.empty_cache()
```
`apply_vram_cap` must be called before the first CUDA tensor is created in the process (the trainer, eval scripts and app do this in `bootstrap` right after `bootstrap_env()`). The cap limits the PyTorch caching allocator's **reserved** memory, which is stricter than H2's "peak allocated" budget. The CUDA context (~0.3–0.6 GiB) sits outside the cap, which is why H2 leaves headroom.

### 9.4 `env/capabilities.json` (one key per machine; each machine updates only its own key)
```json
{
  "laptop_4050": {
    "updated_utc": "<iso8601>",
    "os": "<platform.platform()>",
    "python": "3.12.<x>",
    "torch": "<torch.__version__>",
    "torch_channel": "cu13 | cu126",
    "cuda_runtime": "<torch.version.cuda>",
    "cudnn": "<torch.backends.cudnn.version()>",
    "driver": "<nvml driver string>",
    "gpu": "NVIDIA GeForce RTX 4050 Laptop GPU",
    "compute_capability": [8, 9],
    "vram_total_gib": 6.0,
    "vram_cap_gib": 4.5,
    "vram_cap_fraction": 0.75,
    "bf16": true,
    "bf16_matmul_tflops": "<measured>",
    "sdpa_backends": {"flash": "<bool>", "mem_efficient": "<bool>", "math": "<bool>"},
    "nvml_telemetry": true,
    "java": {"present": false, "version": null},
    "playwright_chromium": false,
    "hf_model_cached": true
  },
  "desktop_3060ti": { "...": "same fields" }
}
```

## 10. Optional: Java for `MarioEval.jar` (V6)

| OS | Install | Verify |
|---|---|---|
| Windows | `winget install -e --id EclipseAdoptium.Temurin.21.JRE` | new shell → `java -version` |
| Ubuntu 24.04 | `sudo apt install -y openjdk-21-jre-headless` | `java -version` |
| Ubuntu 22.04 | `sudo apt install -y openjdk-17-jre-headless` (any Java 8+ works for the jar) | `java -version` |

If Java can't be installed, record `"java": {"present": false}`. G-P5b then uses the 50-level hand-labelled fallback (ledger §9). The jar is only ever run as a subprocess with a timeout (doc 21, NFR-SEC-04).

## 11. Optional: Playwright (Day 13 smoke test) and Node

Python Playwright drives headless Chromium, so **Node.js is not required**.
```powershell
python -m pip install playwright pytest-playwright      # already in requirements-dev.txt
python -m playwright install chromium
python -m playwright install msedge                     # optional, Edge channel for AT-16 (Windows)
```
```bash
python -m playwright install --with-deps chromium       # Linux: also installs system libraries (needs sudo)
```
Node.js is optional, only if someone wants JS linting of `app/static/game/game.js` (`winget install OpenJS.NodeJS.LTS` / `sudo apt install nodejs npm`). Nothing in the build depends on it.

## 12. Streamlit local settings
`.streamlit/config.toml` is specified in doc 28. Environment-level requirements: the local profile binds to `localhost` (the cloud profile is managed by Streamlit Community Cloud, ledger §3b), usage statistics are off (`browser.gatherUsageStats = false`), and the launcher sets `T2M_OFFLINE=1` (NFR-OFF-01, NFR-SEC-02). Always run `streamlit run app/streamlit_app.py` **from the repo root**, so the project `.streamlit/` folder is picked up.

## 13. Day-0 sign-off checklist (do it on DT, then on LT)

- [ ] §3: driver read; channel decided; the same channel noted for both machines in `PROGRESS.md`.
- [ ] §4: Windows settings W1–W11 applied (W8–W10 on the laptop at minimum).
- [ ] §5: venv on Python 3.12; torch installed via the install script; **identical `torch.__version__` on both**.
- [ ] §6: pinned requirements installed; `env/lock_<machine>.txt` written; `compare_env.py` clean.
- [ ] §8: `fetch_models.py` passes, including the offline self-test; `env/models.json` committed.
- [ ] §9: `verify_env.py --full` exits 0; `env/capabilities.json` has this machine's key; the report JSON is in `env/verify/`.
- [ ] §10/§11: Java and Playwright status recorded (optional items).
- [ ] `pre-commit run --all-files` and `pytest -q` pass.
- [ ] Commit: `chore(env): day-0 setup <machine>`.

## 14. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `torch.cuda.is_available()` is False on Windows; `torch.version.cuda` is None | A CPU-only wheel was installed (plain `pip install torch` on Windows can resolve to a CPU build) | `pip uninstall -y torch` → `scripts/install_torch.ps1` with the CUDA index |
| `CUDA driver version is insufficient for CUDA runtime version` | Driver < 580 with a CUDA 13 wheel | Update the driver (§3.3), or move **both** machines to `cu126` (§3.2 step 3) |
| V08 fails (torch versions differ) | Machines were installed on different days | Install the exact same `torch.__version__` on both (§5 rule); update both lock files |
| V09 does not raise OOM | Cap applied after a CUDA allocation, or on the wrong device | Call `apply_vram_cap` right after `bootstrap_env()`, before any `.cuda()`/`device="cuda"` |
| V10 fails on the laptop | Other processes hold VRAM (browser GPU process, overlays, chat apps) | `nvidia-smi` → close them before training. The app itself needs far less than 4 GiB |
| Training on Windows becomes 5–20× slower with no error | **Sysmem fallback** (VRAM spilled to system RAM) | Set W9 *Prefer No Sysmem Fallback*; check Task Manager → GPU → "Shared GPU memory" usage |
| Laptop it/s far below Day-4 measurement | On battery, Quiet mode, thermal throttling, iGPU-only mode | AC power, Performance mode (W7/W8), check `nvidia-smi -q -d PERFORMANCE` clock-event reasons, improve airflow |
| Watchdog stops runs at 87 °C (H9) | Insufficient cooling | Lid open, laptop stand, cooler room; runs resume from the checkpoint via `scripts/queue.py` (doc 07 §10) |
| `Activate.ps1 cannot be loaded because running scripts is disabled` | Execution policy | §4 W1 |
| `UnicodeDecodeError: 'charmap' codec can't decode` | Windows default code page | §4 W2 and explicit `encoding="utf-8"` (doc 05 §3) |
| HF symlink warning on Windows | No Developer Mode | W13 or `HF_HUB_DISABLE_SYMLINKS_WARNING=1` (harmless) |
| Offline load: "couldn't connect to huggingface.co" | Snapshot not under this process's `HF_HOME` (set too late), or revision mismatch | Set `HF_HOME` before importing transformers (`bootstrap_env()` first); re-run `fetch_models.py` online; compare `env/models.json` |
| `Weights only load failed` in `torch.load` | Checkpoint contains non-primitive Python objects (numpy arrays, `random` state tuple) | Fix the saver to store tensors/primitives (doc 07 §5.3). **Do not** switch to `weights_only=False` for files from the sync folder |
| `NVML Shared Library Not Found` (Windows) | Broken driver install | Reinstall the NVIDIA driver (`nvml.dll` lives in `C:\Windows\System32`) |
| `Executable doesn't exist` (Playwright) | Browsers not downloaded | `python -m playwright install chromium` (`--with-deps` on Linux) |
| `java` not recognised | JRE missing or not on PATH | §10, open a new shell; or proceed with the G-P5b fallback |
| Port 8501 in use | Another Streamlit instance | `streamlit run app/streamlit_app.py --server.port 8502` |
| Git "Filename too long" | Windows path limit | §4 W3 |
| Overnight run died at 03:00 | Sleep, hibernate or a Windows Update reboot | §4 W5, W6, W11. Resume with `python scripts/launch.py --resume runs/<id>` (doc 07 §8) |
| Pre-commit takes too long | Slow tests in the fast suite | Mark them `@pytest.mark.slow`; the fast suite must stay < 60 s (NFR-MNT-01) |
| `pip install` is extremely slow on Windows | Defender real-time scanning | Optional W12 exclusion, or be patient |
