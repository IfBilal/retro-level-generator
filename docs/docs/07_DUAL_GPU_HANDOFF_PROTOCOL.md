# 07 — Dual-GPU Handoff Protocol (min-VRAM contract, checkpoints, sync, resume)

> **18-day plan note (2026-11-18, doc 02 §14): this protocol is SIMPLIFIED.** What is required: one owner per run recorded in `HANDOFF.md`, atomic checkpoints, the same hyperparameters on both machines, and **one** minimal resume drill (AT-08, SHOULD). The rest of this document (sync alternatives, recovery of crashed handoffs, fold re-balancing, retention nuances) is reference material to be used only if a problem actually occurs, not a gate. Quoted torch versions in examples are unverified (ledger V2).

> **Authority:** ledger H1–H10, V2, D4, M10–M12, CLAUDE.md §2.5/§2.6/§2.9/§2.12. This document guarantees two properties:
> 1. **Any job runs on either machine.** Desktop RTX 3060 Ti 8 GB (`desktop_3060ti`, tag **DT**) or laptop RTX 4050 6 GB (`laptop_4050`, tag **LT**).
> 2. **A run can move mid-flight** from one machine to the other and continue with full state, verified, without ever being trained on both machines at once.

---

## 0. Two-minute switching checklist

**On the source machine (releasing):**
1. `python scripts/handoff.py release --run <run_id> --note "<why>"`. This creates `runs/<run_id>/STOP`, waits for the trainer's graceful exit (exit code 75), verifies the final checkpoint, copies the bundle to the sync folder, sets status `RELEASED`, and commits and pushes `HANDOFF.md`.
2. Wait until the sync client shows "up to date" (bundle ≈ 0.45 GB).

**On the destination machine (acquiring):**
3. `git pull` → `python scripts/verify_env.py --quick` (must exit 0).
4. `python scripts/handoff.py acquire --run <run_id>`. This checks `RELEASED`, waits for the bundle, verifies every sha256, copies it into `runs/<run_id>/`, sets status `RUNNING` with this machine as owner, and commits and pushes `HANDOFF.md` **before** any training.
5. `python scripts/launch.py --resume runs/<run_id>`. Wait for the console line `RESUME VERIFIED` (§8.4).

If any step fails: **do not train.** Read the error, then §8.6 (resume failures) or §11 (divergence).

---

## 1. The minimum-VRAM contract

### 1.1 Contract
| Clause | Rule | Ledger |
|---|---|---|
| C1 | Every training, evaluation and app process calls `apply_vram_cap(4.5)` before its first CUDA allocation | H2, H3 |
| C2 | Global batch = 64 on both machines; hyperparameters are identical | H4, M10 |
| C3 | Machine YAMLs may override only the allow-listed performance keys (§3.2) | H4, CLAUDE.md §2.5 |
| C4 | The cap is never raised to "make something fit"; reduce the micro-batch and accumulate (§1.3), logged | CLAUDE.md §2.6, §2.12 |
| C5 | Precision bf16 autocast on both, no GradScaler; `torch.compile` off | H5, H6 |
| C6 | Dataset resident on the GPU as uint8; no DataLoader workers | H7 |

Because the desktop runs under the same 4.5 GiB cap, anything that runs there also fits the laptop (H3).

### 1.2 Expected memory budget (verify on Day 4 with `scripts/bench.py`)
| Component | Size | Inside the cap? |
|---|---|---|
| CUDA context + cuBLAS/cuDNN workspaces | ~0.3–0.6 GiB | No (outside the PyTorch allocator; H2 headroom covers it) |
| U-Net weights fp32 (18,312,013 params, H8) | 73 MB | Yes |
| EMA shadow fp32 | 73 MB | Yes |
| Gradients fp32 | 73 MB | Yes |
| AdamW moments (2 × fp32) | 146 MB | Yes |
| Adapter + LayerNorm (66,048 params + LN) | < 1 MB | Yes |
| Frozen MiniLM in bf16 (≈ 22.7 M params, conditional phases only) | ≈ 45 MB | Yes |
| Dataset uint8 (7,687 × 16 × 16) | 2 MB | Yes |
| Activations at batch 64, 16×16, bf16 | measured Day 4 (expected < 1.5 GiB) | Yes |
| **Expected peak allocated** | **< 2 GiB** (H8) | Headroom ≥ 2.5 GiB under the cap |

### 1.3 OOM fallback: micro-batch probe and gradient accumulation
At trainer start (after the cap, before step 1):
1. Probe one forward + backward pass on a synthetic batch with `micro_batch = 64`.
2. On `torch.OutOfMemoryError`: `torch.cuda.empty_cache()`, halve `micro_batch` (32 → 16 → 8), set `accum_steps = 64 / micro_batch`, and retry. Below 8 → exit 1 (something is badly wrong).
3. Log the event `fallback.grad_accum {micro_batch, accum_steps}` to `events.jsonl` and add a note to `PROGRESS.md`.
4. **Equivalence rules** (so accumulation changes nothing but numerics):
   - The model uses GroupNorm only (no BatchNorm), so splitting the batch does not change the normalisation statistics.
   - All random draws for an optimizer step (indices, t, ε, caption augmentation, CFG drops) are made **once for the full global batch of 64** on the CPU generators (§5.3) and then sliced into micro-batches. The random stream is therefore identical with or without accumulation, and across machines.
   - Each micro-batch loss is divided by `accum_steps`; `optimizer.step()`, the LR-scheduler step and the EMA update happen once per global batch.
5. Accumulation settings may differ between machines (they are recorded per lineage segment). They do not change the config hash.

Evaluation follows the same pattern. If a sampling batch OOMs, the evaluator splits it. Every sample's initial noise comes from a **per-sample derived seed** (`derive_seed(prompt_id, seed_idx)`), so batch composition never changes the samples.

### 1.4 Windows "Sysmem Fallback" (silent slowdown instead of OOM)
On Windows, recent NVIDIA drivers can spill VRAM into shared system memory instead of failing an allocation, which makes training 5–20× slower with no error. Required on both Windows machines: NVIDIA Control Panel → *CUDA – Sysmem Fallback Policy* = **Prefer No Sysmem Fallback** for the venv and base `python.exe` (doc 06 §4 W9). As a detector, the watchdog (§10) emits `perf.slowdown` when it/s falls more than 40 % below the `hw_report.json` baseline for 5 minutes with no thermal reason.

### 1.5 Measurement (`scripts/bench.py`, Day 4, both machines)
Writes `runs/<bench_run_id>/hw_report.json`:
```json
{"machine": "laptop_4050", "gpu": "...", "driver": "...", "torch": "...", "cap_gib": 4.5, "cap_fraction": 0.75,
 "micro_batch": 64, "accum_steps": 1, "peak_allocated_gib": 0.0, "peak_reserved_gib": 0.0,
 "it_s_train_median": 0.0, "samples_s_eval_ddim50_cfg": 0.0, "cudnn_benchmark": false, "date_utc": "..."}
```
Acceptance: `peak_allocated_gib ≤ 4.5` (H2) and `accum_steps == 1` on **LT** (expected). The bench measures `cudnn_benchmark` both off and on (§3.2) and records the speed-up.

## 2. Run IDs and run directories

| Item | Rule |
|---|---|
| Format | `{YYYYMMDD}T{HHMM}Z_{config_stem}[_{variant}]_s{seed}_{TAG}` (UTC time; TAG = **birth** machine, `DT` or `LT`) |
| Examples | `20261122T0415Z_cond_main_s0_DT` · `20261126T1600Z_cv_fold4_s0_LT` · `20261124T1530Z_x01_aux_ce_s0_LT` |
| Collisions | Machine suffixes make cross-machine collisions impossible. On the same machine, if the directory exists, append `-2`, `-3`, … |
| Identity | The run ID **never changes when the run moves**. The current owner lives in `HANDOFF.md`; where each step ran lives in `runs/<id>/lineage.json` |
| Fork / orphan names | `<run_id>__fork_<TAG>_<step>` (split-brain branch, §11a) · `<run_id>__orphan_<TAG>_<step>` (steps trained after ownership was lost, §4.3 recover) |
| Directory layout | doc 05 §10, plus `heartbeat.json` (§4.8) |

## 3. Machine profiles and the forbidden-override guard

### 3.1 Machine YAMLs (complete files)
```yaml
# configs/machine/desktop_3060ti.yaml
machine:
  name: desktop_3060ti
  tag: DT
  gpu_name_pattern: "RTX 3060 Ti"
  expected_vram_gib: 8
paths:
  sync_root: "D:/OneDrive/text2mario_sync"        # edit per machine (§7); forward slashes are fine on Windows
  java_bin: null                                  # null = use PATH (doc 21)
overrides:                                        # ONLY allow-listed keys (§3.2)
  perf.num_workers: 0
  perf.cudnn_benchmark: false
  perf.compile: false
  logging.log_every: 50
```
```yaml
# configs/machine/laptop_4050.yaml
machine:
  name: laptop_4050
  tag: LT
  gpu_name_pattern: "RTX 4050"
  expected_vram_gib: 6
paths:
  sync_root: "C:/Users/<you>/OneDrive/text2mario_sync"
  java_bin: null
overrides:
  perf.num_workers: 0
  perf.cudnn_benchmark: false
  perf.compile: false
  logging.log_every: 50
```
Selection: `utils.hw.detect_machine()` matches `gpu_name_pattern` against `torch.cuda.get_device_name(0)`. `T2M_MACHINE` or `--machine` overrides it. An `unknown` machine may not train.

### 3.2 Allow-list
| Key | Default (both) | Why it may differ |
|---|---|---|
| `perf.num_workers` | 0 | Performance only. With H7 the training data never uses workers; the key exists for auxiliary loaders |
| `perf.cudnn_benchmark` | false | Performance only. May be set to `true` on a machine **only if** the Day-4 bench shows ≥ 10 % speed-up. Evaluation always forces `false` (`deterministic_eval`) |
| `perf.compile` | false | H6 (fragile on Windows) |
| `logging.log_every` | 50 | Logging cadence, not a hyperparameter |

Ledger H4 names `num_workers`, `cudnn.benchmark` and `compile`. CLAUDE.md §2.5 adds `log_every`. The allow-list is the union of the two.

### 3.3 Config hash
`config_hash = sha256(canonical_json(cfg_experiment))`, where `cfg_experiment` = `base.yaml` ← experiment overlay ← CLI `--set` values, **minus** the `machine`, `paths` and `overrides` sections **and minus the four allow-listed keys**. Canonical JSON uses sorted keys, separators `(",", ":")`, UTF-8, and `repr`-exact floats. The same experiment therefore has the **same hash on both machines**. The hash is written to `runs/<id>/config_hash.txt`, every checkpoint and every sidecar. Resume requires equality (§8.2).

### 3.4 Guard algorithm (`utils/config.py::load_config`)
```python
ALLOWED = {"perf.num_workers", "perf.cudnn_benchmark", "perf.compile", "logging.log_every"}
MACHINE_SECTIONS = {"machine", "paths", "overrides"}

def guard(machine_yaml: dict, experiment_yaml: dict) -> None:
    if extra := set(machine_yaml) - MACHINE_SECTIONS:
        raise ConfigError(f"machine YAML has forbidden top-level keys {sorted(extra)}")
    if bad := set(flatten(machine_yaml.get("overrides", {}))) - ALLOWED:
        raise ConfigError(f"machine YAML overrides non-allow-listed keys {sorted(bad)} (H4, CLAUDE.md 2.5)")
    if clash := MACHINE_SECTIONS & set(experiment_yaml):
        raise ConfigError(f"experiment YAML must not contain machine sections {sorted(clash)}")
```
The trainer exits with **code 3** on `ConfigError`, printing the offending keys. `tests/test_config_guard.py` covers: an allowed override passes; `optim.lr` in overrides fails; an unknown top-level key fails; the hash is identical for DT and LT profiles of the same experiment; the hash changes when `optim.lr` changes.

## 4. `HANDOFF.md` lock protocol

### 4.1 Principles
1. **Single owner.** At any moment a run has at most one owner machine. Only the owner may write checkpoints for that run ID.
2. **Git is the lock.** Ownership changes are commits to `HANDOFF.md` that must be **pushed before** training starts. A rejected push (non-fast-forward) means someone else changed the state: pull, re-read, and abort if the run is no longer available. This is optimistic locking.
3. **Machine-edited only.** `HANDOFF.md` is edited only by `scripts/handoff.py` and `scripts/launch.py`, through `utils/handoff_state.py`. The machine-readable state lives in a YAML block; the human table and the log are re-rendered from it on every write.
4. **The trainer never does network I/O.** At start it reads the local (freshly pulled) `HANDOFF.md` and refuses to run (**exit 4**) if this machine is not the owner of a `RUNNING`/`PAUSED` run, unless the run is brand new (registered by `launch.py` in the same invocation).

### 4.2 Status state machine
| From | Event (command) | To | Who |
|---|---|---|---|
| — | `launch.py --config …` (new run; registers, commits, pushes) | RUNNING (owner = this machine) | any machine |
| RUNNING | `handoff.py pause --run` (graceful stop, keep ownership) | PAUSED | owner |
| PAUSED | `launch.py --resume` on the owner | RUNNING | owner |
| RUNNING / PAUSED | `handoff.py release --run` | RELEASED (owner = null) | owner |
| RELEASED | `handoff.py acquire --run` | RUNNING (owner = this machine) | any machine |
| RUNNING | trainer finishes (max steps or early stop) → `handoff.py finish --run` | FINISHED | owner |
| RUNNING / PAUSED | owner unreachable → `handoff.py recover --run --reason` | STALE → RUNNING (new owner) | other machine |
| any | `handoff.py abort --run --reason` | ABORTED | anyone (reason required) |

A thermal or battery stop (exit 75, §10) does **not** change the status. The run stays `RUNNING` with the same owner, and `queue.py` resumes it on the same machine.

### 4.3 Commands (`scripts/handoff.py`)
| Command | Steps | Failure behaviour |
|---|---|---|
| `release --run ID [--note]` | 1) touch `runs/ID/STOP`; 2) wait for trainer exit 75 (timeout 15 min); 3) `verify_ckpt` on `latest.json`; 4) build the bundle (§7.3) in the sync folder; 5) state → RELEASED, history entry; 6) commit `handoff: release ID @ step N from TAG`, push (retry with pull-rebase ≤ 3×) | Any failure leaves the status unchanged and prints the reason. Training can be resumed locally |
| `acquire --run ID` | 1) `git pull --rebase`; 2) require status RELEASED; 3) wait for the bundle (poll 10 s, timeout 60 min), verify every sha256 against `bundle_manifest.json`; 4) copy into `runs/ID/` with atomic writes; 5) `verify_ckpt`; 6) state → RUNNING owner = this machine; 7) commit `handoff: acquire ID @ step N by TAG`, push | Push rejected → pull, re-check; if no longer RELEASED → abort without training |
| `pause --run ID` | STOP + wait for exit 75; status PAUSED; commit, push | — |
| `finish --run ID` | Called by `launch.py` after exit 0: status FINISHED, publish final results (§7.3 `published/`) | — |
| `publish --run ID --what final\|snapshot\|eval` | Copy without an ownership change: `final` (final EMA + eval JSON), `snapshot` (latest full checkpoint as a crash-safety copy), `eval` (eval JSON only) | — |
| `recover --run ID --reason "…"` | Owner unavailable (crash, laptop away): mark STALE with the reason; take the newest **verified** bundle or snapshot in the sync folder; claim ownership (RUNNING); commit, push | Steps the old owner trained after its last bundle or snapshot are lost. If that machine returns, its newer local checkpoints become `__orphan_…` and are never merged |
| `status` | Print the rendered table plus heartbeat ages (§4.8) | — |

Snapshot cadence for crash safety: `launch.py --publish-every 10000` publishes a snapshot every 10k steps. It is a CLI flag, not YAML, so it never touches the config hash or the allow-list. Recommended for long runs on the laptop.

### 4.4 `HANDOFF.md` template
````markdown
# HANDOFF — GPU run ownership

> Edited only by `scripts/handoff.py` / `scripts/launch.py`. Manual edits break the lock (docs/07 §4).
> Rule: `git pull` before you start, stop, or move any GPU job.

<!-- handoff-state:begin -->
```yaml
schema: 1
updated_utc: "2026-11-21T17:10:00Z"
updated_by: desktop_3060ti
machines:
  desktop_3060ti: {state: idle, run: null, last_seen_utc: "2026-11-21T17:10:00Z"}
  laptop_4050:    {state: busy, run: "20261121T1500Z_x02_ema995_s0_LT", last_seen_utc: "2026-11-21T16:55:00Z"}
runs:
  "20261121T1330Z_cond_main_s0_DT":
    status: RELEASED            # RUNNING | PAUSED | RELEASED | FINISHED | ABORTED | STALE
    owner: null                 # machine profile while RUNNING/PAUSED
    step: 24000
    bundle: "runs/20261121T1330Z_cond_main_s0_DT/handoff"     # relative to paths.sync_root
    ckpt_sha256: "3fa9c1e2b7d0<...>"
    config_hash: "9b1c44e0a2f1<...>"
    history:
      - {utc: "2026-11-21T13:30:00Z", machine: desktop_3060ti, action: launch,  step: 0}
      - {utc: "2026-11-21T17:10:00Z", machine: desktop_3060ti, action: release, step: 24000, note: "evening switch"}
  "20261121T1500Z_x02_ema995_s0_LT":
    status: RUNNING
    owner: laptop_4050
    step: 6000
    bundle: null
    ckpt_sha256: null
    config_hash: "1d07ab93c5e2<...>"
    history:
      - {utc: "2026-11-21T15:00:00Z", machine: laptop_4050, action: launch, step: 0}
```
<!-- handoff-state:end -->

## Runs (rendered — do not edit)
| run_id | status | owner | step | ckpt sha (12) | updated (PKT) | note |
|---|---|---|---|---|---|---|
| 20261121T1330Z_cond_main_s0_DT | RELEASED | — | 24000 | 3fa9c1e2b7d0 | 2026-11-21 22:10 | evening switch |
| 20261121T1500Z_x02_ema995_s0_LT | RUNNING | LT | 6000 | — | 2026-11-21 20:00 | — |

## Log (append-only, rendered)
- 2026-11-21 22:10 PKT · DT · release · 20261121T1330Z_cond_main_s0_DT @ 24000 · sha 3fa9c1e2b7d0
````
Times are stored in UTC and rendered in the operator's local zone (Asia/Karachi, PKT).

### 4.5 Push-conflict loop (inside every state-changing command)
```text
for attempt in 1..3:
    git pull --rebase
    state = read_state(HANDOFF.md)
    if not precondition(state): abort("run is now <status> owned by <owner>")
    state = apply(state, change); write_state(HANDOFF.md, state)
    git commit -m "<message>"
    if git push succeeds: return OK
    git reset --hard HEAD~1   # drop our commit, retry from fresh state
abort("could not acquire lock after 3 attempts")
```

### 4.6 Offline machine
Ownership changes **require** a working `git push`. A machine without internet may run evaluation and the app (read-only; no `HANDOFF.md` change needed), but must not start, acquire or release training runs. For a machine that must be offline, use USB mode (§7.5) and register the change as soon as it is back online. Until then, the other machine must not touch that run.

### 4.7 `PROGRESS.md` entry template (one entry per session)
```markdown
## 2026-11-21 (Day 8) — laptop_4050
**Done:** A08 adherence.py (tests green); G08 normaliser v1 at 27/30 on P-casual-dev.
**In-flight runs:** 20261121T1330Z_cond_main_s0_DT — RELEASED @24000, to be acquired by LT tonight.
**Blocked:** —
**Next:** Day 9 CFG sweep shard B on LT.
**Assumptions:** (any choice made where docs were silent, with the config flag that controls it)
**Doc conflicts:** (doc A says X, doc B says Y → followed ledger Z)
**Buffer log:** used 0.0 / 1.5 · zone GREEN (doc 04 §6.2)
**Fallbacks used:** none
```

### 4.8 Heartbeat
The trainer writes `runs/<id>/heartbeat.json` every 60 s: `{utc, step, it_s, loss_ema100, gpu_temp_c, vram_gib, status}`. If `paths.sync_root` is set, it also copies this tiny file to `<sync_root>/runs/<id>/heartbeat.json` every 10 minutes, so the other machine can see that the run is alive. `handoff.py status` and `scripts/status.py` flag a RUNNING run whose heartbeat is older than 30 minutes.

## 5. Checkpoint schema

### 5.1 Files written at every save
| File | Content | Loader |
|---|---|---|
| `checkpoints/ckpt_step{N:07d}.pt` | Full training state (§5.2) | `torch.load(path, map_location="cpu", weights_only=True)` |
| `checkpoints/ckpt_step{N:07d}.json` | Sidecar: `{file, sha256, bytes, step, config_hash, created_utc, machine, torch, val_loss_det, fingerprint}` | `json` |
| `checkpoints/ema_step{N:07d}.safetensors` | EMA weights only (U-Net + adapter), with metadata `{config_hash, step, git_sha}` | `safetensors.torch.load_file` (app, eval) |
| `checkpoints/latest.json`, `best.json` | Pointers `{file, step, sha256, metric?, value?}` | `json` |

### 5.2 Keys of the full training state (`.pt`)
| Key | Type | Content |
|---|---|---|
| `schema_version` | int | `1` |
| `kind` | str | `"train_state"` |
| `run_id` | str | §2 |
| `step` | int | Optimizer steps completed |
| `samples_seen` | int | `step × 64` |
| `model` | state_dict (fp32 tensors) | U-Net (+ adapter + LayerNorm in conditional runs) |
| `ema` | state_dict (fp32 tensors) | EMA shadow parameters |
| `ema_num_updates` | int | For the warm-up decay `min(0.999, (1+n)/(10+n))` (M11) |
| `optimizer` | `AdamW.state_dict()` | Moments, step counters, param-group hyperparameters |
| `lr_scheduler` | dict | `{"last_step": int}` (the LR is a pure function `lr_at(step)`, M10) |
| `grad_accum` | dict | `{"micro_batch": int, "accum_steps": int}` of the segment that saved it |
| `sampler_state` | dict | `{"epoch": int, "position": int, "perm": LongTensor, "gen_state": ByteTensor}` |
| `rng.sampler_gen` | ByteTensor | CPU generator for batch indices |
| `rng.aug_gen` | ByteTensor | CPU generator for caption augmentation + CFG null drops (D8) |
| `rng.noise_gen` | ByteTensor | CPU generator for t and ε (§5.3) |
| `rng.torch_cpu` | ByteTensor | `torch.get_rng_state()` |
| `rng.torch_cuda` | list[ByteTensor] | `torch.cuda.get_rng_state_all()` (dropout masks; not portable across GPUs, §9) |
| `rng.numpy` | dict | `np.random.default_rng` PCG64 `bit_generator.state` (nested dict of str/int) |
| `rng.python` | dict | `{"version": int, "internal": list[int] (625), "gauss_next": float or None}` from `random.getstate()` |
| `config` | dict | Resolved experiment config (machine sections removed) |
| `config_hash` | str | §3.3 |
| `split_lock_sha256` | str | Content of `data/splits/LOCK` (D4) |
| `data_manifest_hash` | str | Content hash from `data/MANIFEST.json` (§11e) |
| `git_sha`, `git_dirty` | str, bool | At save time |
| `torch_version`, `cuda_version`, `cudnn_version` | str, str, int | Environment |
| `machine` | dict | `{"name", "tag", "gpu", "driver"}` of the saving machine |
| `best` | dict | `{"metric": "dev_composite", "value": float, "step": int}` (E4) |
| `early_stop` | dict | `{"evals_since_improvement": int, "patience": 5}` (M12) |
| `val_loss_det` | dict | Deterministic per-t-bucket EMA validation loss at this step (`{"t0": ..., ..., "t9": ..., "mean": ...}`) |
| `fingerprint` | dict | Next-batch fingerprint (§8.4 V-R4): `{"next_idx8": list[int], "captions_sha256": str, "eps16": list[float], "t8": list[int]}` |
| `metrics_last` | dict | Last logged train metrics |
| `wallclock_s` | float | Cumulative training time |
| `lineage` | list[dict] | Segments `{machine, tag, start_step, end_step, started_utc, ended_utc, torch, driver, micro_batch, accum_steps}` |
| `created_utc` | str | ISO-8601 |

### 5.3 RNG portability rule (required by this protocol; implemented in doc 16's trainer)
1. Everything that decides **what** is trained in a step (batch indices, timesteps t, noise ε, caption augmentation choices, CFG null-caption drops) is drawn from the **three dedicated CPU `torch.Generator`s** above, for the whole global batch at once, and then copied to the GPU (pinned memory, `non_blocking=True`). Cost: ε is 64 × 13 × 16 × 16 = 212,992 floats ≈ 0.85 MB per step, which is negligible. The result is identical random streams on both machines and under any accumulation setting (§1.3).
2. Dropout inside the network uses the CUDA default generator. Its stream depends on the GPU's launch geometry, so it is **not** portable across GPUs. This is accepted (§9).
3. Every RNG state is stored as a uint8 tensor or as plain Python primitives, so that `torch.load(weights_only=True)` works. Verified on torch 2.15.0 (2026-11-17): Python `random` state as a list, the numpy PCG64 state dict and generator `ByteTensor`s all round-trip. A legacy `np.random.get_state()` tuple (which contains an ndarray) fails with `UnpicklingError`. **Never store it.**

### 5.4 Reference save/restore of RNG state
```python
def capture_rng(gens: dict[str, torch.Generator], np_rng: np.random.Generator) -> dict:
    v, internal, gauss = random.getstate()
    return {
        **{f"rng.{k}": g.get_state() for k, g in gens.items()},          # sampler_gen, aug_gen, noise_gen
        "rng.torch_cpu": torch.get_rng_state(),
        "rng.torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "rng.numpy": np_rng.bit_generator.state,
        "rng.python": {"version": v, "internal": list(internal), "gauss_next": gauss},
    }

def restore_rng(s: dict, gens: dict[str, torch.Generator], np_rng: np.random.Generator) -> None:
    for k, g in gens.items():
        g.set_state(s[f"rng.{k}"])
    torch.set_rng_state(s["rng.torch_cpu"])
    if torch.cuda.is_available() and s["rng.torch_cuda"]:
        torch.cuda.set_rng_state_all(s["rng.torch_cuda"])   # single-GPU on both machines
    np_rng.bit_generator.state = s["rng.numpy"]
    p = s["rng.python"]
    random.setstate((p["version"], tuple(p["internal"]), p["gauss_next"]))
```

### 5.5 Sizes
Full state ≈ 0.29–0.30 GB (weights 73 MB + EMA 73 MB + two AdamW moments 146 MB + small items; gradients are not saved). EMA `.safetensors` ≈ 73 MB. A handoff bundle (one full state + EMA + logs) ≈ 0.45 GB.

## 6. Atomic save and retention

### 6.1 Atomic write (CLAUDE.md §2.9)
```python
def atomic_torch_save(obj: dict, path: Path) -> str:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    digest = sha256_file(tmp)
    os.replace(tmp, path)                       # atomic on the same filesystem (Windows and POSIX)
    if os.name == "posix":                      # persist the rename
        dfd = os.open(path.parent, os.O_RDONLY)
        try: os.fsync(dfd)
        finally: os.close(dfd)
    return digest
```
Order per save: `.pt` → `.safetensors` (same pattern via `safetensors.torch.save_file` to a tmp name) → sidecar `.json` (contains the sha256) → `latest.json` pointer (atomic). A crash at any point leaves the previous `latest.json` pointing at a complete, verified checkpoint.

### 6.2 When a checkpoint is written
Every `checkpoint.every_steps` (default **2,000**, aligned with the M12 dev-evaluation cadence); on graceful stop (STOP file, Ctrl+C, watchdog); at the end of training; and the `best.json` pointer is updated after each dev evaluation that improves E4.

### 6.3 A "good" checkpoint
The file exists, its sha256 equals the sidecar's, it loads with `weights_only=True`, it contains every key in §5.2, and `schema_version == 1`. `scripts/verify_ckpt.py <path>` performs these checks (exit 0/1).

### 6.4 Retention policy
| Run state | Full `.pt` kept | EMA `.safetensors` kept |
|---|---|---|
| Active | **last 2 good** (never fewer, CLAUDE.md §2.9) + the one referenced by `best.json` | latest 2 + best + milestones (if `checkpoint.milestone_every_steps > 0`; `cond_main`/`final` use 10,000) |
| Finished (not frozen) | final + previous + best | final + best (+ milestones for `cond_main`) |
| Frozen (in `FROZEN.json`) | everything listed in `FROZEN.json`, forever; also archived to `<sync_root>/frozen/` | same |

Pruning happens only in the trainer (after a new save verifies) or via `scripts/prune_runs.py` (dry-run by default; `--apply` to delete). A file referenced by `best.json`, `latest.json` or `FROZEN.json` is never deleted. Expected footprint: an active 60k run ≈ 1.2–1.6 GB; a finished run ≈ 1.0 GB; the whole project ≤ 15 GB per machine (NFR-RES-04).

Keys used here (doc 17 must carry them in `base.yaml`):
```yaml
checkpoint:
  every_steps: 2000            # aligned with M12 eval cadence
  keep_last: 2                 # CLAUDE.md 2.9 — never < 2
  milestone_every_steps: 0     # cond_main.yaml / final.yaml set 10000
```

## 7. Sync between machines

### 7.1 Default: a cloud-synced folder
**Default = a cloud-synced folder** (OneDrive, built into Windows; or Google Drive for desktop), at `paths.sync_root` in each machine YAML.

| Why this default | Detail |
|---|---|
| Machines need not be online at the same time | The desktop can release and switch off; the laptop acquires later |
| Built into Windows / free | OneDrive free tier 5 GB (1 TB with Microsoft 365); Google Drive free tier 15 GB |
| Small footprint | Only handoff bundles (≈ 0.45 GB each), published results, and `frozen/` are synced; `handoff.py acquire` deletes older bundles of the same run after a successful acquisition |
| Safety net | Provider recycle bin and version history |

Required client settings: mark `text2mario_sync` **"Always keep on this device"** (OneDrive) or use **mirror** mode (Google Drive), so files are not online-only placeholders. `acquire` also reads every file fully (hydration) before verifying it.

### 7.2 Sync-folder layout
```text
<sync_root>/                                  e.g. D:/OneDrive/text2mario_sync
├── runs/<run_id>/
│   ├── handoff/                              latest release bundle (one per run)
│   │   ├── ckpt.pt  ckpt.json  ema.safetensors
│   │   ├── config.yaml  config_hash.txt  meta.json  lineage.json
│   │   ├── metrics.csv  events.jsonl  tb.zip
│   │   ├── bundle_manifest.json              {file: sha256, bytes} for every file above
│   │   └── BUNDLE_COMPLETE                   written last (informational only)
│   ├── snapshot/                             optional crash-safety copy (--publish-every)
│   ├── published/                            final EMA + eval JSON (FINISHED runs, CV folds)
│   └── heartbeat.json                        §4.8
└── frozen/                                   FROZEN.json, final.yaml, model_ema.safetensors (+ sidecars)
```

### 7.3 Push (release / publish)
1. Copy each file to `<name>.partial`, fsync, rename to `<name>`.
2. Write `bundle_manifest.json` (sha256 and size of every file).
3. Write `BUNDLE_COMPLETE`.
4. **The receiver never trusts markers.** Cloud clients may deliver files out of order, so the receiver verifies every file against `bundle_manifest.json`, which itself must parse and match the `ckpt_sha256` recorded in `HANDOFF.md`.

### 7.4 Pull (acquire)
Poll every 10 s (timeout 60 min, progress printed). For each file: hydrate (read it completely), verify sha256, copy into `runs/<id>/…` with an atomic write. `ckpt.pt` is restored to `checkpoints/ckpt_step{N:07d}.pt` and `latest.json` is rewritten locally. Any mismatch → wait and retry. After the timeout → abort, with the status still RELEASED.

### 7.5 Alternatives
| Option | When to use | How |
|---|---|---|
| **Syncthing** (P2P, LAN or internet) | Both machines are often on at the same time; no cloud quota | Share the `text2mario_sync` folder between the two devices; same layout; same commands |
| **USB / external SSD** | No internet, or viva-day emergency | `handoff.py release --run ID --to E:/t2m_bundle` then `handoff.py acquire --run ID --from E:/t2m_bundle`. The `HANDOFF.md` change is committed when the machine is next online (§4.6) |
| **Private Hugging Face Hub model repo** (optional) | Versioned archive of `frozen/` and milestone EMA weights; off-site backup | `HfApi().create_repo("<user>/text2mario-ckpts", private=True)`, `HfApi().upload_folder(...)`, `hf_hub_download(...)` with a token in `.env`; never needed by the app at runtime |
| Git / Git LFS | **Never** for checkpoints | Blocked by the pre-commit hook (doc 05 §8) |

### 7.6 What is never synced
Raw data (each machine runs `scripts/fetch_data.py` from the pinned commit and must reproduce the same content hashes, §11e), virtual environments, the HF cache (each machine runs `scripts/fetch_models.py` at the pinned revision), and `reports/` (they travel through git).

## 8. Resume procedure and verification

### 8.1 Command
```bash
python scripts/launch.py --resume runs/<run_id> [--publish-every 10000] \
       [--allow-code-change "<reason>"] [--allow-torch-mismatch "<reason>"]
```
Every `--allow-*` flag is logged to `events.jsonl` and `PROGRESS.md` (CLAUDE.md §2.12).

### 8.2 Pre-load checks (refuse on failure)
| # | Check | Exit code on failure |
|---|---|---|
| P1 | `HANDOFF.md` (local, after pull): this machine owns the run, status RUNNING or PAUSED | 4 |
| P2 | `latest.json` → file exists; sha256 equals the sidecar; `verify_ckpt` passes | 5 |
| P3 | `config_hash` recomputed from `runs/<id>/config.yaml` (+ CLI `--set`) equals the checkpoint's | 3 |
| P4 | `split_lock_sha256` equals the current `data/splits/LOCK`; `data_manifest_hash` equals the current content hash | 5 |
| P5 | torch **minor** version equals the checkpoint's (V2), unless `--allow-torch-mismatch` | 5 |
| P6 | `git diff --name-only <ckpt.git_sha>..HEAD -- src/text2mario/{models,diffusion,train,data,text}` is empty, unless `--allow-code-change` | 5 |

### 8.3 Restore order
1. `bootstrap_env()` → `apply_vram_cap(4.5)` → watchdog start.
2. Build the model, adapter, EMA and optimizer from the config; run the micro-batch probe (§1.3). The micro-batch may differ from the previous segment; record it.
3. Load `model`, `ema` (+ `ema_num_updates`), `optimizer` (then move its state tensors to the device), the LR step, `sampler_state`, the three CPU generators, and the global RNGs (§5.4), plus `best` and `early_stop`.
4. Append a new lineage segment `{machine, start_step: step, …}`.

### 8.4 Resume verification (before the first training step)
| ID | Check | Tolerance | Rationale |
|---|---|---|---|
| V-R1 | Recompute the deterministic EMA validation loss (fixed noise and t per dev window) and compare with `val_loss_det` in the checkpoint | relative \|Δ\| ≤ **0.1 %** same machine; ≤ **1.0 %** cross machine (bf16 kernel differences) | Proves the weights and EMA were restored and evaluate consistently |
| V-R2 | `optimizer.param_groups[0]["lr"] == lr_at(step)` | ≤ 1e-12 absolute | LR schedule continuity (M10) |
| V-R3 | `ema_num_updates` equals the stored value; the decay value equals the formula | exact | EMA continuity (M11) |
| V-R4 | **Next-batch fingerprint:** the next 8 sampler indices, sha256 of the next batch's augmented captions, the first 8 timesteps and the first 16 ε values equal `fingerprint` | **exact** (all from CPU generators, §5.3) | Data order, augmentation and noise streams continue exactly, on either GPU |
| V-R5 | During the first 100 steps: running-mean loss ≤ 2 × the running mean of the last 100 pre-handoff steps (`metrics.csv`) | factor 2 | Catches corrupted optimizer state (the Day-4 drill criterion, doc 03) |

On success, print `RESUME VERIFIED step=N machine=TAG Δval=…%` and log `resume_verified`. A V-R1–V-R4 failure stops before training (exit 5); a V-R5 failure saves nothing new and stops (exit 5).

### 8.5 Day-4 resume drill (doc 03 Day 4)
1. DT: `launch.py --config configs/uncond_cosine.yaml --set train.max_steps=1500` with `checkpoint.every_steps=500`. Stop at step 1,000 via `handoff.py release`.
2. LT: `handoff.py acquire` → `launch.py --resume` → train to step 1,500.
3. Pass: V-R1…V-R5 all pass; the step, LR and EMA counters are continuous in `metrics.csv`; the plot `reports/dev/resume_drill.png` shows no discontinuity. Record `reports/dev/resume_drill.json`.

### 8.6 Resume failures
| Failure | Meaning | Action |
|---|---|---|
| P1 (exit 4) | Not the owner | `git pull`; if RELEASED → `acquire`; if owned by the other machine → stop (split-brain risk, §11a) |
| P2 | Corrupt or incomplete file | Use the previous good checkpoint (`verify_ckpt` the second-latest); never delete it |
| P3 (exit 3) | Config changed | Revert the config, or start a **new** run. Never continue a run under a different config |
| P4 | Data or split changed | Regenerate data from the pinned source and compare content hashes (§11e). If the split changed → stop and investigate (D4) |
| P5 | Torch versions differ | Align both machines (doc 06 §5 rule) |
| P6 | Code changed | Revert, or `--allow-code-change "<reason>"` only for changes proven numerically irrelevant (logging, comments) |
| V-R1 | Weights or eval inconsistent | Re-run on the source machine; if it matches there, investigate cross-machine numerics (§9, §11d) |
| V-R4 | RNG or sampler state not restored | Bug in save/restore: fix it before any further training |

## 9. Cross-GPU numerical non-determinism

### 9.1 Sources
| Source | Same machine, same process | Same machine, after resume | Cross machine, after resume |
|---|---|---|---|
| Batch indices, t, ε, caption augmentation, CFG drops (CPU generators) | identical | identical | **identical** |
| Dropout masks (CUDA default generator) | identical | identical (state restored) | different stream (launch geometry differs: 38 SMs on the 3060 Ti vs 20 on the 4050 Laptop) |
| cuBLAS/cuDNN algorithm choice | fixed if `cudnn_benchmark` is off | may differ if benchmark is on | **different kernels and tiling** |
| Atomic-add backward kernels (e.g., nearest-upsample backward) | non-deterministic | non-deterministic | non-deterministic |
| bf16 rounding paths | same | same | differ with the kernels |

### 9.2 Expectations and policy
- **Bitwise equality across GPUs is never expected or claimed** (NFR-REP-05). Per-step losses diverge within a few hundred steps after a cross-machine resume. Curves and final metrics should be **statistically equivalent** (within seed noise).
- Results are reported as mean ± CI over seeds (E3), with lineage showing which steps ran where.

### 9.3 Identical-config cross-machine check (Day 4 evening; SHOULD)
Run `configs/uncond_cosine.yaml --set train.max_steps=3000` with seed 0 on **both** machines, plus seed 1 on DT. At steps 1k/2k/3k, compare the deterministic per-bucket validation loss:
- ΔM = |DT_s0 − LT_s0| (machine effect), ΔS = |DT_s0 − DT_s1| (seed effect).
- Pass if `ΔM ≤ max(1.5 · ΔS, 0.02 · mean)` on the mean loss and on ≥ 8 of 10 buckets. Output: `reports/dev/machine_effect.json`. Failure → §11d.

### 9.4 Deterministic debug mode (bug hunts only)
`launch.py --deterministic`: `torch.use_deterministic_algorithms(True, warn_only=True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8` (set before CUDA initialises), `cudnn.benchmark=False`, `cudnn.deterministic=True`. It is slower and still not cross-GPU bitwise. Use it to check same-machine reproducibility of a suspected bug.

### 9.5 Inference determinism (app and evaluation)
DDIM with η = 0 is deterministic given its inputs. Initial noise comes from per-sample CPU generators seeded by `derive_seed(...)` (NFR-REP-04). Evaluation forces `cudnn_benchmark=False`. On the same machine, results are identical (NFR-REP-03). Across machines, `scripts/gate_p7.py --compare-machines` reports the percentage of identical tiles for the 20-prompt AT set, generated on DT and on LT from the frozen weights. This is reported as an observation (expected ≥ 99 %), not as a gate.

## 10. Laptop thermal and power watchdog (H9)

### 10.1 Configuration (`base.yaml`, identical on both machines; no machine override)
```yaml
hardware:
  vram_cap_gib: 4.5             # H2, H3
  watchdog:
    enabled: true
    poll_s: 5
    max_temp_c: 87              # H9
    sustain_s: 60               # H9
    warn_temp_c: 83
    cooldown_resume_temp_c: 75  # used by scripts/queue.py before auto-resume
    cooldown_hold_s: 120
    require_ac_power: true      # H9 laptop rule; desktops report no battery → always satisfied
    it_drop_warn_frac: 0.40     # §1.4 slowdown detector
```

### 10.2 Algorithm (`utils/watchdog.py::ThermalWatchdog`, daemon thread)
```text
every poll_s:
    temp  = NVML nvmlDeviceGetTemperature(GPU)   (fallback: `nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits`)
    power = psutil.sensors_battery()               (None → AC assumed)
    push temp into a rolling window covering sustain_s
    if all temps in window >= max_temp_c        → request_stop("thermal")          # H9
    elif all temps in window >= warn_temp_c     → event watchdog.warn (once per 10 min)
    if require_ac_power and power and not power.power_plugged → request_stop("battery")
    if it_s < (1 - it_drop_warn_frac) * baseline_it_s for 5 min and no thermal reason → event perf.slowdown
if NVML and nvidia-smi both fail:
    desktop → event watchdog.unavailable, continue
    laptop  → refuse to start unless --no-watchdog (logged)
```
`request_stop` sets a flag that the training loop checks after every optimizer step. The loop then saves a checkpoint (§6.1), logs `stop.{reason}`, and exits with code **75**.

### 10.3 Actions
| Trigger | Trainer | `scripts/queue.py` |
|---|---|---|
| ≥ 87 °C for 60 s | checkpoint, exit 75 (`thermal`) | wait until temp ≤ 75 °C for 120 s → resume the same run (still RUNNING, same owner) |
| On battery (laptop) | checkpoint, exit 75 (`battery`) | wait for AC power → resume |
| STOP file / Ctrl+C | checkpoint, exit 75 (`manual`) | stop the queue (the operator asked) |
| Warn ≥ 83 °C | event only | — |

### 10.4 Trainer exit codes
| Code | Meaning | `queue.py` reaction |
|---|---|---|
| 0 | Finished (max steps or early stop) | `handoff.py finish`; next job |
| 1 | Unexpected error | stop the queue; print the traceback location; event `crash` |
| 3 | Config guard violation (§3.4) or config-hash mismatch | stop the queue |
| 4 | Ownership/lock violation (§4.1) | stop the queue |
| 5 | Resume verification or checkpoint integrity failure (§8) | stop the queue |
| 75 | Graceful stop (thermal / battery / manual), resumable | per §10.3 |

## 11. When the machines diverge

| Case | Symptom | Resolution |
|---|---|---|
| **(a) Split-brain**: the same run was trained on both machines | Two lineages continue past the same step; `HANDOFF.md` push conflict; heartbeat from a non-owner | **Never merge weights.** The branch recorded as owner in `HANDOFF.md` git history keeps the run ID. The other branch is renamed `<run_id>__fork_<TAG>_<step>`, kept for the record, and excluded from all curves and tables. Note it in `PROGRESS.md` |
| **(b) Code/config divergence** | Resume refuses (P3/P6); different config hashes for "the same" experiment | Sync git (`git pull` on both). Continue only under an identical hash. A deliberate change means a **new run** |
| **(c) Environment divergence** | `verify_env` V08/V17 fail; `compare_env.py` shows differences | Reinstall from the pinned files plus the identical torch version (doc 06 §5–6); rewrite both locks |
| **(d) Results divergence** | Same config gives different dev metrics on DT vs LT beyond seed noise; §9.3 fails | Ladder: 1) `compare_env.py`; 2) check `cudnn_benchmark`/`compile` overrides; 3) check `events.jsonl` for fallbacks (accumulation, sysmem slowdown); 4) re-run §9.3 with `--deterministic`; 5) if it persists, treat the machine as a nuisance factor: re-run the decisive experiments on the **reference machine** and report the machine per run |
| **(e) Data divergence** | `data_manifest_hash` differs between machines | Data artifacts are identified by **content hashes over canonical arrays** (`data.manifest.content_hash`), not file bytes, because parquet files embed writer metadata. Regenerate with `fetch_data.py` + `audit_dataset.py` from the pinned commit; the hashes must match. If they do not, there is a non-determinism bug in the pipeline: fix it before any training |
| **(f) Sync divergence** | Bundle sha256 mismatch, missing files | Wait (sync in progress); if it persists, the source re-runs `handoff.py release` (the bundle is rebuilt from the local checkpoint). Never train from an unverified bundle |

**Single-machine rule for final numbers.** The frozen run's training lineage may span both machines (it is recorded). But the D14 freeze evaluation, the D15 one-shot test evaluation and the report's main tables are computed on **one reference machine in one environment**. Default reference machine: **DT**. `FROZEN.json` records `eval_machine`, `torch`, `driver`, and the checkpoint sha256. The viva laptop (LT) runs the app on the same frozen weights; its cross-machine tile agreement is reported per §9.5. Before D16, `frozen/` is pulled to the laptop and its sha256 values are verified against `FROZEN.json`.

## 12. CV folds and parallel experiment distribution

1. **Folds.** 5 level-grouped folds over the 36 non-test levels (D5). This document uses the timeline's labels 1–5. If `split_v1.json` stores folds 0-indexed, label k ↔ index k−1.
2. **Default allocation** (doc 03 D15): DT runs folds 1–3, LT runs folds 4–5. **Re-balance from measured throughput** (`hw_report.json`, Day 4): `k_DT = round(5 · r_DT / (r_DT + r_LT))`, `k_LT = 5 − k_DT`; assign folds in label order (DT first).
3. **Run IDs:** `…_cv_fold{k}_s0_{TAG}`. Each fold is its own run with the same seed and the same fixed step budget (the frozen configuration's step count; doc 10 / doc 23 govern the fold protocol).
4. **Execution:** `python scripts/queue.py --jobs configs/queues/cv_DT.yaml` on DT and `cv_LT.yaml` on LT (sequential, exit-code aware, §10.4). Folds are **not** moved mid-flight. If a machine must leave, `handoff.py pause` and resume later on the same machine, or use the normal release/acquire protocol.
5. **Results:** each finished fold runs `handoff.py publish --what final` (final EMA + `eval/*.json` to `<sync_root>/runs/<id>/published/`). `scripts/aggregate_cv.py` reads the published results from both machines → `reports/cv/CV_RESULTS.md` (mean ± std per metric, with a machine column per fold).
6. **Ablations (D10, D15)** follow the same pattern: one run per ablation, the allocation in doc 04 §5, a shared seed, and results published and aggregated by `scripts/eval_dev.py --merge`.

## 13. Implementation checklist (Day 0 and Day 4)
- [ ] `utils/config.py`: merge order, guard (§3.4), config hash (§3.3) + `tests/test_config_guard.py`.
- [ ] `utils/hw.py`: `detect_machine`, `apply_vram_cap`, `hw_report` + `scripts/bench.py` (§1.5).
- [ ] `utils/io.py`: `atomic_torch_save`, `atomic_write_bytes`, `sha256_file`, `canonical_json` (§6.1).
- [ ] `train/checkpoint.py`: schema (§5.2), RNG capture/restore (§5.4), retention (§6.4), `verify_checkpoint` + `scripts/verify_ckpt.py`.
- [ ] `utils/handoff_state.py` + `scripts/handoff.py` (§4) + `tests/test_handoff_state.py` (state-machine transitions, push-conflict loop with a fake git).
- [ ] `utils/watchdog.py` (§10) + `tests/test_watchdog.py` (simulated temperature series → stop after 60 s ≥ 87 °C; battery stop).
- [ ] `scripts/launch.py --resume` with P1–P6 and V-R1–V-R5 (§8).
- [ ] `scripts/queue.py` with the exit-code table (§10.4).
- [ ] Day-4 resume drill (§8.5) and machine-effect check (§9.3) recorded in `reports/dev/`.
