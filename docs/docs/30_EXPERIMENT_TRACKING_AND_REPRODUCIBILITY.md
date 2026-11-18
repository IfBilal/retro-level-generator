# 30 — Experiment Tracking & Reproducibility

> **Owner track:** B (both). **Used from:** Day 0 onward. **Ledger rows:** V8 (tracking stack), D1/D4 (data manifest, split lock), H2–H5 (machine contract), M11 (EMA for eval), M13 (sampler settings), E2–E4 (prompt sets, statistics, composite).
> **This doc owns:** reproducibility levels, the seeding policy for training, determinism policy, the content of `meta.json`/`lineage.json`/checkpoints, logging formats, manifests and the chain of custody, the runs index, number provenance, retention *policy*, the `PROGRESS.md` and experiment-log templates, and the end-to-end reproduction chain.
> **Defers to:** `05_REPO_STRUCTURE.md` §10 (file **locations** in a run directory; doc 05 wins on paths), `07_DUAL_GPU_HANDOFF_PROTOCOL.md` (§2 run-ID implementation `make_run_id`, §3.3 `config_hash` implementation, §4 `HANDOFF.md`, §5 lineage mechanics, §6 retention mechanics), `22_EVALUATION_PROTOCOL_AND_METRICS.md` (§2.3 evaluation seeds, §6 evaluation output paths, §6.5 `eval_plan.json` schema and test guards), `26_ASSETS_PIPELINE.md` §7 (`ASSET_MANIFEST.json` schema), `17_HYPERPARAMETERS_AND_CONFIGS.md` (YAML contents). If this doc and one of those disagree on *their* topic, theirs wins; log a `C-xxx` doc conflict in `PROGRESS.md` (CLAUDE.md §1).

---

## 1. What "reproducible" means in this project

There are three levels. §8 states which level each artifact must reach.

| Level | Definition | Required for |
|---|---|---|
| **L1 — Bitwise** | Re-running the same command on the **same machine** with the same versions gives byte-identical outputs (sha256 equal). | All data artifacts (index, splits, prompt sets, validation noise); all metrics recomputed from saved `samples.npz`; evaluation of a fixed checkpoint on one machine; `check_determinism.py` in repro mode (§7). |
| **L2 — Numerical** | Identical *inputs* on a different GPU give outputs equal within floating-point tolerance. | Training inputs (§6) and initial weights. Evaluation samples only if the CPU-noise upgrade in §6.4 is adopted. |
| **L3 — Statistical** | Re-training from scratch, or evaluating on the other machine, gives metrics inside the reported CI / CV spread. | Every model-quality claim in the report (stated with CIs, E3, and 5-fold CV spread, D5). |

Why not demand L1 for training across machines? The RTX 3060 Ti (sm_86, 38 SMs) and RTX 4050 Laptop (sm_89, 20 SMs) select different cuBLAS/cuDNN kernels and reduction orders, and bf16 autocast (H5) is not associative. Byte-identical trajectories across them are not achievable at reasonable cost. Instead, **every input to training is identical on both machines**: data order, augmentation draws, CFG-dropout masks, diffusion t and ε, initial weights (§6). Any divergence is then purely numerical, and it is covered by seed variance (L3).

---

## 2. Run identity

### 2.1 Run ID (normative implementation: `utils/runid.py::make_run_id`, doc 07 §2)
```
<YYYYMMDD-HHMM>_<config>_<machine>_s<seed>[_<suffix>]
```
| Field | Rule | Example |
|---|---|---|
| `YYYYMMDD-HHMM` | **UTC** start time of the first segment (`make_run_id(..., now_utc)`). The local time (Asia/Karachi, UTC+05:00) is stored in `meta.json → created_local` for humans. | `20261122-1815` (= 23:15 PKT) |
| `<config>` | stem of the experiment overlay (`configs/<config>.yaml`, or `ablations/x03_pooled` → `x03_pooled`) | `cond_main`, `uncond_cosine`, `x03_pooled`, `final` |
| `<machine>` | machine profile from `utils/hw.py::detect_machine()` | `desktop_3060ti`, `laptop_4050`, `cpu` |
| `s<seed>` | training-run seed (§6.2) | `s1` |
| `_<suffix>` | `fold<k>` for CV folds; `b`, `c` … if the directory already exists | `_fold3`, `_b` |

Examples: `20261122-1815_cond_main_desktop_3060ti_s1`, `20261130-1640_final_laptop_4050_s1_fold4`.

**Rules**
1. **Never parse a run ID.** Underscores occur inside config and machine names. Read `meta.json`.
2. **A resumed run keeps its ID and directory**, even on the other machine. The machine token records where it *started*; every later segment goes into `lineage.json` (§3.3).
3. Evaluations don't create runs. In-training dev evaluations write inside the run (`eval/`, doc 05 §10). Full evaluations write where doc 22 §6 says (`reports/dev/<run_id>/`, `reports/final/test_runs/<UTC>_<gitsha>/`).

### 2.2 Config hash (implementation: `utils/config.py::config_hash`, doc 07 §3.3)
`config_hash` = sha256 of the canonical JSON (`sort_keys=True, separators=(",", ":")`, after interpolation) of the resolved config **with these sections removed**:
- `machine`, `paths`, `overrides` (machine files contain only these, doc 05 §5);
- allow-listed performance knobs (`num_workers`, `cudnn_benchmark`, `compile`, `log_every`; CLAUDE.md rule 5);
- `run.*` (seed, suffix, purpose, notes), `logging.*`, `tracking.*`.

`config_hash.txt` holds the full hex digest. Reports and indexes show its first 12 characters.

Consequences:
- Replicates (different seeds) and the same config on the other machine **share a `config_hash`**. The runs index groups them by it.
- `launch.py` warns if a new run duplicates an existing finished run's `config_hash` **and** seed.
- Resume refuses if the checkpoint's `config_hash` differs from the resolved config, unless `--allow-config-change "<reason>"` is given. The reason is logged as an `events.jsonl` event (`config_change_allowed`).

> If doc 07 §3.3 excludes a different key set, doc 07 wins. The non-negotiable property is: **seed and machine knobs must not change the hash; anything that changes the model or the data must.**

### 2.3 Purpose tags (`meta.json → purpose`)
`smoke` · `drill` (resume/handoff drills) · `phase3` · `main` · `ablation:X<n>` · `cv:fold<k>` · `final` · `debug`. The purpose drives the git policy (§2.4) and retention (§13).

### 2.4 Git-state policy
- `launch.py` records `git_sha` and `git_dirty` (true if `git status --porcelain` is non-empty, ignoring `runs/`, `PROGRESS.md`, `HANDOFF.md`).
- For `main`, `ablation:*`, `cv:*`, `final`: **a dirty tree is refused.** Commit first.
- For `smoke`, `drill`, `debug`: allowed. The diff is saved to `runs/<id>/git.diff` and the run is marked dirty. **Dirty runs can never be cited** (`check_provenance.py`, §12).

---

## 3. Run directory

### 3.1 Layout (locations per doc 05 §10; additions from this doc marked ✚)
```text
runs/<run_id>/
├── config.yaml              resolved experiment config (machine sections removed)
├── config_hash.txt          full sha256 (§2.2)
├── meta.json                identity + status (§3.2)
├── lineage.json             machine segments (§3.3)
├── env_lock.txt          ✚  copy of env/lock_<machine>.txt at launch (pip freeze)
├── env.json              ✚  python, torch, CUDA runtime, cuDNN, driver, GPU name, VRAM, OS, hostname
├── git.diff              ✚  only if git_dirty
├── hw_report.json           peak VRAM, it/s, thermal (§3.5)
├── metrics.csv              training scalars: step, loss, loss_t0..9, lr, grad_norm, it_s, vram_gib (doc 05)
├── metrics.jsonl         ✚  superset log of every scalar incl. val/dev/sys with CI fields (§4.1)
├── events.jsonl             lifecycle, fallbacks, warnings (§4.2)
├── tb/                      TensorBoard
├── checkpoints/
│   ├── ckpt_step0002000.pt  ckpt_step0002000.json      full state + sidecar {step, sha256, bytes, dev_S, saved_utc, machine}
│   ├── ema_step0002000.safetensors                     inference EMA weights (+ adapter)
│   ├── latest.json  best.json                          pointers {file, step, sha256}
├── eval/dev_step0002000.json …                         in-training dev suite outputs (E1, E4; doc 22 §6.1)
├── samples/                 PNG grids at eval steps (+ ✚ samples_step0002000.npz, uint8 grids)
├── STOP                     control file: create to request a graceful stop (doc 07)
└── HANDOFF_READY.json       written on graceful stop for handoff (doc 07 §4)
```
> Doc 22 §6.1 also names a per-run `dev_eval.jsonl` (one summary line per dev eval) and a `best_ema.pt`. Write `dev_eval.jsonl` as an append-only summary next to `eval/` (it doesn't conflict). Resolve "best" through `checkpoints/best.json` → its `ema_step*.safetensors`. If `best_ema.pt` is required by doc 22's scripts, it is a copy of that EMA file. This naming overlap is logged as a doc conflict to reconcile (doc 05 vs doc 22).

**Write rules**
- Checkpoints, pointers and sidecars are written **atomically**: write `*.tmp` → `flush` + `os.fsync` → `os.replace`. The newest two good checkpoints are never deleted (CLAUDE.md rule 9).
- `metrics.*` and `events.jsonl` are append-only and flushed after every write, so a crash loses at most one line.
- No symlinks anywhere (Windows, sync tools). "Latest" and "best" are JSON pointers.
- ✚ **Sample grids are kept as uint8 `.npz`** (256 × 16 × 16 B ≈ 64 KB per eval step). Every metric must be recomputable from saved grids without regenerating.

### 3.2 `meta.json`
```json
{
  "schema": "t2m.run_meta/v1",
  "run_id": "20261122-1815_cond_main_desktop_3060ti_s1",
  "purpose": "main",
  "config": "cond_main",
  "config_hash": "a3f9c2e1b7d4…",
  "seeds": {"run": 1, "project": 20261117},
  "git_sha": "4e1d0c9…", "git_dirty": false,
  "parent_run_id": null, "init_from": null,
  "data_manifest_sha256": "…", "split_lock_sha256": "…", "prompts_lock_sha256": "…",
  "env_lock_ref": "env/lock_desktop_3060ti.txt",
  "text_encoder": {"name": "sentence-transformers/all-MiniLM-L6-v2", "revision": "<HF commit sha>"},
  "created_utc": "2026-11-22T18:15:04Z", "created_local": "2026-11-22T23:15:04+05:00",
  "status": "running",
  "best": {"step": null, "dev_S": null, "ema_file": null},
  "notes": ""
}
```
`status` ∈ `running | finished | early_stopped | crashed | killed | handed_off | superseded`. It is updated atomically at every lifecycle event.

### 3.3 `lineage.json` (mechanics: doc 07 §5)
```json
{"segments": [
  {"seg": 0, "machine": "desktop_3060ti", "host": "DESKTOP-XYZ", "torch": "2.15.0+cu130", "driver": "…",
   "gpu": "NVIDIA GeForce RTX 3060 Ti", "start_step": 0, "end_step": 24000,
   "started_utc": "…", "ended_utc": "…", "end_reason": "handoff"},
  {"seg": 1, "machine": "laptop_4050", "start_step": 24000, "end_step": null, "started_utc": "…", "end_reason": null}
]}
```
`end_reason` ∈ `finished | early_stop | handoff | crash | watchdog_thermal | user_stop | nan_abort`. Segments must be contiguous in steps. Overlapping segments mean two machines ran the same run, which is risk R15 (doc 33).

### 3.4 Checkpoint contents (`ckpt_step*.pt`)
Doc 16 owns *when* checkpoints are written and doc 07 §5–6 the save/verify mechanics. This table is the **required content**.

| Key | Content |
|---|---|
| `model` | U-Net `state_dict` (training weights) |
| `ema` | EMA `state_dict` + `ema_step` (M11 warm-up needs the step) |
| `adapter` | text-adapter `state_dict` (conditional runs). MiniLM is frozen and never saved; it is re-loaded by name + revision. |
| `optimizer`, `lr_scheduler` | AdamW and LR-schedule `state_dict`s |
| `step` | optimizer step count (the budget unit, M12) |
| `sampler_state` | `ResumableBatchSampler.state_dict()`: generator state, epoch, position in the permutation |
| `rng` | the CPU generators of the `data`, `aug`, `cfg`, `diffusion` streams (§6.3), `torch_cpu`, `torch_cuda` (all devices) |
| `early_stop` | best S, best step, evals without improvement (patience 5, M12) |
| `meta` | `config_hash`, `git_sha`, `run_id`, `torch_version`, `machine`, `saved_utc`, `text_encoder` |

- **`torch.load(weights_only=True)` compatibility** (doc 05 §4.6): store every RNG state as a tensor or plain Python primitives (ints, floats, lists, dicts, tuples). **No numpy objects**: use the `bit_generator.state` dict of a `np.random.Generator`, never the legacy `np.random.get_state()` tuple (it contains an ndarray).
- **Load-time checks** (resume and eval): `config_hash` equal (§2.2); **torch minor version equal** (V2); text-encoder name + revision equal; sha256 equals the sidecar. Any mismatch → refuse with a message naming the field.
- Sizes (H8: 18,312,013 + 66,048 params): full checkpoint ≈ 4 fp32 copies (weights, EMA, 2 AdamW moments) ≈ **0.30 GB**; bf16 EMA safetensors ≈ **37 MB**.

### 3.5 `hw_report.json` (written by `scripts/bench.py` and at the end of every segment)
```json
{"machine": "laptop_4050", "gpu": "NVIDIA GeForce RTX 4050 Laptop GPU", "total_vram_gib": 6.0,
 "cap_gib": 4.5, "cap_fraction": 0.75, "peak_allocated_gib": null, "peak_reserved_gib": null,
 "batch": 64, "accum_steps": 1, "accum_fallback_used": false,
 "it_per_s_median": null, "it_per_s_p10": null, "eval_seconds_per_dev_suite": null,
 "gpu_temp_max_c": null, "throttle_events": 0, "det_mode": "fast", "det_slowdown_vs_fast": null,
 "measured_utc": "…", "steps_measured": 200}
```
Day 4 requires this file from **both** machines. `peak_allocated_gib` must be ≤ 4.5 (H2).

---

## 4. Logging formats

### 4.1 `metrics.jsonl` (one JSON object per line)
```json
{"ts":"2026-11-22T20:02:03Z","step":12000,"split":"val","name":"loss","value":0.0412,"t_bucket":3}
{"ts":"…","step":12000,"split":"dev","name":"cscore","value":0.71,"ci_lo":0.68,"ci_hi":0.74,"w":5.0,"set":"P-dev","n":351}
```
Required: `ts` (UTC), `step`, `split` ∈ {`train`, `val`, `dev`, `sys`}, `name`, `value`. Optional: `t_bucket` (0–9), `w`, `set`, `n`, `ci_lo`, `ci_hi`.

**Canonical names** (also the TensorBoard tags `<split>/<name>[_t<k>]`):
- `train`: `loss`, `loss_t0`…`loss_t9`, `lr`, `grad_norm`, `ema_decay`, `skipped_nonfinite`
- `val` (deterministic noise, §6.5): `loss`, `loss_t0`…`loss_t9`
- `dev` (doc 22 metric names): `cscore`, `em`, `broken_rate`, `floor_valid`, `playable`, `novelty_med`, `copy_rate`, `diversity`, `js`, `S`
- `sys`: `vram_peak_gib`, `gpu_temp_c`, `it_s`, `eval_wall_s`

The 10 t-buckets are equal-width: bucket k covers t ∈ [100k, 100k + 99]. They are the basis of doc 19's per-bucket plateau rules (audit #26).

### 4.2 `events.jsonl` (closed vocabulary; extend only here)
```json
{"ts":"…","step":24000,"machine":"desktop_3060ti","type":"handoff_out","detail":{"ckpt":"checkpoints/ckpt_step0024000.pt","sha256":"…"}}
```
`start`, `resume`, `handoff_out`, `handoff_in`, `ckpt_saved`, `ckpt_verified`, `eval_start`, `eval_done`, `new_best`, `early_stop`, `finished`, `oom_fallback` (accumulation enabled; detail = micro-batch), `nonfinite_skip`, `nan_abort`, `watchdog_thermal`, `config_change_allowed`, `fallback` (placeholder sprites, CPU mode, …; CLAUDE.md rule 12), `gate_result`, `determinism_warning`, `warning`, `crash`.

---

## 5. Ownership map (who defines what)

| Concern | Owner | This doc adds |
|---|---|---|
| Run-ID string, `make_run_id` | doc 07 §2 | format rules and parsing ban (§2.1) |
| `config_hash` code | doc 07 §3.3 | required invariants (§2.2) |
| Run-directory paths | doc 05 §10 | additions ✚ (§3.1) |
| Checkpoint save/verify mechanics | doc 07 §5–6, doc 16 | required content (§3.4) |
| Evaluation seeds, CRN, output paths, `eval_plan.json` schema, test guards | doc 22 §2.3, §6, §8 | its place in the chain of custody (§9.5) |
| Asset manifest schema | doc 26 §7 | verification in `verify_hashes.py` (§9.4) |
| `HANDOFF.md` template | doc 07 §4 | — |
| `PROGRESS.md` registers + session entries | **this doc §14** | — |

---

## 6. Seeding policy

### 6.1 Principle: named, independent RNG streams
Every source of randomness has its **own** generator with a stable seed. Changing one stream (an augmentation ablation, say) never shifts the others (CFG mask, noise draws, data order). That is what makes paired comparisons between ablations meaningful (E3; doc 22 §7.2).

```python
# src/text2mario/utils/seed.py (doc 05 §4.9)
import hashlib
def derive_seed(*parts) -> int:
    """Stable 63-bit seed across processes, OSes and machines. NEVER Python's hash() (salted per process)."""
    key = "|".join(map(str, parts)).encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "little") & ((1 << 63) - 1)
```
Library code takes generator arguments and never seeds globally (doc 05 §3). Only entry points call `seed_everything`.

### 6.2 Roots
| Root | Value | Drives |
|---|---|---|
| **Project root** `PROJECT_SEED` | **20261117** (ledger D3) | Everything identical for all runs: split selection, CV folds, P-rand-dev (doc 22 §2.1 uses 20261117; P-rand-test uses 20261118), validation noise (§6.5), the paraphrase-lexicon split, stride-8 offsets. |
| **Run root** `run.seed` | `1` for the primary run of a config; `2`, `3` for replicates | Init, data order, augmentation, CFG dropout, training t/ε, dropout masks. |
| **Evaluation seeds** | `sample_seed(set, prompt_index, replicate) = SET_BASE[set] + 1000·prompt_index + replicate` (**doc 22 §2.3**) | Initial x_T and all sampler noise. Independent of the run seed, so every checkpoint is judged on the same noise (common random numbers). |

### 6.3 Training streams
| Stream | Generator | Seed | Device | In checkpoint | Notes |
|---|---|---|---|---|---|
| `init` | `torch.manual_seed` before building the model | `derive_seed(run.seed, "init")` | **CPU** (build on CPU, then `.to(device)`) | n/a | Initial weights bit-identical on both machines. |
| `data` | `torch.Generator("cpu")` in `ResumableBatchSampler` | `derive_seed(run.seed, "data")` | CPU | yes | `randperm` per epoch; position saved → exact mid-epoch resume. |
| `aug` | `torch.Generator("cpu")` passed to `augment_caption` (doc 05 §4.1) | `derive_seed(run.seed, "aug")` | CPU | yes | Phrase shuffle, phrase dropout, paraphrase (D8 a–c). |
| `cfg` | `torch.Generator("cpu")` | `derive_seed(run.seed, "cfg")` | CPU | yes | Null-caption mask (D8 d, p = 0.10). A separate stream, so augmentation ablations don't move the CFG pattern. If doc 11 implements null dropout inside `augment_caption`, it must use this generator for that draw. |
| `diffusion` | `torch.Generator("cpu")` | `derive_seed(run.seed, "diffusion")` | **CPU**, non-blocking copy to GPU | yes | t ~ U{0..999} and ε ~ N(0, I): 64 × 13 × 16 × 16 = 212,992 floats/step, ~1–2 ms on CPU (a few % of a step). Flag `repro.cpu_noise: true` (default). `false` draws on the GPU (faster, not cross-machine identical) and is allowed only for `smoke`/`debug`. |
| `dropout` | global CUDA RNG | `torch.cuda.manual_seed_all(derive_seed(run.seed, "dropout"))` | GPU | yes (`torch_cuda`) | Not cross-machine identical. Accepted. |

### 6.4 Evaluation noise (doc 22 §2.3) and the cross-machine upgrade
Doc 22 draws each sample's x_T with `torch.Generator(device).manual_seed(sample_seed)` in fp32 **on the generating device**, one generator per sample, all stochastic sampler steps in a fixed order, batching never changing the noise. Consequences:
- **Same machine:** evaluation is L1 (bitwise repeatable). This holds for every comparison doc 22 makes, since it compares within one machine/run.
- **Across machines:** CUDA random kernels are not guaranteed to produce the same stream on GPUs with different SM counts, so x_T may differ between the 3060 Ti and the 4050. Cross-machine agreement is therefore checked at the **metric level** (L3, §8).
- **Optional upgrade (recommended; needs a doc 22 change + a ledger row):** draw x_T and all sampler noise from `torch.Generator("cpu")` seeded with the same `sample_seed`, then move to the GPU. x_T becomes identical on both machines, and the Day 14 check can be the stronger tile-level test (≥ 99 % tile agreement). Cost: negligible (13 × 16 × W floats per sample).

### 6.5 Deterministic validation loss (fixed noise)
`scripts/make_val_noise.py` creates `data/processed/val_noise_v1.pt` **once**, on CPU, from `derive_seed(PROJECT_SEED, "val_noise", "v1")`: for every dev window (stride 1), a fixed `t` (stratified, equal counts per t-bucket) and a fixed fp32 ε. Its sha256 goes into `data/MANIFEST.json → derived`. Every run computes `val/loss_t*` on exactly these (x₀, t, ε) triples (`train/validate.py::deterministic_val_loss`). Validation loss is then comparable **across runs, seeds and machines**, which is what doc 19's overfitting overlays need.

### 6.6 Evaluation discipline (applies everywhere)
EMA weights (M11), DDIM 50, η = 0, x̂₀ clipped to [−1, 1], CFG `ε_u + w(ε_c − ε_u)` (M13), `torch.no_grad()`, `model.eval()`, `cudnn.deterministic=True`, `benchmark=False`. Outputs don't depend on eval batch size: per-sample generators, and GroupNorm is per-sample (no BatchNorm anywhere).

---

## 7. Determinism flags and their cost

| Mode | Settings | Use for | Cost |
|---|---|---|---|
| **fast** (training default) | `cudnn.benchmark` from the machine allow-list; no deterministic algorithms | all training runs, including `final` | baseline |
| **repro** | `cudnn.benchmark=False`, `cudnn.deterministic=True`, `torch.use_deterministic_algorithms(True, warn_only=True)`, env `CUBLAS_WORKSPACE_CONFIG=:4096:8`; force the SDPA math backend if a non-deterministic attention-backward warning appears | `scripts/check_determinism.py`; bisecting a divergence | Measured on Day 4 → `hw_report.json → det_slowdown_vs_fast`. Plan for 10–40 % slower at this size. |
| **eval** | `cudnn.deterministic=True`, `benchmark=False`, `no_grad` | every evaluation | negligible |

- `warn_only=True`: some CUDA backward kernels have no deterministic implementation (the list depends on the torch version: certain interpolation, scatter/index and fused-attention backward paths). PyTorch then warns instead of raising. Every warning becomes a `determinism_warning` event. Replace the offending op (e.g. force the SDPA math backend) and re-check.
- **`check_determinism.py`** (slow test, Day 4): same machine, repro mode, 200 steps twice from the same seed → model `state_dict`s bitwise equal and loss sequences equal to the last bit. It repeats in fast mode and **reports** (doesn't fail on) the max absolute weight difference, documenting how non-deterministic fast mode is.
- Training in fast mode is a deliberate trade. Report claims are L3 (CIs, CV). The final checkpoint is pinned by sha256 in `FROZEN.json` (§9.6), so everything evaluated from it is L1 on one machine.

---

## 8. Cross-GPU reproducibility matrix

| Artifact / process | Desktop vs laptop | Mechanism | Verified by |
|---|---|---|---|
| `index.parquet`, `scenes_u8.npy`, `split_v1.json`, prompt sets, `val_noise_v1.pt`, `REFERENCES.json` | **L1** | Pure CPU code, project-root seeds | sha256 vs `data/MANIFEST.json`, `LOCK`, `PROMPTS_LOCK` on every `verify_env --quick` |
| Initial weights | **L1** | Built on CPU under `init` | `tests/test_seeding.py::test_init_bitwise` |
| Training inputs (order, captions after aug, CFG mask, t, ε) | **L1** | CPU generators (§6.3) | `tests/test_seeding.py::test_training_inputs_bitwise`: hash the first 50 batches' inputs on each machine and diff the hash lists |
| Training trajectory | **not identical** | Kernels, reduction order, bf16 | Loss curves overlap within seed noise (Day 4 drill) |
| Resumed run, other machine | **Continues correctly** (full state restored) but diverges numerically from an uninterrupted run | §3.4 | Day 4 drill: no spike > 2× the running mean; LR/EMA/step continuous |
| Eval samples from the same checkpoint | **L1** same machine; across machines L3 (doc 22 device noise) or **L2 with the §6.4 upgrade** | Per-sample generators | `scripts/check_cross_machine_eval.py` (Day 14), below |
| Metrics recomputed from the same saved grids | **L1** | Deterministic CPU metric code on uint8 grids | Recompute on the other machine → `metrics.json` sha256 equal |
| Retraining from scratch | **L3** | Seeds + CV | Day 15 CV spread; replicate seeds (doc 23) |

**`check_cross_machine_eval.py` (Day 14, before the freeze).** Evaluate the candidate final checkpoint on P-dev (3 replicates) on **both** machines. Pass criteria:
- (a) Always: every headline metric (C-score, playable, broken, copy rate, novelty_med) differs by less than the half-width of its 95 % CI (doc 22 §7.6: ≈ ±0.03 C-score, ≈ ±3 pp proportions).
- (b) Only with the §6.4 upgrade: ≥ 99 % tile agreement, \|ΔC-score\| ≤ 0.01.

Result → `reports/repro/cross_machine.json`. If (a) fails, the usual cause is a batch-dependent op or an unseeded draw. Fix it; never waive.

**Policy:** a number may come from either machine. `RESULTS.md` records the machine of every evaluation (§12).

---

## 9. Manifests: the chain of custody

All manifests are JSON, git-tracked, and checked by `scripts/verify_hashes.py`. It is cross-platform (no `sha256sum`) and runs inside `verify_env.py --quick`.

**Chain:** `data/MANIFEST.json` → `data/splits/LOCK` + `data/prompts/PROMPTS_LOCK` → `reports/references/REFERENCES.json` → runs (`meta.json` records the three hashes) → `FROZEN.json` (Day 14) → `reports/final/eval_plan.json` (references FROZEN; doc 22 §6.5) → `reports/final/test_runs/<UTC>_<gitsha>/` (Day 16) → `RESULTS.md` + `number_provenance.json`.

### 9.1 `data/MANIFEST.json` (Day 1)
```json
{
  "schema": "t2m.data_manifest/v1",
  "source": {"repo": "https://github.com/schrum2/MarioDiffusion", "commit": "<40-hex>", "licence": "MIT", "fetched_utc": "…"},
  "files": [
    {"path": "data/raw/MarioDiffusion/datasets/Mar1and2_LevelsAndCaptions-regular.json", "sha256": "…", "bytes": 42028678},
    {"path": "data/raw/MarioDiffusion/datasets/smb.json", "sha256": "…", "bytes": null}
  ],
  "derived": [
    {"path": "data/processed/index.parquet", "content_hash": "…", "produced_by": "scripts/audit_dataset.py@<git_sha>"},
    {"path": "data/processed/scenes_u8.npy", "content_hash": "…", "produced_by": "scripts/audit_dataset.py@<git_sha>"},
    {"path": "data/processed/val_noise_v1.pt", "content_hash": "…", "produced_by": "scripts/make_val_noise.py@<git_sha>"}
  ],
  "facts_verified": {"F3": 7687, "F4": 7620, "F7": 42, "F9": [3606, 4081], "F11_total_columns": 8317}
}
```
`content_hash` uses `data/manifest.py::content_hash` (canonical array bytes, doc 05 §4.1), so the hash doesn't depend on parquet writer metadata. `audit_dataset.py` refuses to write derived files if any F-fact differs from the ledger.

### 9.2 Split lock (Day 2)
- `data/splits/split_v1.json`: train/dev/test level IDs, the 5 CV folds, seed, algorithm version, `constraints_report` (doc 10).
- `data/splits/LOCK`: one line, `<sha256>  split_v1.json`. `tests/test_splits.py` recomputes it on every `pytest`. Changing the split after Day 2 is a ledger-level change, and it invalidates every run whose `meta.json → split_lock_sha256` differs.

### 9.3 Prompt lock (Day 2; doc 22 §2.2)
`data/prompts/PROMPTS_LOCK` = sha256 over the test-side prompt files. Model-facing code never opens test-side prompt files before Day 16 (doc 22).

### 9.4 `assets/ASSET_MANIFEST.json` (Day 0; schema owned by doc 26 §7)
It records every fetched or built asset: source URL, licence SPDX + licence-file path (e.g. `app/static/vendor/LICENSE.phaser.md`), sha256, bytes, and whether a fallback was used. `verify_hashes.py` checks the files on disk against it. A fallback (placeholder sprites, A2) must also appear in `events.jsonl` and `PROGRESS.md` (CLAUDE.md rule 12).

### 9.5 `reports/final/eval_plan.json` — pre-registration (schema and guards: doc 22 §6.5)
- Written **after** `FROZEN.json` and committed **before** the test run. It contains `frozen_json_sha256` (doc 22), so the dependency points from the plan to the freeze. **`FROZEN.json` must not contain the plan's hash**; that would be circular.
- Its job in the chain: fix the prompt sets, sampling settings, metrics, comparators (B0–B3, optional B4†), hypotheses H1–H4, bootstrap settings and anomaly policy **before** anyone sees a test number (D4; audit #29).
- `eval_test.py` checks (doc 22 §6.5b) that the plan's commit precedes the run, its hashes match the disk, and the w/sampling settings come from the plan, not the CLI.

### 9.6 `FROZEN.json` (Day 14, tag `v0.9-frozen`)
```json
{
  "schema": "t2m.frozen/v1",
  "frozen_utc": "…", "git_sha": "…", "git_tag": "v0.9-frozen",
  "config": {"path": "configs/final.yaml", "config_hash": "…"},
  "checkpoint": {"run_id": "…", "full": "runs/<run_id>/checkpoints/ckpt_step00NNNNN.pt", "full_sha256": "…",
                 "ema": "runs/<run_id>/checkpoints/ema_step00NNNNN.safetensors", "ema_sha256": "…", "step": null},
  "text_encoder": {"name": "sentence-transformers/all-MiniLM-L6-v2", "revision": "…"},
  "sampler": {"type": "ddim", "steps": 50, "eta": 0.0, "clip_x0": true, "cfg_convention": "eps_u + w*(eps_c - eps_u)"},
  "default_w": null,
  "full_level": {"stride": 8, "resample_r": 2, "resample_frac": 0.6, "framing": "G3",
                 "max_attempts_window": 5, "max_attempts_total": 20},
  "hashes": {"split_lock": "…", "prompts_lock": "…", "data_manifest": "…", "references": "…",
             "physics_json": "…", "normalizer_lexicon": "…", "detector_src": "…", "asset_manifest": "…"}
}
```
After the freeze, any change to a hashed file makes `eval_test.py` and the app's report mode refuse to run until `FROZEN.json` is re-issued, with a ledger change-log row explaining why.

### 9.7 Other tracked manifests
- `env/lock_desktop_3060ti.txt`, `env/lock_laptop_4050.txt` (pip freeze), `env/capabilities.json` (Java available, CUDA runtime, driver, free disk): Day 0.
- `reports/references/REFERENCES.json` (doc 22 §4): Day 2, regenerated whenever `LOCK` changes.
- `reports/gates/P<N>/RESULT.json` = `{"gate", "pass", "metrics", "thresholds", "git_sha", "run_ids", "at_utc"}` (CLAUDE.md §3).

---

## 10. Runs index

### 10.1 Files
`runs/` is git-ignored and only partly synced (doc 07), so each machine indexes **its own** runs. That avoids git merge conflicts.
- `reports/runs_index_desktop_3060ti.csv`, `reports/runs_index_laptop_4050.csv`: written by `scripts/index_runs.py`, git-tracked.
- `reports/runs_index.csv`: merged view (`index_runs.py --merge`). A run spanning machines appears once; `machines` lists both (from `lineage.json`).

### 10.2 Columns
`run_id, purpose, config, config_hash12, seed, parent_run_id, machines, git_sha, git_dirty, status, steps_done, best_step, dev_S, dev_cscore, dev_em, dev_broken, dev_playable, dev_novelty_med, dev_copy_rate, dev_diversity, dev_js, wall_hours, peak_vram_gib, accum_fallback, experiment_id (X<n> | —), notes`

### 10.3 Rules
- `index_runs.py` runs at every session close (it is in the §14 checklist).
- Any run cited in the report, a gate or the ledger must be in the index with `git_dirty = false`.
- `runs/cv/fold<k>.json` holds pointers (`run_id`, machine) to the five CV runs, so `make_figures.py` can find them from either machine.

---

## 11. TensorBoard and optional W&B

**TensorBoard (default, V8)**
- `tensorboard --logdir runs --port 6006`. To compare named runs: `--logdir_spec a:runs/<id1>,b:runs/<id2>`.
- Scalars: the §4.1 tags. Overlaying `train/loss_t*` with `val/loss_t*` is the primary overfitting view (doc 19).
- Images: `samples/grid` at every dev eval (8×8 sheet); `samples/contrast` (PDF contrast prompts, conditional runs).
- Text: resolved config at step 0; the normalised dev prompt list.
- HParams: at run end, `add_hparams({lr, ema, schedule, w, aux_ce, pooled_text, …}, {best dev metrics})`. This feeds the ablation comparison.
- Both machines in one view: point TensorBoard at doc 07's sync folder, which carries `tb/`.

**Weights & Biases (optional, off by default)**
- `tracking.wandb: false` in `configs/base.yaml`. If enabled: `WANDB_MODE=offline` by default; `wandb sync` later. Same metric names as §4.1; project `text2mario`, group = `config_hash12`, job type = `purpose`.
- Never the source of truth. Figure scripts read only local files.

---

## 12. Number provenance (every reported number traces to a file)

- **Tag format** (inline in `RESULTS.md`; footnote or appendix in the report; speaker notes on slides):
  `[src=<file path> | key=<json key> | run=<run_id> | set=<prompt set> | reps=<n> | machine=<machine> | git=<sha7>]`
- **`reports/final/number_provenance.json`** (written by `make_figures.py` / doc 22's `make_results.py`): maps each figure/table cell → `{file, key, run_id, machine, git_sha}`.
- **`scripts/check_provenance.py`** fails if:
  - any number in `RESULTS.md` lacks a tag, or a tag points to a missing file/key;
  - a tagged run is `git_dirty`;
  - a test number doesn't come from `reports/final/test_runs/`;
  - a test-run folder isn't recorded in `reports/final/TEST_LEDGER.md` (doc 22).

  It is on the Day 17 checklist.

---

## 13. Retention and disk policy (mechanics: doc 07 §6, `scripts/prune_runs.py`)

| Artifact | main / final | ablation / cv | smoke / drill / debug |
|---|---|---|---|
| `ckpt_step*.pt` (+ sidecar) | every 10k-step checkpoint + newest 2 | newest 2 | newest 1 |
| `ema_step*.safetensors` | the one `best.json` points to + newest 2 | the best one | none after the run ends |
| `samples_step*.npz`, `eval/*.json`, `dev_eval.jsonl` | forever | forever | 7 days |
| `samples/*.png` | forever | forever | 7 days |
| logs, `metrics.*`, `events.jsonl`, `meta.json`, `lineage.json` | forever | forever | forever (tiny) |

- `prune_runs.py --dry-run` lists deletions first. It never deletes the newest two good checkpoints of a run (CLAUDE.md rule 9), nor anything referenced by `FROZEN.json`, a gate `RESULT.json`, or `runs/cv/fold*.json`.
- `launch.py` refuses to start if the run drive has < 10 GB free. Day 0 records free disk in `env/capabilities.json`; ≥ 30 GB free on both machines is recommended.
- Budget: a main run ≈ 6 full checkpoints × 0.30 GB ≈ 1.8 GB; an ablation ≈ 0.6 GB; project worst case ≈ 15–25 GB.
- **Day 20 archive:** `release/text2mario_v1.0_artifacts.zip` = the frozen run directory (full checkpoint + EMA + `eval/` + samples), the five CV best EMA files, `reports/`, and all manifests. Its sha256 is recorded in `PROGRESS.md`.

---

## 14. `PROGRESS.md` template

Git-tracked. **Newest entry first**; cumulative registers above the entries. (The `HANDOFF.md` state block is doc 07 §4's.)

```markdown
# PROGRESS — text2mario

## Registers (cumulative)
### Assumptions
| ID | Date | Assumption | Why | Config flag | Revisit by |
|---|---|---|---|---|---|
| A-001 | 2026-11-18 | … | … | … | Day N |
### Doc conflicts
| ID | Date | Doc A says | Doc B / ledger says | Followed | Ledger change-log row? |
|---|---|---|---|---|---|
| C-001 | … | … | … | ledger / owner doc | yes / no |
### Fallbacks in use
| ID | Since | Fallback | Reason | Removal plan |
|---|---|---|---|---|

---

## 2026-11-20 · Day 3 · laptop_4050 · session 2
**Phase / gate:** Phase 2 → G-P2 ✅ (reports/gates/P2/RESULT.json @ <sha7>)
### Done
- scheduler.py (linear + cosine), q_sample, ddim_step, renoise; 14 property tests green
### In-flight runs
| run_id | machine | step / target | status | next check | handoff? |
|---|---|---|---|---|---|
### Gates
| Gate | Status | Day | Evidence |
|---|---|---|---|
| G-P1 | ✅ | 2 | reports/gates/P1/RESULT.json |
| G-P2 | ✅ | 3 | reports/gates/P2/RESULT.json |
### Blocked
- (none)
### Risks triggered (doc 33)
- R16 laptop thermal: 86 °C at 14:10 PKT; cooling pad added
### New assumptions / conflicts / fallbacks
- A-004 …
### Next (ordered)
1. Day 4: unet.py per doc 14 …
### Session-close checklist
- [ ] pytest -q green  - [ ] scripts/index_runs.py  - [ ] HANDOFF.md updated  - [ ] pushed
```

---

## 15. Experiment log template (`reports/experiments/X<n>_<slug>.md`)

Write it **before** launching. The hypothesis and decision rule are pre-registered; fill in the results after. Decision thresholds follow doc 22 §7.2 (minimum effects: ΔS ≥ 0.01, ΔC-score ≥ 0.02, Δ playable/broken ≥ 2 pp) and doc 23.

```markdown
# X3 — Pooled text embedding added to the time embedding
- **Status:** planned | running | done   **Track:** M   **Planned day:** 10   **Ledger rows:** M4, M7
- **Baseline:** config v0 (`cond_main`, config_hash …), run …

## Hypothesis (pre-registered)
Adding the mean-pooled adapter output to the time embedding improves count adherence
(enemies, coins) without raising the copy rate.

## Change vs baseline
```diff
- model.text.pooled_to_time: false
+ model.text.pooled_to_time: true
```
## Budget & seeds
40k steps, run seed 1 (+ seed 2 if the decision is borderline); evaluation seeds per doc 22 §2.3 (CRN).

## Decision rule (pre-registered)
Adopt iff the paired-bootstrap 95 % CI of ΔS (P-dev) lies above 0 and ΔS ≥ 0.01, and Δcopy_rate ≤ +0.02.
Otherwise keep the baseline (simpler wins ties).

## Runs
| run_id | seed | machine(s) | steps | best_step | status |
|---|---|---|---|---|---|

## Results (P-dev, EMA, DDIM 50, w as in baseline)
| metric | baseline | X3 | Δ | 95 % paired CI |
|---|---|---|---|---|

## Decision & rationale
## Side effects / surprises
## Follow-ups
## Provenance
src paths · git_sha · figure: reports/figures/fig_results_ablation_forest.png
```

---

## 16. Reproducing any reported number end-to-end

### 16.1 From the frozen checkpoint (L1 on the producing machine, L3 on the other; L2 with the §6.4 upgrade)
```bash
git clone <repo> text2mario && cd text2mario && git checkout v1.0
python -m venv .venv && . .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r env/lock_<machine>.txt && pip install -e .   # torch minor must equal FROZEN's (V2)
python scripts/verify_env.py                              # GPU, VRAM cap, versions, manifest hashes
python scripts/fetch_models.py                            # MiniLM at the pinned revision
python scripts/fetch_data.py                              # pinned commit from data/MANIFEST.json
python scripts/audit_dataset.py                           # asserts F1–F17; rebuilds index + scenes; content hashes must match MANIFEST
python scripts/make_splits.py --verify-lock               # regenerates split_v1.json; must equal LOCK
python scripts/make_prompt_sets.py --verify               # must equal PROMPTS_LOCK (doc 22 §2.2)
python scripts/make_val_noise.py --verify
python scripts/compute_references.py                      # REFERENCES.json (doc 22 §4)
# restore the frozen checkpoint files from the release archive into runs/<run_id>/checkpoints/ (sha256 per FROZEN.json)
python scripts/eval_dev.py --ckpt <FROZEN.checkpoint.ema> --sets p_dev,p_rand_dev,p_casual_dev --w <FROZEN.default_w> --seeds 3 --out reports/dev/<run_id>/
python scripts/eval_test.py --rerun-reason "reproduction: <who>, <date>"   # doc 22 guard 3; logged in TEST_LEDGER.md
python scripts/make_figures.py && python scripts/check_provenance.py
```
A reproduction run writes a **new** `reports/final/test_runs/<UTC>_<gitsha>/` folder and an entry in the append-only `TEST_LEDGER.md`. It never replaces the reported run. Compare with `scripts/compare_to_reported.py --run-dir <new>`.

### 16.2 From scratch (L3)
```bash
python scripts/launch.py --exp configs/final.yaml --seed 1 --purpose final
python scripts/eval_dev.py --ckpt runs/<new_run_id>/checkpoints/best.json --sets p_dev --w <FROZEN.default_w> --seeds 3 --out reports/dev/<new_run_id>/
python scripts/compare_to_reported.py --dev reports/dev/<new_run_id>/metrics.json   # each metric inside reported CI / CV spread?
```

### 16.3 Tracing one number backwards (illustrative values, not results)
`RESULTS.md: test C-score 0.81 [src=reports/final/test_runs/20261203-0510_4e1d0c9/metrics.json | key=p_test.B3.cscore.value | run=…_final_… | set=P-test | reps=3 | machine=desktop_3060ti | git=4e1d0c9]`
→ `metrics.json` aggregates `per_sample.parquet` (doc 22 §8), computed by `src/text2mario/eval/adherence.py` at that git SHA
→ from the grids in `samples.npz`, generated by the checkpoint whose sha256 is in `FROZEN.json`
→ trained by run `…` (`config.yaml`, `config_hash.txt`) on the split in `LOCK`, from data in `MANIFEST` (pinned commit).

---

## 17. Reproducibility checklist (before the Day 14 freeze and the Day 17 figures)
- [ ] `verify_hashes.py` green on both machines (data, scenes, split, prompts, val noise, references, assets).
- [ ] Same torch **minor** version on both machines; locks committed.
- [ ] `check_determinism.py` passed (repro mode bitwise; fast-mode drift documented).
- [ ] `test_training_inputs_bitwise` passed on both machines.
- [ ] `check_cross_machine_eval.py` passed (criterion a; plus b if the §6.4 upgrade is adopted).
- [ ] Every cited run: `git_dirty = false`, listed in `runs_index.csv`.
- [ ] `FROZEN.json` committed and tagged → then `eval_plan.json` committed (references FROZEN) → only then `eval_test.py`.
- [ ] `check_provenance.py` green (Day 17).
- [ ] Release archive built; its sha256 recorded (Day 20).
