# 11 — Data Pipeline and Representation

> **Day:** 2 (dataset + augmentation), Day 7 (text side wired to the model). **Ledger rows:** F5, F6, H4, H7, D6, D7, D8, D9, M7, M8.
> **Modules:** `src/text2mario/data/tiles.py`, `provenance.py`, `splits.py`, `dataset.py`, `sampler.py`, `augment.py`; `scripts/measure_tokens.py`.
> **Design goals:** (1) identical batches on both GPUs for the same seed and step; (2) zero DataLoader workers (H7); (3) any training step resumable from `(seed, step)` alone; (4) the test split physically unreachable before the freeze.

---

## 1. The tile table (`tiles.py`) — single source of tile truth

```python
CHARS = ('-', '<', '>', '?', 'B', 'E', 'Q', 'S', 'X', '[', ']', 'b', 'o')   # ledger F6, ids 0..12
NAMES = ('sky', 'pipe_top_left', 'pipe_top_right', 'question_full', 'cannon_top', 'enemy',
         'question_empty', 'breakable', 'ground', 'pipe_body_left', 'pipe_body_right',
         'cannon_support', 'coin')
SOLID    = frozenset({1, 2, 3, 4, 6, 7, 8, 9, 10, 11})
PASSABLE = frozenset({0, 12})
ENEMY    = frozenset({5})
PIPE     = frozenset({1, 2, 9, 10})
CANNON   = frozenset({4, 11})
N_TILES  = 13
CHAR_TO_ID = {c: i for i, c in enumerate(CHARS)}
```
- A test asserts `CHARS == tuple(sorted(json.load(open("data/source/smb.json"))["tiles"]))` and that `SOLID/PASSABLE/ENEMY/PIPE` agree with the descriptors in `smb.json`.
- Placeholder colours for debug renders (doc 26 defines the real sprites): sky `#9ccfff`, ground `#a0522d`, breakable `#c86432`, question `#f8d030`, empty-question `#a08060`, enemy `#e03030`, pipe tops `#20c040`, pipe bodies `#109030`, cannon top `#303030`, cannon support `#606060`, coin `#ffd700`.

## 2. Tensor formats

| Name | dtype | Shape | Values | Where |
|---|---|---|---|---|
| `grid` | `uint8` (storage) / `int64` (indexing) | (16, W) or (B, 16, W) | 0…12 | storage, detector, renderer, playability, exports |
| `x0` | `bfloat16` (training under autocast) or `float32` (tests, eval) | (B, 13, 16, W) | {−1, +1} | model input/target space |
| `x̂0` | `float32` | (B, 13, 16, W) | ℝ, clipped to [−1, 1] at sampling (M13) | sampler |
| decoded grid | `int64` | (B, 16, W) | `argmax(x̂0, dim=1)` (ties → lowest id, `torch.argmax` semantics) | everywhere after sampling |
| `token_ids` | `int64` | (B, L) | WordPiece ids, L ≤ `max_len` | text encoder |
| `attn_mask` | `bool` | (B, L) | True = real token | text encoder, cross-attention key-padding mask |
| `ctx` | `bfloat16` | (B, L, 128) | adapter output | U-Net cross-attention (doc 15) |

Axis convention is fixed everywhere: **row 0 = top, row 15 = bottom (floor), column 0 = left**. Image renders use the same orientation, so a render with the floor at the top is a bug (G-P1 check).

## 3. One-hot ±1 representation: the maths you need

For a grid g with cell tiles g_ij ∈ {0…12}, the model space is
```
x_{c,i,j} = 2·[g_ij = c] − 1,     c = 0…12          (D7)
```
Properties (use these in tests and in the viva):
1. **Per-cell structure:** each cell vector has exactly one +1 and twelve −1 entries. Its mean is (1 − 12)/13 = −11/13 ≈ −0.846 and its squared norm is 13.
2. **Distance ↔ Hamming:** for two grids, `‖x(g) − x(h)‖² = 8 · Hamming(g, h)` (each differing cell contributes two channels that each differ by 2). So nearest neighbours in x-space are nearest neighbours in Hamming, and Hamming matrices can be computed as `256 − ⟨e(g), e(h)⟩` from one-hot dot products (doc 10 §3.2, doc 22 novelty).
3. **Per-channel statistics:** with tile frequency p_c (F13), channel mean `μ_c = 2p_c − 1` and variance `σ_c² = 4p_c(1 − p_c)`.
   - sky: p = 0.8713 → μ = +0.743, σ² = 0.449
   - ground: p = 0.0866 → μ = −0.827, σ² = 0.316
   - cannon support: p = 0.0003 → μ ≈ −0.9994, σ² ≈ 0.0012

   The data is **not** zero-mean/unit-variance. That is fine for DDPM because x_T is pure noise to within √ᾱ_T ≈ 0.006 (linear) or 5e-5 (cosine) (G-P2). But rare channels carry very little variance, so with plain ε-MSE their errors are cheap. This is the mechanistic reason behind the "sky overwhelms everything" failure mode and its rollbacks (channel weights, auxiliary CE: M9, doc 25).
4. **Decoding is argmax, never thresholding.** argmax is invariant to any per-cell additive shift, so a model that predicts uniformly offset logits still decodes correctly.
5. **Clipping x̂0 to [−1, 1]** (M13) keeps predictions inside the bounding box of valid encodings. It does not project onto the simplex of valid one-hots, and doesn't need to, because decoding is argmax.
6. **Signal-to-noise:** `x_t = √ᾱ_t x0 + √(1−ᾱ_t) ε` has per-element SNR ᾱ_t/(1−ᾱ_t) relative to the ±1 signal. The tile identity of a cell is effectively lost once the noise std exceeds roughly the ±1 margin, which happens around ᾱ_t ≈ 0.2–0.5. That's why the cosine schedule (more steps in that range) often helps small discrete grids (M2 ablation).

## 4. GPU-resident dataset (`dataset.py`)

```python
@dataclass(frozen=True)
class DataConfig:
    scenes_path: str = "data/processed/scenes_u8.npy"
    index_path: str = "data/processed/index.parquet"
    split_path: str = "data/splits/split_v1.json"
    dtype: str = "bfloat16"                 # x0 dtype; tests use float32
    max_len: int = 48                       # M7; asserted by measure_tokens.py (§6)
    aug: AugConfig = AugConfig()            # §5

class MarioWindows:
    """All windows of the requested levels, resident on `device` as uint8."""
    def __init__(self, split: Literal["train", "dev", "test", "all"] | None = None, *,
                 levels: Sequence[int] | None = None,     # explicit level ids (CV folds) instead of split
                 device: str | torch.device = "cuda",
                 cfg: DataConfig = DataConfig(),
                 allow_test: bool = False) -> None: ...
    # attributes
    grids: torch.Tensor          # uint8 (N, 16, 16) on device
    level_id: torch.Tensor       # int16 (N,) on device
    col_offset: torch.Tensor     # int16 (N,)
    game: torch.Tensor           # int8 (N,) 0=SMB1, 1=SMB2
    captions: list[str]          # CPU, stored captions (generator order)
    phrases: list[tuple[str, ...]]   # CPU, pre-split phrases
    source_idx: torch.Tensor     # int32 (N,) index into the full corpus
    reused: torch.Tensor         # bool (N,) reused-segment flag (doc 10 §6), False for train
    def __len__(self) -> int: ...
    def onehot(self, rows: torch.Tensor, dtype: torch.dtype | None = None) -> torch.Tensor:
        """(B,) row indices → (B, 13, 16, 16) in {−1, +1} on device."""
    def eval_rows(self, stride: int = 8) -> torch.Tensor:
        """Rows with col_offset % stride == 0 (D6), level order."""
    def batch(self, rows: torch.Tensor, step: int, *, uncond: bool = False) -> "Batch": ...
```

Implementation notes:
- Load `scenes_u8.npy` (verified by `provenance.verify()`), select rows of the requested levels, then **one** `torch.from_numpy(...).to(device)`. Train ≈ 5.7k × 256 B ≈ 1.5 MB of VRAM.
- `onehot`: `F.one_hot(self.grids[rows].long(), 13).permute(0, 3, 1, 2).to(dtype).mul_(2).sub_(1)`. That's exact in bf16 (±1 are representable).
- **Test guard (T10 of doc 10):** if `split == "test"` (or `levels` intersects the test levels) and not (`allow_test` and `Path("FROZEN.json").exists()`), raise `PermissionError("Test split is locked until FROZEN.json exists (D4)")`. Only `scripts/eval_test.py` passes `allow_test=True`.
- **CV folds:** `MarioWindows(levels=fold_train_levels)`. The guard still applies if a fold mistakenly includes a test level.
- **No PyTorch `Dataset`/`DataLoader`** in training. (A thin `torch.utils.data.Dataset` wrapper exists only for ad-hoc notebooks.)

```python
@dataclass
class Batch:
    x0: torch.Tensor            # (B, 13, 16, 16) {−1,+1}, device, cfg dtype
    grid: torch.Tensor          # (B, 16, 16) uint8, device (metrics/debug)
    captions: list[str]         # post-augmentation; "" = null caption
    is_null: torch.Tensor       # (B,) bool, device (True for CFG-dropped or uncond)
    rows: torch.Tensor          # (B,) int64 dataset rows (CPU)
    level_id: torch.Tensor      # (B,) int16 (CPU)
    aug: list[dict] | None      # per-sample augmentation record (when cfg.aug.record=True)
```
Text tokenisation and encoding happen in the trainer through `TextConditioner` (doc 15), using `captions`. Its string-keyed LRU cache sees roughly 2,668 base captions plus augmented variants.

## 5. Sampler and caption augmentation

### 5.1 Stateless, resumable index sampler (`sampler.py`)
Every batch is a pure function of `(seed, step)`. Resume needs nothing but the step counter, and both machines produce identical index streams.

```python
class WindowSampler:
    def __init__(self, n: int, batch_size: int = 64, seed: int = 20261117,
                 mode: Literal["epoch_perm", "balanced"] = "epoch_perm",
                 weights: np.ndarray | None = None): ...
    def rows(self, step: int) -> torch.Tensor:      # (batch_size,) int64, CPU
    def state_dict(self) -> dict: return {"mode": ..., "seed": ..., "n": ..., "batch_size": ...}
```
- **`epoch_perm` (default):** conceptually concatenate permutations π₀, π₁, … of `range(n)`, where πₑ = `torch.randperm(n, generator=torch.Generator("cpu").manual_seed(mix(seed, e)))`. Batch t = elements `[64t, 64t+64)` of the concatenation, spanning an epoch boundary when needed (no sample is ever dropped). Epoch e = ⌊64t / n⌋. Cache the current and next permutation.
- **`balanced` (D9 ablation):** `torch.multinomial(weights, 64, replacement=True, generator=Generator("cpu").manual_seed(mix(seed, 1_000_000 + t)))`, with `weights_i = max_{c ∈ concepts(i)} f_c^(−1/2)` (f_c = train-split concept frequency), normalised. Every window has ≥ 1 phrase (F14), so weights are well defined.
- `mix(a, b) = int.from_bytes(hashlib.blake2b(f"{a}:{b}".encode(), digest_size=8).digest(), "little") & (2**63 − 1)`. **Never use Python's `hash()`** for seeding (string hashing is salted per process via `PYTHONHASHSEED`).
- **CPU generators only.** CUDA RNG streams differ between GPU models and drivers; CPU `torch.Generator` streams are portable.
- The checkpoint stores `sampler.state_dict()` and the global step (doc 16). Changing `n` (a different split) invalidates a resume, and the trainer refuses to continue.

### 5.2 Caption augmentation (`augment.py`) — order and probabilities (D8)
For sample i of step t, use `rng = numpy.random.default_rng(mix(seed, mix(t, i)))`. That's stateless, so the exact augmentation is reproducible from (seed, t, i) and is identical on both machines.

| Order | Operation | Probability / rule | Config key |
|---|---|---|---|
| 0 | Parse stored caption into phrases (`split(".")`, strip, drop empty); lowercase; collapse whitespace | always | — |
| 1 | **Phrase dropout** (D8b): keep each phrase independently with prob 0.9; if none survive, keep one uniformly chosen phrase | p_drop = 0.10 per phrase | `aug.phrase_dropout_p` |
| 2 | **Casual paraphrase** (D8c): with prob 0.30 the caption is paraphrased; inside a paraphrased caption each surviving phrase is rewritten with prob 0.5 (at least one phrase) using **only `split: train` lexicon variants** (doc 12 §6) | 0.30 per caption × 0.5 per phrase | `aug.paraphrase_caption_p`, `aug.paraphrase_phrase_p` |
| 3 | **Phrase shuffle** (D8a): uniform random permutation | always | `aug.shuffle` (True) |
| 4 | **Join** as `"p1. p2. … pk."` (single space after each period, final period) — the same format the normalizer emits | always | — |
| 5 | **CFG null dropout** (D8d): replace the whole caption with `""` | p = 0.10, applied last | `aug.null_p` |
| — | Unconditional phase (Phase 3): every caption is `""`, `is_null = True`; the model uses its learned null context (M7) | `uncond=True` | — |

**Paraphrase rewrite of one phrase** (`paraphrase_phrase(phrase, lexicon, rng)`):
1. Parse with the canonical grammar (doc 12 §2) into `(topic, bucket, form)`.
2. Quantity: replace the bucket word with a casual equivalent that the normalizer maps back to the **same bucket**: one → {"1", "a", "a single"}; two → {"2", "a couple of", "a pair of"}; a few → {"3", "4", "some", "a handful of"}; several → {"5", "6", "7", "8", "9", "a bunch of"}; many → {"10", "12", "lots of", "tons of"}. Pick uniformly; keep the canonical word with prob 0.25.
3. Noun: replace the canonical noun with a random **train-split** synonym of matching number (e.g., "enemies" → "goombas" / "bad guys"; "pipes" → "tubes").
4. Floor/ceiling templates use the template paraphrases in doc 12 §6.2 (e.g., "full floor" → "solid ground"; "floor with two gaps" → "ground with 2 pits").
5. Invariant: `normalize(paraphrase_phrase(p)) == p` for every phrase and every rng draw (unit-tested). This guarantees augmentation never teaches a label inconsistent with inference-time normalisation.

**Why phrase dropout is on despite its side effect.** Dropout makes training captions under-describe their scene, so the model learns to honour partial prompts ("three pipes" alone). The cost: "phrase absent ⇒ concept absent" becomes only a soft prior. Doc 12 §5.6 handles explicit negations ("no enemies") with a `forbid` list for the generator's verify loop instead of relying on omission.

### 5.3 Expected augmentation statistics (asserted over 10,000 draws, ±1 pp)
- Null captions: 10.0 % (test T-A3).
- Captions with at least one dropped phrase (given n = 4 phrases): 1 − 0.9⁴ ≈ 34.4 %.
- Captions containing at least one casual word: ≈ 0.9 × 30 % ≈ 27 % (the 0.9 factor because null-dropped captions contain nothing).
- Phrase order equal to the stored order (given n = 4): 1/4! ≈ 4.2 %.

## 6. Tokenizer facts — measured on Day 1/2 (`scripts/measure_tokens.py`)

The MiniLM tokenizer is the BERT-uncased WordPiece tokenizer (vocabulary 30,522). The audit sandbox could not reach Hugging Face, so these are **to be measured**, not assumed:
1. Token length (with `[CLS]`/`[SEP]`) of all 7,687 stored captions: report min / mean / p99 / max. Bound: ≤ 38 whitespace tokens including periods (doc 09) + 2 special tokens; WordPiece splits can add a few.
2. Token length of 10,000 augmented captions (§5.2, including casual paraphrases).
3. Token length of the 60 normalised P-casual prompts.
4. `""` → `[CLS] [SEP]` = **2 tokens** (the null context of M7).
5. List any caption words that split into multiple word pieces (e.g., "staircases" might become "staircase ##s").

**Assertion:** max over (1)–(3) ≤ `cfg.max_len` (48, M7). If violated, set `max_len = 64` in `configs/text.yaml`, record it in the ledger change log, and rerun. The encoder **raises** on overflow; it never truncates silently. Padding is to the batch maximum (dynamic), not to `max_len`, so a larger `max_len` costs nothing in the common case.

## 7. Edge cases and their handling

| Case | Handling |
|---|---|
| Caption with trailing/double spaces or uppercase | normalised at parse (step 0); stored captions are already lowercase |
| A phrase that the grammar cannot parse (corrupt data) | hard error in `augment.py` (cannot happen with the audited corpus; guards future datasets) |
| Paraphrase target topic without train-split synonyms | keep the canonical noun (log at debug level) |
| Batch spans an epoch boundary | handled by the concatenated-permutation definition (§5.1) |
| Duplicate windows (F4) | sampled like any other (D: duplicates kept, doc 09 §7) |
| Resume with a different split or batch size | trainer refuses (sampler state mismatch) |
| CPU-only machine / CI | `device="cpu"`, `dtype=float32`; identical index stream |
| Wide grids (W > 16) at inference | `onehot`-style encoding works for any W; the dataset only ever yields W = 16 |
| Reused-segment windows in dev/test | carried via `reused` for metric stratification (doc 10 §6) |
| Accidental test access | `PermissionError` (§4) |

## 8. Performance budget (laptop RTX 4050, expected)
- Index sampling: < 0.1 ms per step (cached permutation slice).
- One-hot on GPU: < 0.2 ms for (64, 13, 16, 16).
- Caption augmentation: ≈ 0.05 ms per sample in pure Python → ≈ 3 ms per batch. This is the dominant data cost, still negligible next to the U-Net step. If profiling shows otherwise, precompute augmented captions for the next N steps in a background thread (threads are fine on Windows; processes are not needed).
- Text encoding: cached per unique string; a cache miss is one MiniLM forward pass of ≤ 64 short sequences in bf16 (~2–4 ms).

## 9. Unit tests (`tests/test_data.py`, `tests/test_sampler.py`, `tests/test_augment.py`)

| ID | Test |
|---|---|
| T-D1 | `tiles.CHARS` equals the sorted `smb.json` keys; descriptor sets agree |
| T-D2 | Round trip: `argmax(onehot(grid)) == grid` for 1,000 random corpus windows (all dtypes) |
| T-D3 | `‖x(g) − x(h)‖² == 8·Hamming(g, h)` on random pairs |
| T-D4 | Orientation: corpus window idx 0 has non-sky cells in row 15 and only sky in row 0 |
| T-D5 | `MarioWindows("train")` levels equal `split_v1.json["splits"]["train"]`; window count equals the split file's count |
| T-D6 | Test guard raises without `FROZEN.json`; passes in a temp dir with a dummy `FROZEN.json` and `allow_test=True` |
| T-S1 | `WindowSampler.rows(t)` identical across two fresh instances, and across `cpu` vs a `cuda`-resident dataset |
| T-S2 | Over one epoch's worth of steps (⌈n/64⌉), every row appears once or twice (boundary spill) and no row is missing |
| T-S3 | Resume equivalence: rows(t) for t ∈ [1000, 1100) are identical whether computed fresh or after "restoring" at step 1000 |
| T-S4 | Balanced mode: empirical concept frequencies move toward uniform (rare-concept share rises by ≥ 2× for descending staircases) |
| T-A1 | Augmentation is a pure function of (seed, t, i) |
| T-A2 | At least one phrase always survives dropout; no empty non-null caption |
| T-A3 | Null rate 0.10 ± 0.01 over 10k draws (the PDF's "verify 10 % null-prompt dropout" rollback item, turned into a test) |
| T-A4 | Paraphrase invariant: `normalize(paraphrase_phrase(p)) == p` for all 62 phrases × 50 draws |
| T-A5 | Paraphrases use only `split: train` lexicon entries (scan output tokens against the held-out list) |
| T-A6 | Joined caption format matches the regex `^([a-z0-9 ?'\-]+\. )*[a-z0-9 ?'\-]+\.$` or `""` (casual variants may contain `?`, `'`, `-`, e.g. "?-blocks", "bad guys") |
| T-A7 | `measure_tokens.py` assertion: max tokens ≤ `max_len` |
