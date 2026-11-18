# 15 — Text Conditioning and Classifier-Free Guidance

> **Purpose.** Exactly how a caption becomes the tensors the U-Net's cross-attention reads, how the null condition works, how CFG is trained and applied, how w is chosen, and how to tell when conditioning is broken.
> **Ledger rows:** V3 (encoder), M6 (cross-attention), M7 (adapter, null), M8 (online encoding + cache), D8 (caption augmentation incl. CFG dropout), M13 (CFG convention, w selection), G-P4 (gate).
> **Code:** `src/text2mario/text/encoder.py` (frozen MiniLM + cache), `src/text2mario/text/adapter.py`, `src/text2mario/text/conditioner.py` (public API), `src/text2mario/diffusion/guidance.py` (`guided_eps`). Normalizer (casual → canonical) lives in doc 12.

---

## 1. Pipeline at a glance

```
                       TRAINING                                            INFERENCE / EVAL
 stored caption ─► augmentation (D8, doc 11):                 user prompt ─► normalizer (doc 12) ─► canonical scene captions
                    a) phrase shuffle (always)                                              │
                    b) phrase dropout p=0.10 (keep ≥1)                                      │
                    c) paraphrase subst. p=0.30 (train lexicon)                             │
                    d) CFG null "" p=0.10  (applied LAST)                                    │
                          │                                                                  │
                          └──────────────► caption strings (B) ◄─────────────────────────────┘
                                               │
                     WordPiece tokenizer (uncased, max_len 48, pad-to-longest, attention mask)
                                               │
                     frozen all-MiniLM-L6-v2 (bf16, no_grad) ──► last_hidden_state (B, L, 384)   [string-keyed LRU cache]
                                               │
                     adapter (trainable): Linear(384,128) → SiLU → Linear(128,128) → LayerNorm(128)
                                               │
                     ctx (B, L, 128)  +  mask (B, L) bool (True = real token)
                                               │
                     11 cross-attention blocks (8×8 ×5, 4×4 ×6): Q = pixels, K/V = tokens   (doc 14)
```

---

## 2. The frozen encoder: `sentence-transformers/all-MiniLM-L6-v2` (V3)

| Property | Value |
|---|---|
| Architecture | BERT-style encoder, **6 layers**, hidden **384**, **12** heads, FFN 1536, max positions 512 |
| Tokenizer | WordPiece, **uncased** (lower-cases input), vocab **30,522** (bert-base-uncased vocab) |
| Parameters | **22,713,216** incl. the pooler (unused); **22,565,376** without; embeddings 11,918,592 (verified by instantiating the architecture) |
| What we read | `last_hidden_state`: **token states (B, L, 384)**. **No mean pooling and no L2 normalisation** (the sentence-transformers pipeline would do both; we bypass it) |
| Mode | `.eval()`, `requires_grad_(False)`, weights cast to **bf16**, every call under `torch.no_grad()` |
| Revision | pinned to a commit SHA on Day 7 → `cfg.text.encoder_revision`; stored in every checkpoint |
| Offline | `scripts/fetch_text_encoder.py` downloads it once into `models/hf/` (set `HF_HOME`); at runtime load with `local_files_only=True` so the viva works with Wi-Fi off (doc 06, doc 28) |
| Cost | forward 66.6 GFLOP per batch of 64 at L = 48; 21.9 GFLOP at L = 16; 1.04 GFLOP for one prompt at L = 48 (verified with `FlopCounterMode`) |

**Why token states, not a pooled vector.** Cross-attention lets each pixel attend to individual words ("pipe", "enemies", "three"). A pooled sentence vector averages them away, which hurts counts and the co-presence of several concepts. (The paper's MiniLM models fed pooled vectors, one per caption or one per phrase. This project follows the PDF: `(Batch, Seq_Len, 384)` token states feeding cross-attention.)

**Why frozen (corrected rationale; audit #12).** The PDF's reason, "prevents gradient collapse", is not the real one. The real reasons:
1. **Tiny caption language.** The corpus has 2,668 distinct caption strings built from 62 phrases and 44 words (F14). Fine-tuning 22.7 M parameters on that would collapse MiniLM's general English geometry onto those few phrases (catastrophic forgetting). We need that geometry to keep casual words ("tubes", "goombas", "pits") near their canonical counterparts.
2. **Stability and simplicity.** The only trainable text parameters are the 66,048 adapter weights. No LM learning-rate tuning, no LM weights in checkpoints (only the revision string), and no LM optimizer state in VRAM.
3. **Compute and memory.** No backward pass through 22.7 M parameters; frozen bf16 weights take 45 MB.

---

## 3. Tokenisation

- Call: `tokenizer(captions, max_length=48, truncation=True, padding="longest", return_attention_mask=True, return_tensors="pt")`.
- `""` → `[CLS] [SEP]` (2 tokens). Every sequence therefore has ≥ 2 valid tokens, so no cross-attention row is ever fully masked.
- **Length budget (measured from the data on 2026-11-17).** Counting one token per word plus one per period plus [CLS]/[SEP]: **max 40, mean 17.4** tokens. Worst case, if every one of the 14 plural/rare words (`staircases`, `cannons`, `chunks`, `clusters`, …) splits into two WordPieces: **max 44** ≤ 48 ✓. (The real tokenizer could not be downloaded in the planning sandbox. Test T1 re-measures with it on Day 7, and if max > 48 the fix is a ledger change to `max_len: 64`.)
- **Canonical formatting contract** (shared with the normalizer, doc 12, and the dataset, doc 11): lower-case phrases joined by `". "` with a trailing `"."`, e.g. `"full floor. one enemy. a few question blocks."`. Training captions, eval prompts and normalizer outputs must all use this exact surface form (test T8).
- Inference prompts longer than 48 tokens after normalisation are **truncated with a visible UI warning** (logged as `prompt_truncated`). The multi-scene splitter normally keeps every scene prompt far below the limit.

---

## 4. Adapter and context tensor (M7)

```python
adapter = nn.Sequential(nn.Linear(384, 128), nn.SiLU(), nn.Linear(128, 128), nn.LayerNorm(128))   # 66,048 params
ctx = adapter(token_states.to(compute_dtype))     # (B, L, 128); applied per token, pads included (they are masked later)
```
- **LayerNorm at the end** makes every token's key/value input unit-scale, whatever MiniLM's per-token norm is ([CLS] vs word tokens differ a lot), so cross-attention logits start well-conditioned.
- Init: PyTorch defaults. The adapter is in the optimizer (weights → decay group; biases and LayerNorm → no-decay group; doc 16 §5) and in the EMA.
- The same `ctx`/`mask` feeds all 11 cross-attention layers; each layer has its own $W_k, W_v$ (128 → C).
- **Optional (X3):** `pooled_text_to_time: true` adds `Linear(128, 256)(masked_mean(ctx))` (zero-init, +33,024 params) to the time embedding e(t), giving every ResBlock a global caption summary.

---

## 5. Online encoding and the LRU cache (M8, with corrected expectations)

- **Key:** the final caption string after augmentation. **Value:** MiniLM token states for that string, **unpadded** `(L_i, 384)` bf16, stored on **CPU** (pinned memory).
- **Capacity:** byte-capped at `text.cache_max_mb = 512` (LRU eviction). A byte cap, unlike an entry-count cap, bounds RAM on both machines regardless of caption length. Mean entry ≈ 17 tokens × 384 × 2 B ≈ 13 KB, so 512 MB ≈ 40k entries.
- **Batch assembly:** look up all B strings, encode the misses **together in one MiniLM call** (padded to the longest miss), insert them, then gather all B, pad to the batch maximum, build the mask, and transfer with `non_blocking=True`.
- **Expected hit rates (simulated on the real caption distribution, 6k steps × 64):**

| Setting | Hit rate | Distinct strings seen |
|---|---|---|
| Full D8 augmentation, cache 20k entries | 33.5 % | 214,769 |
| Full D8 augmentation, cache 100k entries | 42.2 % | 214,921 |
| Without paraphrase substitution | 68.9 % | 118,770 |
| Eval / inference (fixed prompt sets) | ≈ 100 % after the first pass | — |

**Correction to M8's rationale.** "Captions repeat heavily" is true of the raw corpus (2,668 distinct strings), **not** of augmented training strings: phrase shuffling alone allows 14.6 M distinct orderings. In training the cache saves roughly a third of encoder calls. Its real value is at eval and inference. With dynamic padding, uncached encoding adds about 10–25 % compute to a training step (≈ 22–67 GFLOP vs ≈ 238 GFLOP for the U-Net step); with the cache, about 6–15 %. Both are acceptable. The trainer logs `text_cache_hit_rate` and `text_encode_ms` every log interval (doc 16 §15), so the real number is known by Day 7. (Reported to the ledger owner as a rationale fix, not a decision change.)

---

## 6. The null condition and CFG training (M7, D8d)

- **Null caption = `""`** encoded by the same frozen MiniLM (→ `[CLS][SEP]`, `(1, 2, 384)`). It is computed **once** at start-up and stored as a buffer, because the encoder is frozen. It still passes through the **trainable** adapter on every use, so the null context co-evolves with the adapter.
- **Training dropout:** after augmentations (a)–(c), each sample's caption string is replaced by `""` with probability **0.10**, independently per sample, using the dataset's CPU generator (machine-independent; doc 16 §9). Test T6 checks the empirical rate is 0.10 ± 0.01 over 10k draws.
- **The unconditional Phase-3 model** has no text path at all: its cross-attention reads a learned `null_ctx` (doc 14 §7).
- *Considered, not adopted:* a learned null embedding for the conditional model. Using `""` keeps the training-dropout path and the inference unconditional branch byte-identical (M7).

---

## 7. Guided inference (M13)

```python
def guided_eps(model, x_t, t, ctx_c, mask_c, ctx_u, mask_u, w):
    L = max(ctx_c.shape[1], ctx_u.shape[1])                     # pad both contexts to the same length
    ctx  = torch.cat([pad(ctx_c, L),  pad(ctx_u, L)])           # (2B, L, 128)
    mask = torch.cat([pad(mask_c, L), pad(mask_u, L)])          # (2B, L) — padding = False
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(torch.cat([x_t, x_t]), torch.cat([t, t]), ctx, mask)
    eps_c, eps_u = out.float().chunk(2)
    return eps_u + w * (eps_c - eps_u)                           # w = 1 → conditional; w = 0 → unconditional
```
- One forward pass on a 2B batch per step (≈ 2.48 GFLOP per sample at 16×16). On CPU this is the ≈ 75 ms pass in H10.
- The DDIM update then clips $\hat x_0$ to [−1, 1] and recomputes $\hat\epsilon$ from the clipped $\hat x_0$ (doc 13 §7.2).
- **Diagnostic logged per sampling run: clip rate** = fraction of $\hat x_0$ elements changed by clipping, averaged over steps. It rises steeply when w is too high.

### 7.1 Choosing w
- **Search space (M13):** w ∈ {1, 2, 3, 5, 7.5, 10}, on **P-dev** (dev-window captions at stride 8), **3 seeds**, EMA weights, DDIM 50.
- **Selection rule:** maximise the dev composite score E4. **Tie** (difference inside the bootstrap CI) → the **smaller w**, which gives more diversity, fewer broken structures and a lower copy rate.
- **Report curves vs w:** C-score, per-concept exact match, broken-structure rate, playability, novelty (median NN distance), copy rate, within-prompt diversity, clip rate. These form one figure in doc 32.
- **Training-time evaluations use a fixed w = 5.0** (the PDF prior), so checkpoint selection is not coupled to the w search (doc 16 §11). The w sweep runs once, on Day 9, for the selected checkpoint, and is confirmed for the final config on Day 14.

### 7.2 CFG failure modes and remedies

| Symptom | Likely cause | Check | Remedy (doc 25 has the full matrix) |
|---|---|---|---|
| Counts ignored (asked for 3 pipes, got 0) | w too low, or conditioning not learned | C-score vs w is flat; conditioning gain (§8) ≈ 0 | verify adapter in optimizer (T3); raise w; check null-dropout rate (T6) |
| Half pipes / broken cannons appear as w grows | over-extrapolation pushes $\hat x_0$ off the vertex set | broken rate and clip rate rise with w | lower w; optional simplex projection (doc 13 §7.4); optional interval guidance (below) |
| Same layout for every seed | guidance collapses onto a mode | diversity falls, copy rate rises with w | lower w; check copy rate against the dev reference |
| Caricatures ("many enemies" → enemy carpet) | over-sharpening of $p(c\mid x)^w$ | exact-match falls while C-score holds | lower w; per-concept w analysis (X7) |
| Noisy, unstable guidance at all w | weak unconditional branch | the null-caption val loss is much worse than the conditional one in high-t buckets | raise p_uncond to 0.2 (X10) |
| Contradictory prompt ("full floor" + "giant gap") produces garbage | the caption is impossible | normalizer warnings | normalizer rejects or resolves contradictions (doc 12) |

**Interval guidance (documented option, off by default):** apply w only for strided steps with $\tau\in[t_{lo}, t_{hi}]$ (e.g. [100, 900]) and use w = 1 elsewhere (Kynkäänniemi et al., 2024). Config `sampling.guidance_interval: null | [lo, hi]`. Enable only via a dev experiment.

**Negative prompts (future seam):** $\hat\epsilon = \epsilon_u + w(\epsilon_c-\epsilon_u) - w_{neg}(\epsilon_{neg}-\epsilon_u)$ with a 3B batch. The API keeps `negative_caption: str | None = None` (MarioDiffusion supports this; we reserve it, doc 31). Not trained for, not evaluated.

---

## 8. Is the model actually listening? Conditioning diagnostics

1. **Conditioning gain** (computed in every validation pass, doc 16 §10): $G_b = \mathcal L^{\text{val}}_{b}(\varnothing) - \mathcal L^{\text{val}}_{b}(c)$ per t-bucket b, on the same fixed (window, t, ε) triples. It must become **> 0 in the mid-t buckets** within the Day-7 smoke run (3k steps) and keep growing in Phase 4. If $G\approx0$, the model ignores the text (PDF rollback "model ignores the prompt"): check T3, T6 and the mask (T7) before anything else.
2. **Cross-attention entropy** (Day-7 smoke test): for each CA layer, mean over heads, pixels and batch of $H = -\sum_j p_{ij}\log p_{ij}$, normalised by $\log L_{\text{valid}}$. Pass: at least one CA layer per resolution has normalised entropy < 0.9 after 3k steps (attention is not uniform). Uses `return_attn=True` (doc 14 §9).
3. **The PDF contrast test** (Day 9, G-P4): canonical forms of "full floor with three pipes and no enemies" and "staircase with gap and several enemies" × 8 seeds, rendered side by side, plus their detected captions.
4. **Phrase-order sensitivity** (measured, not gated): C-score of the same phrase set under 5 orderings. Since training shuffles phrases, the spread should be small. A large spread means the model leans on position (expected only if shuffling was off).

---

## 9. Interfaces (`src/text2mario/text/conditioner.py`)

```python
class TextConditioner(nn.Module):
    """Frozen MiniLM + trainable adapter + CPU LRU cache. Only the adapter is trainable."""
    def __init__(self, cfg_text, device): ...
    @torch.no_grad()
    def encode_frozen(self, captions: list[str]) -> list[torch.Tensor]:     # list of (L_i, 384) bf16, via cache
    def forward(self, captions: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns ctx (B, L, 128) in compute dtype and mask (B, L) bool. '' is a valid caption (null)."""
    def null(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:   # ctx/mask for ""
    def trainable_parameters(self): return self.adapter.parameters()
    def state_dict_for_ckpt(self) -> dict:   # adapter weights + encoder name + revision + probe vector; NEVER MiniLM weights
    def check_probe(self, saved_probe) -> None:  # raises if the local MiniLM differs (see below)
    cache_stats: dict  # hits, misses, bytes, evictions
```

**Cross-machine safety check (probe vector).** Each checkpoint stores the MiniLM token states of the fixed probe sentence `"full floor. two pipes. a few enemies."` (12 tokens incl. [CLS]/[SEP] if every word is one WordPiece → ≈ 12 × 384 bf16 values). On load, the conditioner re-encodes the probe and requires cosine similarity > 0.999 for every token. This catches a different MiniLM revision, tokenizer or corrupted cache on the other machine (doc 07) before it silently degrades the model. The tolerance absorbs bf16 kernel differences between the RTX 3060 Ti and RTX 4050.

---

## 10. Tests (`tests/test_text.py`)

| ID | Test | Pass condition |
|---|---|---|
| T1 | token lengths | real tokenizer over all 7,687 captions: max ≤ 48 (expected ≤ 44); `""` → 2 tokens |
| T2 | encoder frozen | every MiniLM param `requires_grad == False`; none appears in `optimizer.param_groups` |
| T3 | adapter trained | every adapter param appears in exactly one optimizer group (PDF rollback #3 as a test) and in the EMA |
| T4 | MiniLM unchanged by training | probe output hash identical before and after 50 training steps |
| T5 | shapes / dtypes | ctx (B, L, 128), mask (B, L) bool, ≥ 2 True per row |
| T6 | null dropout rate | empirical rate 0.10 ± 0.01 over 10k draws (seeded) |
| T7 | mask correctness | changing token states at masked positions changes ε̂ by < 1e-5 (fp32) |
| T8 | canonical surface form | dataset captions, normalizer outputs and eval prompts all match the regex `^([a-z ]+\. )*[a-z ]+\.$` (or the empty string) |
| T9 | cache equivalence | cached vs freshly encoded token states: cosine > 0.999 per token (bf16 tolerance); LRU respects the byte cap |
| T10 | guidance algebra | `guided_eps` with w = 1 equals the conditional forward, w = 0 equals the unconditional forward; the 2B batched result equals two separate forwards (fp32, atol 1e-5) |
| T11 | probe check | a perturbed probe (simulated wrong revision) raises; the correct one passes |
| T12 (smoke, slow) | attention entropy + conditioning gain | after 3k steps: entropy criterion of §8.2 holds; $G_b > 0$ in buckets 3–6 |

---

## 11. Alternatives considered (documented; not in the plan)

| Option | Pro | Con | Status |
|---|---|---|---|
| Per-phrase pooled embeddings (one vector per phrase, as in MarioDiffusion "multiple") | invariant to phrase order by construction; only ~62 canonical phrases → perfect caching | departs from the PDF's token-level design; loses within-phrase word binding | candidate future experiment |
| Train a small text encoder from scratch (the paper's best model was an MLM transformer) | strong on canonical captions | no robustness to casual words without the normalizer; extra training stage | out of scope |
| LoRA fine-tuning of MiniLM | adapts to Mario vocabulary | overfitting on 62 phrases; more VRAM; more knobs | out of scope |
| Learned null embedding | common in image models | breaks byte-identity between training dropout and inference null | rejected (M7) |

---

## 12. Seams for future features (doc 31)
- **"See what it heard" (word highlighting):** `return_attn=True` exposes per-token cross-attention maps. Aggregating over layers and heads, then upsampling to 16×16, gives a heat map per word. A second, model-free source is the caption detector's per-phrase tile coordinates (F19).
- **"Erase and re-ask":** the same RePaint machinery (doc 13 §9) with a user mask and a new caption for the erased region.
- **Negative prompts:** `negative_caption` argument reserved (§7.2).
