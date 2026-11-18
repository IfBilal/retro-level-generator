# 14 — Model Architecture: the Conditional 2-Stage U-Net (exact specification)

> **Purpose.** A layer-by-layer specification precise enough that two independent implementations produce the same parameter count, the same state-dict structure and (with copied weights) the same outputs. **Oracle:** `reference/ref_unet.py`. The production `src/text2mario/models/unet.py` must reproduce its parameter count **18,312,013** and pass the positional weight-copy parity test (§11, test U2).
> **Ledger rows:** M3 (channels, blocks, GroupNorm, dropout), M4 (time embedding), M5 (self-attention), M6 (cross-attention), M7 (text adapter, null), H7–H8 (memory), H10 (CPU cost).
> **Maths of every block:** doc 13 §12. **Text path:** doc 15. **Trainer:** doc 16.

---

## 1. Overview

```
 x_t (B,13,16,W) ─► conv_in ─► [L0: 16×16, 64ch]  Res ─ Res ─────────────────────────────────────────────── skip×3 ─┐
                                   │ downsample (conv s2)                                                            │
                               [L1: 8×8, 128ch]   Res─SA─CA ─ Res─SA─CA ──────────────────────────────── skip×3 ─┐   │
                                   │ downsample (conv s2)                                                        │   │
                               [L2: 4×4, 256ch]   Res─SA─CA ─ Res─SA─CA ──────────────────────────── skip×3 ─┐   │   │
                                   │                                                                         │   │   │
                               [mid: 4×4, 256ch]  Res ─ SA ─ CA ─ Res                                         │   │   │
                                   │                                                                         │   │   │
                               [dec L2: 4×4]  (cat)Res─SA─CA ×3  ◄──────────────────────────────────────────┘   │   │
                                   │ upsample (nearest×2 + conv)                                                 │   │
                               [dec L1: 8×8]  (cat)Res─SA─CA ×3  ◄──────────────────────────────────────────────┘   │
                                   │ upsample (nearest×2 + conv)                                                     │
                               [dec L0: 16×16](cat)Res ×3        ◄──────────────────────────────────────────────────┘
                                   │
                               head: GN → SiLU → conv3×3(64→13)  ─► ε̂ (B,13,16,W)

 t ─► sinusoidal(128) ─► Linear(128,256) ─ SiLU ─ Linear(256,256) = e(t) ─► every ResBlock (FiLM scale/shift)
 caption ─► frozen MiniLM (B,L,384) ─► adapter MLP + LayerNorm (B,L,128) = ctx ─► every CrossAttn (K,V) + key-padding mask
```

- **Resolutions:** 16×16 → 8×8 → 4×4 (two downsamplings, PDF Phase 3.2).
- **Channels:** 64 / 128 / 256 (base 64, multipliers (1, 2, 4), M3).
- **Blocks:** 2 ResBlocks per encoder level, 3 per decoder level (each decoder ResBlock consumes one skip), 2 in the bottleneck.
- **Attention:** self-attention (SA) and cross-attention (CA) after **every** ResBlock at 8×8 and 4×4, in the encoder, the bottleneck and the decoder. None at 16×16 (M5, M6).
- **Normalisation / activation:** GroupNorm(8 groups, ε = 1e-5), SiLU. **Dropout 0.1** inside ResBlocks (between the second activation and the second conv).

---

## 2. Building blocks (definitions + parameter formulas)

Let $C_{in}, C_{out}$ be channel counts, $E = 256$ the time-embedding width, $d_{ctx} = 128$ the context width.

### 2.1 `ResBlock(C_in, C_out)` (FiLM / AdaGN, M4)
```
h = conv1( SiLU( GN1(x) ) )                       # 3×3, C_in → C_out
γ, β = split( Linear_t( SiLU(e(t)) ), 2 )          # E → 2·C_out
h = GN2(h) * (1 + γ) + β
h = conv2( Dropout_0.1( SiLU(h) ) )                # 3×3, C_out → C_out   (ZERO-INIT, §6)
return h + skip(x)                                  # skip = 1×1 conv if C_in ≠ C_out else identity
```
Parameters: $2C_{in} + (9C_{in}C_{out}+C_{out}) + (2E\,C_{out}+2C_{out}) + 2C_{out} + (9C_{out}^2+C_{out}) + [C_{in}C_{out}+C_{out}]_{C_{in}\ne C_{out}}$.
Example 64→64: 128 + 36,928 + 32,896 + 128 + 36,928 = **107,008** ✓.

### 2.2 `SelfAttention(C)` (M5) — 4 heads
```
h = GN(x).flatten(2).transpose(1,2)               # (B, N=H·W, C)
q,k,v = Wq h, Wk h, Wv h                          # each C→C, with bias; reshape to (B, 4, N, C/4)
o = SDPA(q, k, v)                                 # softmax(q kᵀ/√(C/4)) v
return x + reshape( Wo o )                         # Wo: C→C (ZERO-INIT, §6)
```
Parameters: $2C + 4(C^2 + C) = 4C^2 + 6C$ → C = 128: **66,304**; C = 256: **263,680**.
**Heads:** fixed at **4 per layer**, so the head dimension is **32 at 8×8 and 64 at 4×4**. This matches the oracle. Ledger M5's wording "32-d heads" holds only at 8×8. The parameter count and FLOPs do not depend on the head split. (Reported as a ledger wording fix.)

### 2.3 `CrossAttention(C, d_ctx=128)` (M6) — 4 heads, key-padding mask
```
h = GN(x).flatten(2).transpose(1,2)               # (B, N, C) queries
q = Wq h ;  k = Wk ctx ;  v = Wv ctx              # Wq: C→C, Wk/Wv: 128→C ; ctx: (B, L, 128)
o = SDPA(q, k, v, attn_mask = mask[:,None,None,:])  # mask: True = valid token
return x + reshape( Wo o )                         # ZERO-INIT
```
Parameters: $2C + (C^2+C) + 2(d_{ctx}C + C) + (C^2+C) = 2C^2 + 262C$ (for $d_{ctx} = 128$) → C = 128: **66,304**; C = 256: **198,144**.

### 2.4 `Downsample(C)` — Conv3×3, stride 2, padding 1: $9C^2 + C$ → 64: 36,928; 128: 147,584.
### 2.5 `Upsample(C)` — nearest ×2 then Conv3×3 (padding 1): $9C^2 + C$ → 256: 590,080; 128: 147,584.
### 2.6 `conv_in` — Conv3×3, 13 → 64: 7,552. `head` — GN(64) → SiLU → Conv3×3 64 → 13 (ZERO-INIT): 128 + 7,501 = 7,629.
### 2.7 `time_mlp` — sinusoidal(128) → Linear(128, 256) → SiLU → Linear(256, 256): 33,024 + 65,792 = 98,816.

---

## 3. Exact layer table (default config; B = batch, L = caption tokens ≤ 48)

Verified by forward hooks on the oracle (2026-11-17). **Sum = 18,312,013.**

| # | Module (production name) | Op | Input | Output | Params |
|---|---|---|---|---|---|
| 0 | `time_mlp` | sinusoidal(128) → Linear → SiLU → Linear | (B,) int64 | (B,256) | 98,816 |
| 1 | `conv_in` | Conv3×3 | (B,13,16,16) | (B,64,16,16) | 7,552 |
| 2 | `enc.0.blocks.0.res` | ResBlock 64→64 | (B,64,16,16) | (B,64,16,16) | 107,008 |
| 3 | `enc.0.blocks.1.res` | ResBlock 64→64 | (B,64,16,16) | (B,64,16,16) | 107,008 |
| 4 | `enc.0.down` | Conv3×3 s2 | (B,64,16,16) | (B,64,8,8) | 36,928 |
| 5 | `enc.1.blocks.0.res` | ResBlock 64→128 | (B,64,8,8) | (B,128,8,8) | 295,936 |
| 6 | `enc.1.blocks.0.sa` | SelfAttn C=128 | (B,128,8,8) | (B,128,8,8) | 66,304 |
| 7 | `enc.1.blocks.0.ca` | CrossAttn C=128 | (B,128,8,8) + ctx (B,L,128) | (B,128,8,8) | 66,304 |
| 8 | `enc.1.blocks.1.res` | ResBlock 128→128 | (B,128,8,8) | (B,128,8,8) | 361,472 |
| 9 | `enc.1.blocks.1.sa` | SelfAttn C=128 | (B,128,8,8) | (B,128,8,8) | 66,304 |
| 10 | `enc.1.blocks.1.ca` | CrossAttn C=128 | (B,128,8,8) | (B,128,8,8) | 66,304 |
| 11 | `enc.1.down` | Conv3×3 s2 | (B,128,8,8) | (B,128,4,4) | 147,584 |
| 12 | `enc.2.blocks.0.res` | ResBlock 128→256 | (B,128,4,4) | (B,256,4,4) | 1,050,624 |
| 13 | `enc.2.blocks.0.sa` | SelfAttn C=256 | (B,256,4,4) | (B,256,4,4) | 263,680 |
| 14 | `enc.2.blocks.0.ca` | CrossAttn C=256 | (B,256,4,4) | (B,256,4,4) | 198,144 |
| 15 | `enc.2.blocks.1.res` | ResBlock 256→256 | (B,256,4,4) | (B,256,4,4) | 1,312,768 |
| 16 | `enc.2.blocks.1.sa` | SelfAttn C=256 | (B,256,4,4) | (B,256,4,4) | 263,680 |
| 17 | `enc.2.blocks.1.ca` | CrossAttn C=256 | (B,256,4,4) | (B,256,4,4) | 198,144 |
| 18 | `mid.res0` | ResBlock 256→256 | (B,256,4,4) | (B,256,4,4) | 1,312,768 |
| 19 | `mid.sa` | SelfAttn C=256 | (B,256,4,4) | (B,256,4,4) | 263,680 |
| 20 | `mid.ca` | CrossAttn C=256 | (B,256,4,4) | (B,256,4,4) | 198,144 |
| 21 | `mid.res1` | ResBlock 256→256 | (B,256,4,4) | (B,256,4,4) | 1,312,768 |
| 22 | `dec.2.blocks.0.res` | ResBlock (256+256)→256 | (B,512,4,4) | (B,256,4,4) | 2,034,432 |
| 23 | `dec.2.blocks.0.sa` | SelfAttn C=256 | (B,256,4,4) | (B,256,4,4) | 263,680 |
| 24 | `dec.2.blocks.0.ca` | CrossAttn C=256 | (B,256,4,4) | (B,256,4,4) | 198,144 |
| 25 | `dec.2.blocks.1.res` | ResBlock (256+256)→256 | (B,512,4,4) | (B,256,4,4) | 2,034,432 |
| 26 | `dec.2.blocks.1.sa` | SelfAttn C=256 | (B,256,4,4) | (B,256,4,4) | 263,680 |
| 27 | `dec.2.blocks.1.ca` | CrossAttn C=256 | (B,256,4,4) | (B,256,4,4) | 198,144 |
| 28 | `dec.2.blocks.2.res` | ResBlock (256+128)→256 | (B,384,4,4) | (B,256,4,4) | 1,706,496 |
| 29 | `dec.2.blocks.2.sa` | SelfAttn C=256 | (B,256,4,4) | (B,256,4,4) | 263,680 |
| 30 | `dec.2.blocks.2.ca` | CrossAttn C=256 | (B,256,4,4) | (B,256,4,4) | 198,144 |
| 31 | `dec.2.up` | Nearest×2 + Conv3×3 | (B,256,4,4) → (B,256,8,8) | (B,256,8,8) | 590,080 |
| 32 | `dec.1.blocks.0.res` | ResBlock (256+128)→128 | (B,384,8,8) | (B,128,8,8) | 706,176 |
| 33 | `dec.1.blocks.0.sa` | SelfAttn C=128 | (B,128,8,8) | (B,128,8,8) | 66,304 |
| 34 | `dec.1.blocks.0.ca` | CrossAttn C=128 | (B,128,8,8) | (B,128,8,8) | 66,304 |
| 35 | `dec.1.blocks.1.res` | ResBlock (128+128)→128 | (B,256,8,8) | (B,128,8,8) | 542,080 |
| 36 | `dec.1.blocks.1.sa` | SelfAttn C=128 | (B,128,8,8) | (B,128,8,8) | 66,304 |
| 37 | `dec.1.blocks.1.ca` | CrossAttn C=128 | (B,128,8,8) | (B,128,8,8) | 66,304 |
| 38 | `dec.1.blocks.2.res` | ResBlock (128+64)→128 | (B,192,8,8) | (B,128,8,8) | 460,032 |
| 39 | `dec.1.blocks.2.sa` | SelfAttn C=128 | (B,128,8,8) | (B,128,8,8) | 66,304 |
| 40 | `dec.1.blocks.2.ca` | CrossAttn C=128 | (B,128,8,8) | (B,128,8,8) | 66,304 |
| 41 | `dec.1.up` | Nearest×2 + Conv3×3 | (B,128,8,8) → (B,128,16,16) | (B,128,16,16) | 147,584 |
| 42 | `dec.0.blocks.0.res` | ResBlock (128+64)→64 | (B,192,16,16) | (B,64,16,16) | 193,344 |
| 43 | `dec.0.blocks.1.res` | ResBlock (64+64)→64 | (B,128,16,16) | (B,64,16,16) | 152,256 |
| 44 | `dec.0.blocks.2.res` | ResBlock (64+64)→64 | (B,128,16,16) | (B,64,16,16) | 152,256 |
| 45 | `head` | GN → SiLU → Conv3×3 (64→13) | (B,64,16,16) | (B,13,16,16) | 7,629 |
| | **Total U-Net** | | | | **18,312,013** |

**Stage sums:** time-MLP 98,816 · conv_in 7,552 · encoder (rows 2–17) 4,608,192 · bottleneck 3,087,360 · decoder (rows 22–44) 10,502,464 · head 7,629.
*Ledger note:* H8's per-stage list omits `conv_in` (7,552), so its stages sum to 18,304,461. The **total** 18,312,013 is correct (reported for a ledger fix).

**Other trainable modules (not in the U-Net):** text adapter Linear(384,128) → SiLU → Linear(128,128) → LayerNorm(128) = **66,048** (doc 15). Conditional model trainable total = **18,378,061**. Frozen MiniLM = 22,713,216 (22,565,376 without its unused pooler).

**Attention inventory:** 11 SA blocks (5 at 8×8, 6 at 4×4) = **1,913,600** params; 11 CA blocks = **1,520,384** params.

---

## 4. Skip-connection bookkeeping (the classic U-Net bug source)

The encoder pushes **9** tensors onto a stack; the decoder pops them in reverse, one per decoder ResBlock (3 levels × 3 blocks = 9).

| Push order | Source | Shape | Popped by | Concat input → output |
|---|---|---|---|---|
| s0 | `conv_in` | (64,16,16) | `dec.0.blocks.2.res` | 64+64 = 128 → 64 |
| s1 | `enc.0.blocks.0.res` | (64,16,16) | `dec.0.blocks.1.res` | 64+64 = 128 → 64 |
| s2 | `enc.0.blocks.1.res` | (64,16,16) | `dec.0.blocks.0.res` | 128+64 = 192 → 64 |
| s3 | `enc.0.down` | (64,8,8) | `dec.1.blocks.2.res` | 128+64 = 192 → 128 |
| s4 | `enc.1.blocks.0.res` (+SA,CA) | (128,8,8) | `dec.1.blocks.1.res` | 128+128 = 256 → 128 |
| s5 | `enc.1.blocks.1.res` (+SA,CA) | (128,8,8) | `dec.1.blocks.0.res` | 256+128 = 384 → 128 |
| s6 | `enc.1.down` | (128,4,4) | `dec.2.blocks.2.res` | 256+128 = 384 → 256 |
| s7 | `enc.2.blocks.0.res` (+SA,CA) | (256,4,4) | `dec.2.blocks.1.res` | 256+256 = 512 → 256 |
| s8 | `enc.2.blocks.1.res` (+SA,CA) | (256,4,4) | `dec.2.blocks.0.res` | 256+256 = 512 → 256 |

Rules: (1) the skip pushed after an attended level is the output **after** its SA and CA; (2) concatenation is `torch.cat([h, skip], dim=1)` (decoder features first, then skip); this order is part of the state-dict contract, because it determines the conv weight layout; (3) after the forward pass the stack must be empty (`assert not skips`).

---

## 5. Width-agnostic property (needed for outpainting and the direct-wide ablation)

- The network is fully convolutional plus attention, with **no positional embeddings**, so any **H, W that are multiples of 4** work. Two stride-2 convs with padding 1 map W → ⌈W/2⌉ → ⌈W/4⌉, and nearest ×2 upsampling only restores the skip shapes when W ≡ 0 (mod 4). Example: W = 18 → 9 → 5 → up to 10 ≠ 9, so the concat fails. `UNet.forward` asserts `H % 4 == 0 and W % 4 == 0`.
- Verified forward FLOPs at H = 16: W = 16 → 1.241 GFLOP; W = 24 → 1.843 GFLOP; W = 112 → 8.856 GFLOP (L = 48 tokens).
- Production generation uses 16×16 windows (G1). Wider inputs are used by the **direct wide sampling** fallback (doc 25) and by tests.
- Positional information comes only from zero padding at the borders, which is how the model knows where row 15 (the floor row) is. This works because H is always exactly 16. Coordinate channels (CoordConv) are a documented, not-adopted option.

**Receptive field.** Conv-only, along one axis: **189 tiles** (computed). Self-attention at 8×8 already makes it **global**: every output tile can depend on every input tile, even at W = 112.

---

## 6. Initialisation (decision)

**Decision:** ADM-style zero-initialisation of every residual branch's last layer plus the output head:
1. `ResBlock.conv2` weight and bias = 0 (22 ResBlocks)
2. `SelfAttention.Wo`, `CrossAttention.Wo` weight and bias = 0 (22 attention blocks)
3. `head` conv weight and bias = 0

Everything else keeps PyTorch defaults (Kaiming-uniform for conv/linear, GroupNorm/LayerNorm γ = 1, β = 0). The FiLM linear (`Linear_t`) keeps default init; with the $(1+\gamma)$ form, small random $\gamma$ is near identity. The learned null context (§7) is initialised $\mathcal N(0, 0.02^2)$.

**Why:**
- Every residual block starts as the identity, so the initial network is a short, well-conditioned path. Early training is stable at lr 2e-4 without special warm-up tricks beyond the 1,000-step LR warm-up (M10).
- $\epsilon_\theta \equiv 0$ at step 0 ⇒ **initial loss = E[ε²] = 1**, a free sanity check. Verified: **0.997** (B = 64). With default init the initial loss is 1.096 and less predictable.
- Parameter count is unchanged: **18,312,013** (verified).

**Consequences for tests (important):**
- At step 0 only the head conv receives a gradient (verified: 2 of 430 parameter tensors). By step 1, 114 do; **by step 2, all 430 do** (verified with AdamW lr 2e-4). Test U5 therefore checks "every parameter has had a non-zero gradient at least once **within 3 optimizer steps**", not at step 0.
- At init, the output does not depend on t or on the caption, because it is identically 0. Sensitivity tests (U6, U7) run with `zero_init: false` or after ≥ 50 training steps.
- The oracle parity test (U2) copies weights positionally into both models, so init does not matter there.

---

## 7. Variants

| Variant | Config | Params | Use |
|---|---|---|---|
| **Conditional (default)** | `cross_attention: true`, `text.conditioning: true` | 18,312,013 + adapter 66,048 | Phase 4+, all app use |
| **Unconditional (Phase 3)** | `cross_attention: true`, `text.conditioning: false` | **18,312,141** (adds a learned `null_ctx` ∈ ℝ^{1×1×128}) | Phase 3 baseline. Same module tree as the conditional model, so the Phase-3 checkpoint can initialise Phase 4 if ever needed (`train.init_from`, off by default). Each CA layer attends to a single learned token, which reduces it to a learned per-layer bias. |
| No cross-attention | `cross_attention: false` | 16,791,629 | debugging only |
| X8: no self-attention | `self_attention: false` | **16,398,413** | tests M5's claim that SA reduces broken pipes |
| X12: half width | `base_channels: 32` | **5,160,621** | capacity / bias–variance probe (doc 19) |
| X3: pooled text → time | `pooled_text_to_time: true` | +33,024 (Linear(128,256), zero-init) | adds a global caption vector to e(t) |
| (reference only) base 96 | `base_channels: 96` | 39,553,005 | not planned; listed for capacity context |

**Null context semantics (M7):** in the *conditional* model the null condition is the embedding of `""` (doc 15). The learned `null_ctx` exists **only** in the unconditional variant. If a conditional model is called with `ctx=None`, it raises (no silent fallback, CLAUDE.md rule 12).

---

## 8. Compute and memory (verified estimates; replaced by measured `hw_report.json` on Day 4)

| Quantity | Value | Source |
|---|---|---|
| Forward FLOPs, 16×16, L = 48 | **1.241 GFLOP / sample** (conv 1.022, linear 0.195, attention matmuls 0.025) | `FlopCounterMode` on oracle |
| Forward FLOPs, 16×16, L = 16 | 1.197 GFLOP / sample | idem |
| Training FLOPs (≈ 3× fwd) | ≈ 3.7 GFLOP / sample → **≈ 238 GFLOP / step** at B = 64 | |
| Text encoder (frozen MiniLM) | 66.6 GFLOP per batch of 64 at L = 48; 21.9 at L = 16 (only on cache misses; doc 15 §5) | idem |
| Guided DDIM step (cond + uncond) | 2.48 GFLOP / sample | |
| One 112-wide level (13 windows × 110 guided passes, G1) | ≈ 3.5 TFLOP | |
| Weights (fp32) | 73.2 MB | |
| Train state: weights + EMA + grads + Adam m, v (fp32) | ≈ 366 MB | |
| Saved activations for backward, B = 64 | **714 MiB fp32**; ≈ 373 MiB under bf16 autocast (CPU-autocast proxy) | `saved_tensors_hooks` byte count |
| Saved activations, B = 32, fp32 | 391 MiB | |
| Expected CUDA peak, B = 64 | ≈ 1.0–1.5 GiB allocated (+ CUDA context) | far under the 4.5 GiB cap (H2) |
| CPU guided forward (16×16, 4 threads) | ≈ 75 ms | H10 |

Throughput is **not** stated here. It depends on kernels and drivers and is measured on Day 4 for both GPUs (`hw_report.json`: median it/s over steps 50–200).

---

## 9. Module and API contract (`src/text2mario/models/`)

```
models/
  blocks.py    SinusoidalTimeEmbedding, TimeMLP, ResBlock, SelfAttention, CrossAttention, Downsample, Upsample,
               Block (= one ResBlock + optional SA + optional CA, registered in that order)
  unet.py      UNet (the tree in §3; attribute names exactly: time_mlp, conv_in, enc, mid, dec, head, [null_ctx], [pooled_proj])
  factory.py   build_unet(cfg.model, conditional: bool) -> UNet ; count_params(model) -> dict
```

**Module tree (registration order = parameter order = oracle order):**
```
UNet
├─ time_mlp            Sequential(Linear(128,256), SiLU, Linear(256,256))
├─ conv_in             Conv2d(13,64,3,p=1)
├─ enc                 ModuleList[ Level0, Level1, Level2 ]
│   └─ Level l         blocks: ModuleList[Block × 2], down: Downsample (levels 0,1 only)
│       └─ Block       res: ResBlock, sa: SelfAttention (l ≥ 1), ca: CrossAttention (l ≥ 1)
├─ mid                 res0: ResBlock, sa, ca, res1: ResBlock            (registered in this order)
├─ dec                 ModuleDict with INSERTION ORDER "2", "1", "0"  (decoder forward order)
│   └─ Level l         blocks: ModuleList[Block × 3], up: Upsample (levels 2,1 only; registered AFTER blocks)
├─ head                Sequential(GroupNorm(8,64), SiLU, Conv2d(64,13,3,p=1))
├─ null_ctx            Parameter (1,1,128) — unconditional variant only, registered LAST
└─ pooled_proj         Linear(128,256) — X3 only, registered LAST
```
Interleaving `res → sa → ca` inside each `Block` is **required**. Separate per-level lists (`res[]`, `sa[]`, `ca[]`) would reorder the parameters and break the positional parity test U2.

```python
class UNet(nn.Module):
    def forward(self,
                x: torch.Tensor,            # (B,13,H,W) float, H%4==0, W%4==0
                t: torch.Tensor,            # (B,) int64 code indices 0..999
                ctx: torch.Tensor | None,   # (B,L,128) adapter output; None only for the unconditional variant
                mask: torch.Tensor | None,  # (B,L) bool, True = valid token
                *, return_attn: bool = False):
        ...  # returns eps_hat (B,13,H,W); if return_attn: (eps_hat, {layer_name: (B,heads,N,L) cross-attn probs})
```
- `return_attn=True` exists for the **future word-highlighting feature** (doc 31) and the Day-7 attention-entropy diagnostic. It must not change outputs, and it is off in training (SDPA fast path).
- **Parameter creation order must match the oracle** (`time_mlp`, `conv_in`, encoder in forward order, mid, decoder in forward order, head; inside a ResBlock: n1, c1, t, n2, c2, skip; inside attention: n, q, k, v, o). Test U2 depends on it.
- **State-dict keys are a contract.** A golden list `tests/golden/unet_state_dict_keys.txt` is committed. Test U10 fails on any rename, because renames break every checkpoint.
- `ctx` and `mask` for the SA blocks are ignored. CA blocks receive the same `ctx` and `mask` at every layer (no per-layer text projection beyond W_k, W_v).

---

## 10. Configuration knobs (`cfg.model`, full YAML in doc 17)

| Key | Default | Allowed | Effect |
|---|---|---|---|
| `in_channels` | 13 | 13 (fixed by F6) | tiles |
| `base_channels` | 64 | {32, 64, 96} | width (X12 = 32) |
| `channel_mult` | [1, 2, 4] | fixed | 16→8→4 hierarchy |
| `num_res_blocks_down` | 2 | {1, 2, 3} | encoder depth |
| `num_res_blocks_up` | 3 | = down + 1 | must equal skips per level |
| `groups` | 8 | divides every channel count | GroupNorm |
| `dropout` | 0.1 | [0.0, 0.3] | ResBlock dropout (regulariser, doc 19) |
| `attn_levels` | [1, 2] | subsets of {0,1,2} | where SA/CA sit (level 0 = 16×16) |
| `self_attention` | true | bool | X8 = false |
| `cross_attention` | true | bool | debugging only |
| `heads` | 4 | divides C at each attended level | |
| `time_embed_dim` / `time_mlp_dim` | 128 / 256 | fixed | M4 |
| `context_dim` | 128 | fixed | M6/M7 |
| `zero_init` | true | bool | §6 |
| `pooled_text_to_time` | false | bool | X3 |
| `null_ctx_tokens` | 1 | ≥ 1 | unconditional variant only |

Changing any key changes the config hash (doc 17 §1.5) and therefore the run identity.

---

## 11. Tests (`tests/test_unet.py`; doc 24 lists all suites)

| ID | Test | Pass condition |
|---|---|---|
| U1 | parameter counts | default **18,312,013**; uncond **18,312,141**; adapter **66,048**; X8 **16,398,413**; X12 **5,160,621**; X3 adds **33,024** |
| U2 | oracle parity | copy `reference/ref_unet.py` weights positionally (`zip(ref.parameters(), ours.parameters())`, shapes equal), `eval()`, fp32 → max abs diff < 1e-5 on random x, t, ctx, mask |
| U3 | shapes | output == input shape for W ∈ {16, 24, 112}; W = 18 raises `AssertionError` |
| U4 | initial loss | zero_init model, B = 64, random one-hot x₀ → ε-MSE ∈ [0.95, 1.05] |
| U5 | gradient flow | within 3 AdamW steps, every parameter tensor has had a non-zero gradient |
| U6 | time sensitivity | (`zero_init: false`) ‖f(x,t=10) − f(x,t=900)‖ > 1e-3 |
| U7 | mask / pad invariance | changing ctx values at masked positions changes output by < 1e-5 (fp32) |
| U8 | eval determinism | `eval()`: two identical forwards are bit-identical (dropout off) |
| U9 | bf16 closeness (CUDA only) | relative L2 error of autocast-bf16 vs fp32 output < 2e-2 |
| U10 | state-dict golden keys | key list equals `tests/golden/unet_state_dict_keys.txt` |
| U11 | return_attn | shapes (B, 4, N, L); probabilities sum to 1 over valid tokens; outputs unchanged vs `return_attn=False` |
| U12 (slow) | FLOP budget | FlopCounterMode within ±2 % of 1.241 GFLOP (16×16, L = 48) |
| U13 (slow, GPU) | memory | B = 64 train step peak allocated ≤ 2.0 GiB (H8 expectation; hard cap 4.5 GiB) |
