"""The conditional 2-stage U-Net (doc 14): 16x16 -> 8x8 -> 4x4, 64/128/256 channels, FiLM time, self+cross attention.

Explain-me:
  What: predicts the noise eps in a noisy level x_t, given the step t and (optionally) the caption's word vectors.
        Encoder halves the resolution twice, the decoder restores it using skip connections; attention sits at 8x8 and 4x4.
  In/Out: forward(x (B,13,H,W), t (B,) int, ctx (B,L,128) | None, mask (B,L) bool) -> eps_hat (B,13,H,W); H, W multiples of 4.
  Previous: scheduler.q_sample gives x_t; text/conditioner gives ctx.  Next: losses.loss_simple.
  Contract: 18,312,013 parameters by default; parameter creation order and state-dict keys match the oracle
            (tests/test_unet.py), so never rename a submodule.
  Try: UNet(base=32) has 5,160,621 parameters (X12); set zero_init=False to see a non-zero initial output.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from text2mario.models.blocks import Block, Downsample, ResBlock, TimeMLP, Upsample, Attention


class Level(nn.Module):
    def __init__(self, blocks: list[Block], sampler: nn.Module | None, key: str):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        if sampler is not None:
            setattr(self, key, sampler)   # "down" for the encoder, "up" for the decoder (registered AFTER blocks)


class Mid(nn.Module):
    def __init__(self, c, self_attention, cross_attention, ctx_dim, heads, groups, dropout):
        super().__init__()
        self.res0 = ResBlock(c, c, groups=groups, dropout=dropout)
        if self_attention:
            self.sa = Attention(c, None, heads, groups)
        if cross_attention:
            self.ca = Attention(c, ctx_dim, heads, groups)
        self.res1 = ResBlock(c, c, groups=groups, dropout=dropout)


class UNet(nn.Module):
    def __init__(self, in_channels: int = 13, base_channels: int = 64, channel_mult=(1, 2, 4), num_res_blocks_down: int = 2,
                 groups: int = 8, dropout: float = 0.1, attn_levels=(1, 2), self_attention: bool = True,
                 cross_attention: bool = True, heads: int = 4, context_dim: int = 128, zero_init: bool = True,
                 pooled_text_to_time: bool = False, uncond: bool = False, null_ctx_tokens: int = 1):
        super().__init__()
        self.uncond, self.cross_attention, self.pooled = uncond, cross_attention, pooled_text_to_time
        n_up = num_res_blocks_down + 1
        ch = [base_channels * m for m in channel_mult]
        kw = dict(self_attention=self_attention, cross_attention=cross_attention, ctx_dim=context_dim, heads=heads,
                  groups=groups, dropout=dropout)
        self.time_mlp = TimeMLP(128, 256)
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        enc, skip_ch, c = [], [base_channels], base_channels
        for i, co in enumerate(ch):
            blocks = []
            for _ in range(num_res_blocks_down):
                blocks.append(Block(c, co, i in attn_levels, **kw))
                c = co
                skip_ch.append(c)
            down = Downsample(c) if i < len(ch) - 1 else None
            if down is not None:
                skip_ch.append(c)
            enc.append(Level(blocks, down, "down"))
        self.enc = nn.ModuleList(enc)
        self.mid = Mid(c, self_attention, cross_attention, context_dim, heads, groups, dropout)
        self.dec = nn.ModuleDict()
        for i in reversed(range(len(ch))):
            blocks = []
            for _ in range(n_up):
                blocks.append(Block(c + skip_ch.pop(), ch[i], i in attn_levels, **kw))
                c = ch[i]
            self.dec[str(i)] = Level(blocks, Upsample(c) if i > 0 else None, "up")
        assert not skip_ch
        self.head = nn.Sequential(nn.GroupNorm(groups, c), nn.SiLU(), nn.Conv2d(c, in_channels, 3, padding=1))
        if uncond:                                         # Phase 3: a learned "null" caption instead of text
            self.null_ctx = nn.Parameter(torch.randn(1, null_ctx_tokens, context_dim) * 0.02)
        if pooled_text_to_time:                            # experiment X3
            self.pooled_proj = nn.Linear(context_dim, 256)
        if zero_init:
            self._zero_init()

    def _zero_init(self):
        """ADM-style: every residual branch starts as identity and eps_hat == 0 at step 0 (initial loss ~ 1)."""
        for m in self.modules():
            if isinstance(m, ResBlock):
                nn.init.zeros_(m.c2.weight), nn.init.zeros_(m.c2.bias)
            elif isinstance(m, Attention):
                nn.init.zeros_(m.o.weight), nn.init.zeros_(m.o.bias)
        nn.init.zeros_(self.head[-1].weight), nn.init.zeros_(self.head[-1].bias)
        if self.pooled:
            nn.init.zeros_(self.pooled_proj.weight), nn.init.zeros_(self.pooled_proj.bias)

    def forward(self, x, t, ctx=None, mask=None, *, return_attn: bool = False):
        B, _, H, W = x.shape
        assert H % 4 == 0 and W % 4 == 0, "H and W must be multiples of 4 (two stride-2 downsamplings)"
        if self.uncond:
            ctx = self.null_ctx.to(x.dtype).expand(B, -1, -1)
            mask = torch.ones(B, ctx.shape[1], dtype=torch.bool, device=x.device)
        elif ctx is None and self.cross_attention:
            raise ValueError("a conditional model needs ctx (use the null caption embedding, never None)")
        te = self.time_mlp(t)
        if self.pooled and ctx is not None:
            m = mask.unsqueeze(-1).to(ctx.dtype)
            te = te + self.pooled_proj((ctx * m).sum(1) / m.sum(1).clamp(min=1))
        log = {} if return_attn else None
        h = self.conv_in(x)
        skips = [h]
        for li, lvl in enumerate(self.enc):
            for bi, blk in enumerate(lvl.blocks):
                h = blk(h, te, ctx, mask, log, f"enc.{li}.{bi}")
                skips.append(h)
            if hasattr(lvl, "down"):
                h = lvl.down(h)
                skips.append(h)
        m = self.mid
        h = m.res0(h, te)
        if hasattr(m, "sa"):
            h = m.sa(h)
        if hasattr(m, "ca"):
            if log is not None:
                h, log["mid"] = m.ca(h, ctx, mask, return_attn=True)
            else:
                h = m.ca(h, ctx, mask)
        h = m.res1(h, te)
        for key, lvl in self.dec.items():
            for bi, blk in enumerate(lvl.blocks):
                h = blk(torch.cat([h, skips.pop()], dim=1), te, ctx, mask, log, f"dec.{key}.{bi}")
            if hasattr(lvl, "up"):
                h = lvl.up(h)
        assert not skips, "skip stack must be empty"
        out = self.head(h)
        return (out, log) if return_attn else out
