"""U-Net building blocks (doc 14 §2): ResBlock (FiLM), self/cross attention, down/up-sampling, time embedding.

Explain-me:
  What: the small pieces the U-Net is assembled from.  ResBlock = conv, norm, add the timestep as scale+shift
        (FiLM/AdaGN), conv, with a skip connection.  Attention blocks let every tile look at all other tiles
        (self) or at the caption's word vectors (cross).
  In/Out: feature maps (B, C, H, W); the time vector e(t) (B, 256); caption context (B, L, 128) plus a bool mask.
  Order of parameters matters: tests/test_unet.py copies weights positionally from reference/ref_unet.py.
  Try: change `heads` or `dropout` in the model config and print the parameter count (drill 11).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_embedding(t: torch.Tensor, dim: int = 128) -> torch.Tensor:
    """Fixed sin/cos features of the integer step index (doc 13 §12.1). Feed the real code index 0..999."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
    a = t.float()[:, None] * freqs[None]
    return torch.cat([a.sin(), a.cos()], dim=1)


class TimeMLP(nn.Module):
    """sinusoidal(128) -> Linear(128,256) -> SiLU -> Linear(256,256) = e(t)."""

    def __init__(self, dim: int = 128, out: int = 256):
        super().__init__()
        self.dim = dim
        self.net = nn.Sequential(nn.Linear(dim, out), nn.SiLU(), nn.Linear(out, out))

    def forward(self, t):
        return self.net(sinusoidal_embedding(t, self.dim))


class ResBlock(nn.Module):
    def __init__(self, cin: int, cout: int, temb: int = 256, groups: int = 8, dropout: float = 0.1):
        super().__init__()
        # registration order = parameter order = oracle order (n1, c1, t, n2, c2, skip)
        self.n1 = nn.GroupNorm(groups, cin)
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.t = nn.Linear(temb, 2 * cout)
        self.n2 = nn.GroupNorm(groups, cout)
        self.drop = nn.Dropout(dropout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, te):
        h = self.c1(F.silu(self.n1(x)))
        scale, shift = self.t(F.silu(te))[:, :, None, None].chunk(2, 1)   # FiLM: gamma, beta from the timestep
        h = self.n2(h) * (1 + scale) + shift
        h = self.c2(self.drop(F.silu(h)))
        return h + self.skip(x)


class Attention(nn.Module):
    """Shared implementation: self-attention when ctx_dim is None, cross-attention otherwise (4 heads)."""

    def __init__(self, c: int, ctx_dim: int | None = None, heads: int = 4, groups: int = 8):
        super().__init__()
        self.n = nn.GroupNorm(groups, c)
        self.heads = heads
        self.q = nn.Linear(c, c)
        self.k = nn.Linear(ctx_dim or c, c)
        self.v = nn.Linear(ctx_dim or c, c)
        self.o = nn.Linear(c, c)
        self.is_cross = ctx_dim is not None

    def forward(self, x, ctx=None, mask=None, return_attn: bool = False):
        B, C, H, W = x.shape
        h = self.n(x).flatten(2).transpose(1, 2)                    # (B, N=H*W, C) queries
        kv = h if ctx is None else ctx
        d = C // self.heads
        q = self.q(h).view(B, -1, self.heads, d).transpose(1, 2)    # (B, heads, N, d)
        k = self.k(kv).view(B, -1, self.heads, d).transpose(1, 2)
        v = self.v(kv).view(B, -1, self.heads, d).transpose(1, 2)
        am = None if mask is None else mask[:, None, None, :]       # True = attend (key padding mask)
        probs = None
        if return_attn:                                              # explicit softmax so the map can be returned
            logits = q @ k.transpose(-1, -2) / math.sqrt(d)
            if am is not None:
                logits = logits.masked_fill(~am, float("-inf"))
            probs = logits.softmax(-1)
            o = probs @ v
        else:
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=am)
        o = self.o(o.transpose(1, 2).reshape(B, H * W, C)).transpose(1, 2).view(B, C, H, W)
        return (x + o, probs) if return_attn else x + o


class Downsample(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class Block(nn.Module):
    """One ResBlock plus optional self-attention and cross-attention, registered in that order."""

    def __init__(self, cin, cout, attn: bool, self_attention: bool, cross_attention: bool, ctx_dim: int,
                 heads: int, groups: int, dropout: float):
        super().__init__()
        self.res = ResBlock(cin, cout, groups=groups, dropout=dropout)
        if attn and self_attention:
            self.sa = Attention(cout, None, heads, groups)
        if attn and cross_attention:
            self.ca = Attention(cout, ctx_dim, heads, groups)

    def forward(self, x, te, ctx, mask, attn_log=None, name=""):
        h = self.res(x, te)
        if hasattr(self, "sa"):
            h = self.sa(h)
        if hasattr(self, "ca"):
            if attn_log is not None:
                h, p = self.ca(h, ctx, mask, return_attn=True)
                attn_log[name] = p
            else:
                h = self.ca(h, ctx, mask)
        return h
