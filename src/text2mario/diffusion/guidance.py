"""Classifier-free guidance (ledger M13, doc 13 §8, doc 15 §7).

Explain-me:
  What: eps_hat = eps_uncond + w * (eps_cond - eps_uncond).  w = 1 is the plain conditional model, w = 0 the
        unconditional one, w > 1 pushes harder towards the caption.  Both predictions come from ONE network call
        on a doubled batch (cond first, null second).
  In/Out: model, x_t (B,13,H,W), t (B,), contexts + masks -> guided eps (B,13,H,W), float32.
  Try: sweep w in {1, 3, 7.5} on one prompt and watch broken structures appear as w grows (drill 1).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _pad_len(x: torch.Tensor, L: int, value=0) -> torch.Tensor:
    """Pad dim 1 of ctx (B,L,C) or mask (B,L) up to length L."""
    extra = L - x.shape[1]
    if extra == 0:
        return x
    pad = (0, 0, 0, extra) if x.dim() == 3 else (0, extra)
    return F.pad(x, pad, value=value)


def guided_eps(model, x_t, t, ctx_c, mask_c, ctx_u, mask_u, w: float, autocast_dtype: torch.dtype | None = None):
    L = max(ctx_c.shape[1], ctx_u.shape[1])                       # pad both contexts to a common length
    ctx = torch.cat([_pad_len(ctx_c, L), _pad_len(ctx_u, L)])      # (2B, L, C)
    mask = torch.cat([_pad_len(mask_c, L, False), _pad_len(mask_u, L, False)])   # padding = False
    xx, tt = torch.cat([x_t, x_t]), torch.cat([t, t])
    if autocast_dtype is not None:
        with torch.autocast(x_t.device.type, dtype=autocast_dtype):
            out = model(xx, tt, ctx, mask)
    else:
        out = model(xx, tt, ctx, mask)
    eps_c, eps_u = out.float().chunk(2)
    return eps_u + w * (eps_c - eps_u)
