"""Guided DDIM sampling with an optional known-region (RePaint) hook (doc 13 §7, §9; doc 20).

Explain-me:
  What: starts from pure noise and runs S DDIM steps; at each step the network predicts the noise (with
        classifier-free guidance if a caption is given), x0-hat is clipped to [-1, 1], and x moves to the next step.
        With `known=(x0_known, mask)` the known columns are re-noised to the current level and pasted back at every
        step; `resample_r` extra jumps back-and-forth in the first 60 % of steps smooth the seams (RePaint).
  In/Out: returns the final x0-hat (B,13,H,W) float32 (decode with renderer.tensor_to_grid) and a stats dict
          (clip_rate = share of x0-hat elements that clipping changed, averaged over steps).
  Previous: the trained U-Net + scheduler.  Next: eval (metrics on decoded grids), generate/full_level (stitching).
  Try: steps=25, resample_r=0 is the CPU "fast mode" (H10); w=1 disables guidance (drill 7, drill 8).
"""
from __future__ import annotations

import torch

from text2mario.diffusion.guidance import guided_eps


@torch.no_grad()
def ddim_sample(model, schedule, shape, *, ctx_c=None, mask_c=None, ctx_u=None, mask_u=None, w: float = 1.0,
                steps: int = 50, eta: float = 0.0, generator: torch.Generator | None = None, clip: bool = True,
                known: tuple[torch.Tensor, torch.Tensor] | None = None, resample_r: int = 0,
                resample_fraction: float = 0.6, autocast_dtype: torch.dtype | None = None, device="cpu"):
    """shape = (B, 13, H, W).  Noise is drawn on CPU from `generator` (portable) and moved to `device`."""
    B = shape[0]
    x = torch.randn(shape, generator=generator).to(device)
    ts = schedule.ddim_timesteps(steps).to(device)
    clipped, n_obs = 0.0, 0
    if known is not None:
        x0_kn, m = known
        x0_kn, m = x0_kn.to(device, x.dtype), m.to(device, x.dtype)

    def predict(x_t, t):
        tt = t.repeat(B)
        if ctx_c is None:                                  # unconditional model: no guidance possible
            out = model(x_t, tt)
            return out.float()
        if w == 1.0:                                       # plain conditional: a single pass is enough
            return _forward(model, x_t, tt, ctx_c, mask_c, autocast_dtype)
        return guided_eps(model, x_t, tt, ctx_c, mask_c, ctx_u, mask_u, w, autocast_dtype)

    for i, t in enumerate(ts):
        s = ts[i + 1] if i + 1 < len(ts) else torch.tensor(-1, device=device)
        reps = 1 + (resample_r if (known is not None and i < resample_fraction * len(ts) and s >= 0) else 0)
        for rep in range(reps):
            if known is not None:                          # known region: fresh noise at the current level, then merge
                z = torch.randn(shape, generator=generator).to(device)
                x = m * schedule.q_sample(x0_kn, t.repeat(B), z) + (1 - m) * x
            eps = predict(x, t)
            x0_hat = schedule.predict_x0_from_eps(x, t.repeat(B), eps)
            clipped += (x0_hat.abs() > 1).float().mean().item()
            n_obs += 1
            x_s = schedule.ddim_step(eps, t.repeat(B), s.repeat(B), x, eta, generator, clip)
            x = schedule.renoise(x_s, s.repeat(B), t.repeat(B), generator) if rep < reps - 1 else x_s
    if known is not None:
        x = m * x0_kn + (1 - m) * x                        # the known region is returned exactly
    return x, {"clip_rate": clipped / max(n_obs, 1), "steps": steps}


def _forward(model, x_t, t, ctx, mask, autocast_dtype):
    if autocast_dtype is not None:
        with torch.autocast(x_t.device.type, dtype=autocast_dtype):
            return model(x_t, t, ctx, mask).float()
    return model(x_t, t, ctx, mask).float()
