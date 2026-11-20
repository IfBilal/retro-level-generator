"""Load a checkpoint's EMA weights and draw samples (doc 22 §2.3, CLAUDE.md rule 7: evaluate with EMA).

Explain-me:
  What: rebuilds the U-Net from the config stored inside the checkpoint, swaps in the EMA weights, and samples
        levels with DDIM.  Every sample has its OWN seed (seed_base + 1000 * index), so batching or the machine's
        batch size never changes which noise a sample starts from.
  In/Out: load_ema_model(path) -> (unet, schedule, cfg);  sample_unconditional(...) -> uint8 grids (n, 16, 16).
  Previous: train/loop.py wrote the checkpoint.  Next: eval/structure.py scores the grids, scripts/gate_p3.py decides.
  Try: steps=25 (CPU "fast mode") vs 50 and compare the structure metrics (drill 7).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from text2mario.diffusion.sampler import ddim_sample
from text2mario.diffusion.scheduler import DiffusionSchedule
from text2mario.models.factory import build_unet
from text2mario.render.renderer import tensor_to_grid
from text2mario.train import checkpoint as ckpt


def load_ema_model(path: str | Path, device: str = "cpu"):
    """-> (unet in eval mode carrying the EMA weights, DiffusionSchedule, config dict, checkpoint dict)."""
    s = ckpt.load(path)
    cfg = json.loads(s["config_json"])
    unet = build_unet(cfg["model"], conditional=bool(cfg["text"]["enabled"])).to(device)
    unet.load_state_dict(s["unet"])
    with torch.no_grad():
        for n, p in unet.named_parameters():
            p.copy_(s["ema"]["shadow"][f"unet.{n}"].to(p.dtype))
    unet.eval()
    return unet, DiffusionSchedule(cfg["diffusion"]["schedule"], cfg["diffusion"]["T"], device), cfg, s


def sample_unconditional(unet, schedule, n: int, seed_base: int, steps: int = 50, batch: int = 64, device: str = "cpu",
                         autocast_dtype: torch.dtype | None = None) -> np.ndarray:
    """n samples from the unconditional model; sample i starts from noise seeded with seed_base + 1000 * i."""
    out = []
    for s in range(0, n, batch):
        idx = range(s, min(s + batch, n))
        x_init = torch.stack([torch.randn((13, 16, 16), generator=torch.Generator().manual_seed(seed_base + 1000 * i))
                              for i in idx])
        x, _ = ddim_sample(unet, schedule, (len(idx), 13, 16, 16), steps=steps, x_init=x_init, device=device,
                           autocast_dtype=autocast_dtype)
        out.append(tensor_to_grid(x))
    return np.concatenate(out)
