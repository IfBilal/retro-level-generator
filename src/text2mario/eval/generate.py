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


# ------------------------------------------------------------------------------------------------ conditional (Phase 4)
def load_ema_conditional(path: str | Path, device: str = "cpu", max_len: int | None = None):
    """-> (unet with EMA weights, schedule, TextConditioner with EMA adapter, cfg, checkpoint dict).
    The frozen MiniLM is rebuilt from models/minilm and verified against the probe vector stored in the checkpoint."""
    from text2mario.text.conditioner import TextConditioner
    unet, sched, cfg, s = load_ema_model(path, device)
    # max_len defaults to the training value; evaluation of synthetic prompts may need more (the check never truncates silently)
    cond = TextConditioner(cfg["text"]["encoder_dir"], max_len or cfg["data"]["max_len"], cfg["text"]["adapter_hidden"],
                           cfg["model"]["context_dim"], cfg["text"]["cache_max_mb"], device)
    with torch.no_grad():
        for n, p in cond.adapter.named_parameters():
            p.copy_(s["ema"]["shadow"][f"adapter.{n}"].to(p.dtype))
    cond.adapter.eval()
    cond.check_probe(s["text"]["probe"])
    return unet, sched, cond, cfg, s


def sample_conditional(unet, schedule, cond, prompts: list[str], w: float, seeds: list[int], steps: int = 50,
                       batch: int = 64, device: str = "cpu", autocast_dtype: torch.dtype | None = None,
                       eta: float = 0.0) -> np.ndarray:
    """One 16x16 grid per (prompt, seed) pair; the pair's own seed fixes its starting noise (portable across batch sizes).
    CFG: eps_u + w * (eps_c - eps_u); w = 1 is plain conditional sampling (doc 15)."""
    assert len(prompts) == len(seeds)
    out = []
    for s in range(0, len(prompts), batch):
        ps, sd = prompts[s:s + batch], seeds[s:s + batch]
        with torch.no_grad():
            ctx_c, mask_c = cond(ps)
            ctx_u, mask_u = cond.null(len(ps))
        x_init = torch.stack([torch.randn((13, 16, 16), generator=torch.Generator().manual_seed(int(i))) for i in sd])
        x, _ = ddim_sample(unet, schedule, (len(ps), 13, 16, 16), ctx_c=ctx_c, mask_c=mask_c, ctx_u=ctx_u, mask_u=mask_u,
                           w=w, steps=steps, eta=eta, x_init=x_init, device=device, autocast_dtype=autocast_dtype)
        out.append(tensor_to_grid(x))
    return np.concatenate(out)


# ------------------------------------------------------------------------------------------------ exported (cloud) bundle
def load_exported(bundle_dir: str | Path, device: str = "cpu"):
    """Load the small inference bundle written by scripts/export_cloud_ckpt.py:
         <dir>/config.json, <dir>/t2m_ema.safetensors (unet.* + adapter.* + probe), <dir>/minilm/ (fp16 encoder + tokenizer).
    Returns the same tuple as load_ema_conditional: (unet, schedule, cond, cfg, state)."""
    from safetensors.torch import load_file
    from text2mario.text.conditioner import TextConditioner
    d = Path(bundle_dir)
    cfg = json.loads((d / "config.json").read_text())
    tensors = load_file(str(d / "t2m_ema.safetensors"))
    unet = build_unet(cfg["model"], conditional=True).to(device)
    with torch.no_grad():
        for n, p in unet.named_parameters():
            p.copy_(tensors[f"unet.{n}"].to(p.dtype))
    unet.eval()
    cond = TextConditioner(str(d / "minilm"), cfg["data"]["max_len"], cfg["text"]["adapter_hidden"], cfg["model"]["context_dim"],
                           cfg["text"]["cache_max_mb"], device)
    with torch.no_grad():
        for n, p in cond.adapter.named_parameters():
            p.copy_(tensors[f"adapter.{n}"].to(p.dtype))
    cond.adapter.eval()
    cond.check_probe(tensors["probe"].float())
    sched = DiffusionSchedule(cfg["diffusion"]["schedule"], cfg["diffusion"]["T"], device)
    return unet, sched, cond, cfg, {"step": int(cfg.get("export", {}).get("step", -1)), "exported": True}
