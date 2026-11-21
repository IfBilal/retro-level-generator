"""AdamW parameter groups and the warm-up + cosine learning-rate schedule (ledger M10, doc 13 §13, doc 16 §5).

Explain-me:
  What: weight decay (0.01) applies only to Conv/Linear weights (incl. the text adapter); norm scales, biases and the
        learned null context get none.  LR rises linearly for 1,000 steps then follows a cosine down to 1e-5.
        The frozen text encoder is never passed to the optimiser.
  In/Out: param_groups(modules, wd) -> list of groups;  lr_at(step, cfg) -> float.
  Try: print([lr_at(n, cfg) for n in (0, 500, 1000, 5000, 15000)]) -> 0, 1e-4, 2e-4, 1.64e-4, 1e-5 (15k horizon).
"""
from __future__ import annotations

import math

import torch


def param_groups(modules: dict[str, torch.nn.Module], weight_decay: float) -> list[dict]:
    decay, no_decay = [], []
    for m in modules.values():
        for n, p in m.named_parameters():
            if not p.requires_grad:
                continue
            is_weight = p.ndim >= 2 and "null_ctx" not in n          # conv (4-D) and linear (2-D) weights only
            (decay if is_weight else no_decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def build_optimizer(modules: dict[str, torch.nn.Module], cfg: dict) -> torch.optim.Optimizer:
    o = cfg["optim"]
    return torch.optim.AdamW(param_groups(modules, o["weight_decay"]), lr=o["lr"], betas=tuple(o["betas"]))


def lr_at(step: int, cfg: dict) -> float:
    """Linear warm-up to lr over `warmup_steps`, then cosine decay to lr_final at train.max_steps."""
    o, N = cfg["optim"], cfg["train"]["max_steps"]
    W, hi, lo = o["warmup_steps"], o["lr"], o["lr_final"]
    if step < W:
        return hi * step / W
    prog = min(1.0, (step - W) / max(1, N - W))
    return lo + 0.5 * (hi - lo) * (1 + math.cos(math.pi * prog))
