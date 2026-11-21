"""Exponential moving average of weights with warm-up (ledger M11, doc 13 §13.4).

Explain-me:
  What: keeps a smoothed copy of the weights; decay d_n = min(0.999, (1+n)/(10+n)) so it forgets the random start quickly.
        Evaluation and export always use the EMA copy.
  In/Out: EMA(modules).update(n_completed_steps); .apply_to(modules) swaps the EMA weights in; .restore() swaps back.
  Try: print([ema_decay(n) for n in (0, 10, 100, 1000, 8990)]) -> 0.1, 0.55, 0.918, 0.991, 0.999.
"""
from __future__ import annotations

import torch


def ema_decay(n: int, max_decay: float = 0.999) -> float:
    return min(max_decay, (1 + n) / (10 + n))


class EMA:
    def __init__(self, modules: dict[str, torch.nn.Module], max_decay: float = 0.999):
        self.max_decay = max_decay
        self.shadow = {f"{k}.{n}": p.detach().clone().float() for k, m in modules.items() for n, p in m.named_parameters()}
        self._backup: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def update(self, modules: dict[str, torch.nn.Module], n: int) -> None:
        d = ema_decay(n, self.max_decay)
        for k, m in modules.items():
            for name, p in m.named_parameters():
                s = self.shadow[f"{k}.{name}"]
                s.mul_(d).add_(p.detach().float(), alpha=1 - d)

    @torch.no_grad()
    def apply_to(self, modules: dict[str, torch.nn.Module]) -> None:
        self._backup = {}
        for k, m in modules.items():
            for name, p in m.named_parameters():
                self._backup[f"{k}.{name}"] = p.detach().clone()
                p.copy_(self.shadow[f"{k}.{name}"].to(p.dtype))

    @torch.no_grad()
    def restore(self, modules: dict[str, torch.nn.Module]) -> None:
        for k, m in modules.items():
            for name, p in m.named_parameters():
                p.copy_(self._backup[f"{k}.{name}"])
        self._backup = {}

    def state_dict(self) -> dict:
        return {"shadow": self.shadow, "max_decay": self.max_decay}

    def load_state_dict(self, sd: dict) -> None:
        # checkpoints are loaded on the CPU; the shadow must stay on the device the live parameters use (found when resuming on CUDA)
        dev = next(iter(self.shadow.values())).device if self.shadow else torch.device("cpu")
        self.shadow = {k: v.clone().to(dev) for k, v in sd["shadow"].items()}
