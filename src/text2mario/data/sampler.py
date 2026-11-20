"""Stateless, resumable batch-index sampler (doc 11 §5.1).

Explain-me:
  What: batch t is a pure function of (seed, t): think of an endless list made of one shuffled copy of the dataset
        per epoch, and batch t is items [64t, 64t+64) of it.  So resume only needs the step counter, and both
        GPUs/machines draw identical indices.
  In/Out: WindowSampler(n, batch_size, seed).rows(step) -> int64 tensor of dataset rows (CPU).
  Try: build two samplers with the same seed and compare rows(1000) (they must be equal).
"""
from __future__ import annotations

import hashlib
from typing import Literal

import numpy as np
import torch


def mix(a: int, b: int) -> int:
    """Deterministic 63-bit seed from two ints. Never use Python hash() (string hashing is salted per process)."""
    return int.from_bytes(hashlib.blake2b(f"{a}:{b}".encode(), digest_size=8).digest(), "little") & (2 ** 63 - 1)


class WindowSampler:
    def __init__(self, n: int, batch_size: int = 64, seed: int = 20261117,
                 mode: Literal["epoch_perm", "balanced"] = "epoch_perm", weights: np.ndarray | None = None):
        if mode == "balanced" and weights is None:
            raise ValueError("balanced mode needs per-row weights")
        self.n, self.batch_size, self.seed, self.mode = n, batch_size, seed, mode
        self.weights = None if weights is None else torch.as_tensor(weights, dtype=torch.float64)
        self._perm_cache: dict[int, torch.Tensor] = {}

    def _perm(self, epoch: int) -> torch.Tensor:
        if epoch not in self._perm_cache:
            if len(self._perm_cache) > 4:
                self._perm_cache.clear()
            g = torch.Generator("cpu").manual_seed(mix(self.seed, epoch))  # CPU generators are portable across GPUs
            self._perm_cache[epoch] = torch.randperm(self.n, generator=g)
        return self._perm_cache[epoch]

    def rows(self, step: int) -> torch.Tensor:
        """(batch_size,) int64 dataset rows for this step."""
        if self.mode == "balanced":
            g = torch.Generator("cpu").manual_seed(mix(self.seed, 1_000_000 + step))
            return torch.multinomial(self.weights, self.batch_size, replacement=True, generator=g)
        pos = torch.arange(step * self.batch_size, (step + 1) * self.batch_size)  # positions in the endless list
        epochs = pos // self.n
        out = torch.empty(self.batch_size, dtype=torch.int64)
        for e in epochs.unique().tolist():
            m = epochs == e
            out[m] = self._perm(int(e))[pos[m] % self.n]
        return out

    def state_dict(self) -> dict:
        return {"mode": self.mode, "seed": self.seed, "n": self.n, "batch_size": self.batch_size}

    def check_resume(self, state: dict) -> None:
        """Refuse to resume with a different dataset/batch size (doc 11 §7)."""
        if state != self.state_dict():
            raise RuntimeError(f"sampler state mismatch: {state} vs {self.state_dict()}")
