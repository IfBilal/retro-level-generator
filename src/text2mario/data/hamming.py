"""Hamming-distance helpers on one-hot windows (doc 10 §3.2, doc 11 §3).

Explain-me:
  What: Hamming(g, h) = 256 - <onehot(g), onehot(h)>, so all-pairs distances are one matrix product.
  In/Out: uint8 grids (N,16,W) -> float one-hot (N, 13*16*W); NN distances to another set, optionally per level.
  Used by: family detection and the leakage tests (splits), novelty / copy rate (evaluation).
"""
from __future__ import annotations

import numpy as np
import torch


def onehot_flat(grids: np.ndarray) -> torch.Tensor:
    """(N,16,W) uint8 -> (N, 13*16*W) float32 one-hot."""
    t = torch.from_numpy(np.asarray(grids).astype(np.int64))
    return torch.nn.functional.one_hot(t, 13).reshape(len(grids), -1).float()


def nn_distance(a: np.ndarray, b: np.ndarray, chunk: int = 1024):
    """For each window of `a`: (min Hamming distance to `b`, argmin index into b)."""
    A, B = onehot_flat(a), onehot_flat(b)
    cells = a.shape[1] * a.shape[2]
    dist = np.zeros(len(a), np.int32)
    arg = np.zeros(len(a), np.int64)
    for s in range(0, len(a), chunk):
        d = cells - (A[s:s + chunk] @ B.T)
        m = d.min(dim=1)
        dist[s:s + chunk] = m.values.round().int().numpy()
        arg[s:s + chunk] = m.indices.numpy()
    return dist, arg


def nn_distance_by_level(scenes: np.ndarray, level_id: np.ndarray, n_levels: int, chunk: int = 1024) -> np.ndarray:
    """(N, n_levels): min Hamming distance from each window to the windows of each level.
    The entry for a window's own level is set to a large value (we only care about other levels)."""
    A = onehot_flat(scenes)
    n = len(scenes)
    out = np.full((n, n_levels), 10 ** 6, np.int32)
    cols = [np.flatnonzero(level_id == k) for k in range(n_levels)]
    for s in range(0, n, chunk):
        d = 256 - (A[s:s + chunk] @ A.T)
        for k in range(n_levels):
            out[s:s + chunk, k] = d[:, cols[k]].min(dim=1).values.round().int().numpy()
    out[np.arange(n), level_id] = 10 ** 6
    return out
