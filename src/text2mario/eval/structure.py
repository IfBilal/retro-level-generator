"""Structure and distribution metrics on generated grids (doc 22 §3.4-3.10).

Explain-me:
  What: broken-structure counts (via the caption detector), floor validity, nearest-neighbour novelty and copy rate
        against the train windows, within-prompt diversity and the tile-distribution Jensen-Shannon divergence.
        These are the numbers the gates G-P3/G-P5 and the dev composite are made of.
  In/Out: grids uint8 (n,16,16) -> floats.  Metrics are computed on RAW samples, before any sanitizer repair (rule 8).
  Try: tile_js(all-sky grids, train) is about 0.0676 - the "collapse to sky" reference (doc 22 §3.10).
"""
from __future__ import annotations

import numpy as np
import torch

from text2mario.captions.detector import detect
from text2mario.data import tiles as T
from text2mario.data.hamming import nn_distance

FLOOR_ALPHABET = {T.SKY, T.BREAKABLE, T.GROUND, T.PIPE_BL, T.PIPE_BR}      # the only tiles ever seen in row 15


def broken_counts(grid: np.ndarray) -> tuple[int, int, int]:
    """(broken pipe components, broken cannon cells, total pipe components) of one grid."""
    d = {x.topic: x.count for x in detect(grid)}
    return d.get("broken pipe", 0), d.get("broken cannon", 0), d.get("pipe", 0) + d.get("upside down pipe", 0) + d.get("broken pipe", 0)


def broken_rate(grids: np.ndarray) -> float:
    """Share of samples with >= 1 broken pipe component OR >= 1 broken cannon (primary metric, gates use this)."""
    return float(np.mean([(lambda b: b[0] > 0 or b[1] > 0)(broken_counts(g)) for g in grids]))


def valid_floor(g: np.ndarray) -> bool:
    """Doc 22 §3.5: row 15 only has legal tiles AND (>= 8 solid cells OR the window is not a void)."""
    row = g[15]
    if not all(int(v) in FLOOR_ALPHABET for v in row):
        return False
    if sum(int(v) in T.SOLID for v in row) >= 8:
        return True
    return any(int(v) in T.SOLID for v in g[2:15].ravel())


def floor_valid_rate(grids: np.ndarray) -> float:
    return float(np.mean([valid_floor(g) for g in grids]))


def floorless_share(grids: np.ndarray) -> float:
    """Diagnostic: share of samples whose bottom row is entirely passable (real data: 16.64 %)."""
    return float(np.mean([all(int(v) in T.PASSABLE for v in g[15]) for g in grids]))


def novelty(grids: np.ndarray, train: np.ndarray) -> dict:
    """NN Hamming distance to the train windows: median (primary), mean, p10, p90, exact-copy share."""
    d, _ = nn_distance(np.asarray(grids), train)
    return {"novelty_med": float(np.median(d)), "mean": float(d.mean()), "p10": float(np.percentile(d, 10)),
            "p90": float(np.percentile(d, 90)), "exact_copy_rate": float((d == 0).mean()), "nn": d}


def copy_rates(nn: np.ndarray) -> dict:
    return {"copy_rate": float((nn <= 4).mean()), "copy_rate_strict": float((nn <= 2).mean()), "exact_copy_rate": float((nn == 0).mean())}


def diversity(groups: list[np.ndarray]) -> float:
    """Mean over prompts of the mean pairwise Hamming distance between that prompt's samples."""
    vals = []
    for g in groups:
        n = len(g)
        if n < 2:
            continue
        flat = np.asarray(g).reshape(n, -1)
        dist = [(flat[i] != flat[j]).sum() for i in range(n) for j in range(i + 1, n)]
        vals.append(np.mean(dist))
    return float(np.mean(vals))


def amed_self(grids: np.ndarray) -> float:
    """Mean over samples of the min Hamming distance to any OTHER sample in the same set (always report n)."""
    flat = torch.from_numpy(np.asarray(grids).reshape(len(grids), -1).astype(np.int64))
    d = (flat[:, None, :] != flat[None, :, :]).sum(-1).float()
    d.fill_diagonal_(float("inf"))
    return float(d.min(dim=1).values.mean())


def tile_hist(grids: np.ndarray) -> np.ndarray:
    return np.bincount(np.asarray(grids).ravel(), minlength=T.N_TILES).astype(np.float64)


def tile_js(grids: np.ndarray, train: np.ndarray) -> float:
    """JS divergence (log base 2, range [0,1]) between the sample tile histogram and the train histogram."""
    p, q = tile_hist(grids), tile_hist(train)
    p, q = p / p.sum(), q / q.sum()
    m = (p + q) / 2

    def kl(a, b):
        mask = a > 0
        return float((a[mask] * np.log2(a[mask] / b[mask])).sum())

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)
