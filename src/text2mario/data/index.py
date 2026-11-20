"""Level (chain) reconstruction and the level index (ledger F10, doc 09 §2).

Explain-me:
  What: the corpus has no level id, but window i+1 is window i shifted by one column. We cut the record list
        wherever that stops being true, which recovers the 42 levels; `level_grid` rebuilds each full level.
  In/Out: scenes uint8 (N,16,16) -> chain starts, level_id[N], col_offset[N], full level grids.
  Next: scripts/audit_dataset.py and make_splits.py use this.  Try: look at chain_starts() on a shuffled array.
"""
from __future__ import annotations

import numpy as np

N_SMB1_WINDOWS = 3606  # ledger F9: records [0:3606] are SMB1


def chain_starts(scenes: np.ndarray) -> list[int]:
    """Record indices where a new level begins (window i does not continue window i-1)."""
    cont = (scenes[:-1, :, 1:] == scenes[1:, :, :-1]).all(axis=(1, 2))  # (N-1,)
    return [0] + [int(i) + 1 for i in np.flatnonzero(~cont)]


def level_ids(scenes: np.ndarray):
    """-> (level_id[N] int16, col_offset[N] int16, chains [(start, end_exclusive)])."""
    n = len(scenes)
    starts = chain_starts(scenes)
    ends = starts[1:] + [n]
    lid = np.zeros(n, np.int16)
    off = np.zeros(n, np.int16)
    for k, (b, e) in enumerate(zip(starts, ends)):
        lid[b:e] = k
        off[b:e] = np.arange(e - b)
    return lid, off, list(zip(starts, ends))


def level_grid(scenes: np.ndarray, start: int, end: int) -> np.ndarray:
    """Full-width level (16, width) = first window + last column of every later window."""
    cols = [scenes[start]] + [scenes[i][:, -1:] for i in range(start + 1, end)]
    return np.concatenate(cols, axis=1)
