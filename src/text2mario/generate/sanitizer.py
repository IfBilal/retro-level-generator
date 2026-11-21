"""Multi-tile sanitizer (ledger G4): DETECT broken pipes / cannons on the raw grid; optional repair for DISPLAY only.

Explain-me:
  What: counts broken pipe components and broken cannon cells (the same detector as the metrics), and can produce a
        repaired copy for display: broken pipes and orphan cannon cells are removed (replaced by sky).
        The raw grid is never modified; all metrics use the raw grid (CLAUDE.md rule 8).
  In/Out: detect(grid) -> dict(counts);  repair(grid) -> (new grid, n_removed_cells).
  Previous: generate/full_level.py.  Next: render/schema.py (sanitizer block of the level JSON).
"""
from __future__ import annotations

import numpy as np

from text2mario.captions.detector import detect as detect_phrases
from text2mario.data import tiles as T


def detect(grid: np.ndarray) -> dict:
    d = {x.topic: x.count for x in detect_phrases(np.asarray(grid))}
    return {"broken_pipes": int(d.get("broken pipe", 0)), "broken_cannons": int(d.get("broken cannon", 0))}


def repair(grid: np.ndarray):
    """Display-time repair: delete every cell of every component the detector classifies as a broken pipe or broken cannon,
    repeating until none is left (removing one structure can expose another).  The input grid is never modified."""
    g = np.asarray(grid).copy()
    removed = 0
    for _ in range(12):
        bad = [x for x in detect_phrases(g) if x.topic in ("broken pipe", "broken cannon")]
        if not bad:
            break
        for x in bad:
            for (r, c) in x.cells:
                if g[r, c] != T.SKY:
                    g[r, c] = T.SKY
                    removed += 1
    return g, removed
