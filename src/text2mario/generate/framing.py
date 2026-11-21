"""Deterministic level framing (ledger G3): a flat spawn zone in front and an end template (stairs + flag) behind.

Explain-me:
  What: the generated content is wrapped by 3 full-floor columns (spawn zone) and 6 template columns (a 4-step staircase,
        then a flat top with the flag).  These columns are NOT generated and are flagged `templated` in the level JSON.
  In/Out: frame(content[16,W]) -> (framed grid, framing dict as in doc 29).  Previous: generate/full_level.py.
  Try: change LEAD or the stair heights below and watch the spawn zone / end template change in the app.
"""
from __future__ import annotations

import numpy as np

from text2mario.data import tiles as T

LEAD, TAIL = 3, 6
STAIRS = (1, 2, 3, 4, 4, 4)          # block height of each of the 6 tail columns (flag stands on the last three)


def frame(content: np.ndarray):
    content = np.asarray(content, dtype=np.uint8)
    rows, w = content.shape
    lead = np.zeros((rows, LEAD), np.uint8)
    lead[rows - 1, :] = T.GROUND
    tail = np.zeros((rows, TAIL), np.uint8)
    for c, h in enumerate(STAIRS):
        tail[rows - 1 - h + 1:, c] = T.GROUND          # h blocks tall (the floor row counts as the first)
        tail[rows - 1, c] = T.GROUND
    grid = np.concatenate([lead, content, tail], axis=1)
    cols = grid.shape[1]
    top_h = STAIRS[-1]
    flag_col = cols - TAIL + 3                          # a column on the flat top
    framing = {"templated_cols": [[0, LEAD - 1], [cols - TAIL, cols - 1]],
               "spawn": {"col": 1, "row": rows - 2},
               "flag": {"col": flag_col, "row_top": rows - top_h - 6, "row_bottom": rows - top_h - 1},
               "checkpoints": [{"col": 1}] + ([{"col": LEAD + w // 2}] if w >= 48 else [])}
    return grid, framing
