"""Comparator rows for every results table (doc 22 §5): what a generator WITHOUT learning would score on the same prompts.

Explain-me:
  What: (1) constant generators (always empty, always 'full floor.', always the modal caption); (2) a RETRIEVAL baseline that returns the train window
        whose caption has the highest C-score against the prompt (ties: first).  Retrieval is the honest "just look it up" competitor: it adheres to prompts
        well by construction but copies training data (copy rate 1.0, novelty 0) and cannot invent anything.
  In/Out: retrieval(prompts, train_captions) -> indices;  evaluate_baselines(prompts, train_caps, train_grids) -> dict of metrics.
  Previous: eval/adherence.py, eval/structure.py.  Next: scripts/compare_baselines.py -> reports/dev_eval/baselines.json (dev prompts only).
  Try: compare retrieval's C-score with the model's: the gap shows what generation adds over lookup (novelty) and costs (adherence).
"""
from __future__ import annotations

import numpy as np

from text2mario.eval import adherence as A
from text2mario.eval import structure as S

CONSTANTS = {"always_empty": "", "always_full_floor": "full floor.", "always_modal": "full floor. a few enemies."}


def retrieval(prompts: list[str], train_captions: list[str]) -> list[int]:
    """Index of the best-adhering train caption per prompt (C-score; exact ties -> the first)."""
    uniq = sorted(set(train_captions))
    first = {c: train_captions.index(c) for c in uniq}
    out = []
    for p in prompts:
        best = max(uniq, key=lambda c: (A.c_score(p, c), -first[c]))
        out.append(first[best])
    return out


def evaluate_baselines(prompts: list[str], train_captions: list[str], train_grids: np.ndarray) -> dict:
    res = {}
    for name, cap in CONSTANTS.items():
        res[name] = {"c_score": float(np.mean([A.c_score(p, cap) for p in prompts])),
                     "exact_match": float(np.mean([A.exact_match(p, cap) for p in prompts]))}
    idx = retrieval(prompts, train_captions)
    grids = train_grids[idx]
    caps = [train_captions[i] for i in idx]
    res["retrieval_from_train"] = {"c_score": float(np.mean([A.c_score(p, c) for p, c in zip(prompts, caps)])),
                                   "exact_match": float(np.mean([A.exact_match(p, c) for p, c in zip(prompts, caps)])),
                                   "broken_rate": S.broken_rate(grids), "floor_valid": S.floor_valid_rate(grids), "copy_rate": 1.0, "novelty_med": 0.0,
                                   "note": "returns a real training window: perfect structure, zero novelty"}
    return res
