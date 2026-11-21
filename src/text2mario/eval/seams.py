"""Seam metric (doc 20 §5, doc 22 §3.11): are window joins as plausible as the rest of the level?

Explain-me:
  What: a smoothed column-bigram model is fitted on TRAIN levels (a column = the 16 tile ids of one x position).  The NLL of
        every column-to-column transition is computed; seam_ratio = mean NLL at the generator's seams / mean NLL elsewhere.
        1.0 means seams look like ordinary transitions; the gate wants <= 1.3.
  In/Out: ColumnBigram.fit(levels).seam_ratio(level) -> float (and nll arrays).  Seams sit at columns c = 16, 24, 32, ...
        (a new window's right half joins the kept left half).  Real dev levels with pseudo-seams give the reference.
  Previous: generate/full_level.py output (content columns only, framing excluded).  Next: scripts/gate_p5.py.
  Try: k (smoothing) 0.1 vs 5 changes the absolute NLLs but barely the ratio.
"""
from __future__ import annotations

from collections import Counter
from itertools import pairwise

import numpy as np

from text2mario.utils import appcfg

_C = appcfg.load()
SEAM_START, SEAM_STEP = int(_C["window"]), int(_C["stride"])     # a new window's right part joins at column window, window+stride, ...


class ColumnBigram:
    def __init__(self, k: float = 1.0):
        self.k = k
        self.uni: Counter = Counter()
        self.bi: Counter = Counter()
        self.ctx: Counter = Counter()
        self.total = 0

    @staticmethod
    def _cols(level: np.ndarray):
        return [bytes(level[:, c].astype(np.uint8)) for c in range(level.shape[1])]

    def fit(self, levels):
        for lv in levels:
            cols = self._cols(lv)
            self.uni.update(cols)
            for a, b in pairwise(cols):
                self.bi[(a, b)] += 1
                self.ctx[a] += 1
        self.total = sum(self.uni.values())
        self.V = len(self.uni) + 1                      # +1 for 'unseen column'
        return self

    def _p_uni(self, b) -> float:
        return (self.uni.get(b, 0) + 1.0) / (self.total + self.V)

    def nll_transitions(self, level: np.ndarray) -> np.ndarray:
        """-log P(col_{t+1} | col_t), interpolated with the smoothed unigram: (count + k * p_uni) / (ctx + k)."""
        cols = self._cols(level)
        out = []
        for a, b in pairwise(cols):
            p = (self.bi.get((a, b), 0) + self.k * self._p_uni(b)) / (self.ctx.get(a, 0) + self.k)
            out.append(-np.log(p))
        return np.array(out)

    def seam_ratio(self, level: np.ndarray):
        """-> (ratio, mean_seam_nll, mean_interior_nll).  Transition index t joins column t and t+1; seams are at column
        c = 16, 24, ... i.e. transition t = c - 1."""
        nll = self.nll_transitions(level)
        idx = np.arange(len(nll))
        is_seam = (idx + 1 >= SEAM_START) & ((idx + 1 - SEAM_START) % SEAM_STEP == 0)
        if not is_seam.any():
            return float("nan"), float("nan"), float(nll.mean())
        s, i = float(nll[is_seam].mean()), float(nll[~is_seam].mean())
        return s / i, s, i
