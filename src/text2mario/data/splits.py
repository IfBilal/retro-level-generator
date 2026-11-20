"""Level-grouped splitting: near-duplicate families, exhaustive deterministic selection, CV folds, LOCK (doc 10).

Explain-me:
  What: assigns whole LEVELS (never single windows) to train / dev / test. Levels that reuse each other's layouts
        form "families" and stay in train. Test = 3 SMB1 + 3 SMB2 levels, dev = 5, chosen by exhaustively scoring
        every allowed combination (no randomness), then the result is hashed into data/splits/LOCK.
  In/Out: per-level statistics + the cross-level distance matrix -> split dict (see doc 10 §10).
  Used by: scripts/make_splits.py (build), dataset.py (load with verify), gate G-P1 tests.
  Try: change family_link_threshold in configs/data.yaml and watch the families (and the whole split) change.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SPLIT_PATH = ROOT / "data" / "splits" / "split_v1.json"
LOCK_PATH = ROOT / "data" / "splits" / "LOCK"


# ------------------------------------------------------------------ families (doc 10 §3)
def link_matrix(nn_by_level: np.ndarray, level_id: np.ndarray, n_levels: int, hamming: int = 2) -> np.ndarray:
    """L[a, b] = share of level a's windows whose nearest window in level b is within `hamming` tiles."""
    L = np.zeros((n_levels, n_levels))
    for a in range(n_levels):
        rows = nn_by_level[level_id == a]
        L[a] = (rows <= hamming).mean(axis=0)
    np.fill_diagonal(L, 0.0)
    return L


def find_families(L: np.ndarray, tau: float) -> list[list[int]]:
    """Connected components (size >= 2) of the graph with an edge when max(L[a,b], L[b,a]) >= tau."""
    n = len(L)
    adj = np.maximum(L, L.T) >= tau
    seen, fams = set(), []
    for s in range(n):
        if s in seen:
            continue
        comp, stack = [], [s]
        seen.add(s)
        while stack:
            a = stack.pop()
            comp.append(a)
            for b in np.flatnonzero(adj[a]):
                if int(b) not in seen:
                    seen.add(int(b))
                    stack.append(int(b))
        if len(comp) >= 2:
            fams.append(sorted(comp))
    return sorted(fams)


# ------------------------------------------------------------------ objective (doc 10 §5.2)
def js2(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Jensen-Shannon divergence in bits along the last axis; 0*log0 := 0 (zeros are masked, never eps-added)."""
    m = (p + q) / 2

    def kl(a, b):
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(a > 0, a * np.log2(a / b), 0.0)
        return t.sum(axis=-1)

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def _score(conc: np.ndarray, tile: np.ndarray, share: np.ndarray, c_all, t_all, target: float) -> np.ndarray:
    cn = conc / conc.sum(axis=-1, keepdims=True)
    tn = tile / tile.sum(axis=-1, keepdims=True)
    return js2(cn, c_all) + js2(tn, t_all) + 2 * (share - target) ** 2


def _lex_best(J: np.ndarray, combos: list[tuple[int, ...]], feasible: np.ndarray):
    """Index of the feasible combo with the lowest J; ties broken by the lexicographically smallest tuple."""
    idx = np.flatnonzero(feasible)
    if len(idx) == 0:
        return None
    best = J[idx].min()
    tied = [i for i in idx if J[i] <= best + 1e-12]
    return min(tied, key=lambda i: tuple(sorted(combos[i])))


def select_test(stats: dict, eligible: dict[str, list[int]], n_total: int, cfg: dict):
    """Stage 1: all 3 SMB1 + 3 SMB2 combinations of eligible levels (100,100 in the measured data)."""
    conc, tile, win = stats["concept"], stats["tile"], stats["windows"]
    c_all, t_all = conc.sum(0) / conc.sum(), tile.sum(0) / tile.sum()
    c1 = list(itertools.combinations(eligible["SMB1"], cfg["test_smb1"]))
    c2 = list(itertools.combinations(eligible["SMB2"], cfg["test_smb2"]))
    s1 = lambda arr, cs: np.stack([arr[list(c)].sum(0) for c in cs])
    conc_s = s1(conc, c1)[:, None, :] + s1(conc, c2)[None, :, :]
    tile_s = s1(tile, c1)[:, None, :] + s1(tile, c2)[None, :, :]
    win_s = s1(win[:, None], c1)[:, None, 0] + s1(win[:, None], c2)[None, :, 0]
    share = win_s / n_total
    J = _score(conc_s, tile_s, share, c_all, t_all, cfg["test_share"])
    feas = (np.abs(share - cfg["test_share"]) <= cfg["test_share_tol"]) & (conc_s >= cfg["min_concept_windows"]).all(-1)
    combos = [a + b for a in c1 for b in c2]
    i = _lex_best(J.ravel(), combos, feas.ravel())
    n_feas = int(feas.sum())
    return (None, n_feas) if i is None else (sorted(combos[i]), n_feas, float(J.ravel()[i]))


def select_dev(stats: dict, eligible_left: dict[str, list[int]], test: list[int], n_total: int, cfg: dict):
    """Stage 2: dev = 5 levels (2+3 or 3+2 by game) from the remaining eligible levels."""
    conc, tile, win, has = stats["concept"], stats["tile"], stats["windows"], stats["concept"] > 0
    c_all, t_all = conc.sum(0) / conc.sum(), tile.sum(0) / tile.sum()
    levels_with = has.sum(0)                       # number of levels containing each concept
    test_has = has[test].sum(0)
    combos, J, feas = [], [], []
    for k1, k2 in ((2, 3), (3, 2)):
        for a in itertools.combinations(eligible_left["SMB1"], k1):
            for b in itertools.combinations(eligible_left["SMB2"], k2):
                lv = list(a + b)
                c, t, w = conc[lv].sum(0), tile[lv].sum(0), win[lv].sum()
                share = w / n_total
                score = _score(c, t, np.array(share), c_all, t_all, cfg["dev_share"])
                train_levels_with = levels_with - test_has - has[lv].sum(0)   # C7 on the implied train set
                ok = (abs(share - cfg["dev_share"]) <= cfg["dev_share_tol"] and (c >= cfg["min_concept_windows"]).all()
                      and (train_levels_with >= cfg["min_train_levels_per_concept"]).all())
                combos.append(a + b)
                J.append(float(score))
                feas.append(ok)
    J, feas = np.array(J), np.array(feas)
    i = _lex_best(J, combos, feas)
    return (None, int(feas.sum())) if i is None else (sorted(combos[i]), int(feas.sum()), float(J[i]))


# ------------------------------------------------------------------ CV folds (doc 10 §8)
def make_cv_folds(levels_df, families: list[list[int]], test: list[int], k: int = 5) -> list[list[int]]:
    """Greedy longest-processing-time assignment of units (families / singleton levels) to k folds,
    balancing windows and the SMB1/SMB2 mix.  Test levels are excluded."""
    win = dict(zip(levels_df.level_id, levels_df.windows))
    game = dict(zip(levels_df.level_id, levels_df.game))
    pop = [l for l in levels_df.level_id if l not in set(test)]
    in_fam = {l for f in families for l in f}
    units = [[l for l in f if l in set(pop)] for f in families] + [[l] for l in pop if l not in in_fam]
    units = [u for u in units if u]
    units.sort(key=lambda u: (-sum(win[l] for l in u), min(u)))
    total = {g: sum(1 for l in pop if game[l] == g) for g in ("SMB1", "SMB2")}
    folds: list[list[int]] = [[] for _ in range(k)]
    def cost(f, u):
        """Windows in fold f after adding unit u, plus a penalty for an SMB1/SMB2 mix far from the global mix."""
        wins = sum(win[l] for l in folds[f]) + sum(win[l] for l in u)
        pen = sum(abs(sum(1 for l in folds[f] + u if game[l] == g) - total[g] / k) for g in total)
        return wins + 50 * pen

    for u in units:
        best = min(range(k), key=lambda f: (cost(f, u), f))
        folds[best].extend(u)
    return [sorted(f) for f in folds]


# ------------------------------------------------------------------ LOCK (doc 10 §7)
def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def lock_hash(split: dict, corpus_hash: str) -> str:
    """sha256(canonical_json(split) + newline + corpus content hash)."""
    return hashlib.sha256((canonical_json(split) + "\n" + corpus_hash).encode("utf-8")).hexdigest()


def load(verify: bool = True) -> dict:
    """Read split_v1.json; with verify=True check it against LOCK (any edit after Day 2 raises)."""
    from text2mario.data import provenance
    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    if verify:
        want = LOCK_PATH.read_text().strip()
        got = lock_hash(split, provenance.corpus_content_hash())
        if want != got:
            raise RuntimeError("split_v1.json does not match data/splits/LOCK (D4): the split is locked")
    return split
