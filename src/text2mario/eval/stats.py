"""Level-clustered bootstrap statistics (ledger E3, doc 22 §7, CLAUDE.md rule 16).

Explain-me:
  What: prompts from the same real level are not independent, so confidence intervals resample whole LEVELS (clusters),
        never single prompts.  Also gives a paired version for A/B comparisons (e.g. CFG w* against w = 1).
  In/Out: cluster_bootstrap(values, clusters) -> (mean, lo, hi);  paired_cluster_bootstrap(a, b, clusters) -> (mean diff, lo, hi).
  Previous: eval/devloop.py rows (one value per prompt + its source level).  Next: gate_p4, scripts/eval_dev.py reports.
  Try: n_boot=200 vs 5000 and see how little the interval moves; fewer clusters make it much wider (honest with 5 dev levels).
"""
from __future__ import annotations

import numpy as np


def _groups(values, clusters):
    values = np.asarray(values, float)
    clusters = np.asarray(["_none" if c is None else str(c) for c in clusters])
    names = sorted(set(clusters))
    return [values[clusters == n] for n in names]


def cluster_bootstrap(values, clusters, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05):
    """Mean with a percentile interval from resampling clusters with replacement (weights = cluster sizes)."""
    gs = _groups(values, clusters)
    rng = np.random.default_rng(seed)
    allv = np.concatenate(gs)
    means = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(gs), len(gs))
        means[b] = np.concatenate([gs[i] for i in pick]).mean()
    return float(allv.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_cluster_bootstrap(a, b, clusters, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05):
    """Mean of (a - b) per prompt with a cluster-bootstrap interval; a and b are aligned per prompt."""
    d = np.asarray(a, float) - np.asarray(b, float)
    return cluster_bootstrap(d, clusters, n_boot, seed, alpha)


def n_clusters(clusters) -> int:
    return len({("_none" if c is None else str(c)) for c in clusters})
