"""Caption adherence: the C-score of doc 22 §3.1, whole-caption exact match and per-concept metrics (§3.2-3.3).

Explain-me:
  What: compares the caption of a PROMPT with the caption our detector reads off the GENERATED grid, topic by topic
        (18 topics), and averages the per-topic scores into C in [-1, 1].  Written from the specification (a clean
        rewrite, not a port); the golden vectors of doc 22 are tests.
  In/Out: c_score(prompt_caption, generated_caption) -> float;  exact_match(...);  concept_report(pairs) -> dict.
  Previous: captions/detector.py (grid -> caption).  Next: train/eval loop, scripts/eval_dev.py, the CFG sweep.
  Try: change QUANTITIES' order or the 0.1 'phrases differ' score and re-run tests/test_adherence.py.
Credit: the metric idea is from Lee & Simo-Serra (MarioDiffusion, MIT); see DATA_CARD.md.
"""
from __future__ import annotations

import math

# T: the 18 topics in the spec's order (substring match; 'enem' covers enemy/enemies, 'rectangular'/'irregular' are
# the first words of the block-cluster phrases).  The order matters: specific topics come before general ones.
TOPICS = ("floor", "ceiling", "broken pipe", "upside down pipe", "pipe", "coin line", "coin", "platform", "tower",
          "broken cannon", "cannon", "ascending staircase", "descending staircase", "rectangular", "irregular",
          "question block", "loose block", "enem")
T16 = tuple(t for t in TOPICS if not t.startswith("broken"))
QUANTITIES = ("one", "two", "a few", "several", "many")   # Q (ledger F15), index = iota
NO_MATCH_SCORE = 0.1                                      # phrases differ and at most one carries a quantity


def phrases(caption: str) -> list[str]:
    """Split on '.', strip, drop empties."""
    return [p.strip().lower() for p in caption.split(".") if p.strip()]


def extract(caption: str) -> dict[str, str | None]:
    """Topic -> the FIRST phrase containing it that no earlier topic has taken (None if there is none)."""
    free = phrases(caption)
    out: dict[str, str | None] = {}
    for t in TOPICS:
        hit = next((p for p in free if t in p), None)
        if hit is not None:
            free.remove(hit)
            if hit.startswith("no "):
                hit = None            # an absence phrase means "topic not present"
        out[t] = hit
    return out


def normalise(p: str) -> str:
    """Plural normalisation: 'enemies' -> 'enemy', then drop a trailing 's' of words that end in 's' but not 'ss'."""
    p = p.replace("enemies", "enemy")
    return " ".join(w[:-1] if w.endswith("s") and not w.endswith("ss") else w for w in p.split())


def quantity(p: str) -> int | None:
    """Index of the first quantity term (in Q's order) found in the normalised phrase."""
    for i, q in enumerate(QUANTITIES):
        if q in p:
            return i
    return None


def topic_score(a: str | None, b: str | None) -> float:
    if a is None and b is None:
        return 1.0
    if (a is None) != (b is None):
        return -1.0
    na, nb = normalise(a), normalise(b)
    if na == nb:
        return 1.0
    qa, qb = quantity(na), quantity(nb)
    if qa is not None and qb is not None:
        return 1.0 - abs(qa - qb) / (len(QUANTITIES) - 1)
    return NO_MATCH_SCORE


def c_score(prompt: str, generated: str) -> float:
    """C(prompt, G) = mean over the 18 topics of the per-topic match (range [-1, 1])."""
    pa, pb = extract(prompt), extract(generated)
    return sum(topic_score(pa[t], pb[t]) for t in TOPICS) / len(TOPICS)


def exact_match(prompt: str, generated: str) -> bool:
    """Whole-caption exact match: equal phrase sets after normalisation (order-free)."""
    return {normalise(p) for p in phrases(prompt)} == {normalise(p) for p in phrases(generated)}


def concept_report(pairs: list[tuple[str, str]], min_support: int = 5) -> dict:
    """Per-concept presence precision/recall/F1, strict exact match, absence respect, bucket MAE/bias, support (§3.3).

    `pairs` = [(prompt caption, generated caption)].  Concepts with support < min_support are flagged n/a."""
    rep = {}
    for t in T16:
        tp = fp = fn = tn = em_n = em_hit = ar_n = ar_hit = 0
        diffs: list[int] = []
        for prompt, gen in pairs:
            a, b = extract(prompt)[t], extract(gen)[t]
            req, got = a is not None, b is not None
            tp += req and got
            fp += (not req) and got
            fn += req and (not got)
            tn += (not req) and (not got)
            if req:
                em_n += 1
                em_hit += got and normalise(a) == normalise(b)
                qa, qb = quantity(normalise(a)), (quantity(normalise(b)) if got else None)
                if qa is not None and qb is not None:
                    diffs.append(qb - qa)
            else:
                ar_n += 1
                ar_hit += not got
        support = tp + fn
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / support if support else float("nan")
        f1 = 2 * prec * rec / (prec + rec) if not (math.isnan(prec) or math.isnan(rec)) and prec + rec else float("nan")
        rep[t] = {"support": support, "na": support < min_support, "precision": prec, "recall": rec, "f1": f1,
                  "exact_match": em_hit / em_n if em_n else float("nan"),
                  "absence_respect": ar_hit / ar_n if ar_n else float("nan"),
                  "bucket_mae": sum(abs(d) for d in diffs) / len(diffs) if diffs else float("nan"),
                  "bucket_bias": sum(diffs) / len(diffs) if diffs else float("nan")}
    return rep


RARE = ("rectangular", "ceiling", "cannon", "upside down pipe", "irregular", "descending staircase")


def rare_exact_match(report: dict) -> float:
    """Support-weighted mean of the strict exact match over the rare concepts (trigger for X4, doc 22 §3.3)."""
    num = den = 0.0
    for t in RARE:
        r = report[t]
        if r["support"] and r["exact_match"] == r["exact_match"]:
            num += r["support"] * r["exact_match"]
            den += r["support"]
    return num / den if den else float("nan")
