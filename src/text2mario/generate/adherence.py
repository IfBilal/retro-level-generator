"""Prompt adherence for full levels: plan counts across windows, then verify the RAW level against what was asked (audit P2).

Explain-me:
  What: the model was trained on 16x16 windows whose captions describe the whole window ("one pipe" = exactly one pipe in 16 columns).
        A prompt describes the whole level, so giving every window of a scene the scene caption asked for one pipe PER WINDOW
        (a 48-column level has 5 windows -> several pipes).  This module (1) turns each scene's requested counts into a
        level-level target, (2) writes each window's caption from what is still missing (plan_caption), and (3) counts the raw
        level with the repo's own detector and says which requested counts are met (evaluate); the generator resamples the offending
        window until they are met or the attempt budget ends.  No tile is ever edited here: a miss is reported, never hidden.
  In/Out: plan_caption(...) -> caption str;  evaluate(content, scenes, scene_of, ...) -> dict(status, constraints, unmet);
          check_feasible(norm, width) raises ValueError for requests that cannot be met (contradiction, too many for the width).
  Previous: text/normalizer.py (Scene.want / Scene.forbid).  Next: generate/full_level.py (the resample loop), app (message).
  Try: configs/generate/app.yaml -> adherence.enabled: false restores the old behaviour; max_new_per_window changes how objects are spread.
"""
from __future__ import annotations

import math

import numpy as np

from text2mario.captions.grammar import CONCEPTS, counted_phrase, parse_phrase
from text2mario.eval.count_audit import detector_objects
from text2mario.text import normalizer as N
from text2mario.utils import appcfg

_A = appcfg.load()["adherence"]
WINDOW, STRIDE = int(appcfg.load()["window"]), int(appcfg.load()["stride"])
COUNTED = [t for t in CONCEPTS if t not in ("floor", "ceiling")]


def enabled() -> bool:
    return bool(_A["enabled"])


def count_as(topic: str) -> list[str]:
    """Detector topics that count as `topic` (a hanging or broken pipe is still 'a pipe' to the person looking at the level)."""
    return list(_A["count_as"].get(topic, [topic]))


def anchors(grid: np.ndarray) -> dict[str, list[int]]:
    """Requested topic -> sorted leftmost columns of the objects found in `grid` (detector definition, see eval/count_audit.py)."""
    det = detector_objects(grid) if grid.shape[1] >= 1 else {}
    out = {t: sorted(a for d in count_as(t) for a in det.get(d, [])) for t in COUNTED}
    return out


def region_bounds(scene_of: list[int], width: int) -> list[tuple[int, int]]:
    """Columns each scene OWNS.  Window k owns the 8 columns it adds (cols 0-15 for window 0, 8k+8..8k+15 later); a scene owns the
    columns added by its windows.  An object belongs to a scene by its leftmost column."""
    firsts = {}
    for k, s in enumerate(scene_of):
        firsts.setdefault(s, k)
    order = sorted(firsts)
    starts = [0 if firsts[s] == 0 else firsts[s] * STRIDE + (WINDOW - STRIDE) for s in order]
    return [(starts[i], starts[i + 1] if i + 1 < len(order) else width) for i in range(len(order))]


def target_of(lo: int, hi: int) -> int:
    """Count to plan for a requested range: exact numbers as they are, a vague bucket at its middle (or its low end when open-ended)."""
    if hi - lo > 20 or _A["range_target"] == "low":
        return lo
    return lo + (hi - lo) // 2


def constraints(norm) -> list[dict]:
    """Flat list of {scene, topic, lo, hi} from the normalized scenes (a prohibition is lo=hi=0)."""
    out = []
    for i, sc in enumerate(norm.scenes):
        for t, (lo, hi) in sc.want.items():
            if t in COUNTED:
                out.append({"scene": i, "topic": t, "lo": int(lo), "hi": int(hi)})
        for t in sc.forbid:
            if t in COUNTED:
                out.append({"scene": i, "topic": t, "lo": 0, "hi": 0})
    return out


def check_feasible(norm, width: int) -> None:
    """Raise ValueError (a user-facing sentence) when the request cannot be met at this width."""
    bad = [w for w in norm.warnings if w.startswith("CONTRADICTION")]
    if bad:
        topics = ", ".join(sorted({w.split(":", 1)[1] for w in bad}))
        raise ValueError(f"The sentence asks for and rules out the same thing ({topics}). Please say one or the other.")
    need: dict[str, int] = {}
    for c in constraints(norm):
        need[c["topic"]] = need.get(c["topic"], 0) + c["lo"]
    for t, n in need.items():
        cap = int(_A["capacity_per_16_cols"].get(t, 99) * width / WINDOW)
        if n > cap:
            raise ValueError(f"{n} x {t} will not fit in {width} columns (at most about {cap}). Ask for fewer or choose a wider level.")


def _phrases_of(caption: str) -> list[str]:
    return [p.strip() for p in caption.split(".") if p.strip()]


def plan_caption(scene, scene_caption: str, k: int, first_k: int, last_k: int, content_before: np.ndarray, known_half: np.ndarray | None,
                 region_start: int) -> str:
    """Caption for window k of a scene spanning windows first_k..last_k.

    For each requested topic: planned_cumulative = round((i+1) * target / m) for the i-th of the scene's m windows, `have` = objects of this
    scene already in the finished columns, the window adds `planned_cumulative - have` (0..max_new_per_window).  The caption states
    the objects ALREADY inside the window's known left half plus the new ones, because that is what a 16-column caption means."""
    if not _A["plan"] or not scene.want:
        return scene_caption
    m, i = last_k - first_k + 1, k - first_k
    have = anchors(content_before) if content_before is not None and content_before.shape[1] else {t: [] for t in COUNTED}
    known = anchors(known_half) if known_half is not None else {t: [] for t in COUNTED}
    base = [p for p in _phrases_of(scene_caption) if parse_phrase(p)[0] in ("floor", "ceiling")]
    phrases = list(base)
    for t in COUNTED:
        if t in scene.forbid:
            continue
        n_known = len(known[t])
        new = 0
        if t in scene.want:
            lo, hi = scene.want[t]
            tgt = target_of(lo, hi)
            done = sum(1 for a in have[t] if a >= region_start)
            cum = math.floor((i + 1) * tgt / m + 0.5) if k < last_k else tgt      # +0.5: a single object lands mid-scene, not at its end
            new = max(0, min(int(_A["max_new_per_window"].get(t, 1)), cum - done))
        q = n_known + new
        if q > 0:
            b = N._clamp(t, N.bucket(q), [])
            phrases.append(counted_phrase(t, N.BUCKET_REP[b]))
    return "".join(p + ". " for p in phrases).strip() or str(_A["neutral_caption"])


def evaluate(content: np.ndarray, norm, scene_of: list[int], width: int) -> dict:
    """Count the RAW content and compare with every constraint.  status: met | unmet | not_applicable."""
    cons = constraints(norm)
    if not cons:
        return {"status": "not_applicable", "constraints": [], "unmet": []}
    a = anchors(content)
    bounds = region_bounds(scene_of, width)
    rows = []
    for c in cons:
        lo_c, hi_c = bounds[min(c["scene"], len(bounds) - 1)]
        pos = [x for x in a[c["topic"]] if lo_c <= x < hi_c]
        rows.append({**c, "got": len(pos), "met": c["lo"] <= len(pos) <= c["hi"], "anchors": pos, "region": [lo_c, hi_c]})
    unmet = [r for r in rows if not r["met"]]
    return {"status": "met" if not unmet else "unmet", "constraints": rows, "unmet": unmet}


def window_to_resample(row: dict, scene_of: list[int], n_windows: int) -> int:
    """Which window to resample for one unmet constraint: the window that owns the first surplus object, or for a shortfall the first
    window of the scene's owned columns that holds none of the topic (else the scene's last window)."""
    owner = lambda col: 0 if col < WINDOW else min(n_windows - 1, (col - (WINDOW - STRIDE)) // STRIDE)  # noqa: E731
    if row["got"] > row["hi"]:
        return owner(row["anchors"][row["hi"]])
    ks = [k for k, s in enumerate(scene_of) if s == row["scene"]]
    for k in ks:
        c0 = 0 if k == 0 else k * STRIDE + (WINDOW - STRIDE)
        c1 = WINDOW if k == 0 else c0 + STRIDE
        if not any(c0 <= x < c1 for x in row["anchors"]):
            return k
    return ks[-1]
