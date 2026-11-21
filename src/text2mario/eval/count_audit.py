"""Object counting on a RAW generated grid, used only by the adherence audit (scripts/audit_adherence.py).

Explain-me:
  What: counts "how many pipes / enemies / ..." in a level and checks a prompt's expectations.
        (1) detector counts: the repo's own caption detector (the definition the training captions used: pipes = connected
        components, enemies / coins / cannons = cells, platforms = runs, staircases = diagonal runs).
        (2) an independent re-count (pipes = vertical runs of left-half pipe tiles; enemies, coins, cannons, question blocks = tiles)
        to catch a wrong detector.  An object belongs to a scene by its LEFTMOST column: scene s owns [start_s, start_{s+1}).
  In/Out: grid (16, W) uint8 content columns -> anchor columns per topic;  check_expectation(...) -> (ok, details).
  Previous: generate/full_level.py (raw_grid).  Next: scripts/audit_adherence.py, tests/test_count_audit.py.
  Try: edit quantity_ranges in configs/audit/adherence.yaml (e.g. several -> 4-9) and re-run the audit.
"""
from __future__ import annotations

import numpy as np

from text2mario.captions.detector import detect
from text2mario.data import tiles as T

BLOCKS = frozenset(T.SOLID) - frozenset(T.PIPE) - frozenset(T.CANNON)


def visible_towers(grid: np.ndarray) -> list[int]:
    """Leftmost column of every tower a PERSON would call a tower: a group of 1-2 adjacent solid columns, each at least 3 blocks above the floor row
    and standing on it, whose left and right neighbour columns are at least 2 blocks lower.  The caption detector (the training-caption definition)
    also calls the narrow top of a staircase a tower and merges neighbouring pillars; on 30 'N towers' levels it disagreed with what is visible in 10
    (reports/audit/NOTES_towers.md), so towers are counted with this rule (ledger AD9) and the detector's count is kept only as a cross-check."""
    g = np.asarray(grid)
    h, w = g.shape
    height = []
    for c in range(w):
        k = h - 1
        while k >= 0 and int(g[k, c]) in BLOCKS:                 # plain blocks only: pipes and cannons are not towers
            k -= 1
        height.append((h - 1) - k - 1)                           # solid blocks above the floor row in this column (-1 when there is no floor)
    out, c = [], 0
    while c < w:
        if height[c] < 3:
            c += 1
            continue
        c2 = c
        while c2 + 1 < w and height[c2 + 1] >= 3:
            c2 += 1
        low = min(height[c:c2 + 1])
        left = height[c - 1] if c > 0 else 0
        right = height[c2 + 1] if c2 + 1 < w else 0
        if c2 - c + 1 <= 2 and left <= low - 2 and right <= low - 2:
            out.append(c)
        c = c2 + 1
    return out


def detector_objects(grid: np.ndarray, visible_tower_rule: bool = True) -> dict[str, list[int]]:
    """Detector topic -> anchor column (leftmost column) of each counted object.

    enemies / coins / cannons / question blocks are one object per cell (that is how the detector counts them).  Every other
    topic arrives as ONE Detection holding d.count objects and the union of their cells, so the union is split into d.count
    column groups; if the split does not give d.count groups the detector count stays authoritative and anchors are spread evenly."""
    out: dict[str, list[int]] = {}
    for d in detect(np.asarray(grid)):
        if d.count is None or not d.cells:
            continue
        cells = sorted(d.cells, key=lambda rc: (rc[1], rc[0]))
        if d.topic in ("enemy", "coin", "cannon", "question block"):
            out.setdefault(d.topic, []).extend(c for _, c in cells)
            continue
        cols = sorted({c for _, c in cells})
        groups, cur = [], [cols[0]]
        for c in cols[1:]:
            if c - cur[-1] > 1:
                groups.append(cur)
                cur = [c]
            else:
                cur.append(c)
        groups.append(cur)
        if len(groups) == d.count:
            anchors = [g[0] for g in groups]
        else:
            step = max(1, len(cols) // d.count)
            anchors = [cols[min(i * step, len(cols) - 1)] for i in range(d.count)]
        out.setdefault(d.topic, []).extend(anchors)
    if visible_tower_rule:
        out["tower (detector)"] = out.pop("tower", [])             # kept for the cross-check
        out["tower"] = visible_towers(grid)
    return out


def independent_counts(grid: np.ndarray) -> dict[str, int]:
    """A second, simpler counter used to cross-check the detector (it does not call the detector)."""
    g = np.asarray(grid)
    h, w = g.shape
    pipes = 0
    for c in range(w):                    # a pipe has exactly one LEFT column: one vertical run of left-half tiles
        run = False
        for r in range(h):
            left = int(g[r, c]) in (T.PIPE_TL, T.PIPE_BL)
            if left and not run:
                pipes += 1
            run = left
    return {"pipe": pipes, "enemy": int((g == T.ENEMY_ID).sum()), "coin": int((g == T.COIN).sum()),
            "cannon": int((g == T.CANNON_TOP).sum()), "question block": int(np.isin(g, [T.QBLOCK, T.QEMPTY]).sum())}


def objects_by_topic(grid: np.ndarray, topics_cfg: dict) -> dict[str, list[int]]:
    """Audit topic (pipe, staircase...) -> sorted anchor columns, summing the detector topics listed in the config."""
    det = detector_objects(grid)
    return {name: sorted(a for t in spec["det"] for a in det.get(t, [])) for name, spec in topics_cfg.items()}


def scene_starts(prompt_scenes: list[dict], stride: int, window: int = 16) -> list[int]:
    """First content column each scene OWNS.  Window k adds the 8 columns after the previous window (window 0 owns columns 0-15), so a
    scene whose first window is k > 0 owns columns from k*stride + (window - stride).  The scene of a window is the "scene" index the
    generator records; older levels without it use "the caption changed since the previous window"."""
    firsts: list[int] = []
    sid, prev_cap = -1, None
    for k, s in enumerate(prompt_scenes):
        if "scene" in s:
            new_scene = s["scene"] != sid
            sid = s["scene"]
        else:
            new_scene = s["caption"] != prev_cap
            sid += 1 if new_scene else 0
        prev_cap = s["caption"]
        if new_scene:
            firsts.append(k)
    return [0 if k == 0 else k * stride + (window - stride) for k in firsts]


def check_expectation(anchors: dict[str, list[int]], expect: list[dict], starts: list[int], width: int, order: list[str] | None = None):
    """Compare per-scene expected ranges with the anchors found.  Returns (ok, details).

    If the generator produced a different number of scenes than the prompt asks for, scene-by-scene judging is impossible, so the
    expectations are merged and judged on the whole level (and the mismatch is recorded in details)."""
    details = {"scene_count_expected": len(expect), "scene_count_actual": len(starts)}
    if len(starts) != len(expect):
        details["note"] = "scene count differs from the prompt: judged on the whole level"
        merged: dict[str, list[int]] = {}
        for sc in expect:
            for t, rng in sc.items():
                lo, hi = merged.get(t, [0, 0])
                merged[t] = [lo + rng[0], hi + rng[1]]
        expect, starts = [merged], [0]
    bounds = list(starts) + [width]
    ok, per = True, []
    for i, sc in enumerate(expect):
        for topic, (lo, hi) in sc.items():
            n = sum(1 for a in anchors.get(topic, []) if bounds[i] <= a < bounds[i + 1])
            good = lo <= n <= hi
            per.append({"scene": i, "topic": topic, "want": [lo, hi], "got": n, "ok": good})
            ok &= good
    details["checks"] = per
    if order:
        firsts = [min(anchors[t]) if anchors.get(t) else None for t in order]
        seq = [f for f in firsts if f is not None]
        good = None not in firsts and seq == sorted(seq) and len(set(seq)) == len(seq)
        details["order"] = {"topics": order, "first_cols": firsts, "ok": bool(good)}
        ok &= good
    return bool(ok), details


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval for k passes in n levels (each seed is its own generated level, so a level is the cluster; rule 16)."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, min(p, c - h)), min(1.0, max(p, c + h)))        # float rounding must never push the interval off its own point estimate
