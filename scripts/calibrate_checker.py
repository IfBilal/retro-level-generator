"""Calibrate the playability checker on REAL train+dev levels (doc 21 §6, doc 27 §4.5).  Never touches test levels.

Explain-me:
  What: runs the tier-1 checker on every train/dev real level at full width and on 16-wide windows (framed with the
        flat spawn/end columns of doc 20), prints the solved rates and writes reports/gates/P5b/calibration.json.
        Failures are listed with the column where the search got stuck, for manual classification.
  In/Out: reads data/processed (scenes, index) + data/splits; writes the JSON.  Previous: playability/checker.py.
  Try: --enemies hazard for the conservative enemy model; edit physics.json + derive_physics.py to see the rate move.
Real-level conventions: spawn = the lowest standable cell of column 0; goal = the column before the tall solid wall
that VGLC draws for the flagpole near the end of most levels (an artefact of the tile set), else the last column.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from text2mario.data import splits
from text2mario.data.index import level_grid
from text2mario.playability.checker import Framing, check_level, frame_window
from text2mario.playability.physics import Physics


def real_levels(which=("train", "dev")):
    """-> {level_id: (name, full grid, split)} for the requested splits only (the test split is refused here)."""
    assert "test" not in which, "calibration must never look at test levels"
    S = splits.load(verify=True)
    ids = sorted(l for w in which for l in S["splits"][w])
    scenes = np.load(ROOT / "data/processed/scenes_u8.npy")
    df = pd.read_parquet(ROOT / "data/processed/index.parquet")
    names = {l["level_id"]: l["name"] for l in S["levels"]}
    out = {}
    for lid in ids:
        rows = df[df.level_id == lid].sort_values("col_offset")
        a, b = int(rows.idx.min()), int(rows.idx.max()) + 1
        out[lid] = (names[lid], level_grid(scenes, a, b), S["levels"][lid]["split"])
    return out


def real_framing(g: np.ndarray, ph: Physics):
    """(spawn, goal_col) of a real full level, or None when column 0 has no standable cell."""
    solid = np.isin(g, list(ph.solid))
    h, w = g.shape
    rows = [r for r in range(h - 1) if not solid[r, 0] and solid[r + 1, 0]]
    if not rows:
        return None
    goal = w - 1
    for c in range(max(0, w - 40), w):   # flagpole artefact: a wall reaching high up near the end
        if solid[:, c].sum() >= 9 and solid[6, c]:
            goal = c - 1
            break
    return Framing((0, max(rows)), goal)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=16, help="window stride for the 16-wide windows")
    ap.add_argument("--budget", type=int, default=20000)
    ap.add_argument("--enemies", default="ignore", choices=["ignore", "hazard"])
    ap.add_argument("--out", default=str(ROOT / "reports/gates/P5b/calibration.json"))
    a = ap.parse_args()
    ph = Physics.load()
    lv = real_levels()
    rep = {"physics_sha256": ph.sha256, "enemies": a.enemies, "levels": [], "windows": {}}
    t0, solved = time.time(), 0
    for lid, (name, g, sp) in lv.items():
        fr = real_framing(g, ph)
        r = check_level(g, mode="level", framing=fr, physics=ph, budget_nodes=a.budget, enemies=a.enemies) if fr else None
        status = r.status if r else "no_spawn"
        solved += status == "solved"
        rep["levels"].append({"level_id": lid, "name": name, "split": sp, "width": g.shape[1], "status": status,
                              "goal_col": fr.flag_col if fr else None, "stuck_at_col": r.fail_col if r else None,
                              "nodes": r.nodes if r else 0, "seconds": round(r.seconds, 2) if r else 0})
        print(f"{lid:2d} {name:10s} {sp:5s} w={g.shape[1]:3d} {status:8s} goal={fr.flag_col if fr else -1:3d} stuck_at={r.fail_col if r else -1:3d}")
    rep["full_level_solved_rate"] = solved / len(lv)
    print(f"full levels solved: {solved}/{len(lv)} = {solved / len(lv):.3f}   ({time.time() - t0:.0f}s)")
    st, n, t1 = {"solved": 0, "unsolved": 0, "timeout": 0}, 0, time.time()
    for lid, (name, g, sp) in lv.items():
        for c0 in range(0, g.shape[1] - 15, a.stride):
            fg, fr = frame_window(g[:, c0:c0 + 16])
            st[check_level(fg, mode="level", framing=fr, physics=ph, budget_nodes=a.budget, enemies=a.enemies).status] += 1
            n += 1
    rep["windows"] = {"n": n, **st, "solved_rate": st["solved"] / n, "timeout_rate": st["timeout"] / n}
    print("framed 16-wide windows", rep["windows"], f"({time.time() - t1:.0f}s)")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
