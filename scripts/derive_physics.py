"""Derive the checker's jump envelopes from the physics constants (doc 27 §4.3, doc 21 §5).

Explain-me:
  What: re-implements the game's per-step integrator (semi-implicit Euler, 60 Hz, tile units, y up), simulates a
        full jump and writes the `derived` block of app/static/game/physics.json (apex, airtime, reach, and the
        max gap / rise the checker may assume).  Run with --check to verify the stored block is not stale.
  In/Out: reads the base constants of physics.json -> rewrites `derived`.  The checker only reads `derived`.
  Previous: doc 27 §4.1 (owner of the numbers).  Next: playability/checker.py, app/static/game/game.js.
  Try: set jump_velocity to 20.5 in physics.json, run this script, and watch checker_max_gap_* shrink.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

PHYSICS = Path(__file__).resolve().parents[1] / "app" / "static" / "game" / "physics.json"
EPS = 1e-9  # values exactly on a floor boundary are resolved by the arithmetic below, never by hand


def jump_trace(p: dict):
    """Per-step heights of a full jump: v = jump_velocity; each step v -= g*dt, y += v*dt (until back at y <= 0)."""
    dt, g, v, y = 1.0 / p["fps"], p["gravity"], p["jump_velocity"], 0.0
    ys = [0.0]
    while True:
        v = max(v - g * dt, -p["max_fall_speed"])
        y += v * dt
        ys.append(y)
        if y <= 0.0 and v < 0:
            return ys, dt


def air_time_to(ys: list[float], dt: float, rise: int) -> float | None:
    """Seconds until the descending arc first reaches height `rise` (negative = lands lower); None if never."""
    apex_i = max(range(len(ys)), key=lambda i: ys[i])
    if rise > ys[apex_i]:
        return None
    for i in range(apex_i, len(ys) - 1):  # descending part; linear interpolation between steps
        if ys[i] >= rise >= ys[i + 1]:
            frac = 0.0 if ys[i] == ys[i + 1] else (ys[i] - rise) / (ys[i] - ys[i + 1])
            return (i + frac) * dt
    # lands lower than the end of the trace: continue free fall
    return None


def full_trace(p: dict, max_rise_below: int = 4):
    """Trace long enough to reach `-max_rise_below` below take-off."""
    dt, g, v, y = 1.0 / p["fps"], p["gravity"], p["jump_velocity"], 0.0
    ys = [0.0]
    while y > -max_rise_below - 1:
        v = max(v - g * dt, -p["max_fall_speed"])
        y += v * dt
        ys.append(y)
    return ys, dt


def derive(p: dict) -> dict:
    ys, dt = full_trace(p)
    apex = max(ys)
    apex_i = ys.index(apex)
    same_i = next(i for i in range(apex_i, len(ys)) if ys[i] <= 0.0)
    airtime = same_i * dt  # whole integrator steps back at take-off height (doc 27 §4.3: 51 steps = 0.850 s)
    bw, mh, mv = p["player"]["body_w"], p["checker_margins"]["horizontal"], p["checker_margins"]["vertical"]
    speeds = {"walk": p["walk_speed"], "run": p["run_speed"]}
    max_rise = math.floor(apex - mv + EPS)

    def gap(speed: float, rise: int) -> int:
        t = air_time_to(ys, dt, rise)
        return -1 if t is None else math.floor(speed * t + bw - mh + EPS)

    out = {
        "generated_by": "scripts/derive_physics.py",
        "apex_rise_full": round(apex, 3), "apex_time_s": round(apex_i * dt, 3),
        "airtime_same_height_s": round(airtime, 3),
        "jump_distance_walk": round(speeds["walk"] * airtime, 2), "jump_distance_run": round(speeds["run"] * airtime, 2),
        "checker_max_rise": max_rise,
        "checker_max_gap_same_height": {k: gap(v, 0) for k, v in speeds.items()},
        "checker_max_gap_landing_lower": {k: {str(d): gap(v, -d) for d in range(1, 5)} for k, v in speeds.items()},
        "checker_max_gap_landing_higher": {k: {str(d): gap(v, d) for d in range(1, 5)} for k, v in speeds.items()},
    }
    return out


def physics_sha256(p: dict) -> str:
    """sha256 of the canonical JSON of the whole file (doc 27 §4.1)."""
    return hashlib.sha256(json.dumps(p, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="fail if the stored derived block differs")
    ap.add_argument("--path", default=str(PHYSICS))
    a = ap.parse_args()
    path = Path(a.path)
    p = json.loads(path.read_text(encoding="utf-8"))
    d = derive(p)
    if a.check:
        keep = {k: v for k, v in p["derived"].items() if k in d}
        ok = keep == d
        print("derived block", "matches" if ok else "is STALE", "| sha256", physics_sha256(p)[:16])
        if not ok:
            print(json.dumps(d, indent=1))
        return 0 if ok else 1
    p["derived"] = d
    path.write_text(json.dumps(p, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(d, indent=1)); print("physics_sha256", physics_sha256(p))
    return 0


if __name__ == "__main__":
    sys.exit(main())
