"""Agreement of the tier-1 checker with MarioEval.jar on REAL train+dev windows and levels (doc 21 §7, G-P5b support).

Explain-me:
  What: runs the jar once per item (verdicts cached by grid hash in reports/gates/P5b/jar_cache.json), runs the fast
        checker on the same items for each requested variant, and prints agreement, Cohen's kappa and the
        false-'verified' rate (fast=solved, jar=not beaten).  Only train/dev levels are used (never test).
  In/Out: reads data + split, writes the cache and reports/gates/P5b/agreement_<tag>.json.
  Previous: playability/jar_bridge.py + checker.py.  Next: choosing the enemy mode / physics before the freeze.
  Try: --enemies hazard to see how much stricter the conservative enemy model is.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from calibrate_checker import real_framing, real_levels  # noqa: E402

from text2mario.playability.checker import check_level, frame_window  # noqa: E402
from text2mario.playability.jar_bridge import jar_solve_many  # noqa: E402
from text2mario.playability.physics import Physics  # noqa: E402

CACHE = ROOT / "reports/gates/P5b/jar_cache.json"


def key(g: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(g).tobytes() + bytes(g.shape)).hexdigest()[:20]


def items(stride: int):
    """[(label, grid, framing)] : every train/dev window (16 wide) framed with the flat spawn/end columns (doc 20 G3),
    so both solvers share the same spawn and goal; plus every full level (spawn/goal as in calibrate_checker)."""
    out = []
    ph = Physics.load()
    for lid, (name, g, sp) in real_levels().items():
        fr = real_framing(g, ph)
        if fr is not None:
            out.append((f"L{lid}:{name}:full", g, fr))
        for c0 in range(0, g.shape[1] - 15, stride):
            fg, ff = frame_window(g[:, c0:c0 + 16])
            out.append((f"L{lid}:{name}:w{c0}", fg, ff))
    return out


def kappa(a: list[bool], b: list[bool]) -> float:
    a, b = np.array(a), np.array(b)
    po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--enemies", default="ignore", choices=["ignore", "hazard"])
    ap.add_argument("--no-full", action="store_true", help="skip full-width levels (slow in the jar)")
    ap.add_argument("--strip-enemies", action="store_true", help="replace enemy tiles by sky for the jar (structure-only comparison)")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    its = [it for it in items(a.stride) if not (a.no_full and it[0].endswith(":full"))]
    if a.strip_enemies:  # the fast checker runs on the original grid with enemies='ignore'; the jar sees no enemies
        its = [(lab, np.where(g == 5, 0, g).astype(g.dtype), fr) for lab, g, fr in its]
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    todo = [(lab, g) for lab, g, _ in its if key(g) not in cache]
    print(f"{len(its)} items, {len(todo)} need the jar")
    t0 = time.time()
    for i in range(0, len(todo), 48):  # chunks so the cache survives an interruption
        chunk = todo[i:i + 48]
        for (lab, g), r in zip(chunk, jar_solve_many([g for _, g in chunk], workers=a.workers, timeout=120)):
            cache[key(g)] = r
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(cache))
        print(f"  jar {min(i + 48, len(todo))}/{len(todo)}  {time.time() - t0:.0f}s", flush=True)
    ph = Physics.load()
    rows = []
    for lab, g, fr in its:
        j = cache[key(g)]
        f = check_level(g, mode="level", framing=fr, physics=ph, enemies=a.enemies).status == "solved"
        rows.append((lab, f, j))
    ok = [(l, f, j) for l, f, j in rows if j is not None]
    fa, ja = [f for _, f, _ in ok], [bool(j) for _, _, j in ok]
    agree = float(np.mean([x == y for x, y in zip(fa, ja)]))
    fv = [f and not j for _, f, j in ok]
    rep = {"enemies": a.enemies, "physics_sha256": ph.sha256, "n": len(rows), "jar_timeouts": len(rows) - len(ok),
           "jar_solved_rate": float(np.mean(ja)), "fast_solved_rate": float(np.mean(fa)), "agreement": agree,
           "kappa": kappa(fa, ja), "false_verified_rate_of_fast_solved": float(sum(fv) / max(1, sum(fa))),
           "fast_solved_jar_failed": int(sum(fv)), "fast_failed_jar_solved": int(sum((not f) and j for _, f, j in ok))}
    full = [(l, f, j) for l, f, j in ok if l.endswith(":full")]
    if full:
        rep["full_levels"] = {"n": len(full), "jar_solved": int(sum(bool(j) for _, _, j in full)),
                              "fast_solved": int(sum(f for _, f, _ in full))}
    print(json.dumps(rep, indent=1))
    (CACHE.parent / f"agreement_{a.enemies}{a.tag}.json").write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
