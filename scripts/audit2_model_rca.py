"""Model root-cause measurements for the second audit (H1 scope, H3 counting, H6 CFG): python scripts/audit2_model_rca.py

Explain-me:
  What: runs the frozen model with the adherence planner/verifier OFF (so we see the raw model) on the caption "full floor. one pipe." etc.
        and counts objects in the RAW grid with the repo's detector.  H1: counts vs number of windows (width 16/48/112).  H3: asked bucket
        vs got count at width 16 (one window).  H6: the same at several CFG scales.  Seeds are fixed, so the run is repeatable.
  In/Out: -> reports/audit2/model_rca.json and MODEL_RCA_tables.md.   Previous: eval/count_audit.py.  Next: reports/audit2/MODEL_RCA.md.
  Try: change SEEDS or WIDTHS below.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from text2mario.app.service import LevelService
from text2mario.generate import adherence
from text2mario.generate.adherence import anchors

SEEDS = list(range(1, 21))
TOPICS = {"one pipe": "pipe", "two pipes": "pipe", "one enemy": "enemy", "two enemies": "enemy", "a few enemies": "enemy"}


def count(svc, caption, topic, seed, width, w):
    lvl = svc.run(caption, seed=seed, w=w, width=width, fast=True, verify=False).level
    return len(anchors(np.array(lvl["raw_grid"]))[topic])


def main():
    adherence._A["enabled"] = False                       # the raw model: no per-window planning, no verification (the situation of the first audit)
    svc = LevelService()
    out = {"h1": {}, "h3": {}, "h6": {}}
    t0 = time.time()
    for cap, topic in list(TOPICS.items())[:1] + [("two enemies", "enemy")]:
        for width in (16, 48, 112):
            c = [count(svc, "full floor. " + cap + ".", topic, s, width, 7.5) for s in SEEDS]
            out["h1"][f"{cap}@{width}"] = {"mean": float(np.mean(c)), "counts": c}
    for cap, topic in TOPICS.items():
        c = [count(svc, "full floor. " + cap + ".", topic, s, 16, 7.5) for s in SEEDS]
        out["h3"][cap] = {"hist": {str(k): int(v) for k, v in zip(*np.unique(c, return_counts=True), strict=True)}, "counts": c}
    for w in (1.0, 3.0, 5.0, 7.5, 10.0):
        c = [count(svc, "full floor. one pipe.", "pipe", s, 16, w) for s in SEEDS]
        out["h6"][str(w)] = {"exactly_one": float(np.mean([x == 1 for x in c])), "mean": float(np.mean(c))}
    out["seconds"] = round(time.time() - t0, 1)
    (ROOT / "reports/audit2/model_rca.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "h3"}, indent=1)[:3000])
    print(json.dumps({k: v["hist"] for k, v in out["h3"].items()}))


if __name__ == "__main__":
    main()
