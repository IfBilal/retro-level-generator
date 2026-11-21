"""How exactly does the frozen model hit a count INSIDE one 16x16 window?  python scripts/audit_window_exactness.py [--n 30]

Explain-me:
  What: the frozen model was trained on window captions ("one pipe" = one pipe in 16 columns).  This script measures, per caption, how often a single
        sampled window contains exactly the requested number of objects (detector definition), and two diagnostics that explain the audit's planner:
        (1) a caption WITHOUT a topic vs WITH it when the known half already contains that object (does the model repeat it?),
        (2) neutral captions ("full floor." vs "") for how often unrequested objects appear.
  In/Out: -> reports/audit/window_exactness.json (+ printed table).  Uses the same sampler call as the app (fast mode, CFG 7.5, no RePaint).
  Previous: generate/full_level.sample_window.  Next: configs/generate/app.yaml adherence.max_new_per_window / default_floor (the numbers there come from here).
  Try: --n 60 for tighter estimates.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from text2mario.app.service import LevelService
from text2mario.data import tiles as T
from text2mario.generate import adherence as A
from text2mario.generate.full_level import sample_window

EXACT = [("one pipe.", "pipe", 1), ("full floor. one pipe.", "pipe", 1), ("two pipes.", "pipe", 2), ("full floor. two pipes.", "pipe", 2),
         ("one enemy.", "enemy", 1), ("full floor. one enemy.", "enemy", 1), ("two enemies.", "enemy", 2), ("full floor. a few enemies.", "enemy", 3),
         ("one coin.", "coin", 1), ("full floor. a few coins.", "coin", 3), ("one cannon.", "cannon", 1), ("full floor. one cannon.", "cannon", 1),
         ("one platform.", "platform", 1), ("full floor. one platform.", "platform", 1), ("one question block.", "question block", 1),
         ("full floor. one tower.", "tower", 1), ("full floor. one ascending staircase.", "ascending staircase", 1), ("full floor. one coin line.", "coin line", 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    n = ap.parse_args().n
    svc = LevelService()
    out = {"device": svc.device, "n": n, "exact": {}, "known_half": {}, "neutral": {}}

    def draw(cap, seed, known=None):
        return sample_window(svc.bundle, cap, 7.5, 25, 0, seed, known, svc.device)

    for cap, topic, want in EXACT:
        cs = [len(A.anchors(draw(cap, 60000 + s))[topic]) for s in range(n)]
        out["exact"][cap] = {"topic": topic, "want": want, "exact": sum(c == want for c in cs), "dist": {str(k): cs.count(k) for k in sorted(set(cs))}}
        print(f"{cap!r:42} exact {out['exact'][cap]['exact']:3d}/{n}  {out['exact'][cap]['dist']}", flush=True)
    prev = np.zeros((16, 16), np.uint8)
    prev[15, :] = T.GROUND
    prev[11, 10], prev[11, 11] = T.PIPE_TL, T.PIPE_TR
    for r in range(12, 15):
        prev[r, 10], prev[r, 11] = T.PIPE_BL, T.PIPE_BR
    for cap in ("one pipe.", "full floor.", "full floor. one pipe.", "full floor. two pipes."):
        new = [sum(1 for x in A.anchors(draw(cap, 70000 + s, prev[:, 8:]))["pipe"] if x >= 8) for s in range(n)]
        out["known_half"][cap] = {"new_half_pipes": {str(k): new.count(k) for k in sorted(set(new))}}
        print(f"known half has a pipe, caption {cap!r:26} new-half pipes {out['known_half'][cap]['new_half_pipes']}", flush=True)
    for cap in ("full floor.", "", "full floor. one enemy."):
        ws = [A.anchors(draw(cap, 90000 + s)) for s in range(n)]
        out["neutral"][cap] = {"pipes_mean": float(np.mean([len(w["pipe"]) for w in ws])), "enemies_mean": float(np.mean([len(w["enemy"]) for w in ws])),
                               "windows_without_pipe": sum(len(w["pipe"]) == 0 for w in ws)}
        print(f"neutral caption {cap!r:26} {out['neutral'][cap]}", flush=True)
    p = ROOT / "reports/audit/window_exactness.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
