"""Determinism evidence (rubric row 7: consistent outputs): same prompt + seed + settings -> the same level; a different seed -> a different level.

Explain-me:
  What: with the exported model it (1) generates the same level twice and compares the raw grids, (2) changes only the seed and checks that the level
        changes, (3) samples one window alone and inside a batch (per-sample seeding) and reports how many cells agree, (4) runs the normalizer and the
        checker twice.  Writes reports/gates/determinism.json.  Machine-specific: cross-machine bit equality is not claimed (NFR-REP-05).
  In/Out: app/model -> reports/gates/determinism.json.  Previous: generate/full_level.py.  Try: python scripts/check_determinism.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import torch

from text2mario.app.service import LevelService
from text2mario.eval.generate import sample_conditional
from text2mario.playability.checker import Framing, check_level
from text2mario.text.normalizer import normalize


def main():
    svc = LevelService()
    p = "flat ground, one pipe, then several coins and a staircase"
    a = svc.run(p, seed=11, width=48).level
    b = svc.run(p, seed=11, width=48).level
    c = svc.run(p, seed=12, width=48).level
    same = a["raw_grid"] == b["raw_grid"]
    diff = float(np.mean(np.asarray(a["raw_grid"]) != np.asarray(c["raw_grid"])))
    unet, sched, cond = svc.bundle[0], svc.bundle[1], svc.bundle[2]
    dev = svc.device
    ac = torch.bfloat16 if dev == "cuda" else None
    cap = "full floor. two pipes."
    alone = sample_conditional(unet, sched, cond, [cap], 5.0, [77], 50, device=dev, autocast_dtype=ac)[0]
    in_batch = sample_conditional(unet, sched, cond, ["full floor.", cap, "one platform."], 5.0, [5, 77, 9], 50, device=dev, autocast_dtype=ac)[1]
    g = np.asarray(a["raw_grid"], np.uint8)
    fr = a["framing"]
    fr_obj = Framing((fr["spawn"]["col"], fr["spawn"]["row"]), fr["flag"]["col"])
    r1, r2 = check_level(g, mode="level", framing=fr_obj), check_level(g, mode="level", framing=fr_obj)
    out = {"device": dev, "same_seed_same_level": bool(same), "different_seed_share_of_cells_changed": round(diff, 3),
           "window_alone_vs_in_batch_cells_equal": round(float((alone == in_batch).mean()), 4),
           "normalizer_repeatable": [s.caption for s in normalize(p).scenes] == [s.caption for s in normalize(p).scenes],
           "checker_repeatable": r1.path == r2.path and r1.status == r2.status,
           "note": "same machine and settings; bf16 batching can change a few cells, cross-machine equality is not claimed"}
    (ROOT / "reports/gates").mkdir(parents=True, exist_ok=True)
    (ROOT / "reports/gates/determinism.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    assert same and diff > 0.02 and out["normalizer_repeatable"] and out["checker_repeatable"]


if __name__ == "__main__":
    main()
