"""Measure the app's latency BEFORE pinning any threshold (CLAUDE.md rule 14, ledger G-P7): cold start and per-level time.

Explain-me:
  What: loads the model like the app does (cold start), then times full generate + verify + render for several prompts at widths
        48 and 112, on the GPU (full quality, 50 steps r=2) and on the CPU (fast mode, 25 steps r=0); reports median and p95.
  In/Out: a checkpoint or exported bundle -> reports/app/latency.json.  Previous: app/service.py.  Next: the report quotes the file.
  Try: python scripts/bench_app.py --prompts 20 --ckpt app/model     (the CPU numbers describe THIS machine, not the cloud)
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PROMPTS = ["full floor with two pipes and a few enemies", "flat ground, one pipe, then several coins and a staircase",
           "an underground level: three platforms, then a cannon and two enemies", "mostly pits with two islands of ground, then a coin line",
           "two pipes followed by a coin line", "a few question blocks and several enemies", "full floor. one ascending staircase. two towers.",
           "lots of coins and a couple of goombas", "begin with a platform and end with stairs going down", "three pipes and no enemies"]


def stats(xs):
    return {"p50": float(np.median(xs)), "p95": float(np.quantile(xs, 0.95)), "n": len(xs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--prompts", type=int, default=10)
    ap.add_argument("--cpu", action="store_true", help="also time the CPU fast mode")
    ap.add_argument("--skip-full", action="store_true", help="skip the slow full-quality timings")
    a = ap.parse_args()
    from text2mario.app.service import LevelService
    out = {}
    t0 = time.time()
    svc = LevelService(a.ckpt)
    out["cold_start_s"] = round(time.time() - t0, 2)
    out["device"] = svc.device
    out["fast_mode_default"] = True
    svc.run(PROMPTS[0], seed=0, width=48)                       # warm-up (cuDNN autotune, caches)
    for width in (48, 112):
        if not a.skip_full:
            ts = [svc.run(PROMPTS[i % len(PROMPTS)], seed=i, width=width).seconds["total"] for i in range(a.prompts)]
            for k, v in stats(ts).items():
                out[f"{svc.device}_{width}_{k}"] = v
        tf = [svc.run(PROMPTS[i % len(PROMPTS)], seed=i, width=width, fast=True).seconds["total"] for i in range(a.prompts)]
        for k, v in stats(tf).items():                       # the app's default: 25 steps, no resampling, checker budget 1000
            out[f"{svc.device}_{width}_fast_{k}"] = v
    if a.cpu and svc.device != "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        cpu = LevelService(a.ckpt, device="cpu")
        for width in (48, 112):
            ts = [cpu.run(PROMPTS[i % len(PROMPTS)], seed=i, width=width, fast=True).seconds["total"] for i in range(min(a.prompts, 5))]
            out[f"cpu_{width}_p50"], out[f"cpu_{width}_p95"] = stats(ts)["p50"], stats(ts)["p95"]
    out["note"] = "measured on this machine; cloud CPU numbers are measured separately after deployment"
    old = json.loads((ROOT / "reports/app/latency.json").read_text()) if (ROOT / "reports/app/latency.json").exists() else {}
    old.update({k: v for k, v in out.items() if not isinstance(v, dict)})          # merge: keep earlier full-quality numbers
    (ROOT / "reports/app/latency.json").write_text(json.dumps(old, indent=1))
    print(json.dumps({k: v for k, v in out.items() if not isinstance(v, dict)}, indent=1))


if __name__ == "__main__":
    main()
