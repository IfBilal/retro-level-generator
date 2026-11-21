"""Gate G-P5: full-level generation (doc 24 §G-P5, doc 20).

Explain-me:
  What: generates L-single-dev = 50 dev prompts x 2 seeds = 100 levels (112 wide, 50 DDIM steps, r = 2, the selected w*),
        and checks P5.01 seam ratio <= 1.3 (level-clustered CI), P5.02 broken structures = 0 after the display repair,
        P5.03 the verify loop never exceeded 5 attempts per window / 20 per level, P5.04 diagnostics.
  In/Out: a conditional checkpoint -> reports/gates/P5/RESULT.json, levels/*.json, renders, seam figure.
  Previous: generate/full_level.py, eval/seams.py.  Next: the app (G-P7) and the freeze.
  Try: --n 10 --width 48 for a quick plumbing run (flagged as not the gate).
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from text2mario.utils.env import bootstrap_env

bootstrap_env()
import numpy as np
import torch
from calibrate_checker import real_levels

from text2mario.eval.devloop import load_prompts
from text2mario.eval.generate import load_ema_conditional
from text2mario.eval.seams import ColumnBigram
from text2mario.eval.stats import cluster_bootstrap
from text2mario.generate.full_level import generate
from text2mario.generate.sanitizer import detect
from text2mario.render.renderer import grid_to_sprite_image
from text2mario.utils.gates import write_result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--w", type=float, default=5.0)
    ap.add_argument("--width", type=int, default=112)
    ap.add_argument("--n", type=int, default=100, help="number of levels (50 prompts x 2 seeds = 100 is the gate)")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--reuse", action="store_true", help="reload saved levels and only recompute the display repair + metrics")
    a = ap.parse_args()
    out = Path(a.out_dir) if a.out_dir else ROOT / "reports/gates/P5"
    (out / "levels").mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    bundle = load_ema_conditional(a.ckpt, dev)
    lv = real_levels(("train",))
    lm = ColumnBigram().fit([g for _, g, _ in lv.values()])
    ref_dev = [lm.seam_ratio(g[:, :a.width + 16])[0] for _, g, _ in real_levels(("dev",)).values() if g.shape[1] >= a.width + 16]
    prompts = load_prompts("p_dev")[:50]
    jobs = [(p, s) for p in prompts for s in (0, 1)][:a.n]
    rows, ratios, clusters = [], [], []
    t0 = time.time()
    for i, (p, s) in enumerate(jobs):
        lf = out / "levels" / f"{i:03d}.json"
        if a.reuse and lf.exists():
            from text2mario.generate.sanitizer import repair
            lvl = json.loads(lf.read_text())
            disp, removed = repair(np.asarray(lvl["raw_grid"], np.uint8))      # the repair is deterministic post-processing
            lvl["display_grid"] = disp.astype(int).tolist()
            lvl["sanitizer"].update({"repaired": bool(removed), "removed_cells": int(removed)})
        else:
            lvl = generate(bundle, p["prompt"], seed=7_000_000 + 10 * i + s, w=a.w, width=a.width, steps=50, resample_r=2, device=dev)
        lf.write_text(json.dumps(lvl))
        grid = np.asarray(lvl["raw_grid"], np.uint8)
        content = grid[:, 3:3 + a.width]
        ratio = lm.seam_ratio(content)[0]
        disp = np.asarray(lvl["display_grid"], np.uint8)
        post = detect(disp[:, 3:3 + a.width])
        pre = lvl["sanitizer"]
        att = lvl["verification"]["attempt_log"]
        per_win = {}
        for e in att:
            per_win[e["window"]] = max(per_win.get(e["window"], 0), e["attempt"])
        rows.append({"i": i, "pid": p["pid"], "level": (p.get("source") or {}).get("level"), "seam_ratio": ratio,
                     "broken_pre": pre["broken_pipes"] + pre["broken_cannons"], "broken_post": post["broken_pipes"] + post["broken_cannons"],
                     "verified": lvl["verification"]["status"] == "verified", "attempts_total": lvl["verification"]["attempts_total"],
                     "max_window_attempts": max(per_win.values()) if per_win else 0, "seconds": lvl["generation"]["seconds"]})
        ratios.append(ratio)
        clusters.append((p.get("source") or {}).get("level"))
        if i % 10 == 0:
            print(f"{i + 1}/{len(jobs)}  seam={ratio:.2f}  verified={rows[-1]['verified']}  {rows[-1]['seconds']}s", flush=True)
    ok_r = [(r, c) for r, c in zip(ratios, clusters) if r == r]
    m, lo, hi = cluster_bootstrap([r for r, _ in ok_r], [c for _, c in ok_r])
    worst = sorted(range(len(rows)), key=lambda i: -rows[i]["seam_ratio"])[:3]
    best = sorted(range(len(rows)), key=lambda i: rows[i]["seam_ratio"])[:3]
    for tag, ids in (("best", best), ("worst", worst)):
        for i in ids:
            lv_ = json.loads((out / "levels" / f"{i:03d}.json").read_text())
            grid_to_sprite_image(np.asarray(lv_["display_grid"], np.uint8)).save(out / f"render_{tag}_{i:03d}.png")
    checks = [{"id": "P5.01", "name": "seam ratio (mean over levels)", "value": round(m, 4), "ci95": [round(lo, 4), round(hi, 4)], "op": "<=", "threshold": 1.3,
               "pass": m <= 1.3},
              {"id": "P5.02", "name": "broken structures after display repair", "value": int(sum(r["broken_post"] for r in rows)), "op": "==", "threshold": 0,
               "pass": sum(r["broken_post"] for r in rows) == 0},
              {"id": "P5.03", "name": "verify-loop limits (<=5 per window, <=20 per level)", "value": {"max_window": max(r["max_window_attempts"] for r in rows),
                                                                                                   "max_total": max(r["attempts_total"] for r in rows)},
               "op": "<=", "pass": max(r["max_window_attempts"] for r in rows) <= 5 and max(r["attempts_total"] for r in rows) <= 21}]
    hist = {}
    for r in rows:
        hist[r["attempts_total"]] = hist.get(r["attempts_total"], 0) + 1
    diag = {"n_levels": len(rows), "width": a.width, "w": a.width and a.w, "real_dev_pseudo_seam_ratio_mean": float(np.nanmean(ref_dev)) if ref_dev else None,
            "pre_repair_broken_rate": float(np.mean([r["broken_pre"] > 0 for r in rows])), "verified_rate": float(np.mean([r["verified"] for r in rows])),
            "attempts_histogram": hist, "seconds_per_level_mean": float(np.mean([r["seconds"] for r in rows])),
            "seconds_per_level_p95": float(np.quantile([r["seconds"] for r in rows], 0.95)), "total_minutes": (time.time() - t0) / 60,
            "is_full_gate": len(rows) == 100 and a.width == 112}
    (out / "per_level.json").write_text(json.dumps(rows, indent=1))
    write_result("P5", all(c["pass"] for c in checks) and diag["is_full_gate"], {"checkpoint": str(a.ckpt), "w": a.w}, checks, out_dir=out, diagnostics=diag)
    print(json.dumps({c["id"]: (c["value"], c["pass"]) for c in checks}, default=str))
    print(json.dumps({k: diag[k] for k in ("verified_rate", "pre_repair_broken_rate", "seconds_per_level_mean")}))
    print("G-P5", "PASS" if all(c["pass"] for c in checks) and diag["is_full_gate"] else "FAIL / partial")


if __name__ == "__main__":
    main()
