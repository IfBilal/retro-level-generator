"""Gate G-P3 (unconditional baseline) + the linear-vs-cosine schedule choice (doc 24 §G-P3, ledger M2/E4/§9).

Explain-me:
  What: for each run (U-lin, U-cos) takes its milestone checkpoints (5k/10k/15k), draws 256 EMA samples with DDIM 50,
        and scores them on RAW samples (no repair): floor validity, broken-structure rate, tile-JS, copy rate, novelty.
        The checkpoint per schedule is the one with the best S_noplay = 0.5*floor_valid + 0.25*(1-broken) + 0.25*min(1, novelty/ref)
        (the `playable` term is dropped until the checker is validated, ledger E4).  A schedule PASSES if P3.01-P3.04 hold;
        the winner is the better-S passing schedule, a tie (|dS| < 0.01) goes to cosine (M2).
  In/Out: runs/U-lin, runs/U-cos (+ reports/references/REFERENCES.json) -> reports/gates/P3/RESULT.json, sample sheets,
          nearest-train figures.  Thresholds come from configs/gates.yaml.
  Try: python scripts/gate_p3.py --runs runs/U-lin runs/U-cos            (the real gate; ~minutes per checkpoint on a GPU)
       python scripts/gate_p3.py --runs <one run> --n 16 --steps 5 --out-dir /tmp/p3test   (plumbing test, not a result)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from text2mario.utils.env import bootstrap_env  # noqa: E402

bootstrap_env()
from PIL import Image  # noqa: E402

from text2mario.data.dataset import MarioWindows  # noqa: E402
from text2mario.data.hamming import nn_distance  # noqa: E402
from text2mario.eval import structure as S  # noqa: E402
from text2mario.eval.generate import load_ema_model, sample_unconditional  # noqa: E402
from text2mario.render.renderer import grid_to_image, sheet  # noqa: E402
from text2mario.utils.gates import require, write_result  # noqa: E402


def s_noplay(floor_valid: float, broken: float, novelty_med: float, novelty_ref: float) -> float:
    """Phase-3 composite without the playable term (weights renormalised to 0.5 / 0.25 / 0.25, ledger E4)."""
    return 0.5 * floor_valid + 0.25 * (1 - broken) + 0.25 * min(1.0, novelty_med / novelty_ref)


def evaluate_checkpoint(path: Path, train: np.ndarray, refs: dict, gcfg: dict, steps: int, n: int, device: str):
    unet, sched, cfg, state = load_ema_model(path, device)
    grids = sample_unconditional(unet, sched, n, gcfg["seed_base"], steps, device=device,
                                 autocast_dtype=torch.bfloat16 if device == "cuda" else None)
    nov = S.novelty(grids, train)
    m = {"step": state["step"], "schedule": cfg["diffusion"]["schedule"], "n": len(grids),
         "floor_valid": S.floor_valid_rate(grids), "broken_rate": S.broken_rate(grids), "tile_js": S.tile_js(grids, train),
         "novelty_med": nov["novelty_med"], "floorless_share": S.floorless_share(grids), **S.copy_rates(nov["nn"])}
    m["S_noplay"] = s_noplay(m["floor_valid"], m["broken_rate"], m["novelty_med"], refs["novelty_ref_median_stride8"])
    return m, grids, nov["nn"], state


def checks_for(m: dict, refs: dict, g: dict) -> list[dict]:
    copy_thr = refs["copy_rate_ref"] + g["copy_margin"]
    rows = [("P3.01", "floor_valid", m["floor_valid"], ">=", g["floor_valid_min"]),
            ("P3.02", "broken_rate", m["broken_rate"], "<=", g["broken_rate_max"]),
            ("P3.03", "tile_js", m["tile_js"], "<=", g["tile_js_max"]),
            ("P3.04", "copy_rate", m["copy_rate"], "<=", round(copy_thr, 4))]
    return [{"id": i, "name": n, "value": round(v, 4), "op": op, "threshold": t, "n": m["n"],
             "pass": bool(v >= t if op == ">=" else v <= t), "waivable": False} for i, n, v, op, t in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="run directories (U-lin, U-cos)")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--out-dir", default=None, help="write results here instead of reports/gates/P3 (plumbing tests)")
    ap.add_argument("--ckpt-glob", default="step_*.pt")
    a = ap.parse_args()
    out_dir = Path(a.out_dir) if a.out_dir else ROOT / "reports" / "gates" / "P3"
    if not a.out_dir:
        require("P2")
    out_dir.mkdir(parents=True, exist_ok=True)
    g = yaml.safe_load((ROOT / "configs/gates.yaml").read_text())["P3"]
    n = a.n or g["n_samples"]
    refs = json.loads((ROOT / "reports/references/REFERENCES.json").read_text())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train = MarioWindows("train").grids.numpy()

    per_sched, all_rows, warnings, artifacts = {}, [], [], []
    for run in map(Path, a.runs):
        # milestones (5k/10k/15k) if present, otherwise every checkpoint that exists (e.g. a short test run)
        ckpts = sorted(run.glob(a.ckpt_glob.replace(".pt", ".milestone")))
        ckpts = [c.with_suffix(".pt") for c in ckpts] or sorted(run.glob(a.ckpt_glob))
        if not ckpts:
            sys.exit(f"no checkpoints in {run}")
        best = None
        for c in ckpts:
            m, grids, nn, state = evaluate_checkpoint(c, train, refs, g, a.steps, n, device)
            m["checkpoint"], m["checkpoint_sha256"] = c.name, (c.with_suffix(".pt.sha256").read_text().strip() if c.with_suffix(".pt.sha256").exists() else None)
            all_rows.append(m)
            print(f"{run.name} {c.name}: " + "  ".join(f"{k}={m[k]:.3f}" for k in ("floor_valid", "broken_rate", "tile_js", "copy_rate", "novelty_med", "S_noplay")))
            if best is None or m["S_noplay"] > best[0]["S_noplay"]:
                best = (m, grids, nn)
        m, grids, nn = best
        kind = m["schedule"]
        m["checks"] = checks_for(m, refs, g)
        m["passes"] = all(c["pass"] for c in m["checks"])
        per_sched[kind] = m
        sheet(list(grids[:64]), cols=8, tile_px=8).save(out_dir / f"sample_sheet_{kind}.png")
        artifacts.append(f"sample_sheet_{kind}.png")
        # the 8 lowest-nn samples beside their nearest train windows (memorisation check)
        order = np.argsort(nn)[:8]
        _, arg = nn_distance(grids[order], train)
        pairs = [im for i, j in zip(order, arg) for im in (grid_to_image(grids[i], 8, False), grid_to_image(train[j], 8, False))]
        w, h = pairs[0].size
        fig = Image.new("RGB", (4 * (w + 4) + 4, 4 * (h + 4) + 4), "white")
        for k, im in enumerate(pairs):
            fig.paste(im, (4 + (k % 4) * (w + 4), 4 + (k // 4) * (h + 4)))
        fig.save(out_dir / f"nearest_train_{kind}.png")
        artifacts.append(f"nearest_train_{kind}.png")
        if m["tile_js"] > g["tile_js_warn"]:
            warnings.append(f"{kind}: tile JS {m['tile_js']:.3f} > drift warning {g['tile_js_warn']}")

    passing = {k: v for k, v in per_sched.items() if v["passes"]}
    choice = None
    if passing:
        if len(passing) == 2 and abs(per_sched["linear"]["S_noplay"] - per_sched["cosine"]["S_noplay"]) < g["tie_delta"]:
            choice = "cosine"                                              # tie -> cosine (M2)
        else:
            choice = max(passing, key=lambda k: passing[k]["S_noplay"])
    passed = choice is not None
    write_result("P3", passed, {"copy_rate_ref": refs["copy_rate_ref"], "novelty_ref": refs["novelty_ref_median_stride8"]},
                 checks=[{"schedule": k, **c} for k, v in per_sched.items() for c in v["checks"]], out_dir=out_dir,
                 diagnostics={"S_noplay": {k: round(v["S_noplay"], 4) for k, v in per_sched.items()}, "schedule_choice": choice,
                              "composite": "S_noplay (playable term dropped: checker not validated yet, ledger E4)",
                              "all_checkpoint_rows": all_rows},
                 warnings=warnings, artifacts=artifacts, device=device, ddim_steps=a.steps, n_samples=n)
    print("schedule choice:", choice, "| warnings:", warnings or "none")
    print("G-P3", "PASS" if passed else "FAIL")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
