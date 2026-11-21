"""Contrarian check: does the counter agree with what a person SEES?  python scripts/audit_visual_check.py --run round5 --out evidence/audit/<ts>

Explain-me:
  What: picks levels from an audit run (default: the first seed of 12 chosen prompts, passing and failing ones), renders their RAW grid with the real
        game sprites and writes one contact sheet; every panel is titled with the prompt and what the counter found, so a person can check by eye that
        "one pipe" really shows one pipe.  It also writes the list of what was drawn (levels.json) so the check can be repeated.
  In/Out: reports/audit/<run>/rows.jsonl -> <out>/visual_check_<run>.png + visual_check_<run>.json.
  Previous: scripts/audit_adherence.py.  Next: a human looks at the PNG (the audit report says what was seen).
  Try: python scripts/audit_visual_check.py --run round5 --ids cnt_pipe_2,ex_button_2 --seeds 1001,1002
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from text2mario.data import tiles as T
from text2mario.render.renderer import grid_to_sprite_image

DEFAULT_IDS = ["cnt_pipe_1", "cnt_pipe_2", "cnt_enemy_3", "cnt_platform_3", "cnt_tower_2", "cnt_cannon_2", "ex_button_2", "ex_button_3",
               "ord_three", "neg_two_pipes_no_enemies", "syn_chimney", "two_scene_same"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ids", default=",".join(DEFAULT_IDS))
    ap.add_argument("--seeds", default="1000,5000")
    a = ap.parse_args()
    seeds = {int(s) for s in a.seeds.split(",")}
    want = a.ids.split(",")
    rows = [json.loads(line) for line in (ROOT / "reports/audit" / a.run / "rows.jsonl").read_text(encoding="utf-8").splitlines()]
    pick = []
    for pid in want:
        for r in rows:
            if r["id"] == pid and r.get("grid") and r["seed"] in seeds:
                pick.append(r)
                break
    cols = 2
    n = len(pick)
    fig, axes = plt.subplots((n + cols - 1) // cols, cols, figsize=(15, 2.6 * ((n + cols - 1) // cols)))
    axes = np.atleast_1d(axes).ravel()
    out_items = []
    for ax, r in zip(axes, pick):
        grid = np.array([[T.CHARS.index(c) for c in row] for row in r["grid"]], np.uint8)
        ax.imshow(grid_to_sprite_image(grid))
        counts = {k: v for k, v in r["counts"].items() if v}
        checks = ", ".join(f"{c['topic']} {c['got']} (asked {c['want'][0] if c['want'][0] == c['want'][1] else c['want']})" for c in r["details"]["checks"])
        ax.set_title(f"{r['text'][:70]}  [seed {r['seed']}]  {'PASS' if r['ok'] else 'FAIL'}\nchecked: {checks} | all objects found: {counts}", fontsize=7, loc="left")
        ax.axis("off")
        out_items.append({"id": r["id"], "seed": r["seed"], "text": r["text"], "ok": r["ok"], "checks": r["details"]["checks"], "all_counts": counts})
    for ax in axes[len(pick):]:
        ax.axis("off")
    fig.tight_layout()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"visual_check_{a.run}.png", dpi=110)
    (out / f"visual_check_{a.run}.json").write_text(json.dumps(out_items, indent=1), encoding="utf-8")
    print("wrote", out / f"visual_check_{a.run}.png", n, "levels")


if __name__ == "__main__":
    main()
