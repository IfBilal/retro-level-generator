"""Tower definition check (ledger AD9): python scripts/audit_towers.py --run round5 [--run round4 ...]

Explain-me:
  What: for the "N towers" prompts of an audit run, compares the caption detector's tower count (the training-caption definition) with the visible-tower
        count (`eval/count_audit.visible_towers`) on the stored raw grids, and how often each satisfies the prompt.  The detector counts the narrow top of
        a staircase as a tower and merges neighbouring pillars, so the two disagree on many levels; the audit counts what a person sees.
  In/Out: reports/audit/<run>/rows.jsonl -> reports/audit/NOTES_towers.md (+ printed).
  Previous: scripts/audit_adherence.py.  Next: the audit report.
  Try: python scripts/audit_towers.py --run round5 --run baseline_full
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from text2mario.data import tiles as T
from text2mario.eval.count_audit import detector_objects, visible_towers

WANT = {"cnt_tower_1": 1, "cnt_tower_2": 2, "cnt_tower_3": 3}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True)
    runs = ap.parse_args().run
    md = ["# Tower definition check (ledger AD9)", "",
          "| run | levels | detector says it matches | a person would say it matches (visible rule) | the two disagree on the count |", "|---|---|---|---|---|"]
    for run in runs:
        n = det_ok = vis_ok = dis = 0
        for line in (ROOT / "reports/audit" / run / "rows.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r["id"] not in WANT or not r.get("grid"):
                continue
            g = np.array([[T.CHARS.index(c) for c in row] for row in r["grid"]], np.uint8)
            det = len(detector_objects(g, visible_tower_rule=False).get("tower", []))
            vis = len(visible_towers(g))
            n += 1
            det_ok += det == WANT[r["id"]]
            vis_ok += vis == WANT[r["id"]]
            dis += det != vis
        md.append(f"| {run} | {n} | {det_ok} | {vis_ok} | {dis} |")
    out = ROOT / "reports/audit/NOTES_towers.md"
    out.write_text("\n".join(md) + "\n\nRuns generated BEFORE the visible-tower rule was enforced show the gap (the generator was satisfying the detector's definition); "
                   "runs after it show both agree on the matching share.\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
