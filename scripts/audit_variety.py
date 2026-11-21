"""Contrarian check: does the adherence fix make levels degenerate?  python scripts/audit_variety.py --before baseline_full --after round5

Explain-me:
  What: for every prompt, compares the 10 seeds' RAW levels in two audit runs: how many distinct grids, the mean pairwise normalised Hamming distance
        between the grids (share of cells that differ), and the share of cells that are plain floor/sky (a degenerate level is nearly empty).
        A fix that "passes" by always producing the same easy level would show distinct = 1 and distance ~ 0.
  In/Out: reports/audit/<run>/rows.jsonl (grids) -> reports/audit/variety.json and variety.md.
  Previous: scripts/audit_adherence.py.  Next: reports/audit/AUDIT_REPORT.md.
  Try: python scripts/audit_variety.py --before baseline_full --after round6_fresh_seeds
"""
import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CHARS = "-<>?BEQSX[]bo"                       # tile ids 0..12 in order (data/tiles.py CHARS) only used to decode the stored grids
SKY, GROUND = 0, 8


def load(run):
    out = defaultdict(list)
    for line in (ROOT / "reports/audit" / run / "rows.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r.get("grid") and r["cat"] != "adversarial":
            out[r["id"]].append(np.array([[CHARS.index(c) for c in row] for row in r["grid"]], np.uint8))
    return out


def stats(grids):
    pairs = list(itertools.combinations(range(len(grids)), 2))
    dist = float(np.mean([np.mean(grids[i] != grids[j]) for i, j in pairs])) if pairs else float("nan")
    distinct = len({g.tobytes() for g in grids})
    plain = float(np.mean([np.mean(np.isin(g, [SKY, GROUND])) for g in grids]))
    return {"n": len(grids), "distinct": distinct, "mean_pairwise_hamming": round(dist, 4), "plain_floor_or_sky_share": round(plain, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", required=True)
    ap.add_argument("--after", required=True)
    a = ap.parse_args()
    b, f = load(a.before), load(a.after)
    rows, agg = [], {"before": [], "after": []}
    for pid in f:
        if pid not in b:
            continue
        sb, sf = stats(b[pid]), stats(f[pid])
        rows.append({"id": pid, "before": sb, "after": sf})
        agg["before"].append(sb)
        agg["after"].append(sf)
    summ = {k: {m: round(float(np.mean([s[m] for s in v])), 4) for m in ("distinct", "mean_pairwise_hamming", "plain_floor_or_sky_share")} for k, v in agg.items()}
    summ["min_distinct_after"] = min(r["after"]["distinct"] for r in rows)
    (ROOT / "reports/audit/variety.json").write_text(json.dumps({"before_run": a.before, "after_run": a.after, "summary": summ, "per_prompt": rows}, indent=1), encoding="utf-8")
    md = [f"# Variety check: `{a.before}` (before) vs `{a.after}` (after)", "",
          (f"Mean over prompts: distinct grids of 10 seeds {summ['before']['distinct']} -> {summ['after']['distinct']}; mean pairwise Hamming "
           f"{summ['before']['mean_pairwise_hamming']} -> {summ['after']['mean_pairwise_hamming']}; share of plain floor/sky cells "
           f"{summ['before']['plain_floor_or_sky_share']} -> {summ['after']['plain_floor_or_sky_share']}; "
           f"fewest distinct grids for any prompt after the fix: {summ['min_distinct_after']} of 10."), "",
          "| prompt id | distinct (before / after) | Hamming (before / after) | plain share (before / after) |", "|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['id']} | {r['before']['distinct']} / {r['after']['distinct']} | {r['before']['mean_pairwise_hamming']} / {r['after']['mean_pairwise_hamming']} | "
                  f"{r['before']['plain_floor_or_sky_share']} / {r['after']['plain_floor_or_sky_share']} |")
    (ROOT / "reports/audit/variety.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(summ))


if __name__ == "__main__":
    main()
