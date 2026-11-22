"""Seam-ratio comparison of the generation path before and after the adherence work (G-P5.01, same jobs): python scripts/audit_seam_compare.py

Explain-me:
  What: G-P5 generates 112-column levels for the dev prompts.  The official result (reports/gates/P5) was produced by the unmodified pipeline; a 20-level re-run with the
        final pipeline went to reports/gates/_p5_adherence_check (never overwriting the official result).  This script recomputes the seam ratio of the FIRST 20 levels of both
        sets with the same column-bigram model, so the comparison uses the same prompts and seeds, and writes the numbers to reports/audit/gate_p5_compare.json.
  In/Out: reports/gates/P5/levels, reports/gates/_p5_adherence_check/levels -> reports/audit/gate_p5_compare.json.
  Previous: scripts/gate_p5.py.  Next: reports/audit/AUDIT_REPORT.md.
  Try: python scripts/audit_seam_compare.py
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from calibrate_checker import real_levels

from text2mario.eval.seams import ColumnBigram


def main():
    lm = ColumnBigram().fit([g for _, g, _ in real_levels(("train",)).values()])
    out = {"limit": 1.3, "n_levels": 20}
    for tag, d in (("before_unmodified_pipeline", "reports/gates/P5"), ("after_final_pipeline", "reports/gates/_p5_adherence_check")):
        rs, ver = [], []
        for f in sorted(glob.glob(str(ROOT / d / "levels/*.json")))[:20]:
            j = json.loads(Path(f).read_text(encoding="utf-8"))
            rs.append(lm.seam_ratio(np.array(j["raw_grid"], np.uint8)[:, 3:3 + 112])[0])
            ver.append(j["verification"]["status"] == "verified")
        out[tag] = {"mean": round(float(np.nanmean(rs)), 3), "median": round(float(np.nanmedian(rs)), 3), "max": round(float(np.nanmax(rs)), 2), "verified": int(sum(ver)), "n": len(rs)}
    (ROOT / "reports/audit/gate_p5_compare.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out))


if __name__ == "__main__":
    main()
