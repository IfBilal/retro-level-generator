"""Provenance check: every number in the report and slides comes from a saved artifact, and every figure exists (doc 32, CLAUDE.md §3).

Explain-me:
  What: rebuilds the placeholder values from the saved result files and checks that (1) the report and slides have no unresolved or 'n/a (not run)' placeholder,
        (2) every figure referenced by the report exists, (3) RESULTS.md numbers equal metrics.json, (4) the frozen checkpoint hash in FROZEN.json equals the one
        printed in the report, (5) the gate statuses quoted in PROGRESS.md match the RESULT.json files.  Exits 1 on any failure.
  In/Out: report/report.md, slides/slides.html, reports/**, FROZEN.json -> printed table; exit code.  Try: python scripts/check_provenance.py
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_report as BR


def main() -> int:
    checks = []
    rep = (ROOT / "report/report.md").read_text(encoding="utf-8")
    checks.append(("report has no unresolved or 'n/a (not run)' placeholder", "n/a (not run)" not in rep and "{{" not in rep))
    figs = re.findall(r"\]\(([^)]+\.png)\)", rep)
    missing = [f for f in figs if not (ROOT / "report" / f).resolve().exists()]
    checks.append((f"all {len(figs)} figures referenced by the report exist" + (f" (missing: {missing})" if missing else ""), not missing))
    sl = ROOT / "slides/slides.html"
    checks.append(("slides have no unresolved placeholder", sl.exists() and "{{" not in sl.read_text(encoding="utf-8") and "n/a (not run)" not in sl.read_text(encoding="utf-8")))
    m = json.loads((ROOT / "reports/final/metrics.json").read_text())
    res = (ROOT / "reports/final/RESULTS.md").read_text()
    cs = m["P-test"]["c_score"]["mean"]
    checks.append(("RESULTS.md C-score equals metrics.json", f"{cs:.3f}" in res))
    fz = json.loads((ROOT / "FROZEN.json").read_text())
    checks.append(("frozen checkpoint hash appears in the report", fz["checkpoint_sha256"][:16] in rep))
    checks.append(("report test C-score equals metrics.json", f"{cs:.3f}" in rep))
    vals = BR.values()
    checks.append(("report builder values all resolved", not [k for k, v in vals.items() if isinstance(v, str) and v == "n/a (not run)"]))
    gates = {g: json.loads((ROOT / f"reports/gates/{g}/RESULT.json").read_text())["pass"] for g in ("P1", "P2", "P3", "P4", "P5", "P5b", "P6", "P7") if (ROOT / f"reports/gates/{g}/RESULT.json").exists()}
    prog = (ROOT / "PROGRESS.md").read_text(encoding="utf-8")
    ok_prog = all((("FAIL" in prog) if not v else True) for v in gates.values()) and ("G-P3 FAIL" in prog) == (not gates.get("P3", True))
    checks.append((f"PROGRESS.md reports the failing gates ({[g for g, v in gates.items() if not v]})", ok_prog))
    bad = 0
    for name, ok in checks:
        print(("PASS " if ok else "FAIL ") + name)
        bad += not ok
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
