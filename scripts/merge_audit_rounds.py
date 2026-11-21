"""Merge audit rows from several runs into one results set: python scripts/merge_audit_rounds.py --out final --parts round5:non-adversarial round4_adv:adversarial

Explain-me:
  What: the baseline was run before the injection metric was corrected and before the extra payloads existed, so its 18 adversarial rows are not
        comparable with later runs.  This script takes the NON-adversarial rows of one run and the ADVERSARIAL rows of another run (same harness,
        same code under test) and writes a merged rows.jsonl / results.json / RESULTS.md so before and after cover the same 630 levels.
        The original runs are never modified.
  In/Out: reports/audit/<run>/rows.jsonl -> reports/audit/<out>/{rows.jsonl, results.json, RESULTS.md}.
  Previous: scripts/audit_adherence.py.  Next: scripts/build_report.py (reads baseline_full and final).
  Try: python scripts/merge_audit_rounds.py --out baseline_full --parts baseline:non-adversarial baseline_adv:adversarial
"""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import audit_adherence as AA


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--parts", nargs="+", required=True, help="run:non-adversarial | run:adversarial | run:all")
    a = ap.parse_args()
    rows, meta = [], None
    for part in a.parts:
        run, which = part.split(":")
        d = ROOT / "reports/audit" / run
        for line in (d / "rows.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if which == "all" or (which == "adversarial") == (r["cat"] == "adversarial"):
                rows.append(r)
        m = json.loads((d / "results.json").read_text(encoding="utf-8"))
        meta = meta or dict(m["meta"])
        meta.setdefault("merged_from", []).append(f"{run} ({which}) git {m['meta']['git']}")
    out = ROOT / "reports/audit" / a.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "rows.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    per, cat, tot = AA.summarise(rows)
    mode = SimpleNamespace(mode="fast")
    (out / "results.json").write_text(json.dumps({"meta": meta, "mode": "fast", "overall": tot, "by_category": cat, "per_prompt": per}, indent=1), encoding="utf-8")
    AA.write_md(out / "RESULTS.md", a.out, mode, per, cat, tot, rows, meta)
    print(f"{a.out}: {tot[0]}/{tot[1]} levels pass; merged from {meta['merged_from']}")


if __name__ == "__main__":
    main()
