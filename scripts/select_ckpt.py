"""Pick the checkpoint by the dev composite S_noplay (doc 22 §3.13; playable term excluded while G-P5b has not passed).

Explain-me:
  What: reads reports/dev_eval/<run>_<step>_p_dev_w5.json for every evaluated milestone, scores
        S = 0.5 C + 0.25 (1 - broken) + 0.25 min(1, novelty / reference novelty), and picks the best; |dS| < 0.01 ties go to the EARLIER step.
        Dev numbers only.  Writes reports/dev_eval/selection_<run>.json.
  In/Out: reports/dev_eval/*.json -> selection json + a table on stdout.  Next: gate_p4.py --ckpt <chosen>.
  Try: python scripts/select_ckpt.py C-main
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REF = json.loads((ROOT / "reports/references/REFERENCES.json").read_text())["novelty_ref_median_stride8"]


def s_noplay(r):
    return 0.5 * r["c_score"] + 0.25 * (1 - r["broken_rate"]) + 0.25 * min(1.0, r["novelty_med"] / REF)


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else "C-main"
    rows = []
    for f in sorted((ROOT / "reports/dev_eval").glob(f"{run}_*_p_dev_w5.json")):
        r = json.loads(f.read_text())
        rows.append({"step": r["step"], "c_score": r["c_score"], "broken_rate": r["broken_rate"], "novelty_med": r["novelty_med"],
                     "copy_rate": r["copy_rate"], "S_noplay": s_noplay(r)})
    rows.sort(key=lambda x: x["step"])
    best = rows[0]
    for r in rows[1:]:
        if r["S_noplay"] > best["S_noplay"] + 0.01:                  # a later step must win by more than the tie margin
            best = r
    for r in rows:
        print(f"{run} step {r['step']:6d}  C={r['c_score']:.3f} broken={r['broken_rate']:.3f} nov={r['novelty_med']:.1f} copy={r['copy_rate']:.3f}  S={r['S_noplay']:.3f}" + ("  <== selected" if r is best else ""))
    (ROOT / "reports/dev_eval" / f"selection_{run}.json").write_text(json.dumps({"run": run, "selected_step": best["step"], "rule": "max S_noplay; ties (|dS|<0.01) -> earlier step",
                                                                                "table": rows}, indent=1))
    print("selected step:", best["step"])


if __name__ == "__main__":
    main()
