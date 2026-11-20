"""Check our caption detector against the 7,687 stored captions (doc 09 §6.4; ledger O3).

Explain-me:
  What: runs detector.caption on every corpus window; compares phrase SETS (and exact strings) with the stored caption.
  In/Out: data/source corpus -> console summary + reports/eda/detector_mismatches.jsonl.
  Exit criterion (timeline Day 1): 100 % phrase-set agreement, or >= 99.5 % with every exception documented.
"""
import collections
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from text2mario.captions.detector import caption
from text2mario.captions.grammar import caption_set
from text2mario.data import provenance


def main():
    provenance.verify()
    recs = json.loads((ROOT / "data/source/Mar1and2_LevelsAndCaptions-regular.json").read_text())
    n_set = n_str = 0
    miss = []
    diff_phrases = collections.Counter()
    for i, r in enumerate(recs):
        ours = caption(np.array(r["scene"], dtype=np.uint8))
        a, b = caption_set(ours), caption_set(r["caption"])
        n_set += a == b
        n_str += ours == r["caption"]
        if a != b:
            miss.append({"idx": i, "stored": r["caption"], "ours": ours})
            for p in a ^ b:
                diff_phrases[("ours-only " if p in a else "stored-only ") + p] += 1
    n = len(recs)
    print(f"phrase-set agreement {n_set}/{n} = {n_set / n:.4%};  exact string {n_str}/{n} = {n_str / n:.4%}")
    out = ROOT / "reports/eda/detector_mismatches.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(m) + "\n" for m in miss))
    for k, v in diff_phrases.most_common(15):
        print(f"  {v:5d}  {k}")
    sys.exit(0 if n_set / n >= 0.995 else 1)


if __name__ == "__main__":
    main()
