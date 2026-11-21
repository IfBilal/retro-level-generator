"""Evaluate the prompt normalizer on the P-casual DEV prompts (doc 12 §9).  Test prompts are never read here.

Explain-me:
  What: parses the 30 development prompts and their expected RAW topics from doc 12 §8.1 into
        data/prompts/p_casual_dev.json (once), runs the normalizer and reports N1 scene count, N2 topic F1, N3 bucket
        accuracy and N4 exact match (scenes, raw topics, buckets and forbid sets all equal).
  In/Out: docs -> data/prompts/p_casual_dev.json, reports/normalizer/dev.json.  Previous: text/normalizer.py.
  Try: python scripts/eval_normalizer.py --show   (prints every miss with expected vs produced)
"""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from text2mario.text.normalizer import normalize  # noqa: E402

DOC = ROOT / "docs/docs/12_CAPTION_GRAMMAR_AND_PROMPT_NORMALIZER.md"
OUT = ROOT / "data/prompts/p_casual_dev.json"
ROW = re.compile(r"\|\s*(cas-(?:dev|test)-\d+)\s+([A-H])\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*$")


def parse_dev() -> list[dict]:
    rows, on = [], False
    for line in DOC.read_text(encoding="utf-8").splitlines():
        if line.startswith("### 8.1"):
            on = True
        elif line.startswith("### 8.2"):
            break
        m = ROW.match(line)
        if on and m:
            pid, cat, prompt, raw, caps = m.groups()
            rows.append({"pid": pid, "category": cat, "prompt": prompt, "expected_raw": raw, "expected_caption_fullcorpus": caps})
    return rows


def parse_test() -> list[dict]:
    """The 30 held-out casual prompts (doc 12 §8.2).  ONLY scripts/eval_test.py may call this, once, after FROZEN.json exists."""
    rows, on = [], False
    for line in DOC.read_text(encoding="utf-8").splitlines():
        if line.startswith("### 8.2"):
            on = True
        elif line.startswith("### 8.3"):
            break
        m = ROW.match(line)
        if on and m:
            pid, cat, prompt, raw, caps = m.groups()
            rows.append({"pid": pid, "category": cat, "prompt": prompt, "expected_raw": raw, "expected_caption_fullcorpus": caps})
    return rows


def parse_raw(raw: str):
    """'S1 floor full, pipe one · S2 enemy many (forbid cannon)' -> [({topic: bucket}, {forbid})], one entry per scene."""
    scenes = []
    glob = {}
    m = re.match(r"global (ceiling|floor) (\w+)\s*·\s*(.*)", raw)
    if m:
        glob, raw = {m.group(1): m.group(2)}, m.group(3)
    for part in re.split(r"\s*·\s*", raw):
        part = re.sub(r"^S\d\s*", "", part.strip())
        forbid = set(re.findall(r"forbid ([a-z ]+?)(?:[,)]|$)", part))
        part = re.sub(r"\([^)]*\)", "", part).split(";")[0]
        topics = dict(glob)
        for item in [i.strip() for i in part.split(",") if i.strip()]:
            m = re.match(r"(floor|ceiling)\s+(full|gaps:[\w ]+|chunks:[\w ]+)$", item)
            if m:
                topics[m.group(1)] = m.group(2).strip()
                continue
            m = re.match(r"(.+?)\s+(one|two|a few|several|many|default)$", item)
            if m:
                topics[m.group(1).strip()] = m.group(2)
        scenes.append((topics, forbid))
    return scenes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()
    rows = parse_dev()
    OUT.write_text(json.dumps(rows, indent=1))
    n_scene = em = tp = fp = fn = bucket_ok = bucket_n = 0
    miss = []
    for r in rows:
        out = normalize(r["prompt"])
        if "PASSTHROUGH" in r["expected_raw"]:        # canonical input must come back unchanged
            ok_pt = [s_.caption for s_ in out.scenes] == [r["prompt"]]
            n_scene += 1
            em += ok_pt
            if not ok_pt:
                miss.append((r["pid"], r["prompt"], r["expected_raw"], [s_.caption for s_ in out.scenes], out.warnings))
            continue
        exp = parse_raw(r["expected_raw"])
        got = [(dict(s.topics), set(s.forbid), s) for s in out.scenes]
        n_scene += len(got) == len(exp)
        ok = len(got) == len(exp)
        for (et, ef), (gt, gf, sc) in zip(exp, got):
            if "IMPLIED_COINS" in sc.warnings:
                gt.pop("coin", None)
                et.pop("coin", None)
            for k in set(et) | set(gt):
                if k in et and k in gt:
                    tp += 1
                    match = et[k] == gt[k] or (et[k] == "default" and "BARE_PLURAL_DEFAULT" in sc.warnings)
                    bucket_n += 1
                    bucket_ok += match
                    ok &= match
                elif k in et:
                    fn += 1
                    ok = False
                else:
                    fp += 1
                    ok = False
            ok &= ef == gf
        em += ok
        if not ok:
            miss.append((r["pid"], r["prompt"], r["expected_raw"], [(g[0], sorted(g[1])) for g in got], out.warnings))
    prec, rec = tp / max(1, tp + fp), tp / max(1, tp + fn)
    rep = {"n": len(rows), "N1_scene_count_acc": n_scene / len(rows), "N2_topic_f1": 2 * prec * rec / max(1e-9, prec + rec),
           "N3_bucket_acc": bucket_ok / max(1, bucket_n), "N4_exact_match": em / len(rows), "misses": [m[0] for m in miss],
           "target": {"N4": 0.90, "N2": 0.97}}
    (ROOT / "reports/normalizer").mkdir(parents=True, exist_ok=True)
    (ROOT / "reports/normalizer/dev.json").write_text(json.dumps(rep, indent=1))
    print(f"dev prompts {len(rows)}  N1 {rep['N1_scene_count_acc']:.3f}  N2 F1 {rep['N2_topic_f1']:.3f}  "
          f"N3 {rep['N3_bucket_acc']:.3f}  N4 exact {em}/{len(rows)}")
    if a.show:
        for pid, p, e, g, w in miss:
            print(f"\n{pid}: {p}\n  expected {e}\n  got      {g}  {w}")


if __name__ == "__main__":
    main()
