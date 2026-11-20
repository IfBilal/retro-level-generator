"""Re-measure every ledger data fact and fail loudly if one differs (doc 09 §1; gate G-P1 part 1).

Explain-me:
  What: ~35 assertions (F1-F17, A1-A13) on the pinned corpus. Each writes {id, expected, observed, pass, severity}
        to reports/eda/AUDIT.json; any hard (H) failure makes the script exit non-zero. Nothing is "fixed" here:
        a mismatch means the data or our understanding changed, so we stop and investigate.
  In/Out: data/source/*, data/processed/* -> reports/eda/AUDIT.json, data/MANIFEST.json (audit.passed).
  Try: change an expected value below and watch that row fail.
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from text2mario.captions.detector import detect  # noqa: E402
from text2mario.captions.grammar import caption_set  # noqa: E402
from text2mario.data import provenance  # noqa: E402
from text2mario.data import tiles as T  # noqa: E402
from text2mario.data.index import N_SMB1_WINDOWS, level_grid, level_ids  # noqa: E402

SRC = ROOT / "data" / "source"
RESULTS: list[dict] = []


def check(cid: str, desc: str, expected, observed, severity: str = "H", tol: float | None = None) -> None:
    """Record one assertion. `tol` makes numbers compare within +-tol."""
    if tol is not None:
        ok = abs(float(observed) - float(expected)) <= tol
    else:
        ok = expected == observed
    RESULTS.append({"id": cid, "description": desc, "expected": expected, "observed": observed,
                    "pass": bool(ok), "severity": severity})
    print(f"[{'PASS' if ok else 'FAIL'}] {cid} ({severity}) {desc}: expected {expected}, observed {observed}")


def onehot_flat(scenes: np.ndarray) -> torch.Tensor:
    """(N,16,16) uint8 -> (N, 13*256) float32 one-hot (Hamming = 256 - dot product, doc 10 §3.2)."""
    t = torch.from_numpy(scenes.astype(np.int64))
    return torch.nn.functional.one_hot(t, 13).reshape(len(scenes), -1).float()


def nn_distance(a: np.ndarray, b: np.ndarray, mask_same=None, chunk: int = 1024) -> np.ndarray:
    """Nearest-neighbour Hamming distance from each window of `a` to the windows of `b`.
    `mask_same[i, j]` (optional callable on index arrays) excludes pairs, e.g. same level."""
    A, B = onehot_flat(a), onehot_flat(b)
    out = np.zeros(len(a), np.int32)
    for s in range(0, len(a), chunk):
        d = 256 - (A[s:s + chunk] @ B.T)
        if mask_same is not None:
            d[mask_same(np.arange(s, min(s + chunk, len(a))))] = 10 ** 6
        out[s:s + chunk] = d.min(dim=1).values.round().int().numpy()
    return out


def main() -> None:
    provenance.verify()
    manifest = json.loads((ROOT / "data/MANIFEST.json").read_text())
    exp_hash = "6a15264a4beb67247a0488dba0439f257f7a94955c0df60b709b7e85024ade2c"
    recs = json.loads((SRC / "Mar1and2_LevelsAndCaptions-regular.json").read_text())
    check("F1", "corpus content hash", exp_hash, provenance.corpus_content_hash())

    ok_schema = isinstance(recs, list) and all(
        isinstance(r, dict) and set(r) == {"prompt", "scene", "caption"} and r["prompt"] is None
        and len(r["scene"]) == 16 and all(len(row) == 16 for row in r["scene"])
        and isinstance(r["caption"], str) and r["caption"] for r in recs)
    check("F2", "record schema", True, ok_schema)
    check("F3", "number of records", 7687, len(recs))
    scenes = np.array([r["scene"] for r in recs], dtype=np.int64)
    check("F5", "tile ids in 0..12", (0, 12), (int(scenes.min()), int(scenes.max())))
    scenes = scenes.astype(np.uint8)
    n = len(scenes)

    hashes = [hashlib.sha1(s.tobytes()).hexdigest() for s in scenes]
    cnt = Counter(hashes)
    check("F4", "unique scenes", 7620, len(cnt))
    dups = {h: c for h, c in cnt.items() if c > 1}
    check("F4b", "duplicate groups / extra copies", (41, 67), (len(dups), sum(c - 1 for c in dups.values())), "W")

    smb = json.loads((SRC / "smb.json").read_text())
    check("F6", "tile chars match smb.json (sorted) and tiles.CHARS", list(T.CHARS), sorted(smb["tiles"]))
    desc = smb["tiles"]
    check("F6b", "solid set matches descriptors",
          {T.CHAR_TO_ID[c] for c, d in desc.items() if "solid" in d}, set(T.SOLID))
    check("F6c", "pipe / cannon / enemy sets match descriptors", [sorted(T.PIPE), sorted(T.CANNON), sorted(T.ENEMY)],
          [sorted(T.CHAR_TO_ID[c] for c, d in desc.items() if tag in d) for tag in ("pipe", "cannon", "enemy")])

    lid, off, chains = level_ids(scenes)
    check("F7", "chain count", 42, len(chains))
    check("F7b", "SMB1 / SMB2 boundary", N_SMB1_WINDOWS, chains[20][0])
    within = int((scenes[:-1, :, 1:] == scenes[1:, :, :-1]).all(axis=(1, 2)).sum())
    check("F8", "within-chain stride-1 pairs", 7645, within)
    s1 = json.loads((SRC / "SMB1_Levels.json").read_text())
    s2 = json.loads((SRC / "SMB2_Levels.json").read_text())
    check("F9", "records[:3606]==SMB1_Levels, rest==SMB2_Levels", True,
          np.array_equal(scenes[:N_SMB1_WINDOWS], np.array(s1, np.uint8))
          and np.array_equal(scenes[N_SMB1_WINDOWS:], np.array(s2, np.uint8)))
    check("F10", "no level-id field", True, all(set(r) == {"prompt", "scene", "caption"} for r in recs))

    lens = [e - b for b, e in chains]
    widths = [e - b + 15 for b, e in chains]
    grids = [level_grid(scenes, b, e) for b, e in chains]
    check("F11a", "chain length min/max", (130, 358), (min(lens), max(lens)))
    check("F11b", "level width min/median/max", (145, 193, 373), (min(widths), int(np.median(widths)), max(widths)))
    all_cols = np.concatenate(grids, axis=1)
    distinct = {c.tobytes() for c in all_cols.T}
    check("F11c", "sum of widths / distinct column patterns", (8317, 653), (all_cols.shape[1], len(distinct)))
    ok_resl = all(np.array_equal(np.stack([g[:, i:i + 16] for i in range(g.shape[1] - 15)]), scenes[b:e])
                  for g, (b, e) in zip(grids, chains))
    check("F11d", "full levels re-slice to the stored windows", True, ok_resl)

    top = scenes[:, :2, :]
    nz = top[top != 0]
    check("F12", "rows 0-1 non-sky cells / tile ids / windows",
          (64, [9, 10], 17), (int((top != 0).sum()), sorted(set(nz.tolist())), int((top != 0).any(axis=(1, 2)).sum())))
    check("F12b", "row 2 non-sky cells", 32, int((scenes[:, 2, :] != 0).sum()))

    freq = np.bincount(scenes.ravel(), minlength=13) / scenes.size * 100
    exp_freq = {0: 87.13, 1: 0.18, 2: 0.18, 3: 0.05, 4: 0.04, 5: 0.62, 6: 0.19, 7: 1.70, 8: 8.66, 9: 0.43,
                10: 0.43, 11: 0.03, 12: 0.34}
    bad = {k: round(float(freq[k]), 2) for k, v in exp_freq.items() if abs(freq[k] - v) > 0.01}
    check("F13", "tile frequencies (%) within 0.01", {}, bad)

    caps = [r["caption"] for r in recs]
    words = [len(c.replace(".", " ").split()) for c in caps]
    phr = [len(caption_set(c)) for c in caps]
    check("F14a", "caption words min/max/mean", (2, 28, 11.39), (min(words), max(words), round(float(np.mean(words)), 2)))
    check("F14b", "phrases min/max/mean", (1, 10, 4.03), (min(phr), max(phr), round(float(np.mean(phr)), 2)))
    all_phr = Counter(p for c in caps for p in caption_set(c))
    vocab = {w for c in caps for w in c.replace(".", " ").split()}
    check("F14c", "distinct phrases / vocabulary", (62, 44), (len(all_phr), len(vocab)))
    qw = Counter(p.split()[0] if not p.startswith("a few") else "a few" for p in all_phr)
    many_topics = sorted({p for p in all_phr if p.startswith("many ")})
    check("F15", "'many' only for coins and question blocks", ["many coins", "many question blocks"], many_topics)
    check("A10", "every caption with 'coin line' has a coin phrase", True,
          all(any(" coin" in p and "coin line" not in p for p in caption_set(c)) for c in caps if "coin line" in c))
    check("A11", "no 'broken' in captions", 0, sum("broken" in c for c in caps))
    check("A9", "unique captions / unique phrase sets", (2668, 2665),
          (len(set(caps)), len({caption_set(c) for c in caps})), "W")
    check("F19", "detector returns contributing cells", True, all(len(d.cells) > 0 for d in detect(scenes[0])))

    # upstream split sanity (never used for training) F16/F17
    up = {k: len(json.loads(Path(ROOT / f"data/raw/MarioDiffusion/datasets/Mar1and2_LevelsAndCaptions-regular-{k}.json").read_text()))
          for k in ("train", "validate", "test")}
    check("F16", "upstream split sizes", (6918, 384, 385), (up["train"], up["validate"], up["test"]), "W")
    ut = json.loads((ROOT / "data/raw/MarioDiffusion/datasets/Mar1and2_LevelsAndCaptions-regular-test.json").read_text())
    utr = json.loads((ROOT / "data/raw/MarioDiffusion/datasets/Mar1and2_LevelsAndCaptions-regular-train.json").read_text())
    d_up = nn_distance(np.array([r["scene"] for r in ut], np.uint8), np.array([r["scene"] for r in utr], np.uint8))
    check("F17a", "upstream test->train NN Hamming median", 8, int(np.median(d_up)), "W", tol=1)
    check("F17b", "upstream test within 16 tiles of train", 83.9, round(float((d_up <= 16).mean() * 100), 1), "W", tol=1.0)
    # which levels does the upstream test set cover?
    lookup = {h: int(lid[i]) for i, h in reversed(list(enumerate(hashes)))}
    covered = {lookup[hashlib.sha1(np.array(r["scene"], np.uint8).tobytes()).hexdigest()] for r in ut}
    check("F17c", "upstream test spans all 42 levels", 42, len(covered), "W")

    # A3 bottom row, A6 pipe tops
    bottom = scenes[:, 15, :]
    solid_frac = np.isin(bottom, list(T.SOLID)).mean(axis=1)
    check("A3", "bottom row fully solid / >=50% / none (%)", (35.01, 71.00, 16.64),
          (round(float((solid_frac == 1).mean() * 100), 2), round(float((solid_frac >= .5).mean() * 100), 2),
           round(float((solid_frac == 0).mean() * 100), 2)), tol=None)
    bad_pipe = 0
    for s in scenes:
        for r in range(16):
            for c in range(16):
                v = s[r, c]
                if v == T.PIPE_TL and c < 15 and s[r, c + 1] != T.PIPE_TR:
                    bad_pipe += 1
                    break
                if v == T.PIPE_TR and c > 0 and s[r, c - 1] != T.PIPE_TL:
                    bad_pipe += 1
                    break
            else:
                continue
            break
    check("A6", "windows with a broken pipe lip in the raw data", 0, bad_pipe)

    # A12 / A13 cross-level near-duplicates and adjacent-window distance
    A = onehot_flat(scenes)
    cross = np.zeros(n, np.int32)
    for s_ in range(0, n, 1024):
        d = 256 - (A[s_:s_ + 1024] @ A.T)
        same = torch.from_numpy(lid[s_:s_ + 1024, None] == lid[None, :])
        d[same] = 10 ** 6
        cross[s_:s_ + 1024] = d.min(dim=1).values.round().int().numpy()
    got = [round(float((cross <= k).mean() * 100), 2) for k in (0, 2, 4, 8, 16)]
    check("A12", "share within Hamming 0/2/4/8/16 of another level (%)", [0.99, 7.49, 16.09, 40.17, 75.40], got, "W")
    adj = [(scenes[i] != scenes[i + 1]).sum() for i in range(n - 1) if lid[i] == lid[i + 1]]
    check("A13", "adjacent-window Hamming median/mean", (14, 16.19), (int(np.median(adj)), round(float(np.mean(adj)), 2)), "W")

    # VGLC (A1/A2) is verified byte-for-byte in build_index.py; read its result
    lv = pd.read_csv(ROOT / "data/processed/levels.csv")
    check("A1", "VGLC level names attached (42/42 chains reproduced)", 42, int(lv.vglc_height.notna().sum()))
    heights = {str(int(k)): int(v) for k, v in lv.vglc_height.value_counts().items()}
    check("A2", "VGLC heights 14/15/16", {"14": 20, "15": 21, "16": 1}, heights)

    hard_fail = [r for r in RESULTS if not r["pass"] and r["severity"] == "H"]
    passed = not hard_fail
    out = ROOT / "reports/eda/AUDIT.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"passed": passed, "python": sys.version.split()[0], "numpy": np.__version__,
                               "torch": torch.__version__, "checks": RESULTS}, indent=2, default=str) + "\n")
    manifest["audit"] = {"script": "scripts/audit_dataset.py", "passed": passed, "report": "reports/eda/AUDIT.json"}
    (ROOT / "data/MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\n{sum(r['pass'] for r in RESULTS)}/{len(RESULTS)} checks pass; hard failures: {len(hard_fail)}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
