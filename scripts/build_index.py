"""Build data/processed/{scenes_u8.npy, index.parquet, levels.csv} from the pinned corpus (doc 09 §2, §5).

Explain-me:
  What: turns the JSON corpus into one uint8 array (7687,16,16) plus a table with one row per window:
        level_id, level name, game, column offset, caption concepts/buckets, duplicate group, ...
  In/Out: data/source/* -> data/processed/*.  Next: scripts/audit_dataset.py, scripts/make_splits.py.
  Try: print(pd.read_parquet('data/processed/index.parquet').head()) to see the columns.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from text2mario.captions.grammar import CONCEPTS, caption_set, parse_phrase
from text2mario.data import provenance
from text2mario.data import tiles as T
from text2mario.data.index import N_SMB1_WINDOWS, level_grid, level_ids

PROC = ROOT / "data" / "processed"
VGLC = ROOT / "data" / "source" / "vglc"


def col(concept: str) -> str:
    return concept.replace(" ", "_")


def load_vglc_level(path: Path) -> np.ndarray:
    """VGLC text -> grid (16, W): top-pad with sky to 16 rows and extend pipe bodies of the top row upward
    through the padding (that is how the corpus builder made hanging pipes, doc 08 §1.1)."""
    lines = [ln.rstrip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    # (SMB1 8-3 has a stray trailing space on its last two rows; rstrip() above removes it)
    g = np.array([[T.CHAR_TO_ID[c] for c in ln] for ln in lines], dtype=np.uint8)
    pad = 16 - g.shape[0]
    if pad:
        top = np.zeros((pad, g.shape[1]), np.uint8)
        for c in range(g.shape[1]):
            if g[0, c] in T.PIPE:  # extend the pipe tile upward
                top[:, c] = g[0, c]
        g = np.concatenate([top, g], axis=0)
    return g, 16 - pad


def level_name(tag: str, fname: str) -> str:
    if tag == "SMB1":
        m = re.match(r"mario-(\d+-\d+)\.txt", fname)
        return f"SMB1 {m.group(1)}"
    m = re.match(r"SuperMarioBros2\(J\)-World(\w+-\d+)\.txt", fname)
    return f"SMB2 {m.group(1)}"


def main() -> None:
    provenance.verify()
    recs = json.loads((ROOT / "data/source/Mar1and2_LevelsAndCaptions-regular.json").read_text())
    scenes = np.array([r["scene"] for r in recs], dtype=np.uint8)
    captions = [r["caption"] for r in recs]
    n = len(scenes)
    lid, off, chains = level_ids(scenes)
    grids = [level_grid(scenes, b, e) for b, e in chains]

    # level names + heights from VGLC when available; verify they reproduce the chains byte-for-byte (A1)
    names = [f"L{k:02d}" for k in range(len(chains))]
    heights = [None] * len(chains)
    vglc_ok = False
    files = [("SMB1", p) for p in sorted((VGLC / "SMB1").glob("*.txt"))] + \
            [("SMB2", p) for p in sorted((VGLC / "SMB2").glob("*.txt"))]
    if len(files) == len(chains):
        vglc_ok = True
        for k, (tag, p) in enumerate(files):
            g, h = load_vglc_level(p)
            if g.shape != grids[k].shape or not np.array_equal(g, grids[k]):
                vglc_ok = False
                print(f"VGLC mismatch for chain {k} ({p.name})")
                break
            names[k], heights[k] = level_name(tag, p.name), h
    print("VGLC names attached:", vglc_ok)

    widths = np.array([g.shape[1] for g in grids])
    df = pd.DataFrame({"idx": np.arange(n, dtype=np.int32), "level_id": lid.astype(np.int8)})
    df["level_name"] = [names[k] for k in lid]
    df["game"] = np.where(np.arange(n) < N_SMB1_WINDOWS, "SMB1", "SMB2")
    df["col_offset"] = off
    df["level_width"] = widths[lid].astype(np.int16)
    df["vglc_height"] = [heights[k] for k in lid]
    df["caption"] = captions
    df["n_phrases"] = [len(caption_set(c)) for c in captions]
    q = {c: np.zeros(n, np.int8) for c in CONCEPTS}
    for i, cap in enumerate(captions):
        for ph in caption_set(cap):
            concept, code = parse_phrase(ph)
            q[concept][i] = code
    for c in CONCEPTS:
        df[f"c_{col(c)}"] = q[c] > 0
    for c in CONCEPTS:
        df[f"q_{col(c)}"] = q[c]
    hashes = [hashlib.sha1(s.tobytes()).hexdigest() for s in scenes]
    df["scene_hash"] = hashes
    grp = df.groupby("scene_hash")["idx"].transform("size")
    codes = pd.factorize(df["scene_hash"])[0]
    df["dup_group"] = np.where(grp > 1, codes, -1).astype(np.int32)
    df["n_nonsky"] = (scenes != 0).sum(axis=(1, 2)).astype(np.int16)
    solid = np.isin(scenes[:, 15, :], list(T.SOLID))
    df["floor_solid_frac"] = solid.mean(axis=1).astype(np.float32)

    PROC.mkdir(parents=True, exist_ok=True)
    np.save(PROC / "scenes_u8.npy", scenes)
    df.to_parquet(PROC / "index.parquet", index=False)
    lv = pd.DataFrame({"level_id": range(len(chains)), "level_name": names, "game": ["SMB1"] * 20 + ["SMB2"] * 22,
                       "windows": [e - b for b, e in chains], "width": widths, "vglc_height": heights})
    lv.to_csv(PROC / "levels.csv", index=False)
    print(f"wrote {n} windows, {len(chains)} levels; scenes sha256 =",
          hashlib.sha256(scenes.tobytes()).hexdigest()[:16])


if __name__ == "__main__":
    main()
