"""Leakage and integrity tests for split_v1 (doc 10 §6, T1-T10). Part of gate G-P1."""
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from conftest import ROOT, needs_data
from text2mario.captions.grammar import CONCEPTS
from text2mario.data import provenance, splits

pytestmark = needs_data
COLS = ["c_" + c.replace(" ", "_") for c in CONCEPTS]


@pytest.fixture(scope="module")
def S():
    return splits.load(verify=True)


@pytest.fixture(scope="module")
def df(S):
    d = pd.read_parquet(ROOT / "data/processed/index.parquet")
    m = {l: sp for sp, ls in S["splits"].items() for l in ls}
    d["split"] = d.level_id.map(m)
    return d


def test_t1_disjoint_and_complete(S):
    tr, dv, te = (set(S["splits"][k]) for k in ("train", "dev", "test"))
    assert not (tr & dv) and not (tr & te) and not (dv & te)
    assert tr | dv | te == set(range(42))


def test_t2_families_in_train(S):
    train = set(S["splits"]["train"])
    for f in S["families"]:
        assert set(f["levels"]) <= train


def test_t3_composition(S):
    game = {l["level_id"]: l["game"] for l in S["levels"]}
    assert sorted(game[l] for l in S["splits"]["test"]) == ["SMB1"] * 3 + ["SMB2"] * 3
    assert len(S["splits"]["dev"]) == 5 and len(S["splits"]["train"]) == 31
    assert sorted(S["splits"]["dev"] + S["splits"]["test"] + S["splits"]["train"]) == list(range(42))


def test_t4_shares(S):
    n = sum(S["window_counts"].values())
    assert 0.11 <= S["window_counts"]["test"] / n <= 0.17
    assert 0.09 <= S["window_counts"]["dev"] / n <= 0.15


def test_t5_coverage(S, df):
    for sp in ("dev", "test"):
        assert (df[df.split == sp][COLS].sum() >= 10).all(), sp
    train = df[df.split == "train"]
    levels_with = train.groupby("level_id")[COLS].sum() > 0
    assert (levels_with.sum() >= 2).all()


def test_t6_nn_leakage(S):
    for sp in ("dev", "test"):
        lk = S["leakage"][sp]
        assert lk["le2"] <= 0.03 and lk["eq0"] <= 0.005 and lk["nn_median"] >= 12


def test_t7_reused_segments_list_is_current(S):
    """Rebuild the list from scratch and compare, so a stale file fails."""
    from text2mario.data.hamming import nn_distance
    df = pd.read_parquet(ROOT / "data/processed/index.parquet")
    scenes = np.load(ROOT / "data/processed/scenes_u8.npy")
    m = {l: sp for sp, ls in S["splits"].items() for l in ls}
    sp = df.level_id.map(m).to_numpy()
    tr = np.flatnonzero(sp == "train")
    listed = {r["idx"] for r in json.loads((ROOT / "data/splits/reused_segments.json").read_text())}
    now = set()
    for s in ("dev", "test"):
        idx = np.flatnonzero(sp == s)
        dist, _ = nn_distance(scenes[idx], scenes[tr])
        now |= set(idx[dist <= 2].tolist())
    assert now == listed


def test_t8_lock(S):
    want = (ROOT / "data/splits/LOCK").read_text().strip()
    assert splits.lock_hash(S, provenance.corpus_content_hash()) == want


def test_t8b_lock_detects_edit(S):
    edited = json.loads(json.dumps(S))
    edited["splits"]["test"][0] = 99
    assert splits.lock_hash(edited, provenance.corpus_content_hash()) != (ROOT / "data/splits/LOCK").read_text().strip()


@pytest.mark.slow
def test_t9_determinism():
    out = subprocess.run([sys.executable, str(ROOT / "scripts/make_splits.py"), "--check"], capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr


def test_cv_folds_cover_non_test_levels(S):
    flat = sorted(l for f in S["cv_folds"] for l in f)
    assert flat == sorted(S["splits"]["train"] + S["splits"]["dev"])
    for f in S["families"]:  # a family is indivisible
        assert sum(set(f["levels"]) <= set(fold) for fold in S["cv_folds"]) == 1


def test_split_json_is_canonical_and_free_of_timestamps(S):
    assert "created_utc" not in S
