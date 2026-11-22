"""Prompt normalizer tests (doc 12 §9): quantity rows, markers, floor/ceiling, negation, clamping, passthrough, dev gate."""
import json

import pytest
from conftest import ROOT, needs_data

from text2mario.captions.grammar import parse_phrase
from text2mario.text.normalizer import bucket, normalize


def caps(text):
    return [s.caption for s in normalize(text).scenes]


@pytest.mark.parametrize("text,expected", [
    ("a pipe", "one pipe."), ("two pipes", "two pipes."), ("a couple of pipes", "two pipes."),
    ("three pipes", "a few pipes."), ("a handful of pipes", "a few pipes."),
    ("6 enemies", "several enemies."), ("lots of coins", "several coins."),   # coins: 'many' is clamped on the train split
])
def test_quantity_expressions(text, expected):
    assert caps(text) == [expected] or "CLAMPED" in str(normalize(text).scenes[0].warnings)


def test_bucket_boundaries():
    assert [bucket(n) for n in (1, 2, 3, 4, 5, 9, 10, 40)] == ["one", "two", "a few", "a few", "several", "several", "many", "many"]


def test_synonyms_train_and_heldout():
    assert caps("one tube") == ["one pipe."]                     # train variant
    assert caps("one chimney") == ["one pipe."]                  # held-out variant
    assert caps("two bullet bills") == ["two cannons."]
    assert caps("a bad guy") == ["one enemy."]                   # 'guy' + 's' plural bug found while building this


def test_floor_and_ceiling_templates():
    assert caps("solid ground, one pipe") == ["full floor. one pipe."]
    assert caps("ground with 2 holes and an enemy") == ["floor with two gaps. one enemy."]
    assert "full ceiling" in caps("an underground stretch with a few bricks")[0]


def test_negation_goes_to_forbid_not_the_caption():
    n = normalize("full floor with three pipes and no enemies")
    assert n.scenes[0].forbid == ["enemy"] and "enem" not in n.scenes[0].caption
    assert caps("no gaps and zero cannons, two towers")[0].startswith("full floor.")


def test_multi_scene_markers_and_anchors():
    assert len(normalize("two pipes followed by a coin line").scenes) == 2
    n = normalize("begin with a mystery block and end with stairs going down")
    assert [s.caption for s in n.scenes] == ["one question block.", "one descending staircase."]
    assert len(normalize("a few pipes, two coins and a platform").scenes) == 1   # commas never split


def test_coin_line_implies_coins():
    n = normalize("two pipes followed by a coin line")
    assert n.scenes[1].caption == "a few coins. one coin line." and "IMPLIED_COINS" in n.scenes[1].warnings


def test_clamping_is_reported():
    n = normalize("nine towers")
    assert "CLAMPED" in " ".join(n.scenes[0].warnings)
    assert parse_phrase(n.scenes[0].caption.rstrip("."))                      # the clamped caption is still canonical


def test_output_is_always_canonical_grammar():
    for t in ["3 goombas and 2 pipes", "a ledge with a ? block above it", "5 coins in a row", "pyramid", "lava castle at night"]:
        for s in normalize(t).scenes:
            for ph in [p.strip() for p in s.caption.split(".") if p.strip()]:
                assert parse_phrase(ph)                                                       # raises (or is empty) outside the grammar


@needs_data
def test_every_stored_caption_passes_through_unchanged():
    import pandas as pd
    for c in sorted(set(pd.read_parquet(ROOT / "data/processed/index.parquet").caption.tolist()))[::7]:
        if c:
            assert [s.caption for s in normalize(c).scenes] == [c]


def test_dev_gate_n4_at_least_90_percent():
    import importlib.util
    spec = importlib.util.spec_from_file_location("eval_normalizer", ROOT / "scripts" / "eval_normalizer.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.sys.argv = ["x"]
    mod.main()
    rep = json.loads((ROOT / "reports/normalizer/dev.json").read_text())
    assert rep["N4_exact_match"] >= 0.90 and rep["N2_topic_f1"] >= 0.97
