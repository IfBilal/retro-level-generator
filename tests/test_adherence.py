"""C-score golden vectors (doc 22 §3.1), symmetry on real caption pairs, exact match and per-concept metrics."""
import itertools
import random

import pytest
from conftest import needs_data

from text2mario.eval.adherence import c_score, concept_report, exact_match, extract, rare_exact_match

GOLDEN = [
    ("full floor. one enemy.", "full floor. one enemy.", 1.000000),
    ("full floor. one enemy.", "one enemy. full floor.", 1.000000),
    ("full floor. one enemy.", "full floor. two enemies.", 0.986111),
    ("full floor. a few enemies. one pipe.", "floor with one gap. a few enemies.", 0.838889),
    ("floor with one gap. two enemies. one platform. one tower.",
     "giant gap with one chunk of floor. two enemies. one platform. one tower.", 1.000000),
    ("full floor. several coins.", "full floor. many coins. one broken pipe.", 0.875000),
    ("a few platforms.", "", 0.888889),
    ("full floor. one coin line. a few coins.", "full floor. two coin lines.", 0.875000),
    ("full floor. many enemies.", "full floor. one enemy.", 0.944444),
    ("one upside down pipe. full floor.", "full floor. one pipe.", 0.777778),
    ("full floor. one ascending staircase.", "full floor. one descending staircase.", 0.777778),
    ("full floor. two pipes.", "full floor. one broken cannon. two pipes.", 0.888889),
]


@pytest.mark.parametrize("prompt,gen,expected", GOLDEN)
def test_golden_vectors(prompt, gen, expected):
    assert c_score(prompt, gen) == pytest.approx(expected, abs=1e-6)


def test_topic_order_keeps_specific_before_general():
    e = extract("one broken pipe. one pipe. one coin line. one coin.")
    assert e["broken pipe"] == "one broken pipe" and e["pipe"] == "one pipe"
    assert e["coin line"] == "one coin line" and e["coin"] == "one coin"


def test_exact_match_is_order_free_and_plural_blind():
    assert exact_match("full floor. two enemies.", "two enemy. full floor.")
    assert not exact_match("full floor. two enemies.", "full floor. one enemy.")


def test_concept_report_counts():
    pairs = [("full floor. one pipe.", "full floor. one pipe.")] * 5 + [("full floor. one pipe.", "full floor.")] * 5
    r = concept_report(pairs)
    assert r["pipe"]["support"] == 10 and r["pipe"]["recall"] == 0.5 and r["pipe"]["exact_match"] == 0.5
    assert r["floor"]["f1"] == 1.0 and not r["pipe"]["na"] and r["ceiling"]["na"]
    assert rare_exact_match(r) != rare_exact_match(r)  # nan: no rare concept requested


@needs_data
def test_symmetry_on_real_caption_pairs():
    import pandas as pd
    from conftest import ROOT
    caps = pd.read_parquet(ROOT / "data/processed/index.parquet").caption.tolist()
    rng = random.Random(0)
    for a, b in (rng.sample(caps, 2) for _ in range(2000)):
        assert c_score(a, b) == pytest.approx(c_score(b, a), abs=1e-12)
    assert c_score(caps[0], caps[0]) == 1.0
    _ = itertools  # (kept for readability of the sampling above)
