"""Structure / novelty / JS metric tests (doc 22 §3.4-3.10) against hand-built grids and the real-data references."""
import json

import numpy as np
import pytest
from conftest import ROOT, needs_data

from text2mario.data.tiles import CHAR_TO_ID
from text2mario.eval import structure as S


def grid(rows_from_bottom, width=16):
    g = np.zeros((16, width), np.uint8)
    for k, line in enumerate(rows_from_bottom):
        g[15 - k] = [CHAR_TO_ID[c] for c in line.ljust(width, "-")]
    return g


FULL = "X" * 16


def test_valid_floor_cases():
    assert S.valid_floor(grid([FULL]))
    void = grid([])
    assert not S.valid_floor(void)                                   # a window that is entirely sky is a void
    floating = grid([])
    floating[8, 3:6] = CHAR_TO_ID["X"]
    assert S.valid_floor(floating)                                   # floorless but with structure: legal gap pattern
    bad = grid([FULL])
    bad[15, 4] = CHAR_TO_ID["o"]                                     # a coin in the floor row is illegal
    assert not S.valid_floor(bad)


def test_broken_counts_and_rate():
    ok = grid([FULL])
    bad = grid([FULL])
    bad[12, 4] = CHAR_TO_ID["<"]                                     # lip without partner
    cannon = grid([FULL])
    cannon[13, 2] = CHAR_TO_ID["b"]                                  # support without cannon above
    assert S.broken_counts(ok) == (0, 0, 0)
    assert S.broken_counts(bad)[0] == 1 and S.broken_counts(cannon)[1] == 1
    assert S.broken_rate(np.stack([ok, bad, cannon, ok])) == 0.5


def test_novelty_and_copy_rate():
    rng = np.random.default_rng(0)
    train = rng.integers(0, 13, (50, 16, 16)).astype(np.uint8)
    q = train[:10].copy()
    q[5:, 0, 0] = (q[5:, 0, 0] + 1) % 13                              # 5 exact copies, 5 at distance 1
    n = S.novelty(q, train)
    assert n["exact_copy_rate"] == 0.5 and n["nn"].max() == 1
    assert S.copy_rates(n["nn"])["copy_rate"] == 1.0 and S.copy_rates(n["nn"])["copy_rate_strict"] == 1.0


def test_diversity_and_amed():
    a = np.zeros((16, 16), np.uint8)
    b = a.copy()
    b[0, :4] = 1                                                      # distance 4 from a
    assert S.diversity([np.stack([a, b])]) == 4.0
    assert S.amed_self(np.stack([a, b, b])) == pytest.approx((4 + 0 + 0) / 3)


def test_js_extremes():
    train = np.random.default_rng(1).integers(0, 13, (40, 16, 16)).astype(np.uint8)
    assert S.tile_js(train, train) == pytest.approx(0.0, abs=1e-12)
    a, b = np.zeros((2, 16, 16), np.uint8), np.full((2, 16, 16), 5, np.uint8)
    assert S.tile_js(a, b) == pytest.approx(1.0)                      # disjoint supports -> 1 bit


@needs_data
def test_real_reference_values():
    ref = json.loads((ROOT / "reports/references/REFERENCES.json").read_text())
    assert ref["floor_valid_real_dev"] >= 0.95                        # calibration rule for G-P3 (Q9)
    assert ref["broken_structures_in_real_train_dev"] == 0            # real data has no broken structures (A11, doc 22 §3.4)
    assert ref["novelty_ref_median_stride8"] == 13.0                  # E4 reference for the locked split
    assert 0.06 < ref["tile_js_all_sky"] < 0.07 and ref["tile_js_real_dev_vs_train"] < 0.025
    assert ref["G-P3_copy_rate_threshold"] == pytest.approx(ref["copy_rate_ref"] + 0.05, abs=1e-4)
