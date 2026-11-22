"""Fuzzing: nothing a user can type and nothing the model can sample may crash the pipeline (rubric row 7: invalid / boundary cases)."""
import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from text2mario.app.service import validate_prompt, validate_settings
from text2mario.captions.detector import caption, detect
from text2mario.captions.grammar import parse_phrase
from text2mario.eval import adherence as A
from text2mario.generate.sanitizer import repair
from text2mario.playability.checker import check_level
from text2mario.text.normalizer import normalize

FAST = settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])


@FAST
@given(st.text(max_size=600))
def test_normalizer_never_crashes_and_only_emits_canonical_captions(text):
    out = normalize(text)
    for sc in out.scenes:
        for ph in [p.strip() for p in sc.caption.split(".") if p.strip()]:
            concept, code = parse_phrase(ph)          # raises if the output is outside the training grammar
            assert concept and code >= 1


@FAST
@given(st.lists(st.sampled_from(["pipe", "pipes", "goomba", "coin", "no", "without", "then", "three", "a few", "lots of", "gap", "ceiling",
                                 "staircase", "stairs going down", "?", ",", ".", "and", "zero", "big", "left", "castle", "10", "2-3"]), max_size=30))
def test_normalizer_on_vocabulary_soup(words):
    out = normalize(" ".join(words))
    assert isinstance(out.scenes, list) and all(sc.caption and isinstance(sc.want, dict) for sc in out.scenes)


@FAST
@given(st.one_of(st.none(), st.integers(), st.floats(allow_nan=True), st.text(max_size=50), st.lists(st.integers(), max_size=3), st.binary(max_size=10)))
def test_prompt_validator_accepts_any_python_object(x):
    ok, msg = validate_prompt(x)
    assert isinstance(ok, bool) and isinstance(msg, str) and (ok or msg)


@FAST
@given(st.one_of(st.none(), st.integers(), st.floats(allow_nan=True), st.text(max_size=5)), st.one_of(st.none(), st.integers(), st.floats(allow_nan=True), st.text(max_size=5)),
       st.one_of(st.none(), st.integers(), st.floats(allow_nan=True), st.text(max_size=5)))
def test_settings_validator_accepts_any_python_object(seed, w, width):
    ok, msg = validate_settings(seed, w, width)
    assert isinstance(ok, bool) and (ok or msg)


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large])
@given(st.integers(0, 2**31 - 1), st.sampled_from([16, 24, 48]), st.floats(0.0, 1.0))
def test_detector_sanitizer_and_checker_survive_random_grids(seed, width, density):
    """Model samples can be arbitrary tile soup: the detector, the display repair and the checker must still return."""
    rng = np.random.default_rng(seed)
    g = np.where(rng.random((16, width)) < density, rng.integers(0, 13, (16, width)), 0).astype(np.uint8)
    cap = caption(g)
    detect(g)
    fixed, _ = repair(g)
    assert fixed.shape == g.shape
    A.c_score("full floor. two pipes.", cap)
    r = check_level(g, mode="scene", budget_nodes=300)
    assert r.status in ("solved", "unsolved", "timeout")


@pytest.mark.parametrize("g", [np.zeros((16, 16), np.uint8), np.full((16, 16), 8, np.uint8), np.full((16, 16), 5, np.uint8), np.full((16, 16), 12, np.uint8),
                               np.full((16, 1), 8, np.uint8), np.zeros((16, 200), np.uint8)])
def test_degenerate_grids(g):
    assert isinstance(caption(g), str)
    assert check_level(g, mode="scene", budget_nodes=300).status in ("solved", "unsolved", "timeout")
