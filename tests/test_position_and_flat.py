"""Regression tests for the second audit (U3, U5, U7): references are not counts, flat means flat, nothing mapped is refused.

Explain-me: each test fails on the code before this change (the headline prompt used to become 'two enemies. one pipe.' with the last
mention winning; 'flatland' / 'empty level' used to run the UNCONDITIONED model; 'castle with lava' too; position words were dropped silently).
"""
import pytest

from text2mario.app.service import explain_warning
from text2mario.generate import adherence
from text2mario.text.normalizer import normalize

HEADLINE = "three pipes and 5 enemies, 2 enemies before the first pipe and the rest after the 4th pipe"


def test_headline_keeps_the_declared_counts_and_flags_the_impossible_reference():
    n = normalize(HEADLINE)
    assert n.scenes[0].want["pipe"] == [3, 3]            # was [1, 1]: "the first pipe" / "the 4th pipe" are references, not extra pipes
    assert n.scenes[0].want["enemy"] == [5, 5]           # was [2, 2]: the later "2 enemies before..." is a part of the 5, not a new total
    assert "CONTRADICTION_ORDINAL:pipe:4:3" in n.warnings
    assert "POSITION_NOT_PLANNED" in n.warnings


def test_headline_is_refused_with_a_clear_sentence():
    with pytest.raises(ValueError, match="no 4th pipe|pipe number 4"):
        adherence.check_feasible(normalize(HEADLINE), 48)


def test_a_reachable_ordinal_is_accepted():
    n = normalize("three pipes and 5 enemies, 2 enemies before the third pipe")
    assert not any(w.startswith("CONTRADICTION") for w in n.warnings)
    adherence.check_feasible(n, 48)


@pytest.mark.parametrize("text", ["flatland", "empty level", "just a floor", "a completely flat level", "blank"])
def test_flat_requests_become_a_floor_with_everything_ruled_out(text):
    n = normalize(text)
    assert len(n.scenes) == 1 and n.scenes[0].caption == "full floor."
    for t in ("enemy", "pipe", "coin", "cannon", "question block", "platform"):
        assert t in n.scenes[0].forbid
    assert "FLAT_REQUEST" in n.warnings and "EMPTY_SCENE" not in n.warnings


@pytest.mark.parametrize("text", ["castle with lava", "make it spooky", "qwerty"])
def test_unmapped_sentences_are_refused_not_generated_unconditioned(text):
    n = normalize(text)
    assert not n.scenes
    with pytest.raises(ValueError, match="could not find anything"):
        adherence.check_feasible(n, 48)


def test_position_words_are_reported_not_dropped():
    n = normalize("tall pipe at the start and cannon at the end")
    assert "POSITION_NOT_PLANNED" in n.warnings
    assert "NOT controlled" in explain_warning("POSITION_NOT_PLANNED")


def test_negation_alone_still_works_and_is_not_flat():
    n = normalize("no enemies")
    assert n.scenes[0].forbid == ["enemy"] and "FLAT_REQUEST" not in n.warnings
