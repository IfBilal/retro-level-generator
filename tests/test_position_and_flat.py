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
    n = normalize("a pipe before the first coin and an enemy after it")
    assert "POSITION_NOT_PLANNED" in n.warnings
    assert "NOT controlled" in explain_warning("POSITION_NOT_PLANNED")


def test_negation_alone_still_works_and_is_not_flat():
    n = normalize("no enemies")
    assert n.scenes[0].forbid == ["enemy"] and "FLAT_REQUEST" not in n.warnings


@pytest.mark.parametrize("text,expected", [("tall pipe at the start and cannon at the end", ["one pipe.", "one cannon."]),
                                           ("cannon at the end and two coins at the start", ["two coins.", "one cannon."])])
def test_start_and_end_phrases_become_ordered_scenes(text, expected):
    n = normalize(text)
    assert [s.caption for s in n.scenes] == expected      # was one scene 'one cannon. one pipe.' (order flipped, position dropped)
    assert "POSITION_NOT_PLANNED" not in n.warnings       # honoured, so nothing to apologise for


def test_counts_are_also_judged_on_what_the_player_sees(monkeypatch):
    """audit2 H8: the display repair deletes a pipe cut in half by the end template, so 3 raw pipes could show as 2.  evaluate_both must say unmet."""
    import numpy as np

    from text2mario.generate import sanitizer
    g = np.zeros((16, 48), np.uint8)
    g[15, :] = 8
    for c0 in (6, 20, 34):
        g[11, c0:c0 + 2] = (1, 2)
        g[12:15, c0:c0 + 2] = (9, 10)
    n = normalize("exactly 3 pipes")
    assert adherence.evaluate(g, n, [0] * 5, 48)["status"] == "met"          # the raw level is fine
    assert adherence.evaluate_both(g, n, [0] * 5, 48)["status"] == "met"      # and so is the shown one while nothing is deleted

    def delete_last_pipe(framed):
        d = framed.copy()
        d[11:15, 34 + 3:34 + 5] = 0                                          # +3: the spawn lead the framing puts in front
        return d, 8
    monkeypatch.setattr(sanitizer, "repair", delete_last_pipe)
    both = adherence.evaluate_both(g, n, [0] * 5, 48)
    assert both["status"] == "unmet" and both["constraints"][0]["got"] == 2
