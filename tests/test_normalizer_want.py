"""Decision coverage of the normalizer's `want` / `forbid` logic and of the plain-English warning texts (audit P2, rubric row 7)."""
import pytest

from text2mario.app.service import WARNING_TEXT, explain_warning
from text2mario.text.normalizer import normalize


@pytest.mark.parametrize("text,scenes", [
    ("a pyramid", [{"ascending staircase": [1, 1]}]),
    ("two pyramids", [{"ascending staircase": [2, 2]}]),
    ("a ? block", [{"question block": [1, 1]}]),
    ("5 coins in a row", [{"coin": [5, 5], "coin line": [1, 1]}]),
    ("a coin line", [{"coin line": [1, 1], "coin": [3, 4]}]),                      # a coin line implies a few coins (IMPLIED_COINS)
    ("a coin line and one coin", [{"coin line": [1, 1], "coin": [2, 2]}]),         # one coin cannot form a line: two
    ("two coin lines", [{"coin line": [2, 2]}]),
    ("two pipes and three pipes", [{"pipe": [3, 3]}]),                             # duplicate topic: last wins, with a warning
    ("a few coins and several coins", [{"coin": [5, 9]}]),
    ("one pipe and one pipe", [{"pipe": [1, 1]}]),
    ("an underground level with one pipe", [{"pipe": [1, 1]}]),
    ("three towers", [{"tower": [3, 3]}]),                                         # caption clamps to 'two'; the request stays 3
    ("one pipe, then two pipes", [{"pipe": [1, 1]}, {"pipe": [2, 2]}]),
])
def test_want_per_scene(text, scenes):
    n = normalize(text)
    assert [dict(s.want) for s in n.scenes] == scenes


@pytest.mark.parametrize("text,forbid", [("no ceiling", ["ceiling"]), ("zero enemies", ["enemy"]), ("no more pipes", ["pipe"]),
                                         ("without cannons", ["cannon"]), ("no gaps", [])])
def test_negation_forms(text, forbid):
    n = normalize(text)
    assert n.scenes and [f for s in n.scenes for f in s.forbid] == forbid
    assert all(not s.want for s in n.scenes)                                       # nothing positive was asked for


def test_duplicate_topic_warns():
    assert "DUPLICATE_TOPIC" in normalize("two pipes and three pipes").scenes[0].warnings


def test_clamped_request_keeps_the_exact_number_and_says_so():
    s = normalize("three towers").scenes[0]
    assert s.caption == "two towers." and s.want["tower"] == [3, 3] and any(w.startswith("CLAMPED") for w in s.warnings)


def test_contradiction_in_one_scene_vs_across_scenes():
    n = normalize("one pipe and no pipes")
    assert any(w.startswith("CONTRADICTION") for w in n.warnings)
    n = normalize("one pipe, then no pipes")                                       # different scenes: allowed, not a contradiction
    assert not any(w.startswith("CONTRADICTION") for w in n.warnings)


def test_empty_and_unknown_text_have_no_constraints():
    from text2mario.generate import adherence as A
    for t in ("", "   ", "flibber the glorp", "dos tuberias"):
        n = normalize(t)
        assert not n.scenes and A.constraints(n) == []


# ---------------------------------------------------------------------------------------------- plain-English warnings
def test_every_warning_code_the_normalizer_can_emit_has_a_sentence():
    codes = ["BARE_PLURAL_DEFAULT", "DUPLICATE_TOPIC", "EMPTY_SCENE", "IMPLIED_COINS", "NEGATION_FORBID", "NEGATION_ONLY", "POSITIONAL_IGNORED",
             "STYLE_HINT", "THEME_UNSUPPORTED", "TOO_MANY_SCENES"]
    for c in codes:
        assert c in WARNING_TEXT and explain_warning(c) and "_" not in explain_warning(c)


def test_parametrised_and_unknown_codes():
    assert "became" in explain_warning("CLAMPED:tower:a few->two") and "tower" in explain_warning("CLAMPED:tower:a few->two")
    assert "pipe" in explain_warning("CONTRADICTION:pipe")
    assert explain_warning("CANONICAL_PASSTHROUGH") == ""                         # internal bookkeeping is not shown
    assert explain_warning("SOME_NEW_CODE") == "some new code"                     # unknown codes degrade to readable text, never crash


@pytest.mark.parametrize("text", ["500 pipes", "three towers", "two pipes and three pipes", "a coin line", "no ceiling", "a pyramid",
                                  "one pipe, then two enemies, then no coins", "castle with lava", "make it hard and brutal"])
def test_every_visible_warning_of_real_prompts_is_plain_text(text):
    n = normalize(text)
    for w in list(n.warnings) + [w for s in n.scenes for w in s.warnings]:
        msg = explain_warning(w)
        assert msg == "" or (msg[0].isupper() and msg.endswith("."))
