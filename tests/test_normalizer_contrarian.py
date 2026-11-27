"""Regression tests for the contrarian audit findings (reports/audit2/CONTRARIAN.md): each case was mis-read silently before."""
import pytest

from text2mario.app.service import explain_warning, validate_prompt
from text2mario.text.normalizer import normalize


def one(text):
    return normalize(text).scenes[0]


@pytest.mark.parametrize("text,forbidden", [
    ("without any pipes or enemies", {"pipe", "enemy"}), ("I don't want any enemies", {"enemy"}), ("not a single enemy", {"enemy"}),
    ("avoid pipes", {"pipe"}), ("a level free of enemies", {"enemy"}), ("a pipe-free level", {"pipe"}), ("skip the enemies", {"enemy"}),
    ("remove all pipes", {"pipe"}), ("get rid of the cannons", {"cannon"}), ("enemies: none", {"enemy"}),
])
def test_many_ways_of_saying_no_all_mean_no(text, forbidden):
    sc = one(text)
    assert set(sc.forbid) >= forbidden and not (set(sc.want) & forbidden)        # was: two enemies / two pipes requested


def test_a_colon_after_a_prohibition_keeps_the_prohibition():
    sc = one("no pipes: 2 coins")
    assert sc.forbid == ["pipe"] and sc.want == {"coin": [2, 2]}


def test_a_colon_after_a_noun_introduces_its_amount():
    assert one("pipes: 2").want == {"pipe": [2, 2]}


def test_ordinal_words_do_not_delete_the_nouns_around_them():
    sc = one("2 pipes in the first half, 3 coins in the second half")
    assert sc.want == {"pipe": [2, 2], "coin": [3, 3]}                           # was: the coins vanished
    assert normalize("before the first pipe, 2 enemies").scenes[0].want == {"enemy": [2, 2]}        # was: EMPTY_SCENE


@pytest.mark.parametrize("text", ["an empty gap", "a plain gap", "there is nothing between the pipes"])
def test_flat_words_inside_a_real_request_do_not_forbid_everything(text):
    n = normalize(text)
    assert "FLAT_REQUEST" not in n.warnings and not any(len(sc.forbid) > 3 for sc in n.scenes)


@pytest.mark.parametrize("text,n", [("twenty-one pipes", 21), ("three separate pipes", 3), ("3pipes", 3), ("３ pipes", 3), ("pipes (4)", 4)])
def test_numbers_written_in_other_ways_are_read(text, n):
    assert one(text).want == {"pipe": [n, n]}


@pytest.mark.parametrize("text", ["2.5 pipes", "3-4 pipes", "3 to 5 pipes"])
def test_ranges_and_decimals_are_refused_with_a_message(text):
    ok, msg = validate_prompt(text)
    assert not ok and "whole number" in msg


def test_a_staircase_up_is_one_staircase():
    n = normalize("a staircase up then a staircase down")
    assert [sc.want for sc in n.scenes] == [{"ascending staircase": [1, 1]}, {"descending staircase": [1, 1]}]


def test_gap_counts_are_flagged_as_not_verified():
    assert "GAPS_NOT_VERIFIED" in normalize("nine gaps").warnings
    assert "NOT checked" in explain_warning("GAPS_NOT_VERIFIED")


@pytest.mark.parametrize("text", ["2 pipes - 3 coins", "level 1 - 3 pipes", "two pipes\r\nthree coins"])
def test_dashes_and_windows_line_breaks_are_not_errors(text):
    assert validate_prompt(text)[0]


def test_a_real_negative_number_is_still_refused():
    assert not validate_prompt("-3 pipes")[0] and not validate_prompt("pipes (-3)")[0]
