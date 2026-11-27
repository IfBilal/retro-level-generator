"""Regression tests for reports/audit2/ADVERSARY2.md (570 invented prompts, about 22% disagreed): the fixed groups, plus the rule that
what the parser cannot follow is REFUSED or REPORTED, never silently misread."""
import pytest

from text2mario.app.service import explain_warning, validate_prompt
from text2mario.generate.adherence import check_feasible
from text2mario.text.normalizer import normalize


def forbids(text):
    return {t for sc in normalize(text).scenes for t in sc.forbid}


@pytest.mark.parametrize("text,topic", [("w/o pipes", "pipe"), ("sans pipes", "pipe"), ("other than pipes", "pipe"), ("don't use cannons", "cannon"),
                                        ("stay away from pipes", "pipe"), ("ban the enemies", "enemy"), ("without a single pipe", "pipe"),
                                        ("not one pipe", "pipe"), ("3 coins, but not pipes", "pipe"), ("I'd rather not have enemies", "enemy"),
                                        ("neither pipes nor coins", "pipe")])
def test_more_ways_of_saying_no(text, topic):
    assert topic in forbids(text)


@pytest.mark.parametrize("text", ["not many coins", "everything but pipes", "3 pipes with 2 goombas each", "at most 2 pipes", "2 or 3 pipes",
                                  "at least 3 coins", "up to 4 enemies"])
def test_phrases_that_change_the_meaning_of_a_number_are_refused(text):
    with pytest.raises(ValueError, match="exact amounts"):
        check_feasible(normalize(text), 48)


def test_unknown_words_are_reported_not_dropped():
    n = normalize("two pipez and a coin")
    assert any(w.startswith("UNKNOWN_WORDS:") and "pipez" in w for w in n.warnings)
    assert "pipez" in explain_warning(next(w for w in n.warnings if w.startswith("UNKNOWN_WORDS")))


def test_a_clean_sentence_has_no_unknown_word_warning():
    assert not any(w.startswith("UNKNOWN_WORDS") for w in normalize("full floor with two pipes and a few enemies").warnings)


@pytest.mark.parametrize("text,n", [("a goomba and a koopa", 2), ("2 pipes and 3 tubes", 5)])
def test_different_words_for_the_same_object_add_up(text, n):
    sc = normalize(text).scenes[0]
    assert next(iter(sc.want.values())) == [n, n] and "DUPLICATE_TOPIC" in normalize(text).warnings


def test_colon_lists_with_words_and_equals_signs():
    assert normalize("pipes: three, coins: two").scenes[0].want == {"pipe": [3, 3], "coin": [2, 2]}
    assert normalize("pipes = 3, coins = 2").scenes[0].want == {"pipe": [3, 3], "coin": [2, 2]}
    assert normalize("hard: 3 goombas").scenes[0].want == {"enemy": [3, 3]}


def test_thousands_separator_is_not_zero_and_decimal_commas_are_refused():
    assert normalize("2,000 coins").scenes[0].want == {"coin": [2000, 2000]}     # was: no coins
    assert not validate_prompt("2,5 pipes")[0]


def test_world_numbers_are_not_ranges():
    assert validate_prompt("world 1-1 with two pipes")[0]


def test_absurd_digit_strings_do_not_crash_the_parser():
    normalize("9" * 5000 + " pipes")                                          # was ValueError: Exceeds the limit (4300 digits)


def test_bare_plural_default_is_visible():
    assert "BARE_PLURAL_DEFAULT" in normalize("stairs up").warnings
