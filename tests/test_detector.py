"""Caption detector tests (doc 09 §6.4): one hand-built grid per rule, plus reproduction of the stored captions."""
import json

import numpy as np
import pytest
from conftest import ROOT, needs_data

from text2mario.captions.detector import caption, detect
from text2mario.captions.grammar import counted_phrase, parse_phrase, quantity_word
from text2mario.data.tiles import CHAR_TO_ID


def grid(rows_from_bottom: list[str], width: int = 16) -> np.ndarray:
    """16 x width grid; `rows_from_bottom[0]` is row 15, [1] is row 14, ...; unspecified rows are sky."""
    g = np.zeros((16, width), np.uint8)
    for k, line in enumerate(rows_from_bottom):
        line = line.ljust(width, "-")
        g[15 - k] = [CHAR_TO_ID[c] for c in line]
    return g


FULL = "X" * 16


def test_quantity_words_and_plurals():
    assert [quantity_word(n) for n in (1, 2, 3, 4, 5, 9, 10, 40)] == \
        ["one", "two", "a few", "a few", "several", "several", "many", "many"]
    assert counted_phrase("enemy", 2) == "two enemies"
    assert counted_phrase("coin line", 1) == "one coin line"


def test_parse_phrase_codes():
    assert parse_phrase("full floor") == ("floor", 1)
    assert parse_phrase("giant gap with two chunks of floor") == ("floor", 8)
    assert parse_phrase("several loose blocks") == ("loose block", 4)
    with pytest.raises(ValueError):
        parse_phrase("three pipes")


def test_full_floor():
    assert caption(grid([FULL])) == "full floor."


def test_no_floor_phrase_when_all_passable():
    assert caption(grid([])) == ""


def test_bottom_row_of_enemies_only_has_no_floor_phrase():
    """Regression: a generated bottom row with no solid cell and not all passable must not crash (was quantity_word(0))."""
    assert "floor" not in caption(grid(["E" * 16])), "no floor phrase expected"


def test_floor_gaps_are_counted_as_runs():
    assert caption(grid(["XXXX--XXXXXXXXXX"])) == "floor with one gap."
    assert caption(grid(["XXX--XXX--XXXXXX"])) == "floor with two gaps."


def test_giant_gap_counts_chunks():
    assert caption(grid(["XXX-----XXX-----"])) == "giant gap with two chunks of floor."


def test_enemy_on_floor_row_counts_as_gap():
    d = detect(grid(["XXXXEXXXXXXXXXXX"]))
    assert d[0].phrase == "floor with one gap" and d[1].phrase == "one enemy"


def test_double_thick_floor_is_not_loose_blocks():
    assert caption(grid([FULL, FULL])) == "full floor."


def test_ceiling():
    g = grid([FULL])
    g[4] = CHAR_TO_ID["X"]
    assert caption(g) == "full floor. full ceiling."
    g[4, 5] = 0
    assert caption(g) == "full floor. ceiling with one gap."


def test_enemy_question_coin_counts():
    g = grid([FULL])
    g[10, 2] = g[10, 5] = CHAR_TO_ID["E"]
    g[6, 3] = CHAR_TO_ID["?"]
    g[6, 9] = CHAR_TO_ID["Q"]
    g[3, 1] = g[5, 7] = g[8, 12] = CHAR_TO_ID["o"]
    assert caption(g) == "full floor. two enemies. two question blocks. a few coins."


def test_coin_line():
    g = grid([FULL])
    g[8, 2:6] = CHAR_TO_ID["o"]
    assert caption(g) == "full floor. a few coins. one coin line."


def test_cannon():
    g = grid([FULL])
    g[13, 4], g[14, 4] = CHAR_TO_ID["B"], CHAR_TO_ID["b"]
    assert "one cannon" in caption(g)


def test_broken_cannon_support_detected():
    g = grid([FULL])
    g[13, 4] = CHAR_TO_ID["b"]  # support with no cannon above
    assert "one broken cannon" in caption(g)


def test_platform():
    g = grid([FULL])
    g[9, 3:7] = CHAR_TO_ID["S"]
    assert caption(g) == "full floor. one platform."


def pipe(g, col, top, bottom=14, flip=False):
    """Draw a pipe with a lip on top (or on the bottom if flip) between rows top..bottom."""
    for r in range(top, bottom + 1):
        lip = (r == bottom) if flip else (r == top)
        g[r, col], g[r, col + 1] = (CHAR_TO_ID["<"], CHAR_TO_ID[">"]) if lip else (CHAR_TO_ID["["], CHAR_TO_ID["]"])


def test_pipe_and_two_adjacent_pipes_stay_separate():
    g = grid([FULL])
    pipe(g, 3, 12)
    assert caption(g) == "full floor. one pipe."
    pipe(g, 5, 12)
    assert caption(g) == "full floor. two pipes."


def test_left_edge_partial_pipe_is_valid():
    g = grid([FULL])
    g[12, 0], g[13, 0], g[14, 0] = CHAR_TO_ID[">"], CHAR_TO_ID["]"], CHAR_TO_ID["]"]
    assert caption(g) == "full floor. one pipe."


def test_upside_down_pipe_from_row_zero():
    g = grid([FULL])
    pipe(g, 6, 0, 4, flip=True)
    assert "one upside down pipe" in caption(g)


def test_floating_upside_down_pipe_is_broken():
    g = grid([FULL])
    pipe(g, 6, 3, 6, flip=True)  # sky above the body -> invalid
    assert "one broken pipe" in caption(g)


def test_pipe_without_partner_is_broken():
    g = grid([FULL])
    g[12, 4] = CHAR_TO_ID["<"]
    assert "one broken pipe" in caption(g)


def test_tower_vs_rectangular_cluster():
    g = grid([FULL])
    g[11:15, 4] = CHAR_TO_ID["X"]  # 1 wide, 4 tall, standing on the floor
    assert caption(g) == "full floor. one tower."
    g = grid([FULL])
    g[12:15, 4:8] = CHAR_TO_ID["S"]  # 4 wide, 3 tall block
    assert caption(g) == "full floor. one rectangular block cluster."


def test_irregular_cluster():
    g = grid([FULL])
    g[13:15, 4] = g[14, 5] = g[14, 6] = CHAR_TO_ID["S"]  # L-shape, not a rectangle, wider than a tower
    assert caption(g) == "full floor. one irregular block cluster."


def stairs(g, col, steps, ascending=True):
    for k in range(steps):
        height = k + 1 if ascending else steps - k
        g[15 - height:15, col + k] = CHAR_TO_ID["X"]


def test_ascending_and_descending_staircases():
    g = grid([FULL])
    stairs(g, 2, 4, ascending=True)
    assert caption(g) == "full floor. one ascending staircase."
    g = grid([FULL])
    stairs(g, 2, 4, ascending=False)
    assert caption(g) == "full floor. one descending staircase."


def test_loose_blocks_remain_after_structures():
    g = grid([FULL])
    g[10, 2] = CHAR_TO_ID["S"]
    g[7, 9] = CHAR_TO_ID["X"]
    assert caption(g) == "full floor. two loose blocks."


def test_detections_carry_cells():
    g = grid([FULL])
    g[10, 2] = CHAR_TO_ID["E"]
    enemy = next(d for d in detect(g) if d.topic == "enemy")
    assert enemy.cells == frozenset({(10, 2)})


@needs_data
def test_reproduces_all_stored_captions():
    """The detector was written from the grammar; it must reproduce the stored captions (doc 09 §6.4)."""
    recs = json.loads((ROOT / "data/source/Mar1and2_LevelsAndCaptions-regular.json").read_text())
    bad = [i for i, r in enumerate(recs) if caption(np.array(r["scene"], np.uint8)) != r["caption"]]
    assert not bad, f"{len(bad)} mismatches, first: {bad[:5]}"
