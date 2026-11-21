"""Unit tests for the adherence counter on HAND-BUILT grids whose true answer is known (audit section 2: 'is the counter itself right?')."""
import numpy as np
import yaml
from conftest import ROOT

from text2mario.data import tiles as T
from text2mario.eval import count_audit as CA

CFG = yaml.safe_load((ROOT / "configs/audit/adherence.yaml").read_text(encoding="utf-8"))
TOPICS = CFG["topics"]


def floor(w=40):
    g = np.zeros((16, w), np.uint8)
    g[15, :] = T.GROUND
    return g


def put_pipe(g, col, top):
    """Upright pipe: lip on row `top`, body below down to the floor (rows top+1..14)."""
    g[top, col], g[top, col + 1] = T.PIPE_TL, T.PIPE_TR
    for r in range(top + 1, 15):
        g[r, col], g[r, col + 1] = T.PIPE_BL, T.PIPE_BR


def put_hanging_pipe(g, col, length):
    """Upside-down pipe hanging from row 0: body rows 0..length-2, lip on row length-1."""
    for r in range(length - 1):
        g[r, col], g[r, col + 1] = T.PIPE_BL, T.PIPE_BR
    g[length - 1, col], g[length - 1, col + 1] = T.PIPE_TL, T.PIPE_TR


def counts(g):
    a = CA.objects_by_topic(g, TOPICS)
    return {k: len(v) for k, v in a.items()}


def test_one_pipe_is_one_even_though_it_has_many_tiles():
    g = floor()
    put_pipe(g, 10, 11)
    assert counts(g)["pipe"] == 1 and CA.independent_counts(g)["pipe"] == 1
    assert int(np.isin(g, list(T.PIPE)).sum()) > 6          # many tiles, still one pipe


def test_two_separate_pipes_and_two_touching_pipes():
    g = floor()
    put_pipe(g, 5, 12)
    put_pipe(g, 20, 10)
    assert counts(g)["pipe"] == 2 and CA.independent_counts(g)["pipe"] == 2
    t = floor()
    put_pipe(t, 8, 12)
    put_pipe(t, 10, 12)                                       # side by side: right half of A touches left half of B
    assert counts(t)["pipe"] == 2 and CA.independent_counts(t)["pipe"] == 2


def test_pipe_next_to_tower_counts_each_once():
    g = floor()
    put_pipe(g, 6, 11)
    g[12:15, 9] = T.GROUND                                    # a 1-wide, 3-high tower touching the pipe's right half
    g[12:15, 10] = T.GROUND
    c = counts(g)
    assert c["pipe"] == 1 and c["tower"] == 1
    assert CA.independent_counts(g)["pipe"] == 1


def test_upside_down_pipe_counts_as_a_pipe():
    g = floor()
    put_hanging_pipe(g, 12, 5)
    assert counts(g)["pipe"] == 1 and CA.independent_counts(g)["pipe"] == 1


def test_broken_pipe_is_still_counted_so_it_cannot_hide_a_wrong_count():
    g = floor()
    g[13, 4], g[13, 5] = T.PIPE_TL, T.PIPE_TR                 # a lip floating with no body above the floor row 14 -> body missing is fine
    g[14, 4] = T.PIPE_BL                                      # but only one half of the body: malformed
    assert counts(g)["pipe"] == 1


def test_enemies_and_coins_are_counted_per_cell_and_cannons_per_top():
    g = floor()
    g[14, 3] = g[14, 9] = T.ENEMY_ID
    g[5, 2:6] = T.COIN
    g[14, 20] = T.CANNON_TOP
    c = counts(g)
    assert c["enemy"] == 2 and c["coin"] == 4 and c["cannon"] == 1
    ind = CA.independent_counts(g)
    assert ind["enemy"] == 2 and ind["coin"] == 4 and ind["cannon"] == 1


def test_platform_and_staircase_and_coin_line():
    g = floor()
    g[9, 4:8] = T.GROUND                                      # a floating platform
    for k, h in enumerate((1, 2, 3, 4)):                      # an ascending staircase on the floor
        g[15 - h:15, 20 + k] = T.GROUND
    g[3, 30:35] = T.COIN                                      # a coin line of 5
    c = counts(g)
    assert c["platform"] == 1 and c["staircase"] == 1 and c["coin line"] == 1 and c["coin"] == 5


def test_scene_ownership_by_leftmost_column():
    anchors = {"pipe": [5, 30]}
    ok, d = CA.check_expectation(anchors, [{"pipe": [1, 1]}, {"pipe": [1, 1]}], starts=[0, 24], width=48)
    assert ok, d
    ok, d = CA.check_expectation(anchors, [{"pipe": [2, 2]}, {"pipe": [0, 0]}], starts=[0, 24], width=48)
    assert not ok


def test_order_must_be_strictly_left_to_right():
    a = {"pipe": [4], "coin": [20]}
    assert CA.check_expectation(a, [{}, {}], [0, 24], 48, order=["pipe", "coin"])[0]
    assert not CA.check_expectation(a, [{}, {}], [0, 24], 48, order=["coin", "pipe"])[0]
    assert not CA.check_expectation({"pipe": [4]}, [{}], [0], 48, order=["pipe", "coin"])[0]      # missing topic fails


def test_scene_count_mismatch_is_reported_and_judged_on_the_whole_level():
    ok, d = CA.check_expectation({"pipe": [3], "coin": [30, 31, 32, 33, 34]}, [{"pipe": [1, 1]}, {"coin": [5, 9]}], starts=[0], width=48)
    assert ok and d["scene_count_actual"] == 1 and "note" in d


def test_wilson_interval_is_sane():
    lo, hi = CA.wilson(10, 10)
    assert hi == 1.0 and 0.6 < lo < 0.75
    lo, hi = CA.wilson(0, 10)
    assert lo == 0.0 and 0.2 < hi < 0.4
    assert CA.wilson(5, 10)[0] < 0.5 < CA.wilson(5, 10)[1]
