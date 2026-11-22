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


# ---------------------------------------------------------------------------------------------- towers as a person sees them (ledger AD9)
def column_stack(g, col, height):
    g[15 - height:15, col] = T.GROUND


def test_isolated_pillars_are_towers_one_or_two_wide():
    g = floor()
    column_stack(g, 5, 3)
    for c in (12, 13):
        column_stack(g, c, 6)
    assert CA.visible_towers(g) == [5, 12]


def test_staircase_top_is_not_a_tower():
    g = floor()
    for k, h in enumerate((1, 2, 3, 4, 5)):
        column_stack(g, 20 + k, h)
    assert CA.visible_towers(g) == []                              # the detector calls the narrow top a tower; a person sees stairs
    assert counts(g)["tower"] == 0 and counts(g)["staircase"] == 1


def test_pillar_next_to_a_staircase_counts_once_and_pipes_never():
    g = floor(48)
    for k, h in enumerate((1, 2, 3)):
        column_stack(g, 5 + k, h)
    column_stack(g, 20, 4)
    put_pipe(g, 30, 11)
    assert CA.visible_towers(g) == [20]


def test_neighbour_only_one_lower_is_not_a_tower_and_wide_blocks_are_not_towers():
    g = floor()
    column_stack(g, 8, 3)
    column_stack(g, 9, 2)                                          # the right neighbour is only ONE lower: a step, not a pillar
    for c in range(12, 16):
        column_stack(g, c, 4)                                      # a 4-wide block
    for c in (20, 21, 22):
        column_stack(g, c, 5)                                      # a 3-wide block
    assert CA.visible_towers(g) == []
    h = floor()
    column_stack(h, 4, 4)
    column_stack(h, 5, 3)                                          # uneven 2-wide pair with open sides is one (notched) tower
    assert CA.visible_towers(h) == [4]


def test_two_pillars_separated_by_a_single_gap_are_two_towers():
    g = floor()
    column_stack(g, 8, 4)
    column_stack(g, 10, 4)
    assert CA.visible_towers(g) == [8, 10]


def test_tower_on_a_platform_or_without_floor_is_not_counted():
    g = floor()
    g[15, 6] = T.SKY                                               # a pit: the column has no floor
    column_stack(g, 6, 4)
    assert CA.visible_towers(g) == []
    assert CA.visible_towers(np.zeros((16, 8), np.uint8)) == []


def test_detector_tower_count_is_kept_as_a_cross_check():
    g = floor()
    for k, h in enumerate((1, 2, 3, 4, 5)):
        column_stack(g, 20 + k, h)
    obj = CA.detector_objects(g)
    assert "tower (detector)" in obj and obj["tower"] == []
    assert "tower (detector)" not in CA.detector_objects(g, visible_tower_rule=False)            # the old definition is still available


def test_a_two_high_stub_is_not_a_tower_but_a_three_high_pillar_is():
    """Mutation audit (C1): the height threshold of 3 was not pinned by any test."""
    g = floor()
    column_stack(g, 6, 2)
    assert CA.visible_towers(g) == []
    h = floor()
    column_stack(h, 6, 3)
    assert CA.visible_towers(h) == [6]


def test_wilson_interval_contains_its_own_estimate_for_every_k_and_n():
    """Mutation audit (C5): the lower clamp was not pinned.  Exhaustive, not sampled: every (k, n) with n <= 150."""
    for n in range(1, 151):
        for k in range(n + 1):
            lo, hi = CA.wilson(k, n)
            assert 0.0 <= lo <= k / n <= hi <= 1.0, (k, n, lo, hi)
