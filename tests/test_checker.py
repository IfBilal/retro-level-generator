"""Playability checker tests (doc 21 §10): micro-levels, physics contract, determinism, timeout, jar bridge format."""
import json

import numpy as np
import pytest

from text2mario.data.tiles import CHAR_TO_ID
from text2mario.playability.checker import Framing, check_level, frame_window
from text2mario.playability.physics import PHYSICS_PATH, Physics
from text2mario.render.renderer import grid_to_vglc_text, parse_vglc


def level(floor: str, **rows) -> np.ndarray:
    """16 x len(floor) grid; the bottom row (15) is `floor`; extra rows are given as r14="...." etc."""
    w = len(floor)
    r = ["-" * w] * 16
    r[15] = floor
    for k, v in rows.items():
        r[int(k[1:])] = v.ljust(w, "-")
    return np.array([[CHAR_TO_ID[c] for c in line] for line in r], np.uint8)


def solved(g, **kw) -> bool:
    return check_level(g, **kw).status == "solved"


def test_flat_floor_is_solved():
    assert solved(level("X" * 16))


@pytest.mark.parametrize("gap,expect", [(1, True), (2, True), (3, True), (5, True), (6, True), (7, False), (9, False)])
def test_gap_width_against_the_envelope(gap, expect):
    """Run-jump envelope at the same height is read from physics.json (6 with the frozen physics)."""
    ph = Physics.load()
    limit = ph.gap_same["run"]
    g = level("X" * 5 + "-" * gap + "X" * 12)
    assert solved(g) == (gap <= limit)
    if gap in (1, 2, 3):
        assert solved(g)


@pytest.mark.parametrize("h", [1, 2, 3, 4, 5, 6])
def test_wall_height_against_max_rise(h):
    ph = Physics.load()
    rows = {f"r{15 - k}": "-" * 8 + "X" for k in range(1, h + 1)}
    assert solved(level("X" * 16, **rows)) == (h <= ph.max_rise)


def test_staircase_is_solved():
    rows = {f"r{15 - k}": "-" * (4 + 2 * k) + "X" * (12 - 2 * k) for k in range(1, 5)}
    assert solved(level("X" * 16, **rows))


def test_pipe_within_jump_height_is_solved():
    g = level("X" * 16, r14="-" * 8 + "[]", r13="-" * 8 + "[]", r12="-" * 8 + "<>")
    assert solved(g)


def test_no_floor_is_unsolved():
    assert not solved(level("-" * 16))


def test_unsolved_reports_furthest_column():
    r = check_level(level("X" * 5 + "-" * 12 + "X" * 4))
    assert r.status == "unsolved" and 0 < r.fail_col < 8


def test_deterministic_and_hash_recorded():
    g = level("X" * 6 + "---" + "X" * 7)
    a, b = check_level(g), check_level(g)
    assert a.path == b.path and a.status == b.status
    assert a.physics_sha256 == Physics.load().sha256 and len(a.physics_sha256) == 64


def test_timeout_is_reported_as_timeout():
    r = check_level(level("X" * 40 + "-" * 12 + "X" * 40), budget_nodes=3)
    assert r.status == "timeout"


def test_framed_window_spawn_and_goal():
    fg, fr = frame_window(level("X" * 16))
    assert fg.shape == (16, 25) and fr.spawn == (1, 14) and fr.flag_col == 23
    assert solved(fg, mode="level", framing=fr)


def test_enemy_modes():
    g = level("X" * 16, r14="-" * 7 + "EEEE")
    assert solved(g, enemies="ignore")                      # enemies can be killed or avoided
    # the conservative model must never be MORE permissive than the permissive one
    assert not (solved(g, enemies="hazard") and not solved(g, enemies="ignore"))


def test_physics_single_source_and_derived_not_stale():
    """physics.json's derived block equals what derive_physics computes from its constants (doc 27 §4.4)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("derive_physics", PHYSICS_PATH.parents[3] / "scripts" / "derive_physics.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    p = json.loads(PHYSICS_PATH.read_text())
    d = mod.derive(p)
    assert {k: p["derived"][k] for k in d} == d
    assert p["max_fall_speed"] == p["jump_velocity"], "Arcade clamps symmetrically (doc 27 §4.2)"


def test_jar_bridge_format_roundtrip():
    g = level("X" * 16, r14="--E--?--")
    assert np.array_equal(parse_vglc(grid_to_vglc_text(g)), g)
    assert len(grid_to_vglc_text(g).splitlines()) == 16
