"""Regression tests for the prompt-adherence fixes (audit P2): exact counts parsed, bare negation kept, contradictions and impossible
requests refused, per-window planning, and the raw-level verifier.  CPU only, no model."""
import numpy as np
import pytest

from text2mario.app.service import validate_prompt
from text2mario.data import tiles as T
from text2mario.generate import adherence as A
from text2mario.generate.full_level import STRIDE, WINDOW, _assemble, n_windows, scene_for_windows
from text2mario.text.normalizer import normalize


def test_numerals_are_exact_and_vague_words_keep_their_range():
    w = lambda t: normalize(t).scenes[0].want  # noqa: E731
    assert w("one pipe") == {"pipe": [1, 1]}
    assert w("three platforms") == {"platform": [3, 3]}            # the caption can only say "a few" (3-4); the request stays exactly 3
    assert w("several coins") == {"coin": [5, 9]}
    assert w("a few enemies") == {"enemy": [3, 4]}
    assert w("just 2 pipes") == {"pipe": [2, 2]}


def test_bare_negation_survives_as_a_scene_with_a_forbid():
    """'no pipes' used to produce NO scene (EMPTY_SCENE) -> an unconditioned level, free to contain pipes."""
    n = normalize("no pipes")
    assert len(n.scenes) == 1 and n.scenes[0].forbid == ["pipe"]
    n = normalize("a flat level without pipes")
    assert n.scenes[0].forbid == ["pipe"]
    assert A.constraints(n) == [{"scene": 0, "topic": "pipe", "lo": 0, "hi": 0}]


def test_contradiction_is_refused_with_a_clear_message():
    n = normalize("one pipe and no pipes")
    assert any(w.startswith("CONTRADICTION") for w in n.warnings)
    with pytest.raises(ValueError, match="rules out the same thing"):
        A.check_feasible(n, 48)


def test_impossible_count_is_refused_not_clamped_silently():
    with pytest.raises(ValueError, match="will not fit"):
        A.check_feasible(normalize("500 pipes"), 48)
    A.check_feasible(normalize("three pipes"), 48)                 # fits
    with pytest.raises(ValueError):
        A.check_feasible(normalize("five pipes"), 16)              # 16 columns hold about two


def test_negative_numbers_are_rejected_by_validation_but_ranges_are_not():
    ok, msg = validate_prompt("-3 pipes")
    assert not ok and "Negative" in msg
    assert validate_prompt("5-6 coins")[0] and validate_prompt("two pipes, then 3 coins")[0]


def _pipe_window(cols):
    g = np.zeros((16, 16), np.uint8)
    g[15, :] = T.GROUND
    for c in cols:
        g[11, c], g[11, c + 1] = T.PIPE_TL, T.PIPE_TR
        for r in range(12, 15):
            g[r, c], g[r, c + 1] = T.PIPE_BL, T.PIPE_BR
    return g


def _plan(text, windows_so_far):
    """Caption planned for the NEXT window given already-sampled windows."""
    norm = normalize(text)
    caps = [s.caption for s in norm.scenes]
    n = n_windows(48)
    so = scene_for_windows(caps, n)
    bounds = A.region_bounds(so, 48)
    k = len(windows_so_far)
    ks = [j for j, s in enumerate(so) if s == so[k]]
    before = _assemble(windows_so_far, STRIDE * (k - 1) + WINDOW) if k else None
    return A.plan_caption(norm.scenes[so[k]], caps[so[k]], k, ks[0], ks[-1], before, bounds[so[k]][0])[0]


def test_one_pipe_is_not_requested_in_every_window():
    """Root cause of P2: every window used to get 'one pipe', i.e. one pipe PER 16 columns."""
    empty = np.zeros((16, 16), np.uint8)
    empty[15, :] = T.GROUND
    asked = [_plan("one pipe", [empty] * k) for k in range(5)]
    assert sum("pipe" in c for c in asked) < 5
    assert asked[0] == "full floor."                               # nothing planned yet, neutral caption
    done = [empty, empty, _pipe_window([10])]                      # a pipe exists in window 2 (content columns 26-27)
    assert "pipe" not in _plan("one pipe", done)                   # window 3: the pipe is placed; asking for 'one pipe' again would add a SECOND one


def test_objects_already_in_the_known_half_are_not_repeated_in_the_caption():
    """Measured on the frozen model: caption 'one pipe' with a pipe already in the known half -> another pipe in 19/24 samples;
    a caption without it -> none (24/24).  So the caption states only what the window must ADD."""
    w0 = _pipe_window([10])                                        # pipe in window 0's right half = window 1's known half
    assert "pipe" not in _plan("one pipe", [w0])


def test_window_ok_checks_only_the_new_columns():
    norm = normalize("one pipe")
    sc = norm.scenes[0]
    w = _pipe_window([10])                                         # pipe at window column 10 = NEW half of window k >= 1
    assert A.window_ok(w, 1, sc, {"pipe": 1}) and not A.window_ok(w, 1, sc, {"pipe": 0})
    known_pipe = _pipe_window([2])                                 # pipe in the KNOWN half (columns 0-7) of window 1: not this window's doing
    assert A.window_ok(known_pipe, 1, sc, {"pipe": 0}) and not A.window_ok(known_pipe, 0, sc, {"pipe": 0})   # window 0 owns all 16 columns
    assert not A.window_ok(w, 1, normalize("no pipes").scenes[0], {})                                     # a forbidden topic must not appear


def test_evaluate_counts_raw_content_per_scene_region():
    norm = normalize("one pipe, then two pipes")
    so = scene_for_windows([s.caption for s in norm.scenes], 5)
    content = np.zeros((16, 48), np.uint8)
    content[15, :] = T.GROUND
    for c in (5, 36, 42):                                          # one pipe in scene 1 (cols 0-31), two in scene 2 (cols 32-47)
        content[11, c], content[11, c + 1] = T.PIPE_TL, T.PIPE_TR
        content[12:15, c], content[12:15, c + 1] = T.PIPE_BL, T.PIPE_BR
    r = A.evaluate(content, norm, so, 48)
    assert r["status"] == "met", r
    content[11:15, 36:38] = 0                                      # remove a pipe: scene 2 now has only one
    r = A.evaluate(content, norm, so, 48)
    assert r["status"] == "unmet" and r["unmet"][0]["scene"] == 1 and r["unmet"][0]["got"] == 1


def test_window_to_resample_points_at_the_surplus_and_at_the_gap():
    row = {"scene": 0, "topic": "pipe", "lo": 1, "hi": 1, "got": 3, "anchors": [5, 20, 40]}
    assert A.window_to_resample(row, [0] * 5, 5) == (20 - 8) // 8        # the second pipe (first surplus) sits in window 1
    row = {"scene": 0, "topic": "pipe", "lo": 2, "hi": 2, "got": 1, "anchors": [5]}
    assert A.window_to_resample(row, [0] * 5, 5) == 1                    # window 0 owns columns 0-15 and holds the pipe; window 1 is the first empty one
