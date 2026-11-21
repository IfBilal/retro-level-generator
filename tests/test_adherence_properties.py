"""Property-based and decision-coverage tests for the adherence layer (doc 24 §4): planner completeness, region partition, counter agreement,
normalizer totality and numeral exactness, input safety.  `pytest tests/test_adherence_properties.py --hypothesis-seed=0`."""
import numpy as np
import pytest
from conftest import ROOT
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from text2mario.app.service import validate_prompt
from text2mario.data import tiles as T
from text2mario.eval import count_audit as CA
from text2mario.generate import adherence as A
from text2mario.generate import full_level as FL
from text2mario.text.normalizer import BUCKET_RANGE, normalize

FAST = settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture])
SLOWER = settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture])


def flat():
    g = np.zeros((16, 16), np.uint8)
    g[15, :] = T.GROUND
    return g


def add_pipe(g, c):
    g[11, c], g[11, c + 1] = T.PIPE_TL, T.PIPE_TR
    for r in range(12, 15):
        g[r, c], g[r, c + 1] = T.PIPE_BL, T.PIPE_BR


# ----------------------------------------------------------------------------------------------- partition / bounds
@given(st.sampled_from([16, 48, 80, 112]), st.integers(1, 8))
@FAST
def test_scene_regions_partition_the_level(width, n_scenes):
    n = FL.n_windows(width)
    so = FL.scene_for_windows([f"s{i}" for i in range(n_scenes)], n)
    b = A.region_bounds(so, width)
    assert b[0][0] == 0 and b[-1][1] == width
    assert all(b[i][1] == b[i + 1][0] for i in range(len(b) - 1))                 # contiguous, no gap, no overlap
    assert all(lo < hi for lo, hi in b)
    assert len(b) == len(set(so))


@given(st.sampled_from([16, 48, 80, 112]), st.integers(1, 6))
@FAST
def test_audit_and_generator_agree_on_who_owns_a_column(width, n_scenes):
    """The audit's scene_starts (independent code) must equal the generator's region starts."""
    n = FL.n_windows(width)
    so = FL.scene_for_windows([f"s{i}" for i in range(n_scenes)], n)
    prompt_scenes = [{"caption": f"c{s}", "scene": s} for s in so]
    assert CA.scene_starts(prompt_scenes, FL.STRIDE) == [lo for lo, _ in A.region_bounds(so, width)]
    legacy = [{"caption": f"c{s}"} for s in so]                                    # older levels without the scene index
    assert CA.scene_starts(legacy, FL.STRIDE) == CA.scene_starts(prompt_scenes, FL.STRIDE)


# ----------------------------------------------------------------------------------------------- statistics
@given(st.integers(0, 200), st.integers(1, 200))
@FAST
def test_wilson_interval_properties(k, n):
    k = min(k, n)
    lo, hi = CA.wilson(k, n)
    assert 0.0 <= lo <= k / n <= hi <= 1.0 + 1e-12
    if k < n:
        assert CA.wilson(k + 1, n)[0] >= lo - 1e-12                                # monotone in k


def test_wilson_zero_n():
    assert CA.wilson(0, 0) == (0.0, 0.0)


# ----------------------------------------------------------------------------------------------- normalizer
NOUNS = {"pipe": "pipes", "enemy": "enemies", "coin": "coins", "cannon": "cannons", "platform": "platforms", "tower": "towers"}
WORDS = "zero one two three four five six seven eight nine".split()


@given(st.integers(1, 9), st.sampled_from(sorted(NOUNS)), st.booleans())
@FAST
def test_numerals_give_exact_wants(n, noun, as_digit):
    word = str(n) if as_digit else WORDS[n]
    sc = normalize(f"{word} {NOUNS[noun] if n != 1 else noun}").scenes
    assert len(sc) == 1 and sc[0].want[noun] == [n, n]
    assert not sc[0].forbid


@given(st.sampled_from(sorted(BUCKET_RANGE)), st.sampled_from(sorted(NOUNS)))
@FAST
def test_vague_words_keep_their_bucket_range(bucket, noun):
    word = {"one": "one", "two": "two", "a few": "a few", "several": "several", "many": "many"}[bucket]
    sc = normalize(f"{word} {NOUNS[noun] if word != 'one' else noun}").scenes[0]
    lo, hi = sc.want[noun]
    assert (lo, hi) == BUCKET_RANGE[bucket]


@given(st.text(max_size=300))
@settings(max_examples=150, deadline=None)
def test_normalize_is_total_and_wants_are_sane(text):
    n = normalize(text)
    for sc in n.scenes:
        assert sc.caption
        for t, (lo, hi) in sc.want.items():
            assert 0 <= lo <= hi
        assert not (set(sc.want) & set(sc.forbid)) or any(w.startswith("CONTRADICTION") for w in n.warnings + sc.warnings)


@given(st.text(max_size=500))
@settings(max_examples=150, deadline=None)
def test_validate_prompt_never_raises(text):
    ok, msg = validate_prompt(text)
    assert isinstance(ok, bool) and isinstance(msg, str) and (ok or msg)


@given(st.sampled_from(sorted(NOUNS)))
@FAST
def test_negation_forms_all_create_a_forbid(noun):
    for form in (f"no {NOUNS[noun]}", f"without {NOUNS[noun]}", f"zero {NOUNS[noun]}", f"a flat level with no {NOUNS[noun]}"):
        n = normalize(form)
        assert n.scenes and n.scenes[0].forbid == [noun], form
        assert A.constraints(n) == [{"scene": 0, "topic": noun, "lo": 0, "hi": 0}]


def test_scene_counts_follow_the_sentence_order():
    n = normalize("one pipe, then two enemies, then a cannon")
    assert [list(s.want) for s in n.scenes] == [["pipe"], ["enemy"], ["cannon"]]
    assert [c["scene"] for c in A.constraints(n)] == [0, 1, 2]


# ----------------------------------------------------------------------------------------------- counters
@given(st.lists(st.integers(0, 6), min_size=0, max_size=6, unique=True))
@FAST
def test_detector_and_independent_counter_agree_on_random_pipes(slots):
    """Pipes on slots 4 columns apart (never touching): both counters must say len(slots)."""
    g = np.zeros((16, 48), np.uint8)
    g[15, :] = T.GROUND
    for s in slots:
        add_pipe(g, 2 + 6 * s)
    assert len(CA.objects_by_topic(g, {"pipe": {"det": ["pipe", "upside down pipe", "broken pipe"]}})["pipe"]) == len(slots)
    assert CA.independent_counts(g)["pipe"] == len(slots)


@given(st.lists(st.integers(0, 15), max_size=10, unique=True))
@FAST
def test_enemy_and_coin_counts_equal_cells(cols):
    g = np.zeros((16, 48), np.uint8)
    g[15, :] = T.GROUND
    for c in cols:
        g[14, 3 * c] = T.ENEMY_ID
        g[3, 3 * c + 1 if 3 * c + 1 < 48 else 0] = T.COIN
    ind = CA.independent_counts(g)
    got = {k: len(v) for k, v in CA.objects_by_topic(g, {"enemy": {"det": ["enemy"]}, "coin": {"det": ["coin"]}}).items()}
    assert got["enemy"] == ind["enemy"] == len(cols) and got["coin"] == ind["coin"]


@given(st.dictionaries(st.sampled_from(["pipe", "enemy", "coin"]), st.lists(st.integers(0, 47), max_size=8), max_size=3))
@FAST
def test_check_expectation_is_consistent_with_its_checks(anchors):
    expect = [{"pipe": [0, 3]}, {"enemy": [1, 2]}]
    ok, d = CA.check_expectation(anchors, expect, [0, 24], 48)
    assert ok == all(c["ok"] for c in d["checks"])
    for c in d["checks"]:
        assert c["ok"] == (c["want"][0] <= c["got"] <= c["want"][1])


# ----------------------------------------------------------------------------------------------- the planner end to end (stub sampler)
def _stub(monkeypatch, per_window_exact=True):
    """Stub sampler: draws exactly the pipes (and enemies) its caption asks for in the NEW columns; copies the known columns."""
    def fake(bundle, caption, w, steps, r, gen_seed, known_cols, device):
        g = flat()
        if known_cols is not None:
            g[:, :known_cols.shape[1]] = known_cols
        n = 2 if "two pipes" in caption else 1 if "one pipe" in caption else 0
        base = 9 if known_cols is not None else 2
        for i in range(n):
            add_pipe(g, base + 4 * i)
        e = 1 if "one enemy" in caption else 0
        if e:
            g[14, 14 if known_cols is not None else 13] = T.ENEMY_ID
        return g

    monkeypatch.setattr(FL, "sample_window", fake)


@given(st.integers(1, 4), st.sampled_from([48, 80, 112]))
@SLOWER
def test_planner_reaches_any_feasible_pipe_target(monkeypatch, target, width):
    """Completeness: with a sampler that does what the caption says, the planned level has EXACTLY the requested pipes."""
    _stub(monkeypatch)
    words = ["zero", "one", "two", "three", "four"]
    lvl = FL.generate(None, f"{words[target]} pipes" if target > 1 else "one pipe", seed=7, width=width, device="cpu")
    adh = lvl["verification"]["adherence"]
    assert adh["status"] == "met" and adh["constraints"][0]["got"] == target
    assert adh["judged_on"] == "raw"


@given(st.integers(1, 3), st.integers(1, 3))
@SLOWER
def test_two_scenes_both_met(monkeypatch, a, b):
    _stub(monkeypatch)
    words = ["", "one pipe", "two pipes", "three pipes"]
    lvl = FL.generate(None, f"{words[a]}, then {words[b]}", seed=3, width=112, device="cpu")
    got = [c["got"] for c in lvl["verification"]["adherence"]["constraints"]]
    assert got == [a, b] and lvl["verification"]["adherence"]["status"] == "met"


@given(st.sampled_from([48, 80]))
@SLOWER
def test_planner_is_deterministic(monkeypatch, width):
    _stub(monkeypatch)
    a = FL.generate(None, "two pipes and one enemy", seed=5, width=width, device="cpu")
    b = FL.generate(None, "two pipes and one enemy", seed=5, width=width, device="cpu")
    assert a["raw_grid"] == b["raw_grid"] and a["prompt"]["scenes"] == b["prompt"]["scenes"]


# ----------------------------------------------------------------------------------------------- decision coverage: every config branch
def test_adherence_disabled_restores_the_old_behaviour(monkeypatch):
    _stub(monkeypatch)
    monkeypatch.setitem(A._A, "enabled", False)
    lvl = FL.generate(None, "one pipe", seed=1, width=48, device="cpu", verify=False)
    assert lvl["verification"]["adherence"]["status"] == "not_applicable"
    assert all(s["caption"] == "one pipe." for s in lvl["prompt"]["scenes"])           # every window got the scene caption


def test_plan_off_keeps_scene_captions_but_still_verifies(monkeypatch):
    _stub(monkeypatch)
    monkeypatch.setitem(A._A, "plan", False)
    lvl = FL.generate(None, "one pipe", seed=1, width=48, device="cpu")
    assert all(s["caption"] == "one pipe." for s in lvl["prompt"]["scenes"])
    assert lvl["verification"]["adherence"]["planned"] is False


def test_verify_off_skips_local_redraws_and_resampling(monkeypatch):
    calls = []
    _stub(monkeypatch)
    real = FL.sample_window
    monkeypatch.setattr(FL, "sample_window", lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    monkeypatch.setitem(A._A, "verify", False)
    FL.generate(None, "one pipe", seed=1, width=48, device="cpu")
    assert len(calls) == 5                                                              # one draw per window, no redraws


def test_verification_skipped_means_no_redraws_either(monkeypatch):
    calls = []
    _stub(monkeypatch)
    real = FL.sample_window
    monkeypatch.setattr(FL, "sample_window", lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    lvl = FL.generate(None, "one pipe", seed=1, width=48, device="cpu", verify=False)
    assert len(calls) == 5 and lvl["verification"]["status"] == "not_verified"


@pytest.mark.parametrize("value", ["", "full floor"])
def test_default_floor_toggle(monkeypatch, value):
    monkeypatch.setitem(A._A, "default_floor", value)
    n = normalize("one pipe")
    cap, plan = A.plan_caption(n.scenes[0], n.scenes[0].caption, 2, 0, 4, None, 0)
    assert ("full floor" in cap) == bool(value) and plan == {"pipe": 1}


def test_scene_with_its_own_floor_phrase_is_not_given_a_second_one():
    n = normalize("floor with two gaps and one pipe")
    cap, _ = A.plan_caption(n.scenes[0], n.scenes[0].caption, 2, 0, 4, None, 0)
    assert cap.count("floor") == 1 and "gap" in cap


def test_range_target_low_vs_mid(monkeypatch):
    assert A.target_of(5, 9) == 7 and A.target_of(3, 4) == 3 and A.target_of(2, 2) == 2 and A.target_of(10, 99) == 10
    monkeypatch.setitem(A._A, "range_target", "low")
    assert A.target_of(5, 9) == 5


def test_capacity_scales_with_width():
    A.check_feasible(normalize("five pipes"), 112)
    with pytest.raises(ValueError):
        A.check_feasible(normalize("five pipes"), 16)
    with pytest.raises(ValueError, match="enemy"):
        A.check_feasible(normalize("thirty enemies"), 48)


def test_cap_never_below_what_the_scene_needs_per_window(monkeypatch):
    """Eight coins in a 16-column level (one window): the per-window cap of 1 must not make the plan impossible."""
    n = normalize("eight coins")
    cap, plan = A.plan_caption(n.scenes[0], n.scenes[0].caption, 0, 0, 0, None, 0)
    assert plan["coin"] == 8


def test_time_budget_stops_local_redraws(monkeypatch):
    draws = []

    def never_right(bundle, caption, w, steps, r, gen_seed, known_cols, device):
        draws.append(1)
        return flat()                                                                   # never has the pipe the plan wants

    monkeypatch.setattr(FL, "sample_window", never_right)
    lvl = FL.generate(None, "one pipe", seed=1, width=16, device="cpu", time_budget_s=1e-9)
    assert len(draws) <= 3, "the time guard must cut the redraw loops (cloud profile)"
    assert lvl["verification"]["reason"] in ("time budget", "adherence budget exhausted") or lvl["verification"]["adherence"]["status"] == "unmet"


def test_unmet_after_budget_is_reported_not_a_success(monkeypatch):
    monkeypatch.setattr(FL, "sample_window", lambda *a, **k: flat())
    lvl = FL.generate(None, "two pipes", seed=1, width=16, device="cpu")
    adh = lvl["verification"]["adherence"]
    assert adh["status"] == "unmet" and adh["attempts"] <= int(A._A["max_attempts"])
    assert lvl["verification"]["reason"] == "adherence budget exhausted"
    assert [c for c in adh["constraints"] if not c["met"]]


def test_surplus_triggers_resample_of_the_owning_window(monkeypatch):
    seen = []

    def sometimes_extra(bundle, caption, w, steps, r, gen_seed, known_cols, device):
        g = flat()
        if known_cols is not None:
            g[:, :known_cols.shape[1]] = known_cols
        seen.append(caption)
        if "one pipe" in caption:
            add_pipe(g, 9 if known_cols is not None else 2)
            if len(seen) < 6:                                                           # first draws carry a surplus pipe, later ones do not
                add_pipe(g, 13 if known_cols is not None else 8)
        return g

    monkeypatch.setattr(FL, "sample_window", sometimes_extra)
    lvl = FL.generate(None, "one pipe", seed=2, width=48, device="cpu")
    assert lvl["verification"]["adherence"]["status"] == "met"
    assert lvl["verification"]["adherence"]["window_redraws"] >= 1


def test_forbidden_topic_in_a_later_scene_only_applies_to_that_scene(monkeypatch):
    def pipes_early(bundle, caption, w, steps, r, gen_seed, known_cols, device):
        g = flat()
        if known_cols is not None:
            g[:, :known_cols.shape[1]] = known_cols
        elif "pipe" in caption or True:
            add_pipe(g, 2)                                                              # a pipe in window 0 only
        return g

    monkeypatch.setattr(FL, "sample_window", pipes_early)
    n = normalize("one pipe, then no pipes")
    assert [s.forbid for s in n.scenes] == [[], ["pipe"]]
    lvl = FL.generate(None, "one pipe, then no pipes", seed=1, width=48, device="cpu")
    assert lvl["verification"]["adherence"]["status"] == "met"


# ----------------------------------------------------------------------------------------------- input safety (property)
@given(st.text(max_size=120))
@settings(max_examples=60, deadline=None)
def test_any_prompt_text_cannot_break_the_game_page(text):
    pytest.importorskip("PIL")
    if not (ROOT / "app/static/vendor/phaser.min.js").exists():
        pytest.skip("run scripts/fetch_assets.py first")
    from test_level_schema import good_level

    from text2mario.render.game_html import build_game_html
    clean = build_game_html(good_level())
    lvl = good_level()
    lvl["prompt"]["text"] = text + "</script><!--<script>"
    html = build_game_html(lvl)
    assert html.count("</script>") == clean.count("</script>")
    part = html.split("window.__LEVEL__=")[1].split(";window.__PHYSICS__")[0]
    assert "</" not in part and "<!--" not in part


# ----------------------------------------------------------------------------------------------- remaining branches
def test_forbidding_a_non_counted_topic_creates_no_count_constraint():
    n = normalize("no ceiling and two pipes")
    assert "ceiling" in n.scenes[0].forbid
    assert A.constraints(n) == [{"scene": 0, "topic": "pipe", "lo": 2, "hi": 2}]               # 'ceiling' is not a counted topic


def test_shortfall_with_every_window_already_holding_one_resamples_the_last_window():
    row = {"scene": 0, "topic": "pipe", "lo": 6, "hi": 6, "got": 5, "anchors": [4, 20, 28, 36, 44]}
    assert A.window_to_resample(row, [0] * 5, 5) == 4


def test_legacy_forbidden_word_path_when_adherence_is_off(monkeypatch):
    """adherence.enabled = false keeps the OLD caption-text forbid check (CLAUDE.md rule 12: the fallback is explicit and tested)."""
    def pipe_everywhere(bundle, caption, w, steps, r, gen_seed, known_cols, device):
        g = flat()
        if known_cols is not None:
            g[:, :known_cols.shape[1]] = known_cols
        elif True:
            add_pipe(g, 2)
        return g

    monkeypatch.setattr(FL, "sample_window", pipe_everywhere)
    monkeypatch.setitem(A._A, "enabled", False)
    lvl = FL.generate(None, "full floor with no pipes", seed=1, width=16, device="cpu")
    log = lvl["verification"]["attempt_log"]
    assert log and all(a["status"] == "forbidden" for a in log) and lvl["verification"]["status"] == "not_verified"
    assert lvl["verification"]["adherence"]["status"] == "not_applicable"
