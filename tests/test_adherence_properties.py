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

pytestmark = pytest.mark.usefixtures("batch_via_single")

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
WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]


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
        for lo, hi in sc.want.values():
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
        assert A.constraints(n) == [{"scene": 0, "topic": noun, "lo": 0, "hi": 0, "implicit": False}]


def test_scene_counts_follow_the_sentence_order():
    n = normalize("one pipe, then two enemies, then a cannon")
    assert [list(s.want) for s in n.scenes] == [["pipe"], ["enemy"], ["cannon"]]
    explicit = [c for c in A.constraints(n) if not c["implicit"]]
    assert [c["scene"] for c in explicit] == [0, 1, 2]


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
    _cap, plan = A.plan_caption(n.scenes[0], n.scenes[0].caption, 0, 0, 0, None, 0)
    assert plan["coin"] == 8


def test_time_budget_stops_local_redraws(monkeypatch):
    draws = []

    def never_right(bundle, caption, w, steps, r, gen_seed, known_cols, device):
        draws.append(1)
        return flat()                                                                   # never has the pipe the plan wants

    monkeypatch.setattr(FL, "sample_window", never_right)
    lvl = FL.generate(None, "one pipe", seed=1, width=16, device="cpu", time_budget_s=-1.0)      # already expired
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
        else:
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
    assert A.constraints(n) == [{"scene": 0, "topic": "pipe", "lo": 2, "hi": 2, "implicit": False}]               # 'ceiling' is not a counted topic


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


@pytest.mark.parametrize("width", [16, 48, 112])
def test_unsatisfiable_request_has_a_hard_bounded_cost(monkeypatch, width):
    """Audit latency defect: a seed that cannot hit 3 enemies took 177 s (16 whole-level resamples x 8 redraws per window).
    Extra window draws are now capped by adherence.max_extra_draws, so the worst case is deterministic and small."""
    singles, batches = [], []

    def never_right(bundle, caption, w, steps, r, gen_seed, known_cols, device):
        singles.append(1)
        return flat()

    def batch_never_right(bundle, caption, w, steps, r, gen_seed, known_cols, device, n):
        batches.append(n)
        return np.stack([flat() for _ in range(n)])

    monkeypatch.setattr(FL, "sample_window", never_right)
    monkeypatch.setattr(FL, "sample_window_batch", batch_never_right)
    lvl = FL.generate(None, "three enemies", seed=1, width=width, device="cpu")
    n, cap = FL.n_windows(width), int(A._A["max_extra_draws"])
    assert len(singles) + len(batches) <= 2 * (n + cap + n)                   # a draw (+ at most one small batch) per window, capped extras
    adh = lvl["verification"]["adherence"]
    assert adh["status"] == "unmet" and adh["draw_cap"] == cap and lvl["verification"]["reason"] == "adherence budget exhausted"


# ----------------------------------------------------------------------------------------------- batched candidates (CUDA path, stubbed)
def _batch_stub(monkeypatch, pipes_per_candidate):
    """sample_window_batch stub: candidate i gets pipes_per_candidate[i] pipes in the new columns; sample_window must NOT be used."""
    seen = []

    def fake_batch(bundle, caption, w, steps, r, gen_seed, known_cols, device, n):
        seen.append(n)
        out = []
        for i in range(n):
            g = flat()
            if known_cols is not None:
                g[:, :known_cols.shape[1]] = known_cols
            for p in range(pipes_per_candidate[i % len(pipes_per_candidate)]):
                add_pipe(g, 2 + 5 * p if known_cols is None else 9 + 4 * p)
            out.append(g)
        return np.stack(out)

    def no_single(*a, **k):
        raise AssertionError("the sequential sampler must not run on the batch path")

    monkeypatch.setattr(FL, "sample_window_batch", fake_batch)
    monkeypatch.setattr(FL, "sample_window", no_single)
    return seen


def test_batch_path_keeps_the_exact_candidate(monkeypatch):
    seen = _batch_stub(monkeypatch, [0, 2, 3, 0, 0, 1, 2, 0])                       # only candidate 5 (and 13) has exactly one pipe
    lvl = FL.generate(None, "one pipe", seed=1, width=16, device="cuda")
    assert lvl["verification"]["adherence"]["status"] == "met" and _count_pipes(lvl) == 1
    assert seen == [int(A._A["batch_tries"])]                                       # ONE forward pass for the whole window


def test_batch_path_keeps_the_closest_when_none_is_exact(monkeypatch):
    seen = _batch_stub(monkeypatch, [3, 0, 3, 3])                                   # want 1: the 0-pipe candidate misses by 1, the 3-pipe ones by 2
    lvl = FL.generate(None, "one pipe", seed=1, width=16, device="cuda", verify=True)
    assert lvl["verification"]["adherence"]["constraints"][0]["got"] in (0, 3) and seen
    assert lvl["verification"]["adherence"]["window_redraws"] >= 1


def test_batch_path_is_skipped_for_scenes_without_counts_and_when_out_of_time(monkeypatch):
    seen = _batch_stub(monkeypatch, [1])
    monkeypatch.setattr(FL, "sample_window", lambda *a, **k: flat())
    FL.generate(None, "full floor", seed=1, width=16, device="cuda")
    assert seen == []                                                               # a floor-only scene has nothing to count
    FL.generate(None, "one pipe", seed=1, width=16, device="cuda", time_budget_s=-1.0)      # already expired (a 1e-9 budget can tie with the clock tick)
    assert seen == []                                                               # time guard: no batch is started after the budget


def test_cpu_draws_one_first_and_a_batch_only_on_a_miss(monkeypatch):
    seen = _batch_stub(monkeypatch, [1])
    monkeypatch.setattr(FL, "sample_window", lambda *a, **k: flat())            # the single draw has no pipe -> a miss -> one small batch
    lvl = FL.generate(None, "one pipe", seed=1, width=16, device="cpu")
    assert seen == [int(A._A["batch_tries_cpu"])] and lvl["verification"]["adherence"]["status"] == "met"
    seen.clear()

    def exact(*a, **k):                                                            # the single draw is already right -> no batch at all
        g = flat()
        add_pipe(g, 2)
        return g

    monkeypatch.setattr(FL, "sample_window", exact)
    FL.generate(None, "one pipe", seed=1, width=16, device="cpu")
    assert seen == []


def test_cpu_batch_can_be_switched_off_for_sequential_redraws(monkeypatch):
    monkeypatch.setitem(A._A, "batch_tries_cpu", 1)
    seen = _batch_stub(monkeypatch, [1])
    calls = []
    monkeypatch.setattr(FL, "sample_window", lambda *a, **k: (calls.append(1), flat())[1])
    FL.generate(None, "one pipe", seed=1, width=16, device="cpu")
    assert seen == [] and len(calls) > 1                                           # redraws one by one, no batch


def _count_pipes(lvl):
    from text2mario.eval.count_audit import objects_by_topic
    raw = np.array(lvl["raw_grid"], dtype=np.uint8)[:, 3:3 + lvl["generation"]["width"]]
    return len(objects_by_topic(raw, {"pipe": {"det": ["pipe", "upside down pipe", "broken pipe"]}})["pipe"])


@given(st.sampled_from([16, 48, 80, 112]))
@FAST
def test_more_scenes_than_windows_is_refused_not_silently_dropped(width):
    """Audit self-review: scene_for_windows([a, b, c], 1) maps every window to scene 0, so scenes 1 and 2 were never generated but still judged."""
    windows = FL.n_windows(width)
    text = ", then ".join(["one pipe"] * (windows + 1)) if windows + 1 <= 8 else None
    if text is None:
        return
    with pytest.raises(ValueError, match="has room for"):
        A.check_feasible(normalize(text), width)
    A.check_feasible(normalize(", then ".join(["one pipe"] * windows)), width)               # exactly as many scenes as windows is fine


# ----------------------------------------------------------------------------------------------- exclusive scenes ("then") and plan-guided resampling
def test_then_means_what_is_asked_elsewhere_is_ruled_out_here():
    n = normalize("one pipe, then two enemies, then a cannon")
    f = A.scene_forbids(n)
    assert f == [{"enemy", "cannon"}, {"pipe", "cannon"}, {"pipe", "enemy"}]
    rows = A.constraints(n)
    assert sum(1 for c in rows if c["implicit"]) == 6 and all(c["lo"] == c["hi"] == 0 for c in rows if c["implicit"])


def test_exclusivity_does_not_forbid_a_topic_the_scene_asks_for_and_is_off_for_one_scene(monkeypatch):
    assert A.scene_forbids(normalize("one pipe, then two pipes")) == [set(), set()]            # same topic in both scenes: nothing to exclude
    assert A.scene_forbids(normalize("two pipes and one enemy")) == [set()]                    # a single scene: nothing to exclude
    monkeypatch.setitem(A._A, "exclusive_scenes", False)
    assert A.scene_forbids(normalize("one pipe, then two enemies")) == [set(), set()]


def test_explicit_prohibition_is_not_marked_implicit():
    n = normalize("one pipe, then no enemies")
    rows = {(c["scene"], c["topic"]): c["implicit"] for c in A.constraints(n)}
    assert rows[(1, "enemy")] is False and rows[(1, "pipe")] is True and (0, "enemy") not in rows       # ruling something out elsewhere asks for nothing here


def test_a_stray_enemy_before_the_pipe_is_reported_unmet():
    """Audit ordering failure: 'one pipe, then two enemies, then a cannon' with an unrequested enemy in the pipe's scene."""
    norm = normalize("one pipe, then two enemies, then a cannon")
    so = FL.scene_for_windows([s.caption for s in norm.scenes], 5)
    g = np.zeros((16, 48), np.uint8)
    g[15, :] = T.GROUND
    add_pipe(g, 10)
    g[14, 4] = T.ENEMY_ID                                                                    # stray enemy BEFORE the pipe
    g[14, 24] = g[14, 28] = T.ENEMY_ID
    g[14, 43] = T.CANNON_TOP
    r = A.evaluate(g, norm, so, 48)
    assert r["status"] == "unmet"
    assert [(u["topic"], u["scene"], u["implicit"]) for u in r["unmet"]] == [("enemy", 0, True)]
    g[14, 4] = T.SKY
    assert A.evaluate(g, norm, so, 48)["status"] == "met"


def test_shortfall_is_redrawn_where_the_plan_put_the_object():
    row = {"scene": 0, "topic": "tower", "lo": 2, "hi": 2, "got": 1, "anchors": [17]}
    plans = [{"tower": 0}, {"tower": 1}, {"tower": 0}, {"tower": 1}, {"tower": 0}]
    assert A.window_to_resample(row, [0] * 5, 5, plans) == 3                                 # window 1 holds its tower (anchor 17); window 3 does not
    assert A.window_to_resample(row, [0] * 5, 5) == 0                                        # without the plan the old guess is the first empty window


# ----------------------------------------------------------------------------------------------- route budget spent -> counts are still fixed
class _FakeCheck:
    def __init__(self, status="unsolved", fail_col=10):
        self.status, self.fail_col, self.seconds, self.path = status, fail_col, 0.0, []


def test_counts_keep_being_fixed_after_the_route_budget_is_spent(monkeypatch):
    """Audit: every failing 'underground' level had stopped at 'attempt budget exhausted' (route) with a count still unmet.
    The route budget now ends only the ROUTE search; the counts go on until their own budget."""
    checks = {"n": 0}

    def no_route(*a, **k):
        checks["n"] += 1
        return _FakeCheck("unsolved", 10)                                                       # a level with no route, ever

    monkeypatch.setattr(FL, "check_level", no_route)

    def late_pipe(bundle, caption, w, steps, r, gen_seed, known_cols, device):
        g = flat()
        if known_cols is not None:
            g[:, :known_cols.shape[1]] = known_cols
        if "one pipe" in caption and checks["n"] >= 9:                                           # the pipe only appears once the route budget is spent
            add_pipe(g, 9 if known_cols is not None else 2)
        return g

    monkeypatch.setattr(FL, "sample_window", late_pipe)
    lvl = FL.generate(None, "one pipe", seed=1, width=48, device="cpu")
    v = lvl["verification"]
    log = v["attempt_log"]
    route = [i for i, a in enumerate(log) if a["status"] != "adherence"]
    adh = [i for i, a in enumerate(log) if a["status"] == "adherence"]
    assert route and adh and min(adh) > max(route[:6])                                           # counts were worked on AFTER the route attempts
    assert v["status"] == "not_verified" and v["reason"] == "attempt budget exhausted" and v["attempts_total"] <= 20
    assert v["adherence"]["status"] == "met"                                                     # and they got fixed although the level has no route


def test_attempts_total_never_exceeds_the_ledger_cap_when_failures_spread_over_windows(monkeypatch):
    """attempts_total counted the attempt that hit the cap (21) and validate_level would reject the level."""
    state = {"n": 0}

    def spread(*a, **k):
        state["n"] += 1
        return _FakeCheck("unsolved", 3 + 8 * (state["n"] % 12))                                  # a different window every time

    monkeypatch.setattr(FL, "check_level", spread)
    monkeypatch.setattr(FL, "sample_window", lambda *a, **k: flat())
    lvl = FL.generate(None, "full floor", seed=1, width=112, device="cpu")                      # validate_level runs inside generate()
    assert lvl["verification"]["attempts_total"] <= 20 and lvl["verification"]["reason"] == "attempt budget exhausted"


# ----------------------------------------------------------------------------------------------- hypothesis (f) of the audit brief: the sanitizer cannot ADD objects
@given(st.integers(0, 2**32 - 1), st.sampled_from([16, 32, 48]))
@settings(max_examples=60, deadline=None)
def test_display_repair_only_removes_cells_never_adds_them(seed, width):
    """Rule 8: repair is for display only.  Property: every cell that differs between raw and repaired is SKY in the repaired grid, so the repair can
    never create or duplicate an object, and it never touches the raw grid it is given."""
    from text2mario.generate.sanitizer import repair
    rng = np.random.default_rng(seed)
    g = rng.integers(0, 13, size=(16, width)).astype(np.uint8)
    g[rng.random((16, width)) < 0.6] = T.SKY                                       # mostly sky, some of every tile (broken pipes / cannons included)
    before = g.copy()
    out, removed = repair(g)
    assert np.array_equal(g, before), "repair must not modify its input"
    changed = out != g
    assert (out[changed] == T.SKY).all() and int(changed.sum()) == removed
    for t in range(13):
        if t != T.SKY:
            assert int((out == t).sum()) <= int((g == t).sum())                    # no tile type ever increases


# ----------------------------------------------------------------------------------------------- last branches
def test_constraints_ignore_a_want_key_that_is_not_a_counted_topic():
    from types import SimpleNamespace
    scene = SimpleNamespace(want={"floor": [1, 1], "pipe": [1, 1]}, forbid=[], caption="x")
    norm = SimpleNamespace(scenes=[scene], warnings=[])
    assert [c["topic"] for c in A.constraints(norm)] == ["pipe"]


def test_shortfall_falls_back_to_the_first_empty_window_when_the_plan_is_already_satisfied():
    row = {"scene": 0, "topic": "pipe", "lo": 1, "hi": 1, "got": 0, "anchors": []}
    assert A.window_to_resample(row, [0] * 5, 5, [{"pipe": 0}] * 5) == 0                   # nothing was planned anywhere: the first empty window


def test_route_search_stops_at_the_time_budget(monkeypatch):
    monkeypatch.setattr(FL, "check_level", lambda *a, **k: _FakeCheck("unsolved", 10))
    monkeypatch.setattr(FL, "sample_window", lambda *a, **k: flat())
    lvl = FL.generate(None, "full floor", seed=1, width=48, device="cpu", time_budget_s=-1.0)          # already expired
    v = lvl["verification"]
    assert v["reason"] == "time budget" and v["status"] == "not_verified" and len(v["attempt_log"]) == 1


def test_asking_for_staircases_of_one_direction_rules_out_the_other_direction_in_that_scene():
    """Fresh-seed audit miss: 'three staircases' got 4 because an unrequested DESCENDING staircase appeared; a person counts it as a staircase."""
    n = normalize("three staircases")
    assert A.scene_forbids(n) == [{"descending staircase"}]
    rows = A.constraints(n)
    assert {"scene": 0, "topic": "descending staircase", "lo": 0, "hi": 0, "implicit": True} in rows
    n = normalize("one ascending staircase and one descending staircase")                    # both asked for: nothing to rule out
    assert A.scene_forbids(n) == [set()]
    n = normalize("stairs going down")
    assert A.scene_forbids(n) == [{"ascending staircase"}]


def test_a_stray_descending_staircase_makes_three_staircases_unmet():
    norm = normalize("three staircases")
    so = FL.scene_for_windows([s.caption for s in norm.scenes], 5)
    g = np.zeros((16, 48), np.uint8)
    g[15, :] = T.GROUND
    for start in (2, 12, 22):                                                                # three ascending staircases (3 steps each)
        for k in range(3):
            g[15 - (k + 1):15, start + k] = T.GROUND
    for k in range(3):                                                                       # one DESCENDING staircase
        g[15 - (3 - k):15, 34 + k] = T.GROUND
    r = A.evaluate(g, norm, so, 48)
    assert r["status"] == "unmet" and [(u["topic"], u["got"]) for u in r["unmet"]] == [("descending staircase", 1)]
    g[:, 34:37] = 0
    g[15, 34:37] = T.GROUND
    assert A.evaluate(g, norm, so, 48)["status"] == "met"


def test_a_topic_that_is_both_requested_and_forbidden_is_never_put_in_the_caption():
    """Mutation audit (A17): normalize() refuses a contradiction, but plan_caption must also be safe on its own (defence in depth)."""
    from types import SimpleNamespace
    scene = SimpleNamespace(want={"pipe": [2, 2], "coin": [3, 3]}, forbid=["pipe"], caption="two pipes. a few coins.")
    cap, plan = A.plan_caption(scene, scene.caption, 2, 0, 4, None, 0)
    assert "pipe" not in cap and "pipe" not in plan and "coin" in cap


# ----------------------------------------------------------------------------------------------- mutation audit: the sequential (CPU, batch off) redraw loop
def test_sequential_redraws_stop_at_the_extra_draw_cap(monkeypatch):
    """Mutation audit (F4): with batches off, the per-window redraw loop must obey adherence.max_extra_draws, not only max_window_tries."""
    monkeypatch.setitem(A._A, "batch_tries_cpu", 1)
    monkeypatch.setitem(A._A, "max_extra_draws", 3)
    monkeypatch.setitem(A._A, "max_window_tries", 50)                      # far above the cap: only the cap can stop the loop
    monkeypatch.setitem(A._A, "max_attempts", 0)                           # no whole-level resampling, isolate the local loop
    calls = []
    monkeypatch.setattr(FL, "sample_window", lambda *a, **k: (calls.append(1), flat())[1])    # never has the pipe
    FL.generate(None, "one pipe", seed=1, width=16, device="cpu")
    assert len(calls) <= 1 + 3 + 1, len(calls)                              # first draw + 3 extra draws (+ one of slack for the last check)


def test_sequential_redraws_keep_the_closest_candidate_not_the_worst(monkeypatch):
    """Mutation audit (F8): none of the draws is exact; the loop must keep the one with the smallest miss (2 pipes for 'one pipe'), not the worst (3)."""
    monkeypatch.setitem(A._A, "batch_tries_cpu", 1)
    monkeypatch.setitem(A._A, "max_attempts", 0)
    pipes = iter([3, 2, 0, 3, 3, 3, 3, 3] * 20)

    def scripted(bundle, caption, w, steps, r, gen_seed, known_cols, device):
        g = flat()
        for i in range(next(pipes)):
            add_pipe(g, 1 + 5 * i)
        return g

    monkeypatch.setattr(FL, "sample_window", scripted)
    lvl = FL.generate(None, "one pipe", seed=1, width=16, device="cpu")
    assert _count_pipes(lvl) == 2                                           # misses were 2, 1, 1, 2...: the first minimum (2 pipes) wins


# ----------------------------------------------------------------------------------------------- mutation audit: scene-specific prohibitions and the draw budget
def test_each_window_is_checked_against_ITS_scene_prohibitions(monkeypatch):
    """Mutation audit (H7): 'no pipes, then one enemy' forbids pipes in scene 1 only.  A sampler that draws a pipe together with the enemy is therefore fine in scene 2:
    with the right prohibition set the first draw of every window is accepted (5 draws); if scene 2 inherited scene 1's prohibition every such window would be redrawn."""
    calls = []

    def fake(bundle, caption, w, steps, r, gen_seed, known_cols, device):
        calls.append(caption)
        g = flat()
        if known_cols is not None:
            g[:, :known_cols.shape[1]] = known_cols
        if "one enemy" in caption:
            g[14, 14] = T.ENEMY_ID
            add_pipe(g, 9)
        return g

    monkeypatch.setattr(FL, "sample_window", fake)
    lvl = FL.generate(None, "no pipes, then one enemy", seed=1, width=48, device="cpu")
    assert len(calls) == 5, calls
    assert lvl["verification"]["adherence"]["status"] == "met"


def test_a_scene_with_only_an_implied_prohibition_is_still_checked_window_by_window(monkeypatch):
    """Mutation audit (H8): 'flat ground, then one pipe' - the first scene asks for nothing counted, but a pipe there is a violation of 'then'.  A pipe drawn in its first
    window must be redrawn LOCALLY (no whole-level resample needed)."""
    state = {"n": 0}

    def fake(bundle, caption, w, steps, r, gen_seed, known_cols, device):
        state["n"] += 1
        g = flat()
        if known_cols is not None:
            g[:, :known_cols.shape[1]] = known_cols
        if state["n"] == 1:
            add_pipe(g, 4)                                                          # a stray pipe in the very first window (scene 1)
        elif "one pipe" in caption:
            add_pipe(g, 9)
        return g

    monkeypatch.setattr(FL, "sample_window", fake)
    lvl = FL.generate(None, "flat ground, then one pipe", seed=1, width=48, device="cpu")
    adh = lvl["verification"]["adherence"]
    assert adh["status"] == "met" and adh["attempts"] == 0 and adh["window_redraws"] >= 1


def test_whole_level_resamples_spend_the_extra_draw_budget(monkeypatch):
    """Mutation audit (H9): with window redraws off, only whole-level resamples can use the budget; they must stop at max_extra_draws, not at max_attempts."""
    monkeypatch.setitem(A._A, "batch_tries_cpu", 1)
    monkeypatch.setitem(A._A, "max_window_tries", 1)
    monkeypatch.setitem(A._A, "max_extra_draws", 10)
    monkeypatch.setitem(A._A, "max_attempts", 16)
    monkeypatch.setattr(FL, "sample_window", lambda *a, **k: flat())                 # three enemies are never drawn
    lvl = FL.generate(None, "three enemies", seed=1, width=48, device="cpu")
    adh = lvl["verification"]["adherence"]
    assert adh["attempts"] < 16 and adh["extra_draws"] <= 10 + FL.n_windows(48), adh
