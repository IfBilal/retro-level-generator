"""Full-level generation tests (doc 20 §8) with a stub sampler: windows, framing, sanitizer, verify budgets, determinism."""
import numpy as np
import pytest

from text2mario.data import tiles as T
from text2mario.generate import full_level as FL
from text2mario.generate.framing import LEAD, STAIRS, TAIL, frame
from text2mario.generate.sanitizer import detect, repair
from text2mario.render.schema import validate_level


def flat_window(*_a, **_k):
    g = np.zeros((16, 16), np.uint8)
    g[15, :] = T.GROUND
    return g


@pytest.fixture
def stub(monkeypatch):
    """Replace the diffusion sampler by flat floors and count the calls; the 'bundle' is never touched."""
    calls = []

    def fake(bundle, caption, w, steps, r, gen_seed, known_cols, device):
        calls.append((caption, gen_seed, None if known_cols is None else known_cols.copy()))
        g = flat_window()
        if known_cols is not None:
            g[:, :8] = known_cols
        return g

    monkeypatch.setattr(FL, "sample_window", fake)
    return calls


def test_window_counts_and_scene_spread():
    assert [FL.n_windows(w) for w in (16, 48, 80, 112)] == [1, 5, 9, 13]
    assert FL.scene_for_windows(["a", "b"], 4) == [0, 0, 1, 1]
    assert FL.scene_for_windows(["a"], 5) == [0] * 5


def test_window_seed_is_deterministic_and_distinct():
    assert FL.window_seed(1, 2, 0) == FL.window_seed(1, 2, 0)
    assert len({FL.window_seed(1, k, a) for k in range(13) for a in range(5)}) == 65


def test_framing_exact_and_flagged():
    g, fr = frame(np.zeros((16, 48), np.uint8))
    assert g.shape == (16, LEAD + 48 + TAIL)
    assert (g[15, :LEAD] == T.GROUND).all() and (g[:15, :LEAD] == T.SKY).all()
    for c, h in enumerate(STAIRS):
        assert (g[16 - h:, LEAD + 48 + c] == T.GROUND).all() and (g[:16 - h, LEAD + 48 + c] == T.SKY).all()
    assert fr["templated_cols"] == [[0, LEAD - 1], [LEAD + 48, LEAD + 48 + TAIL - 1]]
    assert fr["spawn"] == {"col": 1, "row": 14}


def test_known_columns_are_carried_between_windows(stub):
    lvl = FL.generate(None, "full floor", seed=3, width=48, device="cpu", verify=False)
    assert len(stub) == 5 and stub[0][2] is None
    assert all(c[2] is not None and c[2].shape == (16, 8) for c in stub[1:])
    validate_level(lvl)
    assert lvl["size"]["cols"] == 3 + 48 + 6 and lvl["verification"]["status"] == "not_verified"


def test_determinism_same_seed_same_level(stub):
    a = FL.generate(None, "two pipes", seed=11, width=16, device="cpu", verify=False)
    b = FL.generate(None, "two pipes", seed=11, width=16, device="cpu", verify=False)
    assert a["raw_grid"] == b["raw_grid"] and [c[1] for c in stub[:1]] == [c[1] for c in stub[1:2]]


def test_flat_level_is_verified(stub):
    lvl = FL.generate(None, "full floor", seed=1, width=48, device="cpu")
    assert lvl["verification"]["status"] == "verified" and lvl["verification"]["checker"]["physics_sha256"]


def test_verify_loop_respects_budgets(monkeypatch):
    """An unwinnable level (no floor in window 2) must stop after <= 5 attempts per window and <= 20 in total."""
    def pit_window(bundle, caption, w, steps, r, gen_seed, known_cols, device):
        g = np.zeros((16, 16), np.uint8)            # never any floor -> never solvable
        if known_cols is not None:
            g[:, :8] = known_cols
        return g
    monkeypatch.setattr(FL, "sample_window", pit_window)
    lvl = FL.generate(None, "x", seed=1, width=48, device="cpu")
    v = lvl["verification"]
    assert v["status"] == "not_verified" and v["reason"] == "attempt budget exhausted"
    assert v["attempts_total"] <= 21 and all(a["attempt"] <= 5 for a in v["attempt_log"])
    per_window = {}
    for a in v["attempt_log"]:
        per_window[a["window"]] = per_window.get(a["window"], 0) + 1
    assert max(per_window.values()) <= 6


def test_sanitizer_detects_and_never_alters_raw():
    g = np.zeros((16, 16), np.uint8)
    g[15, :] = T.GROUND
    g[14, 3] = T.PIPE_TL                             # cap without its right half -> broken pipe
    g[13, 8] = T.CANNON_SUP                          # support with no barrel above -> broken cannon
    before = g.copy()
    d = detect(g)
    assert d["broken_pipes"] >= 1 and d["broken_cannons"] >= 1
    fixed, n = repair(g)
    assert np.array_equal(g, before) and n >= 2
    assert detect(fixed) == {"broken_pipes": 0, "broken_cannons": 0}


def test_repair_keeps_valid_pipes():
    g = np.zeros((16, 16), np.uint8)
    g[15, :] = T.GROUND
    g[13, 4:6] = (T.PIPE_TL, T.PIPE_TR)
    g[14, 4:6] = (T.PIPE_BL, T.PIPE_BR)
    fixed, n = repair(g)
    assert n == 0 and np.array_equal(fixed, g)
