"""The test set is locked until the freeze and used exactly once (CLAUDE.md hard rules 1-2; rubric: leakage prevention)."""
import importlib.util
import json

import pytest
from conftest import ROOT, needs_data


@needs_data
def test_test_windows_are_refused_before_the_freeze():
    from text2mario.data.dataset import MarioWindows
    with pytest.raises(PermissionError):
        MarioWindows("test")
    with pytest.raises(PermissionError):
        MarioWindows("all")
    MarioWindows("dev")   # dev and train are fine
    MarioWindows("train")


def test_dev_tooling_refuses_test_prompts():
    from text2mario.eval.devloop import load_prompts
    with pytest.raises(PermissionError):
        load_prompts("p_test")
    assert len(load_prompts("p_dev")) == 115


def test_calibration_never_touches_test_levels():
    spec = importlib.util.spec_from_file_location("calibrate_checker", ROOT / "scripts" / "calibrate_checker.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    with pytest.raises(AssertionError):
        mod.real_levels(("train", "test"))
    S = json.loads((ROOT / "data/splits/split_v1.json").read_text())["splits"]
    assert set(mod.real_levels()) == set(S["train"]) | set(S["dev"])
    assert not set(mod.real_levels()) & set(S["test"])


def test_split_has_no_level_in_two_sets():
    S = json.loads((ROOT / "data/splits/split_v1.json").read_text())["splits"]
    assert not (set(S["train"]) & set(S["dev"])) and not (set(S["train"]) & set(S["test"])) and not (set(S["dev"]) & set(S["test"]))


def test_one_shot_script_refuses_without_a_freeze(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("eval_test_mod", ROOT / "scripts" / "eval_test.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if (ROOT / "FROZEN.json").exists():
        pytest.skip("already frozen")
    with pytest.raises(SystemExit):
        mod.guards(None)
