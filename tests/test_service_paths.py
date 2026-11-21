"""LevelService checkpoint discovery and refusal paths (no model needed): every branch of default_ckpt, the missing-model error, invalid inputs."""
import json

import pytest

from text2mario.app import service as S


def fake_root(tmp_path, monkeypatch, *, bundle=False, ckpt_json=None, frozen=None, env=None):
    monkeypatch.setattr(S, "ROOT", tmp_path)
    monkeypatch.delenv("T2M_CKPT", raising=False)
    if env:
        monkeypatch.setenv("T2M_CKPT", env)
    (tmp_path / "app/model").mkdir(parents=True)
    if bundle:
        (tmp_path / "app/model/t2m_ema.safetensors").write_bytes(b"x")
    if ckpt_json:
        (tmp_path / "app/model/ckpt.json").write_text(json.dumps({"path": ckpt_json}))
    if frozen:
        (tmp_path / "FROZEN.json").write_text(json.dumps({"checkpoint": frozen}))


def test_default_ckpt_prefers_env_then_bundle_then_ckpt_json_then_frozen_then_none(tmp_path, monkeypatch):
    fake_root(tmp_path, monkeypatch, bundle=True, ckpt_json="runs/a.pt", frozen="runs/b.pt", env="/somewhere/c.pt")
    assert str(S.default_ckpt()).replace("\\", "/").endswith("/somewhere/c.pt")
    monkeypatch.delenv("T2M_CKPT")
    assert S.default_ckpt() == tmp_path / "app/model"                                  # the exported bundle wins over ckpt.json and FROZEN
    (tmp_path / "app/model/t2m_ema.safetensors").unlink()
    assert S.default_ckpt() == tmp_path / "runs/a.pt"
    (tmp_path / "app/model/ckpt.json").unlink()
    assert S.default_ckpt() == tmp_path / "runs/b.pt"
    (tmp_path / "FROZEN.json").unlink()
    assert S.default_ckpt() is None


def test_missing_model_is_a_clear_error_not_a_crash(tmp_path, monkeypatch):
    fake_root(tmp_path, monkeypatch)
    with pytest.raises(FileNotFoundError, match="no model checkpoint"):
        S.LevelService()
    with pytest.raises(FileNotFoundError):
        S.LevelService(tmp_path / "does_not_exist.pt")


def test_run_refuses_bad_inputs_before_touching_the_model():
    svc = S.LevelService.__new__(S.LevelService)                                       # no model loaded: validation must come first
    svc.device, svc.profile = "cpu", "local"
    for text, seed, w, width in (("", 1, 7.5, 48), ("one pipe", -1, 7.5, 48), ("one pipe", 1, 99.0, 48), ("one pipe", 1, 7.5, 50), (None, 1, 7.5, 48),
                                 ("-3 pipes", 1, 7.5, 48), ("one pipe", True, 7.5, 48)):
        with pytest.raises(ValueError):
            svc.run(text, seed=seed, w=w, width=width)


@pytest.mark.parametrize("seed,ok", [(0, True), (2**53 - 1, True), (2**53, False), (-1, False), (1.5, False), (True, False), ("7", False)])
def test_validate_settings_seed_boundaries(seed, ok):
    assert S.validate_settings(seed, 7.5, 48)[0] is ok


@pytest.mark.parametrize("w,ok", [(1.0, True), (10.0, True), (0.99, False), (10.01, False), (True, False), ("5", False)])
def test_validate_settings_cfg_boundaries(w, ok):
    assert S.validate_settings(5, w, 48)[0] is ok


@pytest.mark.parametrize("text,ok", [("a" * 400, True), ("a" * 401, False), ("  one pipe  ", True), ("one\npipe", True), ("one\x00pipe", False), ("\t", False),
                                     ("5-6 coins", True), ("- 3 pipes", False), ("one pipe -2", False)])
def test_validate_prompt_boundaries(text, ok):
    assert S.validate_prompt(text)[0] is ok
