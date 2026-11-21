"""Optimiser groups, LR schedule, EMA and atomic checkpoints (doc 16 §5-6, doc 13 §13)."""
import pytest
import torch

from text2mario.models.ema import EMA, ema_decay
from text2mario.models.factory import build_unet
from text2mario.train import checkpoint as ckpt
from text2mario.train.optim import build_optimizer, lr_at, param_groups
from text2mario.utils.config import ConfigError, config_hash, load_config

CFG = load_config()


def test_lr_schedule_reference_values():
    c = {"optim": CFG["optim"], "train": {"max_steps": 15000}}
    assert lr_at(0, c) == 0 and lr_at(500, c) == pytest.approx(1e-4) and lr_at(1000, c) == pytest.approx(2e-4)
    assert lr_at(5000, c) == pytest.approx(1.64e-4, rel=2e-2) and lr_at(10000, c) == pytest.approx(6.38e-5, rel=2e-2)
    assert lr_at(15000, c) == pytest.approx(1e-5)
    c60 = {"optim": CFG["optim"], "train": {"max_steps": 60000}}
    assert lr_at(30000, c60) == pytest.approx(1.08e-4, rel=2e-2)


def test_ema_decay_ramp():
    assert ema_decay(0) == pytest.approx(0.1) and ema_decay(10) == pytest.approx(0.55)
    assert ema_decay(100) == pytest.approx(0.918, abs=1e-3) and ema_decay(1000) == pytest.approx(0.991, abs=1e-3)
    assert ema_decay(8990) == pytest.approx(0.999, abs=1e-5) and ema_decay(20000) == 0.999


def test_param_groups_decay_only_weights():
    u = build_unet({"base_channels": 16}, conditional=False)
    g = param_groups({"unet": u}, 0.01)
    decay, no = g[0]["params"], g[1]["params"]
    assert all(p.ndim >= 2 for p in decay) and all(p.ndim < 2 or p.ndim == 3 for p in no)
    assert sum(p.numel() for p in decay + no) == sum(p.numel() for p in u.parameters())
    assert any(p is u.null_ctx for p in no)                 # the learned null context is never decayed
    assert g[1]["weight_decay"] == 0.0


def test_optimizer_has_adapter_and_not_minilm():
    from text2mario.text.conditioner import TextConditioner
    cond = TextConditioner()
    u = build_unet({"base_channels": 16})
    opt = build_optimizer({"unet": u, "adapter": cond.adapter}, CFG)
    ids = {id(p) for gr in opt.param_groups for p in gr["params"]}
    assert all(id(p) in ids for p in cond.adapter.parameters())                 # T3: adapter trained
    assert not any(id(p) in ids for p in cond.encoder.parameters())             # T2: MiniLM frozen and absent
    assert not any(p.requires_grad for p in cond.encoder.parameters())


def test_ema_tracks_and_swaps():
    m = torch.nn.Linear(3, 3)
    ema = EMA({"m": m}, 0.9)
    before = m.weight.detach().clone()
    with torch.no_grad():
        m.weight.add_(1.0)
    ema.update({"m": m}, 100)
    ema.apply_to({"m": m})
    assert not torch.allclose(m.weight, before + 1.0) and (m.weight - before).abs().max() < 1.0
    ema.restore({"m": m})
    assert torch.allclose(m.weight, before + 1.0)


def test_checkpoint_roundtrip_atomic_and_corruption(tmp_path):
    state = {"step": 5, "w": torch.randn(4, 4), "rng": torch.get_rng_state()}
    p = ckpt.save(state, tmp_path / "step_0000005.pt")
    assert not list(tmp_path.glob("*.tmp")) and p.with_suffix(".pt.sha256").exists()
    back = ckpt.load(p)
    assert back["step"] == 5 and torch.equal(back["w"], state["w"])
    data = bytearray(p.read_bytes())
    data[-10] ^= 0xFF
    p.write_bytes(bytes(data))
    with pytest.raises(ckpt.CheckpointError):
        ckpt.load(p)


def test_prune_keeps_last_two_and_milestones(tmp_path):
    for s in (1, 2, 3, 4, 5):
        ckpt.save({"step": s}, tmp_path / f"step_{s:07d}.pt")
    (tmp_path / "step_0000002.milestone").write_text("")
    ckpt.prune(tmp_path, keep_last=1)                        # asking for 1 still keeps 2
    left = sorted(p.name for p in tmp_path.glob("step_*.pt"))
    assert left == ["step_0000002.pt", "step_0000004.pt", "step_0000005.pt"]


def test_safetensors_export(tmp_path):
    from safetensors.torch import load_file
    p = ckpt.export_safetensors({"a": torch.ones(2, 2)}, tmp_path / "e.safetensors", {"step": 3})
    assert torch.equal(load_file(str(p))["a"], torch.ones(2, 2))


def test_machine_yaml_cannot_change_hyperparameters(tmp_path, monkeypatch):
    bad = tmp_path / "bad.yaml"
    bad.write_text("lr: 1.0e-3\n")
    import text2mario.utils.config as c
    monkeypatch.setattr(c, "ROOT", tmp_path)
    (tmp_path / "configs" / "machine").mkdir(parents=True)
    (tmp_path / "configs" / "base.yaml").write_text("optim: {lr: 0.0002}\nlog_every: 1\n")
    (tmp_path / "configs" / "machine" / "evil.yaml").write_text("optim: {lr: 0.1}\n")
    with pytest.raises(ConfigError):
        c.load_config(None, "evil")


def test_config_hash_ignores_machine_knobs_but_not_hyperparameters():
    a = load_config("configs/uncond_linear.yaml", "cpu_dev")
    b = load_config("configs/uncond_linear.yaml", "laptop_4050")
    assert config_hash(a) == config_hash(b)
    assert config_hash(load_config("configs/uncond_linear.yaml", None, ["optim.lr=0.001"])) != config_hash(a)
    with pytest.raises(ConfigError):
        load_config("configs/uncond_linear.yaml", None, ["optim.nope=1"])


def test_override_parses_scientific_notation_as_number():
    cfg = load_config(None, None, ["optim.lr=1e-3", "train.batch_size=8", "train.uncond=false", "diffusion.schedule=cosine"])
    assert cfg["optim"]["lr"] == 0.001 and isinstance(cfg["optim"]["lr"], float)
    assert cfg["train"]["batch_size"] == 8 and cfg["train"]["uncond"] is False and cfg["diffusion"]["schedule"] == "cosine"


def test_ema_load_state_dict_keeps_the_live_device():
    """Regression: resuming on CUDA crashed because the EMA shadow loaded from a CPU checkpoint stayed on the CPU."""
    import torch

    from text2mario.models.ema import EMA
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = {"unet": torch.nn.Linear(4, 4).to(dev)}
    ema = EMA(m, 0.999)
    saved = {k: v.cpu() for k, v in ema.state_dict()["shadow"].items()}      # what torch.load(map_location="cpu") returns
    ema.load_state_dict({"shadow": saved, "max_decay": 0.999})
    assert all(v.device.type == torch.device(dev).type for v in ema.shadow.values())
    ema.update(m, 5)                                                         # would raise "two devices" before the fix
