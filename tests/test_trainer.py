"""Trainer behaviour on a tiny CPU config (doc 16 §18). These test the training PATH; no real run is started."""
import json

import pytest
import torch

from conftest import needs_data
from text2mario.train import checkpoint as ckpt
from text2mario.train.loop import Trainer
from text2mario.utils.config import load_config

pytestmark = needs_data

TINY = ["model.base_channels=16", "train.batch_size=8", "optim.warmup_steps=2", "log_every=1", "checkpoint.every_steps=2"]


def cfg(overlay="configs/uncond_cosine.yaml", extra=()):
    return load_config(overlay, "cpu_dev", TINY + list(extra))


@pytest.fixture(scope="module")
def data():
    from text2mario.data.dataset import MarioWindows
    return MarioWindows("train")


def make(tmp_path, data, overlay="configs/uncond_cosine.yaml", extra=(), name="run"):
    torch.manual_seed(0)
    return Trainer(cfg(overlay, extra), tmp_path / name, "cpu", dataset=data)


def test_initial_loss_is_about_one_and_step_counter_advances(tmp_path, data):
    tr = make(tmp_path, data)
    m = tr.train_step()
    assert 0.9 <= m["loss"] <= 1.1 and tr.step == 1 and m["lr"] == 0.0     # step 0 has lr 0 (warm-up start)


def test_conditional_trainer_runs_and_updates_adapter(tmp_path, data):
    tr = make(tmp_path, data, "configs/cond_smoke.yaml")
    before = [p.detach().clone() for p in tr.cond.adapter.parameters()]
    for _ in range(3):
        m = tr.train_step()
    assert torch.isfinite(torch.tensor(m["loss"]))
    assert any(not torch.equal(a, b) for a, b in zip(before, tr.cond.adapter.parameters()))   # adapter is trained
    assert all(not p.requires_grad for p in tr.cond.encoder.parameters())


def test_run_writes_logs_checkpoints_and_ema(tmp_path, data):
    tr = make(tmp_path, data)
    tr.run(max_steps=4)
    d = tmp_path / "run"
    assert (d / "metrics.csv").exists() and (d / "events.jsonl").exists() and (d / "ema.safetensors").exists()
    events = [json.loads(l)["event"] for l in (d / "events.jsonl").read_text().splitlines()]
    assert events[0] == "started" and events[-1] == "finished" and "checkpoint" in events
    assert len(list(d.glob("step_*.pt"))) >= 2                                  # never fewer than the last two


def test_resume_equals_continuous_run(tmp_path, data):
    a = make(tmp_path, data, name="a", extra=["model.dropout=0.1"])
    for _ in range(2):
        a.train_step()
    p = a.save("step_0000002.pt")
    for _ in range(2):
        a.train_step()
    b = make(tmp_path, data, name="b", extra=["model.dropout=0.1"])
    b.resume(p)
    assert b.step == 2
    for _ in range(2):
        b.train_step()
    for (n, x), (_, y) in zip(a.unet.state_dict().items(), b.unet.state_dict().items()):
        assert torch.allclose(x, y, atol=1e-6), n


def test_resume_refuses_different_config(tmp_path, data):
    a = make(tmp_path, data, name="a")
    a.train_step()
    p = a.save("step_0000001.pt")
    b = make(tmp_path, data, name="b", extra=["optim.lr=0.001"])
    with pytest.raises(RuntimeError, match="config hash"):
        b.resume(p)


def test_gradient_accumulation_equals_full_batch(tmp_path, data):
    """Micro-batching is mathematically the same step (GroupNorm is batch independent)."""
    a = make(tmp_path, data, name="a", extra=["model.dropout=0.0", "optim.warmup_steps=0"])
    b = make(tmp_path, data, name="b", extra=["model.dropout=0.0", "optim.warmup_steps=0"])
    b.unet.load_state_dict(a.unet.state_dict())
    b.micro_batch = 4
    ma, mb = a.train_step(), b.train_step()
    assert ma["loss"] == pytest.approx(mb["loss"], rel=1e-5)
    for x, y in zip(a.unet.parameters(), b.unet.parameters()):
        assert torch.allclose(x, y, atol=1e-5)


def test_nan_loss_is_skipped_and_five_in_a_row_abort(tmp_path, data, monkeypatch):
    tr = make(tmp_path, data)
    monkeypatch.setattr(tr, "_loss", lambda *a, **k: torch.tensor(float("nan"), requires_grad=True))
    for _ in range(4):
        assert tr.train_step()["skipped"] == 1
    with pytest.raises(RuntimeError, match="nan_abort"):
        tr.train_step()
    assert tr.step == 0                                                         # the weights were never touched


def test_stop_file_saves_and_stops(tmp_path, data):
    tr = make(tmp_path, data)
    (tmp_path / "run").mkdir(exist_ok=True)
    (tmp_path / "run" / "STOP").write_text("")
    tr.run(max_steps=10)
    assert tr.step == 0 and list((tmp_path / "run").glob("step_*.pt"))


def test_aux_ce_and_channel_weighted_losses_run(tmp_path, data):
    for extra in (["loss.aux_ce_lambda=0.001"], ["loss.type=channel_weighted"]):
        tr = make(tmp_path, data, extra=extra, name="x" + extra[0][:9])
        assert torch.isfinite(torch.tensor(tr.train_step()["loss"]))


def test_batches_are_identical_on_two_machines_by_construction(tmp_path, data):
    """Index stream and noise depend only on (seed, step): same on any machine (doc 16 §8)."""
    a, b = make(tmp_path, data, name="a"), make(tmp_path, data, name="b")
    assert torch.equal(a.sampler.rows(7), b.sampler.rows(7))


def test_entry_point_enforces_gates_and_dry_runs(tmp_path, monkeypatch):
    from text2mario.train import train as entry
    entry.main(["--config", "configs/uncond_linear.yaml", "--machine", "cpu_dev", "--device", "cpu", "--dry-run",
                "--run-dir", str(tmp_path / "dry")] + sum([["--set", s] for s in TINY], []))
    import text2mario.utils.gates as g
    monkeypatch.setattr(g, "ROOT", tmp_path)                                    # no gate results here -> must refuse
    monkeypatch.setattr(entry, "require", g.require)
    with pytest.raises(g.GateNotPassedError):
        entry.main(["--config", "configs/uncond_linear.yaml", "--machine", "cpu_dev", "--device", "cpu", "--dry-run",
                    "--run-dir", str(tmp_path / "dry2")])


@pytest.mark.slow
def test_overfit_one_batch_sanity(tmp_path, data):
    """Optimiser-path check: 8 fixed scenes at fixed t must be memorised (loss < 0.01). Tiny model, CPU, not a real run."""
    tr = make(tmp_path, data, extra=["train.batch_size=8", "optim.warmup_steps=20", "optim.lr=1e-3", "model.dropout=0.0"])
    rows = torch.arange(8)
    x0 = data.onehot(rows)
    g = torch.Generator().manual_seed(0)
    eps = torch.randn(x0.shape, generator=g)
    t = torch.full((8,), 300)
    xt = tr.schedule.q_sample(x0, t, eps)
    last = 1.0
    for i in range(1500):                                      # the 1,500-step budget of doc 16 §18
        for grp in tr.opt.param_groups:
            grp["lr"] = 1e-3
        tr.opt.zero_grad()
        loss = ((tr.unet(xt, t, None, None) - eps) ** 2).mean()
        loss.backward()
        tr.opt.step()
        last = loss.item()
    assert last < 0.01
