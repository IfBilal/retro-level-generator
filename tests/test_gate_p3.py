"""G-P3 decision logic and per-sample seeding (doc 24 §G-P3, doc 22 §2.3)."""
import importlib.util

import pytest
import torch
import yaml
from conftest import ROOT

from text2mario.diffusion.scheduler import DiffusionSchedule
from text2mario.eval.generate import sample_unconditional
from text2mario.models.factory import build_unet

spec = importlib.util.spec_from_file_location("gate_p3", ROOT / "scripts/gate_p3.py")
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)
G = yaml.safe_load((ROOT / "configs/gates.yaml").read_text())["P3"]
REFS = {"copy_rate_ref": 0.104, "novelty_ref_median_stride8": 13.0}


def metrics(**kw):
    base = {"floor_valid": 0.95, "broken_rate": 0.0, "tile_js": 0.01, "copy_rate": 0.10, "n": 256}
    return {**base, **kw}


def test_s_noplay_weights():
    assert gate.s_noplay(1.0, 0.0, 13, 13) == pytest.approx(1.0)
    assert gate.s_noplay(0.0, 1.0, 0, 13) == pytest.approx(0.0)
    assert gate.s_noplay(1.0, 0.0, 40, 13) == pytest.approx(1.0)       # novelty term is capped at 1


def test_thresholds_come_from_the_ledger_and_copy_threshold_is_calibrated():
    assert (G["floor_valid_min"], G["broken_rate_max"], G["tile_js_max"]) == (0.85, 0.05, 0.05)
    rows = {c["id"]: c for c in gate.checks_for(metrics(), REFS, G)}
    assert rows["P3.04"]["threshold"] == pytest.approx(0.154) and all(c["pass"] for c in rows.values())


@pytest.mark.parametrize("field,value,failing", [("floor_valid", 0.80, "P3.01"), ("broken_rate", 0.06, "P3.02"),
                                                 ("tile_js", 0.06, "P3.03"), ("copy_rate", 0.16, "P3.04")])
def test_each_check_can_fail(field, value, failing):
    bad = [c["id"] for c in gate.checks_for(metrics(**{field: value}), REFS, G) if not c["pass"]]
    assert bad == [failing]


def test_boundary_values_pass():
    assert all(c["pass"] for c in gate.checks_for(metrics(floor_valid=0.85, broken_rate=0.05, tile_js=0.05, copy_rate=0.154), REFS, G))


def test_samples_do_not_depend_on_batch_size():
    torch.manual_seed(0)
    m = build_unet({"base_channels": 16, "zero_init": False}, conditional=False).eval()
    s = DiffusionSchedule("cosine")
    a = sample_unconditional(m, s, 6, 5_000_000, steps=3, batch=6)
    b = sample_unconditional(m, s, 6, 5_000_000, steps=3, batch=2)
    assert (a == b).all()                                             # same seeds -> same samples, whatever the batching
    assert not (a == sample_unconditional(m, s, 6, 5_000_001, steps=3)).all()
