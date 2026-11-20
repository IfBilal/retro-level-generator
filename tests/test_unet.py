"""U-Net tests (doc 14 §11, U1-U12)."""
import importlib.util

import pytest
import torch
from conftest import ROOT

from text2mario.diffusion.losses import loss_simple
from text2mario.models.factory import build_unet, count_params


def _oracle():
    spec = importlib.util.spec_from_file_location("ref_unet", ROOT / "reference" / "ref_unet.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def rand_inputs(b=4, w=16, l=20, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(b, 13, 16, w, generator=g)
    t = torch.randint(0, 1000, (b,), generator=g)
    ctx = torch.randn(b, l, 128, generator=g)
    mask = torch.ones(b, l, dtype=torch.bool)
    mask[:, l - 5:] = False
    return x, t, ctx, mask


def test_u1_parameter_counts():
    assert count_params(build_unet())["total"] == 18_312_013
    assert count_params(build_unet(conditional=False))["total"] == 18_312_141
    assert count_params(build_unet({"self_attention": False}))["total"] == 16_398_413
    assert count_params(build_unet({"base_channels": 32}))["total"] == 5_160_621
    assert count_params(build_unet({"cross_attention": False}))["total"] == 16_791_629
    assert count_params(build_unet({"pooled_text_to_time": True}))["total"] == 18_312_013 + 33_024


def test_u2_oracle_parity():
    """Copy the oracle's weights positionally into our U-Net: outputs must agree (eval mode, fp32)."""
    ref = _oracle().UNet().eval()
    ours = build_unet({"zero_init": False}).eval()
    rp, op = list(ref.parameters()), list(ours.parameters())
    assert len(rp) == len(op) == 430
    with torch.no_grad():
        for a, b in zip(rp, op):
            assert a.shape == b.shape
            b.copy_(a)
    x, t, ctx, mask = rand_inputs()
    with torch.no_grad():
        assert (ref(x, t, ctx, mask) - ours(x, t, ctx, mask)).abs().max() < 1e-5


@pytest.mark.parametrize("w", [16, 24, 112])
def test_u3_shapes(w):
    m = build_unet().eval()
    x, t, ctx, mask = rand_inputs(2, w)
    assert m(x, t, ctx, mask).shape == x.shape


def test_u3_width_not_multiple_of_four_is_rejected():
    m = build_unet().eval()
    x, t, ctx, mask = rand_inputs(2, 18)
    with pytest.raises(AssertionError):
        m(x, t, ctx, mask)


def test_u4_initial_loss_is_one():
    m = build_unet().eval()
    g = torch.Generator().manual_seed(1)
    x0 = torch.nn.functional.one_hot(torch.randint(0, 13, (64, 16, 16), generator=g), 13).permute(0, 3, 1, 2).float() * 2 - 1
    eps = torch.randn(x0.shape, generator=g)
    t = torch.randint(0, 1000, (64,), generator=g)
    _, _, ctx, mask = rand_inputs(64)
    with torch.no_grad():
        out = m(x0 + eps, t, ctx, mask)
    assert out.abs().max() == 0                        # zero-init head
    assert 0.95 <= loss_simple(out, eps).item() <= 1.05


def test_u5_every_parameter_gets_gradient_within_three_steps():
    m = build_unet().train()
    opt = torch.optim.AdamW(m.parameters(), lr=2e-4)
    seen = {n: False for n, _ in m.named_parameters()}
    x, t, ctx, mask = rand_inputs(8)
    for _ in range(3):
        opt.zero_grad()
        loss_simple(m(x, t, ctx, mask), torch.randn_like(x)).backward()
        for n, p in m.named_parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                seen[n] = True
        opt.step()
    missing = [n for n, ok in seen.items() if not ok]
    assert not missing, missing[:5]


def test_u6_time_sensitivity():
    m = build_unet({"zero_init": False}).eval()
    x, _, ctx, mask = rand_inputs(2)
    with torch.no_grad():
        d = (m(x, torch.full((2,), 10), ctx, mask) - m(x, torch.full((2,), 900), ctx, mask)).norm()
    assert d > 1e-3


def test_u7_masked_positions_do_not_matter():
    m = build_unet({"zero_init": False}).eval()
    x, t, ctx, mask = rand_inputs(3)
    ctx2 = ctx.clone()
    ctx2[:, 15:] = torch.randn_like(ctx2[:, 15:]) * 50      # garbage in the padded positions
    with torch.no_grad():
        assert (m(x, t, ctx, mask) - m(x, t, ctx2, mask)).abs().max() < 1e-5


def test_u8_eval_is_deterministic():
    m = build_unet({"zero_init": False}).eval()
    x, t, ctx, mask = rand_inputs(2)
    with torch.no_grad():
        assert torch.equal(m(x, t, ctx, mask), m(x, t, ctx, mask))


def test_u10_state_dict_keys_golden():
    got = [f"{k} {tuple(v.shape)}" for k, v in build_unet().state_dict().items()]
    want = (ROOT / "tests/golden/unet_state_dict_keys.txt").read_text().splitlines()
    assert got == want, "state-dict keys changed: this breaks every checkpoint"


def test_u11_return_attn_probabilities():
    m = build_unet({"zero_init": False}).eval()
    x, t, ctx, mask = rand_inputs(2)
    with torch.no_grad():
        out, attn = m(x, t, ctx, mask, return_attn=True)
        assert torch.allclose(out, m(x, t, ctx, mask), atol=1e-5)    # recording the maps must not change the output
    assert len(attn) == 11                                           # 11 cross-attention blocks (doc 14 §3)
    p = attn["enc.1.0"]
    assert p.shape == (2, 4, 64, 20) and torch.allclose(p.sum(-1), torch.ones(2, 4, 64), atol=1e-5)
    assert (p[..., 15:] == 0).all()                                  # padded tokens get zero attention


def test_conditional_model_refuses_missing_context():
    m = build_unet().eval()
    x, t, _, _ = rand_inputs(2)
    with pytest.raises(ValueError):
        m(x, t, None, None)


def test_unconditional_variant_runs_without_context_and_ignores_it():
    m = build_unet({"zero_init": False}, conditional=False).eval()
    x, t, ctx, mask = rand_inputs(2)
    with torch.no_grad():
        assert torch.equal(m(x, t), m(x, t, ctx, mask))


def test_unknown_config_key_rejected():
    with pytest.raises(ValueError):
        build_unet({"basechannels": 64})


@pytest.mark.slow
def test_u12_flop_budget():
    from torch.utils.flop_counter import FlopCounterMode
    m = build_unet({"zero_init": False}).eval()
    x, t, ctx, mask = rand_inputs(1, 16, 48)
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        m(x, t, ctx, mask)
    assert abs(fc.get_total_flops() / 1e9 - 1.241) / 1.241 < 0.05    # counter counts FLOPs (2 per MAC)
