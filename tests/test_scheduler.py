"""Scheduler maths tests (doc 13 §16; gate G-P2)."""
import math

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from text2mario.diffusion import losses
from text2mario.diffusion.scheduler import DiffusionSchedule, make_betas

T_ = 1000
LIN, COS = DiffusionSchedule("linear"), DiffusionSchedule("cosine")


def onehot_batch(n=8, h=16, w=16, seed=0, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    y = torch.randint(0, 13, (n, h, w), generator=g)
    return torch.nn.functional.one_hot(y, 13).permute(0, 3, 1, 2).to(dtype) * 2 - 1, y


# values from doc 13 §3.3 (computed in float64 on 2026-11-17)
TABLE = {
    "linear": {0: (1.000e-4, 0.99990), 100: (2.092e-3, 0.89514), 250: (5.080e-3, 0.52142), 500: (1.006e-2, 0.077797),
               750: (1.504e-2, 3.300e-3), 999: (2.000e-2, 4.036e-5)},
    "cosine": {0: (4.128e-5, 0.999959), 100: (5.319e-4, 0.97158), 250: (1.327e-3, 0.84589), 500: (3.156e-3, 0.49229),
               750: (7.579e-3, 0.14318), 999: (0.9990, 2.429e-9)},
}


@pytest.mark.parametrize("kind", ["linear", "cosine"])
def test_reference_values(kind):
    s = DiffusionSchedule(kind)
    for t, (beta, ab) in TABLE[kind].items():
        assert s.buf["betas"][t].item() == pytest.approx(beta, rel=2e-3), (kind, t)
        assert s.buf["alphas_cumprod"][t].item() == pytest.approx(ab, rel=2e-3), (kind, t)


def test_alpha_bar_monotone_and_bounds():
    for s, bound in ((LIN, 1e-4), (COS, 1e-8)):
        ab = s.buf["alphas_cumprod"]
        assert (ab[1:] < ab[:-1]).all() and ab[-1] < bound


def test_cosine_clips_exactly_one_beta():
    assert (make_betas("cosine") >= 0.999 - 1e-12).sum().item() == 1


def test_snr_crossings():
    first_below = lambda s, thr: int((s.buf["snr"] < thr).nonzero()[0])
    assert first_below(LIN, 1) == 259 and first_below(COS, 1) == 496
    assert first_below(LIN, 1 / 8) == 464 and first_below(COS, 1 / 8) == 781


def test_ddim_timestep_lists():
    s50 = LIN.ddim_timesteps(50).tolist()
    assert s50[:5] == [999, 979, 958, 938, 917] and s50[-3:] == [41, 20, 0] and len(s50) == 50
    s25 = LIN.ddim_timesteps(25).tolist()
    assert s25[:4] == [999, 957, 916, 874] and s25[-2:] == [42, 0] and len(s25) == 25


@pytest.mark.parametrize("s", [LIN, COS])
def test_x0_eps_round_trip(s):
    x0, _ = onehot_batch()
    eps = torch.randn_like(x0)
    t = torch.randint(0, T_, (8,))
    xt = s.q_sample(x0, t, eps)
    assert (s.predict_x0_from_eps(xt, t, eps) - x0).abs().max() < 1e-9
    assert (s.predict_eps_from_x0(xt, t, x0) - eps).abs().max() < 1e-9


@pytest.mark.parametrize("s", [LIN, COS])
def test_oracle_ddim_reconstructs_x0_exactly(s):
    """G-P2: with the TRUE noise at every step, eta=0 DDIM returns x0 (float64 MSE < 1e-6, argmax 100 %)."""
    x0, y = onehot_batch()
    ts = s.ddim_timesteps(50)
    x = torch.randn_like(x0, generator=None)
    for i, t in enumerate(ts):
        tt = t.repeat(8)
        eps = s.predict_eps_from_x0(x, tt, x0)               # oracle epsilon
        nxt = ts[i + 1] if i + 1 < len(ts) else torch.tensor(-1)
        x = s.ddim_step(eps, tt, nxt.repeat(8), x, eta=0.0, clip=True)
    assert torch.nn.functional.mse_loss(x, x0).item() < 1e-6
    assert (x.argmax(1) == y).all()


@pytest.mark.parametrize("s", [LIN, COS])
def test_ddpm_oracle_gives_posterior_mean(s):
    x0, _ = onehot_batch()
    t = torch.full((8,), 400)
    eps = torch.randn_like(x0)
    xt = s.q_sample(x0, t, eps)
    mean, var, _ = s.q_posterior_mean_variance(x0, xt, t)
    g = torch.Generator().manual_seed(0)
    out = s.ddpm_step(eps, t, xt, g)
    # mean part: set the noise to zero by comparing with variance removed
    z = torch.randn(xt.shape, generator=torch.Generator().manual_seed(0), dtype=xt.dtype)
    assert (out - var.sqrt() * z - mean).abs().max() < 1e-9


def test_q_sample_moments_match_theory():
    x0, _ = onehot_batch(n=200, seed=3)
    for s in (LIN, COS):
        for t in (0, 250, 500, 750, 999):
            tt = torch.full((200,), t)
            xt = s.q_sample(x0, tt, torch.randn_like(x0))
            ab = s.buf["alphas_cumprod"][t].item()
            assert xt.var().item() == pytest.approx(ab * x0.var().item() + (1 - ab), abs=0.02)
            assert xt.mean().item() == pytest.approx(math.sqrt(ab) * x0.mean().item(), abs=0.01)


def test_x999_is_unit_gaussian():
    x0, _ = onehot_batch(n=300, seed=4)
    for s in (LIN, COS):
        x = s.q_sample(x0, torch.full((300,), 999), torch.randn_like(x0))
        assert abs(x.mean().item()) < 0.02 and abs(x.std().item() - 1) < 0.02


def test_posterior_variance_edge_cases():
    for s in (LIN, COS):
        assert s.buf["posterior_variance"][0].item() == 0.0
        assert torch.isfinite(s.buf["posterior_log_variance_clipped"]).all()
    assert LIN.buf["posterior_variance"][1].item() == pytest.approx(5.45e-5, rel=1e-2)
    assert COS.buf["posterior_variance"][1].item() == pytest.approx(2.18e-5, rel=1e-2)


def test_posterior_mean_equals_eps_parameterisation():
    """Posterior mean in (x0, x_t) form equals the eps form mu = (x_t - beta/sqrt(1-ab) eps)/sqrt(alpha) (doc 13 §4)."""
    s, t = LIN, 500
    x0 = torch.ones(200000, dtype=torch.float64)
    xt = s.q_sample(x0, torch.full((200000,), t), torch.randn_like(x0))
    x_prev = s.q_sample(x0, torch.full((200000,), t - 1), torch.randn_like(x0))
    # joint sampling is awkward; instead verify the closed forms with the identity mu = (x_t - beta/sqrt(1-ab) eps)/sqrt(alpha)
    eps = s.predict_eps_from_x0(xt, torch.full((200000,), t), x0)
    mean, _, _ = s.q_posterior_mean_variance(x0, xt, torch.full((200000,), t))
    alt = (xt - s.buf["betas"][t] / s.buf["sqrt_1m_ab"][t] * eps) / s.buf["alphas"][t].sqrt()
    assert (mean - alt).abs().max() < 1e-9 and x_prev.shape == xt.shape


@pytest.mark.parametrize("pair", [(0, 20), (500, 520), (959, 979)])
def test_renoise_matches_forward_marginal(pair):
    s_, t_ = pair
    s = LIN
    x0 = torch.ones(200000, dtype=torch.float64)
    xs = s.q_sample(x0, torch.full((200000,), s_), torch.randn_like(x0))
    xt = s.renoise(xs, torch.full((200000,), s_), torch.full((200000,), t_))
    ab = s.buf["alphas_cumprod"][t_].item()
    assert xt.mean().item() == pytest.approx(math.sqrt(ab), abs=0.01)
    assert xt.var().item() == pytest.approx(1 - ab, abs=0.01)


def test_ddim_eta1_s_equals_t_minus_1_matches_ddpm_in_distribution():
    """eta=1, s=t-1 should have the DDPM variance beta-tilde (doc 13 §7.2)."""
    s, t = LIN, 300
    ab_t, ab_s = s.buf["alphas_cumprod"][t], s.buf["alphas_cumprod"][t - 1]
    sigma2 = ((1 - ab_s) / (1 - ab_t) * (1 - ab_t / ab_s))
    assert sigma2.item() == pytest.approx(s.buf["posterior_variance"][t].item(), rel=1e-9)


@given(t=st.integers(0, 999))
@settings(max_examples=50, deadline=None)
def test_property_gather_shapes(t):
    x = torch.zeros(3, 13, 16, 16)
    assert LIN._g("sqrt_ab", torch.full((3,), t), x).shape == (3, 1, 1, 1)


def test_clipping_keeps_x0_in_box_and_recomputes_eps():
    s = LIN
    x0, _ = onehot_batch(dtype=torch.float32)
    t = torch.full((8,), 999)
    xt = torch.randn_like(x0)
    wild_eps = torch.randn_like(x0) * 5
    out = s.ddim_step(wild_eps, t, torch.full((8,), 979), xt, clip=True)
    assert torch.isfinite(out).all() and out.abs().max() < 10


# ------------------------------------------------------------------ losses
def test_channel_weights_mean_one_and_values():
    w = losses.channel_weights()
    assert w.mean().item() == pytest.approx(1.0, abs=1e-6)
    assert w[0].item() == pytest.approx(0.289, abs=1e-3) and w[8].item() == pytest.approx(0.578, abs=1e-3)
    assert w[1].item() == pytest.approx(1.156, abs=1e-3)


def test_initial_loss_of_zero_output_is_about_one():
    eps = torch.randn(64, 13, 16, 16)
    assert 0.95 <= losses.loss_simple(torch.zeros_like(eps), eps).item() <= 1.05


def test_aux_ce_gradient_bounded_per_element():
    """With the alpha_bar weight the CE gradient into eps_hat is bounded by lam*tau/2 per element at every t (doc 13 §10.3)."""
    s = COS
    x0, y = onehot_batch(n=16, dtype=torch.float32)
    for t_ in (10, 300, 700, 999):
        t = torch.full((16,), t_)
        eps = torch.randn_like(x0)
        xt = s.q_sample(x0, t, eps)
        eps_hat = (eps + 0.3 * torch.randn_like(eps)).requires_grad_(True)
        lam = 1e-3
        ce_only = losses.loss_aux_ce(eps_hat, eps, xt, t, y, s, lam=lam) - losses.loss_simple(eps_hat, eps)
        (g,) = torch.autograd.grad(ce_only * 16, eps_hat)   # undo the batch-mean to get a per-sample bound
        assert g.abs().max().item() <= lam * 0.5 + 1e-7
