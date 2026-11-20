"""Guidance algebra (doc 15 T10) and sampler behaviour (doc 13 §7-9)."""
import torch

from text2mario.diffusion.guidance import guided_eps
from text2mario.diffusion.sampler import ddim_sample
from text2mario.diffusion.scheduler import DiffusionSchedule
from text2mario.models.factory import build_unet


def setup_model(seed=0):
    torch.manual_seed(seed)
    return build_unet({"base_channels": 16, "zero_init": False}).eval()


def ctxs(b=2):
    g = torch.Generator().manual_seed(1)
    c = torch.randn(b, 9, 128, generator=g)
    mc = torch.ones(b, 9, dtype=torch.bool)
    mc[:, 7:] = False
    u = torch.randn(b, 2, 128, generator=g)                 # the null context is shorter ([CLS][SEP])
    mu = torch.ones(b, 2, dtype=torch.bool)
    return c, mc, u, mu


def test_t10_guidance_algebra():
    m = setup_model()
    x, t = torch.randn(2, 13, 16, 16), torch.tensor([300, 700])
    c, mc, u, mu = ctxs()
    with torch.no_grad():
        eps_c, eps_u = m(x, t, c, mc), m(x, t, u, mu)
        assert torch.allclose(guided_eps(m, x, t, c, mc, u, mu, 1.0), eps_c, atol=1e-5)      # w = 1: conditional
        assert torch.allclose(guided_eps(m, x, t, c, mc, u, mu, 0.0), eps_u, atol=1e-5)      # w = 0: unconditional
        assert torch.allclose(guided_eps(m, x, t, c, mc, u, mu, 4.0), eps_u + 4 * (eps_c - eps_u), atol=1e-4)


def test_sampler_is_deterministic_and_seed_dependent():
    m, s = setup_model(), DiffusionSchedule("cosine")
    c, mc, u, mu = ctxs(1)
    kw = dict(ctx_c=c, mask_c=mc, ctx_u=u, mask_u=mu, w=3.0, steps=4)
    a, _ = ddim_sample(m, s, (1, 13, 16, 16), generator=torch.Generator().manual_seed(1), **kw)
    b, _ = ddim_sample(m, s, (1, 13, 16, 16), generator=torch.Generator().manual_seed(1), **kw)
    d, _ = ddim_sample(m, s, (1, 13, 16, 16), generator=torch.Generator().manual_seed(2), **kw)
    assert torch.equal(a, b) and not torch.equal(a, d)


def test_known_region_is_returned_exactly_and_resampling_runs():
    m, s = setup_model(), DiffusionSchedule("linear")
    c, mc, u, mu = ctxs(1)
    x0 = torch.nn.functional.one_hot(torch.randint(0, 13, (1, 16, 24)), 13).permute(0, 3, 1, 2).float() * 2 - 1
    mask = torch.zeros(1, 1, 16, 24)
    mask[..., :8] = 1                                          # leftmost 8 columns are known
    out, st = ddim_sample(m, s, (1, 13, 16, 24), ctx_c=c, mask_c=mc, ctx_u=u, mask_u=mu, w=2.0, steps=6,
                          known=(x0, mask), resample_r=2, generator=torch.Generator().manual_seed(0))
    assert torch.equal(out[..., :8], x0[..., :8]) and out.shape == (1, 13, 16, 24)
    assert 0.0 <= st["clip_rate"] <= 1.0


def test_unconditional_model_samples_without_context():
    torch.manual_seed(0)
    m = build_unet({"base_channels": 16, "zero_init": False}, conditional=False).eval()
    out, _ = ddim_sample(m, DiffusionSchedule("linear"), (2, 13, 16, 16), steps=3, generator=torch.Generator().manual_seed(0))
    assert out.shape == (2, 13, 16, 16) and torch.isfinite(out).all()
