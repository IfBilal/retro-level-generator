"""Noise schedules and the forward/reverse maths of DDPM and DDIM (doc 13).

Explain-me:
  What: builds beta/alpha/alpha-bar tables (linear or cosine, float64), adds noise to clean levels (q_sample),
        recovers x0 from a predicted noise, and takes one reverse step (DDPM ancestral or DDIM).  Also the
        RePaint "renoise" jump used by outpainting.
  In/Out: tensors shaped (B, 13, H, W); timesteps are the 0-based code index t in 0..T-1 (t=-1 means "clean").
  Previous: dataset.py gives x0.  Next: the U-Net predicts eps; sampler.py/train.py call these functions.
  Conventions (doc 13 §0): alpha_bar[t] = prod_{i<=t} alpha_i, alpha_bar[-1] := 1.  Buffers are float64 and are
  cast to the tensor's dtype only when used.
  Try: SCHEDULE kind "linear" vs "cosine" (drill 3) - compare alpha_bar at t=500 (0.078 vs 0.492).
"""
from __future__ import annotations

import math

import torch


def make_betas(kind: str, T: int = 1000, beta_min: float = 1e-4, beta_max: float = 0.02, s: float = 0.008) -> torch.Tensor:
    """float64 betas.  linear: linspace(1e-4, 0.02).  cosine: Nichol & Dhariwal with beta clipped at 0.999."""
    if kind == "linear":
        return torch.linspace(beta_min, beta_max, T, dtype=torch.float64)
    if kind == "cosine":
        u = torch.arange(T + 1, dtype=torch.float64) / T
        f = torch.cos((u + s) / (1 + s) * math.pi / 2) ** 2
        ab = f / f[0]
        return (1 - ab[1:] / ab[:-1]).clamp(max=0.999)
    raise ValueError(f"unknown schedule {kind!r}")


class DiffusionSchedule:
    def __init__(self, kind: str = "linear", T: int = 1000, device: str | torch.device = "cpu"):
        self.kind, self.T = kind, T
        b = make_betas(kind, T)
        a = 1.0 - b
        ab = torch.cumprod(a, 0)                        # recomputed from the clipped betas (doc 13 §3.2)
        ab_prev = torch.cat([torch.ones(1, dtype=torch.float64), ab[:-1]])
        post_var = b * (1 - ab_prev) / (1 - ab)         # beta-tilde; 0 at t = 0
        self.buf = {
            "betas": b, "alphas": a, "alphas_cumprod": ab, "alphas_cumprod_prev": ab_prev,
            "sqrt_ab": ab.sqrt(), "sqrt_1m_ab": (1 - ab).sqrt(), "snr": ab / (1 - ab),
            "posterior_variance": post_var,
            "posterior_log_variance_clipped": torch.log(torch.cat([post_var[1:2], post_var[1:]])),
            "posterior_mean_coef1": ab_prev.sqrt() * b / (1 - ab),
            "posterior_mean_coef2": a.sqrt() * (1 - ab_prev) / (1 - ab),
        }
        self.buf = {k: v.to(device) for k, v in self.buf.items()}

    # ------------------------------------------------------------ helpers
    def to(self, device):
        self.buf = {k: v.to(device) for k, v in self.buf.items()}
        return self

    def _g(self, name: str, t, like: torch.Tensor) -> torch.Tensor:
        """Gather buffer[t] as (B,1,1,1) in like's dtype/device.  t = -1 is allowed only for alpha_bar (=1)."""
        t = torch.as_tensor(t, device=like.device).long().reshape(-1)
        buf = self.buf[name]
        if name == "alphas_cumprod":
            v = torch.where(t >= 0, buf[t.clamp(min=0)], torch.ones((), dtype=buf.dtype, device=buf.device))
        else:
            v = buf[t]
        return v.to(like.dtype).view(-1, *([1] * (like.dim() - 1)))

    def ab(self, t, like):
        return self._g("alphas_cumprod", t, like)

    def snr(self, t) -> torch.Tensor:
        return self.buf["snr"][torch.as_tensor(t).long()]

    # ------------------------------------------------------------ forward process (doc 13 §2)
    def q_sample(self, x0: torch.Tensor, t, eps: torch.Tensor) -> torch.Tensor:
        """x_t = sqrt(ab_t) x0 + sqrt(1-ab_t) eps."""
        return self._g("sqrt_ab", t, x0) * x0 + self._g("sqrt_1m_ab", t, x0) * eps

    # ------------------------------------------------------------ conversions (doc 13 §6)
    def predict_x0_from_eps(self, x_t, t, eps):
        return (x_t - self._g("sqrt_1m_ab", t, x_t) * eps) / self._g("sqrt_ab", t, x_t)

    def predict_eps_from_x0(self, x_t, t, x0):
        return (x_t - self._g("sqrt_ab", t, x_t) * x0) / self._g("sqrt_1m_ab", t, x_t)

    # ------------------------------------------------------------ posterior (doc 13 §4)
    def q_posterior_mean_variance(self, x0, x_t, t):
        mean = self._g("posterior_mean_coef1", t, x_t) * x0 + self._g("posterior_mean_coef2", t, x_t) * x_t
        return mean, self._g("posterior_variance", t, x_t), self._g("posterior_log_variance_clipped", t, x_t)

    # ------------------------------------------------------------ DDPM (doc 13 §7.1)
    def ddpm_step(self, eps_hat, t, x_t, generator: torch.Generator | None = None):
        """Ancestral step x_t -> x_{t-1}; at t = 0 no noise is added (variance is 0) and the result is x0-hat."""
        mean = (x_t - self._g("betas", t, x_t) / self._g("sqrt_1m_ab", t, x_t) * eps_hat) / self._g("alphas", t, x_t).sqrt()
        var = self._g("posterior_variance", t, x_t)
        z = torch.randn(x_t.shape, generator=generator, dtype=x_t.dtype, device=x_t.device)
        return mean + var.sqrt() * z

    # ------------------------------------------------------------ DDIM (doc 13 §7.2-7.3)
    def ddim_timesteps(self, steps: int) -> torch.Tensor:
        """S strided timesteps, from T-1 down to 0 (inclusive)."""
        return torch.linspace(self.T - 1, 0, steps).round().long()

    def ddim_step(self, eps_hat, t, s, x_t, eta: float = 0.0, generator: torch.Generator | None = None, clip: bool = True):
        """One DDIM update t -> s (s = -1 returns x0-hat).  With clip, x0-hat is clamped to [-1,1] and eps is
        recomputed from the clamped x0 so the update stays self-consistent (M13)."""
        x0 = self.predict_x0_from_eps(x_t, t, eps_hat)
        if clip:
            x0 = x0.clamp(-1.0, 1.0)
            eps_hat = self.predict_eps_from_x0(x_t, t, x0)
        ab_t, ab_s = self.ab(t, x_t), self.ab(s, x_t)
        sigma = eta * ((1 - ab_s) / (1 - ab_t)).sqrt() * (1 - ab_t / ab_s).clamp(min=0).sqrt()
        coef = (1 - ab_s - sigma ** 2).clamp(min=0).sqrt()
        out = ab_s.sqrt() * x0 + coef * eps_hat
        if eta > 0:
            out = out + sigma * torch.randn(x_t.shape, generator=generator, dtype=x_t.dtype, device=x_t.device)
        return out

    # ------------------------------------------------------------ RePaint jump back (doc 13 §9)
    def renoise(self, x_s, s, t, generator: torch.Generator | None = None):
        """Jump from the level of s back up to the (noisier) level of t, s < t:
        x_t = sqrt(ab_t/ab_s) x_s + sqrt(1 - ab_t/ab_s) z."""
        ratio = self.ab(t, x_s) / self.ab(s, x_s)
        z = torch.randn(x_s.shape, generator=generator, dtype=x_s.dtype, device=x_s.device)
        return ratio.sqrt() * x_s + (1 - ratio).sqrt() * z
