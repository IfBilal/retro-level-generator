"""Training losses (ledger M9, doc 13 §5, §10).

Explain-me:
  What: loss_simple is the baseline epsilon-MSE.  loss_channel_weighted (rollback) weights tile channels so rare tiles
        count more.  loss_aux_ce (experiment X1) adds a small cross-entropy on the decoded x0-hat.
  In/Out: eps_hat/eps (B,13,H,W) float32 (cast the network output with .float() first); returns a scalar tensor.
  Try: set lambda_ce to 0.01 and run 50 steps (drill 13) to see the loss values change.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from text2mario.data import tiles as T

# Raw channel weights (M9): sky 0.5, ground + breakable 1.0, the ten rarer tiles 2.0.
_RAW = {T.SKY: 0.5, T.GROUND: 1.0, T.BREAKABLE: 1.0}


def channel_weights() -> torch.Tensor:
    """Normalised so the mean over the 13 channels is 1 (keeps the loss scale, hence the effective LR)."""
    raw = torch.tensor([_RAW.get(c, 2.0) for c in range(T.N_TILES)], dtype=torch.float32)
    return raw / raw.mean()


def loss_simple(eps_hat: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
    """L_simple = mean over all B*D elements of (eps - eps_hat)^2."""
    return F.mse_loss(eps_hat.float(), eps.float())


def loss_channel_weighted(eps_hat: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
    w = channel_weights().to(eps.device).view(1, -1, 1, 1)
    return (w * (eps_hat.float() - eps.float()) ** 2).mean()


def loss_aux_ce(eps_hat: torch.Tensor, eps: torch.Tensor, x_t: torch.Tensor, t, y: torch.Tensor, schedule,
                lam: float = 1e-3, tau: float = 1.0, weighting: str = "alpha_bar") -> torch.Tensor:
    """L = L_simple + lam * mean_b[ w_t * sum_ij CE(softmax(tau * x0_hat), y) ],  w_t = alpha_bar_t (or 1 for the
    exact-paper variant).  x0_hat is UNCLIPPED (clipping would kill the gradient) and computed in float32."""
    eps_hat, eps, x_t = eps_hat.float(), eps.float(), x_t.float()
    x0_hat = schedule.predict_x0_from_eps(x_t, t, eps_hat)
    ce = F.cross_entropy(tau * x0_hat, y.long(), reduction="none").sum(dim=(1, 2))   # (B,) sum over pixels
    w = schedule.ab(t, ce.new_zeros(len(ce), 1)).view(-1) if weighting == "alpha_bar" else torch.ones_like(ce)
    return loss_simple(eps_hat, eps) + lam * (w * ce).mean()
