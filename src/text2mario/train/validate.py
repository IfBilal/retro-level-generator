"""Deterministic per-t-bucket validation loss and conditioning gain (doc 16 §10, doc 15 §8).

Explain-me:
  What: evaluates the dev windows on a FIXED set of (window, t, noise) triples (made once from a fixed seed), so the
        curve does not jitter between evaluations.  Reports the loss in 10 t-buckets (bucket k = t in [100k, 100k+99]),
        the conditioning gain G_b = loss(null caption) - loss(caption) (must turn positive in mid-t buckets), and
        simplex_dev = mean |sum_c x0_hat_c + 11| (an ideal denoiser gives 0).
  In/Out: Validator(dev_windows, schedule).run(model, conditioner) -> dict of floats.
  Try: compare val/loss_t3 (t in 300..399) between linear and cosine - only within one schedule, never across.
"""
from __future__ import annotations

import torch

from text2mario.diffusion.losses import loss_simple


class Validator:
    def __init__(self, dev_windows, schedule, n_buckets: int = 10, per_bucket: int = 2, seed: int = 12345,
                 batch: int = 64, device="cpu"):
        self.d, self.schedule, self.nb, self.batch, self.device = dev_windows, schedule, n_buckets, batch, device
        rows = dev_windows.eval_rows(8)                                   # stride-8 subset (D6)
        g = torch.Generator().manual_seed(seed)
        T = schedule.T
        width = T // n_buckets
        self.rows, self.t, self.bucket = [], [], []
        for k in range(n_buckets):
            for r in rows.tolist():
                for _ in range(per_bucket):
                    self.rows.append(r)
                    self.bucket.append(k)
                    self.t.append(k * width + int(torch.randint(0, width, (1,), generator=g)))
        self.rows = torch.tensor(self.rows)
        self.t = torch.tensor(self.t)
        self.bucket = torch.tensor(self.bucket)
        self.eps = torch.randn((len(self.rows), 13, 16, 16), generator=g)   # fixed noise

    @torch.no_grad()
    def run(self, model, conditioner=None, autocast_dtype=None) -> dict:
        was = model.training
        model.eval()
        sums = {k: [0.0, 0] for k in range(self.nb)}
        null_sums = {k: [0.0, 0] for k in range(self.nb)}
        simplex = {k: [0.0, 0] for k in range(self.nb)}
        for s in range(0, len(self.rows), self.batch):
            sl = slice(s, s + self.batch)
            rows, t, eps, bk = self.rows[sl], self.t[sl].to(self.device), self.eps[sl].to(self.device), self.bucket[sl]
            x0 = self.d.onehot(rows)
            xt = self.schedule.q_sample(x0, t, eps)
            variants = [("cond", [self.d.captions[r] for r in rows.tolist()])]
            if conditioner is not None:
                variants.append(("null", [""] * len(rows)))
            for name, caps in variants:
                if conditioner is not None:
                    ctx, mask = conditioner(caps)
                else:
                    ctx = mask = None
                if autocast_dtype is not None:
                    with torch.autocast(self.device if isinstance(self.device, str) else self.device.type, dtype=autocast_dtype):
                        out = model(xt, t, ctx, mask).float()
                else:
                    out = model(xt, t, ctx, mask).float()
                per_item = ((out - eps) ** 2).mean(dim=(1, 2, 3)).cpu()
                tgt = sums if name == "cond" else null_sums
                for k in range(self.nb):
                    m = bk == k
                    tgt[k][0] += per_item[m].sum().item()
                    tgt[k][1] += int(m.sum())
                if name == "cond":
                    x0_hat = self.schedule.predict_x0_from_eps(xt.float(), t, out)
                    dev = (x0_hat.sum(1) + 11).abs().mean(dim=(1, 2)).cpu()
                    for k in range(self.nb):
                        m = bk == k
                        simplex[k][0] += dev[m].sum().item()
                        simplex[k][1] += int(m.sum())
        res = {f"val/loss_t{k}": sums[k][0] / max(sums[k][1], 1) for k in range(self.nb)}
        res["val/loss"] = sum(v[0] for v in sums.values()) / max(sum(v[1] for v in sums.values()), 1)
        for k in range(self.nb):
            res[f"val/simplex_dev_t{k}"] = simplex[k][0] / max(simplex[k][1], 1)
            if conditioner is not None:
                res[f"val/gain_t{k}"] = null_sums[k][0] / max(null_sums[k][1], 1) - res[f"val/loss_t{k}"]
        model.train(was)
        return res
