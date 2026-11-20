"""The trainer: one step = corrupt a real window with noise, predict the noise, nudge the weights (doc 16).

Explain-me (for the viva): every step takes 64 real 16x16 windows, picks a random noise strength t per window,
  adds that much noise (q_sample), asks the U-Net to predict the noise (given the caption's word vectors, or a
  learned 'null' in Phase 3), measures the mean-squared error and updates the weights with AdamW.  A slow-moving
  average of the weights (EMA) is kept; evaluation and export use the EMA copy.
  In/Out: Trainer(cfg, run_dir, device).run() writes metrics.csv, events.jsonl, step_*.pt checkpoints, EMA safetensors.
  Previous: dataset.py (batches), scheduler.py (noise).  Next: validate.py / evaluation on the EMA weights.
  Try: --set optim.lr=1e-3 or --set loss.aux_ce_lambda=0.001 (drill 13); see CLAUDE.md for the rules a run must obey.
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import torch

from text2mario.data.augment import AugConfig
from text2mario.data.dataset import DataConfig, MarioWindows
from text2mario.data.sampler import WindowSampler, mix
from text2mario.diffusion import losses
from text2mario.diffusion.scheduler import DiffusionSchedule
from text2mario.models.ema import EMA
from text2mario.models.factory import build_unet
from text2mario.train import checkpoint as ckpt
from text2mario.train.optim import build_optimizer, lr_at
from text2mario.utils.config import config_hash


class Trainer:
    def __init__(self, cfg: dict, run_dir: str | Path, device: str = "cpu", dataset: MarioWindows | None = None):
        self.cfg, self.run_dir, self.device = cfg, Path(run_dir), torch.device(device)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.autocast = torch.bfloat16 if (self.device.type == "cuda" and cfg["train"]["precision"] == "bf16") else None
        self.conditional = bool(cfg["text"]["enabled"])
        a = cfg["aug"]
        self.data = dataset or MarioWindows("train", device=device, cfg=DataConfig(
            seed=cfg["seed"], max_len=cfg["data"]["max_len"], aug=AugConfig(**a)))
        self.sampler = WindowSampler(len(self.data), cfg["train"]["batch_size"], cfg["seed"])
        self.schedule = DiffusionSchedule(cfg["diffusion"]["schedule"], cfg["diffusion"]["T"], self.device)
        self.unet = build_unet(cfg["model"], conditional=self.conditional).to(self.device)
        self.cond = None
        if self.conditional:
            from text2mario.text.conditioner import TextConditioner
            self.cond = TextConditioner(cfg["text"]["encoder_dir"], cfg["data"]["max_len"], cfg["text"]["adapter_hidden"],
                                        cfg["model"]["context_dim"], cfg["text"]["cache_max_mb"], self.device)
        self.modules = {"unet": self.unet, **({"adapter": self.cond.adapter} if self.cond else {})}
        self.opt = build_optimizer(self.modules, cfg)
        self.ema = EMA(self.modules, cfg["ema"]["decay"])
        self.step = 0
        self.micro_batch = cfg["train"]["batch_size"]       # reduced automatically (and logged) only after an OOM
        self.nonfinite_streak = 0
        self.chash = config_hash(cfg)
        self._csv_header_written = (self.run_dir / "metrics.csv").exists()

    # ------------------------------------------------------------------ logging
    def event(self, name: str, **kw) -> None:
        with open(self.run_dir / "events.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"t": time.time(), "step": self.step, "event": name, **kw}) + "\n")

    def log_metrics(self, row: dict) -> None:
        path = self.run_dir / "metrics.csv"
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if not self._csv_header_written:
                w.writeheader()
                self._csv_header_written = True
            w.writerow(row)

    # ------------------------------------------------------------------ one optimiser step
    def _loss(self, xt, t, eps, x0_grid, ctx, mask, schedule):
        if self.autocast:
            with torch.autocast(self.device.type, dtype=self.autocast):
                eps_hat = self.unet(xt, t, ctx, mask)
        else:
            eps_hat = self.unet(xt, t, ctx, mask)
        eps_hat = eps_hat.float()                               # loss maths always in float32 (doc 13 §14)
        lc = self.cfg["loss"]
        if lc["aux_ce_lambda"] > 0:
            return losses.loss_aux_ce(eps_hat, eps, xt, t, x0_grid, schedule, lc["aux_ce_lambda"], lc["aux_ce_tau"], lc["aux_ce_weighting"])
        if lc["type"] == "channel_weighted":
            return losses.loss_channel_weighted(eps_hat, eps)
        return losses.loss_simple(eps_hat, eps)

    def train_step(self) -> dict:
        cfg, step = self.cfg, self.step
        B = cfg["train"]["batch_size"]
        t0 = time.time()
        rows = self.sampler.rows(step)
        batch = self.data.batch(rows, step, uncond=not self.conditional)
        g = torch.Generator().manual_seed(mix(cfg["seed"], mix(step, 1)))        # CPU noise stream: machine independent
        t = torch.randint(0, cfg["diffusion"]["T"], (B,), generator=g).to(self.device)
        eps = torch.randn(batch.x0.shape, generator=g).to(self.device)
        xt = self.schedule.q_sample(batch.x0, t, eps)
        lr = lr_at(step, cfg)
        for grp in self.opt.param_groups:
            grp["lr"] = lr
        self.unet.train()
        self.opt.zero_grad(set_to_none=True)
        total = 0.0
        mb = self.micro_batch
        try:
            for s in range(0, B, mb):                                           # accumulation == full batch (GroupNorm is batch-independent)
                sl = slice(s, s + mb)
                ctx = mask = None
                if self.cond is not None:
                    ctx, mask = self.cond(batch.captions[sl])
                loss = self._loss(xt[sl], t[sl], eps[sl], batch.grid[sl], ctx, mask, self.schedule)
                if not torch.isfinite(loss):
                    return self._skip("loss")
                (loss * (xt[sl].shape[0] / B)).backward()
                total += loss.item() * xt[sl].shape[0] / B
        except torch.cuda.OutOfMemoryError:
            self.micro_batch = max(1, mb // 2)
            self.event("oom_fallback", micro_batch=self.micro_batch)             # no silent fallbacks (rule 12)
            self.opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            return {"skipped": 1, "loss": float("nan")}
        params = [p for m in self.modules.values() for p in m.parameters() if p.requires_grad]
        gnorm = torch.nn.utils.clip_grad_norm_(params, cfg["optim"]["grad_clip"])
        if not torch.isfinite(gnorm):
            return self._skip("grad")
        self.opt.step()
        self.ema.update(self.modules, step)
        self.nonfinite_streak = 0
        self.step += 1
        return {"loss": total, "lr": lr, "grad_norm": float(gnorm), "step_s": time.time() - t0, "skipped": 0}

    def _skip(self, what: str) -> dict:
        self.opt.zero_grad(set_to_none=True)
        self.nonfinite_streak += 1
        self.event("skipped_nonfinite", what=what)
        if self.nonfinite_streak >= self.cfg["train"]["max_nonfinite_skips"]:
            self.event("nan_abort")
            raise RuntimeError("nan_abort: 5 consecutive non-finite steps (doc 25 R-3.3)")
        return {"skipped": 1, "loss": float("nan")}

    # ------------------------------------------------------------------ checkpoints
    def state(self) -> dict:
        s = {"step": self.step, "unet": self.unet.state_dict(), "opt": self.opt.state_dict(), "ema": self.ema.state_dict(),
             "sampler": self.sampler.state_dict(), "rng": torch.get_rng_state(), "config_hash": self.chash,
             "config_json": json.dumps(self.cfg, sort_keys=True)}
        if self.cond is not None:
            s["text"] = self.cond.state_dict_for_ckpt()
        return s

    def save(self, name: str | None = None) -> Path:
        p = ckpt.save(self.state(), self.run_dir / (name or f"step_{self.step:07d}.pt"))
        ckpt.prune(self.run_dir, self.cfg["checkpoint"]["keep_last"])
        self.event("checkpoint", path=p.name)
        return p

    def resume(self, path: str | Path) -> None:
        s = ckpt.load(path)
        if s["config_hash"] != self.chash:
            raise RuntimeError("config hash differs from the checkpoint: refusing to resume (doc 16 §7)")
        self.sampler.check_resume(s["sampler"])
        if self.cond is not None:
            self.cond.check_probe(s["text"]["probe"])
            self.cond.adapter.load_state_dict(s["text"]["adapter"])
        self.unet.load_state_dict(s["unet"])
        self.opt.load_state_dict(s["opt"])
        self.ema.load_state_dict(s["ema"])
        torch.set_rng_state(s["rng"])
        self.step = s["step"]
        self.event("resumed", path=str(path))

    def export_ema(self, path: str | Path | None = None) -> Path:
        tensors = {k: v for k, v in self.ema.shadow.items()}
        meta = {"config_hash": self.chash, "step": self.step, "schedule": self.cfg["diffusion"]["schedule"]}
        return ckpt.export_safetensors(tensors, path or self.run_dir / "ema.safetensors", meta)

    # ------------------------------------------------------------------ main loop
    def run(self, max_steps: int | None = None) -> None:
        cfg = self.cfg
        N = max_steps or cfg["train"]["max_steps"]
        self.event("started", max_steps=N, config_hash=self.chash, conditional=self.conditional)
        while self.step < N:
            if (self.run_dir / "STOP").exists():                                 # graceful stop (doc 16 §1)
                self.save()
                self.event("stopped_by_file")
                return
            m = self.train_step()
            if self.step % cfg["log_every"] == 0 and not m.get("skipped"):
                self.log_metrics({"step": self.step, **{k: round(v, 6) if isinstance(v, float) else v for k, v in m.items()}})
            if self.step and self.step % cfg["checkpoint"]["every_steps"] == 0:
                self.save()
        self.save()
        self.export_ema()
        self.event("finished")
