"""Build the U-Net from a config dict and count parameters (doc 14 §9-10).

Explain-me:
  What: maps the `model:` section of the YAML config to UNet arguments; count_params() reports the pieces.
  In/Out: dict -> UNet.  Try: build_unet({'base_channels': 32}) for the half-width probe (X12).
"""
from __future__ import annotations

from text2mario.models.unet import UNet

_KEYS = {"in_channels", "base_channels", "channel_mult", "num_res_blocks_down", "groups", "dropout", "attn_levels",
         "self_attention", "cross_attention", "heads", "context_dim", "zero_init", "pooled_text_to_time", "null_ctx_tokens"}


def build_unet(model_cfg: dict | None = None, conditional: bool = True) -> UNet:
    cfg = dict(model_cfg or {})
    cfg.pop("num_res_blocks_up", None)       # must equal down + 1; derived
    unknown = set(cfg) - _KEYS
    if unknown:
        raise ValueError(f"unknown model config keys: {sorted(unknown)}")
    return UNet(**cfg, uncond=not conditional)


def count_params(model) -> dict:
    total = sum(p.numel() for p in model.parameters())
    return {"total": total, "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)}
