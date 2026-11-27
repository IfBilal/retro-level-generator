"""MiniLM with its last k layers trainable (audit2 H5 variants b/d).

Explain-me:
  What: like TextConditioner, but the top `unfreeze_layers` transformer layers of MiniLM are trained together with the adapter (small LR from
        the optimiser's adapter group), so the encoder itself can learn to separate "one pipe" from "two pipes".  No string cache (the weights
        change every step); the encoder runs in fp32 (it is tiny) and the lower layers stay frozen.
  In/Out: same interface as TextConditioner.  Checkpoints store adapter + the tuned layers under adapter.*; the probe is taken from the PRETRAINED
          encoder at construction, so the "right MiniLM revision" check still works.
  Previous: text/conditioner.py.  Next: UNet cross-attention.   Try: `text.unfreeze_layers: 1` in configs/ablations/text_minilm_tuned.yaml.
"""
from __future__ import annotations

import torch
from torch import nn

from text2mario.text.conditioner import TextConditioner


class _Tuned(nn.Module):
    def __init__(self, proj: nn.Module, layers: nn.ModuleList):
        super().__init__()
        self.proj, self.layers = proj, layers


class TunedConditioner(TextConditioner):
    def __init__(self, encoder_dir, max_len, adapter_hidden, context_dim, device="cpu", unfreeze_layers: int = 2):
        super().__init__(encoder_dir, max_len, adapter_hidden, context_dim, 1, device, compute_dtype=torch.float32)
        self._probe0 = super().probe()                              # pretrained encoder, before anything is trained
        top = self.encoder.encoder.layer[-unfreeze_layers:]
        for p in top.parameters():
            p.requires_grad_(True)
        self.unfreeze_layers = unfreeze_layers
        self.adapter = _Tuned(self.adapter, top)                    # the trainer/EMA/checkpoints see proj + the tuned layers as "adapter"

    def forward(self, captions: list[str], dtype: torch.dtype = torch.float32):
        self._token_len_check(list(dict.fromkeys(captions)))
        enc = self.tok(captions, padding="longest", return_tensors="pt", truncation=False).to(self.device_)
        hs = self.encoder(**enc).last_hidden_state                 # gradients flow into the unfrozen top layers
        ctx = self.adapter.proj(hs)
        return ctx.to(dtype), enc["attention_mask"].bool()

    def probe(self) -> torch.Tensor:
        return self._probe0.clone()

    def trainable_parameters(self):
        return self.adapter.parameters()


class _Both(nn.Module):
    def __init__(self, tuned: nn.Module, struct: nn.Module):
        super().__init__()
        self.tuned, self.struct = tuned, struct


class BothConditioner(nn.Module):
    """H5 variant (d): the tuned-MiniLM tokens AND the structured (concept, quantity) tokens, concatenated as the cross-attention context."""

    def __init__(self, encoder_dir, max_len, adapter_hidden, context_dim, device="cpu", unfreeze_layers: int = 2):
        super().__init__()
        from text2mario.text.structured import StructuredConditioner
        self.t = TunedConditioner(encoder_dir, max_len, adapter_hidden, context_dim, device, unfreeze_layers)
        self.s = StructuredConditioner(adapter_hidden, context_dim, device)
        self.adapter = _Both(self.t.adapter, self.s.adapter)
        self.cache_stats = self.t.cache_stats
        self.revision = self.t.revision

    def forward(self, captions: list[str], dtype: torch.dtype = torch.float32):
        a, ma = self.t(captions, dtype)
        b, mb = self.s(captions, dtype)
        return torch.cat([a, b], 1), torch.cat([ma, mb], 1)

    def null(self, batch_size: int, dtype: torch.dtype = torch.float32):
        ctx, mask = self([""], dtype)
        return ctx.expand(batch_size, -1, -1), mask.expand(batch_size, -1)

    def train(self, mode: bool = True):
        super().train(mode)
        self.t.encoder.eval()
        return self

    def trainable_parameters(self):
        return self.adapter.parameters()

    def probe(self) -> torch.Tensor:
        return self.t.probe()

    def state_dict_for_ckpt(self) -> dict:
        return {"adapter": {k: v.detach().cpu() for k, v in self.adapter.state_dict().items()},
                "encoder": "sentence-transformers/all-MiniLM-L6-v2 (top layers tuned) + structured", "revision": self.revision, "probe": self.probe()}

    def check_probe(self, saved_probe: torch.Tensor, threshold: float = 0.999) -> None:
        self.t.check_probe(saved_probe, threshold)
