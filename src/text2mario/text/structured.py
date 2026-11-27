"""Count-aware structured conditioning (audit2 H5 variant c): caption phrases -> learned (concept, quantity) embeddings, no MiniLM.

Explain-me:
  What: the captions come from a closed grammar (62 phrases), so each phrase is parsed to (concept, code) and embedded as
        E_concept[concept] + E_code[code] (+ a small MLP).  The quantity code is shared by all concepts, so "one" means one
        for pipes and for enemies alike, which a free-text encoder has to learn from 62 strings.  The null caption "" is one learned token.
  In/Out: StructuredConditioner(...)(['full floor. two pipes.', '']) -> (ctx (B, L, 128), mask (B, L)), the same interface as TextConditioner.
  Previous: captions.grammar.parse_phrase.  Next: UNet cross-attention reads ctx/mask.
  Try: set `text.kind: structured` in a config overlay (configs/ablations/text_structured.yaml) and train; `adapter_hidden` is the MLP width.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from text2mario.captions.grammar import CONCEPTS, TOPICS_18, parse_phrase

PROBE = "full floor. two pipes. a few enemies."
_CONCEPTS = list(TOPICS_18)                 # concept -> index; the last index is reserved for "unknown phrase"
_UNK = len(_CONCEPTS)
_N_CODES = 12                               # floor codes 1-11, counted buckets 1-5; index 0 is "no code" (null / unknown)


class _Adapter(nn.Module):
    """Holds every trainable tensor, so the trainer's `{"adapter": cond.adapter}` and the EMA/checkpoint code work unchanged."""

    def __init__(self, hidden: int, context_dim: int):
        super().__init__()
        self.concept = nn.Embedding(len(_CONCEPTS) + 2, hidden)         # + unknown + null
        self.code = nn.Embedding(_N_CODES + 1, hidden)
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(hidden, context_dim), nn.LayerNorm(context_dim))

    def forward(self, concept_ids: torch.Tensor, code_ids: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.concept(concept_ids) + self.code(code_ids))


class StructuredConditioner(nn.Module):
    def __init__(self, adapter_hidden: int = 128, context_dim: int = 128, device: str | torch.device = "cpu", **_ignored):
        super().__init__()
        self.device_ = torch.device(device)
        self.adapter = _Adapter(adapter_hidden, context_dim).to(self.device_)
        self.cache_stats = {"hits": 0, "misses": 0, "evictions": 0}
        self.revision = None

    def ids(self, caption: str) -> list[tuple[int, int]]:
        """Caption -> [(concept index, code)]; '' -> one null token; a phrase outside the grammar -> the unknown concept."""
        phrases = [p.strip() for p in caption.split(".") if p.strip()]
        if not phrases:
            return [(_UNK + 1, 0)]
        out = []
        for p in phrases:
            try:
                c, code = parse_phrase(p)
                out.append((_CONCEPTS.index(c) if c in _CONCEPTS else _UNK, code))
            except ValueError:
                out.append((_UNK, 0))
        return out

    def forward(self, captions: list[str], dtype: torch.dtype = torch.float32):
        rows = [self.ids(c) for c in captions]
        L = max(len(r) for r in rows)
        cid = torch.zeros(len(rows), L, dtype=torch.long)
        kid = torch.zeros(len(rows), L, dtype=torch.long)
        mask = torch.zeros(len(rows), L, dtype=torch.bool)
        for i, r in enumerate(rows):
            for j, (c, k) in enumerate(r):
                cid[i, j], kid[i, j], mask[i, j] = c, k, True
        ctx = self.adapter(cid.to(self.device_), kid.to(self.device_))
        return ctx.to(dtype), mask.to(self.device_)

    def null(self, batch_size: int, dtype: torch.dtype = torch.float32):
        ctx, mask = self([""], dtype)
        return ctx.expand(batch_size, -1, -1), mask.expand(batch_size, -1)

    def trainable_parameters(self):
        return self.adapter.parameters()

    @torch.no_grad()
    def probe(self) -> torch.Tensor:
        return torch.tensor(self.ids(PROBE), dtype=torch.float32)       # the parse of the probe caption: ids, not activations

    def state_dict_for_ckpt(self) -> dict:
        return {"adapter": {k: v.detach().cpu() for k, v in self.adapter.state_dict().items()}, "encoder": "structured", "revision": None,
                "probe": self.probe()}

    def check_probe(self, saved_probe: torch.Tensor, threshold: float = 0.999) -> None:
        cur = self.probe()
        if cur.shape != saved_probe.shape or not torch.equal(cur, saved_probe.float()):
            raise RuntimeError("structured conditioner probe mismatch: the caption grammar changed since this checkpoint was trained")


assert len(CONCEPTS) <= len(_CONCEPTS) and F is not None
