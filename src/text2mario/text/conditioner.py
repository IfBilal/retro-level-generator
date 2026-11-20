"""Text path: frozen MiniLM token states -> trainable adapter -> context for cross-attention (doc 15, ledger M7/M8).

Explain-me:
  What: turns caption strings into ctx (B, L, 128) + mask (B, L).  MiniLM is frozen (no gradients, not in the
        optimiser); only the 66,048-parameter adapter learns.  Encoded token states are cached per string
        (byte-capped LRU).  The null caption "" is just [CLS][SEP] run through the same path (classifier-free guidance).
  In/Out: TextConditioner(...)(['full floor. two pipes.', '']) -> (ctx, mask).  Previous: dataset gives captions;
          next: UNet cross-attention reads ctx/mask.
  Safety: a probe sentence's token states are stored in checkpoints; load checks they still match, so a different
          MiniLM revision or a corrupted cache on another machine is caught immediately.
  Try: print(cond.cache_stats) after a few batches, or change adapter_hidden in the config.
"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[3]
PROBE = "full floor. two pipes. a few enemies."


class TextConditioner(nn.Module):
    def __init__(self, encoder_dir: str | Path = "models/minilm", max_len: int = 48, adapter_hidden: int = 128,
                 context_dim: int = 128, cache_max_mb: int = 512, device: str | torch.device = "cpu",
                 compute_dtype: torch.dtype | None = None):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer
        path = Path(encoder_dir)
        path = path if path.is_absolute() else ROOT / path
        if not (path / "model.safetensors").exists():
            raise FileNotFoundError(f"MiniLM not found at {path}; run `python scripts/fetch_models.py` once (needs network)")
        self.device_ = torch.device(device)
        self.max_len = max_len
        # bf16 on GPU (H5); fp32 on CPU, where bf16 matmuls are slow
        self.enc_dtype = compute_dtype or (torch.bfloat16 if self.device_.type == "cuda" else torch.float32)
        self.tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.encoder = AutoModel.from_pretrained(path, local_files_only=True).to(self.enc_dtype).eval()
        for p in self.encoder.parameters():
            p.requires_grad_(False)                       # frozen: never in the optimiser, never in checkpoints
        self.encoder.to(self.device_)
        self.adapter = nn.Sequential(nn.Linear(384, adapter_hidden), nn.SiLU(), nn.Linear(adapter_hidden, context_dim),
                                     nn.LayerNorm(context_dim))
        self.adapter.to(self.device_)
        self._cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._cache_bytes, self._cache_cap = 0, cache_max_mb * 1024 * 1024
        self.cache_stats = {"hits": 0, "misses": 0, "evictions": 0}
        self.revision = None
        mf = ROOT / "data" / "MANIFEST.json"
        if mf.exists():
            import json
            self.revision = json.loads(mf.read_text()).get("sources", {}).get("MiniLM", {}).get("hf_commit")

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()                               # the frozen encoder never leaves eval mode
        return self

    # ---------------------------------------------------------------- frozen encoding with cache
    def _token_len_check(self, captions: list[str]) -> None:
        lens = [len(x) for x in self.tok(captions, add_special_tokens=True, truncation=False)["input_ids"]]
        if max(lens) > self.max_len:
            raise ValueError(f"caption needs {max(lens)} tokens > max_len {self.max_len}; raise text.max_len, never truncate silently")

    @torch.no_grad()
    def encode_frozen(self, captions: list[str]) -> list[torch.Tensor]:
        """List of (L_i, 384) token states (CPU, encoder dtype), via the string-keyed LRU cache."""
        misses = [c for c in dict.fromkeys(captions) if c not in self._cache]
        self.cache_stats["hits"] += len(captions) - sum(c in set(misses) for c in captions)
        self.cache_stats["misses"] += sum(c in set(misses) for c in captions)
        if misses:
            self._token_len_check(misses)
            enc = self.tok(misses, padding="longest", return_tensors="pt", truncation=False).to(self.device_)
            hs = self.encoder(**enc).last_hidden_state                  # (M, L, 384): token states, no pooling
            for i, c in enumerate(misses):
                n = int(enc["attention_mask"][i].sum())
                v = hs[i, :n].detach().cpu()
                self._cache[c] = v
                self._cache_bytes += v.numel() * v.element_size()
            while self._cache_bytes > self._cache_cap and len(self._cache) > 1:
                _, old = self._cache.popitem(last=False)
                self._cache_bytes -= old.numel() * old.element_size()
                self.cache_stats["evictions"] += 1
        out = []
        for c in captions:
            self._cache.move_to_end(c)
            out.append(self._cache[c])
        return out

    # ---------------------------------------------------------------- trainable part
    def forward(self, captions: list[str], dtype: torch.dtype = torch.float32):
        """-> ctx (B, L, 128), mask (B, L) bool (True = real token).  '' is a valid caption (the null)."""
        states = self.encode_frozen(captions)
        L = max(s.shape[0] for s in states)
        x = torch.zeros(len(states), L, 384, dtype=states[0].dtype)
        mask = torch.zeros(len(states), L, dtype=torch.bool)
        for i, s in enumerate(states):
            x[i, : s.shape[0]] = s
            mask[i, : s.shape[0]] = True
        x, mask = x.to(self.device_, non_blocking=True), mask.to(self.device_, non_blocking=True)
        ctx = self.adapter(x.to(self.adapter[0].weight.dtype))          # applied per token, pads included (masked later)
        return ctx.to(dtype), mask

    def null(self, batch_size: int, dtype: torch.dtype = torch.float32):
        ctx, mask = self([""], dtype)
        return ctx.expand(batch_size, -1, -1), mask.expand(batch_size, -1)

    def trainable_parameters(self):
        return self.adapter.parameters()

    # ---------------------------------------------------------------- checkpoint safety
    @torch.no_grad()
    def probe(self) -> torch.Tensor:
        return self.encode_frozen([PROBE])[0].float().clone()

    def state_dict_for_ckpt(self) -> dict:
        """Adapter weights + encoder identity + probe vector.  NEVER MiniLM's own weights."""
        return {"adapter": {k: v.detach().cpu() for k, v in self.adapter.state_dict().items()},
                "encoder": "sentence-transformers/all-MiniLM-L6-v2", "revision": self.revision, "probe": self.probe()}

    def check_probe(self, saved_probe: torch.Tensor, threshold: float = 0.999) -> None:
        cur = self.probe()
        if cur.shape != saved_probe.shape:
            raise RuntimeError("text encoder probe has a different token count: tokenizer/revision mismatch")
        cos = F.cosine_similarity(cur, saved_probe.float(), dim=-1)
        if cos.min() < threshold:
            raise RuntimeError(f"text encoder probe mismatch (min cosine {cos.min():.4f} < {threshold}): wrong MiniLM revision?")
