"""CUDA-graph replay of the U-Net forward (inference only, optional): removes the kernel-launch overhead of ~300 small modules per forward.

Explain-me:
  What: a sampling step is latency-bound on a GPU (batch 1-64 windows of 16 x 16 cost about the same).  This wrapper captures the U-Net forward ONCE per
        input shape in a CUDA graph and replays it, which keeps every operation identical and only skips Python/launch overhead.  It is used by the app's
        window sampler (generate/full_level.py); training and the evaluation scripts never use it.
  In/Out: GraphedUNet(unet)(x, t, ctx, mask) -> same tensor as unet(x, t, ctx, mask); falls back to the plain call for other signatures, on CPU, or on any capture error.
  Previous: models/unet.py.  Next: diffusion/sampler.py (calls it like the model).
  Try: configs/generate/app.yaml -> cuda_graph: false restores the plain forward; tests/test_cuda_graph.py checks equivalence, determinism, thread safety and the fallback.
"""
from __future__ import annotations

import threading

import torch


class GraphedUNet:
    def __init__(self, unet, ctx_len: int = 48):
        self.unet = unet
        self.ctx_len = ctx_len
        self._graphs: dict = {}
        self.disabled = False
        self._lock = threading.Lock()                 # the static buffers are shared: two Streamlit sessions must not replay at the same time

    def __getattr__(self, name):                      # anything else the sampler might read from the model
        if name == "unet":                             # not set yet (copy / pickle): do not recurse
            raise AttributeError(name)
        return getattr(self.unet, name)

    def _pad(self, ctx: torch.Tensor, mask: torch.Tensor):
        extra = self.ctx_len - ctx.shape[1]
        if extra < 0:
            return None
        if extra:
            ctx = torch.nn.functional.pad(ctx, (0, 0, 0, extra))
            mask = torch.nn.functional.pad(mask, (0, extra), value=False)       # padding is masked, exactly as guidance._pad_len does
        return ctx, mask

    def __call__(self, x, t, *rest):
        if self.disabled or len(rest) != 2 or not x.is_cuda:
            return self.unet(x, t, *rest)
        padded = self._pad(*rest)
        if padded is None:
            return self.unet(x, t, *rest)
        ctx, mask = padded
        key = (tuple(x.shape), x.dtype, tuple(ctx.shape), torch.is_autocast_enabled(), torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled() else None)
        with self._lock:
            entry = self._graphs.get(key)
            if entry is None:
                try:
                    entry = self._capture(x, t, ctx, mask)
                except Exception:                     # noqa: BLE001 - any capture problem: use the plain forward from now on, never fail a request
                    self.disabled = True
                    return self.unet(x, t, *rest)
                self._graphs[key] = entry
            graph, sx, st, sc, sm, out = entry
            sx.copy_(x)
            st.copy_(t)
            sc.copy_(ctx)
            sm.copy_(mask)
            graph.replay()
            return out.clone()

    def _capture(self, x, t, ctx, mask):
        sx, st, sc, sm = x.clone(), t.clone(), ctx.clone(), mask.clone()
        ac = torch.is_autocast_enabled()
        # With autocast ON, PyTorch caches the bf16 copies of the weights; those copies would be allocated INSIDE the capture pool and freed when the
        # caller's autocast block exits, leaving the graph pointing at freed memory (illegal memory access on a later replay).  Capture with the cache off.
        with torch.autocast("cuda", dtype=torch.get_autocast_dtype("cuda") if ac else torch.bfloat16, enabled=ac, cache_enabled=False):
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):                    # warm-up so cuDNN/cuBLAS pick their algorithms before capture
                    self.unet(sx, st, sc, sm)
            torch.cuda.current_stream().wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = self.unet(sx, st, sc, sm)
        return graph, sx, st, sc, sm, out


_CACHE: dict = {}


def graphed(unet) -> GraphedUNet:
    """One wrapper (and its captured graphs) per U-Net instance."""
    g = _CACHE.get(id(unet))
    if g is None or g.unet is not unet:
        g = _CACHE[id(unet)] = GraphedUNet(unet)
    return g
