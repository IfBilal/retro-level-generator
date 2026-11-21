"""CUDA-graph U-Net replay (diffusion/graph.py): equivalence to the plain forward, thread safety, fallbacks.  GPU tests skip cleanly without CUDA."""
import threading

import numpy as np
import pytest
import torch
from conftest import ROOT

from text2mario.diffusion import graph as G

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
needs_model = pytest.mark.skipif(not (ROOT / "app/model/t2m_ema.safetensors").exists(), reason="no exported model bundle")


def tiny_unet():
    """A small module with the U-Net call signature (x, t, ctx, mask), cheap enough for a unit test."""
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.c = torch.nn.Conv2d(13, 13, 3, padding=1)
            self.p = torch.nn.Linear(8, 13)

        def forward(self, x, t, ctx, mask):
            cond = (self.p(ctx) * mask.unsqueeze(-1)).sum(1)                       # padded (masked) tokens must not matter
            return self.c(x) + cond[:, :, None, None] + t.float()[:, None, None, None] * 0.001

    return Tiny().eval()


def test_cpu_input_uses_the_plain_forward():
    u = tiny_unet()
    gu = G.GraphedUNet(u)
    x, t = torch.randn(2, 13, 16, 16), torch.tensor([3, 3])
    ctx, mask = torch.randn(2, 5, 8), torch.ones(2, 5, dtype=torch.bool)
    assert torch.equal(gu(x, t, ctx, mask), u(x, t, ctx, mask))                    # CPU input: never graphed
    assert gu._graphs == {} and not gu.disabled
    assert gu.c is u.c                                                              # attribute access is forwarded to the wrapped model


@needs_cuda
def test_replay_matches_the_plain_forward_and_ignores_masked_padding():
    u = tiny_unet().cuda()
    gu = G.GraphedUNet(u, ctx_len=12)
    x, t = torch.randn(4, 13, 16, 16, device="cuda"), torch.tensor([7, 7, 7, 7], device="cuda")
    ctx, mask = torch.randn(4, 5, 8, device="cuda"), torch.ones(4, 5, dtype=torch.bool, device="cuda")
    with torch.no_grad():
        ref = u(x, t, ctx, mask)
        for _ in range(3):                                                         # first call captures, later calls replay
            out = gu(x, t, ctx, mask)
            assert torch.allclose(out, ref, atol=1e-5)
        assert not gu.disabled and len(gu._graphs) == 1
        x2 = torch.randn_like(x)
        assert torch.allclose(gu(x2, t, ctx, mask), u(x2, t, ctx, mask), atol=1e-5)   # new inputs through the same graph
        long_ctx, long_mask = torch.randn(4, 20, 8, device="cuda"), torch.ones(4, 20, dtype=torch.bool, device="cuda")
        assert torch.allclose(gu(x, t, long_ctx, long_mask), u(x, t, long_ctx, long_mask), atol=1e-5)   # longer than ctx_len -> plain forward


@needs_cuda
def test_a_capture_failure_falls_back_for_good_and_never_fails_the_request(monkeypatch):
    u = tiny_unet().cuda()
    gu = G.GraphedUNet(u)
    x, t = torch.randn(2, 13, 16, 16, device="cuda"), torch.tensor([1, 1], device="cuda")
    ctx, mask = torch.randn(2, 5, 8, device="cuda"), torch.ones(2, 5, dtype=torch.bool, device="cuda")

    def boom(*a, **k):
        raise RuntimeError("capture not supported")

    monkeypatch.setattr(torch.cuda, "CUDAGraph", boom)
    with torch.no_grad():
        out = gu(x, t, ctx, mask)
        assert gu.disabled and torch.allclose(out, u(x, t, ctx, mask), atol=1e-6)
        assert torch.allclose(gu(x, t, ctx, mask), out)                             # stays on the plain path


@needs_cuda
def test_concurrent_callers_do_not_corrupt_each_other():
    """Two Streamlit sessions share one service: replays must be serialised (the graph's static buffers are shared)."""
    u = tiny_unet().cuda()
    gu = G.GraphedUNet(u)
    ctx, mask = torch.randn(1, 5, 8, device="cuda"), torch.ones(1, 5, dtype=torch.bool, device="cuda")
    t = torch.tensor([2], device="cuda")
    xs = [torch.randn(1, 13, 16, 16, device="cuda") for _ in range(8)]
    with torch.no_grad():
        want = [u(x, t, ctx, mask) for x in xs]
        got = [None] * 8
        errs = []

        def work(i):
            try:
                for _ in range(40):
                    out = gu(xs[i], t, ctx, mask)
                    if not torch.allclose(out, want[i], atol=1e-5):
                        errs.append(i)
                        return
                got[i] = True
            except Exception as e:  # noqa: BLE001
                errs.append(repr(e))

        threads = [threading.Thread(target=work, args=(i,)) for i in range(8)]
        [th.start() for th in threads]
        [th.join() for th in threads]
    assert not errs and all(got)


@needs_cuda
@needs_model
def test_real_unet_graph_samples_are_equivalent_to_eager_samples():
    """Same seed, same maths: windows agree cell for cell almost everywhere (bf16 kernel choices may flip a few cells); replay itself is deterministic."""
    from text2mario.app.service import LevelService
    from text2mario.generate import full_level as FL
    svc = LevelService(device="cuda")
    known = np.zeros((16, 8), np.uint8)
    old = FL._C.get("cuda_graph")
    try:
        FL._C["cuda_graph"] = False
        eager = [FL.sample_window(svc.bundle, "full floor. one pipe.", 7.5, 25, 0, 100 + i, known, "cuda") for i in range(8)]
        FL._C["cuda_graph"] = True
        graph = [FL.sample_window(svc.bundle, "full floor. one pipe.", 7.5, 25, 0, 100 + i, known, "cuda") for i in range(8)]
        again = [FL.sample_window(svc.bundle, "full floor. one pipe.", 7.5, 25, 0, 100 + i, known, "cuda") for i in range(8)]
    finally:
        FL._C["cuda_graph"] = old
    agree = float(np.mean([np.mean(a == b) for a, b in zip(eager, graph)]))
    assert agree > 0.97, agree
    assert all(np.array_equal(a, b) for a, b in zip(graph, again)), "graph replay itself must be deterministic"


def test_unet_helper_only_graphs_on_cuda_with_the_flag(monkeypatch):
    from text2mario.generate import full_level as FL
    u = tiny_unet()
    assert FL._unet(u, "cpu") is u
    monkeypatch.setitem(FL._C, "cuda_graph", False)
    assert FL._unet(u, "cuda") is u
    monkeypatch.setitem(FL._C, "cuda_graph", True)
    assert isinstance(FL._unet(u, "cuda"), G.GraphedUNet) and FL._unet(u, "cuda") is FL._unet(u, "cuda")    # one wrapper per U-Net


@needs_cuda
def test_capture_under_autocast_survives_the_autocast_cache_being_cleared():
    """Regression (found when the UI tests left CUDA in a sticky 'illegal memory access'): capturing under autocast with its weight cache ON puts the cached
    bf16 weights into the capture pool, and they are freed when the caller's autocast block exits.  The capture now disables that cache."""
    u = tiny_unet().cuda()
    gu = G.GraphedUNet(u)
    x, t = torch.randn(2, 13, 16, 16, device="cuda"), torch.tensor([4, 4], device="cuda")
    ctx, mask = torch.randn(2, 5, 8, device="cuda"), torch.ones(2, 5, dtype=torch.bool, device="cuda")
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16):                          # exactly how guidance.guided_eps calls the model (cache on)
            ref = u(x, t, ctx, mask)
            first = gu(x, t, ctx, mask)                                              # captured in here
        junk = [torch.randn(1 << 20, device="cuda") for _ in range(48)]             # churn the allocator: freed memory would be reused and corrupt a stale graph
        with torch.autocast("cuda", dtype=torch.bfloat16):
            second = gu(x, t, ctx, mask)
        torch.cuda.synchronize()
        del junk
    assert torch.allclose(first.float(), ref.float(), atol=3e-2) and torch.allclose(first.float(), second.float())
