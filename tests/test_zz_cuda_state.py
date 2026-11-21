"""Sentinel: the LAST test module alphabetically.  If any earlier test (or the app code it drives) left the CUDA context in an error state, this fails.
It exists because a CUDA graph captured under autocast with its weight cache on (diffusion/graph.py) freed memory the graph still used: nothing failed
until a later, unrelated test touched the GPU."""
import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no cuda")
def test_cuda_still_healthy_after_all_previous_tests():
    torch.cuda.synchronize()
    a = torch.zeros(10, device="cuda")
    m = torch.nn.Conv2d(13, 13, 3).cuda()
    assert float(a.sum()) == 0.0 and m.weight.is_cuda
