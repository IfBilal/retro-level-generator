"""Environment check (doc 06, timeline Day 0): versions, machine profile, numerics, VRAM cap, data and model presence.

Explain-me:
  What: prints torch/CUDA/GPU, runs a bf16 matmul + scaled-dot-product-attention smoke test, on a GPU applies the 4.5 GiB
        cap and proves it by a deliberate 5 GiB allocation that MUST fail, then checks that the pinned data, split LOCK
        and the local MiniLM copy are present and unchanged.  Exit code 0 = green.
  Try: python scripts/verify_env.py --quick   (skips the slow checks)
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from text2mario.utils.env import bootstrap_env

bootstrap_env()
import torch

from text2mario.utils.hw import apply_vram_cap, detect_machine

ok_all = True


def check(name, ok, detail=""):
    global ok_all
    ok_all &= bool(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    print(f"python {sys.version.split()[0]}  torch {torch.__version__}  cuda {torch.version.cuda}  machine profile: {detect_machine()}")
    check("python is 3.12", sys.version_info[:2] == (3, 12), f"({sys.version.split()[0]})")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        p = torch.cuda.get_device_properties(0)
        print(f"GPU {p.name}  {p.total_memory / 1024 ** 3:.1f} GiB  sm_{p.major}{p.minor}")
    else:
        print("no CUDA device: CPU profile (training and latency numbers must be measured on the GPU machines)")
    dt = torch.bfloat16 if dev == "cuda" else torch.float32
    x = torch.randn(256, 256, device=dev, dtype=dt)
    y = x @ x
    q = torch.randn(2, 4, 16, 32, device=dev, dtype=dt)
    a = torch.nn.functional.scaled_dot_product_attention(q, q, q)
    check(f"matmul + SDPA smoke test ({dt})", torch.isfinite(y).all() and torch.isfinite(a).all())
    if dev == "cuda" and not args.quick:
        apply_vram_cap(4.5)
        try:
            torch.empty(5 * 1024 ** 3, dtype=torch.uint8, device="cuda")
            check("VRAM cap: a 5 GiB allocation must fail", False, "(it succeeded: the cap is not applied)")
        except torch.cuda.OutOfMemoryError:
            check("VRAM cap: a 5 GiB allocation must fail", True)
    from text2mario.data import provenance, splits
    try:
        provenance.verify()
        splits.load(verify=True)
        check("data hashes match MANIFEST and split matches LOCK", True)
    except Exception as e:
        check("data hashes match MANIFEST and split matches LOCK", False, str(e))
    check("MiniLM local copy present", (ROOT / "models/minilm/model.safetensors").exists(), "(scripts/fetch_models.py)")
    if not args.quick:
        from text2mario.models.factory import build_unet, count_params
        n = count_params(build_unet())["total"]
        check("U-Net parameter count is 18,312,013", n == 18_312_013, f"({n})")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
