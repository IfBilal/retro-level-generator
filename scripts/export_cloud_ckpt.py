"""Export a small inference bundle for the cloud / the zip: EMA U-Net + adapter (safetensors), config, fp16 MiniLM (doc 28 §12).

Explain-me:
  What: reads a training checkpoint, writes app/model/{t2m_ema.safetensors, config.json, minilm/, EXPORT.json}.  No optimiser state,
        no raw (non-EMA) weights.  MiniLM is stored in fp16 with its tokenizer, so the bundle is about 120 MB and each file is
        below GitHub's 100 MB limit.  The export's sha256 and the source checkpoint's sha256 are recorded.
  In/Out: step_XXXX.pt -> app/model/.  Next: T2M_CKPT=app/model streamlit run app/streamlit_app.py (a directory = exported bundle).
  Try: python scripts/export_cloud_ckpt.py runs/C-main/step_0060000.pt
"""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import torch  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402

from text2mario.train import checkpoint as ckpt  # noqa: E402


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--out", default=str(ROOT / "app/model"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    s = ckpt.load(a.checkpoint)                                    # verifies the sha256 sidecar
    cfg = json.loads(s["config_json"])
    shadow = s["ema"]["shadow"]
    tensors = {k: v.detach().float().contiguous() for k, v in shadow.items() if k.startswith(("unet.", "adapter."))}
    tensors["probe"] = s["text"]["probe"].float().contiguous()
    save_file(tensors, str(out / "t2m_ema.safetensors"), metadata={"step": str(s["step"]), "ema": "true"})
    cfg["export"] = {"step": s["step"], "source_checkpoint": str(a.checkpoint)}
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    src = ROOT / cfg["text"]["encoder_dir"]
    dst = out / "minilm"
    dst.mkdir(exist_ok=True)
    for f in src.iterdir():
        if f.name == "model.safetensors":
            t16 = {k: v.half() if v.is_floating_point() else v for k, v in load_file(str(f)).items()}
            save_file(t16, str(dst / "model.safetensors"), metadata={"format": "pt"})
        elif f.is_file() and f.suffix in (".json", ".txt"):
            shutil.copy(f, dst / f.name)
    man = {"source_checkpoint_sha256": sha(Path(a.checkpoint)) if Path(a.checkpoint).exists() else None, "step": s["step"],
           "files": {p.name: sha(p) for p in (out / "t2m_ema.safetensors", out / "config.json", dst / "model.safetensors")},
           "sizes_mb": {p.name: round(p.stat().st_size / 1e6, 1) for p in (out / "t2m_ema.safetensors", dst / "model.safetensors")}}
    (out / "EXPORT.json").write_text(json.dumps(man, indent=1))
    print(json.dumps(man, indent=1))


if __name__ == "__main__":
    main()
