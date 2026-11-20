"""Hardware profile, VRAM cap and the hardware report (ledger H2/H3, doc 07).

Explain-me:
  What: detects the machine profile (T2M_MACHINE env var, else GPU name, else cpu_dev), applies the 4.5 GiB
        per-process CUDA memory cap at start-up so anything that runs on the desktop also runs on the laptop,
        and writes a small hw_report.json.
  Try: T2M_MACHINE=laptop_4050 python -c "from text2mario.utils.hw import detect_machine; print(detect_machine())"
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch


def detect_machine() -> str:
    if os.environ.get("T2M_MACHINE"):
        return os.environ["T2M_MACHINE"]
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0).lower()
        if "3060" in name:
            return "desktop_3060ti"
        if "4050" in name:
            return "laptop_4050"
        return "other_gpu"
    return "cpu_dev"


def apply_vram_cap(cap_gib: float) -> float | None:
    """torch.cuda.set_per_process_memory_fraction(cap / total). Returns the fraction, or None on CPU."""
    if not torch.cuda.is_available():
        return None
    total = torch.cuda.get_device_properties(0).total_memory
    frac = min(1.0, cap_gib * 1024 ** 3 / total)
    torch.cuda.set_per_process_memory_fraction(frac, 0)
    return frac


def hw_report(path: str | Path, extra: dict | None = None) -> dict:
    rep = {"machine": detect_machine(), "torch": torch.__version__, "cuda": torch.version.cuda,
           "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", **(extra or {})}
    if torch.cuda.is_available():
        rep["vram_peak_allocated_gib"] = round(torch.cuda.max_memory_allocated() / 1024 ** 3, 3)
        rep["vram_peak_reserved_gib"] = round(torch.cuda.max_memory_reserved() / 1024 ** 3, 3)
    Path(path).write_text(json.dumps(rep, indent=2) + "\n")
    return rep
