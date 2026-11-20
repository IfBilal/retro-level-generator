"""Config loading: base -> overlay -> machine -> CLI overrides, with guards (doc 17 §1).

Explain-me:
  What: merges YAML files, rejects unknown keys, refuses a machine YAML that changes anything except the four
        performance knobs, and computes the config hash that identifies a run (machine knobs excluded).
  In/Out: load_config('configs/uncond_cosine.yaml', machine='cpu_dev') -> plain dict; config_hash(cfg) -> hex.
  Try: put `lr: 1e-3` in a machine YAML and load it - it raises (rule: same hyperparameters on both GPUs).
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
MACHINE_ALLOWED = {"num_workers", "cudnn_benchmark", "compile", "log_every"}


class ConfigError(ValueError):
    pass


def _merge(base: dict, over: dict, path: str = "") -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if k not in base:
            raise ConfigError(f"unknown config key: {path}{k}")
        if isinstance(v, dict) and isinstance(base[k], dict):
            out[k] = _merge(base[k], v, f"{path}{k}.")
        else:
            out[k] = v
    return out


def _parse_value(text: str):
    """YAML value for a --set override.  PyYAML (YAML 1.1) reads `1e-3` as a STRING, so numbers are tried first."""
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return yaml.safe_load(text)


def _read(p: str | Path) -> dict:
    return yaml.safe_load(Path(p).read_text(encoding="utf-8")) or {}


def load_config(overlay: str | Path | None = None, machine: str | None = None, overrides: list[str] | None = None) -> dict:
    cfg = _read(ROOT / "configs" / "base.yaml")
    if overlay:
        cfg = _merge(cfg, _read(overlay))
    if machine:
        m = _read(ROOT / "configs" / "machine" / f"{machine}.yaml")
        illegal = set(m) - MACHINE_ALLOWED
        if illegal:
            raise ConfigError(f"machine YAML may only override {sorted(MACHINE_ALLOWED)}; found {sorted(illegal)}")
        cfg = _merge(cfg, m)
        cfg["machine"] = machine
    for item in overrides or []:                       # --set a.b=value (logged by the caller)
        key, val = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node[p]
        if parts[-1] not in node:
            raise ConfigError(f"unknown config key: {key}")
        node[parts[-1]] = _parse_value(val)
    return cfg


def config_hash(cfg: dict) -> str:
    """sha256 of the canonical JSON of the config, excluding machine-only knobs, `name` and the machine label."""
    c = {k: v for k, v in cfg.items() if k not in MACHINE_ALLOWED | {"machine", "name"}}
    return hashlib.sha256(json.dumps(c, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
