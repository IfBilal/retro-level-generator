"""Re-check source-data hashes against data/MANIFEST.json before any use (doc 08 §8.7).

Explain-me:
  What: verify() recomputes the content hash of each JSON we read and compares it with the manifest.
  In/Out: reads data/source/* + data/MANIFEST.json; raises on mismatch.  Called by the audit and the loaders.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def content_hash(path: Path) -> str:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    return hashlib.sha256(json.dumps(obj, separators=(",", ":"), sort_keys=True).encode("utf-8")).hexdigest()


def corpus_content_hash() -> str:
    """Content hash of the training corpus, as recorded in the manifest."""
    m = json.loads((ROOT / "data" / "MANIFEST.json").read_text(encoding="utf-8"))
    return m["sources"]["MarioDiffusion"]["files"]["datasets/Mar1and2_LevelsAndCaptions-regular.json"]["sha256_content"]


def verify(files=("datasets/Mar1and2_LevelsAndCaptions-regular.json",)) -> None:
    """Raise if a source file no longer matches the manifest (the content hash is the contract)."""
    m = json.loads((ROOT / "data" / "MANIFEST.json").read_text(encoding="utf-8"))
    for rel in files:
        entry = m["sources"]["MarioDiffusion"]["files"][rel]
        actual = content_hash(ROOT / entry["copied_to"])
        if actual != entry["sha256_content"]:
            raise RuntimeError(f"data hash mismatch for {rel}: {actual} != {entry['sha256_content']}")
