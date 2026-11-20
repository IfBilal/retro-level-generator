"""Download the frozen text encoder once, record its revision and file hashes (doc 08 §5, ledger V3).

Explain-me:
  What: snapshots sentence-transformers/all-MiniLM-L6-v2 (config, weights, tokenizer files) into models/minilm/
        and writes revision + sha256 hashes + licence into data/MANIFEST.json.  After this, everything loads with
        local_files_only=True, so training, evaluation and the app work with Wi-Fi off.
  Try: python scripts/fetch_models.py --check   (verify the local copy against the manifest, no network)
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = "sentence-transformers/all-MiniLM-L6-v2"
FILES = ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "vocab.txt", "special_tokens_map.json"]
DEST = ROOT / "models" / "minilm"


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    mpath = ROOT / "data" / "MANIFEST.json"
    manifest = json.loads(mpath.read_text())
    if args.check:
        rec = manifest["sources"]["MiniLM"]["files"]
        bad = [f for f, e in rec.items() if sha(DEST / f) != e["sha256_bytes"]]
        print("local MiniLM matches the manifest" if not bad else f"MISMATCH: {bad}")
        sys.exit(1 if bad else 0)
    from huggingface_hub import HfApi, snapshot_download
    info = HfApi().model_info(REPO)
    snapshot_download(REPO, revision=info.sha, allow_patterns=FILES, local_dir=DEST)
    manifest["sources"]["MiniLM"] = {
        "hf_repo": REPO, "hf_commit": info.sha, "license": (info.card_data or {}).get("license", "apache-2.0"),
        "files": {f: {"sha256_bytes": sha(DEST / f)} for f in FILES}, "source": "huggingface"}
    mpath.write_text(json.dumps(manifest, indent=2) + "\n")
    print("MiniLM revision", info.sha, "->", DEST)


if __name__ == "__main__":
    main()
