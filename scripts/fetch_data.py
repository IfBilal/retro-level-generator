"""Fetch the pinned MarioDiffusion + VGLC data, verify hashes, write data/MANIFEST.json (doc 08).

Explain-me:
  What: clones two third-party repos at pinned commits into data/raw/, copies the files we use into
        data/source/ (read-only), checks sha256 hashes against configs/data.yaml, writes the manifest.
  In/Out: configs/data.yaml -> data/source/*, data/MANIFEST.json.  Next step: scripts/audit_dataset.py.
  Safety: nothing from data/raw/ is ever imported or executed (CLAUDE.md rule 10); files are only read.
  Try: python scripts/fetch_data.py --dry-run
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import stat
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
SOURCE = ROOT / "data" / "source"


def sha256_bytes(path: Path) -> str:
    """Hash of the exact file bytes (provenance)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_content(path: Path) -> str:
    """Hash of the parsed JSON, robust to whitespace / line endings (doc 08 §3)."""
    with open(path, encoding="utf-8") as f:
        obj = json.load(f)
    return hashlib.sha256(json.dumps(obj, separators=(",", ":"), sort_keys=True).encode("utf-8")).hexdigest()


def git(*args: str, cwd: Path | None = None) -> str:
    out = subprocess.run(["git", "-c", "core.autocrlf=false", *args], cwd=cwd, check=True,
                         capture_output=True, text=True)
    return out.stdout.strip()


def ensure_clone(url: str, commit: str, dest: Path, sparse: list[str]) -> None:
    """Idempotent sparse clone pinned to `commit`; never continues on the wrong SHA."""
    if (dest / ".git").exists() and git("rev-parse", "HEAD", cwd=dest) == commit:
        print(f"  {dest.name}: already at {commit[:8]}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in (1, 2):  # one retry after 10 s (doc 08 §6)
        try:
            if not (dest / ".git").exists():
                git("clone", "-q", "--filter=blob:none", "--no-checkout", "--sparse", url, str(dest))
            git("sparse-checkout", "set", *sparse, cwd=dest)
            git("checkout", "-q", commit, cwd=dest)
            break
        except subprocess.CalledProcessError as e:
            if attempt == 2:
                sys.exit(f"exit 3: clone failed ({e.stderr}); see doc 08 §7 for the manual fallback")
            time.sleep(10)
    if git("rev-parse", "HEAD", cwd=dest) != commit:
        sys.exit("exit 4: wrong commit checked out")


def copy_readonly(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.chmod(stat.S_IWUSR | stat.S_IRUSR)
    shutil.copy2(src, dst)
    dst.chmod(0o444)  # read-only after fetch (doc 08 §8.8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-vglc", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / "configs" / "data.yaml").read_text())
    if args.dry_run:
        print("would clone", cfg["mariodiffusion_url"], "@", cfg["mariodiffusion_commit"])
        print("would copy", list(cfg["files"]))
        return

    manifest = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "created_by": "scripts/fetch_data.py", "sources": {}, "audit": {"passed": None}}

    # --- S-1: MarioDiffusion ---
    print("MarioDiffusion")
    md_dir = RAW / "MarioDiffusion"
    ensure_clone(cfg["mariodiffusion_url"], cfg["mariodiffusion_commit"], md_dir, ["datasets"])
    files = {}
    ok = True
    for rel, exp in cfg["files"].items():
        src = md_dir / rel
        if not src.exists():
            sys.exit(f"exit 5: missing {rel}")
        dst = SOURCE / Path(rel).name
        copy_readonly(src, dst)
        b = sha256_bytes(dst)
        entry = {"copied_to": str(dst.relative_to(ROOT)), "bytes": dst.stat().st_size, "sha256_bytes": b}
        if "sha256_content" in exp:
            c = sha256_content(dst)
            entry["sha256_content"] = c
            if c != exp["sha256_content"]:
                ok = False
                print(f"  CONTENT HASH MISMATCH {rel}")
        else:  # doc 08 §3: "compute on Day 1, record"
            entry["sha256_content"] = sha256_content(dst)
        if b != exp["sha256_bytes"]:
            entry["byte_hash_mismatch"] = True
            print(f"  warning: byte hash differs for {rel} (content hash is authoritative)")
        files[rel] = entry
        print(f"  {rel}: ok")
    manifest["sources"]["MarioDiffusion"] = {"url": cfg["mariodiffusion_url"], "commit": cfg["mariodiffusion_commit"],
                                             "license": "MIT", "files": files}
    if not ok:
        sys.exit("exit 6: content hash mismatch; do NOT edit the expected hashes (doc 08 §3)")

    # --- S-2: VGLC (optional, gives level names) ---
    if not args.skip_vglc:
        print("TheVGLC")
        try:
            v_dir = RAW / "TheVGLC"
            ensure_clone(cfg["vglc_url"], cfg["vglc_commit"], v_dir,
                         ["Super Mario Bros/Processed", "Super Mario Bros 2 (Japan)/Processed"])
            vfiles = {}
            for tag, sub in (("SMB1", "Super Mario Bros/Processed"), ("SMB2", "Super Mario Bros 2 (Japan)/Processed")):
                for p in sorted((v_dir / sub).glob("*.txt")):
                    dst = SOURCE / "vglc" / tag / p.name
                    copy_readonly(p, dst)
                    vfiles[f"{tag}/{p.name}"] = {"sha256_bytes": sha256_bytes(dst)}
            manifest["sources"]["TheVGLC"] = {"url": cfg["vglc_url"], "commit": cfg["vglc_commit"],
                                              "license": "MIT", "files": vfiles}
            manifest["level_names_available"] = True
            print(f"  {len(vfiles)} level files")
        except SystemExit as e:  # non-fatal: levels are then named L00..L41 (doc 08 §10)
            manifest["level_names_available"] = False
            print("  VGLC unavailable:", e)

    out = ROOT / "data" / "MANIFEST.json"
    out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT)}; next: python scripts/audit_dataset.py")


if __name__ == "__main__":
    main()
