"""Build the single submission zip (doc 35 §4): python scripts/build_submission_zip.py [--out ../text2mario_AI2002.zip]

Explain-me:
  What: zips the source, app (with the exported model bundle and showcase), configs, tests, docs, locked splits/prompts, reports, evidence,
        report and slides.  Excludes training runs, raw third-party data, caches, virtualenvs, git history and checkpoints (step_*.pt).
        Writes the zip's size and sha256 next to it (for the upload record).  Fails if a forbidden file slipped in.
  In/Out: repository -> one zip + <zip>.sha256.  Previous: collect_evidence.py, build_report.py, build_slides.py.  Next: clean-room test.
  Try: python scripts/build_submission_zip.py --list   (print what would be included, no zip)
"""
import argparse
import hashlib
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCLUDE_DIRS = {".hypothesis", ".git", ".venv", "venv", "runs", "__pycache__", ".pytest_cache", ".ruff_cache", "cache", "raw", "node_modules", "_downloads", "models", ".claude"}
EXCLUDE_NAMES = {".env"}
EXCLUDE_SUFFIX = {".pt", ".pyc", ".tmp"}
FORBIDDEN = ("data/raw/", "runs/", "step_00")


def files():
    for p in sorted(ROOT.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(ROOT)
        parts = set(rel.parts)
        if parts & EXCLUDE_DIRS or p.name in EXCLUDE_NAMES or p.suffix in EXCLUDE_SUFFIX:
            continue
        if rel.parts[:2] == ("assets", "kenney") and p.suffix == ".png" and p.name.startswith("tile_"):
            continue                                              # single-tile PNGs are redundant with the packed sheets
        yield p, str(rel).replace("\\", "/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT.parent / "text2mario_AI2002.zip"))
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    items = list(files())
    bad = [r for _, r in items if any(f in r for f in FORBIDDEN) or r.endswith(".pt")]
    if bad:
        sys.exit(f"forbidden files would be included: {bad[:5]}")
    if a.list:
        total = sum(p.stat().st_size for p, _ in items)
        print(len(items), "files", round(total / 1e6, 1), "MB uncompressed")
        for _, r in items[:60]:
            print(" ", r)
        return
    out = Path(a.out)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p, r in items:
            z.write(p, f"text2mario/{r}")
    h = hashlib.sha256(out.read_bytes()).hexdigest()
    Path(str(out) + ".sha256").write_text(f"{h}  {out.name}\n")
    print(f"{out}  {out.stat().st_size / 1e6:.1f} MB  {len(items)} files  sha256 {h[:16]}...")


if __name__ == "__main__":
    main()
