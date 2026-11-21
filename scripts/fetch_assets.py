"""Fetch the two runtime asset sets: Phaser 3.90.0 (MIT) and the Kenney Pixel Platformer pack (CC0)  (doc 26 §4).

Explain-me:
  What: downloads phaser.min.js (size + sha256 checked against doc 26 §2.1), then the Kenney zip (kenney.nl first,
        OpenGameArt mirror second), unpacks only the allow-listed files and records every source tried in
        assets/ASSET_MANIFEST.json.  Idempotent: files that already verify are not downloaded again.
  In/Out: network -> app/static/vendor/{phaser.min.js,PHASER_LICENSE.md}, assets/kenney/*, assets/ASSET_MANIFEST.json.
  Previous: none (Day 0).  Next: scripts/make_atlas.py builds the 13-tile atlas from the Kenney tilemap.
  Try: delete app/static/vendor/phaser.min.js and re-run: it is fetched again and re-verified.
Failure policy (CLAUDE.md rule 12): if Kenney cannot be fetched the placeholder skin is used and the manifest says so.
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import sys
import time
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urljoin

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "app" / "static" / "vendor"
KENNEY = ROOT / "assets" / "kenney"
MANIFEST = ROOT / "assets" / "ASSET_MANIFEST.json"
UA = "text2mario-asset-fetch/1.0 (+offline academic project)"
PHASER_URLS = ["https://cdn.jsdelivr.net/npm/phaser@3.90.0/dist/phaser.min.js",
               "https://unpkg.com/phaser@3.90.0/dist/phaser.min.js"]
PHASER_SIZE = 1_196_122
PHASER_SHA = "e92ddef111ba42e92d316979c732311757093688ea1810591cb7aa2858eba7a7"
PHASER_LICENSE_URL = "https://cdn.jsdelivr.net/npm/phaser@3.90.0/LICENSE.md"
KENNEY_PAGES = [("https://kenney.nl/assets/pixel-platformer", r'href="([^"]*kenney_pixel-platformer[^"]*\.zip)"'),
                ("https://opengameart.org/content/pixel-platformer-0",
                 r'href="([^"]*/sites/default/files/[^"]*pixel-platformer[^"]*\.zip)"')]
ALLOW = re.compile(r"(?i)(^|/)(Tilemap/[^/]+\.png|Tilesheet\.txt|License\.txt|Tiles/[^/]+\.png|Tiles/Characters/[^/]+\.png)$")


def http_get(url: str, tries: int = 3) -> bytes:
    """GET with a 30 s timeout and 1/2/4 s backoff; 403/404 means 'try the next source', not 'retry'."""
    last = None
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=30) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (403, 404):
                break
        except Exception as e:  # network hiccup
            last = e
        time.sleep(2 ** i)
    raise RuntimeError(f"{url}: {last}")


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def fetch_phaser(tried: list) -> dict:
    dest = VENDOR / "phaser.min.js"
    if dest.exists() and dest.stat().st_size == PHASER_SIZE and sha(dest.read_bytes()) == PHASER_SHA:
        return {"file": "app/static/vendor/phaser.min.js", "sha256": PHASER_SHA, "bytes": PHASER_SIZE, "source_used": "already present"}
    for url in PHASER_URLS:
        try:
            b = http_get(url)
            assert len(b) == PHASER_SIZE and sha(b) == PHASER_SHA, f"size/sha mismatch ({len(b)} bytes)"
            VENDOR.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b)
            (VENDOR / "PHASER_LICENSE.md").write_bytes(http_get(PHASER_LICENSE_URL))
            tried.append({"url": url, "result": "ok"})
            return {"file": "app/static/vendor/phaser.min.js", "sha256": PHASER_SHA, "bytes": PHASER_SIZE, "source_used": url,
                    "licence": "MIT (Copyright (c) 2024 Richard Davey, Phaser Studio Inc.)"}
        except Exception as e:
            tried.append({"url": url, "result": f"failed: {e}"})
    raise SystemExit("Phaser could not be fetched from any source")


def fetch_kenney(tried: list) -> dict | None:
    zbytes = used = None
    for page, rx in KENNEY_PAGES:
        try:
            html = http_get(page).decode("utf-8", "replace")
            m = re.search(rx, html)
            if not m:
                tried.append({"url": page, "result": "no zip link found"})
                continue
            url = urljoin(page, m.group(1))
            zbytes = http_get(url)
            used = url
            tried.append({"url": page, "result": "ok", "zip": url})
            break
        except Exception as e:
            tried.append({"url": page, "result": f"failed: {e}"})
    if zbytes is None:
        return None
    z = zipfile.ZipFile(io.BytesIO(zbytes))
    names = z.namelist()
    lic = next((n for n in names if n.lower().endswith("license.txt")), None)
    assert lic and "CC0" in z.read(lic).decode("utf-8", "replace").upper() or "creative commons zero" in z.read(lic).decode("utf-8", "replace").lower(), \
        "licence file does not say CC0: refusing to use the pack (CLAUDE.md rule 11)"
    KENNEY.mkdir(parents=True, exist_ok=True)
    kept = []
    for n in names:
        if ALLOW.search(n) and not n.endswith("/"):
            out = KENNEY / n.split("/", 1)[1] if n.count("/") else KENNEY / n
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(z.read(n))
            kept.append(str(out.relative_to(ROOT)).replace("\\", "/"))
    return {"source_used": used, "zip_sha256": sha(zbytes), "zip_files": len(names), "kept": len(kept),
            "licence": "CC0 1.0 (Kenney.nl)", "licence_file": lic, "sample": kept[:6]}


def main() -> int:
    tried_p, tried_k = [], []
    man = {"generated_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")}
    man["phaser"] = {**fetch_phaser(tried_p), "sources_tried": tried_p}
    k = fetch_kenney(tried_k)
    man["kenney"] = ({**k, "sources_tried": tried_k} if k else
                     {"source_used": None, "sources_tried": tried_k, "fallback": "placeholder skin (doc 26 A2)"})
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(man, indent=2) + "\n")
    print(json.dumps({k_: (v if k_ == "generated_utc" else {x: y for x, y in v.items() if x != "sample"}) for k_, v in man.items()}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
