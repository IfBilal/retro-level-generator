# 08 — Data Acquisition (pinned, hashed, isolated)

> **Day:** 1 (timeline `03`, Phase 1a). **Gate it feeds:** G-P1. **Ledger rows:** F1–F19, D1, D2, V3.
> **Output of this doc:** a reproducible `data/` tree whose every byte is traceable to a pinned commit and a sha256, created by one command: `python scripts/fetch_data.py`.

---

## 1. What we download, and why

| # | Source | What we take | Pin | Licence | Required? |
|---|---|---|---|---|---|
| S-1 | `github.com/schrum2/MarioDiffusion` | `datasets/Mar1and2_LevelsAndCaptions-regular.json` (training corpus, F1), `datasets/smb.json` (tile descriptors, F6), `datasets/SMB1_Levels.json` + `datasets/SMB2_Levels.json` (game-boundary check, F9), `datasets/Mar1and2_RandomTest-regular.json` (paper's random captions, reference only), `create_ascii_captions.py` + `captions/util.py` + `captions/caption_match.py` + `captions/caption_generator.py` (**read as documentation only**, for the detector/C-score/grammar re-implementation), `MarioEval.jar` (optional evaluator, doc 21) | commit **`561c76d4634027fa89ff8c1cfb58a24e8c089197`** (2026-05-28, "edited the wall recognition") — the commit that was audited on 2026-11-17 | MIT (code, captions); level layouts are Nintendo designs transcribed by the VGLC (see §9) | **Yes** |
| S-2 | `github.com/schrum2/TheVGLC` (cleaned VGLC fork used to build S-1) | `Super Mario Bros/Processed/*.txt` (20 files), `Super Mario Bros 2 (Japan)/Processed/*.txt` (22 files) | commit **`e91316a0b53955e981e179c5ec4f8d989d02b2bb`** | MIT (VGLC transcription, © 2016 Summerville, Ontañón, Snodgrass) | Optional (level names + full-width levels for the playability baseline, doc 21) — **strongly recommended** |
| S-3 | Hugging Face `sentence-transformers/all-MiniLM-L6-v2` | `config.json`, `model.safetensors`, `tokenizer.json`, `tokenizer_config.json`, `vocab.txt`, `special_tokens_map.json` | HF snapshot commit hash recorded at first download (V3) | Apache-2.0 | **Yes** (Phase 4+) |

**Never used:** the repo's `-train/-validate/-test.json` split files (F17, D1), its `*.pkl` tokenizers (pickle = arbitrary code execution, §8), its `requirements.txt`, its `.bat` pipeline, its pretrained HF diffusion models.

### 1.1 Verified facts about S-2 (measured 2026-11-17)
- Sorting the 20 SMB1 file names, then the 22 SMB2 file names, **lexicographically** (exactly what S-1's builder does: `sorted(Path(dir).glob("*.txt"))`, SMB1 batch first) and slicing each level with a 16-wide stride-1 window **reproduces all 42 chains of S-1 byte-for-byte** (42/42 exact matches). Level names can therefore be attached to `level_id` 0..41 with certainty (full table in `09_DATA_AUDIT_AND_EDA.md` §3).
- **Level heights are not all 14 rows:** 20 levels are 14 rows, 21 levels are 15 rows (SMB1 8-2, 8-3 and 19 SMB2 levels), and 1 level (SMB2 World 3-1) is 16 rows. S-1's builder top-pads with sky (`-`) to 16 rows, and **extends any pipe tile found in the top VGLC row upward through the padding rows** (hanging/upside-down pipes). That is the source of the 64 non-sky cells in rows 0–1 (all `[`/`]` pipe bodies, in 17 windows). *This refines ledger F12 — see "Ledger conflicts" at the end of doc 09.*

---

## 2. Pinning procedure (do exactly this)

```
data/
  raw/                       # git-ignored; untrusted third-party clones live here ONLY
    MarioDiffusion/          # clone of S-1 at the pinned SHA
    TheVGLC/                 # sparse clone of S-2 at the pinned SHA (optional)
  source/                    # git-ignored; byte copies of the files we use (read-only after fetch)
    Mar1and2_LevelsAndCaptions-regular.json
    smb.json
    SMB1_Levels.json
    SMB2_Levels.json
    Mar1and2_RandomTest-regular.json
    vglc/SMB1/*.txt          # optional, 20 files
    vglc/SMB2/*.txt          # optional, 22 files
  processed/                 # git-ignored; produced by audit_dataset.py (doc 09)
  splits/                    # COMMITTED: split_v1.json, LOCK, reused_segments.json (doc 10)
  prompts/                   # COMMITTED: p_casual.jsonl, lexicon.yaml, p_rand.json (docs 10, 12)
  MANIFEST.json              # COMMITTED: provenance + hashes (§4)
```

**S-1 clone (Windows and Linux identical):**
```bash
git -c core.autocrlf=false clone --filter=blob:none --no-checkout \
    https://github.com/schrum2/MarioDiffusion.git data/raw/MarioDiffusion
git -C data/raw/MarioDiffusion -c core.autocrlf=false checkout 561c76d4634027fa89ff8c1cfb58a24e8c089197
git -C data/raw/MarioDiffusion rev-parse HEAD   # must print the pinned SHA
```
- `core.autocrlf=false` is **mandatory on Windows**: with Git for Windows' default (`true`) the JSON files are checked out with CRLF line endings and their byte sha256 changes (the content hash in §3 still matches, but the byte hash won't).
- `--filter=blob:none` + `--no-checkout` avoids downloading history blobs; the checkout fetches only the pinned tree (~320 MB of datasets; acceptable once per machine). If bandwidth is a problem, use a sparse checkout of `datasets/`, `captions/`, `create_ascii_captions.py`, `MarioEval.jar`, `LICENSE`.

**S-2 sparse clone:**
```bash
git -c core.autocrlf=false clone --filter=blob:none --no-checkout --sparse \
    https://github.com/schrum2/TheVGLC.git data/raw/TheVGLC
git -C data/raw/TheVGLC sparse-checkout set "Super Mario Bros/Processed" "Super Mario Bros 2 (Japan)/Processed"
git -C data/raw/TheVGLC -c core.autocrlf=false checkout e91316a0b53955e981e179c5ec4f8d989d02b2bb
```

**Moving to a newer upstream commit** is allowed only via `fetch_data.py --allow-new-commit <sha>`. That writes the new SHA to `MANIFEST.json`, re-runs `audit_dataset.py` (doc 09), and **must be followed by a ledger change-log row** if any F-fact changes. A changed corpus after Day 2 invalidates `data/splits/LOCK` (doc 10 §7). Do not do this after Day 2 without a written reason in `PROGRESS.md`.

---

## 3. Expected hashes (measured 2026-11-17 at the pinned SHA)

Two hashes per JSON file:
- **byte hash** = `sha256(file bytes)` — exact provenance (requires LF line endings).
- **content hash** = `sha256(json.dumps(json.load(f), separators=(",", ":"), sort_keys=True).encode("utf-8"))` — robust to whitespace/line-ending differences. **The audit asserts the content hash** and only warns on a byte-hash mismatch.

| File | byte sha256 | content sha256 |
|---|---|---|
| `Mar1and2_LevelsAndCaptions-regular.json` | `9d3eb4dbd82a90f3c74cfe869239b9cd18e8cc93670a053fcba999236d5e2a25` | `6a15264a4beb67247a0488dba0439f257f7a94955c0df60b709b7e85024ade2c` |
| `smb.json` | `c1a23d7af82c7cdf0364cd1a1d6fcb4ada14092b0f383ffc7edc0e7f10987dcf` | `833ba2141418b29852951030f24967df90cd2c318d194b8d8b31b665e00da823` |
| `SMB1_Levels.json` | `18ee2e2e4677ba22c030d7759302c04bcf2592e31e9d9db21cac509429df77d5` | (compute on Day 1, record) |
| `SMB2_Levels.json` | `318c0b264d1c5b994c23e28fe4fafa30ac5f6e93ed194755d7600c98259d88d6` | (compute on Day 1, record) |
| `Mar1and2_RandomTest-regular.json` | `06ebc0ba420b6a018310d8c6b985abb0477850dc7bfdaafae77b98ee34df87b4` | (compute on Day 1, record) |

If the content hash of the training corpus differs, **stop**: either the pin was not honoured or upstream rewrote history. Do not "fix" it by updating the expected value without re-running the full audit and logging the change.

---

## 4. `data/MANIFEST.json` (committed)

```json
{
  "schema_version": 1,
  "created_utc": "2026-11-18T09:12:44Z",
  "created_by": "scripts/fetch_data.py",
  "machine_profile": "desktop_3060ti",
  "sources": {
    "MarioDiffusion": {
      "url": "https://github.com/schrum2/MarioDiffusion.git",
      "commit": "561c76d4634027fa89ff8c1cfb58a24e8c089197",
      "license": "MIT",
      "files": {
        "datasets/Mar1and2_LevelsAndCaptions-regular.json": {
          "copied_to": "data/source/Mar1and2_LevelsAndCaptions-regular.json",
          "bytes": 42028678,
          "sha256_bytes": "9d3eb4db…2a25",
          "sha256_content": "6a15264a…de2c"
        },
        "datasets/smb.json": { "copied_to": "data/source/smb.json", "sha256_content": "833ba214…a823" }
      }
    },
    "TheVGLC": {
      "url": "https://github.com/schrum2/TheVGLC.git",
      "commit": "e91316a0b53955e981e179c5ec4f8d989d02b2bb",
      "license": "MIT",
      "files": { "Super Mario Bros/Processed/mario-1-1.txt": { "sha256_bytes": "…" } }
    },
    "MiniLM": {
      "hf_repo": "sentence-transformers/all-MiniLM-L6-v2",
      "hf_commit": "<snapshot hash recorded at first download>",
      "license": "Apache-2.0",
      "files": { "model.safetensors": { "sha256_bytes": "…" }, "tokenizer.json": { "sha256_bytes": "…" } }
    }
  },
  "audit": { "script": "scripts/audit_dataset.py", "passed": true, "report": "reports/eda/AUDIT.json" }
}
```
Rules: full 64-hex hashes in the real file (abbreviated above for readability); `audit.passed` is written by doc 09's script, never by hand.

---

## 5. MiniLM download and caching (S-3)

1. **Where it lives:** set `HF_HOME` per machine in `.env` (git-ignored), e.g. `D:/hf_cache` on Windows, `~/.cache/huggingface` on Linux. Both machines download independently; the snapshot hash must be identical (checked by `verify_env.py --models`).
2. **First download (online):** `python scripts/fetch_models.py` →
   `AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")` and `AutoModel.from_pretrained(...)` — no `sentence-transformers` package needed (V3). Record `model.config._commit_hash` (or the snapshot directory name) as `hf_commit` and the sha256 of `model.safetensors` and `tokenizer.json` in `MANIFEST.json`.
3. **Afterwards (offline, always):** training, evaluation and the app load with `local_files_only=True` and `HF_HUB_OFFLINE=1`. The app must start with Wi-Fi off (G-P7, doc 28). A missing cache is a hard error with an actionable message, not a silent download.
4. **If Hugging Face is unreachable on one machine:** copy the snapshot folder `models--sentence-transformers--all-MiniLM-L6-v2/` through the sync folder (doc 07) and point `HF_HOME` at it; verify hashes.
5. **Fallback mirror (last resort):** export once to `assets/models/minilm/` with `save_pretrained()` (git-ignored, ~91 MB) and load from that path. Record `"source": "local_export"` in the manifest.
6. **Tokenizer facts to measure on Day 1** (doc 11 §6): max token length over all 7,687 captions, over augmented captions, and over normalized P-casual prompts. The ledger's `max_len = 48` (M7) is asserted, never truncated silently.

---

## 6. `scripts/fetch_data.py` — specification

```
usage: python scripts/fetch_data.py [--skip-vglc] [--allow-new-commit SHA]
                                    [--from-dir PATH] [--force] [--dry-run]
```

| Step | Action | Failure behaviour |
|---|---|---|
| 1 | Read pins from `configs/data.yaml` (`mariodiffusion_commit`, `vglc_commit`, expected content hashes). | Missing key → exit 2. |
| 2 | If `data/raw/MarioDiffusion` exists at the pinned SHA and `--force` is not set → skip the clone (idempotent). Else clone per §2. | Network error → retry once after 10 s → exit 3 with the manual instructions of §7. |
| 3 | Verify `git rev-parse HEAD == pin`. | Mismatch → exit 4 (never continue on the wrong commit). |
| 4 | Copy the listed files to `data/source/` (`shutil.copy2`), then set them read-only (`os.chmod(path, 0o444)`; on Windows, `stat.S_IREAD`). | Missing file → exit 5. |
| 5 | Compute byte + content hashes; compare with `configs/data.yaml`. | Content mismatch → exit 6. Byte mismatch only → warning + `"byte_hash_mismatch": true` in the manifest. |
| 6 | Unless `--skip-vglc`: sparse-clone S-2 (§2), verify the SHA, copy the 42 `.txt` files to `data/source/vglc/SMB1|SMB2/`, hash them. | Any failure → warning; `level_names_available = false` in the manifest (doc 09 then names levels `L00`..`L41`). |
| 7 | Write `data/MANIFEST.json` (`audit.passed = null` until doc 09's script runs). | — |
| 8 | Print a summary table: files, sizes, hashes OK/FAIL, next command (`python scripts/audit_dataset.py`). | — |

- `--from-dir PATH`: offline mode. Copy files from a folder the user prepared (e.g., a USB stick), skip git, still verify hashes. This is how the second machine can be set up without internet.
- `--dry-run`: print the plan, touch nothing.
- **No code from `data/raw/` is imported or executed by this script** (§8).
- Unit tests (`tests/test_fetch.py`, offline): hash function on a fixture file; content-hash invariance to CRLF and indentation; refusal on wrong SHA (mock `git`); idempotence (second run makes no copies).

---

## 7. Manual fallback (if git/network fails on a machine)

1. On any machine with internet: download `https://github.com/schrum2/MarioDiffusion/archive/561c76d4634027fa89ff8c1cfb58a24e8c089197.zip` and the same for TheVGLC at `e91316a0b53955e981e179c5ec4f8d989d02b2bb`.
2. Extract each into its own **new, empty** folder (never into the repo).
3. Run `python scripts/fetch_data.py --from-dir <folder with the extracted datasets/>`.
4. Hash verification still applies. Archive-based extraction has LF endings, so byte hashes should match.

---

## 8. Isolation rules for untrusted third-party material (mandatory)

Downloaded repositories are **untrusted data**:

1. **Separate directories.** Each download goes into its own new, empty directory under `data/raw/`. Our scripts live in `scripts/` and `src/` and receive paths as arguments.
2. **Never execute or import their code.** No `import` from `data/raw/`, no `sys.path` insertion pointing there, no running their `.py`/`.bat` files, no `pip install -r data/raw/.../requirements.txt`. Python that reads their files runs with `python -I` (isolated mode: ignores `PYTHONPATH`, user site-packages, and the script's directory) from **our** working directory. Reason: an interpreter started inside the folder would pick up a planted `json.py` on `import json`.
3. **Read their logic as documentation.** The caption detector, C-score and random-caption grammar are re-implemented cleanly in `src/text2mario/captions/` from reading the source, with an MIT attribution header:
   ```python
   # Logic re-implemented from schrum2/MarioDiffusion (MIT License, (c) 2025 Jacob Schrum),
   # commit 561c76d4…, files create_ascii_captions.py, captions/util.py, captions/caption_match.py.
   ```
   Correctness is proven by **behaviour** (reproducing the 7,687 stored captions, doc 09 §6), not by copying code.
4. **Never unpickle.** `*.pkl` files from the repo are never loaded (`pickle.load` can execute arbitrary code). We don't need them.
5. **JSON only via `json.load`.** No `eval`, no `yaml.load` without `SafeLoader`.
6. **The jar.** `MarioEval.jar` (doc 21) runs only via an explicit `subprocess.run(["java", "-jar", <abs path>, ...], cwd=<fresh temp dir>, timeout=60)` with level files written by us. It never runs on the app's request path, and never with network access needed.
7. **Hashes before use.** Every consumer of `data/source/*` calls `text2mario.data.provenance.verify()` first, which re-checks content hashes against `MANIFEST.json` (cheap: ~0.4 s for the 42 MB corpus).
8. **Read-only copies.** `data/source/` files are chmod read-only after fetch. Any code path that opens them for writing is a bug (test: `tests/test_provenance.py` asserts the mode bits).

---

## 9. Licences and usage constraints

| Item | Licence | Our obligations | Committed to our repo? |
|---|---|---|---|
| MarioDiffusion code, captions, JSON structure | MIT © 2025 Jacob Schrum | Keep the copyright notice in `THIRD_PARTY_NOTICES.md`; attribution headers on re-implemented logic | **No** (data files are git-ignored; only hashes/manifest are committed) |
| VGLC (cleaned fork) transcriptions | MIT © 2016 Summerville, Ontañón, Snodgrass | Cite the VGLC paper (arXiv 1606.07487) in the report | **No** |
| Underlying level designs (SMB1, SMB2 Japan) | © Nintendo | Academic, non-commercial research use; no redistribution of the raw corpus; the app never ships the training data; the "copy of a Nintendo level?" check (copy rate, doc 22) is part of honest reporting | **No** |
| all-MiniLM-L6-v2 | Apache-2.0 | Keep the licence/NOTICE when exporting a local copy | **No** (cache only) |
| MarioEval.jar | Bundled in the MIT repo; derived from the Mario AI framework (licence not separately stated) | Use as a local evaluation tool only; do not redistribute | **No** |
| Our split, prompts, lexicon, manifest | Ours | — | **Yes** |

`THIRD_PARTY_NOTICES.md` (repo root, committed) lists all of the above plus the asset licences (doc 26).

---

## 10. Failure modes and fallbacks

| Failure | Detection | Fallback | Logged where |
|---|---|---|---|
| GitHub unreachable | clone fails twice | §7 manual zip + `--from-dir` | `PROGRESS.md`, manifest `source: "zip"` |
| Pinned commit no longer exists upstream (force-push) | checkout fails | Use the zip of the SHA if GitHub still serves it; else any copy with a matching **content hash** (the content hash is the real contract) | ledger change log |
| Content hash mismatch | step 5 | Stop. Investigate; never edit expected hashes to make it pass | `PROGRESS.md` "Blocked" |
| VGLC fork unavailable | step 6 | Continue without level names (`L00`..`L41`); doc 21's full-level baseline is reconstructed from chains instead (the first window plus the last column of each later window recovers the full 16-row level exactly) | manifest |
| Hugging Face blocked | `fetch_models.py` | §5 step 4/5 (copy the snapshot via the sync folder) | manifest `MiniLM.source` |
| Windows CRLF | byte-hash warning | Ignore (content hash is authoritative); re-clone with `core.autocrlf=false` for cleanliness | manifest flag |

---

## 11. Acceptance checklist (end of Day 1, before doc 09's audit)

- [ ] `data/raw/MarioDiffusion` at `561c76d4…`; `data/raw/TheVGLC` at `e91316a0…` (or documented skip).
- [ ] `data/source/` holds the 5 JSON files (+ 42 `.txt`), all read-only.
- [ ] Content hashes of the corpus and `smb.json` equal §3.
- [ ] `data/MANIFEST.json` committed; `data/raw/`, `data/source/`, `data/processed/` are in `.gitignore` (and `git status` shows none of them).
- [ ] MiniLM cached on this machine; `hf_commit` and file hashes recorded; offline load works with `HF_HUB_OFFLINE=1`.
- [ ] `THIRD_PARTY_NOTICES.md` exists.
- [ ] Nothing from `data/raw/` was imported or executed (`grep -r "data/raw" src/ scripts/` only finds path constants passed to readers).
