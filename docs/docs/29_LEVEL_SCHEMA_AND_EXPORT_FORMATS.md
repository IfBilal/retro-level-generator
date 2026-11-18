# 29 — Level JSON Schema v1 and Export Formats

> **Single schema** for generated levels, real dev/test levels (blind-test seam, doc 31), showcase levels, the game input (doc 27) and the app (doc 28). Code: `src/text2mario/render/schema.py` (`validate_level`, `Level` dataclass), `render/renderer.py` (exports). `schema_version` is `1`; any breaking change bumps it and adds a ledger row. Tests: `tests/test_level_schema.py`, `tests/test_exports.py`.

## 1. Conventions
Rows 0..15 top to bottom (row 15 = bottom). Columns left to right. Tile IDs 0..12 per ledger F6. Integers only; seeds are JS-safe (0 <= seed <= 2^53 - 1, never floats). Strings UTF-8. Unknown extra keys are allowed under `ext` only.

## 2. Schema (field, type, meaning)
```jsonc
{
  "schema_version": 1,
  "level_id": "t2m_20261201-1432_ab12cd",        // string, unique
  "kind": "generated",                           // "generated" | "real" | "showcase"
  "size": {"rows": 16, "cols": 121},             // cols = 3 + width + 6 for generated levels
  "raw_grid": [[0,0,...]],                       // int[16][cols], model output + framing; metrics use this
  "display_grid": [[0,0,...]],                   // int[16][cols], after optional display-time repair (may equal raw_grid)
  "prompt": {"text": "...", "canonical": ["..."], "ignored_tokens": ["..."], "scenes": [{"caption": "...", "col_from": 3, "col_to": 50}]},
  "generation": {"seed": 123, "w": 5.0, "width": 112, "ddim_steps": 50, "resample_r": 2, "fast_mode": false,
                 "profile": "local|cloud", "checkpoint_sha256": "...", "config_hash": "...", "git_sha": "...",
                 "machine": "laptop_4050", "created_utc": "2026-12-01T14:32:00Z"},
  "framing": {"templated_cols": [[0,2],[115,120]], "spawn": {"col": 1, "row": 13}, "flag": {"col": 119, "row_top": 5, "row_bottom": 13},
              "checkpoints": [{"col": 1}, {"col": 60}]},
  "sanitizer": {"broken_pipes": 0, "broken_cannons": 0, "repaired": false},
  "verification": {"status": "verified|not_verified|error", "reason": null, "attempts_total": 3,
                   "attempt_log": [{"window": 4, "attempt": 1, "status": "unsolved", "fail_col": 41}],
                   "checker": {"tier": 1, "physics_sha256": "...", "seconds": 0.4, "path_len": 130}},
  "metrics": {"difficulty": null, "cscore": null, "notes": null},   // difficulty reserved (doc 31 S-10); metrics optional
  "provenance": {"source": "app|eval|showcase|real", "dataset_level_id": null, "dataset_level_name": null},
  "ext": {}
}
```
Required: `schema_version, level_id, kind, size, raw_grid, display_grid, framing, verification`; generated levels also require `prompt, generation, sanitizer`. For `kind = "real"`, `generation` may be null and `provenance.dataset_level_id` is set.

## 3. Validation rules (`validate_level`)
grid shapes equal `size`; all IDs in 0..12; `framing.spawn` on a standable cell; `flag.col < cols`; `templated_cols` inside the grid; `verification.checker.physics_sha256` present when `status` is `verified`; `attempts_total <= 20` and per-window attempts <= 5 (G5); seeds JS-safe; no NaN. Round trip `level -> json -> level` is the identity.

## 4. Export formats (FR-A-11, US-07)
| Format | File | Content |
|---|---|---|
| Level JSON | `t2m_<level_id>.level.json` | the schema above |
| PNG | `t2m_<level_id>.png` | sprite render, scale 2, metadata in tEXt chunks (level_id, seed, prompt) |
| CSV | `t2m_<level_id>.ids.csv` | 16 rows of comma-separated tile IDs (display_grid) |
| VGLC text | `t2m_<level_id>.vglc16.txt` | 16 lines of characters per F6 (`-`,`<`,`>`,`?`,`B`,`E`,`Q`,`S`,`X`,`[`,`]`,`b`,`o`); round trip with `parse_vglc` is the identity |
| Playable HTML | `t2m_<level_id>.html` | single self-contained Phaser page (doc 27 standalone), no external URLs |
| Tiled (optional, MAY) | `.tmj` | cut unless time remains |

## 5. Showcase and evidence
`app/showcase/*.level.json` + `*.png` + `manifest.json` (prompt, seed, w, checkpoint sha, `human_played: true`). Evidence of evaluation stores levels in the same schema (doc 35 §3).
