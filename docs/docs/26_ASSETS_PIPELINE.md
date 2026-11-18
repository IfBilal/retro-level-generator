# 26 — Assets Pipeline (autonomous fetch, licences, tile mapping, skins)

> **Owner track:** G (Game/App). **Days:** D0 (fetch), D1 (mapping + gate), D3 (first use by the game), D18 (offline check).
> **Ledger rows:** A1–A4 (assets), V5 (Phaser), F6 (tile IDs), S2 (MVP), G3 (framing objects), G-P7 (app gate).
> **Hard rule (CLAUDE.md §2.11):** only CC0 / MIT assets are committed. Nintendo-derived art is never committed and never ships by default (A3).

---

## 1. Goals and non-goals

**Goals**
1. Claude Code fetches every runtime asset **by itself on Day 0** (no human download step on the happy path). Every file gets a pinned sha256 and its licence text.
2. Before the model is finished (Day 13), the game and the PNG renderer already have **a complete, licence-clean sprite set** for all 13 tile IDs (F6) plus player, enemy, coin, flag and checkpoint art.
3. **Nothing ever blocks.** If a download fails, a deterministic procedural "placeholder" skin is generated locally (A2), and the failure is logged (no silent fallbacks, CLAUDE.md §2.12).
4. **Offline at runtime.** After Day 0, the app and the game never touch the network (viva requirement, Day 18).
5. **One stable interface.** The game and the renderer only see *our* atlas layout (`atlas_layout.json`), never Kenney's internal sheet layout. Swapping skins = rebuilding the atlas, never touching game code.

**Non-goals:** custom art, animated backgrounds, music (sound effects are optional, §9).

---

## 2. Inventory of runtime assets

| Asset | Purpose | Source (licence) | Destination (committed?) |
|---|---|---|---|
| Phaser **3.90.0** `phaser.min.js` | Game engine (V5, A4) | npm `phaser@3.90.0` via jsDelivr / unpkg / npm tarball (**MIT**) | `app/static/vendor/phaser.min.js` + `app/static/vendor/LICENSE.phaser.md` (**committed**) |
| Kenney **Pixel Platformer** | Tiles, characters, enemies, coins, flags (A1) | kenney.nl / OpenGameArt mirror / itch.io (**CC0 1.0**) | Extracted to `app/static/assets/kenney/pixel-platformer/`. **Only an allowlisted subset is committed** (§4.5) |
| Built atlas `t2m_tiles.png` + `t2m_entities.png` | What the game and renderer actually load | Built by `scripts/build_atlas.py` from Kenney frames + procedural fill-ins (own work, CC0) | `app/static/assets/atlas/<skin>/` (**committed**) |
| Placeholder skin | Never-blocking fallback (A2) | `scripts/make_placeholder_sprites.py` (own work, CC0) | `app/static/assets/atlas/placeholder/` (**committed**) |
| Sound effects (optional) | jump / coin / stomp / bump / death / win | Kenney **Digital Audio** (CC0) **or** the built-in WebAudio synth (no files) | `app/static/assets/audio/kenney/*.ogg` (committed subset, optional) |
| Classic skin (local only) | The user's own local preference (A3) | User-supplied, e.g. MarioDiffusion's `mapsheet.png` (128×112 px = 8×7 frames of 16 px; Nintendo-derived) | `assets/local_only/` (**git-ignored, never committed**) |

### 2.1 Verified facts used here (checked 2026-11-17)
- npm `phaser@3.90.0` exists, licence **MIT**. Tarball `https://registry.npmjs.org/phaser/-/phaser-3.90.0.tgz`, npm integrity `sha512-/cziz/5ZIn02uDkC9RzN8VF9x3Gs3XdFFf9nkiMEQT3p7hQlWuyjy4QWosU802qqno2YSLn2BfqwOKLv/sSVfQ==`.
- Inside that tarball: `package/dist/phaser.min.js` = **1,196,122 bytes**, sha256 `e92ddef111ba42e92d316979c732311757093688ea1810591cb7aa2858eba7a7`. `package/dist/phaser-arcade-physics.min.js` (Arcade-only build, no Matter.js) = **1,086,308 bytes**, sha256 `2e1bc84f885ab5c62b11ebb72492bb334159ac5678b664c7cb2942fc979e44e3`. `package/LICENSE.md` = MIT, "Copyright (c) 2024 Richard Davey, Phaser Studio Inc.".
- `phaser.min.js` 3.90.0 contains **no** `</script` substring, so it can be inlined verbatim inside a `<script>` tag (doc 27 §10).
- Kenney Pixel Platformer: **CC0**, tile size **18×18**, "200 files", current page version **1.2** (kenney.nl). The OpenGameArt mirror lists `kenney_pixel-platformer.zip` at **260.3 KB**, CC0, crediting "Kenney.nl" optional.
- Kenney packs ship a tilesheet text file (`Tilesheet.txt` / `tilemap.txt`) that gives tile width, height and spacing (Kenney knowledge base). **The exact folder layout inside the zip could not be verified from the planning sandbox** (the shell cannot reach kenney.nl or OpenGameArt). The fetch script therefore **discovers** sheet files and grid geometry (§4.4) instead of hard-coding them.

> Expected, not verified (from memory of v1.x; the script must confirm and record): `Tilemap/tilemap_packed.png` (18 px frames, no spacing), `Tilemap/tilemap.png` (1 px spacing), `Tilemap/tilemap-characters_packed.png` (24 px frames), `Tilemap/tilemap-backgrounds_packed.png`, `Tiles/tile_NNNN.png`, `Tiles/Characters/tile_NNNN.png`, `Tilesheet.txt`, `License.txt`.

---

## 3. Directory layout (asset-related only)

```
assets/
  ASSET_MANIFEST.json          # committed: every fetched/built file, source, sha256, licence (§7)
  tile_map.json                # committed: role -> source frame / procedural rule (§5)
  atlas_layout.json            # committed: FIXED frame order of our atlases (§5.1) — the game/renderer contract
  _downloads/                  # git-ignored: raw zips/tarballs (cache for re-extraction)
  local_only/                  # git-ignored: classic skin inputs + outputs (A3)
app/static/
  vendor/phaser.min.js         # committed (MIT)
  vendor/LICENSE.phaser.md     # committed
  assets/kenney/pixel-platformer/   # extraction target; .gitignore allowlists the subset (§4.5)
  assets/atlas/kenney/t2m_tiles.png, t2m_entities.png        # committed (built)
  assets/atlas/placeholder/t2m_tiles.png, t2m_entities.png   # committed (built)
  assets/audio/kenney/*.ogg    # optional, committed subset
  game/template.html, game.js, physics.json                  # doc 27
reports/assets/                # contact sheets, skin check renders, gate evidence
scripts/fetch_assets.py, build_atlas.py, make_placeholder_sprites.py,
        make_contact_sheet.py, validate_tile_map.py, gate_assets.py
```
If `05_REPO_STRUCTURE.md` places a file elsewhere, doc 05 wins for location. The ledger fixes `app/static/vendor/phaser.min.js` (V5), and the timeline fixes `app/static/game/physics.json` (Day 3).

---

## 4. `scripts/fetch_assets.py` — specification

### 4.1 CLI
```
python scripts/fetch_assets.py [--only phaser|kenney|audio|all] [--offline] [--repin]
                               [--manual-zip PATH] [--timeout 30] [--retries 3]
```
- `--offline`: no network. Verify that the files exist and match the manifest hashes. Exit 0 or 20.
- `--repin`: accept new hashes (for example, Kenney released v1.3). Rewrites the manifest entries and **invalidates `tile_map.json`** (sets `"needs_review": true`), so `validate_tile_map.py` fails until the mapping is re-checked.
- `--manual-zip`: use a zip the user downloaded by hand (the itch.io path, §4.3 step 3).

**Exit codes:** `0` all assets present and verified · `10` fell back to placeholders for ≥ 1 asset (warning; also written to `PROGRESS.md` and `events.jsonl`) · `20` hash mismatch against a pinned manifest entry (error; requires `--repin`) · `30` unexpected error.

### 4.2 Phaser — ordered sources (first success wins)
1. `https://cdn.jsdelivr.net/npm/phaser@3.90.0/dist/phaser.min.js`
2. `https://unpkg.com/phaser@3.90.0/dist/phaser.min.js`
3. `https://registry.npmjs.org/phaser/-/phaser-3.90.0.tgz`: verify the npm `integrity` (sha512, §2.1), then extract **only** `package/dist/phaser.min.js` and `package/LICENSE.md`. This route usually works even in locked-down networks that allow package registries.
4. `npm pack phaser@3.90.0` (only if `npm` is on PATH), then the same extraction.

After download: assert size = 1,196,122 bytes and sha256 = `e92ddef1…eba7a7` (§2.1). The licence comes from the tarball, or `https://cdn.jsdelivr.net/npm/phaser@3.90.0/LICENSE.md`. Write both files under `app/static/vendor/`.
*Optional size optimisation:* `phaser-arcade-physics.min.js` (−110 KB) is a valid drop-in for this game (Arcade only). It is **off by default** so the file name matches ledger V5. To enable it, change V5 first.

### 4.3 Kenney Pixel Platformer — ordered sources
1. **kenney.nl (canonical publisher, latest version).** GET `https://kenney.nl/assets/pixel-platformer`, then regex the HTML for `href="([^"]*kenney_pixel-platformer[^"]*\.zip)"` (absolute or site-relative; resolve with `urljoin`). Download it.
2. **OpenGameArt mirror.** GET `https://opengameart.org/content/pixel-platformer-0`, then regex for `href="([^"]*/sites/default/files/[^"]*pixel-platformer[^"]*\.zip)"`. Expected filename `kenney_pixel-platformer.zip`, ~260 KB. *Warning:* a mirror can lag behind (older version → different frame order). The manifest records which source and version were used.
3. **itch.io (manual).** `https://kenney-assets.itch.io/pixel-platformer` uses a "name your own price" click-through that can't be automated reliably. Print exact instructions ("download, save to `assets/_downloads/kenney_pixel-platformer.zip`, rerun with `--manual-zip`") and continue with step 4 so nothing blocks.
4. **Placeholder fallback (A2):** run `make_placeholder_sprites.py`, set the active skin to `placeholder`, exit code 10.

HTTP rules: honour `HTTPS_PROXY`/`HTTP_PROXY`; verify TLS with the system/certifi CA bundle (never disable verification); `User-Agent: text2mario-asset-fetch/1.0 (+offline academic project)`; timeout 30 s; 3 retries with exponential backoff (1, 2, 4 s) **per URL**, and then move to the next source. A 403 or 404 means "next source", not "retry forever".

### 4.4 Validation and discovery (Kenney zip)
1. **Zip sanity:** size < 10 MB; ≤ 2,000 entries; reject absolute paths, `..` components and symlinks (zip-slip protection); total uncompressed size < 50 MB.
2. **Licence check:** a file matching `(?i)licen[cs]e.*\.txt` must exist and contain `CC0` or `Creative Commons Zero` (case-insensitive). If not, abort the Kenney route (go to the placeholder fallback) and do not extract.
3. **Extract** into `app/static/assets/kenney/pixel-platformer/` (preserving the internal paths).
4. **Discover sheets:** glob `**/*tilemap*packed*.png` → classify by name: contains `character` → `characters`; contains `background` → `backgrounds`; otherwise → `tiles`. If no `*packed*` file exists, use the spaced `**/*tilemap*.png` and set `spacing = 1` (Kenney convention, confirmed by the text file).
5. **Geometry:** parse `Tilesheet.txt` / `tilemap.txt` if present (lines like `Tile size • 18px × 18px`, `Space between tiles • 1px × 1px`). Otherwise infer: for each sheet and each candidate size `s ∈ [18, 24, 16, 21, 32]`, accept the first `s` for which `(W + spacing) % (s + spacing) == 0` **and** `(H + spacing) % (s + spacing) == 0`. Record `frame_w, frame_h, columns, rows, spacing, margin, frame_count` per sheet.
6. **Expectation check (warning only):** `tiles.frame_w == 18` (A1). If it differs, keep going but set `tile_px` in `atlas_layout.json` to the discovered size. Tile size is a parameter everywhere (A1).
7. Write the manifest entries (§7) with sha256 for the zip and every kept file.

### 4.5 What gets committed (allowlist)
`app/static/assets/kenney/pixel-platformer/.gitignore`:
```
*
!.gitignore
!License.txt
!Tilesheet.txt
!tilemap.txt
!**/
!**/*tilemap*packed*.png
```
Rationale: CC0 allows redistribution, and committing the ~50–150 KB subset means the **second machine works after a plain `git pull`, even offline**. The full extraction (200 single PNGs, sample project) stays local.

### 4.6 Idempotence and two-machine rule
- Fetching happens **once** (Day 0, on either machine). The results are committed. The other machine never refetches: `fetch_assets.py --offline` only verifies hashes.
- If a later fetch finds a different zip hash (Kenney updated the pack), it **refuses to overwrite** without `--repin`, because frame indices in `tile_map.json` are only valid for the pinned sheet hash (§5.2).

---

## 5. Atlas contract and the 13-tile mapping

### 5.1 `assets/atlas_layout.json` — fixed frame order (the contract)
The game (doc 27) and the PNG renderer (doc 11/22 figures) load **only** `t2m_tiles.png` and `t2m_entities.png`. Frame indices **0–12 equal the F6 tile IDs**, so a level grid can feed the tilemap directly: sky becomes `-1`, and enemy/coin cells become `-1` plus a spawned sprite.

`t2m_tiles.png` — `tile_px × tile_px` frames (default 18), **16 columns × 2 rows = 32 frames**, no spacing:

| Frame | Role | Notes |
|---|---|---|
| 0 | `sky` | Fully transparent. Never drawn in the game (index `-1`); the renderer paints the sky colour |
| 1 / 2 | `pipe_top_left` / `pipe_top_right` | Must join seamlessly: right edge of 1 continues into left edge of 2 |
| 3 | `question_full` | |
| 4 | `cannon_top` | |
| 5 | `enemy_icon` | Static enemy picture for PNG renders (the game uses entity frames) |
| 6 | `question_empty` | |
| 7 | `brick` | |
| 8 | `ground_fill` | |
| 9 / 10 | `pipe_body_left` / `pipe_body_right` | Must join seamlessly |
| 11 | `cannon_support` | |
| 12 | `coin` | |
| 13 | `ground_top` | Render-time variant of 8 when the cell above is not solid (looks better; does not change the grid) |
| 14 / 15 / 16 | `flagpole_segment` / `flagpole_top` / `flag_cloth` | Framing objects (G3), drawn from `framing.flag` (doc 29) |
| 17 | `checkpoint_flag` | Drawn at `framing.checkpoints[1:]` |
| 18 | `spawn_marker` | PNG/debug renders only |
| 19–31 | reserved | Future features (e.g. springboard, bullet, difficulty markers) |

`t2m_entities.png` — `entity_px × entity_px` frames (default 24, or the discovered Kenney character size), **8 columns × 2 rows**:

| Frame | Role | Frame | Role |
|---|---|---|---|
| 0 | `player_idle` | 8 | `coin_spin_a` |
| 1 | `player_walk_a` | 9 | `coin_spin_b` |
| 2 | `player_walk_b` | 10 | `coin_pop` (bump effect) |
| 3 | `player_jump` | 11 | `brick_debris` (reserved) |
| 4 | `player_hurt` | 12–15 | reserved |
| 5 | `enemy_walk_a` | | |
| 6 | `enemy_walk_b` | | |
| 7 | `enemy_squashed` | | |

`atlas_layout.json` schema (abridged):
```json
{"schema_version": 1, "tile_px": 18, "entity_px": 24,
 "tiles": {"columns": 16, "rows": 2, "frames": {"sky": 0, "pipe_top_left": 1, "...": "...", "spawn_marker": 18}},
 "entities": {"columns": 8, "rows": 2, "frames": {"player_idle": 0, "...": "...", "coin_pop": 10}},
 "solid_tile_frames": [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 13],
 "sprite_tile_ids": {"enemy": 5, "coin": 12}}
```
`solid_tile_frames` must equal `physics.json → solid_tile_ids` (doc 27 §4, grid IDs only) **plus frame 13** (the render-only `ground_top` variant). A test checks this, because both the game and the playability checker (doc 21) depend on the solid set.

### 5.2 `assets/tile_map.json` — where each role's pixels come from
```json
{
  "schema_version": 1,
  "skin": "kenney",
  "needs_review": false,
  "source_pack": {"name": "kenney_pixel-platformer", "version": "1.2", "zip_sha256": "<64 hex>",
    "sheets": {"tiles": {"path": "app/static/assets/kenney/pixel-platformer/Tilemap/tilemap_packed.png",
                         "sha256": "<64 hex>", "frame_w": 18, "frame_h": 18, "columns": 20, "rows": 9,
                         "spacing": 0, "margin": 0},
               "characters": {"path": "...", "sha256": "<64 hex>", "frame_w": 24, "frame_h": 24, "columns": 9, "rows": 3,
                              "spacing": 0, "margin": 0}}},
  "roles": {
    "ground_fill":    {"source": {"sheet": "tiles", "frame": null}},
    "ground_top":     {"source": {"sheet": "tiles", "frame": null}},
    "brick":          {"source": {"sheet": "tiles", "frame": null}},
    "question_full":  {"source": {"sheet": "tiles", "frame": null}, "procedural_if_missing": "question_full"},
    "question_empty": {"source": {"sheet": "tiles", "frame": null}, "procedural_if_missing": "question_empty"},
    "pipe_top_left":  {"source": null, "procedural": "pipe_top_left", "palette_from": {"sheet": "tiles", "frame": null}},
    "pipe_top_right": {"source": null, "procedural": "pipe_top_right", "palette_from": {"sheet": "tiles", "frame": null}},
    "pipe_body_left": {"source": null, "procedural": "pipe_body_left", "palette_from": {"sheet": "tiles", "frame": null}},
    "pipe_body_right":{"source": null, "procedural": "pipe_body_right", "palette_from": {"sheet": "tiles", "frame": null}},
    "cannon_top":     {"source": null, "procedural": "cannon_top"},
    "cannon_support": {"source": null, "procedural": "cannon_support"},
    "coin":           {"source": {"sheet": "tiles", "frame": null}},
    "flagpole_segment": {"source": {"sheet": "tiles", "frame": null}, "procedural_if_missing": "flagpole_segment"},
    "flagpole_top":   {"source": {"sheet": "tiles", "frame": null}, "procedural_if_missing": "flagpole_top"},
    "flag_cloth":     {"source": {"sheet": "tiles", "frame": null}},
    "checkpoint_flag":{"source": {"sheet": "tiles", "frame": null}},
    "player_idle":    {"source": {"sheet": "characters", "frame": null}},
    "player_walk_a":  {"source": {"sheet": "characters", "frame": null}},
    "player_walk_b":  {"source": {"sheet": "characters", "frame": null}},
    "player_jump":    {"source": {"sheet": "characters", "frame": null}},
    "player_hurt":    {"source": {"sheet": "characters", "frame": null}, "procedural_if_missing": "tint_red:player_idle"},
    "enemy_walk_a":   {"source": {"sheet": "characters", "frame": null}},
    "enemy_walk_b":   {"source": {"sheet": "characters", "frame": null}},
    "enemy_squashed": {"source": null, "procedural": "squash:enemy_walk_a"},
    "enemy_icon":     {"source": null, "procedural": "fit_tile:enemy_walk_a"}
  },
  "palette": {"sky": "#9fd6ff", "hud_text": "#ffffff", "hud_shadow": "#000000"}
}
```
The `frame: null` values are **filled in on Day 1** by the procedure in §5.3. Shown above: `columns 20 / rows 9` and `24 px` are *expected* values; the script overwrites them with discovered ones.

### 5.3 Mapping procedure (Day 1, G track, ~1 h)
1. `python scripts/make_contact_sheet.py --sheet tiles --scale 4` → `reports/assets/contact_tiles.png`: each frame enlarged 4× with nearest-neighbour, a 1 px grid, and **its frame index printed in the corner** (white text with a black outline). Also writes `contact_tiles.html` (hover shows the index) and the same for `characters` and `backgrounds`.
2. Claude Code **looks at the contact sheets** (Read tool on the PNG) and picks a frame for each role using the criteria table below. It writes the indices into `tile_map.json`.
3. Roles with no suitable Kenney art use the **procedural** builder (pipes and cannons are expected to need this). The builder samples colours from the `palette_from` frame (for example a green bush or grass tile for pipes), so procedural tiles match the Kenney palette.
4. `python scripts/validate_tile_map.py` fails if any role has neither a `source.frame` nor a `procedural` rule, if a frame index is out of range, or if `needs_review` is true.
5. `python scripts/build_atlas.py --skin kenney` composes `t2m_tiles.png` / `t2m_entities.png` per `atlas_layout.json` (nearest-neighbour only; never smooth-scale pixel art; entity frames smaller than `entity_px` are centre-bottom padded).
6. `python scripts/gate_assets.py` runs the visual check gate (§8).

**Selection criteria**

| Role | Must read as | Selection rule |
|---|---|---|
| ground_fill / ground_top | Solid earth; top variant has a surface (grass/edge) | Pick a matching pair from the same terrain family |
| brick | Breakable-looking block, clearly different from ground | Different hue **and** pattern from ground |
| question_full / question_empty | Bonus block / used block | Full = bright, with a mark or glyph; empty = dull and flat. If no glyph tile exists, use procedural (draws a `?` glyph on a Kenney block frame) |
| pipe_* | 2-wide tube; top rim wider-looking than body | Procedural by default (Mario-style pipes are not expected in this pack); colours from a green Kenney frame |
| cannon_* | Dark launcher on a support column | Procedural (dark grey + muzzle circle) |
| coin | Small collectible | Kenney coin frame |
| enemy_* | A hostile walker, readable at 18–24 px | A Kenney enemy character. **Must not imitate a Goomba** (no brown mushroom with feet) |
| player_* | The player character | A Kenney character (not a Mario likeness) |
| flag* / checkpoint | Goal and checkpoint | Kenney flag frames; procedural pole if missing |

---

## 6. Placeholder skin — `scripts/make_placeholder_sprites.py` (A2)

Deterministic (no randomness without a fixed seed), pure PIL, so it runs offline in < 1 s. It produces the **same atlas layout** (§5.1) at any `--tile-px` (default 18; 16 and 32 supported) and `--entity-px` (default 24).

| ID | Char | Role | Base colour | Detail (drawn at 18 px; scale proportionally) | Glyph |
|---|---|---|---|---|---|
| 0 | `-` | sky | transparent | — | — |
| 1 | `<` | pipe top L | `#2E9E44` | top rim rows 0–3 lighter `#66D17A`; left 2 px dark `#1B6B2C`; rim drawn edge to edge on the right side so it joins frame 2 | — |
| 2 | `>` | pipe top R | `#2E9E44` | rim rows 0–3 `#66D17A`; right 2 px dark `#1B6B2C`; left edge continuous with frame 1 | — |
| 3 | `?` | question full | `#F2B33D` | 1 px border `#8A5A12`, corner rivets | `?` in `#FFFFFF` |
| 4 | `B` | cannon top | `#3A3A46` | muzzle circle `#15151A` (r = 5 px), 2 rivets | — |
| 5 | `E` | enemy icon | transparent bg | **purple slime** `#8E44AD`, one white eye with a black pupil (generic, non-Nintendo) | — |
| 6 | `Q` | question empty | `#9C6B3E` | 1 px border `#5E3D1F`, flat face | — |
| 7 | `S` | brick | `#C8553D` | mortar `#6E2A1E`: horizontal lines every 6 px, vertical joints offset per course | — |
| 8 | `X` | ground fill | `#8B5A2B` | speckles `#6E4420` from a fixed hash pattern | — |
| 9 | `[` | pipe body L | `#2E9E44` | highlight stripe x = 3–5 `#66D17A`; left 2 px `#1B6B2C` | — |
| 10 | `]` | pipe body R | `#2E9E44` | right 2 px `#1B6B2C` | — |
| 11 | `b` | cannon support | `#4A4A58` | corner rivets `#2A2A33` | — |
| 12 | `o` | coin | transparent bg | ellipse `#FFD54A`, rim `#B8860B` | — |
| 13 | — | ground top | `#8B5A2B` | top 4 px grass `#4CAF50`, 1 px highlight `#7BD67F` | — |
| 14–17 | — | flagpole / top / cloth / checkpoint | pole `#CFD8DC`, ball `#FFD54A`, cloth `#E53935`, checkpoint `#29B6F6` | — | — |
| 18 | — | spawn marker | `#00E5FF` outline square | — | `S` |

Entities: player = blue rounded body `#2D7FF9` with a white visor and two dark feet (a generic robot); walk frames shift the feet by 1 px; jump = feet tucked; hurt = red tint. Enemy walk frames = slime squash/stretch by 1 px; squashed = flattened to 30 % height. Coins = ellipse width 100 % / 50 %.

Rendering details: 1 px outline `#000000` at 35 % alpha on every non-sky tile (helps at small sizes); glyphs from `ImageFont.load_default()` (always available offline). Output: `app/static/assets/atlas/placeholder/t2m_tiles.png`, `t2m_entities.png`, plus a labelled preview `reports/assets/placeholder_preview.png`.

---

## 7. `assets/ASSET_MANIFEST.json` — schema and example

```json
{
  "schema_version": 1,
  "generated_utc": "2026-11-18T09:12:44Z",
  "active_skin": "kenney",
  "items": [
    {"name": "phaser", "version": "3.90.0", "licence": {"spdx": "MIT", "file": "app/static/vendor/LICENSE.phaser.md"},
     "source_used": "https://cdn.jsdelivr.net/npm/phaser@3.90.0/dist/phaser.min.js",
     "sources_tried": [{"url": "https://cdn.jsdelivr.net/npm/phaser@3.90.0/dist/phaser.min.js", "result": "ok"}],
     "files": [{"path": "app/static/vendor/phaser.min.js", "bytes": 1196122,
                "sha256": "e92ddef111ba42e92d316979c732311757093688ea1810591cb7aa2858eba7a7"}]},
    {"name": "kenney_pixel-platformer", "version": "1.2", "licence": {"spdx": "CC0-1.0",
     "file": "app/static/assets/kenney/pixel-platformer/License.txt"},
     "source_used": "https://kenney.nl/…/kenney_pixel-platformer.zip", "zip_sha256": "<64 hex>",
     "sources_tried": [{"url": "https://kenney.nl/assets/pixel-platformer", "result": "ok"}],
     "sheets": {"tiles": {"frame_w": 18, "frame_h": 18, "columns": 0, "rows": 0, "spacing": 0}},
     "files": [{"path": ".../Tilemap/tilemap_packed.png", "bytes": 0, "sha256": "<64 hex>"}]},
    {"name": "atlas:kenney", "built_by": "scripts/build_atlas.py", "inputs_sha256": ["<tile_map.json sha>", "<sheet shas>"],
     "files": [{"path": "app/static/assets/atlas/kenney/t2m_tiles.png", "bytes": 0, "sha256": "<64 hex>"}]}
  ],
  "fallbacks": []
}
```
Rules: every committed binary under `app/static/` must appear in `items[*].files`, and `tests/test_assets.py` recomputes every sha256. `fallbacks` lists any placeholder substitutions with a reason and timestamp, so an empty list means "no fallback in effect".

---

## 8. Visual check gate — `scripts/gate_assets.py` → `reports/gates/ASSETS/RESULT.json`

Runs on Day 1 (and again whenever `tile_map.json` or a skin changes). It **supports** G-P1 (renders must look right) and is a prerequisite for the Day 3 game skeleton.

**Automated checks (all must pass):**
1. **Completeness:** all 32 tile frames and 16 entity frames exist with the right sizes; every frame except sky (and reserved slots) has alpha coverage ≥ 10 %.
2. **Distinguishability:** render frames 1–13 composited over the sky colour; for every pair, the mean absolute RGB difference must be ≥ **12/255**. Pairs that are allowed to be similar: (1, 2), (9, 10), (8, 13).
3. **Contrast vs sky:** for solid frames, the luminance contrast ratio against the sky colour must be ≥ **1.5 : 1**, or the frame must have a dark outline covering ≥ 30 % of its border pixels.
4. **Pipe seams:** the mean absolute difference between the last pixel column of frame 1 and the first column of frame 2 must be ≤ **60/255** (same for 9/10, and for the bottom row of 1/2 vs the top row of 9/10). This catches mismatched halves.
5. **Determinism:** building the placeholder skin twice gives byte-identical PNGs.
6. **Licences:** each skin's inputs trace to manifest items whose SPDX is `CC0-1.0` or `MIT`, or to `own-work`. Any `local_only` input → fail if the output path is outside `assets/local_only/`.

**Human-visible evidence (Claude Code inspects these with the Read tool):**
- `reports/assets/skin_check_kenney.png` and `skin_check_placeholder.png`: the same three real levels rendered (level 0 cols 0–79; the first level containing pipes **and** cannons; one SMB2 level), with floor at the bottom, pipes upright, `<` left of `>`.
- `reports/assets/atlas_kenney_labelled.png`: every atlas frame with its role name.

`RESULT.json` = `{"pass": bool, "skin": "kenney", "checks": {...}, "git_sha": "...", "timestamp": "..."}`.

---

## 9. Sound (optional)

- **Default (zero assets):** a WebAudio synth in `game.js` (`sfx_synth`) makes short procedural sounds (square/triangle blips, noise burst for stomp). No licence question, no bytes to inline.
- **Optional samples:** Kenney **Digital Audio** (`https://kenney.nl/assets/digital-audio`, CC0), fetched by `fetch_assets.py --only audio` with the same scrape → zip → licence-check → manifest flow. Six roles: `jump, coin, stomp, bump, death, win`. Keep OGG only (Chrome/Edge targets), ≤ 25 KB each and ≤ 150 KB total. Choose the files on Day 9 and record them in `tile_map.json → "audio"`.
- Audio starts only after the "click to play" gesture (browser autoplay rules). `M` toggles mute. Default volume 0.4.

---

## 10. Renderer skins (PNG renderer, doc 11 `render/renderer.py`)

| Skin | Availability | Use |
|---|---|---|
| `kenney` (default) | Committed atlas | App, report figures, viva |
| `placeholder` | Always (generated) | Fallback; CI; colour-coded debug renders |
| `classic-local` | Only if the user places a sheet in `assets/local_only/` | Local viewing at the user's discretion (A3). **Never committed, never in report figures** unless the instructor explicitly approves, and then only rendered locally |

`grid_to_image(grid, skin="kenney", scale=2, overlays=("framing",))`:
1. Paint the sky colour (`tile_map.palette.sky`).
2. Draw tile frames by ID, with the `ground_top` variant rule (ID 8 whose upper neighbour is non-solid, or row 0).
3. Overlays: flag (14–16) at `framing.flag`, checkpoints (17), and the spawn marker (18) in debug mode only. Debug overlays (doc 20/22): window seams, provenance colouring, verifier stuck column.
4. Upscale by `scale` with nearest-neighbour.

The `classic-local` builder reads a user-supplied 16 px sheet. It refuses to write anywhere except `assets/local_only/atlas/`, and the app labels levels rendered with it "local skin".

---

## 11. Offline guarantee

1. After Day 0, every runtime asset is in git (Phaser, the Kenney subset, built atlases, the optional audio subset).
2. The game HTML is self-contained (doc 27 §10). `tests/test_offline_assets.py` parses the built HTML and fails on any `src=`, `href=` or CSS `url(` that is not `data:` or `#…`, and on any `fetch(`/`XMLHttpRequest` with an http(s) URL.
3. The Playwright smoke test (doc 27 §12) runs with **all network requests aborted**, and must still pass.
4. Day 18 rehearsal: Wi-Fi off, cold boot, `fetch_assets.py --offline` exits 0, the app plays a level.

---

## 12. Tests and acceptance

| Test | What it proves |
|---|---|
| `tests/test_assets.py::test_manifest_hashes` | Every committed asset matches its sha256 |
| `test_phaser_pinned` | `phaser.min.js` = 1,196,122 B, sha256 `e92ddef1…`; licence file present and MIT |
| `test_tile_map_complete` | `validate_tile_map.py` passes; no `null` frames |
| `test_atlas_layout_contract` | Frame indices 0–12 = F6 IDs; `solid_tile_frames` = `physics.json → solid_tile_ids` ∪ {13} |
| `test_placeholder_deterministic` | Two builds are byte-identical |
| `test_no_nintendo_assets_committed` | `git ls-files` contains no `mapsheet.png` and nothing under `assets/local_only/` |
| `test_offline_assets` | Built game HTML references no remote URLs |
| `gate_assets.py` | §8 |

**Day 0 exit:** Phaser verified; Kenney fetched (or placeholder fallback logged). **Day 1 exit:** `ASSETS` gate `pass: true` for the `kenney` skin, or for the `placeholder` skin with the fallback logged and a retry scheduled.
