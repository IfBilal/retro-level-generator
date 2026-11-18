# 27 — Browser Game Engine ("Type it, play it")

> **Owner track:** G. **Days:** D3 (skeleton plays a real level), D9 (mechanics), D13 (Streamlit embedding, MVP), D18 (hardening).
> **Ledger rows:** S2 (MVP), V5 (Phaser 3.90.0), A1–A4 (assets), G3 (framing), G5 (verification badge), G-P7 (app gate), F6 (tile IDs).
> **Companion docs:** 21 (playability checker, which **must** read the same `physics.json`), 26 (atlas contract), 28 (Streamlit app), 29 (level JSON schema).

---

## 1. What "real game" means here (acceptance)

A level produced by the model opens in the browser as a **playable platformer**:
- The player runs and jumps with the keyboard, collides with solid tiles, collects coins, bumps blocks, stomps enemies, dies in pits, respawns at checkpoints, and wins at the flag.
- It works **offline**, inside the Streamlit page (iframe) **and** as a downloadable single HTML file.
- It uses **exactly the physics the playability checker assumes** (§4). So "verified beatable" in the app means "a human can beat it with these controls".
- It runs at 60 fps on the RTX 4050 laptop in Chrome and Edge (G-P7).

## 2. Architecture

```
app/static/game/
  template.html     # HTML shell with placeholders (§10); no external URLs
  game.js           # all game code, plain ES2019, no build step, target ≤ 1,500 lines
  physics.json      # THE shared physics constants (§4). Also read by src/text2mario/playability/checker.py
app/static/vendor/phaser.min.js            # Phaser 3.90.0 (MIT), vendored (doc 26)
app/static/assets/atlas/<skin>/t2m_tiles.png, t2m_entities.png   # built atlases (doc 26 §5.1)
assets/atlas_layout.json                   # frame contract (doc 26 §5.1)
src/text2mario/app/game_html.py            # build_game_html(): inlines everything into ONE string (§10)
scripts/derive_physics.py                  # simulates the integrator, writes physics.json["derived"] (§4.3)
tests/e2e/test_game_smoke.py               # Playwright (§12)
```
If doc 05 places the Python builder elsewhere, doc 05 wins for location; the function contract stays the same.

**Runtime data flow:**
`level dict (schema v1, doc 29)` → `build_game_html(level, skin, options)` → one HTML string. It holds a JSON payload (level subset + physics + atlas layout + options), Phaser inlined, atlas PNGs as `data:` URIs, and `game.js`. That string goes to `st.iframe(...)` in the app (doc 28), or is saved as `t2m_<level_id>.html`.

**Coordinate system:** tile `(col, row)`, with row 0 at the top and row 15 at the bottom (same as the grid). World pixels: `x = col·tile_px`, `y = row·tile_px`, y pointing down (Phaser convention). `physics.json` uses tiles and seconds, with up = positive jump velocity. `game.js` converts at load (`px = tiles·tile_px`; gravity is +y in Phaser).

## 3. Scenes and states

| Scene | Responsibility | Enters | Leaves |
|---|---|---|---|
| `Boot` | Parse the JSON payload; load atlas spritesheets from data URIs (`this.load.spritesheet(key, dataURI, {frameWidth, frameHeight})`); optional audio; create animations; show the **"Click to play"** overlay | page load | click → `Play` |
| `Play` | Build tilemap, entities, colliders, camera, DOM HUD; main loop | `Boot`, restarts | death → restart(`checkpoint`); flag → launch `Win`; out of lives → launch `Lose` |
| `Win` | Pauses `Play`; DOM overlay with time, coins, deaths; `R` = play again | flag reached | `R` → `Play` from spawn |
| `Lose` | Only when `options.lives` is an integer (default `null` = unlimited, demo-friendly) | deaths ≥ lives | `R` → `Play` from spawn |

Pause is a **state of `Play`** (`this.physics.pause()` + DOM overlay), toggled by `P`/`Esc`, and entered automatically on window blur.

**Persistent state** (`this.registry`): `deaths`, `elapsed_ms` (excludes paused time), `checkpoint_idx`, `coins_at_checkpoint`, `won`. Restarting from a checkpoint rebuilds the whole level from the original grid (blocks, enemies and coins reset), as in the original game. Coins reset to `coins_at_checkpoint`.

## 4. Physics — `app/static/game/physics.json` (single source of truth)

### 4.1 The file (v1 values)
```json
{
  "schema_version": 1,
  "units": {"length": "tile", "time": "s"},
  "tile_px": 18,
  "fps": 60,
  "integrator": "arcade_semi_implicit_euler_per_axis_clamp",
  "gravity": 50.0,
  "jump_velocity": 21.5,
  "jump_cut_velocity": 8.0,
  "max_fall_speed": 21.5,
  "walk_speed": 4.0,
  "run_speed": 6.0,
  "ground_accel": 40.0,
  "air_accel": 25.0,
  "turn_accel": 80.0,
  "ground_drag": 50.0,
  "air_drag": 10.0,
  "coyote_time": 0.10,
  "jump_buffer": 0.10,
  "player": {"body_w": 0.6, "body_h": 0.9},
  "enemy": {"speed": 1.5, "body_w": 0.8, "body_h": 0.8, "walks_off_edges": true, "start_dir": -1,
            "activation_margin_tiles": 2},
  "stomp": {"tolerance": 0.4, "bounce_velocity": 12.0, "bounce_velocity_held": 18.0},
  "respawn_invulnerability": 1.0,
  "death_fall_margin": 1.0,
  "brick": {"breakable": false},
  "cannon": {"fire_enabled": false},
  "solid_tile_ids": [1, 2, 3, 4, 6, 7, 8, 9, 10, 11],
  "passable_tile_ids": [0, 12],
  "sprite_tile_ids": {"enemy": 5, "coin": 12},
  "hazard_tile_ids": [5],
  "checker_margins": {"horizontal": 0.5, "vertical": 0.25},
  "derived": {
    "generated_by": "scripts/derive_physics.py",
    "apex_rise_full": 4.444, "apex_time_s": 0.417, "airtime_same_height_s": 0.850,
    "jump_distance_walk": 3.40, "jump_distance_run": 5.10, "min_hop_rise": 0.919,
    "run_up_to_full_speed": 0.45,
    "checker_max_rise": 4,
    "checker_max_gap_same_height": {"walk": 3, "run": 5},
    "checker_max_gap_landing_lower": {"walk": {"1": 3, "2": 3, "3": 4, "4": 4}, "run": {"1": 5, "2": 5, "3": 5, "4": 6}},
    "checker_max_gap_landing_higher": {"walk": {"1": 3, "2": 3, "3": 2, "4": 2}, "run": {"1": 4, "2": 4, "3": 4, "4": 3}}
  }
}
```
**Hash:** `physics_sha256 = sha256(canonical JSON of the whole file)` (sorted keys, `separators=(",",":")`). The checker writes it into every level (`verification.checker.physics_sha256`, doc 29). If the game's own hash differs from the level's, the HUD shows "verified with different physics" instead of "verified beatable".

### 4.2 Why these numbers (design targets)
| Target (from the plan) | Achieved | How |
|---|---|---|
| Max jump height ≈ 4 tiles | Apex **4.44 tiles**; checker allows **4** | `v0 = 21.5`, `g = 50` → continuous `v0²/2g = 4.62`; the 60 Hz discrete integrator gives 4.444 (§4.3). The 0.44-tile (8 px) margin makes a 4-high ledge humanly reachable; 5-high is impossible |
| Max gap ≈ 4–5 tiles | Checker: **3 (walking), 5 (running)** at the same height | Airtime 0.85 s × run 6.0 tiles/s = 5.10 tiles of travel; crossing a k-wide gap needs `k − body_w` tiles of travel; the checker subtracts a 0.5-tile human timing margin |
| Snappy, readable feel | Apex reached in 0.417 s | Short airtime; variable height via jump-cut (tap ≈ 0.92 tile, full ≈ 4.44) |
| Fits 1-tile corridors | Body 0.6 × 0.9 tiles (11 × 16 px at 18 px tiles) | Height < 1 tile; width < 1 tile, so the player can drop through 1-wide holes |
| No tunnelling | Max per-step displacement 6.45 px < `tileBias` 16 px | `max_fall_speed = jump_velocity = 21.5 tiles/s` → 387 px/s ÷ 60 |
| Native Arcade semantics | `body.setMaxVelocity(run_px, 387)` | Arcade clamps each axis **symmetrically** (`Clamp(v, −max, max)`, verified in the 3.90.0 source). So `max_fall_speed` must equal `jump_velocity`, or the jump would be clipped. A test enforces equality |

### 4.3 Exact integrator (verified against the Phaser 3.90.0 source)
Arcade with `fixedStep: true, fps: 60` advances physics in fixed steps of `dt = 1/60 s`, independent of the monitor refresh rate (the laptop may run at 144 Hz). Each step, `World.computeVelocity` does: `v += gravity·dt`, then **either** `v += accel·dt` **or** linear drag toward 0 if accel is 0, then `v = clamp(v, −max, +max)` per axis. `Body.update` then does `pos += v·dt`. That is **semi-implicit Euler with a per-axis symmetric clamp**.

`scripts/derive_physics.py` re-implements exactly this loop in tile units (y up), simulates jumps, and writes `derived`:
- **Full jump:** `v = 21.5`; each step `v −= 50/60`, `y += v/60`. Apex after **25 steps** (0.417 s) at **4.444 tiles**; back to the take-off height after **51 steps** (0.850 s).
- **Horizontal reach:** at walk 4.0 → 3.40 tiles; at run 6.0 → 5.10 tiles (horizontal speed held constant in the air).
- **Gap rule** for the checker: `gap_max = floor(reach + body_w − margin_h)`, with `margin_h = 0.5` → walk 3, run 5. Landing lower or higher shifts the reach (table in `derived`).
- **Rise rule:** `rise_max = floor(apex − margin_v)`, with `margin_v = 0.25` → 4.
- Values exactly on a floor boundary are resolved by the script's arithmetic, never by hand. `tests/test_physics_contract.py` re-derives them and fails if `physics.json` is stale.

### 4.4 Checker contract (doc 21 MUST honour these; this doc owns them)
1. The checker loads `app/static/game/physics.json` and uses only `derived.checker_*` envelopes (or simulates the §4.3 integrator itself with the same margins). **No physics constants are hard-coded in the checker.**
2. Solid set = `solid_tile_ids` (bricks are **permanently solid**: `brick.breakable = false`; question blocks stay solid after being hit). Passable = `passable_tile_ids` (sky, coin). Enemy cells are hazards with no collision against the level.
3. The checker does **not** rely on coyote time, jump buffering or mid-air steering. It uses fixed-speed walk/run arcs plus walk-offs, so its moves are a **subset** of what a human can do (conservative: a "beatable" verdict is a promise).
4. Pits: falling below row 15 + `death_fall_margin` = death. The left and right world edges are walls; the top is open (the player may jump above row 0).
5. Cannons are solid and **do not fire** (`cannon.fire_enabled = false`). Enabling bullets later requires a checker update **and** a physics hash change.
6. Success = reaching the flag column `framing.flag.col` (doc 29) from `framing.spawn`.

### 4.5 Physics calibration and freeze (protects the evaluation)
- **Day 5–6:** checker v0 runs on all real levels. Calibration target: ≥ **90 %** of the **train + dev** real levels (full width) are judged beatable, and every failure is inspected and classified (VGLC artifact, springboard/warp the tile set can't express, or physics too strict).
- If physics is too strict, tune **only** within `jump_velocity ∈ [20.5, 23.0]` and `run_speed ∈ [5.5, 7.0]`, re-derive, and record the change in the ledger change log. **Never look at test levels for this** (that would be test-set leakage via the checker).
- **Freeze physics by Day 9**, before playability enters the dev composite score (E4) in Phase 4 model selection. Changing physics after the freeze invalidates every stored playability number, and the evaluation must be re-run.

## 5. Level → tilemap (Phaser 3.90 APIs, verified in the source)

```js
// grid: 16 x cols ints (F6 IDs). frames: atlas_layout (doc 26 §5.1) — frames 0..12 == tile IDs.
const TP = PHYS.tile_px;
const SOLID = new Set(PHYS.solid_tile_ids);
const data = grid.map((row, r) => row.map((id, c) => {
  if (id === 0 || id === 5 || id === 12) return -1;            // sky, enemy, coin -> no tile
  if (id === 8 && (r === 0 || !SOLID.has(grid[r - 1][c]))) return 13;  // ground_top render variant
  return id;
}));
const map   = this.make.tilemap({ data, tileWidth: TP, tileHeight: TP });
const tiles = map.addTilesetImage('t2m_tiles', 't2m_tiles', TP, TP, 0, 0); // (name, key, w, h, margin, spacing)
const layer = map.createLayer(0, tiles, 0, 0);                              // (layerID, tileset, x, y)
layer.setCollision([...PHYS.solid_tile_ids, 13]);                           // 13 = ground_top variant
```
- Enemy cells (ID 5) → spawn enemy sprites, feet at the cell bottom. Coin cells (ID 12) → coin sprites in a static group (overlap, not collide).
- Framing objects come from `level.framing` (doc 29), **not** the grid: the flagpole (frames 14/15 + cloth 16) at `flag.col` from `row_top` to `row_bottom`; checkpoint flags (frame 17) at `checkpoints[1:]`; the player spawns at `spawn` (feet on the top of `spawn.row + 1`).
- World bounds: `this.physics.world.setBounds(0, −8·TP, cols·TP, (16 + 16)·TP)` and `setBoundsCollision(true, true, false, false)` (left/right walls; open top and bottom). The player and enemies use `setCollideWorldBounds(true)`.

## 6. Player controller (`game.js`)

**Input abstraction (MUST):** all control reads go through `Input.read() → {left, right, jump, jumpPressed, run}`. The default source is the keyboard (§8). A replay source (§11, reserved for the future ghost bot) can substitute without touching game logic.

**Per frame (scene `update(time, delta)`):**
1. `onGround = body.blocked.down`. If on the ground → `lastGroundTime = time`.
2. If `jumpPressed` → `lastJumpPressTime = time`.
3. **Jump** if `(time − lastJumpPressTime) ≤ jump_buffer` **and** `(onGround || time − lastGroundTime ≤ coyote_time)` and not already rising from this press: `body.setVelocityY(−jump_velocity_px)`; clear both timers; sfx `jump`.
4. **Jump cut:** if `!jump && body.velocity.y < −jump_cut_px` → `body.setVelocityY(−jump_cut_px)`.
5. **Horizontal:** `dir = right − left`; `maxV = (run ? run_speed : walk_speed)`, then `body.setMaxVelocity(maxV_px, jump_velocity_px)`. If `dir ≠ 0`: the acceleration is `turn_accel` when `dir` opposes the current velocity, otherwise `ground_accel` (on the ground) or `air_accel` (in the air); `body.setAccelerationX(dir·a_px)`; `setDragX(0)`. If `dir = 0`: `setAccelerationX(0)`; `setDragX(onGround ? ground_drag_px : air_drag_px)`.
6. **Animation:** idle / walk (frame rate ∝ |vx|) / jump; flip X by facing direction.
7. **Bump detection:** if `body.blocked.up && !wasBlockedUp` → `bumpTileAbove()` (§7.2).
8. **Checkpoints:** when `body.center.x ≥ checkpoints[i].col·TP` for `i > checkpoint_idx` → activate (frame swap, `coins_at_checkpoint = coins`).
9. **Death by pit:** `body.top > (16 + death_fall_margin)·TP` → `die('pit')`.
10. **Win:** overlap with the flag zone `[flag.col·TP, (flag.col+1)·TP] × [row_top·TP, (row_bottom+1)·TP]` → `win()`.

Body setup: `player.body.setSize(round(0.6·TP), round(0.9·TP))`, offset so the body is horizontally centred and its bottom aligns with the sprite's feet. The sprite frame (24 px) may be larger than the body; only the body matters for physics and for the checker.

## 7. World mechanics

### 7.1 Enemies (decision: walk off edges, as in the original)
- Arcade sprites; body `0.8 × 0.8` tiles; velocity `start_dir · speed` (−1 = towards the player's usual approach); world gravity applies.
- **Turn** on `body.blocked.left/right` (walls, pipes, cannons) and on enemy–enemy collision (group self-collider; both reverse). **Do not turn at ledges** (`walks_off_edges = true`): enemies fall into pits and are destroyed below the world.
- **Activation:** frozen (`body.enable = false`, not animating) until `enemy.x < camera.scrollX + camera.width + activation_margin·TP`, so they don't wander off before the player arrives.
- **Stomp vs hit** (player ↔ enemy overlap callback): stomp if `player.body.velocity.y > 0` **and** `player.body.prev.y + player.body.height ≤ enemy.body.prev.y + stomp.tolerance·TP`. Then: enemy → `enemy_squashed` frame, body disabled, destroyed after 0.5 s; player `vy = −(jumpHeld ? bounce_velocity_held : bounce_velocity)`; sfx `stomp`. Otherwise → `die('enemy')`, unless invulnerable (1.0 s after respawn, the sprite flickers).
- Enemies standing on a block that gets bumped from below are killed (SHOULD, nice-to-have).
- The checker treats enemies conservatively (§4.4.2). Doc 21 decides how (for example, enemy cells plus their reachable patrol span are hazardous unless they can be jumped over).

### 7.2 Blocks
- `bumpTileAbove()`: pick the tile under `(body.center.x, body.top − 2)`. If empty, try `body.left + 1` and then `body.right − 1`. Process **one** tile per bump.
  - `3` (question full) → `layer.putTileAt(6, tx, ty)` (becomes empty), spawn a `coin_pop` sprite that rises 1.5 tiles over 250 ms and is destroyed, `coins += 1`, sfx `coin`.
  - `7` (brick) → bump animation only: hide the tile (`tile.setAlpha(0)`), tween a sprite copy up 4 px and back over 120 ms, restore alpha, sfx `bump`. **Never breaks** (`brick.breakable = false`).
  - Any other solid tile → sfx `bump` (quiet).
- Pipes (1, 2, 9, 10), ground (8), empty question (6) and cannons (4, 11) are plain solid tiles.

### 7.3 Coins, pits, death, respawn, flag
- Coins: overlap → destroy, `coins += 1`, sfx `coin`.
- `die(cause)`: freeze input, `player_hurt` frame, disable collisions, `vy = −12 tiles/s`, let it fall off-screen; after 0.9 s → `deaths += 1`, telemetry event, then `Lose` if out of lives, else `scene.restart({checkpoint: checkpoint_idx})`.
- `win()`: freeze, tween the player down the pole to `row_bottom` (400 ms), sfx `win`, `this.scene.launch('Win', stats)`, `this.scene.pause()`.

### 7.4 Camera and view
- View = `min(cols, 24)` tiles wide × 16 tiles tall at native resolution (432 × 288 px at 18 px tiles). Game config `scale: {mode: Phaser.Scale.FIT, autoCenter: Phaser.Scale.CENTER_BOTH, width: VIEW_W, height: VIEW_H}`, `pixelArt: true`, `roundPixels: true`. The iframe's height is set by the app (doc 28: default 600 px).
- `cameras.main.setBounds(0, 0, cols·TP, 16·TP)`; `startFollow(player, true, 0.12, 0.12)`; `setDeadzone(0.15·VIEW_W, VIEW_H)`.
- Background: sky colour from `tile_map.palette.sky` (sent in the payload). Optional Kenney background tiles as a parallax layer (`scrollFactor 0.3`) if present in the atlas (reserved frames).

## 8. Input, focus, HUD, pause, restart, audio

| Action | Keys |
|---|---|
| Move | ← → or A D |
| Jump | Space, ↑, W, Z, K |
| Run | Shift, X, J |
| Pause / resume | P, Esc |
| Restart from spawn | R |
| Mute | M |
| Debug hitboxes (debug builds only) | F1 |

- Capture keys so the page doesn't scroll: `this.input.keyboard.addCapture('UP,DOWN,LEFT,RIGHT,SPACE')`. Gamepad is optional (`input: {gamepad: true}`: d-pad/stick move, A jump, X run).
- **Click-to-focus overlay (MUST):** a DOM `<button id="t2m-focus">` covers the canvas with "▶ Click to play" and the controls legend. On click: hide the overlay, `window.focus()`, `canvas.setAttribute('tabindex','0'); canvas.focus()`, unlock audio (`this.sound.unlock()` / resume the AudioContext), and start or resume `Play`. On `blur` or `visibilitychange` → pause and show "Click to resume". An iframe only receives keys while focused, so without this the demo silently ignores the keyboard.
- **HUD as DOM, not canvas text** (crisp at any scale, no font loading): `#t2m-hud` absolutely positioned over the canvas, updated at **10 Hz** via `textContent` only. Left: `🪙 03  ⏱ 00:42  💀 1`. Right: badge from `level.verification.status`: `✓ verified beatable` (green), `⚠ not verified` (amber), `— not checked` (grey). Bottom (first 5 s): controls hint.
- **Untrusted text:** the user's prompt appears only via `textContent` (HUD title, `Win` overlay), **never** `innerHTML`. It reaches the page only inside the escaped JSON payload (§10.2).
- Audio: WebAudio synth by default (doc 26 §9); optional OGG samples as data URIs; default volume 0.4; `M` toggles mute.

## 9. Telemetry and debug hooks (reserved seams)

- `window.T2M_TELEMETRY` (array, capped at 1,000) receives events `{type: "t2m:event", v: 1, level_id, event, t_ms, col, row, cause, coins, deaths}` for `start | checkpoint | death | win | restart | pause`.
- If `options.telemetry === true` (default **false**), each event is also sent with `window.parent.postMessage(evt, "*")`. A Streamlit Components-v2 bridge can listen later; this is the seam for the difficulty dial and endless mode (doc 31).
- `window.T2M_DEBUG` is always present, with `version`, `levelId` and `getState() → {ready, scene, x, y, vx, vy, col, row, on_ground, alive, coins, deaths, checkpoint, won, paused, fps}`. Positions and velocities are in tiles (feet position), for tests.
- With `options.debug === true` (tests only, never in the app or standalone downloads): also `teleport(col, row)`, `setVirtualInput({left, right, jump, run})`, `showBodies(bool)`, and `replay(actions)` (reserved, §11).

## 10. Embedding into one HTML string

### 10.1 Builder contract
```python
def build_game_html(level: dict, *, skin: str = "kenney", sound: str = "synth",   # "synth" | "samples" | "off"
                    lives: int | None = None, debug: bool = False, telemetry: bool = False,
                    mode: str = "inline") -> str:   # "inline" (default) | "standalone" | "static"
    """Deterministic: the same inputs give byte-identical output (no timestamps, sorted JSON keys).
    That lets Streamlit keep the iframe mounted across reruns (doc 28 §6)."""
```

### 10.2 Template and escaping rules
```html
<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>$TITLE_ESCAPED</title><style>$CSS</style></head>
<body><div id="t2m-root"><div id="t2m-game"></div><div id="t2m-hud"></div>
<button id="t2m-focus" type="button">▶ Click to play</button></div>
<script id="t2m-data" type="application/json">$PAYLOAD_JSON</script>
<script>/* Phaser v3.90.0 | (c) Richard Davey, Phaser Studio Inc. | MIT */$PHASER_JS</script>
<script>window.T2M_ASSETS=$ASSETS_JSON;</script>
<script>$GAME_JS</script></body></html>
```
- `$PAYLOAD_JSON = json.dumps(payload, sort_keys=True, separators=(",",":"), ensure_ascii=False)` followed by `.replace("</", "<\\/")`, `.replace(" ", "\\u2028")` and `.replace(" ", "\\u2029")`. `game.js` reads it with `JSON.parse(document.getElementById('t2m-data').textContent)`. The payload is never executed as script.
- The payload holds `{level_id, grid: level.grid.data, cols, framing, verification: {status, physics_sha256}, prompt: meta.prompt, physics, atlas_layout, palette, options}`. Only what the game needs (≈ 8–12 KB for a 121-column level).
- `$TITLE_ESCAPED = html.escape("text2mario — " + prompt[:80])`.
- `$PHASER_JS` is inlined **raw** (it has no `</script`, verified for 3.90.0; the builder asserts this at build time). If that ever fails (a future version), fall back to base64 plus `new Function(atob(...))()` and log it.
- `$ASSETS_JSON` = `{"tiles": "data:image/png;base64,…", "entities": "data:image/png;base64,…", "sfx": {…}}`.
- `Boot.preload` loads data URIs through the normal loader (`this.load.spritesheet('t2m_tiles', A.tiles, {frameWidth: TP, frameHeight: TP})`), so no async `addBase64` race.

### 10.3 Size budget (asserted by `tests/test_game_html.py`)
| Part | Bytes |
|---|---|
| Phaser 3.90.0 (raw inline) | 1,196,122 |
| `game.js` | ≤ 60,000 |
| Atlases (base64) | ≤ 60,000 |
| Payload (121 cols) | ≤ 20,000 |
| Optional OGG samples (base64) | ≤ 200,000 |
| **Total** | **≤ 1.40 MB** (synth/off) · **≤ 1.60 MB** (samples) |

### 10.4 Alternative "static" mode (optional optimisation, not MVP)
With `server.enableStaticServing = true`, Streamlit serves `app/static/*` at `/app/static/...`. In `mode="static"`, the template references `/app/static/vendor/phaser.min.js`, `/app/static/game/game.js` and the atlas URLs (cached by the browser), and the level travels in the URL fragment: `st.iframe("/app/static/game/index.html#lvl=<base64url(json)>")`.
Pitfall (Windows): Streamlit picks Content-Type with Python's `mimetypes` and sends `X-Content-Type-Options: nosniff`. A Windows registry mapping of `.js` → `text/plain` makes the browser refuse to run the script. Fix: `mimetypes.add_type("text/javascript", ".js")` at app start. Inline mode avoids this whole class of problem, which is why it is the default.

## 11. Standalone download and the replay seam

- `mode="standalone"` → the same game, plus a header (prompt via `textContent`, level ID, verification badge), a "Download level JSON" link (`data:application/json` of the full schema-v1 level), a controls legend and a `<noscript>` notice. File name `t2m_<level_id>.html`. It opens by double-click, offline. Debug API disabled.
- **Replay (reserved for future feature F1, "watch the proof"):** `T2M_DEBUG.replay(actions)`, where `actions` is an array of per-physics-step bitmasks (`1 = left, 2 = right, 4 = jump, 8 = run`). It feeds the input abstraction (§6) one entry per fixed step (listen on `this.physics.world.on('worldstep')`), so the checker's path can be played back exactly. The MVP **must** ship the input abstraction; implementing `replay` is SHOULD (Day 19 buffer).

## 12. Tests

### 12.1 Python unit tests (fast)
- `test_physics_contract.py`: `derived` equals a fresh `derive_physics.py` run (3 decimals); `max_fall_speed == jump_velocity`; per-step displacement `< 16 px`; `solid_tile_ids` equals the atlas `solid_tile_frames` minus `{13}` (doc 26); `physics_sha256` is stable across runs.
- `test_game_html.py`: deterministic output (build twice → identical); size budget; the JSON payload round-trips; a prompt containing `</script><img onerror=…>` comes out escaped; no remote URLs (doc 26 §11).

### 12.2 Playwright smoke test (`tests/e2e/test_game_smoke.py`, `@pytest.mark.slow`)
Setup: `pip install playwright` and `playwright install chromium` (one-time, needs internet; done on Day 13). Build fixtures with `debug=True`: (a) real level 0, cols 0–63, framed; (b) a flat 40-column calibration level; (c) one showcase level. Write each to a temp file and open it with `page.goto("file://…")`. **Every non-`file:`/`data:` request is aborted**, and the test asserts that zero such requests were attempted.
1. No `pageerror`, no console errors; a `<canvas>` exists; `T2M_DEBUG.getState().ready` within 5 s.
2. Click the overlay. Hold `ArrowRight` for 1.0 s → Δx ≈ 4.0 tiles (± 25 %; acceleration ramp). With `Shift` → ≈ 6.0 tiles (± 25 %).
3. **Calibration (flat level):** from standstill, hold `Space` → max rise **4.44 ± 0.15** tiles; airtime **0.85 ± 0.06 s**. Running jump horizontal distance **5.10 ± 0.25** tiles. This proves the game matches `physics.json`, and therefore the checker.
4. **Win path:** `teleport(flag.col − 3, spawn.row)`, hold right → `won === true` within 3 s.
5. **Death/respawn:** teleport above a pit → `deaths` increments, and the player reappears at checkpoint 0 within 2 s.
6. **Focus:** after `page.evaluate("window.dispatchEvent(new Event('blur'))")` → `paused === true`.
7. Soft performance check: `fps ≥ 30` headless (the real 60 fps target is checked manually on the laptop, Day 18).
Run on Chromium; also with `channel="msedge"` when Edge is installed (Windows laptop).

## 13. Performance budget
- 60 fps on the laptop in Chrome/Edge with a 121 × 16 level, ≤ 128 enemies, ≤ 512 coins (larger counts are capped and logged).
- One static tilemap layer (Phaser culls off-camera tiles). Arcade colliders: player↔layer, enemies↔layer, enemies↔enemies, player↔enemies (overlap), player↔coins (overlap), player↔flag zone (overlap).
- No per-frame allocations in `update`; HUD at 10 Hz; physics fixed at 60 Hz.
- Time to interactive < 1.5 s after the iframe loads; tab memory < 150 MB.

## 14. Day-by-day deliverables
| Day | Deliverable | Check |
|---|---|---|
| D3 | `physics.json` + `derive_physics.py`; skeleton: tilemap from a real level, move/jump/collide, camera | Plays level 0; `test_physics_contract` green |
| D9 | Enemies, blocks, coins, pits, checkpoints, flag/win, DOM HUD, pause/restart, synth audio | Manual playthrough of 3 real levels; e2e tests 1–6 green |
| D13 | `build_game_html` inline + standalone; Streamlit embed (doc 28); badge | G-P7 timing; e2e on 20 generated levels |
| D18 | Offline, browser matrix, 768p layout | Day 18 checklist (doc 03) |
