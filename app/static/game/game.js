/* text2mario game (doc 27): reads window.__LEVEL__ (doc 29 schema), window.__PHYSICS__ (physics.json) and
 * window.__ATLAS__ (base64 PNG).  Physics numbers are in tiles / seconds and converted to pixels here (TP px per tile).
 * Test hooks: window.__t2m = {state(), input, runPath(path)} (used by the Playwright smoke test and the ghost-bot
 * validation of the playability checker).  Explain-me: change physics.json (jump_velocity) and rebuild to feel it. */
(function () {
  "use strict";
  var L = window.__LEVEL__, P = window.__PHYSICS__, TP = P.tile_px;
  var ROWS = L.size.rows, COLS = L.size.cols, GRID = L.display_grid;
  var SOLID = P.solid_tile_ids, ZOOM = 2;
  var T = { SKY: 0, QFULL: 3, ENEMY: 5, QEMPTY: 6, COIN: 12 };
  var S = { coins: 0, attempts: 1, deaths: 0, won: false, dead: false, time: 0, x: 0, y: 0, grounded: false, started: false };
  var hooks = { input: null, bot: null };
  window.__t2m = { state: function () { return S; }, input: hooks, runPath: function (p) { hooks.bot = makeBot(p); } };

  var img = new Image();
  img.onload = function () { new Phaser.Game(config()); };
  img.src = "data:image/png;base64," + window.__ATLAS__;

  function config() {
    return { type: Phaser.CANVAS, width: 800, height: ROWS * TP * ZOOM, parent: "game", backgroundColor: "#9ccfff",
      pixelArt: true, physics: { default: "arcade", arcade: { gravity: { y: P.gravity * TP }, fps: P.fps, fixedStep: true } },
      scene: { create: create, update: update } };
  }

  var player, layer, map, enemies, coins, cursors, keys, spawn, flagX, hud, msg, scene, lastGround = 0, jumpBuf = 0;

  function frame(tex, name, x, y, w, h) { tex.add(name, 0, x, y, w, h); }

  function create() {
    scene = this;
    var tex = this.textures.addImage("atlas", img);
    var x = TP * 13;
    ["player_idle", "player_walk", "player_jump", "enemy_a", "enemy_b"].forEach(function (n) { frame(tex, n, x, 0, 24, 24); x += 24; });
    x += 24;
    ["coin", "flag", "pole", "ground_top"].forEach(function (n) { frame(tex, n, x, 0, TP, TP); x += TP; });

    // tile data without sprite tiles (enemies and coins become sprites)
    var data = GRID.map(function (row) { return row.map(function (t) { return (t === T.ENEMY || t === T.COIN) ? 0 : t; }); });
    map = this.make.tilemap({ data: data, tileWidth: TP, tileHeight: TP });
    var ts = map.addTilesetImage("atlas", "atlas", TP, TP, 0, 0);
    layer = map.createLayer(0, ts, 0, 0);
    layer.setCollision(SOLID);
    for (var r = 0; r < ROWS; r++) for (var c = 0; c < COLS; c++)   // grass on exposed ground
      if (GRID[r][c] === 8 && (r === 0 || SOLID.indexOf(GRID[r - 1][c]) < 0)) this.add.image(c * TP, r * TP, "atlas", "ground_top").setOrigin(0, 0).setDepth(1);

    this.physics.world.setBounds(0, -TP * 40, COLS * TP, ROWS * TP + TP * 40 + TP * 6);
    this.physics.world.setBoundsCollision(true, true, false, false);   // side walls; open top; pit at the bottom

    spawn = L.framing.spawn; flagX = L.framing.flag.col * TP;
    player = this.physics.add.sprite((spawn.col + 0.5) * TP, (spawn.row + 1) * TP - 1, "atlas", "player_idle");
    player.setOrigin(0.5, 1).setDepth(5);
    var bw = Math.round(P.player.body_w * TP), bh = Math.round(P.player.body_h * TP);
    player.body.setSize(bw, bh).setOffset((24 - bw) / 2, 24 - bh);
    player.setCollideWorldBounds(true);
    player.body.setMaxVelocity(P.run_speed * TP, P.max_fall_speed * TP);
    this.physics.add.collider(player, layer, onTile, null, this);

    enemies = this.physics.add.group(); coins = this.physics.add.staticGroup();
    for (r = 0; r < ROWS; r++) for (c = 0; c < COLS; c++) {
      if (GRID[r][c] === T.ENEMY) {
        var e = enemies.create((c + 0.5) * TP, (r + 1) * TP - 1, "atlas", "enemy_a").setOrigin(0.5, 1);
        e.body.setSize(Math.round(P.enemy.body_w * TP), Math.round(P.enemy.body_h * TP)).setOffset(3, 24 - Math.round(P.enemy.body_h * TP));
        e.body.enable = false; e.dir = P.enemy.start_dir; e.active2 = false;
      } else if (GRID[r][c] === T.COIN) { coins.create((c + 0.5) * TP, (r + 0.5) * TP, "atlas", "coin"); }
    }
    this.physics.add.collider(enemies, layer);
    this.physics.add.collider(enemies, enemies, function (a, b) { a.dir = -a.dir; b.dir = -b.dir; });
    this.physics.add.overlap(player, enemies, onEnemy, null, this);
    this.physics.add.overlap(player, coins, function (p, k) { k.destroy(); S.coins++; });

    // flag pole
    var f = L.framing.flag;
    for (r = f.row_top; r <= f.row_bottom; r++) this.add.image(flagX + TP / 2, r * TP, "atlas", "pole").setOrigin(0.5, 0).setDepth(2);
    this.add.image(flagX + TP, f.row_top * TP, "atlas", "flag").setOrigin(0, 0).setDepth(2);

    var cam = this.cameras.main; cam.setZoom(ZOOM); cam.setBounds(0, 0, COLS * TP, ROWS * TP); cam.startFollow(player, true, 0.15, 0.15);
    cursors = this.input.keyboard.createCursorKeys();
    keys = this.input.keyboard.addKeys({ a: "A", d: "D", w: "W", space: "SPACE", shift: "SHIFT", r: "R", p: "P" });
    var v = L.verification || {};
    var badge = v.status === "verified" ? "Route found by checker (not a guarantee)" : (v.status === "different_physics" ? "Checked under different physics" : "Not verified");
    hud = this.add.text(4, 4, "", { fontSize: "9px", color: "#ffffff", backgroundColor: "#00000088" }).setScrollFactor(0).setDepth(20);
    hud.setScale(1 / 1);
    msg = this.add.text(COLS ? 100 : 0, 60, "", { fontSize: "14px", color: "#fff", backgroundColor: "#000000aa" }).setScrollFactor(0).setDepth(21);
    this.badge = badge;
    // click-to-focus overlay (keyboard needs the canvas to have focus)
    var ov = document.getElementById("focus");
    if (ov) { ov.onclick = function () { ov.style.display = "none"; S.started = true; scene.game.canvas.focus(); }; }
    else S.started = true;
    window.__t2m.scene = this;
  }

  function onTile() { /* hook kept for clarity: bumping is detected in update() via blocked.up */ }

  function onEnemy(p, e) {
    if (S.won || S.dead || !e.body.enable) return;
    var stomp = p.body.velocity.y > 0 && (p.body.prev.y + p.body.height) <= (e.body.prev.y + P.stomp.tolerance * TP);
    if (stomp) {
      e.body.enable = false; e.setTint(0x888888).setScale(1, 0.4); scene.time.delayedCall(500, function () { e.destroy(); });
      p.body.setVelocityY(-(keyJump() ? P.stomp.bounce_velocity_held : P.stomp.bounce_velocity) * TP);
    } else if (!(scene.time.now < S.invulnUntil)) die("enemy");
  }

  function keyJump() {
    var i = hooks.input; if (i && i.jump !== undefined) return !!i.jump;
    return cursors.up.isDown || keys.w.isDown || keys.space.isDown;
  }
  function ctl() {
    var i = hooks.input, b = hooks.bot ? hooks.bot.tick(S) : null; if (b) i = b;
    if (i) return { l: !!i.left, r: !!i.right, j: !!i.jump, run: !!i.run };
    return { l: cursors.left.isDown || keys.a.isDown, r: cursors.right.isDown || keys.d.isDown, j: keyJump(), run: keys.shift.isDown };
  }

  function die(why) {
    if (S.dead || S.won) return;
    S.dead = true; S.deaths++; S.attempts++; player.body.setVelocity(0, -P.jump_velocity * TP * 0.6); player.setTint(0xff6666);
    scene.time.delayedCall(700, function () {
      var cp = L.framing.spawn, cps = (L.framing.checkpoints || []);
      var best = cps.filter(function (c) { return c.col * TP <= S.maxX; }).pop();
      var col = best ? best.col : cp.col, row = cp.row;
      player.clearTint(); player.body.reset((col + 0.5) * TP, (row + 1) * TP - 1); S.dead = false; S.invulnUntil = scene.time.now + P.respawn_invulnerability * 1000;
    });
  }

  function update(time, delta) {
    if (!S.started) return;
    var b = player.body, c = ctl(), dt = delta / 1000;
    if (Phaser.Input.Keyboard.JustDown(keys.r)) { location.reload(); }
    if (S.won) { b.setAcceleration(0, 0); return; }
    S.time += dt; S.x = b.center.x / TP; S.y = b.bottom / TP; S.maxX = Math.max(S.maxX || 0, b.center.x); S.grounded = b.blocked.down; S.vx = b.velocity.x / TP; S.vy = b.velocity.y / TP;
    var grounded = b.blocked.down;
    if (grounded) lastGround = time;
    if (!S.dead) {
      var cap = (c.run ? P.run_speed : P.walk_speed) * TP, dir = (c.r ? 1 : 0) - (c.l ? 1 : 0);
      b.setMaxVelocityX(cap);
      if (dir !== 0) {
        var acc = (Math.sign(b.velocity.x) === -dir && Math.abs(b.velocity.x) > 5) ? P.turn_accel : (grounded ? P.ground_accel : P.air_accel);
        b.setAccelerationX(dir * acc * TP); player.setFlipX(dir < 0);
        if (Math.abs(b.velocity.x) > cap) b.setVelocityX(dir * cap);
      } else { b.setAccelerationX(0); b.setDragX((grounded ? P.ground_drag : P.air_drag) * TP); }
      if (c.j) jumpBuf = time;
      if ((time - jumpBuf) <= P.jump_buffer * 1000 && (time - lastGround) <= P.coyote_time * 1000 && !S.jumpHeld) {
        b.setVelocityY(-P.jump_velocity * TP); jumpBuf = -1e9; lastGround = -1e9; S.jumpHeld = true;
      }
      if (!c.j) { S.jumpHeld = false; if (b.velocity.y < -P.jump_cut_velocity * TP) b.setVelocityY(-P.jump_cut_velocity * TP); }
      player.setFrame(!grounded ? "player_jump" : (Math.abs(b.velocity.x) > 8 ? ((Math.floor(time / 120) % 2) ? "player_walk" : "player_idle") : "player_idle"));
      // question block bump from below
      if (b.blocked.up) {
        var tx = Math.floor(b.center.x / TP), ty = Math.floor((b.top - 2) / TP), t = map.getTileAt(tx, ty, true, layer);
        if (t && t.index === T.QFULL) { map.putTileAt(T.QEMPTY, tx, ty, true, layer); layer.setCollision(SOLID); S.coins++; S.bumps = (S.bumps || 0) + 1; }
      }
      if (S.invulnUntil && time < S.invulnUntil) player.setAlpha((Math.floor(time / 80) % 2) ? 0.4 : 1); else player.setAlpha(1);
      if (b.top > (ROWS + P.death_fall_margin) * TP) die("pit");
      if (b.center.x >= flagX) { S.won = true; msg.setText("LEVEL CLEAR! coins " + S.coins + "  attempts " + S.attempts).setX(60); }
    }
    enemies.children.iterate(function (e) {
      if (!e || !e.active) return;
      if (!e.active2 && e.x < scene.cameras.main.scrollX + scene.cameras.main.width / ZOOM + P.enemy.activation_margin_tiles * TP) { e.active2 = true; e.body.enable = true; }
      if (!e.body.enable) return;
      if (e.body.blocked.left) e.dir = 1; else if (e.body.blocked.right) e.dir = -1;
      e.body.setVelocityX(e.dir * P.enemy.speed * TP);
      if (e.y > (ROWS + 2) * TP) e.destroy();
    });
    hud.setText("time " + S.time.toFixed(0) + "   coins " + S.coins + "   attempts " + S.attempts + "   " + scene.badge);
  }

  // ---- ghost bot: follows a checker path [[col,row,move],...] by pressing keys (validation of 'verified', doc 21) ----
  function makeBot(path) {
    var i = 1, phase = "approach", jumpStarted = 0, bot = { ranup: false, backing: false, backX: 0 };
    function solidAt(c, r) { return c < 0 || c >= COLS || (r >= 0 && r < ROWS && SOLID.indexOf(GRID[r][c]) >= 0); }
    function supported(c, r) { return c >= 0 && c < COLS && SOLID.indexOf(GRID[r][c]) < 0 && r + 1 < ROWS && SOLID.indexOf(GRID[r + 1][c]) >= 0; }
    return { tick: function (s) {
      s.botIdx = i;
      var cur = [Math.floor(s.x), Math.round(s.y) - 1];                         // the cell we really stand in (closed loop)
      while (s.grounded && i < path.length && cur[0] >= path[i][0] && cur[1] <= path[i][1]) {   // already at/past this step
        i++; phase = "approach"; bot.ranup = false; bot.stepFrom = null;
      }
      if (i >= path.length) return { left: false, right: true, run: false };   // path done: keep walking to the flag
      if (!bot.stepFrom && s.grounded) bot.stepFrom = cur;
      var tgt = path[i], from = bot.stepFrom || path[i - 1], dirn = Math.sign(tgt[0] - from[0]) || 1, cx = s.x * TP, move = tgt[2];
      var run = move.indexOf("run") >= 0, tx = (tgt[0] + 0.5) * TP;
      if (move === "walk" || move === "drop") {
        if (Math.abs(cx - tx) < 3 && (s.grounded || move === "drop")) { i++; phase = "approach"; bot.ranup = false; bot.stepFrom = null; return { left: false, right: false }; }
        return { left: dirn < 0, right: dirn > 0, run: false };
      }
      // jumps: run to the cell edge, jump, hold the direction until we land near the target
      // take-off point: the cell edge, or 0.2 tiles inside it when a wall touches that edge (same rule as the checker)
      var off = solidAt(from[0] + dirn, from[1]) ? 0.2 : 0.5;
      var edge = (from[0] + 0.5 + dirn * off) * TP;
      if (phase === "approach" && !bot.ranup) {   // take a run-up: back off up to 3 supported cells first
        var back = 0;
        for (var k = 1; k <= 3; k++) { if (!supported(from[0] - dirn * k, from[1])) break; back = k; }
        bot.backX = (from[0] + 0.5 - dirn * back) * TP; bot.ranup = true; bot.backing = back > 0;
      }
      if (bot.backing) {
        if ((dirn > 0 && cx <= bot.backX + 2) || (dirn < 0 && cx >= bot.backX - 2)) bot.backing = false;
        else return { left: dirn > 0, right: dirn < 0, run: false, jump: false };
      }
      if (phase === "approach") {
        if ((dirn > 0 && cx >= edge - 2) || (dirn < 0 && cx <= edge + 2)) { phase = "air"; jumpStarted = s.time; }
        return { left: dirn < 0, right: dirn > 0, run: run, jump: false };
      }
      if (s.grounded && s.time - jumpStarted > 0.15) { phase = "approach"; bot.ranup = false; bot.stepFrom = null; i++; return { left: false, right: false }; }
      var past = dirn > 0 ? cx >= tx : cx <= tx;
      // once over the target, brake with the opposite key (the game allows this air control) instead of drifting past a narrow platform
      var moving = (s.vx || 0) * dirn > 0.8;
      return { left: dirn < 0 ? (!past) : (past && moving), right: dirn > 0 ? (!past) : (past && moving), run: run, jump: s.time - jumpStarted < 0.9 };
    } };
  }
})();
