"""Tier-1 playability checker: A* over standing cells with simulated jump arcs (doc 21, doc 27 §4.4).

Explain-me:
  What: decides whether a level can be crossed from the spawn to the goal using only moves a human can also do
        (walk, walk-off, fixed-speed walk/run jumps).  Jumps are simulated with the game's integrator (gravity, wall
        blocking, head bumps, landing), then limited by the margins in physics.json -> conservative "verified".
  In/Out: check_level(grid[16,W], mode, framing, physics) -> CheckResult(status, path, fail_col, nodes, seconds, sha).
  Previous: render/renderer (grid) + playability/physics.py.  Next: eval (playable rate), generate/full_level (resample).
  Try: change budget_nodes (configs/gates.yaml) or jump_velocity in physics.json + derive_physics.py.
Coordinates: x to the right, y DOWN, in tiles; cell (col,row) spans [col,col+1) x [row,row+1).  A standing cell is a
free cell whose cell below is solid.  Left/right grid edges are walls; the top is open; below the grid is a pit.
"""
from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from text2mario.playability.physics import Physics

EPS = 1e-6


@dataclass
class Framing:
    spawn: tuple[int, int]   # (col,row) standing cell
    flag_col: int            # success = standing at a column >= flag_col


@dataclass
class CheckResult:
    status: Literal["solved", "unsolved", "timeout"]
    path: list = field(default_factory=list)   # [(col,row,move)]
    fail_col: int | None = None                # furthest column reached
    nodes: int = 0
    seconds: float = 0.0
    physics_sha256: str = ""


class _World:
    """Tile lookups with the checker's edge rules."""

    def __init__(self, grid: np.ndarray, ph: Physics, enemies: str = "ignore"):
        self.h, self.w = grid.shape
        self.ph = ph
        self.skip = None  # an enemy cell just stomped: it is gone for the moves that start from it
        self.solid = [[int(grid[r, c]) in ph.solid for c in range(self.w)] for r in range(self.h)]
        # enemies='hazard': enemy cells are deadly unless stomped from above; 'ignore': enemies can be killed/avoided
        # (the model MarioEval's A* also assumes).  Chosen by measured agreement with the jar (doc 21 §7, ledger).
        use = ph.hazard if enemies == "hazard" else frozenset()
        self.haz = [[int(grid[r, c]) in use for c in range(self.w)] for r in range(self.h)]

    def is_solid(self, c: int, r: int) -> bool:
        if c < 0 or c >= self.w:
            return True          # side walls
        if r < 0 or r >= self.h:
            return False         # open top, pit below
        return self.solid[r][c]

    def haz_cells(self, x: float, feet: float, hw: float, bh: float):
        """Hazard (enemy) cells overlapped by the body box."""
        c0, c1 = math.floor(x - hw + EPS), math.floor(x + hw - EPS)
        r0, r1 = math.floor(feet - bh + EPS), math.floor(feet - EPS)
        return [(c, r) for c in range(c0, c1 + 1) for r in range(r0, r1 + 1) if self.is_haz(c, r)]

    def col_has_haz(self, c: int) -> bool:
        return 0 <= c < self.w and any(self.haz[r][c] for r in range(self.h))

    def is_haz(self, c: int, r: int) -> bool:
        return 0 <= c < self.w and 0 <= r < self.h and self.haz[r][c] and (c, r) != self.skip

    def free(self, c: int, r: int) -> bool:
        return not self.is_solid(c, r) and not self.is_haz(c, r)

    def standable(self, c: int, r: int) -> bool:
        return 0 <= c < self.w and 0 <= r < self.h - 1 and self.free(c, r) and self.is_solid(c, r + 1)

    def box_hits(self, x: float, feet: float, hw: float, bh: float, haz: bool = False) -> bool:
        """Does the body box (centre x, feet y) overlap any solid cell (or hazard cell if haz)?"""
        c0, c1 = math.floor(x - hw + EPS), math.floor(x + hw - EPS)
        r0, r1 = math.floor(feet - bh + EPS), math.floor(feet - EPS)
        f = self.is_haz if haz else self.is_solid
        for c in range(c0, c1 + 1):
            for r in range(r0, r1 + 1):
                if f(c, r):
                    return True
        return False


def _simulate(wd: _World, c: int, r: int, dirn: int, speed: float, x_end: float, vy0: float):
    """Fly from the centre of standing cell (c,r) toward x_end at `speed` (vy0 > 0 = jump, 0 = walk-off).
    Returns the landing cell (lc,lr) or None if the move is impossible (hazard hit, death fall, bad landing)."""
    ph = wd.ph
    dt, g, hw, bh = 1.0 / ph.fps, ph.gravity, ph.body_w / 2, ph.body_h
    x, feet, vy = c + 0.5, float(r + 1), vy0
    if vy0 > 0:  # a jump starts as near the cell edge as the walls/hazards allow (doc 27 §4.3: travel = gap - body_w)
        for off in (0.5, 0.2, 0.0):
            cand = c + 0.5 + dirn * off
            if not wd.box_hits(cand, feet, hw, bh) and not wd.box_hits(cand, feet, hw, bh, haz=True):
                x = cand
                break
    for _ in range(int(10 * ph.fps)):
        # horizontal: move toward x_end; a solid in the way blocks the move (wall)
        step = speed * dt
        nx = x_end if dirn * (x_end - x) <= step else x + dirn * step
        if not wd.box_hits(nx, feet, hw, bh):
            x = nx
        # vertical: semi-implicit Euler with symmetric clamp (same as the game, doc 27 §4.3)
        vy = max(vy - g * dt, -ph.max_fall_speed)
        nfeet = feet - vy * dt
        if vy > 0:  # rising: a ceiling stops the rise (head bump)
            if wd.box_hits(x, nfeet, hw, bh):
                top_row = math.floor(nfeet - bh + EPS)
                nfeet = top_row + 1 + bh + 2 * EPS
                vy = 0.0
        else:
            hz = wd.haz_cells(x, nfeet, hw, bh)
            if hz:  # touching an enemy: only a STOMP from above (centre over it, feet within tolerance) is survivable
                col, row = hz[0]
                if len(hz) == 1 and col == math.floor(x) and 0 <= nfeet - row <= ph.stomp_tol and wd.is_solid(col, row + 1):
                    return (col, row)     # enemy squashed; the player stands in its cell afterwards
                return None
        if vy <= 0 and wd.box_hits(x, nfeet, hw, bh):  # falling into something: land on the top of that solid row
            lr = math.floor(nfeet - bh * 0 - EPS)          # row of the solid that the feet entered
            feet = float(lr)                               # feet rest on its top edge
            lc, row = math.floor(x), lr - 1                # standing cell is the one above that solid
            if wd.box_hits(x, feet, hw, bh, haz=True):
                return None
            return (lc, row) if wd.standable(lc, row) else None
        feet = nfeet
        if wd.box_hits(x, feet, hw, bh, haz=True):
            return None
        if feet - bh > wd.h + ph.death_fall_margin:  # fully below the pit margin -> death
            return None
    return None


def _moves(wd: _World, c: int, r: int, safety_gap: int = 0):
    """All successor standing cells of (c,r): (cell, move_name, cost_seconds)."""
    ph = wd.ph
    out = []
    wd.skip = (c, r) if wd.haz[r][c] else None
    for dirn in (-1, 1):
        nc = c + dirn
        if wd.free(nc, r):
            if wd.standable(nc, r):
                out.append(((nc, r), "walk", 1.0 / ph.walk_speed))
            else:  # walk off the edge: fall straight down the next column
                land = _simulate(wd, c, r, dirn, ph.walk_speed, nc + 0.5, 0.0)
                if land is not None:
                    out.append((land, "drop", 1.0 / ph.walk_speed + 0.3))
        for sc, speed in (("walk", ph.walk_speed), ("run", ph.run_speed)):
            for dx in range(1, ph.gap_same[sc] + 3):
                # aim at the NEAR edge of the landing cell (+ a hair): the body then rests on it, margins are in the envelopes
                lc_ = c + dirn * dx
                inset = 0.01 if not wd.col_has_haz(lc_ - dirn) else 0.31  # clear a hazard right behind the landing cell
                x_end = (lc_ + inset) if dirn > 0 else (lc_ + 1 - inset)
                land = _simulate(wd, c, r, dirn, speed, x_end, ph.jump_velocity)
                if land is None:
                    continue
                lc, lr = land
                adx, dy = abs(lc - c), r - lr           # dy > 0: landing higher
                if adx == 0 or dy > ph.max_rise:
                    continue
                gap = adx - 1                           # empty columns crossed
                if dy > 0:
                    ok = gap <= ph.gap_higher[sc][min(dy, 4)] - safety_gap
                elif dy == 0:
                    ok = gap <= ph.gap_same[sc] - safety_gap
                else:
                    ok = gap <= ph.gap_lower[sc][min(-dy, 4)] - safety_gap
                if ok:
                    out.append(((lc, lr), f"jump_{sc}", 0.85))
    wd.skip = None
    return out


def check_level(grid: np.ndarray, *, mode: Literal["scene", "level"] = "scene", framing: Framing | None = None,
                physics: Physics | None = None, budget_nodes: int = 20000, noise_profile: float = 0.0,
                enemies: str = "ignore", safety_gap: int = 0) -> CheckResult:
    """A* from the spawn to the goal.  Timeouts count as unsolved (reported as status 'timeout')."""
    t0 = time.perf_counter()
    ph = physics or Physics.load()
    wd = _World(np.asarray(grid), ph, enemies)
    if mode == "level":
        assert framing is not None, "level mode needs framing"
        starts, goal_col = [tuple(framing.spawn)], framing.flag_col
    else:
        starts = [(0, r) for r in range(wd.h) if wd.standable(0, r)]
        goal_col = wd.w - 1
    if not starts:
        return CheckResult("unsolved", [], 0, 0, time.perf_counter() - t0, ph.sha256)

    def h(cell):  # remaining columns at run speed: never overestimates the time to the goal
        return max(0, goal_col - cell[0]) / ph.run_speed

    heap, best, parent, cnt = [], {}, {}, 0
    for s in starts:
        best[s] = 0.0
        heapq.heappush(heap, (h(s), 0.0, cnt, s))
        cnt += 1
    furthest, nodes = max(s[0] for s in starts), 0
    while heap:
        _f, g, _, cell = heapq.heappop(heap)
        if g > best.get(cell, 1e18) + 1e-12:
            continue
        nodes += 1
        furthest = max(furthest, cell[0])
        if cell[0] >= goal_col:
            path, cur = [], cell
            while cur in parent:
                prev, mv = parent[cur]
                path.append((cur[0], cur[1], mv))
                cur = prev
            path.append((cur[0], cur[1], "start"))
            return CheckResult("solved", path[::-1], furthest, nodes, time.perf_counter() - t0, ph.sha256)
        if nodes > budget_nodes:
            return CheckResult("timeout", [], furthest, nodes, time.perf_counter() - t0, ph.sha256)
        for nxt, mv, cost in _moves(wd, *cell, safety_gap):
            ng = g + cost
            if ng < best.get(nxt, 1e18) - 1e-12:
                best[nxt] = ng
                parent[nxt] = (cell, mv)
                heapq.heappush(heap, (ng + h(nxt), ng, cnt, nxt))
                cnt += 1
    return CheckResult("unsolved", [], furthest, nodes, time.perf_counter() - t0, ph.sha256)


def frame_window(g: np.ndarray, lead: int = 3, tail: int = 6, floor_id: int = 8):
    """Wrap a content grid with the flat spawn / end columns of doc 20 (G3) -> (framed grid, Framing).
    Both the fast checker and the jar judge the SAME framed level, so spawn/goal conventions cannot differ."""
    g = np.asarray(g)
    pad = lambda n: np.concatenate([np.zeros((g.shape[0] - 1, n), g.dtype), np.full((1, n), floor_id, g.dtype)])
    framed = np.concatenate([pad(lead), g, pad(tail)], axis=1)
    return framed, Framing(spawn=(1, g.shape[0] - 2), flag_col=framed.shape[1] - 2)
