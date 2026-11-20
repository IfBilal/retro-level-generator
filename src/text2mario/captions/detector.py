"""Caption detector: turns a tile grid into a canonical caption (doc 09 §6).

ORIGINALITY (ledger O3): written from the grammar description in docs 09 §6 / 12 §2 and the paper's published
phrase list; it is CHECKED against the stored captions (scripts/check_detector.py) but not ported from, or
transcribed out of, MarioDiffusion's code.  The dataset and the caption idea are credited in DATA_CARD.md.

Explain-me:
  What: a rule list that explains a 16xW grid as phrases (floor, ceiling, enemies, ... loose blocks), in a fixed
        order.  Cells already explained by a structure (floor, pipe, staircase...) are not counted again.
  In/Out: grid int (16, W) values 0..12 -> list[Detection] (topic, phrase, count, cells) or the caption string.
  Used by: evaluation (C-score), the generation verify loop, the future "see what it heard" feature (cells).
  Try: change TOWER_MIN_HEIGHT and see which grids switch between "tower" and "block cluster".
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from text2mario.captions.grammar import counted_phrase, gap_word, join_phrases, quantity_word
from text2mario.data import tiles as T

TOWER_MAX_WIDTH = 2
TOWER_MIN_HEIGHT = 3
STRUCTURE_MIN_CELLS = 3
CEILING_ROW_START = 4  # = H - 12 for H = 16

Cell = tuple[int, int]


@dataclass(frozen=True)
class Detection:
    topic: str
    phrase: str
    count: int | None  # None for "full floor" / "full ceiling"
    cells: frozenset[Cell]


# ---------------------------------------------------------------- small helpers
def _is_solid(v: int) -> bool:
    return v in T.SOLID


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Maximal runs of True in a 1-D bool array -> [(start, end_exclusive)]."""
    out, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        elif not m and start is not None:
            out.append((start, i))
            start = None
    if start is not None:
        out.append((start, len(mask)))
    return out


def _components(cells: set[Cell], can_step) -> list[list[Cell]]:
    """4-neighbour connected components of `cells`; can_step(a, b) may veto a step (used for pipes).
    Components are returned in row-major discovery order of their first cell."""
    seen: set[Cell] = set()
    comps = []
    for c in sorted(cells):  # row-major
        if c in seen:
            continue
        comp, q = [], deque([c])
        seen.add(c)
        while q:
            a = q.popleft()
            comp.append(a)
            for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                b = (a[0] + d[0], a[1] + d[1])
                if b in cells and b not in seen and can_step(a, b):
                    seen.add(b)
                    q.append(b)
        comps.append(comp)
    return comps


# ---------------------------------------------------------------- floor / ceiling
def _floor_phrase(row: np.ndarray):
    """Rule 2: phrase for the bottom row, or None."""
    w = len(row)
    solid = np.array([_is_solid(int(v)) for v in row])
    passable = np.array([int(v) in T.PASSABLE for v in row])
    s, p = int(solid.sum()), int(passable.sum())
    if s == w:
        return "floor", "full floor", None
    if p == w:
        return None
    if s > p:  # mostly floor: count the holes (passable OR enemy cells)
        n = len(_runs(~solid))
        return "floor", f"floor with {quantity_word(n)} {gap_word(n)}", n
    n = len(_runs(solid))  # mostly holes: count the floor chunks
    return "floor", f"giant gap with {quantity_word(n)} chunk{'' if n == 1 else 's'} of floor", n


def _ceil_info(row: np.ndarray):
    """Rule 3 helper: ('full'|gaps n, rank) for a row or None.  Rank: full < 1 gap < 2 gaps < a few ... (lower=fuller)."""
    w = len(row)
    solid = np.array([_is_solid(int(v)) for v in row])
    s = int(solid.sum())
    if s == w:
        return ("full", None), 0
    if s > w // 2:
        n = len(_runs(~solid))
        rank = {1: 1, 2: 2}.get(n, 3 if n <= 4 else 4 if n <= 9 else 5)
        return ("gaps", n), rank
    return None


def _ceiling(grid: np.ndarray):
    """Rule 3: -> (ceiling_row | None, Detection | None)."""
    for c in range(CEILING_ROW_START, 0, -1):  # c = 4, 3, 2, 1
        reg, hi = _ceil_info(grid[c]), _ceil_info(grid[c - 1])
        if reg is None and hi is None:
            continue
        if reg is not None and not (hi is not None and hi[1] <= reg[1]):
            row, info = c, reg
        else:
            row, info = c - 1, hi
        (kind, n), _ = info
        phrase = "full ceiling" if kind == "full" else f"ceiling with {quantity_word(n)} {gap_word(n)}"
        return row, Detection("ceiling", phrase, None if kind == "full" else n,
                              frozenset((row, x) for x in range(grid.shape[1])))
    return None, None


# ---------------------------------------------------------------- staircases
def _step_ok(grid: np.ndarray, r: int, c: int) -> bool:
    """A staircase step: solid cell whose upper neighbour is non-solid (not checked on row 0)."""
    h, w = grid.shape
    if not (0 <= r < h and 0 <= c < w) or not _is_solid(int(grid[r, c])):
        return False
    return r == 0 or not _is_solid(int(grid[r - 1, c]))


def _staircases(grid: np.ndarray, v: int):
    """Rule 11: count staircases going up (v=-1) or down (v=+1); returns (count, cells)."""
    h, w = grid.shape
    rows = range(3, 16) if v == -1 else range(14)
    count, cells, col = 0, set(), 0
    while col <= w - 3:
        hit = None
        for r in rows:
            if all(_step_ok(grid, r + v * k, col + k) for k in range(3)):
                hit = r
                break
        if hit is None:
            col += 1
            continue
        length = 3
        while col + length < w and _step_ok(grid, hit + v * length, col + length):
            length += 1
        count += 1
        for k in range(length):  # the step plus the solid column beneath it belong to the staircase
            rr = hit + v * k
            while rr < h and _is_solid(int(grid[rr, col + k])):
                cells.add((rr, col + k))
                rr += 1
        col += length
    return count, cells


# ---------------------------------------------------------------- pipes
def _classify_pipe(grid: np.ndarray, comp: list[Cell]) -> str:
    """Rule 12: 'pipe' | 'upside down pipe' | 'broken pipe' from the component's bounding box."""
    h, w = grid.shape
    rs = [r for r, _ in comp]
    cs = [c for _, c in comp]
    r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
    cellset = set(comp)
    width = c1 - c0 + 1
    if width > 2:
        return "broken pipe"
    # Which half-columns exist.  A 1-wide pipe is only valid when cut by a screen edge.
    left_col, right_col = c0, c1
    if width == 1:
        if c0 == 0:
            left_col, right_col = None, 0       # only the right half is visible
        elif c0 == w - 1:
            left_col, right_col = w - 1, None   # only the left half is visible
        else:
            return "broken pipe"
    rows = list(range(r0, r1 + 1))
    if len(cellset) != len(rows) * (1 if width == 1 else 2):
        return "broken pipe"

    def row_tiles(r):
        return (int(grid[r, left_col]) if left_col is not None else None,
                int(grid[r, right_col]) if right_col is not None else None)

    def is_lip(r):
        a, b = row_tiles(r)
        return a in (None, T.PIPE_TL) and b in (None, T.PIPE_TR)

    def is_body(r):
        a, b = row_tiles(r)
        return a in (None, T.PIPE_BL, T.PIPE_TL) and b in (None, T.PIPE_BR, T.PIPE_TR)

    # normal pipe: lip on top, body pairs below
    if is_lip(r0) and all(is_body(r) for r in rows[1:]):
        return "pipe"
    # upside-down pipe: body above, lip at the bottom, and the body must hang from something (row 0 or a solid)
    if is_lip(r1) and all(is_body(r) for r in rows[:-1]):
        if r0 == 0 or all(_is_solid(int(grid[r0 - 1, c])) for c in (left_col, right_col) if c is not None):
            return "upside down pipe"
    return "broken pipe"


def _pipes(grid: np.ndarray):
    """Rule 12: returns [(class, comp)] in discovery order."""
    h, w = grid.shape
    cells = {(r, c) for r in range(h) for c in range(w) if int(grid[r, c]) in T.PIPE}

    def can_step(a, b):  # never step right from a right half, nor left from a left half (keeps pipes apart)
        if b[1] == a[1] + 1 and int(grid[a]) in (T.PIPE_TR, T.PIPE_BR):
            return False
        if b[1] == a[1] - 1 and int(grid[a]) in (T.PIPE_TL, T.PIPE_BL):
            return False
        return True

    return [(_classify_pipe(grid, comp), comp) for comp in _components(cells, can_step)]


# ---------------------------------------------------------------- main
def detect(grid: np.ndarray) -> list[Detection]:
    """Rules 1-14 in emission order (doc 09 §6.2)."""
    grid = np.asarray(grid)
    h, w = grid.shape
    out: list[Detection] = []
    acc: set[Cell] = set()  # cells already explained

    # 1. floor accounting
    floor_row = h - 1
    acc |= {(h - 1, c) for c in range(w)}
    mapped14 = np.where(grid[h - 2] == T.PIPE_TL, T.PIPE_BL, np.where(grid[h - 2] == T.PIPE_TR, T.PIPE_BR, grid[h - 2]))
    if np.array_equal(mapped14, grid[h - 1]):  # double-thick floor
        floor_row = h - 2
        acc |= {(h - 2, c) for c in range(w)}
    # 2. floor phrase
    fp = _floor_phrase(grid[h - 1])
    if fp:
        out.append(Detection(fp[0], fp[1], fp[2], frozenset((h - 1, c) for c in range(w))))
    # 3. ceiling
    ceil_row, cdet = _ceiling(grid)
    if cdet:
        out.append(cdet)
        acc |= set(cdet.cells)
        for r in range(ceil_row - 1, -1, -1):  # thick ceilings: identical rows above
            if np.array_equal(grid[r], grid[ceil_row]):
                acc |= {(r, c) for c in range(w)}

    def cells_of(vals):
        return frozenset((r, c) for r in range(h) for c in range(w) if int(grid[r, c]) in vals)

    # 4. enemies, 5. question blocks, 6. cannons
    for topic, vals in (("enemy", {T.ENEMY_ID}), ("question block", {T.QBLOCK, T.QEMPTY}), ("cannon", {T.CANNON_TOP})):
        cs = cells_of(vals)
        if cs:
            out.append(Detection(topic, counted_phrase(topic, len(cs)), len(cs), cs))
            if topic == "cannon":
                acc |= cs
    # 7. broken cannons (generated samples only): support blocks with no cannon/support above, or on row 0
    broken_b = frozenset((r, c) for r in range(h) for c in range(w) if int(grid[r, c]) == T.CANNON_SUP
                         and (r == 0 or int(grid[r - 1, c]) not in T.CANNON))
    if broken_b:
        out.append(Detection("broken cannon", counted_phrase("broken cannon", len(broken_b)), len(broken_b), broken_b))
    # 8. coins
    cs = cells_of({T.COIN})
    if cs:
        out.append(Detection("coin", counted_phrase("coin", len(cs)), len(cs), cs))
    # 9. coin lines: horizontal runs of >= 2 coins in rows 0..14
    lines, line_cells = 0, set()
    for r in range(h - 1):
        for a, b in _runs(grid[r] == T.COIN):
            if b - a >= 2:
                lines += 1
                line_cells |= {(r, c) for c in range(a, b)}
    if lines:
        out.append(Detection("coin line", counted_phrase("coin line", lines), lines, frozenset(line_cells)))
    # 10. platforms: rows 1..14 (not the ceiling row); runs >= 2 of solid non-pipe cells with free space above and below
    plat, plat_cells = 0, set()
    for r in range(1, h - 1):
        if r == ceil_row:
            continue
        cand = np.array([_is_solid(int(grid[r, c])) and int(grid[r, c]) not in T.PIPE
                         and not _is_solid(int(grid[r - 1, c])) and not _is_solid(int(grid[r + 1, c])) for c in range(w)])
        for a, b in _runs(cand):
            if b - a >= 2:
                plat += 1
                plat_cells |= {(r, c) for c in range(a, b)}
    if plat:
        out.append(Detection("platform", counted_phrase("platform", plat), plat, frozenset(plat_cells)))
        acc |= plat_cells
    # 11. staircases
    for topic, v in (("ascending staircase", -1), ("descending staircase", 1)):
        n, scells = _staircases(grid, v)
        if n:
            out.append(Detection(topic, counted_phrase(topic, n), n, frozenset(scells)))
            acc |= scells
    # 12. pipes, grouped per class in discovery order
    groups: dict[str, list[Cell]] = {}
    counts: dict[str, int] = {}
    for cls, comp in _pipes(grid):
        groups.setdefault(cls, []).extend(comp)
        counts[cls] = counts.get(cls, 0) + 1
        acc |= set(comp)
    for cls, cells in groups.items():
        out.append(Detection(cls, counted_phrase(cls, counts[cls]), counts[cls], frozenset(cells)))
    # 13. solid structures (>= 3 cells, solid, non-pipe, unexplained)
    scells = {(r, c) for r in range(h) for c in range(w)
              if _is_solid(int(grid[r, c])) and int(grid[r, c]) not in T.PIPE and (r, c) not in acc}
    sgroups: dict[str, list[list[Cell]]] = {}
    for comp in _components(scells, lambda a, b: True):
        if len(comp) < STRUCTURE_MIN_CELLS:
            continue
        rs = [r for r, _ in comp]
        cs_ = [c for _, c in comp]
        r0, r1, c0, c1 = min(rs), max(rs), min(cs_), max(cs_)
        bw, bh = c1 - c0 + 1, r1 - r0 + 1
        touches_ceiling = ceil_row is not None and r0 <= ceil_row
        if (not touches_ceiling and bw <= TOWER_MAX_WIDTH and bh >= TOWER_MIN_HEIGHT and r1 >= floor_row - 1):
            cls = "tower"
        elif len(comp) == bw * bh:
            cls = "rectangular block cluster"
        else:
            cls = "irregular block cluster"
        sgroups.setdefault(cls, []).append(comp)
    for cls, comps in sgroups.items():
        flat = [c for comp in comps for c in comp]
        out.append(Detection(cls, counted_phrase(cls, len(comps)), len(comps), frozenset(flat)))
        acc |= set(flat)
    # 14. loose blocks
    loose = frozenset((r, c) for r in range(h) for c in range(w)
                      if int(grid[r, c]) in (T.GROUND, T.BREAKABLE) and (r, c) not in acc)
    if loose:
        out.append(Detection("loose block", counted_phrase("loose block", len(loose)), len(loose), loose))
    return out


def caption(grid: np.ndarray) -> str:
    """The canonical caption of a grid ('' if nothing is detected)."""
    return join_phrases([d.phrase for d in detect(grid)])
