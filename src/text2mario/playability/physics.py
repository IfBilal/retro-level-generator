"""Load the physics contract (app/static/game/physics.json) for the checker (doc 21 §5, doc 27 §4).

Explain-me:
  What: reads physics.json, exposes the numbers the checker needs and the sha256 that is stored in every level.
  In/Out: Physics.load() -> frozen dataclass.  The checker only READS this file (single source of truth).
  Previous: scripts/derive_physics.py writes the `derived` block.  Next: playability/checker.py.
  Try: edit jump_velocity in physics.json, run derive_physics.py, and see physics_sha256 (and the verdicts) change.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PHYSICS_PATH = ROOT / "app" / "static" / "game" / "physics.json"


@dataclass(frozen=True)
class Physics:
    raw: dict
    sha256: str
    fps: int
    gravity: float
    jump_velocity: float
    max_fall_speed: float
    walk_speed: float
    run_speed: float
    body_w: float
    body_h: float
    death_fall_margin: float
    stomp_tol: float
    solid: frozenset
    hazard: frozenset
    max_rise: int
    gap_same: dict
    gap_lower: dict
    gap_higher: dict

    @staticmethod
    def load(path: str | Path = PHYSICS_PATH) -> "Physics":
        p = json.loads(Path(path).read_text(encoding="utf-8"))
        d = p["derived"]
        sha = hashlib.sha256(json.dumps(p, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        return Physics(
            raw=p, sha256=sha, fps=p["fps"], gravity=p["gravity"], jump_velocity=p["jump_velocity"],
            max_fall_speed=p["max_fall_speed"], walk_speed=p["walk_speed"], run_speed=p["run_speed"],
            body_w=p["player"]["body_w"], body_h=p["player"]["body_h"], death_fall_margin=p["death_fall_margin"], stomp_tol=p["stomp"]["tolerance"],
            solid=frozenset(p["solid_tile_ids"]), hazard=frozenset(p["hazard_tile_ids"]),
            max_rise=d["checker_max_rise"], gap_same=d["checker_max_gap_same_height"],
            gap_lower={k: {int(a): b for a, b in v.items()} for k, v in d["checker_max_gap_landing_lower"].items()},
            gap_higher={k: {int(a): b for a, b in v.items()} for k, v in d["checker_max_gap_landing_higher"].items()},
        )
