"""Tier-2 reference solver: MarioEval.jar (Baumgarten A*) run as a separate process (doc 21 §1, §7).

Explain-me:
  What: writes a level as VGLC text, runs the third-party jar in its own JVM and parses "beaten:true/false".  The jar is
        only ever executed (never imported, never needed by the app; CLAUDE.md rule 10, NFR-SEC-04).  Used for the
        G-P5b agreement study and the jar subset of the one-shot test.
  In/Out: jar_solve(grid) -> True / False / None (None = timeout or jar error);  jar_solve_many(grids, workers).
  Previous: render/renderer.grid_to_vglc_text.  Next: scripts/gate_p5b.py.
  Try: workers=1 vs 6 in jar_solve_many to see the speed-up; a 60 s timeout counts as 'not solved by the jar'.
Setup: the jar loads its sprite images from the working directory, so we unpack it once into data/cache/marioeval/.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from text2mario.render.renderer import grid_to_vglc_text

ROOT = Path(__file__).resolve().parents[3]
JAR_SRC = ROOT / "data" / "raw" / "MarioDiffusion" / "MarioEval.jar"
CACHE = ROOT / "data" / "cache" / "marioeval"
_LOCK = threading.Lock()
JAR_MAIN = "edu.southwestern.tasks.mario.level.MarioLevelUtil"


def java_available() -> bool:
    return shutil.which("java") is not None


def _prepare() -> Path:
    """Unpack the jar (and the inner MarioEval.jar) once; returns the working directory the JVM runs in."""
    with _LOCK:  # several worker threads may call this at once
        if not (CACHE / ".done").exists():
            CACHE.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(JAR_SRC) as z:
                z.extractall(CACHE)
            (CACHE / ".done").write_text("ok")
    return CACHE


def jar_solve(grid: np.ndarray, timeout: float = 90.0) -> bool | None:
    """Run the A* agent on the level; True = beaten, False = not beaten, None = timeout / jar failure."""
    work = _prepare()
    sep = ";" if os.name == "nt" else ":"
    cp = sep.join([".", "MarioEval.jar", "gson-2.8.9.jar", "jdom-1.1.jar", "vecmath-1.5.2.jar"])
    with tempfile.NamedTemporaryFile("w", suffix=".txt", dir=work, delete=False) as f:
        f.write(grid_to_vglc_text(np.asarray(grid)) + "\n")
        name = Path(f.name).name
    try:
        p = subprocess.run(["java", "-cp", cp, JAR_MAIN, "astar", name, "false"], cwd=work, capture_output=True,
                           text=True, timeout=timeout)
        out = p.stdout
        if "beaten:true" in out:
            return True
        if "beaten:false" in out:
            return False
        return None
    except subprocess.TimeoutExpired:
        return None
    finally:
        try:
            os.unlink(work / name)
        except OSError:
            pass


def jar_solve_many(grids, workers: int = 4, timeout: float = 90.0) -> list:
    """Parallel batch; order of results = order of grids."""
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(lambda g: jar_solve(g, timeout), grids))
