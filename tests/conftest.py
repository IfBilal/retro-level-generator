"""Shared fixtures. Data-dependent tests skip cleanly if the data has not been fetched yet."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
# The tests run in ONE process whose text2mario modules are current.  The app's drop_stale_modules() guard (for a redeploy into a running server) would otherwise
# purge them on the first AppTest run and leave other test modules holding old module objects (isinstance / monkeypatch mismatches).  Declare them current.
sys._t2m_src_signature = max((f.stat().st_mtime_ns for f in (ROOT / "src").rglob("*.py")), default=0)

needs_data = pytest.mark.skipif(not (ROOT / "data/processed/index.parquet").exists(),
                                reason="run scripts/fetch_data.py and scripts/build_index.py first")


@pytest.fixture
def batch_via_single(monkeypatch):
    """Stub-sampler tests replace FL.sample_window; the batched sampler must then be emulated by n calls of that stub (distinct seeds),
    otherwise it would run the real model.  Opt in with pytestmark = pytest.mark.usefixtures("batch_via_single")."""
    import numpy as np

    from text2mario.generate import full_level as FL

    def emulate(bundle, caption, w, steps, r, gen_seed, known_cols, device, n):
        return np.stack([FL.sample_window(bundle, caption, w, steps, r, gen_seed + i, known_cols, device) for i in range(n)])

    monkeypatch.setattr(FL, "sample_window_batch", emulate)
