"""Shared fixtures. Data-dependent tests skip cleanly if the data has not been fetched yet."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

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
