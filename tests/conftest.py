"""Shared fixtures. Data-dependent tests skip cleanly if the data has not been fetched yet."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

needs_data = pytest.mark.skipif(not (ROOT / "data/processed/index.parquet").exists(),
                                reason="run scripts/fetch_data.py and scripts/build_index.py first")
