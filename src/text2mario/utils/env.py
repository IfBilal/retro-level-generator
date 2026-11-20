"""Process bootstrap: offline flags and UTF-8 (doc 05, NFR-OFF-01). Call FIRST in every entry point.

Explain-me:
  What: sets HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE so nothing ever downloads at run time (models are fetched once by
        scripts/fetch_models.py), and PYTHONUTF8 for Windows consoles.
"""
from __future__ import annotations

import os


def bootstrap_env() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
