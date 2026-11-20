"""GPU-resident window dataset with the ±1 one-hot representation and a locked test split (doc 11 §4).

Explain-me:
  What: holds all windows of the requested levels as uint8 on the device (about 1.5 MB), converts a batch to the
        model space x0 = 2*onehot - 1 on the fly, and attaches augmented captions.  The test split is physically
        unreachable until FROZEN.json exists (ledger D4).
  In/Out: MarioWindows(split=...) -> .grids, .batch(rows, step) -> Batch(x0, grid, captions, is_null, ...).
  Previous module: splits.py (which levels).  Next: the trainer / evaluation.
  Try: MarioWindows("test") raises PermissionError (that is the point).  Change DataConfig.aug.null_p (drill 12).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from text2mario.captions.grammar import caption_set
from text2mario.data import lexicon as lexicon_mod
from text2mario.data import provenance, splits
from text2mario.data.augment import AugConfig, augment_caption, split_phrases

ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class DataConfig:
    scenes_path: str = "data/processed/scenes_u8.npy"
    index_path: str = "data/processed/index.parquet"
    dtype: str = "float32"          # x0 dtype ("bfloat16" for GPU training under autocast)
    max_len: int = 48               # M7 token budget, asserted by scripts/measure_tokens.py
    seed: int = 20261117
    aug: AugConfig = field(default_factory=AugConfig)


@dataclass
class Batch:
    x0: torch.Tensor            # (B, 13, 16, 16) in {-1, +1}
    grid: torch.Tensor          # (B, 16, 16) uint8 (metrics / debugging)
    captions: list[str]         # post-augmentation; "" = null caption
    is_null: torch.Tensor       # (B,) bool
    rows: torch.Tensor          # (B,) dataset rows
    level_id: torch.Tensor      # (B,) int16
    aug: list[dict] | None = None


class MarioWindows:
    """All windows of the requested levels, resident on `device` as uint8."""

    def __init__(self, split: Literal["train", "dev", "test", "all"] | None = None, *,
                 levels: Sequence[int] | None = None, device: str | torch.device = "cpu",
                 cfg: DataConfig = DataConfig(), allow_test: bool = False, frozen_path: Path | None = None):
        provenance.verify()
        S = splits.load(verify=True)
        if levels is None:
            if split is None:
                raise ValueError("give split= or levels=")
            levels = list(range(42)) if split == "all" else S["splits"][split]
        levels = sorted(int(l) for l in levels)
        touches_test = set(levels) & set(S["splits"]["test"])
        frozen = Path(frozen_path) if frozen_path else ROOT / "FROZEN.json"
        if touches_test and not (allow_test and frozen.exists()):
            raise PermissionError("Test split is locked until FROZEN.json exists (D4)")
        self.cfg, self.levels, self.device = cfg, levels, torch.device(device)
        df = pd.read_parquet(ROOT / cfg.index_path)
        scenes = np.load(ROOT / cfg.scenes_path)
        keep = df.level_id.isin(levels).to_numpy()
        sel = df[keep].reset_index(drop=True)
        reused_all = {r["idx"] for r in __import__("json").loads((ROOT / "data/splits/reused_segments.json").read_text())}
        self.grids = torch.from_numpy(scenes[keep]).to(self.device)                       # one transfer
        self.level_id = torch.from_numpy(sel.level_id.to_numpy().astype(np.int16)).to(self.device)
        self.col_offset = torch.from_numpy(sel.col_offset.to_numpy().astype(np.int16)).to(self.device)
        self.game = torch.from_numpy((sel.game == "SMB2").to_numpy().astype(np.int8)).to(self.device)
        self.source_idx = torch.from_numpy(sel.idx.to_numpy().astype(np.int32)).to(self.device)
        self.reused = torch.tensor([int(i) in reused_all for i in sel.idx], device=self.device)
        self.captions: list[str] = sel.caption.tolist()
        self.phrases = [tuple(split_phrases(c)) for c in self.captions]
        self.df = sel
        self._lex = lexicon_mod.load()

    def __len__(self) -> int:
        return len(self.captions)

    def onehot(self, rows: torch.Tensor, dtype: torch.dtype | None = None) -> torch.Tensor:
        """(B,) row indices -> (B, 13, 16, 16) in {-1, +1}.  Exact in bf16 (±1 are representable)."""
        dtype = dtype or getattr(torch, self.cfg.dtype)
        g = self.grids[rows.to(self.device)].long()
        return F.one_hot(g, 13).permute(0, 3, 1, 2).to(dtype).mul_(2).sub_(1)

    def eval_rows(self, stride: int = 8) -> torch.Tensor:
        """Rows whose window starts at a column that is a multiple of `stride` (D6), in level order."""
        m = (self.col_offset % stride == 0).cpu().numpy()
        order = np.lexsort((self.col_offset.cpu().numpy(), self.level_id.cpu().numpy()))
        return torch.from_numpy(np.array([i for i in order if m[i]], dtype=np.int64))

    def batch(self, rows: torch.Tensor, step: int, *, uncond: bool = False) -> Batch:
        caps, null, records = [], [], []
        for i, r in enumerate(rows.tolist()):
            if uncond:  # Phase 3: every caption is the null caption
                caps.append("")
                null.append(True)
                records.append({"null": True})
                continue
            text, rec = augment_caption(self.captions[r], self.cfg.seed, step, i, self.cfg.aug, self._lex)
            caps.append(text)
            null.append(text == "")
            records.append(rec)
        return Batch(x0=self.onehot(rows), grid=self.grids[rows.to(self.device)], captions=caps,
                     is_null=torch.tensor(null, device=self.device), rows=rows,
                     level_id=self.level_id[rows.to(self.device)].cpu(), aug=records if self.cfg.aug.record else None)
