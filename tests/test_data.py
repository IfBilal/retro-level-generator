"""Tile table, representation, sampler, augmentation and dataset tests (doc 11 §9: T-D*, T-S*, T-A*)."""
import json
from collections import Counter

import numpy as np
import pytest
import torch

from conftest import ROOT, needs_data
from text2mario.captions.grammar import caption_set
from text2mario.data import lexicon as lexicon_mod
from text2mario.data import tiles as T
from text2mario.data.augment import AugConfig, augment_caption, paraphrase_phrase, split_phrases
from text2mario.data.sampler import WindowSampler, mix
from text2mario.render.renderer import (grid_to_csv, grid_to_tensor, grid_to_vglc_text, parse_vglc, tensor_to_grid)


@needs_data
def test_td1_tiles_match_smb_json():
    smb = json.loads((ROOT / "data/source/smb.json").read_text())["tiles"]
    assert T.CHARS == tuple(sorted(smb))
    assert T.SOLID == {T.CHAR_TO_ID[c] for c, d in smb.items() if "solid" in d}
    assert T.PIPE == {T.CHAR_TO_ID[c] for c, d in smb.items() if "pipe" in d}


def test_td2_round_trip_and_distance_identity():
    rng = np.random.default_rng(0)
    a, b = rng.integers(0, 13, (2, 8, 16, 16)).astype(np.uint8)
    xa, xb = grid_to_tensor(a), grid_to_tensor(b)
    assert (tensor_to_grid(xa) == a).all()
    ham = (a != b).sum(axis=(1, 2))
    assert np.allclose(((xa - xb) ** 2).sum(dim=(1, 2, 3)).numpy(), 8 * ham)   # |x(g)-x(h)|^2 = 8 Hamming
    assert (xa.sum(1) == -11).all()                                              # one +1 and twelve -1 per cell


def test_text_exports_round_trip():
    g = np.random.default_rng(1).integers(0, 13, (16, 40)).astype(np.uint8)
    assert (parse_vglc(grid_to_vglc_text(g)) == g).all()
    assert grid_to_csv(g).count("\n") == 16


@needs_data
def test_td4_td5_dataset_orientation_and_split():
    from text2mario.data.dataset import MarioWindows
    from text2mario.data import splits
    d = MarioWindows("train")
    assert len(d) == splits.load()["window_counts"]["train"]
    assert sorted(set(d.level_id.tolist())) == splits.load()["splits"]["train"]
    with pytest.raises(PermissionError):   # "all" includes the locked test levels, so it is refused too
        MarioWindows("all")
    w0 = np.load(ROOT / "data/processed/scenes_u8.npy")[0]   # corpus window 0
    assert (w0[15] != 0).any() and (w0[0] == 0).all()        # floor at the bottom, sky on top


@needs_data
def test_td6_test_guard(tmp_path):
    from text2mario.data.dataset import MarioWindows
    with pytest.raises(PermissionError):
        MarioWindows("test")
    with pytest.raises(PermissionError):
        MarioWindows("test", allow_test=True, frozen_path=tmp_path / "FROZEN.json")   # allowed flag but no file
    (tmp_path / "FROZEN.json").write_text("{}")
    assert len(MarioWindows("test", allow_test=True, frozen_path=tmp_path / "FROZEN.json")) > 0
    with pytest.raises(PermissionError):
        MarioWindows(levels=[1, 3])   # level 1 is a test level


def test_ts1_sampler_pure_function():
    a, b = WindowSampler(5709), WindowSampler(5709)
    assert torch.equal(a.rows(123), b.rows(123))
    assert not torch.equal(a.rows(1), a.rows(2))


def test_ts2_every_row_once_or_twice_per_epoch():
    n, bs = 1000, 64
    s = WindowSampler(n, bs)
    steps = -(-n // bs)
    counts = Counter(r for t in range(steps) for r in s.rows(t).tolist())
    assert set(counts) == set(range(n)) and max(counts.values()) <= 2


def test_ts3_resume_equivalence():
    fresh = [WindowSampler(5709).rows(t) for t in range(1000, 1010)]
    s = WindowSampler(5709)
    s.check_resume(s.state_dict())
    assert all(torch.equal(s.rows(t), f) for t, f in zip(range(1000, 1010), fresh))
    with pytest.raises(RuntimeError):
        WindowSampler(5000).check_resume(s.state_dict())


def test_ts4_balanced_mode_raises_rare_share():
    n = 4000
    rare = np.zeros(n, bool)
    rare[:40] = True                        # 1 % rare windows
    w = np.where(rare, 1 / np.sqrt(0.01), 1 / np.sqrt(0.99))
    s = WindowSampler(n, 64, mode="balanced", weights=w)
    share = np.mean([rare[s.rows(t).numpy()].mean() for t in range(300)])
    assert share >= 2 * 0.01


def test_mix_is_stable():
    assert mix(1, 2) == mix(1, 2) and mix(1, 2) != mix(2, 1)
    assert mix(20261117, 0) == int.from_bytes(__import__("hashlib").blake2b(b"20261117:0", digest_size=8).digest(), "little") & (2 ** 63 - 1)


CAP = "full floor. two enemies. a few coins. one coin line. one pipe."
LEX = lexicon_mod.load()


def test_ta1_augmentation_is_pure():
    cfg = AugConfig()
    assert augment_caption(CAP, 7, 3, 5, cfg, LEX) == augment_caption(CAP, 7, 3, 5, cfg, LEX)
    outs = {augment_caption(CAP, 7, 3, i, cfg, LEX)[0] for i in range(20)}
    assert len(outs) > 1     # different samples get different augmentations


def test_ta2_never_empty_unless_null():
    cfg = AugConfig(null_p=0.0, phrase_dropout_p=0.9)   # brutal dropout
    for i in range(500):
        text, rec = augment_caption(CAP, 1, i, 0, cfg, LEX)
        assert text != "" and not rec["null"]


def test_ta3_null_rate():
    cfg = AugConfig()
    nulls = sum(augment_caption(CAP, 11, t, 0, cfg, LEX)[0] == "" for t in range(10000))
    assert abs(nulls / 10000 - 0.10) <= 0.01


def test_ta_dropout_and_paraphrase_statistics():
    cfg = AugConfig(null_p=0.0)
    recs = [augment_caption(CAP, 5, t, 0, cfg, LEX)[1] for t in range(10000)]
    assert abs(np.mean([r["paraphrased"] for r in recs]) - 0.30) <= 0.02
    assert abs(np.mean([r["dropped"] > 0 for r in recs]) - (1 - 0.9 ** 5)) <= 0.02   # 5 phrases


def test_ta5_paraphrase_uses_only_train_vocabulary():
    held = LEX.all_heldout_words()
    cfg = AugConfig(null_p=0.0, paraphrase_caption_p=1.0, paraphrase_phrase_p=1.0)
    for t in range(1500):
        text, _ = augment_caption(CAP + " floor with two gaps. one upside down pipe. several question blocks.", 3, t, 0, cfg, LEX)
        assert not (set(text.replace(".", " ").split()) & held), text


def test_ta6_format():
    import re
    pat = re.compile(r"^([a-z0-9 ?'\-]+\. )*[a-z0-9 ?'\-]+\.$")
    cfg = AugConfig(null_p=0.0)
    for t in range(500):
        text, _ = augment_caption(CAP, 2, t, 0, cfg, LEX)
        assert pat.match(text), text


def test_paraphrase_preserves_quantity_bucket_word_set():
    """A paraphrase of 'N <topic>' must still carry the number of that bucket (digit or word)."""
    rng = np.random.default_rng(0)
    cfg = AugConfig()
    for phrase in ["one pipe", "two enemies", "a few coins", "several platforms", "many question blocks"]:
        outs = {paraphrase_phrase(phrase, LEX, rng, cfg) for _ in range(200)}
        assert all(o.strip() for o in outs) and len(outs) > 3


@needs_data
def test_batch_shapes_and_null_flag():
    from text2mario.data.dataset import MarioWindows
    d = MarioWindows("train")
    s = WindowSampler(len(d))
    b = d.batch(s.rows(0), 0)
    assert b.x0.shape == (64, 13, 16, 16) and len(b.captions) == 64
    assert [c == "" for c in b.captions] == b.is_null.tolist()
    u = d.batch(s.rows(0), 0, uncond=True)
    assert all(c == "" for c in u.captions) and bool(u.is_null.all())
    assert (d.eval_rows(8).numel() > 0) and (d.col_offset[d.eval_rows(8)] % 8 == 0).all()
