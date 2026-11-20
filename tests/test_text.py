"""Text-conditioning tests (doc 15 §10, T1-T11)."""
import numpy as np
import pandas as pd
import pytest
import torch

from conftest import ROOT, needs_data
from text2mario.data.augment import AugConfig, augment_caption
from text2mario.data import lexicon

pytest.importorskip("transformers")
if not (ROOT / "models/minilm/model.safetensors").exists():
    pytest.skip("run scripts/fetch_models.py first", allow_module_level=True)

from text2mario.text.conditioner import PROBE, TextConditioner  # noqa: E402


@pytest.fixture(scope="module")
def cond():
    return TextConditioner()


@needs_data
def test_t1_token_lengths_fit(cond):
    caps = pd.read_parquet(ROOT / "data/processed/index.parquet").caption.tolist()
    lens = [len(x) for x in cond.tok(caps, truncation=False)["input_ids"]]
    assert max(lens) <= 48
    assert len(cond.tok([""])["input_ids"][0]) == 2


def test_overflow_raises_instead_of_truncating(cond):
    with pytest.raises(ValueError):
        cond(["one pipe. " * 40])


def test_t2_encoder_frozen(cond):
    assert not any(p.requires_grad for p in cond.encoder.parameters())
    cond.train()
    assert not cond.encoder.training                          # stays in eval mode even when the conditioner trains


def test_t5_shapes_and_mask(cond):
    ctx, mask = cond(["full floor. two pipes.", "", "a few enemies. one cannon. several coins."])
    assert ctx.shape[0] == 3 and ctx.shape[2] == 128 and mask.dtype == torch.bool and mask.shape == ctx.shape[:2]
    assert (mask.sum(1) >= 2).all()                            # [CLS] and [SEP] are always valid
    assert mask[1].sum() == 2                                  # the null caption is exactly [CLS][SEP]


def test_adapter_has_66048_parameters(cond):
    assert sum(p.numel() for p in cond.adapter.parameters()) == 66_048


def test_t4_minilm_unchanged_by_adapter_training(cond):
    before = cond.probe().clone()
    opt = torch.optim.AdamW(cond.trainable_parameters(), lr=1e-2)
    for _ in range(5):
        ctx, _ = cond(["full floor. two pipes."])
        opt.zero_grad()
        ctx.pow(2).mean().backward()
        opt.step()
    assert torch.equal(cond.probe(), before)


def test_t7_mask_correctness_end_to_end(cond):
    """Garbage in padded positions must not change the U-Net output (through conditioner + model)."""
    from text2mario.models.factory import build_unet
    m = build_unet({"base_channels": 16, "zero_init": False}).eval()
    ctx, mask = cond(["full floor. two pipes.", "one enemy."])
    ctx2 = ctx.clone()
    ctx2[~mask] = torch.randn_like(ctx2[~mask]) * 40
    x, t = torch.randn(2, 13, 16, 16), torch.tensor([100, 400])
    with torch.no_grad():
        assert (m(x, t, ctx, mask) - m(x, t, ctx2, mask)).abs().max() < 1e-5


def test_t9_cache_equivalence_and_byte_cap():
    c = TextConditioner(cache_max_mb=1)
    a = c.encode_frozen(["full floor. two pipes."])[0].clone()
    b = c.encode_frozen(["full floor. two pipes."])[0]
    assert torch.equal(a, b) and c.cache_stats["hits"] >= 1
    fresh = TextConditioner().encode_frozen(["full floor. two pipes."])[0]
    assert torch.nn.functional.cosine_similarity(a.float(), fresh.float(), dim=-1).min() > 0.999
    c._cache_cap = 20_000                                      # tiny cap: LRU must evict
    for i in range(30):
        c.encode_frozen([f"{i + 1} pipes. one enemy."])
    assert c._cache_bytes <= 20_000 + 2 * 384 * 4 * 20 and c.cache_stats["evictions"] > 0


def test_t11_probe_check(cond):
    good = cond.probe()
    cond.check_probe(good)
    bad = good.clone()
    bad[3] = torch.randn_like(bad[3])
    with pytest.raises(RuntimeError):
        cond.check_probe(bad)
    with pytest.raises(RuntimeError):
        cond.check_probe(good[:-1])


def test_checkpoint_state_never_contains_minilm_weights(cond):
    sd = cond.state_dict_for_ckpt()
    assert set(sd) == {"adapter", "encoder", "revision", "probe"} and len(sd["adapter"]) == 6


def test_t6_null_rate_matches_spec():
    lex = lexicon.load()
    n = sum(augment_caption("full floor. one pipe.", 3, s, 0, AugConfig(), lex)[0] == "" for s in range(10000))
    assert abs(n / 10000 - 0.10) <= 0.01


def test_t8_surface_form_of_stored_captions():
    import re
    pat = re.compile(r"^([a-z ]+\. )*[a-z ]+\.$")
    caps = pd.read_parquet(ROOT / "data/processed/index.parquet").caption.unique()
    assert all(pat.match(c) for c in caps)
