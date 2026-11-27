"""audit2 H5 text-path variants: the structured (count-aware) conditioner and the MiniLM-with-top-layers-trainable conditioner."""
import pytest
import torch

from text2mario.text.conditioner import build_conditioner
from text2mario.text.structured import StructuredConditioner


def test_structured_shapes_null_and_mask():
    c = StructuredConditioner(32, 16)
    ctx, mask = c(["full floor. two pipes. a few enemies.", "one coin.", ""])
    assert ctx.shape == (3, 3, 16) and mask.tolist() == [[True] * 3, [True, False, False], [True, False, False]]
    n_ctx, n_mask = c.null(4)
    assert n_ctx.shape == (4, 1, 16) and n_mask.all()


def test_structured_quantity_is_shared_across_concepts():
    """'one pipe' and 'one enemy' use the same quantity id, 'two pipes' a different one: the count is explicit, not learned from text."""
    c = StructuredConditioner(8, 8)
    (pc, pq), = c.ids("one pipe.")
    (ec, eq), = c.ids("one enemy.")
    (tc, tq), = c.ids("two pipes.")
    assert pq == eq and pq != tq and pc == tc and pc != ec


def test_structured_unknown_phrase_does_not_crash_and_probe_checks_the_grammar():
    c = StructuredConditioner(8, 8)
    assert c.ids("a purple dragon.")[0][1] == 0
    c.check_probe(c.probe())
    with pytest.raises(RuntimeError, match="grammar changed"):
        c.check_probe(c.probe() + 1)


def test_factory_default_is_the_frozen_minilm_path():
    cfg = {"adapter_hidden": 8, "encoder_dir": "models/minilm", "cache_max_mb": 4}
    from text2mario.text.conditioner import TextConditioner
    assert type(build_conditioner(cfg, 48, 8, "cpu")) is TextConditioner
    assert type(build_conditioner({**cfg, "kind": "structured"}, 48, 8, "cpu")) is StructuredConditioner


def test_tuned_minilm_trains_only_its_top_layers():
    cfg = {"adapter_hidden": 8, "encoder_dir": "models/minilm", "cache_max_mb": 4, "kind": "minilm_tuned", "unfreeze_layers": 1}
    c = build_conditioner(cfg, 48, 8, "cpu")
    probe0 = c.probe()
    ctx, mask = c(["full floor. two pipes.", ""])
    ctx.sum().backward()
    layers = c.encoder.encoder.layer
    assert layers[-1].attention.self.query.weight.grad is not None
    assert layers[-2].attention.self.query.weight.grad is None            # lower layers stay frozen
    assert mask.shape == ctx.shape[:2] and torch.equal(c.probe(), probe0)


def test_both_concatenates_the_two_token_sets():
    cfg = {"adapter_hidden": 8, "encoder_dir": "models/minilm", "cache_max_mb": 4, "kind": "both", "unfreeze_layers": 1}
    c = build_conditioner(cfg, 48, 8, "cpu")
    ctx, mask = c(["full floor. two pipes."])
    n_text = c.t(["full floor. two pipes."])[0].shape[1]
    assert ctx.shape[1] == n_text + 2 and mask.sum() == n_text + 2          # MiniLM tokens + the 2 structured phrase tokens
    ctx.sum().backward()
    assert c.s.adapter.code.weight.grad is not None
