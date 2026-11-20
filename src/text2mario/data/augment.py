"""Caption augmentation (ledger D8, doc 11 §5.2): phrase dropout, casual paraphrase, shuffle, null dropout.

Explain-me:
  What: turns a stored caption into the text the model sees this step.  Order: drop phrases (p=0.10 each, keep >= 1)
        -> maybe paraphrase into casual wording (train-half lexicon only) -> shuffle phrases -> join -> with p=0.10
        replace everything by "" (the null caption used for classifier-free guidance).
  In/Out: (caption, seed, step, i) -> (augmented caption, record).  A pure function: same inputs -> same output,
          on any machine (RNG = numpy default_rng seeded with blake2b mix of seed/step/sample).
  Try: set null_p to 0.3 in AugConfig and run tests/test_augment.py::test_null_rate (drill 12).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from text2mario.captions.grammar import BUCKETS, join_phrases, parse_phrase
from text2mario.data.lexicon import Lexicon
from text2mario.data.sampler import mix


@dataclass(frozen=True)
class AugConfig:
    phrase_dropout_p: float = 0.10       # D8b
    paraphrase_caption_p: float = 0.30   # D8c, per caption
    paraphrase_phrase_p: float = 0.50    # D8c, per surviving phrase inside a paraphrased caption
    keep_canonical_quantity_p: float = 0.25
    shuffle: bool = True                 # D8a
    null_p: float = 0.10                 # D8d, applied last
    record: bool = False


def split_phrases(caption: str) -> list[str]:
    """'a. b. c.' -> ['a','b','c'] (lowercased, stripped, empty parts dropped)."""
    return [p.strip().lower() for p in caption.split(".") if p.strip()]


def _qty(bucket: str, lex: Lexicon, rng, cfg: AugConfig) -> str:
    """Canonical bucket word with prob keep_canonical_quantity_p, else a casual equivalent of the SAME bucket."""
    if rng.random() < cfg.keep_canonical_quantity_p:
        return bucket
    opts = lex.casual_quantities(bucket)
    return opts[int(rng.integers(len(opts)))]


def _pick(pairs: list[tuple[str, str]], singular: bool, rng) -> str:
    sg, pl = pairs[int(rng.integers(len(pairs)))]
    return sg if singular else pl


def paraphrase_phrase(phrase: str, lex: Lexicon, rng, cfg: AugConfig) -> str:
    """Rewrite one canonical phrase in casual wording using ONLY train-half lexicon entries (doc 11 §5.2 step 2)."""
    concept, code = parse_phrase(phrase)
    if concept == "floor":
        if code == 1:
            return lex.floor("full")[int(rng.integers(len(lex.floor("full"))))][0]
        kind, b = ("gaps", code - 2) if code <= 6 else ("chunks", code - 7)
        bucket, one = BUCKETS[b], b == 0
        q = _qty(bucket, lex, rng, cfg)
        if kind == "gaps":
            return f"{_pick(lex.floor('floor_nouns'), True, rng)} with {q} {_pick(lex.floor('gap_nouns'), one, rng)}"
        return f"{_pick(lex.floor('giant'), True, rng)} with {q} {_pick(lex.floor('chunk_nouns'), one, rng)}"
    if concept == "ceiling":
        if code == 1:
            return lex.ceiling("full")[int(rng.integers(len(lex.ceiling("full"))))][0]
        b = code - 2
        q = _qty(BUCKETS[b], lex, rng, cfg)
        return f"{_pick(lex.ceiling('ceiling_nouns'), True, rng)} with {q} {_pick(lex.floor('gap_nouns'), b == 0, rng)}"
    bucket = BUCKETS[code - 1]
    nouns = lex.train_nouns(concept)
    if not nouns:  # topic without train synonyms: keep the canonical noun
        return phrase
    return f"{_qty(bucket, lex, rng, cfg)} {_pick(nouns, code == 1, rng)}"


def augment_caption(caption: str, seed: int, step: int, i: int, cfg: AugConfig, lex: Lexicon) -> tuple[str, dict]:
    rng = np.random.default_rng(mix(seed, mix(step, i)))
    phrases = split_phrases(caption)
    # 1. phrase dropout (always keep at least one)
    keep = [p for p in phrases if rng.random() >= cfg.phrase_dropout_p]
    if not keep:
        keep = [phrases[int(rng.integers(len(phrases)))]]
    dropped = len(phrases) - len(keep)
    # 2. casual paraphrase
    paraphrased = rng.random() < cfg.paraphrase_caption_p
    if paraphrased:
        flags = [rng.random() < cfg.paraphrase_phrase_p for _ in keep]
        if not any(flags):
            flags[int(rng.integers(len(keep)))] = True
        keep = [paraphrase_phrase(p, lex, rng, cfg) if f else p for p, f in zip(keep, flags)]
    # 3. shuffle, 4. join
    if cfg.shuffle:
        keep = [keep[j] for j in rng.permutation(len(keep))]
    text = join_phrases(keep)
    # 5. CFG null dropout (applied last, overrides everything)
    is_null = rng.random() < cfg.null_p
    if is_null:
        text = ""
    return text, {"dropped": dropped, "paraphrased": bool(paraphrased), "null": bool(is_null)}
