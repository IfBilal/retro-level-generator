"""Canonical caption grammar (doc 12 §2): quantity words, plurals and phrase helpers.

Explain-me:
  What: the closed language of captions, e.g. "full floor. two pipes. a few enemies." Gives helpers to build
        a counted phrase ("q(5) + enemy" -> "several enemies") and to split a caption into its phrase set.
  In/Out: pure functions on ints / strings.  Used by the detector (grid -> caption), the augmenter and normalizer.
  Try: change a bucket boundary in `quantity_word` and re-run tests/test_detector.py (nothing else changes).
"""
from __future__ import annotations

import re

BUCKETS = ("one", "two", "a few", "several", "many")  # ledger F15

# Singular nouns of counted topics (doc 12 §2.1 NOUN(one)); plural = see `plural`.
COUNTED_NOUNS = ("enemy", "question block", "cannon", "coin", "coin line", "platform", "ascending staircase",
                 "descending staircase", "pipe", "upside down pipe", "tower", "rectangular block cluster",
                 "irregular block cluster", "loose block")
# The 16 concepts of the grammar in generator order (doc 12 §2.4) plus the two detector-only topics (F23).
CONCEPTS = ("floor", "ceiling", "enemy", "question block", "cannon", "coin", "coin line", "platform",
            "ascending staircase", "descending staircase", "pipe", "upside down pipe", "tower",
            "rectangular block cluster", "irregular block cluster", "loose block")
TOPICS_18 = CONCEPTS + ("broken pipe", "broken cannon")


def quantity_word(n: int) -> str:
    """Bucket word for a count: 1 one, 2 two, 3-4 a few, 5-9 several, >=10 many (F15)."""
    if n < 1:
        raise ValueError("quantity needs n >= 1")
    if n == 1:
        return "one"
    if n == 2:
        return "two"
    if n <= 4:
        return "a few"
    if n <= 9:
        return "several"
    return "many"


def plural(noun: str) -> str:
    """'enemy' -> 'enemies'; everything else + 's' ('coin line' -> 'coin lines')."""
    return noun[:-1] + "ies" if noun.endswith("y") else noun + "s"


def counted_phrase(noun: str, n: int) -> str:
    """'two pipes', 'one enemy', 'several loose blocks' ..."""
    return f"{quantity_word(n)} {noun if n == 1 else plural(noun)}"


def gap_word(n: int) -> str:
    return "gap" if n == 1 else "gaps"


def caption_set(text: str) -> frozenset[str]:
    """Phrase set of any caption string (order-free comparison used by the reproduction check)."""
    return frozenset(p.strip().lower() for p in re.split(r"\.", text) if p.strip())


def join_phrases(phrases: list[str]) -> str:
    """'p1. p2. p3.' (the exact format of stored captions); '' for no phrases."""
    return "".join(f"{p}. " for p in phrases).strip()


# ---------------------------------------------------------------- phrase parsing (doc 12 §2.3 codes)
_Q = "|".join(BUCKETS)
_NOUN_FORMS = {}
for _n in COUNTED_NOUNS + ("broken pipe", "broken cannon"):
    _NOUN_FORMS[_n] = _n
    _NOUN_FORMS[plural(_n)] = _n
_COUNTED_RE = re.compile(rf"^({_Q}) ({'|'.join(sorted(_NOUN_FORMS, key=len, reverse=True))})$")
_FLOOR_GAP_RE = re.compile(rf"^floor with ({_Q}) gaps?$")
_FLOOR_CHUNK_RE = re.compile(rf"^giant gap with ({_Q}) chunks? of floor$")
_CEIL_GAP_RE = re.compile(rf"^ceiling with ({_Q}) gaps?$")


def parse_phrase(phrase: str) -> tuple[str, int]:
    """Phrase -> (concept, code).  Counted topics: bucket 1..5.  Floor: 1 full, 2-6 gaps, 7-11 chunks.
    Ceiling: 1 full, 2-6 gaps (doc 12 §2.3).  Raises ValueError for text outside the grammar."""
    p = phrase.strip().lower()
    if p == "full floor":
        return "floor", 1
    if p == "full ceiling":
        return "ceiling", 1
    if m := _FLOOR_GAP_RE.match(p):
        return "floor", 1 + BUCKETS.index(m.group(1)) + 1
    if m := _FLOOR_CHUNK_RE.match(p):
        return "floor", 6 + BUCKETS.index(m.group(1)) + 1
    if m := _CEIL_GAP_RE.match(p):
        return "ceiling", 1 + BUCKETS.index(m.group(1)) + 1
    if m := _COUNTED_RE.match(p):
        return _NOUN_FORMS[m.group(2)], BUCKETS.index(m.group(1)) + 1
    raise ValueError(f"not a canonical phrase: {phrase!r}")
