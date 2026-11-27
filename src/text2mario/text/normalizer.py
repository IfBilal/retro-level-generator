"""Rule-based prompt normalizer: free text -> canonical captions, one per scene (doc 12 §3-§7).

Explain-me:
  What: turns "a flat level with 3 goombas and 2 pipes, then lots of coins" into the grammar the model was trained on
        ("full floor. a few enemies. two pipes." / "many coins."), with warnings for everything it clamped, ignored or
        could not understand.  No model involved: lexicon + regular expressions, so every decision can be explained.
  In/Out: normalize(text) -> Normalized(scenes=[Scene(caption, topics, forbid, warnings)], warnings, ignored_tokens, style).
  Previous: the user's sentence (app) or the P-casual prompts.  Next: generate/full_level.py (scenes -> windows).
  Try: add a synonym to data/prompts/lexicon.yaml ("pipe: train: [tube, ...]") and re-run scripts/eval_normalizer.py.
Pipeline per scene: 1 split scenes  2 find nouns (longest match)  3 attach quantities  4 bare-noun defaults  5 floor/ceiling
rules  6 negation -> forbid  7 duplicates  8 implied rules  9 clamp to the train-supported buckets  10 write the caption.
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from text2mario.captions.grammar import (
    BUCKETS,
    CONCEPTS,
    COUNTED_NOUNS,
    counted_phrase,
    join_phrases,
    parse_phrase,
    plural,
)
from text2mario.data import lexicon as lex

ROOT = Path(__file__).resolve().parents[3]
BUCKET_REP = {"one": 1, "two": 2, "a few": 3, "several": 5, "many": 10}
NEUTRAL_CAPTION = "full floor."          # carries a bare prohibition ("no pipes"); also the caption of a window with nothing to describe
NUMBERS = {w: i for i, w in enumerate(["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty"])}
NUMBERS.update({"thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100})
# multi-word and vague quantity expressions -> (value); matched longest first on the token string
VAGUE = {"a couple of": 2, "couple of": 2, "a pair of": 2, "pair of": 2, "a single": 1, "a lone": 1, "single": 1, "lone": 1, "solo": 1,
         "a couple": 2, "double": 2, "twin": 2, "a handful of": 3, "handful of": 3, "a few": 3, "few": 3, "some": 3,
         "a bunch of": 5, "bunch of": 5, "half a dozen": 6, "a number of": 5, "several": 5, "a dozen": 12, "dozens of": 12,
         "many": 12, "lots of": 12, "a lot of": 12, "loads of": 12, "tons of": 12, "a ton of": 12, "plenty of": 12,
         "heaps of": 12, "numerous": 12, "countless": 12, "a swarm of": 12, "an army of": 12, "a crowd of": 12, "a horde of": 12}
NEG = ("no", "without", "zero", "none", "never", "not any", "don't add any", "dont add any", "don't add", "dont add", "no more")
ADJ = {"green", "red", "golden", "gold", "shiny", "big", "small", "tall", "short", "long", "tiny", "huge", "floating", "fun", "cool",
       "awesome", "scary", "classic", "nice", "cute", "epic", "of", "the", "a", "an", "more", "another", "extra", "additional", "little",
       "few", "some", "bad", "nasty", "evil", "angry", "walking", "just", "plain", "simple",
       "separate", "individual", "distinct", "different", "isolated", "standalone", "all", "every", "any"}
POSITIONAL = {"left", "right", "middle", "center", "centre", "top", "bottom", "above", "below", "under", "near", "next", "far", "corner",
              "high", "low", "over", "beginning", "ends"}
# Position words.  The model is conditioned on what a 16-column window holds, not on where in the level it sits, so "before/after the Nth X" and
# "at the start/end" cannot be steered.  They are parsed so they are NOT mistaken for counts (the 4th pipe is a reference, not a second pipe),
# an impossible reference ("the 4th pipe" of 3) is refused, and the rest is reported to the user as not controlled (POSITION_NOT_PLANNED).
ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10}
REF_PREPS = {"before", "after", "between", "past", "beyond", "until", "since"}
# bounds, multipliers, negated quantities and "everything but": the parser only knows exact amounts and plain prohibitions, so these are refused
UNSUPPORTED_RE = re.compile(r"\b(?:at most|at least|no more than|no fewer than|up to|not (?:too )?(?:many|much|few)|(?:\d+|" + "|".join(NUMBERS)
                            + r") or (?:\d+|" + "|".join(NUMBERS) + r")\b|each|per|sets? of|groups? of|everything but|anything but|all but|only (?:pipes|enemies|coins))\b")
POSITION_RE = re.compile(r"\b(?:before|after|between|beyond|past|at the (?:start|end|beginning)|near the (?:start|end))\b")
# a request for an empty / flat level: a plain floor with everything else ruled out (checked on the raw level like any "no X")
FLAT_WORDS = {"flat", "flatland", "empty", "blank", "bare", "barren", "featureless"}
THEMES = {"castle", "lava", "water", "underwater", "snow", "snowy", "ice", "desert", "night", "forest", "space", "ghost"}
STYLE = {"easy": "easy", "chill": "easy", "relaxed": "easy", "beginner": "easy", "medium": "medium", "normal": "medium",
         "hard": "hard", "difficult": "hard", "challenging": "hard", "brutal": "hard", "insane": "hard"}
STOP = {"with", "and", "or", "a", "an", "the", "of", "in", "on", "to", "level", "stage", "section", "part", "make", "me", "please",
        "want", "i", "it", "that", "has", "have", "there", "is", "are", "for", "some", "stretch", "area", "just", "plus", "also", "then"}
SPLIT_RE = re.compile(r"\s*(?:;|->|\||\n|,?\s+and then\s+|,?\s+then\s+|,?\s+followed by\s+|,?\s+after that\s+|,?\s+afterwards\s+|,?\s+later\s+|,\s*next\s+)\s*")
ORD_RE = re.compile(r"^(?:first|second|third|fourth|fifth|last|finally|lastly)[,:]?\s+")
START_RE = re.compile(r"^(?:start with|starting with|begin with|beginning with|open with)\s+")
_AT = r"(?: of the level)?"
AT_START_END = re.compile(r"^(.+?)" + r"\s+at the (?:start|beginning)" + _AT + r"\s*(?:,|and)\s*(.+?)\s+at the end" + _AT + r"$")
AT_END_START = re.compile(r"^(.+?)\s+at the end" + _AT + r"\s*(?:,|and)\s*(.+?)\s+at the (?:start|beginning)" + _AT + r"$")
END_RE = re.compile(r"^(?:end with|ending with|finish with|finishing with|at the end(?: of the level)?,?(?: with)?)\s+")


@dataclass
class Scene:
    caption: str
    topics: dict = field(default_factory=dict)       # raw (pre-clamp): topic -> bucket word / template
    forbid: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    # LEVEL-level quantity the user asked for, topic -> [lo, hi] (numerals are exact, vague words keep their bucket range).
    # The caption above can only say a bucket ("a few" = 3-4) and describes ONE 16-column window, so counts live here (audit P2).
    want: dict = field(default_factory=dict)
    ignored: list = field(default_factory=list)     # words nothing understood (shown to the user, never silently dropped)


# Bucket word -> the counts it stands for (ledger F15 / caption grammar quantity_word)
BUCKET_RANGE = {"one": (1, 1), "two": (2, 2), "a few": (3, 4), "several": (5, 9), "many": (10, 99)}


@dataclass
class Normalized:
    text: str
    scenes: list
    warnings: list = field(default_factory=list)
    ignored_tokens: list = field(default_factory=list)
    style: dict = field(default_factory=dict)


def bucket(n: int) -> str:
    return "one" if n <= 1 else "two" if n == 2 else "a few" if n <= 4 else "several" if n <= 9 else "many"


_CLAMP = json.loads((ROOT / "data/prompts/clamp_table.json").read_text()) if (ROOT / "data/prompts/clamp_table.json").exists() else None


def _build_nouns() -> list[tuple[str, str, bool, str]]:
    """[(surface form, topic, is_plural, kind)] longest first.  kind in {'noun','gap','chunk','ceil'}."""
    L = lex.load()
    out = []
    for t in COUNTED_NOUNS:
        forms = [(t, plural(t))] + (L.train_nouns(t) + L.heldout_nouns(t) if t in L.raw["topics"] else [])
        if t == "rectangular block cluster":
            forms += [("rectangle", "rectangles"), ("rectangular cluster", "rectangular clusters")]
        if t == "irregular block cluster":
            forms += [("irregular cluster", "irregular clusters")]
        if t == "loose block":
            forms += [("block", "blocks")]
        for sg, pl in forms:
            out.append((sg, t, False, "noun"))
            if pl != sg:
                out.append((pl, t, True, "noun"))
            else:
                out.append((sg, t, True, "noun"))
    for key, kind in (("gap_nouns", "gap"), ("chunk_nouns", "chunk")):
        for half in ("train", "heldout"):
            for sg, pl in L.floor(key, half):
                out.append((sg, kind, False, kind))
                out.append((pl, kind, True, kind))
    for sg, pl in L.ceiling("ceiling_nouns", "train"):
        out.append((sg, "ceil", False, "ceil"))
        out.append((pl, "ceil", True, "ceil"))
    for half in ("train", "heldout"):
        for sg, pl in L.ceiling("full", half):
            out.append((sg, "ceil", False, "ceilfull"))
    for half in ("train", "heldout"):
        for sg, pl in L.floor("full", half):
            out.append((sg, "floor", False, "floorfull"))
        for sg, pl in L.floor("giant", half):
            out.append((sg, "floor", False, "giant"))
    for sg, pl in L.floor("floor_nouns", "train") + L.floor("floor_nouns", "heldout"):
        out.append((sg, "floor", False, "floornoun"))
    # extras not in the lexicon
    for w in ("stairs up", "steps up", "stairs"):
        out.append((w, "ascending staircase", True, "noun"))
    out.append(("staircase up", "ascending staircase", False, "noun"))
    for w in ("descending staircase", "descending stairs", "descending steps"):
        out.append((w, "descending staircase", False, "noun"))
    for w in ("descending staircases",):
        out.append((w, "descending staircase", True, "noun"))
    for w in ("flat", "flat level", "flat ground", "level ground", "full floor", "full ground", "complete floor", "whole floor"):
        out.append((w, "floor", False, "floorfull"))
    out.append(("pyramid", "pyramid", False, "pyramid"))
    out.append(("pyramids", "pyramid", True, "pyramid"))
    out.append(("in a row", "coin line", False, "rowmarker"))
    out.sort(key=lambda x: -len(x[0]))
    seen, uniq = set(), []
    for x in out:
        if x[0] not in seen:
            seen.add(x[0]); uniq.append(x)
    return uniq


_NOUNS = None


def _nouns():
    global _NOUNS
    if _NOUNS is None:
        _NOUNS = _build_nouns()
    return _NOUNS


def _clamp(topic: str, b: str, warns: list) -> str:
    if _CLAMP is None:
        return b
    sup = _CLAMP["supported"].get(topic)
    if sup is None or b in sup:
        return b
    nb = _CLAMP["nearest"][topic][b]
    warns.append(f"CLAMPED:{topic}:{b}->{nb}")
    return nb


_DOWN = re.compile(r"\b(staircases?|stairs|steps|stairways?)\s+(?:going|heading|leading|running)?\s*(?:down|downward|downwards)\b")
_ROW = re.compile(r"\b(\d+|" + "|".join(NUMBERS) + r")\s+coins?\s+(?:in a row|in a line|lined up)\b")


_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
_COMPOUND = re.compile(r"\b(" + "|".join(_TENS) + r")[- ](one|two|three|four|five|six|seven|eight|nine)\b")
# many ways to say "no": all become the one word the parser already understands ("avoid pipes" -> "no pipes")
_NEG_PHRASES = re.compile(r"\b(?:i\s+)?(?:do\s*n[o']?t|don't|dont|do not)\s+(?:want|need|add|include|put|have|like)(?:\s+(?:any|a|an))?\b"
                          r"|\bnot\s+(?:a\s+single|even\s+one|one\s+single|any)\b"
                          r"|\b(?:free of|devoid of|lacking|avoid(?:ing)?|skip(?:ping)?|remov(?:e|ing)|exclud(?:e|ing)|delet(?:e|ing)|minus|get rid of|"
                          r"getting rid of|leave out|leaving out|leave off|w/o|wo|sans|besides|other than|apart from|except(?: for)?|"
                          r"stay(?:ing)? away from|keep(?:ing)? (?:the )?(?:\w+ )?out|ban|banning|don't use|dont use|do not use|"
                          r"never use|no use of|rather not have|rather have no)\b"
                          r"|\bwithout\s+(?:a\s+single|even\s+one|one)\b|\bnot\s+one\b|\b(?:neither|nor)\b|\bnot\b(?=\s+(?:a\s+|an\s+|the\s+)?[a-z])")
_HYPHEN_FREE = re.compile(r"\b([a-z]+)-free\b")
_NOUN_COLON_NUM = re.compile(r"\b([a-z]+)\s*[:=]\s*(\d+|none|zero|no|" + "|".join(NUMBERS) + r")\b")
_NOUN_PAREN_NUM = re.compile(r"\b([a-z]+)\s*\((\d+)\)")


def _colon_amount(m) -> str:
    """'pipes: 2' -> '2 pipes', 'enemies: none' -> 'no enemies'; but 'no pipes: 2 coins' keeps its prohibition ('no pipes, 2 coins')."""
    if not _match_nouns([m.group(1)])[0]:                               # "hard: 3 goombas": the word before the colon is not an object
        return f"{m.group(1)}, {m.group(2)}"
    before = m.string[:m.start()].rstrip().split(" ")[-1:] if m.string[:m.start()].strip() else []
    if before and before[0] in ("no", "without", "zero"):
        return f"{m.group(1)}, {m.group(2)}"
    return ("no " if m.group(2) in ("none", "zero", "no") else m.group(2) + " ") + m.group(1)


def _pre(seg: str) -> str:
    """Idioms that need two-word context: '? block', 'N coins in a row', 'stairs going down'; also ways of saying no and of writing numbers."""
    seg = unicodedata.normalize("NFKC", seg).lower()                    # full-width digits, ligatures, odd spaces
    seg = re.sub(r"(?<=\d),(?=\d{3}\b)", "", seg)                   # 2,000 -> 2000
    seg = re.sub(r"\d{6,}", "999999", seg)                            # absurd digit strings never reach int() limits
    seg = _COMPOUND.sub(lambda m: str(_TENS[m.group(1)] + NUMBERS[m.group(2)]), seg)
    seg = re.sub(r"\bwithout any\b", "without", seg)
    seg = _NEG_PHRASES.sub("no", seg)
    seg = _HYPHEN_FREE.sub(r"no \1", seg)
    seg = _NOUN_COLON_NUM.sub(_colon_amount, seg)
    seg = _NOUN_PAREN_NUM.sub(r"\2 \1", seg)
    seg = re.sub(r"(\d)(?!(?:st|nd|rd|th)\b)([a-z])", r"\1 \2", seg)        # "3pipes" -> "3 pipes" (but keep 4th)
    seg = re.sub(r"\?\s*-?\s*blocks?", "question block", seg)
    seg = _DOWN.sub(lambda m: "descending " + m.group(1), seg)
    seg = re.sub(r"\b(?:descending|downward|falling)\s+(staircases?|stairs|steps)\b", r"descending \1", seg)
    seg = _ROW.sub(lambda m: f"{m.group(1)} coins and one coin line", seg)
    return seg


def _tokens(seg: str) -> list[str]:
    seg = _pre(seg).replace("?", " ? ")
    seg = re.sub(r"(\d+)\s*x\b", r"\1 ", seg)
    seg = re.sub(r"[^a-z0-9?'+\- ]", " ", seg)
    return seg.split()


def _match_nouns(toks: list[str]):
    """Greedy longest-match: -> list of (start, end, topic, plural, kind) over token indices."""
    spans, taken = [], [False] * len(toks)
    starts = [0]
    for t in toks[:-1]:
        starts.append(starts[-1] + len(t) + 1)
    for surf, topic, pl, kind in _nouns():
        words = surf.split()
        n = len(words)
        for i in range(len(toks) - n + 1):
            if toks[i:i + n] == words and not any(taken[i:i + n]):
                spans.append((i, i + n, topic, pl, kind))
                for k in range(i, i + n):
                    taken[k] = True
    if "?" in toks:   # a bare '?' next to 'block' is a question block
        pass
    spans.sort()
    return spans, taken


def _qty_before(toks, i, taken, limit=4):
    """Look left of token i (skipping adjectives/stopwords) for a quantity: -> (value|None, neg, first_index_consumed)."""
    j = i - 1
    steps = 0
    while j >= 0 and steps < limit:
        if taken[j]:
            return None, False, i
        # try longest vague phrase ending at j
        for n in (4, 3, 2, 1):
            if j - n + 1 >= 0:
                ph = " ".join(toks[j - n + 1:j + 1])
                if ph in VAGUE:
                    return VAGUE[ph], False, j - n + 1
                if n == 1 and ph in NEG:
                    return 0, True, j
        w = toks[j]
        if re.fullmatch(r"\d+\+?", w):
            return int(w.rstrip("+")), False, j
        if w in NUMBERS:
            return NUMBERS[w], False, j
        if w in ADJ or w in POSITIONAL or w in STOP:
            j -= 1
            steps += 1
            continue
        return None, False, i
    return None, False, i


def _scene(seg: str, g_prefix: dict | None = None) -> Scene | None:
    toks = _tokens(seg)
    spans, taken = _match_nouns(toks)
    sc = Scene("")
    sc.topics = dict(g_prefix or {})
    consumed = [False] * len(toks)
    for a, b, *_ in spans:
        for k in range(a, b):
            consumed[k] = True
    neg_scope = False
    last_end = -1
    counts: dict[str, int] = {}
    surfaces: dict[str, str] = {}               # topic -> the word the first mention used
    ordinals: dict[str, int] = {}               # topic -> highest "Nth <topic>" the sentence refers to
    vague: dict[str, bool] = {}                 # topic -> was the quantity a vague word ("several") rather than a numeral?
    floor_gap_q = floor_chunk_q = ceil_gap_q = None
    for idx, (a, b, topic, pl, kind) in enumerate(spans):
        # negation scope: a negation token directly before (possibly through a coordinated list)
        q, neg, first = _qty_before(toks, a, taken)
        if neg:
            neg_scope = True
        elif q is not None:
            neg_scope = False
        elif neg_scope and all(t in {"or", "and", ",", "any", "nor"} for t in toks[last_end:a]):
            neg = True
        else:
            neg_scope = False
        if q is not None:
            for k in range(first, a):
                consumed[k] = True
        last_prev_end, last_end = max(last_end, 0), b
        if kind in ("floorfull", "giant", "floornoun", "ceilfull", "rowmarker"):
            if kind == "floorfull":
                sc.topics["floor"] = "full"
            elif kind == "giant":
                sc.topics["floor"] = "chunks:one" if "floor" not in sc.topics or str(sc.topics["floor"]).startswith("full") else sc.topics["floor"]
            elif kind == "ceilfull":
                if neg:
                    sc.forbid.append("ceiling")
                else:
                    sc.topics["ceiling"] = "full"
            elif kind == "floornoun" and neg:
                sc.forbid.append("floor")
                sc.warnings.append("NEGATION_FORBID")
            continue
        if kind == "gap":
            # "no gaps / no pits" means a full floor; otherwise a count of floor gaps (or ceiling gaps near a ceiling noun)
            near_ceiling = any(t[2] == "ceil" and abs(t[0] - a) <= 3 for t in spans)
            if neg or q == 0:
                if not near_ceiling:
                    sc.topics["floor"] = "full"
                continue
            val = q if q is not None else (None if pl else 1)
            default = _CLAMP["bare_plural_default"].get("ceiling_gaps" if near_ceiling else "floor_gaps", "two") if _CLAMP else "two"
            bk = bucket(val) if val is not None else default
            if val is None:
                sc.warnings.append("BARE_PLURAL_DEFAULT")
            if near_ceiling:
                ceil_gap_q = bk
            else:
                floor_gap_q = bk
            continue
        if kind == "chunk":
            floor_chunk_q = bucket(q) if q is not None else "one"
            continue
        if kind == "ceil":
            if neg:
                sc.forbid.append("ceiling")
                sc.warnings.append("NEGATION_FORBID")
            elif "ceiling" not in sc.topics:
                sc.topics["ceiling"] = "full"
            continue
        if kind == "pyramid":
            n = q if q is not None else 1
            for t2 in ("ascending staircase", "descending staircase"):
                counts[t2] = counts.get(t2, 0) + n
            continue
        # counted topic.  A REFERENCE ("before the first pipe", "after the 4th pipe") is not a request for more of it.
        near = toks[max(last_prev_end, a - 3):a]                           # only tokens after the previous noun phrase
        ord_n = next((ORDINALS[t] if t in ORDINALS else int(re.sub(r"\D", "", t)) for t in near
                      if t in ORDINALS or re.fullmatch(r"\d+(?:st|nd|rd|th)", t)), None)
        # "the first pipe" / "before the pipe" refer to a pipe; "the first 3 pipes", "2 enemies" and "before a pipe" ask for objects
        is_ref = q is None and (ord_n is not None or ("the" in near and any(t in REF_PREPS for t in near)))
        if not neg and is_ref:
            if ord_n is not None:
                ordinals[topic] = max(ordinals.get(topic, 0), ord_n)
            sc.warnings.append("POSITION_NOT_PLANNED")
            for k in range(max(0, a - 3), b):
                consumed[k] = True
            continue
        after = toks[b:b + 4]                                              # "2 enemies before the first pipe" after "5 enemies": a part of the total
        if q not in (None, 0) and topic in counts and any(t in REF_PREPS for t in after) and not neg:
            sc.warnings.append("POSITION_NOT_PLANNED")
            continue
        if neg or q == 0:
            if topic not in sc.forbid:
                sc.forbid.append(topic)
            sc.warnings.append("NEGATION_FORBID")
            continue
        if q is None:
            if pl:
                bkt = _CLAMP["bare_plural_default"].get(topic, "two") if _CLAMP else "two"
                sc.warnings.append("BARE_PLURAL_DEFAULT")
                val = BUCKET_REP[bkt]
                is_vague = True
            else:
                val = 1
                is_vague = False
        else:
            val = q
            # vague words ("several", "a few", "some", "lots of") keep their bucket range; numerals and "single/couple/pair" are exact
            is_vague = any(" ".join(toks[first:first + n]) in VAGUE for n in (4, 3, 2, 1)) and val > 2
        more = any(t in ("more", "another", "extra", "additional") for t in toks[max(0, a - 3):a])
        surface = " ".join(toks[a:b])
        if topic in counts:
            sc.warnings.append("DUPLICATE_TOPIC")
            adds = more or surfaces.get(topic) not in (None, surface)         # another word for the same thing is another one of them
            counts[topic] = counts[topic] + val if adds else val
            more = adds
            vague[topic] = vague.get(topic, False) and is_vague if more else is_vague
        else:
            counts[topic] = val
            vague[topic] = is_vague
        surfaces.setdefault(topic, surface)
    # '? block' / 'coins in a row' idioms
    text = " ".join(toks)
    if re.search(r"(?:^|\s)\? block", text) and "question block" not in counts and "question block" not in sc.forbid:
        counts["question block"] = 1
    if re.search(r"coins? in a row", text) and "coin" not in counts:
        counts["coin"] = max(counts.get("coin", 0), 5)
    # floor evidence (doc 12 §5.5)
    if floor_chunk_q is not None:
        sc.topics["floor"] = f"chunks:{floor_chunk_q}"
    elif sc.topics.get("floor", "").startswith("chunks") if isinstance(sc.topics.get("floor"), str) else False:
        pass
    elif floor_gap_q is not None:
        sc.topics["floor"] = f"gaps:{floor_gap_q}"
    if ceil_gap_q is not None:
        sc.topics["ceiling"] = f"gaps:{ceil_gap_q}"
    # implied: coin line => coins (A10)
    if "coin line" in counts and "coin" not in counts and "coin" not in sc.forbid:
        counts["coin"] = {1: 3, 2: 5}.get(counts["coin line"], 5)
        vague["coin"] = True
        sc.warnings.append("IMPLIED_COINS")
    elif "coin line" in counts and counts.get("coin", 0) == 1:
        counts["coin"] = 2
        vague["coin"] = False
    for t, v in counts.items():
        sc.topics[t] = bucket(v)
        sc.want[t] = list(BUCKET_RANGE[bucket(v)]) if vague.get(t) else [v, v]
        if t in sc.forbid:                                  # "one pipe and no pipes": the prompt contradicts itself
            sc.warnings.append(f"CONTRADICTION:{t}")
    for t, n_ref in ordinals.items():                       # "the 4th pipe" of 3 pipes can never exist
        if t in sc.want and sc.want[t][1] < n_ref:
            sc.warnings.append(f"CONTRADICTION_ORDINAL:{t}:{n_ref}:{sc.want[t][1]}")
    if any(str(v).startswith(("gaps", "chunks")) for v in sc.topics.values()):
        sc.warnings.append("GAPS_NOT_VERIFIED")
    if POSITION_RE.search(" ".join(toks)) and "POSITION_NOT_PLANNED" not in sc.warnings:
        sc.warnings.append("POSITION_NOT_PLANNED")
    known = (STOP | ADJ | POSITIONAL | THEMES | FLAT_WORDS | set(ORDINALS) | REF_PREPS | set(NUMBERS) | set(STYLE) | set(NEG)
             | {"but", "also", "plus", "then", "and", "or", "not", "no", "x", "than", "more", "less", "up", "down", "out", "all", "i", "you", "me",
                "my", "we", "it", "its", "this", "that", "so", "very", "really", "too", "ground", "floor", "start", "end", "half", "side", "row"})
    vague_words = {w for ph in VAGUE for w in ph.split()}
    sc.ignored = [t for k, t in enumerate(toks) if not consumed[k] and t.isalpha() and len(t) > 2 and t not in known and t not in vague_words]
    if sc.ignored:
        sc.warnings.append("UNKNOWN_WORDS")
    # ignored vocabulary
    for k, t in enumerate(toks):
        if t in POSITIONAL and not consumed[k]:
            sc.warnings.append("POSITIONAL_IGNORED")
        if t in THEMES:
            sc.warnings.append("THEME_UNSUPPORTED")
    return sc


def _caption(sc: Scene) -> str:
    """Raw topics -> clamped canonical caption in generator order (doc 12 §2.4)."""
    phrases = []
    topics = dict(sc.topics)
    f = topics.get("floor")
    if f == "full":
        phrases.append("full floor")
    elif isinstance(f, str) and f.startswith("gaps:"):
        b = _clamp("floor_gaps", f[5:], sc.warnings)
        phrases.append(f"floor with {b} {'gap' if b == 'one' else 'gaps'}")
    elif isinstance(f, str) and f.startswith("chunks:"):
        b = _clamp("floor_chunks", f[7:], sc.warnings)
        phrases.append(f"giant gap with {b} {'chunk' if b == 'one' else 'chunks'} of floor")
    c = topics.get("ceiling")
    if c == "full":
        phrases.append("full ceiling")
    elif isinstance(c, str) and c.startswith("gaps:"):
        b = _clamp("ceiling_gaps", c[5:], sc.warnings)
        phrases.append(f"ceiling with {b} {'gap' if b == 'one' else 'gaps'}")
    for t in CONCEPTS:
        if t in ("floor", "ceiling") or t not in topics or t in sc.forbid:
            continue
        b = _clamp(t, topics[t], sc.warnings)
        phrases.append(counted_phrase(t, BUCKET_REP[b]))
    return "".join(p + ". " for p in phrases).strip()


def _pre_colon(t: str) -> str:
    """'pipes: 2' / 'enemies: none' -> '2 pipes' / 'no enemies' (a colon after a bare noun introduces its amount, not a theme)."""
    return _NOUN_COLON_NUM.sub(lambda m: _colon_amount(m) if _match_nouns([m.group(1)])[0] else m.group(0), t)   # 'underground level: two pipes' keeps its colon


def _split_scenes(text: str, max_scenes: int):
    """Split on sequence markers and order by start/end anchors (doc 12 §7)."""
    t = " " + text.strip().lower() + " "
    t = re.sub(r"\s+", " ", t).strip()
    prefix = None
    t = _pre_colon(t)
    if ":" in t:                                            # global prefix before a colon
        pre, rest = t.split(":", 1)
        gs = _scene(pre) if len(pre.split()) <= 8 else None
        if gs is not None and gs.topics and set(gs.topics) <= {"floor", "ceiling"} and not gs.forbid:
            prefix, t = pre, rest.strip()                   # "underground: three platforms": the prefix only sets the floor / ceiling
        else:
            t = t.replace(":", ",")                         # "no pipes: 2 coins": the prefix has content, so it is part of the sentence
    # "X at the start and Y at the end" is two scenes in that order ("X, then Y"): the one position phrase the scene planner can honour
    m, rev = AT_START_END.match(t), False
    if not m:
        m, rev = AT_END_START.match(t), True
    if m:
        first, second = (m.group(2), m.group(1)) if rev else (m.group(1), m.group(2))
        t = f"{first} then {second}"
    segs = [s.strip(" ,.") for s in SPLIT_RE.split(t) if s and s.strip(" ,.")]
    segs = [re.sub(r"^and\s+", "", s) for s in segs]
    start, mid, end = [], [], []
    for s in segs:
        s2 = ORD_RE.sub("", s)
        m = re.search(r"\b(?:and|,)\s+(?:end with|finish with|ending with|finishing with)\s+", s2)
        if m:                                                   # 'start with X and end with Y' inside one segment
            a, b = s2[:m.start()], s2[m.end():]
            (start if START_RE.match(a) else mid).append(START_RE.sub("", a))
            end.append(b)
        elif START_RE.match(s2):
            start.append(START_RE.sub("", s2))
        elif END_RE.match(s2):
            end.append(END_RE.sub("", s2))
        else:
            mid.append(s2)
    ordered = start + mid + end
    warns = []
    if len(ordered) > max_scenes:
        ordered = ordered[:max_scenes - 1] + [" and ".join(ordered[max_scenes - 1:])]
        warns.append("TOO_MANY_SCENES")
    return prefix, ordered, warns


def _canonical_scene(seg: str) -> Scene | None:
    """If every '.'-separated phrase is in the caption grammar, return the scene as typed (order as typed)."""
    phrases = [p.strip() for p in seg.split(".") if p.strip()]
    if not phrases:
        return None
    try:
        parsed = [(parse_phrase(p), p) for p in phrases]
    except ValueError:
        return None
    if len({c for (c, _), _ in parsed}) != len(parsed):    # a topic twice: let the full rules resolve it
        return None
    sc = Scene(join_phrases([p for _, p in parsed]))
    for (c, code), _ in parsed:                            # raw topic values, same vocabulary as the full rules
        if c == "floor":
            sc.topics[c] = "full" if code == 1 else (f"gaps:{BUCKETS[code - 2]}" if code <= 6 else f"chunks:{BUCKETS[code - 7]}")
        elif c == "ceiling":
            sc.topics[c] = "full" if code == 1 else f"gaps:{BUCKETS[code - 2]}"
        else:
            sc.topics[c] = BUCKETS[code - 1]
    for c, code in ((c, code) for (c, code), _ in parsed):                 # level-level want = the bucket's own range
        if c not in ("floor", "ceiling"):
            sc.want[c] = list(BUCKET_RANGE[BUCKETS[code - 1]])
    sc.warnings.append("CANONICAL_PASSTHROUGH")
    return sc


def normalize(text: str, max_scenes: int = 8) -> Normalized:
    prefix, segs, warns = _split_scenes(text, max_scenes)
    g_topics: dict = {}
    style: dict = {}
    if prefix:
        gs = _scene(prefix)
        if gs:
            g_topics = {k: v for k, v in gs.topics.items() if k in ("floor", "ceiling")}
    scenes = []
    for s in segs:
        canon = None if g_topics else _canonical_scene(s)   # a global prefix (e.g. 'underground') needs the full rules
        if canon is not None:                      # already in the training grammar: pass it through unchanged
            scenes.append(canon)
            continue
        toks = _tokens(s)
        for w in toks:
            if w in STYLE:
                style["difficulty_hint"] = STYLE[w]
                warns.append("STYLE_HINT")
        sc = _scene(s, g_topics)
        if sc is not None:
            warns += [w for w in sc.warnings if w.startswith("CONTRADICTION") or w in ("POSITION_NOT_PLANNED", "GAPS_NOT_VERIFIED")]      # surfaced even if the scene ends up empty
        if sc is None or not (sc.topics or sc.forbid):
            warns.append("EMPTY_SCENE")
            continue
        sc.caption = _caption(sc)
        if not sc.caption and sc.forbid and not any(w.startswith("CONTRADICTION") for w in sc.warnings):
            # "no pipes" alone: there is nothing to describe, but the prohibition must survive (it used to vanish -> an unconditioned
            # level with pipes).  A neutral full floor carries it; the level is verified against the prohibition (generate/adherence.py).
            sc.caption = NEUTRAL_CAPTION
            sc.warnings.append("NEGATION_ONLY")
        if sc.caption:
            scenes.append(sc)
        else:
            warns.append("EMPTY_SCENE")
    toks_all = _tokens(text)
    only_floor = all(not sc.want and not sc.forbid and set(sc.topics) <= {"floor"} and sc.topics.get("floor", "full") == "full" for sc in scenes)
    if scenes and only_floor and any(t in FLAT_WORDS for t in toks_all):
        # "a completely flat level" already maps to a full floor, but flat also means nothing else is on it
        from text2mario.captions.grammar import CONCEPTS
        for sc in scenes:
            sc.forbid = [c for c in CONCEPTS if c not in ("floor", "ceiling")]
            sc.warnings.append("FLAT_REQUEST")
        warns.append("FLAT_REQUEST")
    ignored: list = []
    flat_words = any(t in FLAT_WORDS for t in toks_all) or re.search(r"\b(?:just|only) (?:a )?(?:floor|ground)\b", " ".join(toks_all))
    if not scenes and flat_words:
        from text2mario.captions.grammar import CONCEPTS
        flat = Scene(NEUTRAL_CAPTION)
        flat.forbid = [c for c in CONCEPTS if c not in ("floor", "ceiling")]
        flat.warnings.append("FLAT_REQUEST")
        scenes.append(flat)
        warns = [w for w in warns if w != "EMPTY_SCENE"] + ["FLAT_REQUEST"]
    m = UNSUPPORTED_RE.search(unicodedata.normalize("NFKC", text).lower())     # on the typed words: "not many" must not become "no many" first
    if m:
        warns.append(f"UNSUPPORTED_PHRASE:{m.group(0)}")
    for sc in scenes:                                       # warnings that used to stay inside the scene and never reached the page
        warns += [w for w in sc.warnings if w.startswith("CLAMPED:") or w in ("DUPLICATE_TOPIC", "BARE_PLURAL_DEFAULT")]
        if sc.ignored:
            warns.append("UNKNOWN_WORDS:" + ",".join(dict.fromkeys(sc.ignored)))
        ignored += sc.ignored
    return Normalized(text, scenes, sorted(set(warns)), ignored, style)
