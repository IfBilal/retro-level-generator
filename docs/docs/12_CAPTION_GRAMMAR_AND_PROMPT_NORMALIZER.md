# 12 — Caption Grammar and Prompt Normalizer

> **Days:** 2 (grammar file, lexicon, P-casual set committed), 8 (normalizer v1), 13 (UI wiring). **Ledger rows:** F14, F15, D8c, E2 (P-casual), S2, S3.
> **Modules:** `src/text2mario/captions/grammar.py` (canonical grammar, parser, emitter), `src/text2mario/text/normalizer.py` (casual → canonical), data files `data/prompts/lexicon.yaml`, `data/prompts/clamp_table.json`, `data/prompts/p_casual.jsonl`.
> **Pipeline position:** `user text → normalize() → [SceneSpec…] → canonical caption per scene → MiniLM → U-Net`. The model is trained on canonical captions (plus 27 % casual paraphrases, D8c, doc 11 §5.2). The normalizer makes inference text look like training text.

---

## 1. Why a rule-based normalizer (and not an LLM)

1. **The target language is tiny and closed:** 62 phrases, 44 words, 16 topics, 5 quantity words (F14–F15). A deterministic mapper covers it exactly.
2. **Offline and fast:** the app must run with Wi-Fi off (G-P7). The normalizer runs in < 5 ms with no model download.
3. **Testable and explainable:** every decision traces to a lexicon line or a rule. The UI can show *why* ("'lots of goombas' → 'several enemies' — clamped: the model never saw 'many enemies'"), which is also the seed of the future "see what it heard" feature (doc 31).
4. **No evaluation contamination:** an LLM rewriter would add an uncontrolled, non-reproducible component to every metric.
5. **Seam for later:** an optional LLM rewrite can sit *in front of* the normalizer behind a flag (`normalizer.llm_prepass: false`). It is never used on evaluation paths.

---

## 2. The canonical caption grammar (`grammar.py`)

### 2.1 Formal grammar (EBNF)
```
caption     ::= phrase ( ". " phrase )* "."
phrase      ::= floor | ceiling | counted
floor       ::= "full floor"
              | "floor with " Q " gap" PL
              | "giant gap with " Q " chunk" PL " of floor"
ceiling     ::= "full ceiling" | "ceiling with " Q " gap" PL
counted     ::= Q " " NOUN(Q)
Q           ::= "one" | "two" | "a few" | "several" | "many"
NOUN(one)   ::= "enemy" | "question block" | "cannon" | "coin" | "coin line" | "platform"
              | "ascending staircase" | "descending staircase" | "pipe" | "upside down pipe"
              | "tower" | "rectangular block cluster" | "irregular block cluster" | "loose block"
NOUN(other) ::= the plural of NOUN(one)   ("enemy" → "enemies"; "cluster" → "clusters"; else + "s")
PL          ::= "" if Q = "one" else "s"
```
Semantic constraints: at most one phrase per topic; "coin line" implies a coin phrase with Q ≥ two (A10). Quantity buckets: one = 1, two = 2, a few = 3–4, several = 5–9, many ≥ 10 (F15).
**Detector-only phrases** (appear in generated samples, never in prompts or training captions): "Q broken pipe(s)", "Q broken cannon(s)".

### 2.2 The 62 phrases observed in training (full corpus counts)

| Topic | Phrases (count) |
|---|---|
| floor (9) | full floor (2691) · floor with one gap (1799) · floor with two gaps (589) · floor with a few gaps (195) · floor with several gaps (11) · giant gap with one chunk of floor (639) · giant gap with two chunks of floor (335) · giant gap with a few chunks of floor (146) · giant gap with several chunks of floor (3) |
| ceiling (3) | full ceiling (380) · ceiling with one gap (168) · ceiling with two gaps (8) |
| enemy (4) | one enemy (2053) · two enemies (1587) · a few enemies (1736) · several enemies (226) |
| question block (5) | one (599) · two (206) · a few (330) · several (273) · many (77) |
| cannon (4) | one (349) · two (137) · a few (60) · several (8) |
| coin (5) | one (247) · two (410) · a few (804) · several (408) · many (30) |
| coin line (3) | one (1295) · two (231) · a few (8) |
| platform (4) | one (1707) · two (1560) · a few (1123) · several (109) |
| ascending staircase (2) | one (820) · two (41) |
| descending staircase (2) | one (103) · two (1) |
| pipe (4) | one (1601) · two (545) · a few (238) · several (3) |
| upside down pipe (3) | one (269) · two (30) · a few (13) |
| tower (4) | one (908) · two (142) · a few (27) · several (6) |
| rectangular block cluster (3) | one (566) · two (150) · a few (48) |
| irregular block cluster (3) | one (253) · two (30) · a few (4) |
| loose block (4) | one (1527) · two (741) · a few (326) · several (57) |

`grammar.py` exposes `PHRASES_SEEN` (these 62, generated from the train corpus at build time, not hand-typed) and validates any caption against §2.1.

### 2.3 Codes used in `index.parquet` (`q_<concept>`) and metrics
- Counted topics: 0 absent, 1 one, 2 two, 3 a few, 4 several, 5 many.
- Floor: 0 no floor phrase, 1 full floor, 2–6 "floor with {one…many} gap(s)", 7–11 "giant gap with {one…many} chunk(s) of floor".
- Ceiling: 0 none, 1 full ceiling, 2–6 "ceiling with {one…many} gap(s)".

### 2.4 Generator order (the normalizer's output order)
floor → ceiling → enemies → question blocks → cannons → coins → coin lines → platforms → ascending staircases → descending staircases → pipes → upside down pipes → towers → rectangular block clusters → irregular block clusters → loose blocks.
(In stored captions, pipes vs upside-down pipes and the three structure types appear in *discovery* order (A8). The normalizer uses the fixed order above for determinism; the model is order-invariant thanks to phrase shuffling, D8a.)

### 2.5 Topics: 16 concepts vs 18 scoring topics
The normalizer and SceneSpec use the **16 concepts**. The C-score (doc 22) uses **18 topics** = 16 + broken pipe + broken cannon (the paper's scorer).

---

## 3. Normalizer architecture (`normalizer.py`)

```
S0 preprocess → S1 split scenes → S2 match lexicon (longest match) → S3 attach quantities/negations
→ S4 resolve topics & conflicts → S5 implied rules → S6 clamp to supported buckets → S7 emit
```

| Stage | What it does |
|---|---|
| **S0 Preprocess** | Unicode NFKC; lowercase; curly quotes → straight; `→`/`->`/`=>` → ` -> `; `upside-down` → `upside down`; `?-block`/`?block`/`? block` → `question block`; `power-up` → `power up`; `x3` / `×3` after a noun → quantity 3; collapse whitespace. Keep `;` `:` `,` `.` as tokens. |
| **S1 Split scenes** | §7. Produces ordered segments plus an optional global prefix. |
| **S2 Match** | Tokenise (`[a-z0-9]+(?:'[a-z]+)?` plus punctuation tokens). Longest-match-first over a trie built from the lexicon (§6: topic variants, floor/ceiling templates, quantity expressions, negations, positional words, adjectives, themes, style words). Every token is either consumed by a match or reported as unrecognized (stopwords like "a", "the", "with", "and", "of", "level", "stage", "please", "make", "put", "add" are consumed silently). |
| **S3 Attach** | For each topic match, find its quantity (§5.1–5.3) and negation (§5.6). |
| **S4 Resolve** | One value per topic per scene; floor/ceiling resolution (§5.5); duplicate handling (§5.7). |
| **S5 Implied** | coin line ⇒ coins; pyramid ⇒ ascending + descending staircase; global prefix applied to all scenes (§7.3). |
| **S6 Clamp** | Map each bucket to the nearest supported bucket for its topic (§5.4), with a warning. |
| **S7 Emit** | Canonical caption per scene in generator order (§2.4); SceneSpec objects; warnings; character spans per topic. |

**Passthrough:** if the whole input parses under §2.1 (one scene), it is returned as a single scene **re-ordered to generator order**, with only the clamp applied (warning `CANONICAL_PASSTHROUGH`).

---

## 4. Data structures and API

```python
Bucket = Literal["one", "two", "a few", "several", "many"]

@dataclass(frozen=True)
class SceneSpec:
    topics: dict[str, str]           # concept -> bucket, or for floor: "full" | "gaps:<bucket>" | "chunks:<bucket>",
                                     # for ceiling: "full" | "gaps:<bucket>"
    raw_topics: dict[str, str]       # before clamping (S6) — used by the P-casual test
    forbid: frozenset[str]           # concepts the user negated ("no enemies")
    caption: str                     # canonical, generator order; "" if no topics (unconditional)
    spans: dict[str, list[tuple[int, int]]]   # concept -> char spans in the ORIGINAL text (seam: word highlighting)
    source_text: str

@dataclass(frozen=True)
class NormalizedPrompt:
    scenes: list[SceneSpec]          # ordered; length ≥ 1
    style: dict[str, str]            # e.g. {"difficulty_hint": "hard"} — stored, not used by the MVP model
    warnings: list[tuple[str, str]]  # (code, human message)
    unrecognized: list[str]
    count_scope: Literal["per_scene", "whole_level"]

def normalize(text: str, *, max_scenes: int | None = None,
              count_scope: Literal["per_scene", "whole_level"] = "per_scene",
              lexicon: "Lexicon | None" = None, clamp: "ClampTable | None" = None) -> NormalizedPrompt
```
- `max_scenes` is supplied by the generator (doc 20): the number of content windows for the chosen width. Extra segments are merged into the last scene (`TOO_MANY_SCENES`).
- `count_scope="per_scene"` (MVP default): counts describe each 16-wide scene, exactly as training captions do. `"whole_level"` (seam for the future pacing feature): a count N over K disjoint 16-column blocks becomes `round(N/K)` per block (minimum 1 where N ≥ 1), and the original totals are kept in `style["level_totals"]`.
- Warning codes: `CLAMPED`, `POSITIONAL_IGNORED`, `THEME_UNSUPPORTED`, `STYLE_HINT`, `UNRECOGNIZED_TOKEN`, `DANGLING_QUANTITY`, `DUPLICATE_TOPIC`, `EMPTY_SCENE`, `TOO_MANY_SCENES`, `IMPLIED_COINS`, `NEGATION_FORBID`, `BARE_PLURAL_DEFAULT`, `CANONICAL_PASSTHROUGH`.

---

## 5. Rules

### 5.1 Quantity expressions → value → bucket

| Expression (examples) | Value | Bucket |
|---|---|---|
| `zero`, `no`, `none`, `without`, `not any`, `don't add (any)`, `X-free` | 0 | absent → negation (§5.6) |
| `a`, `an`, `one`, `1`, `single`, `a single`, `lone`, `a lone`, `solo`, `the` (+ singular noun) | 1 | one |
| `two`, `2`, `a couple of`, `couple of`, `a pair of`, `pair of`, `double`, `twin` | 2 | two |
| `three`, `four`, `3`, `4`, `a few`, `few`, `some`, `a handful of`, `handful of` | 3–4 | a few |
| `five`…`nine`, `5`…`9`, `several`, `a bunch of`, `bunch of`, `half a dozen`, `a number of` | 5–9 | several |
| `ten`+, `10`+, `a dozen`, `dozens of`, `many`, `lots of`, `a lot of`, `loads of`, `tons of`, `a ton of`, `plenty of`, `heaps of`, `numerous`, `countless`, `a swarm of`, `an army of`, `a crowd of`, `a horde of` | ≥ 10 | many |
| Ranges `2-3`, `2 to 3`, `two or three`, `one or two` | ⌈midpoint⌉ | bucket of that value (one or two → 2 → two; two or three → 3 → a few) |
| `at least N`, `N or more`, `N+` | N | bucket(N) |
| `more than N`, `over N` | N + 1 | bucket(N+1) |
| `at most N`, `up to N`, `no more than N` | N | bucket(N) |
| `fewer than N`, `less than N` | max(N − 1, 1) | bucket |

Number words `zero`…`twenty`, `thirty`…`ninety`, `hundred` are supported; anything ≥ 10 is "many".

### 5.2 Attachment
A quantity attaches to the nearest topic noun to its **right** within 3 tokens, skipping adjectives and stopwords ("three **green** pipes", "a couple **of big** goombas"). If no noun is to the right, the pattern `<noun> x N`, `<noun>: N`, `<noun> (N)` attaches it to the noun on the **left**. A quantity that attaches to nothing raises `DANGLING_QUANTITY` (e.g., "3 thwomps", where "thwomps" is unknown).

### 5.3 Bare nouns (no quantity)
- Singular ("pipe", "a pipe" is handled by §5.1) → **one**.
- Plural or mass noun ("pipes", "goombas", "gold", "treasure", "rubble") → **default bucket** = the train-split **mode among buckets ≥ two** for that topic (`clamp_table.json["bare_plural_default"]`), warning `BARE_PLURAL_DEFAULT`. Full-corpus values for orientation: enemies → a few; question blocks → a few; coins → a few; all other topics → two (descending staircase → two → clamped to one; ceiling gaps → clamped to one).

### 5.4 Clamping (`clamp_table.json`, computed on the **train split** on Day 2)
- `supported[topic]` = buckets with ≥ **20** train windows (threshold in `configs/data.yaml: clamp_min_support`).
- An unsupported bucket moves to the **nearest supported bucket** in the order one < two < a few < several < many (ties → lower), with warning `CLAMPED` ("you asked for 7 pipes; the model saw at most 'a few pipes' per screen").
- Full-corpus orientation (doc 09 §4.2; regenerate on train): max supported = enemies several · platforms several · loose blocks several · question blocks many · coins many · pipes a few · towers a few · cannons a few · rectangular a few · coin lines two · irregular two · upside-down pipes two · ascending staircases two · descending staircases **one** · floor gaps a few · floor chunks a few · ceiling gaps **one**.

### 5.5 Floor and ceiling resolution (per scene)
Floor evidence collected in S3: `full` template ("solid ground", "flat ground", "no gaps", "no pits", …), gap count (gap nouns: gap, pit, hole, chasm, …, not attached to a ceiling noun), chunk count (island, chunk of ground, …), `giant` template ("giant/big/huge gap", "mostly pits", "sky level", …), `no_floor` ("no floor", "no ground").
1. If chunk count or `giant` → `chunks:<Q>` (Q = chunk count, default **one**).
2. Else if gap count (explicit, or bare plural "pits" → default) → `gaps:<Q>`.
3. Else if `full` → `full`.
4. Else if `no_floor` → omit the floor phrase and add `floor` to `forbid`.
5. Else no floor phrase.
Conflicts (e.g., "no gaps" plus "two pits"): the later mention wins, with `DUPLICATE_TOPIC`.

Ceiling: `full` from "underground", "cave", "roof", "ceiling" (bare), "cavern", "tunnel", "indoors", "covered", "sewer"; `gaps:<Q>` when a gap noun is attached to a ceiling noun ("holes in the ceiling", "a ceiling with 4 holes", "gaps in the roof"); "no ceiling"/"no roof" → forbid ceiling.

### 5.6 Negation → `forbid`
"no X", "without X", "zero X", "not any X", "don't add (any) X", "X-free", "never X". The scope covers a coordinated list right after the trigger ("without coins or pipes" negates both) and ends at the next content token that is not a topic noun or conjunction.
- The topic is **omitted** from the caption and added to `forbid` (warning `NEGATION_FORBID`).
- Special cases: "no gaps/pits/holes" → floor `full` (not a forbid); "no floor/ground" → forbid floor; "no ceiling/roof" → forbid ceiling.
- Why a forbid list: regular captions have no absence phrases, and phrase dropout (D8b) makes omission only a soft signal. The generator's verify loop (doc 20, G5 seam) re-samples a window whose detector output contains a forbidden concept (MVP: soft, up to the same retry budget; reported on the report card).

### 5.7 Duplicates within a scene
Same topic twice: if the second mention has "more", "another", "extra", "additional" → add the values; otherwise the **last** explicit mention wins. Either way, warning `DUPLICATE_TOPIC`.

### 5.8 Implied rules (S5)
- **Coin line ⇒ coins** (A10): if a coin line is present and coins are not, add coins with bucket by data mode: one coin line → a few coins; two coin lines → several coins; a few coin lines → several coins (warning `IMPLIED_COINS`). If coins are present with bucket "one", raise them to "two" (a line has ≥ 2 coins).
- **Pyramid ⇒** one ascending staircase + one descending staircase (each multiplied by the pyramid count, then clamped).

### 5.9 Ignored-with-warning vocabulary
- **Positional** (captions carry no positions, Q6): left, right, middle, center/centre, top, bottom, above, below, over, under, near, next to, far, corner, high, low, beginning/start/end *when not a scene marker* → `POSITIONAL_IGNORED`.
- **Themes not in the data:** castle, lava, water, underwater, snow/snowy, ice, desert, night, forest, space, ghost house → `THEME_UNSUPPORTED`.
- **Style/difficulty** (seam for the future difficulty dial): easy, chill, relaxed, simple, beginner → `easy`; medium, normal → `medium`; hard, difficult, challenging, brutal, insane, rage quit → `hard`. Stored in `style.difficulty_hint`; `STYLE_HINT`.
- **Adjectives consumed silently:** green, red, golden, shiny, big, small, tall, short, long, tiny, huge (except inside floor templates), floating (except inside MWEs), fun, cool, awesome, scary, classic, nice, cute, epic.

---

## 6. The lexicon (`data/prompts/lexicon.yaml`, committed Day 2)

Each variant has `sg`/`pl` forms (mass nouns: `mass: true`) and `split: train | heldout`.
- **train** variants are used by caption augmentation (D8c, doc 11 §5.2), so the model sees them in training.
- **heldout** variants are **never** used in augmentation. They exist only for inference-time normalisation and for measuring model-side robustness with the normalizer off (§9.3).
- The normalizer uses **both** halves.

### 6.1 Topic nouns

| Topic (canonical sg / pl) | train variants | heldout variants |
|---|---|---|
| enemy / enemies | goomba(s), koopa(s), turtle(s), bad guy(s), monster(s), creature(s), walker(s) | koopa troopa(s), baddie(s), foe(s), mob(s), critter(s), villain(s), opponent(s), mushroom guy(s), piranha plant(s), hammer bro(s), spiny / spinies, buzzy beetle(s) |
| question block(s) | ? block(s) (→ S0), mystery block(s), item block(s), bonus block(s), question mark block(s) | power up block(s), lucky block(s), surprise block(s), prize block(s), coin block(s) |
| cannon(s) | bullet bill(s), blaster(s), turret(s), bullet cannon(s) | bill blaster(s), launcher(s), gun(s), artillery (mass) |
| coin(s) | gold coin(s), gold (mass), collectible(s) | money (mass), cash (mass), treasure (mass), loot (mass) |
| coin line(s) | line(s) of coins, row(s) of coins, coin row(s), coins in a row | coin trail(s), string(s) of coins, coin arc(s) |
| platform(s) | ledge(s), floating platform(s), bridge(s), brick platform(s) | shelf / shelves, walkway(s), overhang(s), floating ledge(s) |
| tower(s) | pillar(s), column(s), tall block(s), block tower(s) | post(s), stack(s) of blocks, obelisk(s) |
| ascending staircase(s) | staircase(s) (bare = ascending), stairs (one set), steps, rising stairs, stairs going up, upward stairs, pyramid(s) (→ §5.8) | stairway(s), ramp(s), climbing steps, uphill steps |
| descending staircase(s) | stairs going down, downward stairs, descending steps | falling stairs, downstairs, downhill steps, declining staircase(s) |
| pipe(s) | tube(s), warp pipe(s), green pipe(s) | chimney(s), drain pipe(s), sewer pipe(s) |
| upside down pipe(s) | hanging pipe(s), ceiling pipe(s), pipe(s) hanging down | inverted pipe(s), pipe(s) from the ceiling, dangling pipe(s), pipe(s) from the sky |
| rectangular block cluster(s) | block cluster(s) (bare = rectangular), rectangle(s) of blocks, brick wall(s), big block(s) | wall(s), block rectangle(s), box(es) of bricks |
| irregular block cluster(s) | irregular cluster(s), jagged block cluster(s), block pile(s), pile(s) of blocks | messy blocks, rubble (mass), odd shaped blocks, lumpy blocks |
| loose block(s) | brick(s), block(s) (bare), single block(s), scattered blocks, breakable block(s) | floating block(s), lone block(s), stray block(s), brick block(s) |

### 6.2 Floor and ceiling templates

| Meaning | train | heldout |
|---|---|---|
| floor `full` | solid ground, flat ground, solid floor, flat floor, continuous floor, no gaps, no pits, no holes, flat level | unbroken ground, safe ground, ground all the way |
| gap nouns | gap(s), pit(s), hole(s), break(s) in the floor/ground | chasm(s), drop(s), pitfall(s), abyss |
| floor nouns (context only) | floor, ground | terrain, land |
| `giant` floor | giant gap, big gap, huge gap, massive pit, mostly gaps, mostly pits | sky level, bottomless level |
| chunk nouns | island(s), floating island(s), chunk(s) of ground, island(s) of ground | piece(s) of ground, bit(s) of floor, floor segment(s) |
| ceiling `full` | underground, cave, roof, ceiling (bare) | cavern, tunnel, indoors, covered, sewer |

### 6.3 Determinism of the split
The train/heldout labels above are **fixed in this document** and copied verbatim into `lexicon.yaml`. They are never regenerated randomly, so the augmentation vocabulary can't drift between runs or machines.

---

## 7. Multi-scene splitting (S1)

### 7.1 Markers (whole-word, case-insensitive)
- **Sequence:** `then`, `and then`, `followed by`, `after that`, `afterwards`, `next` (when sentence-initial or after a comma), `later`, `;`, `->`, `|`, newline.
- **Ordinals:** `first`, `second`, `third`, `fourth`, `fifth`, `last`, `finally`, `lastly`.
- **Anchors:** start anchors `start with`, `starting with`, `begin with`, `beginning with`, `open with`; end anchors `end with`, `ending with`, `finish with`, `finishing with`, `at the end`.
- **Commas never split** ("three goombas, two pipes and a coin line" is one scene).

### 7.2 Ordering
Segments get positions: start-anchored segments first (in textual order), then unanchored/ordinal segments in textual order, then end-anchored segments. Example: "end with a tower, start with 2 mystery blocks" → [mystery blocks] then [tower].

### 7.3 Global prefix
Text before a `:` (or a leading segment containing only ceiling/floor-template/theme/style words and no counted topic) is a **global prefix**. Its floor/ceiling values and style hints apply to every scene. Example: "an underground level: two pipes, then 4 goombas" → both scenes get `full ceiling`.

### 7.4 Empty and excess segments
- A segment with no topic (e.g., "then make it harder") contributes only style hints; it is dropped as a scene (`EMPTY_SCENE`).
- If the segment count exceeds `max_scenes`, the trailing segments are merged into the last allowed scene (`TOO_MANY_SCENES`).
- How scenes map onto windows is decided by the generator (doc 20): scenes are spread over the content windows in order, each scene covering a contiguous run of windows.

---

## 8. P-casual: 60 handwritten prompts with expected outputs (E2)

Stored in `data/prompts/p_casual.jsonl`, one record per prompt:
```json
{"pid": "cas-dev-01", "split": "dev", "category": "A", "prompt": "a flat level with 3 goombas and 2 pipes",
 "expected": {"scenes": [{"raw_topics": {"floor": "full", "enemy": "a few", "pipe": "two"}, "forbid": []}],
              "style": {}, "warning_codes": [], "unrecognized": []},
 "expected_caption_fullcorpus": ["full floor. a few enemies. two pipes."]}
```
- `raw_topics` are **pre-clamp** and are what the test compares exactly. `default` means "bare-plural default from `clamp_table.json`".
- The post-clamp caption is derived at test time from the train-split clamp table. The `expected_caption_fullcorpus` column below assumes the full-corpus table (§5.4) and is for readability only.
- Categories: **A** synonyms + numbers · **B** vague quantities · **C** floor/ceiling templates · **D** negation · **E** multi-scene · **F** clamping · **G** noise (adjectives, positions, themes, style, OOV) · **H** canonical passthrough/reordering.

### 8.1 Dev (30) — used for normalizer development

| pid | Prompt | Expected raw topics (forbid) | Expected caption(s) (full-corpus clamp) |
|---|---|---|---|
| cas-dev-01 A | a flat level with 3 goombas and 2 pipes | floor full, enemy a few, pipe two | full floor. a few enemies. two pipes. |
| cas-dev-02 A | one tube, four koopas and a mystery block | pipe one, enemy a few, question block one | a few enemies. one question block. one pipe. |
| cas-dev-03 A | two bullet bills and 6 bricks | cannon two, loose block several | two cannons. several loose blocks. |
| cas-dev-04 A | 5 coins in a row | coin several, coin line one | several coins. one coin line. |
| cas-dev-05 A | a ledge with a ? block above it | platform one, question block one; POSITIONAL | one question block. one platform. |
| cas-dev-06 A | three pillars and a staircase | tower a few, ascending staircase one | one ascending staircase. a few towers. |
| cas-dev-07 B | lots of coins and a couple of goombas | coin many, enemy two | two enemies. many coins. |
| cas-dev-08 B | some pipes and a bunch of bad guys | pipe a few, enemy several | several enemies. a few pipes. |
| cas-dev-09 B | a handful of platforms | platform a few | a few platforms. |
| cas-dev-10 B | plenty of question blocks | question block many | many question blocks. |
| cas-dev-11 B | goombas and tubes | enemy default, pipe default | a few enemies. two pipes. |
| cas-dev-12 C | solid ground, no pits, with one pipe | floor full, pipe one | full floor. one pipe. |
| cas-dev-13 C | ground with 2 holes and an enemy | floor gaps:two, enemy one | floor with two gaps. one enemy. |
| cas-dev-14 C | an underground stretch with a few bricks | ceiling full, loose block a few | full ceiling. a few loose blocks. |
| cas-dev-15 C | mostly pits with two islands of ground | floor chunks:two | giant gap with two chunks of floor. |
| cas-dev-16 D | full floor with three pipes and no enemies | floor full, pipe a few (forbid enemy) | full floor. a few pipes. |
| cas-dev-17 D | a level without coins, just platforms | platform default (forbid coin) | two platforms. |
| cas-dev-18 D | no gaps and zero cannons, two towers | floor full, tower two (forbid cannon) | full floor. two towers. |
| cas-dev-19 E | start with flat ground and a pipe, then lots of goombas, then end with a staircase | S1 floor full, pipe one · S2 enemy many · S3 ascending staircase one | full floor. one pipe. / several enemies. (CLAMPED) / one ascending staircase. |
| cas-dev-20 E | two pipes followed by a coin line | S1 pipe two · S2 coin line one (+ IMPLIED coin a few) | two pipes. / a few coins. one coin line. |
| cas-dev-21 E | first some platforms; second, a cannon; finally a big gap | S1 platform a few · S2 cannon one · S3 floor chunks:one | a few platforms. / one cannon. / giant gap with one chunk of floor. |
| cas-dev-22 E | begin with a mystery block and end with stairs going down | S1 question block one · S2 descending staircase one | one question block. / one descending staircase. |
| cas-dev-23 E | an underground level: two pipes, then 4 goombas | global ceiling full · S1 pipe two · S2 enemy a few | full ceiling. two pipes. / full ceiling. a few enemies. |
| cas-dev-24 F | 7 pipes | pipe several | a few pipes. (CLAMPED) |
| cas-dev-25 F | two staircases going down | descending staircase two | one descending staircase. (CLAMPED) |
| cas-dev-26 G | a fun green pipe on the left with a scary goomba | pipe one, enemy one; POSITIONAL | one enemy. one pipe. |
| cas-dev-27 G | make it hard: lots of enemies and 2 cannons | enemy many, cannon two; style hard | several enemies. two cannons. (CLAMPED) |
| cas-dev-28 G | castle level with lava and three coins | coin a few; THEME ×2 | a few coins. |
| cas-dev-29 H | full floor. two enemies. one pipe. | floor full, enemy two, pipe one; PASSTHROUGH | full floor. two enemies. one pipe. |
| cas-dev-30 H | floor with one gap. a few coins. one coin line. two platforms. | as given; PASSTHROUGH | floor with one gap. a few coins. one coin line. two platforms. |

### 8.2 Test (30) — evaluated once, Day 16 (heavy use of heldout variants; one deliberate OOV)

| pid | Prompt | Expected raw topics (forbid) | Expected caption(s) (full-corpus clamp) |
|---|---|---|---|
| cas-test-01 A | a single warp pipe and two koopa troopas | pipe one, enemy two | two enemies. one pipe. |
| cas-test-02 A | 4 power-up blocks and 1 chimney | question block a few, pipe one | a few question blocks. one pipe. |
| cas-test-03 A | three baddies near a column | enemy a few, tower one; POSITIONAL | a few enemies. one tower. |
| cas-test-04 A | a row of coins over a walkway | coin line one (+ IMPLIED coin a few), platform one; POSITIONAL | a few coins. one coin line. one platform. |
| cas-test-05 A | two bill blasters and a lone brick | cannon two, loose block one | two cannons. one loose block. |
| cas-test-06 A | 8 foes and 3 stray blocks | enemy several, loose block a few | several enemies. a few loose blocks. |
| cas-test-07 B | tons of treasure | coin many | many coins. |
| cas-test-08 B | a pair of hanging pipes | upside down pipe two | two upside down pipes. |
| cas-test-09 B | a few critters and some ledges | enemy a few, platform a few | a few enemies. a few platforms. |
| cas-test-10 B | loads of prize blocks | question block many | many question blocks. |
| cas-test-11 B | pillars and villains | tower default, enemy default | a few enemies. two towers. |
| cas-test-12 C | unbroken ground with 2 obelisks | floor full, tower two | full floor. two towers. |
| cas-test-13 C | terrain with three chasms | floor gaps:a few | floor with a few gaps. |
| cas-test-14 C | a cavern with one inverted pipe | ceiling full, upside down pipe one | full ceiling. one upside down pipe. |
| cas-test-15 C | a sky level with 3 floating islands | floor chunks:a few | giant gap with a few chunks of floor. |
| cas-test-16 D | no turtles, no tubes, just a pyramid | ascending one, descending one (forbid enemy, pipe) | one ascending staircase. one descending staircase. |
| cas-test-17 D | coin-free, with two block piles | irregular two (forbid coin) | two irregular block clusters. |
| cas-test-18 D | don't add any cannons, but put five goombas | enemy several (forbid cannon) | several enemies. |
| cas-test-19 E | begin with a stairway, after that a dozen coins, and finish with a wall | S1 ascending one · S2 coin many · S3 rectangular one | one ascending staircase. / many coins. / one rectangular block cluster. |
| cas-test-20 E | first a stretch of pits, then a bridge with question mark blocks, then two tubes | S1 floor gaps:default · S2 platform one, question block default · S3 pipe two | floor with two gaps. / a few question blocks. one platform. / two pipes. |
| cas-test-21 E | starting with a lone goomba, followed by a pair of cannons, ending with stairs going down | S1 enemy one · S2 cannon two · S3 descending one | one enemy. / two cannons. / one descending staircase. |
| cas-test-22 E | a cave: some coins -> three hanging pipes | global ceiling full · S1 coin a few · S2 upside down pipe a few | full ceiling. a few coins. / full ceiling. two upside down pipes. (CLAMPED) |
| cas-test-23 E | end with a tower, start with 2 mystery blocks | S1 question block two · S2 tower one | two question blocks. / one tower. |
| cas-test-24 F | ten coin lines | coin line many (+ IMPLIED coins) | several coins. two coin lines. (CLAMPED) |
| cas-test-25 F | a ceiling with 4 holes | ceiling gaps:a few | ceiling with one gap. (CLAMPED) |
| cas-test-26 G | an awesome red tube at the top right and two shiny coins | pipe one, coin two; POSITIONAL | two coins. one pipe. |
| cas-test-27 G | rage quit difficulty: many pits and many spinies | floor gaps:many, enemy many; style hard | floor with a few gaps. several enemies. (CLAMPED ×2) |
| cas-test-28 G | a snowy world with 3 thwomps and two pipes | pipe two; THEME; unrecognized ["thwomps"]; DANGLING_QUANTITY | two pipes. |
| cas-test-29 H | giant gap with one chunk of floor. two platforms. one enemy. | as given (reordered); PASSTHROUGH | giant gap with one chunk of floor. one enemy. two platforms. |
| cas-test-30 H | full ceiling. floor with one gap. several question blocks. | as given (reordered); PASSTHROUGH | floor with one gap. full ceiling. several question blocks. |

Notes: (i) cas-test-24: "ten coin lines" → raw coin line many → clamp to two → implied coins for two lines = several. The implied rule runs on the clamped value (S5 after S6 for implied topics; implement S6 for explicit topics first, then S5, then clamp implied topics). (ii) cas-test-28 is the deliberate OOV case: graceful degradation is the expected behaviour, not a mapping.

### 8.3 Honesty about P-casual-test
The test prompts are published in this plan, so the implementer can see them. P-casual-test is therefore a **frozen acceptance set**, not a blind test. Two safeguards:
1. **Code review rule:** the normalizer source and tests may not contain any P-casual-test prompt substring (`tests/test_normalizer_hygiene.py` greps for them). Behaviour must come from the lexicon and rules of §5–§7.
2. **P-casual-blind (optional, recommended):** on Day 16 the user writes 10 fresh casual prompts with expected topics *without showing them to Claude Code first*, and runs them once. Report them separately as the true out-of-sample normalizer estimate.

---

## 9. Evaluation of the normalizer

### 9.1 Metrics (`scripts/eval_normalizer.py`)
| ID | Metric | Definition |
|---|---|---|
| N1 | Scene-count accuracy | share of prompts with the correct number of scenes |
| N2 | Topic F1 | micro precision/recall/F1 over (scene, topic) pairs (scenes aligned by order) |
| N3 | Bucket accuracy | share of correctly detected topics with the correct **raw** bucket/template |
| N4 | **Exact match** (headline) | scenes, raw topics, buckets and forbid sets all equal |
| N5 | Forbid accuracy | exact match of forbid sets |
| N6 | Warning recall | expected warning codes ⊆ produced codes |
| N7 | Unrecognized-token rate | unconsumed content tokens / all content tokens |

### 9.2 Targets
- **Dev (gate for normalizer v1, Day 8):** N4 ≥ 90 % (≥ 27/30), N2 F1 ≥ 0.97, N6 = 100 %.
- **Test (Day 16, report only):** reported with exact counts. Expectation: N4 ≥ 80 %.
- Unit tests: every lexicon variant, sg and pl, maps to its topic; every §5.1 row; every §7 marker; the paraphrase invariant of doc 11 (T-A4).

### 9.3 Downstream and robustness studies (reported in doc 22/23)
- **P-casual generation quality:** generate with the normalizer ON (the product path) and score each scene with the C-score against its expected canonical caption.
- **Normalizer-OFF ablation:** feed the raw casual prompt directly to MiniLM (no normalization) and compare train-variant prompts against heldout-variant prompts. The gap measures what casual-paraphrase augmentation (D8c) buys and how well MiniLM's semantics generalise to unseen synonyms.

---

## 10. Implementation checklist (Day 8 normalizer v1)
- [ ] `lexicon.yaml` exactly mirrors §6 (a test diffs the YAML against a table parsed from this doc's §6 snapshot committed in `tests/fixtures/lexicon_expected.json`).
- [ ] `clamp_table.json` built from the **train split** on Day 2: `supported`, `nearest`, `bare_plural_default`, `implied_coins`.
- [ ] `normalize()` meets §9.2 dev targets; hygiene test passes.
- [ ] Spans returned for every topic (seam for word highlighting, doc 31).
- [ ] The Streamlit UI shows the canonical caption(s) and warnings beneath the prompt box (doc 28).
