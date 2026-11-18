# 09 — Data Audit and Exploratory Data Analysis

> **Day:** 1 (Phase 1a). **Gate:** G-P1 (part 1: all facts re-verified). **Ledger rows:** F1–F19, D2, D6, D7.
> **Scripts:** `scripts/audit_dataset.py` (assertions → `reports/eda/AUDIT.json`, exit ≠ 0 on any hard failure), `scripts/build_index.py` (→ `data/processed/index.parquet`, `levels.csv`), `scripts/eda.py` (→ `reports/eda/EDA.md` + figures), `scripts/check_detector.py` (caption-detector reproduction, §6).
> **All expected values below were measured on 2026-11-17** at MarioDiffusion commit `561c76d4…` (doc 08 §2) and the VGLC fork commit `e91316a0…`. The audit re-measures them; a mismatch is a hard stop, not a number to update.

---

## 1. Assertion list (`scripts/audit_dataset.py`)

Each assertion writes `{id, description, expected, observed, pass, severity}` to `AUDIT.json`. **Severity H** = hard (non-zero exit); **W** = warning.

### 1.1 Ledger facts F1–F19

| ID | Assertion | Expected value | Sev. |
|---|---|---|---|
| F1 | Training file present at `data/source/Mar1and2_LevelsAndCaptions-regular.json`, content hash equals doc 08 §3 | `6a15264a…de2c` | H |
| F2 | Top-level type is `list`; every record is a `dict` with exactly the keys `{"prompt","scene","caption"}`; `prompt is None`; `scene` is a list of 16 lists of 16 ints; `caption` is a non-empty `str` | 100 % of records | H |
| F3 | `len(records)` | **7,687** | H |
| F4 | Unique scenes (hash of the uint8 bytes) | **7,620** → 67 extra copies in **41 duplicate groups** (16 within-level, 25 cross-level; largest group 20 copies) | H (count), W (group stats) |
| F5 | Every scene value is an `int` in `[0, 12]` (no characters, no `%` padding id 13) | min 0, max 12 | H |
| F6 | `sorted(smb.json["tiles"].keys())` equals `['-','<','>','?','B','E','Q','S','X','[',']','b','o']` and equals `text2mario.data.tiles.CHARS` | exact | H |
| F7 | Chain count (§2) = 42; chains 0–19 are SMB1, 20–41 are SMB2 | 42 = 20 + 22 | H |
| F8 | Within every chain, `S[i][:,1:] == S[i+1][:,:-1]` (stride 1) | all 7,645 within-chain pairs | H |
| F9 | `records[0:3606]` scenes equal `SMB1_Levels.json` element-wise; `records[3606:]` equal `SMB2_Levels.json` | exact | H |
| F10 | No record has a level-id field | true | H |
| F11 | Chain lengths: min **130**, max **358**; level widths (= windows + 15): min **145**, median **193**, max **373**; Σ widths = **8,317 columns in total (with repeats); only 653 distinct column patterns** | exact | H |
| F12 | Rows 0–1 contain exactly **64** non-sky cells, all tile ids **9 or 10** (`[`/`]`, 32 each), in **17** windows; row 2 contains **32** non-sky cells | exact | H |
| F13 | Window-level tile frequencies (%) | `- 87.13, < 0.18, > 0.18, ? 0.05, B 0.04, E 0.62, Q 0.19, S 1.70, X 8.66, [ 0.43, ] 0.43, b 0.03, o 0.34` (±0.01) | H |
| F14 | Captions: words 2–28 (mean 11.39); phrases 1–10 (mean 4.03); **62** distinct phrases; **44**-word vocabulary | exact | H |
| F15 | Every quantity word ∈ {one, two, a few, several, many}; `many` occurs **only** with coins (30) and question blocks (77) | exact | H |
| F16 | Upstream split file sizes 6,918 / 384 / 385 (sanity only; files are never used) | exact | W |
| F17 | Upstream test split: spans 42 levels; NN-Hamming to upstream train: median 8, 83.9 % ≤ 16, 2.1 % exact | ±0.5 pp | W (documentation of the leak) |
| F18 | (literature) 7,160 / 7,687 solvable by Baumgarten A* — **not asserted**; re-measured by doc 21's checker on real windows as the playability baseline | — | — |
| F19 | Our detector (§6) returns per-phrase contributing cells | API check | H |

### 1.2 Additional facts measured for this pack (A-series)

| ID | Assertion | Expected value | Sev. |
|---|---|---|---|
| A1 | VGLC fork (if available): the lexicographically sorted SMB1 then SMB2 `.txt` files, top-padded with sky to 16 rows (pipe tiles in the top VGLC row extended upward), sliced at stride 1, **equal the 42 chains byte-for-byte** | 42 / 42 | H (if VGLC present) |
| A2 | VGLC level heights | **14 rows: 20 levels; 15 rows: 21 levels; 16 rows: 1 level** (SMB2 World 3-1) | H (if VGLC present) |
| A3 | Bottom row (row 15) of windows: fully solid **35.01 %**; ≥ 50 % solid **71.00 %**; zero solid **16.64 %** (solid = tile ids {1,2,3,4,6,7,8,9,10,11}) | ±0.05 pp | H |
| A4 | Per-row non-sky % (rows 0…15): `0.0, 0.0, 0.0, 3.8, 4.9, 3.4, 9.8, 7.1, 5.0, 7.0, 14.6, 14.1, 11.4, 15.8, 43.2, 65.6` | ±0.1 pp | W |
| A5 | Enemy tiles never occur in rows 0–2 or row 15 (enemy-cell counts by row 3…14: `74, 128, 366, 518, 489, 507, 1191, 992, 1207, 1446, 2406, 2908`, window-level, overlapping) | row 15 count = 0 | W |
| A6 | Pipe-top integrity in raw data: no window has `<` not immediately followed by `>` (excluding the right edge) or `>` not immediately preceded by `<` (excluding the left edge) | 0 windows | H |
| A7 | Concept window counts (§4.3 table) | exact | H |
| A8 | Stored caption phrase order follows generator order (§6.2): **0** windows violate it, except that (pipe, upside-down pipe) and (tower, rectangular, irregular) appear in discovery order (146 and 104 adjacent inversions respectively) | exact | W |
| A9 | Unique captions **2,668**; unique phrase sets **2,665** | exact | W |
| A10 | Every caption that contains "coin line" also contains a coin phrase | 100 % | H |
| A11 | No caption contains "broken" (no broken pipes/cannons in the training data) | 0 | H |
| A12 | Cross-level near-duplicates: share of windows whose nearest window **in another level** is within Hamming 0 / 2 / 4 / 8 / 16: **0.99 / 7.49 / 16.09 / 40.17 / 75.40 %**; median 10 | ±0.1 pp | W |
| A13 | Adjacent-window Hamming (same level, offset 1): median **14**, mean **16.19**, p90 **30** | ±0.1 | W |

`AUDIT.json` also stores the environment (Python, numpy versions, machine profile) and the MANIFEST hashes it checked.

---

## 2. Chain (level) reconstruction

### 2.1 Algorithm
```text
input : S  uint8[N,16,16]   (N = 7687, record order preserved)
output: level_id[N], col_offset[N], chains = [(start, end_exclusive)]

starts = [0]
for i in 1..N-1:
    if not array_equal(S[i-1][:, 1:], S[i][:, :-1]):   # window i does not continue i-1
        starts.append(i)
ends = starts[1:] + [N]
for k, (b, e) in enumerate(zip(starts, ends)):
    level_id[b:e]   = k
    col_offset[b:e] = arange(e - b)                     # window start column in its level
game = SMB1 if level_id < 20 else SMB2                   # by F9: records[0:3606] are SMB1
```

### 2.2 Validations (all hard)
1. `len(chains) == 42`; chain 19 ends at index **3,606** exactly (game boundary).
2. **False-continuation guard:** two different levels could in principle continue each other by accident (level A's last window shifted by one equals level B's first window). The F9 boundary check plus A1 (VGLC match) rule this out. If VGLC is unavailable, also check that the number of chains equals the number of upstream level files the paper reports (20 + 22).
3. **Full-level reconstruction:** `level_grid[k] = concat(S[b], S[b+1][:, -1:], …, S[e-1][:, -1:], axis=1)` → shape (16, width_k). Assert `width_k == (e - b) + 15` and that re-slicing it reproduces `S[b:e]`. This recovers all 42 full-width levels from the corpus alone (used by doc 21's whole-level playability baseline and by doc 10's distinct-column statistics).
4. **Level naming (optional, A1):** load VGLC files in sorted order (SMB1 dir, then SMB2 dir), pad as described in doc 08 §1.1, compare against `level_grid[k]`. On success, `level_name[k]` is set from the file name (`mario-1-1.txt` → `SMB1 1-1`, `SuperMarioBros2(J)-World5-2.txt` → `SMB2 5-2`). On failure or without VGLC, names default to `L00`…`L41` and a warning is logged.

---

## 3. Per-level reference table (verified)

`windows` = 16-wide windows at stride 1; `width` = full level width in tiles; `H` = VGLC height before padding; `x-level dup≤2` = share of the level's windows whose nearest window in *another* level differs in ≤ 2 tiles; concept columns = number of the level's windows whose caption contains that concept; `family` = near-duplicate family from doc 10 §3 (link threshold 5 %).

| id | level | VGLC file | H | width | windows | x-level dup≤2 | ceil | pipe | ud-pipe | cannon | asc | desc | coin line | enemy | family |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | SMB1 1-1 | `mario-1-1.txt` | 14 | 202 | 187 | 7.5 % | 0 | 79 | 0 | 0 | 49 | 31 | 0 | 128 | F1 |
| 1 | SMB1 1-2 | `mario-1-2.txt` | 14 | 158 | 143 | 0.0 % | 130 | 29 | 0 | 0 | 16 | 0 | 53 | 139 | — |
| 2 | SMB1 1-3 | `mario-1-3.txt` | 14 | 150 | 135 | 1.5 % | 0 | 0 | 0 | 0 | 0 | 0 | 99 | 94 | — |
| 3 | SMB1 2-1 | `mario-2-1.txt` | 14 | 197 | 182 | 2.2 % | 0 | 95 | 0 | 0 | 16 | 0 | 0 | 170 | — |
| 4 | SMB1 3-1 | `mario-3-1.txt` | 14 | 197 | 182 | 3.3 % | 0 | 67 | 0 | 0 | 54 | 0 | 0 | 166 | — |
| 5 | SMB1 3-2 | `mario-3-2.txt` | 14 | 209 | 194 | 50.5 % | 0 | 17 | 0 | 0 | 21 | 0 | 33 | 177 | F1 |
| 6 | SMB1 3-3 | `mario-3-3.txt` | 14 | 149 | 134 | 0.0 % | 0 | 0 | 0 | 0 | 0 | 0 | 55 | 77 | — |
| 7 | SMB1 4-1 | `mario-4-1.txt` | 14 | 222 | 207 | 21.3 % | 0 | 66 | 0 | 0 | 20 | 0 | 62 | 75 | F1 |
| 8 | SMB1 4-2 | `mario-4-2.txt` | 14 | 187 | 172 | 0.0 % | 152 | 86 | 0 | 0 | 28 | 14 | 39 | 124 | — |
| 9 | SMB1 4-3 | `mario-4-3.txt` | 14 | 145 | 130 | 0.0 % | 0 | 0 | 0 | 0 | 0 | 0 | 71 | 47 | — |
| 10 | SMB1 5-1 | `mario-5-1.txt` | 14 | 198 | 183 | 32.2 % | 0 | 48 | 0 | 43 | 17 | 0 | 0 | 173 | F1 |
| 11 | SMB1 5-2 | `mario-5-2.txt` | 14 | 198 | 183 | 2.7 % | 0 | 51 | 0 | 29 | 55 | 0 | 57 | 167 | — |
| 12 | SMB1 5-3 | `mario-5-3.txt` | 14 | 150 | 135 | 0.0 % | 0 | 0 | 0 | 0 | 0 | 0 | 99 | 93 | — |
| 13 | SMB1 6-1 | `mario-6-1.txt` | 14 | 184 | 169 | 7.1 % | 0 | 17 | 0 | 0 | 82 | 0 | 45 | 16 | F1 |
| 14 | SMB1 6-2 | `mario-6-2.txt` | 14 | 215 | 200 | 1.0 % | 0 | 197 | 0 | 0 | 35 | 0 | 0 | 196 | — |
| 15 | SMB1 6-3 | `mario-6-3.txt` | 14 | 165 | 150 | 1.3 % | 0 | 0 | 0 | 0 | 0 | 0 | 118 | 0 | — |
| 16 | SMB1 7-1 | `mario-7-1.txt` | 14 | 176 | 161 | 11.2 % | 0 | 69 | 0 | 110 | 28 | 0 | 0 | 136 | F1 |
| 17 | SMB1 8-1 | `mario-8-1.txt` | 14 | 373 | 358 | 12.6 % | 0 | 143 | 0 | 0 | 18 | 0 | 65 | 309 | F1 |
| 18 | SMB1 8-2 | `mario-8-2.txt` | 15 | 217 | 202 | 4.0 % | 0 | 50 | 0 | 88 | 50 | 0 | 0 | 177 | — |
| 19 | SMB1 8-3 | `mario-8-3.txt` | 15 | 214 | 199 | 30.7 % | 0 | 51 | 0 | 46 | 0 | 15 | 0 | 143 | F2 |
| 20 | SMB2 1-1 | `SuperMarioBros2(J)-World1-1.txt` | 14 | 183 | 168 | 16.7 % | 0 | 87 | 0 | 0 | 32 | 0 | 21 | 161 | F2 |
| 21 | SMB2 1-2 | `SuperMarioBros2(J)-World1-2.txt` | 15 | 172 | 157 | 0.0 % | 132 | 76 | 0 | 0 | 33 | 0 | 0 | 102 | — |
| 22 | SMB2 1-3 | `SuperMarioBros2(J)-World1-3.txt` | 15 | 165 | 150 | 0.7 % | 0 | 0 | 0 | 0 | 0 | 0 | 52 | 83 | — |
| 23 | SMB2 2-1 | `SuperMarioBros2(J)-World2-1.txt` | 15 | 226 | 211 | 5.2 % | 0 | 54 | 0 | 0 | 48 | 0 | 0 | 145 | F2 |
| 24 | SMB2 2-2 | `SuperMarioBros2(J)-World2-2.txt` | 15 | 257 | 242 | 4.1 % | 0 | 118 | 0 | 0 | 16 | 30 | 95 | 235 | — |
| 25 | SMB2 2-3 | `SuperMarioBros2(J)-World2-3.txt` | 15 | 172 | 157 | 3.2 % | 0 | 0 | 0 | 0 | 6 | 0 | 39 | 107 | — |
| 26 | SMB2 3-1 | `SuperMarioBros2(J)-World3-1.txt` | 16 | 180 | 165 | 3.0 % | 0 | 71 | 0 | 29 | 17 | 0 | 16 | 142 | — |
| 27 | SMB2 3-3 | `SuperMarioBros2(J)-World3-3.txt` | 15 | 182 | 167 | 0.0 % | 0 | 80 | 0 | 0 | 0 | 0 | 45 | 148 | — |
| 28 | SMB2 4-1 | `SuperMarioBros2(J)-World4-1.txt` | 15 | 193 | 178 | 11.2 % | 0 | 97 | 0 | 0 | 36 | 0 | 0 | 84 | F2 |
| 29 | SMB2 4-2 | `SuperMarioBros2(J)-World4-2.txt` | 15 | 201 | 186 | 14.0 % | 0 | 41 | 0 | 45 | 0 | 0 | 0 | 143 | F2 |
| 30 | SMB2 4-3 | `SuperMarioBros2(J)-World4-3.txt` | 15 | 182 | 167 | 3.6 % | 0 | 0 | 0 | 0 | 0 | 0 | 73 | 67 | — |
| 31 | SMB2 5-1 | `SuperMarioBros2(J)-World5-1.txt` | 15 | 315 | 300 | 16.7 % | 0 | 116 | 61 | 32 | 36 | 0 | 16 | 234 | F2 |
| 32 | SMB2 5-2 | `SuperMarioBros2(J)-World5-2.txt` | 15 | 172 | 157 | 0.0 % | 142 | 77 | 38 | 0 | 16 | 0 | 41 | 111 | — |
| 33 | SMB2 6-1 | `SuperMarioBros2(J)-World6-1.txt` | 15 | 244 | 229 | 4.8 % | 0 | 119 | 51 | 16 | 53 | 14 | 30 | 193 | F2 |
| 34 | SMB2 6-3 | `SuperMarioBros2(J)-World6-3.txt` | 14 | 213 | 198 | 0.0 % | 0 | 0 | 0 | 0 | 6 | 0 | 71 | 135 | — |
| 35 | SMB2 8-1 | `SuperMarioBros2(J)-World8-1.txt` | 15 | 209 | 194 | 0.0 % | 0 | 90 | 85 | 45 | 16 | 0 | 0 | 194 | — |
| 36 | SMB2 A-1 | `SuperMarioBros2(J)-WorldA-1.txt` | 15 | 206 | 191 | 6.3 % | 0 | 84 | 34 | 0 | 17 | 0 | 19 | 183 | F2 |
| 37 | SMB2 A-3 | `SuperMarioBros2(J)-WorldA-3.txt` | 15 | 160 | 145 | 0.0 % | 0 | 0 | 0 | 0 | 0 | 0 | 95 | 39 | — |
| 38 | SMB2 B-1 | `SuperMarioBros2(J)-WorldB-1.txt` | 15 | 193 | 178 | 1.1 % | 0 | 114 | 0 | 0 | 0 | 0 | 0 | 171 | — |
| 39 | SMB2 B-3 | `SuperMarioBros2(J)-WorldB-3.txt` | 15 | 204 | 189 | 1.6 % | 0 | 22 | 0 | 0 | 0 | 0 | 33 | 90 | — |
| 40 | SMB2 C-2 | `SuperMarioBros2(J)-WorldC-2.txt` | 15 | 189 | 174 | 3.4 % | 0 | 0 | 0 | 0 | 8 | 0 | 92 | 61 | — |
| 41 | SMB2 D-1 | `SuperMarioBros2(J)-WorldD-1.txt` | 15 | 193 | 178 | 0.0 % | 0 | 76 | 43 | 71 | 32 | 0 | 0 | 167 | — |

Things to notice (they drive doc 10's split constraints):
- **Ceilings** exist in only **4 levels** (SMB1 1-2, 4-2; SMB2 1-2, 5-2: the underground levels).
- **Upside-down pipes** exist only in **6 SMB2 levels** (5-1, 5-2, 6-1, 8-1, A-1, D-1).
- **Descending staircases** exist in only **5 levels** (SMB1 1-1, 4-2, 8-3; SMB2 2-2, 6-1): 104 windows in total.
- **Cannons** exist in 11 levels. SMB1 6-3 has **zero** enemy windows; SMB1 6-1 has only 16.
- SMB1 3-2 is the most duplicated level: half its windows have a ≤ 2-tile near-copy in another level (mostly 5-1, 1-1, 8-1).

---

## 4. EDA plan (`scripts/eda.py` → `reports/eda/EDA.md`)

Every figure is generated by the script from `index.parquet` (no notebooks needed; a notebook may call the same functions). Figures are saved as both PNG (report) and SVG. Window-level statistics count interior columns about 16× (windows overlap); where it matters, a **distinct-column** version (from the 42 reconstructed levels) is shown alongside.

### 4.1 Figures

| Fig | Content | Expected headline numbers |
|---|---|---|
| EDA-1 | Tile frequency, log-scale bars, all / SMB1 / SMB2 | sky 87.13 / 88.14 / 86.24 %; ground X 8.66 / 7.80 / 9.42 %; distinct-column sky 87.19 % |
| EDA-2 | Per-row non-sky share (16-row strip) | A4 values; rows 14–15 dominate (43.2 %, 65.6 %) |
| EDA-3 | Enemy row histogram | A5; enemies sit on rows 3–14, mostly 13–14 |
| EDA-4 | Phrases-per-caption histogram | 1:225, 2:959, 3:1672, 4:2016, 5:1636, 6:785, 7:298, 8:79, 9:15, 10:2 |
| EDA-5 | Frequency of the 62 phrases (horizontal bars, sorted) | top: "full floor" 2,691; bottom: "two descending staircases" 1 (full list in doc 12 §2) |
| EDA-6 | 16 × 16 concept co-occurrence (lift = P(a,b) / P(a)P(b)), clustered | e.g., coin line ⇒ coin always (A10) |
| EDA-7 | 42 × 16 per-level concept heatmap (row-normalised by window count), ordered by game then level | shows the rare-concept concentration of §3 |
| EDA-8 | Bottom-row solidity histogram | A3: 35.0 % full, 16.6 % empty |
| EDA-9 | Cross-level NN-Hamming histogram + CDF; adjacent-window Hamming histogram | A12, A13 |
| EDA-10 | Level width distribution | min 145, median 193, max 373 |
| EDA-11 | Quantity-bucket table per concept (§4.2) | — |
| EDA-12 | Upstream-split leakage (re-computation of F17) as a CDF of NN-Hamming test→train; overlaid later with our split's curve (doc 10 §6) | median 8 (upstream) vs ≈ 14 (ours, example) |
| EDA-13 | Level near-duplicate graph (nodes = levels, edges = link ≥ 5 %), coloured by family | 2 families (doc 10 §3) |

### 4.2 Quantity buckets observed per concept (full corpus; window counts)

| Concept | one | two | a few | several | many | Max bucket with ≥ 20 examples |
|---|---|---|---|---|---|---|
| enemies | 2053 | 1587 | 1736 | 226 | 0 | several |
| platforms | 1707 | 1560 | 1123 | 109 | 0 | several |
| loose blocks | 1527 | 741 | 326 | 57 | 0 | several |
| question blocks | 599 | 206 | 330 | 273 | 77 | many |
| coins | 247 | 410 | 804 | 408 | 30 | many |
| pipes | 1601 | 545 | 238 | 3 | 0 | a few |
| coin lines | 1295 | 231 | 8 | 0 | 0 | two |
| towers | 908 | 142 | 27 | 6 | 0 | a few |
| cannons | 349 | 137 | 60 | 8 | 0 | a few |
| rectangular block clusters | 566 | 150 | 48 | 0 | 0 | a few |
| irregular block clusters | 253 | 30 | 4 | 0 | 0 | two |
| upside down pipes | 269 | 30 | 13 | 0 | 0 | two |
| ascending staircases | 820 | 41 | 0 | 0 | 0 | two |
| descending staircases | 103 | 1 | 0 | 0 | 0 | one |
| floor gaps ("floor with N gaps") | 1799 | 589 | 195 | 11 | 0 | a few |
| floor chunks ("giant gap with N chunks of floor") | 639 | 335 | 146 | 3 | 0 | a few |
| ceiling gaps ("ceiling with N gaps") | 168 | 8 | 0 | 0 | 0 | one |
| "full floor" / "full ceiling" | 2691 / 380 | | | | | — |

The last column feeds the prompt normalizer's **clamp table** (doc 12 §5.4). It must be **recomputed on the train split only** on Day 2: the full-corpus numbers above are for orientation.

### 4.3 Concept window counts (A7)

| Concept | All | SMB1 | SMB2 | Levels containing it |
|---|---|---|---|---|
| floor (any floor phrase) | 6,408 | 3,263 | 3,145 | 42 |
| ceiling | 556 | 282 | 274 | 4 |
| pipe | 2,387 | 1,065 | 1,322 | 31 |
| upside down pipe | 312 | 0 | 312 | 6 |
| coin line | 1,534 | 796 | 738 | 27 |
| coin | 1,899 | 979 | 920 | 27 |
| cannon | 554 | 316 | 238 | 11 |
| question block | 1,485 | 635 | 850 | 38 |
| enemy | 5,602 | 2,607 | 2,995 | 41 |
| platform | 4,499 | 1,869 | 2,630 | 41 |
| tower | 1,083 | 610 | 473 | 34 |
| ascending staircase | 861 | 489 | 372 | 29 |
| descending staircase | 104 | 60 | 44 | 5 |
| rectangular block cluster | 764 | 159 | 605 | 20 |
| irregular block cluster | 287 | 177 | 110 | 25 |
| loose block | 2,651 | 966 | 1,685 | 42 |

Floor phrase types: "full floor" 2,691; "floor with N gap(s)" 2,594; "giant gap with N chunk(s) of floor" 1,123; **no floor phrase** 1,279 (16.6 %).

---

## 5. `data/processed/index.parquet` schema (one row per record; also `index.csv`)

| Column | Type | Description |
|---|---|---|
| `idx` | int32 | record index in the source JSON (0…7686), primary key |
| `level_id` | int8 | 0…41 (§2) |
| `level_name` | string | e.g. `SMB1 4-2` or `L08` if VGLC unavailable |
| `game` | category | `SMB1` / `SMB2` |
| `col_offset` | int16 | window start column within its level |
| `level_width` | int16 | full width of the level |
| `vglc_height` | int8 | 14/15/16 (null if VGLC unavailable) |
| `caption` | string | stored caption, unchanged |
| `n_phrases` | int8 | phrase count |
| `c_<concept>` | bool × 16 | concept presence (`c_floor`, `c_ceiling`, `c_pipe`, `c_upside_down_pipe`, `c_coin_line`, `c_coin`, `c_cannon`, `c_question_block`, `c_enemy`, `c_platform`, `c_tower`, `c_ascending_staircase`, `c_descending_staircase`, `c_rectangular`, `c_irregular`, `c_loose_block`) |
| `q_<concept>` | int8 × 16 | quantity bucket index (0 = absent, 1 one, 2 two, 3 a few, 4 several, 5 many; floor/ceiling use their own code table in doc 12 §2.3) |
| `scene_hash` | string | sha1 hex of the 256 uint8 bytes |
| `dup_group` | int32 | id of the exact-duplicate group, −1 if unique |
| `nn_cross_level` | int16 | Hamming distance to the nearest window in a different level (A12) |
| `n_nonsky` | int16 | non-sky cells |
| `floor_solid_frac` | float32 | solid share of row 15 |
| `family_id` | int8 | near-duplicate family (doc 10 §3), −1 if none. Written by `make_splits.py` into a side table `data/processed/families.parquet` and joined, so the index stays split-agnostic |

The scene tensor itself is **not** in the parquet. It is stored once as `data/processed/scenes_u8.npy` (uint8, shape (7687, 16, 16), 1.97 MB, sha256 recorded in `AUDIT.json`), which is what the GPU-resident dataset loads (doc 11).

---

## 6. Caption detector (`src/text2mario/captions/detector.py`), written from the grammar and checked against the stored captions

> **Originality rule (ledger O3):** implement the rules from the grammar description in doc 12 §2 and the paper; do not port or transcribe MarioDiffusion's source. The stored captions are the *test oracle*, not the template. Mismatches are listed and explained.

The detector turns any tile grid into a canonical caption. It is the **measuring instrument** for C-score, exact match, the report card, word highlighting (future F5), and the generation verify loop. It must reproduce the stored captions.

### 6.1 Tile semantics (from `smb.json` descriptors)
- **solid** = `{X, S, ?, Q, <, >, [, ], B, b}`; **passable** = `{-, o}`; **enemy** `E` is neither solid nor passable (descriptors `enemy, damaging, hazard, moving`); **pipe** = `{<, >, [, ]}`; **coin** = `{o}`; **cannon** = `{B, b}`.
- Grid height H = 16 is assumed; width W may be any value ≥ 3 (the same rules apply to wider grids, but captions are defined and trained for W = 16). `FLOOR = 15`, `CEILING = 4` (= H − 12).
- Quantity word `q(n)`: 1 → one, 2 → two, 3–4 → a few, 5–9 → several, ≥ 10 → many. Nouns are pluralised when n ≠ 1 (`enemy → enemies`).

### 6.2 Rules, in the exact order phrases are emitted
An "accounted" cell set `ACC` tracks cells already explained (it affects only towers/clusters/loose blocks).

1. **Floor accounting.** All row-15 cells → `ACC`. *Double-thick floor:* if row 14, after mapping `<`→`[` and `>`→`]`, equals row 15, then `floor_row = 14` and row-14 cells → `ACC`; otherwise `floor_row = 15`.
2. **Floor phrase** (row 15 only). s = # solid, p = # passable. If s = W → `full floor`. Else if p = W → *no phrase*. Else if s > p → gaps = number of maximal runs of passable-or-enemy cells → `floor with q(gaps) gap(s)`. Else → chunks = number of maximal runs of solid cells → `giant gap with q(chunks) chunk(s) of floor`.
3. **Ceiling.** For a row r, `ceil(r)`: if all W cells are solid → `full ceiling`; else if solid > ⌊W/2⌋ → gaps = maximal runs of passable-or-enemy cells → `ceiling with q(gaps) gap(s)`; else none. Search c = 4, 3, 2, 1: compute `reg = ceil(c)` and `hi = ceil(c−1)`; stop at the first c where either exists. Rank phrases full < 1 gap < 2 gaps < a few < several < many < none. If `reg` exists and `hi` is not at least as full → the ceiling is row c; else if `hi` exists → row c−1; else no ceiling. The ceiling row's cells → `ACC`, plus every row above it that is identical to it (thick ceilings).
4. **Enemies:** count of `E` anywhere → `q enemy/enemies`.
5. **Question blocks:** count of `?` + `Q` → `q question block(s)`.
6. **Cannons:** count of `B` → `q cannon(s)`; all `B` cells → `ACC`.
7. **Broken cannons** (generated samples only): `b` cells whose upper neighbour is neither `B` nor `b` (or that sit in row 0) → `q broken cannon(s)`.
8. **Coins:** count of `o` → `q coin(s)`.
9. **Coin lines:** maximal horizontal runs of ≥ 2 `o` in rows 0–14 → `q coin line(s)`.
10. **Platforms:** in rows 1–14 excluding the ceiling row: maximal horizontal runs of ≥ 2 cells that are solid and not pipe, where every cell's upper and lower neighbours are non-solid → `q platform(s)`; run cells → `ACC`. (Membership in `ACC` is *not* checked when finding platforms.)
11. **Ascending staircases**, then **descending staircases.** A 3-step staircase at (col, r) with direction v (−1 ascending, +1 descending) needs solid cells at (r, col), (r+v, col+1), (r+2v, col+2), each with a non-solid upper neighbour (unchecked for row 0). Start rows: ascending r ∈ [3, 15], descending r ∈ [0, 13]. Scan col from 0 while col ≤ W−3; at each col, try start rows in increasing order; on the first hit, extend the length while the next 3-step pattern (shifted by one column and one row) still holds and col + length < W; count one staircase, jump col by length; otherwise col += 1. The solid columns beneath each step belong to the staircase. Ascending and descending cells → `ACC` after each pass.
12. **Pipes:** connected components (4-neighbour) of pipe tiles, never stepping right from `>`/`]` or left from `<`/`[` (so adjacent pipes stay separate). Classify each component from its bounding box: *valid pipe* (lip `<>` at the top with `[]`/`<>` pairs below until a non-pipe; partial pipes cut by the left or right screen edge are valid when the visible half is consistent); else *valid upside-down pipe* (mirror: lip at the bottom, body above, sky directly above the body invalidates); else *broken pipe*. Phrases are grouped per class in **discovery order** (row-major scan): `q pipe(s)`, `q upside down pipe(s)`, `q broken pipe(s)`. Pipe cells → `ACC`.
13. **Solid structures:** connected components (4-neighbour) of solid, non-pipe cells not in `ACC`, keeping components of ≥ 3 cells. Classify: **tower** if it does not touch the ceiling row, its bounding box is ≤ 2 wide and ≥ 3 tall, and it touches row `floor_row − 1`; else **rectangular block cluster** if it fills its bounding box; else **irregular block cluster**. Grouped per class in discovery order (`one tower`, `two rectangular block clusters`, …). Cells → `ACC`.
14. **Loose blocks:** count of `X`/`S` cells not in `ACC` → `q loose block(s)`.

The caption is the phrases joined as `"<p1>. <p2>. … <pn>."`. The detector also returns, per phrase: topic, raw count, and the set of contributing cells (F19). That cell set is the seam for word highlighting (doc 31).

### 6.3 Public API
```python
@dataclass(frozen=True)
class Detection:
    topic: str                      # one of TOPICS (18 C-score topics, doc 22)
    phrase: str                     # canonical phrase, e.g. "two pipes"
    count: int | None               # None for "full floor" / "full ceiling"
    cells: frozenset[tuple[int, int]]

def detect(grid: np.ndarray) -> list[Detection]          # grid: int (16, W), values 0..12
def caption(grid: np.ndarray) -> str                     # generator order (§6.2)
def caption_set(text: str) -> frozenset[str]             # phrase set of any caption string
```

### 6.4 Reproduction check (`scripts/check_detector.py`, Day 1 exit criterion)
- For all 7,687 records: `caption_set(caption(scene)) == caption_set(stored_caption)` → target **100 %**. Also report exact-string equality (expected 100 % if discovery order is reproduced).
- Mismatches go to `reports/eda/detector_mismatches.jsonl` (idx, level, stored, ours, rendered PNG path) for triage.
- **Risk note:** this audit could not run the upstream generator against its own stored captions (third-party code is never executed, doc 08 §8). The pinned commit's last message is "edited the wall recognition", so the stored captions *might* predate a rule change. If mismatches remain after fixing our own bugs and they cluster in one rule (e.g., towers vs. clusters), document the rule as "stored-caption behaviour" and match the **stored captions** (the training targets), not the latest upstream code. Acceptance per timeline Day 1: **≥ 99.5 % with every exception documented**, otherwise no evaluation work proceeds.
- Fixture unit tests (`tests/test_detector.py`): hand-built 16×16 grids for each rule: full floor; one/two gaps; giant gap with chunks; enemy on the floor row counted as a gap; double-thick floor; ceiling at row 4 vs row 3; a pipe; two adjacent pipes (`<><>`) staying separate; a left-edge partial pipe; an upside-down pipe hanging from row 0; a broken pipe (`<` without `>`); a tower vs. a 2×3 rectangular cluster touching the floor; an L-shaped irregular cluster; ascending vs descending staircases; a platform; coin line + coins; loose blocks remaining after structures.

---

## 7. Duplicates policy

- **Training keeps all 7,687 records** (duplicates are natural repetitions such as flat ground; removing them would distort the true distribution and break chain contiguity).
- **Novelty/copy-rate reference set** (doc 22) = the **unique** train scenes (min-distance results are identical, and it is faster).
- **Evaluation prompt sets** use stride-8 sub-sampling (D6), which also removes most near-identical consecutive prompts.
- Cross-level duplicates are handled at the split level (families, doc 10 §3), not by deleting data.

---

## 8. Data-quality issues register (what the model and evaluation must live with)

| # | Issue | Evidence | Consequence / handling |
|---|---|---|---|
| Q1 | Mixed VGLC heights (14/15/16) padded to 16 with sky | A2 | Rows 0–1 are almost always sky. The model learns this trivially, so don't over-interpret low loss there. Per-row metrics exclude rows 0–2 when reporting "structure" errors. |
| Q2 | Pipe tiles extended into the padding (hanging pipes) | F12 (64 cells, all `[`/`]`) | Upside-down pipes may legally touch row 0; the sanitizer (doc 20) must not "repair" them. |
| Q3 | Real data is only ~93 % solvable (paper, F18) | literature | Playability targets for generated data are set **relative to real dev windows measured by our checker** (doc 21), never at 100 %. |
| Q4 | Rare concepts concentrated in few levels | §3, §4.3 | Split constraints (doc 10 §4); per-concept metrics carry wide CIs; concept-balanced sampling ablation (D9). |
| Q5 | All enemies are one symbol `E` (Goomba semantics) | paper | "Koopas", "piranha plants" etc. all normalise to "enemy" (doc 12). The game renders one walker type (doc 27). |
| Q6 | Captions carry **no position information** ("pipe on the left" is unrepresentable) | grammar | The normalizer drops positional words with a warning; positional control is a future seam (erase-and-re-ask, doc 31). |
| Q7 | "many" exists only for coins and question blocks; several phrases have < 20 examples | §4.2 | Clamp table (doc 12 §5.4); report which prompts were clamped. |
| Q8 | Enemy on the floor row counts as a floor gap (rule 2) | detector rule | Generated enemies standing on row 15 change floor phrases. This is a known metric quirk; it is documented in the C-score section of doc 22. |
| Q9 | Bottom row: only 71 % of real windows are ≥ 50 % solid; 16.6 % have no floor at all | A3 | Any "valid floor" metric (gate G-P3) must be **calibrated on real dev windows** (real ≥ 95 % pass) before being applied to samples. |
| Q10 | Cross-level near-duplicates (7.49 % of windows within 2 tiles) | A12 | Family-aware splitting (doc 10 §3); copy-rate is judged against the real held-out baseline. |
| Q11 | Caption order in the stored data is fixed (generator order) | A8 | Phrase shuffling at training time (D8a) is required so the model doesn't learn order artefacts; the normalizer emits generator order for determinism (doc 12). |
| Q12 | SMB2 is denser (sky 86.2 % vs 88.1 %) and has the only upside-down pipes | EDA-1, §4.3 | Report all metrics per game. |

---

## 9. Ledger conflicts / refinements found by this audit (for the ledger owner)

1. **F12 (refine):** "VGLC levels are 14 rows; rows 0–1 are top padding" is true for 20 of 42 levels only. Measured heights: 14 rows (20 levels), 15 rows (21), 16 rows (1: SMB2 World 3-1). Padding is 2, 1 or 0 rows. Pipe tiles in the top VGLC row are extended upward into the padding. The corpus-wide observation (64 non-sky cells in rows 0–1, all `[`/`]`) remains correct.
2. **M7 (unverified):** `L ≤ 48` tokens could not be measured here (Hugging Face is unreachable from the audit sandbox). Captions have at most 38 whitespace tokens including periods; WordPiece may split some words. Day-1 measurement in doc 11 §6 decides; if the observed maximum (including augmented captions) exceeds 48, raise `max_len` to 64 and log it.
3. **M12 (minor):** with the example split of doc 10, train = 5,709 windows → 89.2 steps/epoch (the ledger says ≈ 87). Within the stated approximation; recompute after the real split.
4. **G-P1 leakage clause:** with family-aware splitting, the example split still has 1.86 % of test windows (20) within Hamming ≤ 2 of a train window (mostly generic flat-floor windows). Doc 10 §6 makes these the auto-generated "documented reused segments" with a hard cap of 3 %.
5. **G-P3 floor clause:** see Q9. The gate must use a validator calibrated on real data.
6. **C-score topic count:** the paper's scorer averages over **18** topics (16 concepts + broken pipe + broken cannon), not 16. Doc 22 must use 18 to match the paper.
