"""Build the level-grouped split, CV folds, LOCK, prompt sets and clamp table (doc 10; gate G-P1 part 2).

Explain-me:
  What: finds near-duplicate level families, picks test (6 levels) and dev (5 levels) by exhaustive deterministic
        search, assigns everything else to train, measures leakage, writes data/splits/*, data/prompts/* and
        reports/splits/SPLIT_REPORT.md.  No randomness except the fixed seed used for P-rand.
  In/Out: data/processed/* + configs/data.yaml -> data/splits/{split_v1.json, LOCK, reused_segments.json},
          data/prompts/{p_dev,p_test,p_rand}.json, data/prompts/clamp_table.json.
  Try: python scripts/make_splits.py --check   (rebuild in memory and compare with the committed split)
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from text2mario.captions.grammar import BUCKETS, CONCEPTS, caption_set, parse_phrase  # noqa: E402
from text2mario.data import provenance, splits  # noqa: E402
from text2mario.data.hamming import nn_distance, nn_distance_by_level  # noqa: E402
from text2mario.data.index import level_grid  # noqa: E402

PROC = ROOT / "data" / "processed"
COLS = {c: "c_" + c.replace(" ", "_") for c in CONCEPTS}
QCOLS = {c: "q_" + c.replace(" ", "_") for c in CONCEPTS}
# Phrases of the paper's random-caption inventory that never occur in training (doc 10 §9.3).
OOV_PHRASES = {"ceiling": ["ceiling with a few gaps"], "upside down pipe": ["several upside down pipes"],
               "loose block": ["many loose blocks"]}


def level_stats(df, levels, scenes, chains):
    """Per-level concept window counts (42,16), distinct-column tile counts (42,13), window counts (42,)."""
    n = len(levels)
    conc = np.stack([df.groupby("level_id")[COLS[c]].sum().reindex(range(n)).to_numpy() for c in CONCEPTS], axis=1)
    tile = np.stack([np.bincount(level_grid(scenes, b, e).ravel(), minlength=13) for b, e in chains])
    return {"concept": conc.astype(float), "tile": tile.astype(float), "windows": levels.windows.to_numpy().astype(float)}


def clamp_table(df_train: pd.DataFrame, min_support: int) -> dict:
    """Doc 12 §5.4: supported buckets per topic (>= min_support TRAIN windows), nearest-supported map, bare-plural default."""
    topics = {c: df_train[QCOLS[c]].to_numpy() for c in CONCEPTS if c not in ("floor", "ceiling")}
    qf, qc = df_train[QCOLS["floor"]].to_numpy(), df_train[QCOLS["ceiling"]].to_numpy()
    topics["floor_gaps"] = np.where((qf >= 2) & (qf <= 6), qf - 1, 0)
    topics["floor_chunks"] = np.where(qf >= 7, qf - 6, 0)
    topics["ceiling_gaps"] = np.where(qc >= 2, qc - 1, 0)
    out = {"min_support": min_support, "supported": {}, "nearest": {}, "bare_plural_default": {}, "counts": {}}
    for name, q in topics.items():
        counts = [int((q == b).sum()) for b in range(1, 6)]
        sup = [i for i, c in enumerate(counts) if c >= min_support]
        out["counts"][name] = dict(zip(BUCKETS, counts))
        out["supported"][name] = [BUCKETS[i] for i in sup]
        near = {}
        for i, b in enumerate(BUCKETS):
            near[b] = b if i in sup or not sup else BUCKETS[min(sup, key=lambda j: (abs(j - i), j))]
        out["nearest"][name] = near
        ge2 = [(counts[i], -i) for i in range(1, 5)]
        out["bare_plural_default"][name] = BUCKETS[-max(ge2)[1]] if any(c for c, _ in ge2) else "two"
    cl, coin = df_train[QCOLS["coin line"]].to_numpy(), df_train[QCOLS["coin"]].to_numpy()
    imp = {}
    for b in range(1, 4):
        m = coin[cl == b]
        m = m[m > 0]
        if len(m):
            imp[BUCKETS[b - 1]] = BUCKETS[Counter(m.tolist()).most_common(1)[0][0] - 1]
    out["implied_coins"] = imp
    return out


def phrase_inventory(df) -> dict[str, list[str]]:
    """Topic -> sorted phrase list (training phrases + the 3 paper-only OOV phrases) for P-rand."""
    inv: dict[str, set[str]] = {c: set() for c in CONCEPTS}
    for cap in df.caption.unique():
        for ph in caption_set(cap):
            inv[parse_phrase(ph)[0]].add(ph)
    for topic, ph in OOV_PHRASES.items():
        inv[topic].update(ph)
    return {t: sorted(v, key=lambda p: (parse_phrase(p)[1], p)) for t, v in inv.items()}


def make_p_rand(inv: dict, seed: int, n: int, train_phrases: set[str]) -> list[dict]:
    """Paper protocol (doc 10 §9.3): k~U{1..10} distinct topics, one phrase each, coin line => add a coin phrase."""
    rng = np.random.default_rng(seed)
    topics = list(CONCEPTS)
    out = []
    for i in range(n):
        k = int(rng.integers(1, 11))
        chosen = [topics[j] for j in rng.choice(len(topics), size=k, replace=False)]
        phrases = [inv[t][int(rng.integers(len(inv[t])))] for t in chosen]
        if "coin line" in chosen and "coin" not in chosen:
            phrases.append(inv["coin"][int(rng.integers(len(inv["coin"])))])
        order = rng.permutation(len(phrases))
        phrases = [phrases[j] for j in order]
        cap = " ".join(f"{p}." for p in phrases)
        out.append({"pid": f"prand-{i:04d}", "set": "P-rand", "prompt": cap, "expected_caption": cap, "source": None,
                    "flags": {"reused_segment": False, "oov_phrases": [p for p in phrases if p not in train_phrases]}})
    return out


def build(write: bool):
    provenance.verify()
    cfg = yaml.safe_load((ROOT / "configs/data.yaml").read_text())
    sc = cfg["split"]
    df = pd.read_parquet(PROC / "index.parquet")
    levels = pd.read_csv(PROC / "levels.csv")
    scenes = np.load(PROC / "scenes_u8.npy")
    lid = df.level_id.to_numpy().astype(np.int64)
    n_lv, n = len(levels), len(df)
    from text2mario.data.index import level_ids
    _, _, chains = level_ids(scenes)

    nn_lv = nn_distance_by_level(scenes, lid, n_lv)
    L = splits.link_matrix(nn_lv, lid, n_lv, cfg["family_link_hamming"])
    fams = splits.find_families(L, cfg["family_link_threshold"])
    in_fam = {l for f in fams for l in f}
    game = levels.game.tolist()
    eligible = {g: [l for l in range(n_lv) if game[l] == g and l not in in_fam] for g in ("SMB1", "SMB2")}
    stats = level_stats(df, levels, scenes, chains)

    relaxations, test, dev = [], None, None
    c = dict(sc)
    for stage in range(3):
        if stage == 1:
            c["min_concept_windows"] = 5
            relaxations.append("C6 relaxed to >= 5 windows")
        if stage == 2:
            c["test_share_tol"] = c["dev_share_tol"] = 0.05
            relaxations.append("C4/C5 relaxed to +-0.05")
        t = splits.select_test(stats, eligible, n, c)
        if t[0] is None:
            continue
        left = {g: [l for l in eligible[g] if l not in t[0]] for g in eligible}
        d = splits.select_dev(stats, left, t[0], n, c)
        if d[0] is None:
            continue
        test, dev = t, d
        break
    if test is None:
        sys.exit("no feasible split (see doc 10 §5.3 step 5)")
    test_lv, dev_lv = test[0], dev[0]
    train_lv = [l for l in range(n_lv) if l not in test_lv and l not in dev_lv]
    split_of = {**{l: "train" for l in train_lv}, **{l: "dev" for l in dev_lv}, **{l: "test" for l in test_lv}}
    df["split"] = df.level_id.map(split_of)
    wc = {s: int((df.split == s).sum()) for s in ("train", "dev", "test")}

    # leakage of dev/test against train, and the documented reused segments (T6/T7)
    train_idx = np.flatnonzero(df.split == "train")
    leak, reused = {}, []
    for s in ("dev", "test"):
        idx = np.flatnonzero(df.split == s)
        dist, arg = nn_distance(scenes[idx], scenes[train_idx])
        leak[s] = {"nn_median": int(np.median(dist)), "le2": round(float((dist <= 2).mean()), 4),
                   "eq0": round(float((dist == 0).mean()), 4)}
        for j in np.flatnonzero(dist <= 2):
            i, k = int(idx[j]), int(train_idx[arg[j]])
            reused.append({"idx": i, "split": s, "level": int(lid[i]), "nn_idx": k, "nn_level": int(lid[k]),
                           "distance": int(dist[j]), "caption": df.caption[i]})
    reused_set = {r["idx"] for r in reused}

    folds = splits.make_cv_folds(levels, fams, test_lv, sc["cv_folds"])
    inv = phrase_inventory(df)
    train_phrases = {p for cap in df.caption.unique() for p in caption_set(cap)}
    p_rand = make_p_rand(inv, cfg["seed"], sc["n_p_rand"], train_phrases)

    def prompts(name, lvs):
        sel = df[df.level_id.isin(lvs) & (df.col_offset % sc["eval_stride"] == 0)].sort_values(["level_id", "col_offset"])
        return [{"pid": f"{name.lower().replace('-', '')}-{i:04d}", "set": name, "prompt": r.caption,
                 "source": {"idx": int(r.idx), "level": r.level_name, "col_offset": int(r.col_offset)},
                 "expected_caption": r.caption, "flags": {"reused_segment": int(r.idx) in reused_set, "oov_phrases": []}}
                for i, r in enumerate(sel.itertuples())]
    p_dev, p_test = prompts("P-dev", dev_lv), prompts("P-test", test_lv)

    split = {"schema_version": 1, "name": "split_v1", "seed": cfg["seed"],
             "corpus_content_hash": provenance.corpus_content_hash(),
             "algorithm": {"family_link_threshold": cfg["family_link_threshold"], "family_link_hamming": cfg["family_link_hamming"],
                           "test": {"smb1": sc["test_smb1"], "smb2": sc["test_smb2"], "share_target": sc["test_share"], "share_tol": c["test_share_tol"]},
                           "dev": {"n": 5, "share_target": sc["dev_share"], "share_tol": c["dev_share_tol"]},
                           "min_concept_windows": c["min_concept_windows"], "min_train_levels_per_concept": sc["min_train_levels_per_concept"],
                           "objective": "JS2(concept)+JS2(tile)+2*(share-target)^2", "tie_break": "lexicographic level_id",
                           "relaxations_applied": relaxations},
             "families": [{"id": f"F{i + 1}", "levels": f} for i, f in enumerate(fams)],
             "levels": [{"level_id": l, "name": levels.level_name[l], "game": game[l], "windows": int(levels.windows[l]),
                         "split": split_of[l], "family": next((f"F{i + 1}" for i, f in enumerate(fams) if l in f), None)}
                        for l in range(n_lv)],
             "splits": {"train": train_lv, "dev": dev_lv, "test": test_lv}, "window_counts": wc,
             "scores": {"test_J": round(test[2], 5), "dev_J": round(dev[2], 5),
                        "feasible_test": test[1], "feasible_dev": dev[1]},
             "leakage": leak, "cv_folds": folds,
             "prompt_sets": {"P-dev": len(p_dev), "P-test": len(p_test), "P-rand": len(p_rand)}}
    table = clamp_table(df[df.split == "train"], cfg["clamp_min_support"])
    return dict(split=split, reused=reused, p_dev=p_dev, p_test=p_test, p_rand=p_rand, clamp=table, df=df, L=L,
                stats=stats, levels=levels, scenes=scenes, train_phrases=train_phrases)


def write_report(r: dict) -> str:
    s, df = r["split"], r["df"]
    lines = ["# Split report (auto-generated by scripts/make_splits.py)", "",
             f"Families (tau={s['algorithm']['family_link_threshold']}): "
             + "; ".join(f"{f['id']} = {[s['levels'][l]['name'] for l in f['levels']]}" for f in s["families"]), "",
             f"Feasible test combinations: {s['scores']['feasible_test']}; feasible dev combinations: {s['scores']['feasible_dev']}",
             f"Relaxations applied: {s['algorithm']['relaxations_applied'] or 'none'}", "",
             "| Split | Levels | Windows | Share |", "|---|---|---|---|"]
    for sp in ("test", "dev", "train"):
        names = ", ".join(s["levels"][l]["name"] for l in s["splits"][sp]) if sp != "train" else f"{len(s['splits'][sp])} levels"
        lines.append(f"| {sp} | {names} | {s['window_counts'][sp]} | {s['window_counts'][sp] / len(df):.2%} |")
    lines += ["", f"Scores: test J = {s['scores']['test_J']}, dev J = {s['scores']['dev_J']}", "",
              "## Concept windows per split", "", "| Concept | Train | Dev | Test |", "|---|---|---|---|"]
    for c in CONCEPTS:
        cnt = [int(df[df.split == sp][COLS[c]].sum()) for sp in ("train", "dev", "test")]
        lines.append(f"| {c} | {cnt[0]} | {cnt[1]} | {cnt[2]} |")
    lines += ["", "## Leakage vs train (NN Hamming, stride 1)", "", "| Split | median | <=0 | <=2 |", "|---|---|---|---|"]
    for sp in ("dev", "test"):
        lk = s["leakage"][sp]
        lines.append(f"| {sp} | {lk['nn_median']} | {lk['eq0']:.2%} | {lk['le2']:.2%} |")
    lines += ["| *upstream random test (F17)* | *8* | *2.1%* | *-* |", "",
              "## CV folds (level ids)", ""] + [f"- fold {i}: {f}" for i, f in enumerate(s["cv_folds"])]
    lines += ["", f"Prompt sets: {s['prompt_sets']}", ""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="rebuild and compare with the committed files, write nothing")
    args = ap.parse_args()
    r = build(write=not args.check)
    new = splits.canonical_json(r["split"])
    if args.check:
        old = splits.canonical_json(json.loads(splits.SPLIT_PATH.read_text()))
        print("identical to committed split_v1.json" if new == old else "DIFFERENT from committed split_v1.json")
        sys.exit(0 if new == old else 1)
    sdir, pdir = ROOT / "data/splits", ROOT / "data/prompts"
    sdir.mkdir(exist_ok=True), pdir.mkdir(exist_ok=True)
    (sdir / "split_v1.json").write_text(json.dumps(r["split"], indent=1) + "\n")
    (sdir / "LOCK").write_text(splits.lock_hash(r["split"], r["split"]["corpus_content_hash"]) + "\n")
    (sdir / "reused_segments.json").write_text(json.dumps(r["reused"], indent=1) + "\n")
    for name in ("p_dev", "p_test", "p_rand"):
        (pdir / f"{name}.json").write_text(json.dumps(r[name], indent=1) + "\n")
    (pdir / "clamp_table.json").write_text(json.dumps(r["clamp"], indent=1) + "\n")
    rep = ROOT / "reports/splits"
    rep.mkdir(parents=True, exist_ok=True)
    (rep / "SPLIT_REPORT.md").write_text(write_report(r))
    # families side table (doc 09 §5)
    pd.DataFrame(r["L"]).to_parquet(PROC / "families_link_matrix.parquet")
    s = r["split"]
    print("test", s["splits"]["test"], [s["levels"][l]["name"] for l in s["splits"]["test"]])
    print("dev ", s["splits"]["dev"], [s["levels"][l]["name"] for l in s["splits"]["dev"]])
    print("windows", s["window_counts"], "scores", s["scores"], "leakage", s["leakage"])
    print("families", [f["levels"] for f in s["families"]])


if __name__ == "__main__":
    main()
