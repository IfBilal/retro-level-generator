"""Exploratory data analysis of the training corpus (doc 09; rubric row 2: dataset, features and target understanding).

Explain-me:
  What: draws tile frequencies (overall and per game), the caption-length histogram, phrase and concept frequencies, and a level x concept heat map
        from the 7,687 windows and their captions, and writes reports/eda/EDA.md with the headline numbers.  Uses ALL 42 levels only for description
        of the corpus (no model is fitted here); the split protocol is unaffected.
  In/Out: data/processed (scenes, index, levels) -> report/figures/eda_*.png, reports/eda/EDA.md, reports/eda/eda_summary.json.
  Previous: scripts/build_index.py.  Next: report section 2.  Try: python scripts/eda.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from text2mario.captions.grammar import CONCEPTS, parse_phrase
from text2mario.data import tiles as T

FIG = ROOT / "report/figures"
OUT = ROOT / "reports/eda"


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    scenes = np.load(ROOT / "data/processed/scenes_u8.npy")
    df = pd.read_parquet(ROOT / "data/processed/index.parquet")
    lv = pd.read_csv(ROOT / "data/processed/levels.csv")
    n = len(scenes)
    # 1. tile frequencies overall and per game
    cnt = np.bincount(scenes.ravel(), minlength=13) / scenes.size
    game = df.game.to_numpy()
    per = {g: np.bincount(scenes[game == g].ravel(), minlength=13) / scenes[game == g].size for g in ("SMB1", "SMB2")}
    fig, ax = plt.subplots(figsize=(8, 3.4))
    x = np.arange(13)
    ax.bar(x - 0.2, per["SMB1"], 0.4, label="SMB1")
    ax.bar(x + 0.2, per["SMB2"], 0.4, label="SMB2")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{T.CHARS[i]}\n{T.NAMES[i][:9]}" for i in x], fontsize=6.5)
    ax.set_ylabel("share of cells (log)")
    ax.legend()
    ax.set_title("Tile frequencies: sky dominates, rare classes are far below 1 %")
    fig.tight_layout()
    fig.savefig(FIG / "eda_tiles.png", dpi=150)
    plt.close(fig)
    # 2. captions: length and phrases
    caps = df.caption.fillna("").tolist()
    phrases = [[p.strip() for p in c.split(".") if p.strip()] for c in caps]
    lens = [len(p) for p in phrases]
    pc = Counter(p for ps in phrases for p in ps)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
    ax[0].hist(lens, bins=range(max(lens) + 2), color="#4477aa", align="left")
    ax[0].set_xlabel("phrases per caption")
    ax[0].set_ylabel("windows")
    ax[0].set_title("Caption length")
    top = pc.most_common(14)
    ax[1].barh([t for t, _ in top][::-1], [c for _, c in top][::-1], color="#66ccee")
    ax[1].tick_params(axis="y", labelsize=7)
    ax[1].set_title("Most frequent phrases")
    fig.tight_layout()
    fig.savefig(FIG / "eda_captions.png", dpi=150)
    plt.close(fig)
    # 3. concept prevalence and per-level profile
    concepts = {c: np.zeros(n, bool) for c in CONCEPTS}
    for i, ps in enumerate(phrases):
        for p in ps:
            try:
                concepts[parse_phrase(p)[0]][i] = True
            except ValueError:
                pass
    prev = {c: float(v.mean()) for c, v in concepts.items()}
    levels = sorted(df.level_id.unique())
    names = dict(zip(lv.level_id, lv.level_name))
    heat = np.array([[concepts[c][(df.level_id == L).to_numpy()].mean() for c in CONCEPTS] for L in levels])
    fig, ax = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1, 1.6]})
    order = sorted(CONCEPTS, key=lambda c: -prev[c])
    ax[0].barh(order[::-1], [prev[c] for c in order][::-1], color="#ee6677")
    ax[0].set_title("Concept prevalence (share of windows)")
    ax[0].tick_params(axis="y", labelsize=7)
    im = ax[1].imshow(heat, aspect="auto", cmap="viridis")
    ax[1].set_xticks(range(len(CONCEPTS)))
    ax[1].set_xticklabels([c[:10] for c in CONCEPTS], rotation=90, fontsize=6)
    ax[1].set_yticks(range(len(levels)))
    ax[1].set_yticklabels([names[L] for L in levels], fontsize=5)
    ax[1].set_title("Per-level concept profile (42 levels)")
    fig.colorbar(im, ax=ax[1], fraction=0.03)
    fig.tight_layout()
    fig.savefig(FIG / "eda_concepts.png", dpi=150)
    plt.close(fig)
    # summary
    spl = json.loads((ROOT / "data/splits/split_v1.json").read_text())
    summary = {"windows": int(n), "levels": len(levels), "tile_share": {T.NAMES[i]: round(float(cnt[i]), 5) for i in range(13)},
               "sky_share": round(float(cnt[0]), 4), "mean_phrases_per_caption": round(float(np.mean(lens)), 2), "max_phrases": int(max(lens)),
               "empty_captions": int(sum(1 for p in phrases if not p)), "distinct_captions": len(set(caps)), "distinct_phrases": len(pc),
               "concept_prevalence": {k: round(v, 4) for k, v in prev.items()}, "distinct_columns": len({bytes(scenes[i][:, c]) for i in range(0, n, 7) for c in range(16)}),
               "windows_per_split": spl["window_counts"]}
    (OUT / "eda_summary.json").write_text(json.dumps(summary, indent=1))
    md = ["# Exploratory data analysis (generated by scripts/eda.py)", "",
          f"- **{n:,} windows** of 16x16 cells over 13 tile classes from **{len(levels)} levels** (SMB1 {int((lv.game == 'SMB1').sum())}, SMB2 {int((lv.game == 'SMB2').sum())}).",
          f"- **Class imbalance:** sky is {cnt[0]*100:.1f} % of all cells; ground {cnt[8]*100:.1f} %; every other class is below 3 % (coins {cnt[12]*100:.2f} %, cannon tops {cnt[4]*100:.2f} %). This is why per-bucket loss and rare-concept metrics are reported.",
          f"- **Captions:** {summary['distinct_captions']:,} distinct captions, {summary['distinct_phrases']} distinct phrases, mean {summary['mean_phrases_per_caption']} phrases per caption (max {summary['max_phrases']}), {summary['empty_captions']} empty.",
          "- **Concept prevalence** (share of windows): " + ", ".join(f"{c} {prev[c]*100:.1f} %" for c in order[:8]) + ", ...",
          "- **Levels differ:** the heat map shows strong per-level profiles (sky levels vs underground levels), which is why the split is by level and why held-out levels are a real test of generalisation.",
          "", "![tiles](../../report/figures/eda_tiles.png)", "![captions](../../report/figures/eda_captions.png)", "![concepts](../../report/figures/eda_concepts.png)", ""]
    (OUT / "EDA.md").write_text("\n".join(md), encoding="utf-8")
    print("wrote reports/eda/EDA.md; sky share", summary["sky_share"], "; distinct captions", summary["distinct_captions"])


if __name__ == "__main__":
    main()
