"""Regenerate every report figure from saved artifacts:  python scripts/make_figures.py  -> report/figures/*.png

Explain-me:
  What: draws the pipeline diagram, training curves, the G-P3 broken-vs-copy trade-off curves, the checker calibration and
        physics-contract plots, the CFG sweep and copies the key sample sheets.  Missing inputs are skipped with a note, so the
        script can be run at any stage.  Nothing is hand-made (CLAUDE.md §3).
  In/Out: runs/*/metrics.csv, reports/gates/*/RESULT.json, reports/dev_eval, reports/final -> report/figures/.
  Try: delete report/figures and re-run: every file comes back identical (figures are deterministic).
"""
import csv
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

done, skipped = [], []


def note(name, ok):
    (done if ok else skipped).append(name)


def read_csv(p):
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def smooth(y, k=10):
    out, acc = [], []
    for v in y:
        acc.append(v)
        acc = acc[-k:]
        out.append(sum(acc) / len(acc))
    return out


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(12, 3.2))
    ax.axis("off")
    boxes = ["Typed sentence", "Prompt normalizer\n(rules + lexicon)", "Scene captions\n(training grammar)", "MiniLM (frozen)\n+ adapter",
             "U-Net denoiser\n(18.3 M params)\nCFG, DDIM-50", "Sliding windows\n16x16, stride 8\nRePaint outpainting", "Framing +\nsanitizer", "Playability\ncheckerL3", "Playable game\n(Phaser, offline)"]
    boxes[7] = "Playability\nchecker (A*)"
    n = len(boxes)
    for i, t in enumerate(boxes):
        x = i * (1.0 / n) + 0.005
        ax.add_patch(FancyBboxPatch((x, 0.35), 0.1, 0.4, boxstyle="round,pad=0.005", fc="#dbe9ff" if i not in (4, 7) else "#ffe3b3", ec="#335", transform=ax.transAxes))
        ax.text(x + 0.05, 0.55, t, ha="center", va="center", fontsize=7.5, transform=ax.transAxes)
        if i < n - 1:
            ax.annotate("", xy=(x + 0.108, 0.55), xytext=(x + 0.1, 0.55), xycoords="axes fraction", arrowprops=dict(arrowstyle="->"))
    ax.text(0.5, 0.12, "failed check -> re-sample the window where the search got stuck (max 5 per window, 20 per level)", ha="center", fontsize=8, transform=ax.transAxes)
    fig.savefig(FIG / "fig01_pipeline.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    note("fig01_pipeline", True)


def fig_training_curves():
    runs = [("U-lin", "uncond linear"), ("U-cos", "uncond cosine"), ("U-lin-d", "uncond linear (dense run)"), ("U-cos-d", "uncond cosine (dense run)"),
            ("U-lin-ad", "linear + aux CE"), ("C-main", "conditional (cosine)")]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    any_ = False
    for r, lab in runs:
        p = ROOT / "runs" / r / "metrics.csv"
        if p.exists():
            rows = read_csv(p)
            ax.plot([int(x["step"]) for x in rows], smooth([float(x["loss"]) for x in rows]), label=lab, lw=1.2)
            any_ = True
    ax.set_yscale("log"); ax.set_xlabel("step"); ax.set_ylabel("training loss (smoothed)"); ax.legend(fontsize=7); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(FIG / "fig02_training_loss.png", dpi=150); plt.close(fig)
    note("fig02_training_loss", any_)


def fig_p3_tradeoff():
    sources = [("P3_lin_dense", "linear, MSE"), ("P3_cos_dense", "cosine, MSE"), ("P3_lin_auxdense", "linear, aux CE 0.001")]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    any_ = False
    for d, lab in sources:
        p = ROOT / "reports/gates" / d / "RESULT.json"
        if not p.exists():
            continue
        rows = sorted(json.loads(p.read_text())["diagnostics"]["all_checkpoint_rows"], key=lambda r: r["step"])
        ax[0].plot([r["step"] for r in rows], [r["broken_rate"] for r in rows], "o-", label=lab)
        ax[1].plot([r["step"] for r in rows], [r["copy_rate"] for r in rows], "o-", label=lab)
        any_ = True
    ax[0].axhline(0.05, ls="--", c="k"); ax[0].set_title("broken-structure rate (gate <= 0.05)")
    ax[1].axhline(0.1543, ls="--", c="k"); ax[1].set_title("copy rate (gate <= 0.154)")
    for a in ax:
        a.set_xlabel("step"); a.grid(alpha=.3); a.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "fig03_p3_tradeoff.png", dpi=150); plt.close(fig)
    note("fig03_p3_tradeoff", any_)


def fig_val_buckets():
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    any_ = False
    for run, sched in (("U-lin", "linear"), ("U-cos", "cosine")):
        p = ROOT / "runs" / run / "step_0015000.val.json"
        if p.exists():
            v = json.loads(p.read_text())
            b = v.get("buckets") or v.get("val_loss_per_bucket") or v
            vals = [x["val_loss"] if isinstance(x, dict) else x for x in (b if isinstance(b, list) else list(b.values()))][:10]
            ax.plot(range(10), vals, "o-", label=f"{sched} dev loss")
            any_ = True
    ax.set_yscale("log"); ax.set_xlabel("noise bucket (t/100)"); ax.set_ylabel("eps-MSE"); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(FIG / "fig04_val_buckets.png", dpi=150); plt.close(fig)
    note("fig04_val_buckets", any_)


def fig_checker():
    cal = ROOT / "reports/gates/P5b/calibration.json"
    rep = ROOT / "reports/gates/P5b/game_replay.json"
    if not cal.exists():
        note("fig05_checker", False)
        return
    c = json.loads(cal.read_text())
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    ax[0].bar(["full levels\n(36)", "framed windows\n(432)"], [c["full_level_solved_rate"], c["windows"]["solved_rate"]], color=["#4477aa", "#66ccee"])
    ax[0].axhline(0.9, ls="--", c="k"); ax[0].set_ylim(0, 1); ax[0].set_title("checker: share of REAL train/dev solved (target 0.90)")
    if rep.exists():
        r = json.loads(rep.read_text())["rows"]
        names = [x["case"] for x in r]
        ax[1].bar(range(len(r)), [1 if x["checker"] == "solved" else 0 for x in r], color="#ee6677", alpha=.6, label="checker solved")
        ax[1].plot(range(len(r)), [1 if (x.get("single_clean_jump") or x.get("replay_of_checker_path")) else 0 for x in r], "ko", ms=4, label="game engine clears")
        ax[1].set_xticks(range(len(r))); ax[1].set_xticklabels(names, rotation=90, fontsize=6); ax[1].legend(fontsize=7)
        ax[1].set_title("physics contract: checker vs the real engine")
    fig.tight_layout(); fig.savefig(FIG / "fig05_checker.png", dpi=150); plt.close(fig)
    note("fig05_checker", True)


def fig_cfg():
    p = ROOT / "reports/gates/P4/cfg_sweep.csv"
    if not p.exists():
        note("fig06_cfg", False)
        return
    rows = read_csv(p)
    fig, ax = plt.subplots(1, 3, figsize=(11, 3))
    w = [float(r["w"]) for r in rows]
    for a, k, t in zip(ax, ("c_score", "broken_rate", "copy_rate"), ("C-score vs w (gate >= 0.75)", "broken rate vs w", "copy rate vs w")):
        a.plot(w, [float(r[k]) for r in rows], "o-"); a.set_title(t); a.set_xlabel("CFG w"); a.grid(alpha=.3)
    ax[0].axhline(0.75, ls="--", c="k")
    fig.tight_layout(); fig.savefig(FIG / "fig06_cfg.png", dpi=150); plt.close(fig)
    note("fig06_cfg", True)


def copy_sheets():
    for src, dst in (("reports/gates/P3_cos_dense/sample_sheet_cosine.png", "fig07_uncond_samples.png"),
                     ("reports/gates/P4/sample_sheet_best_w.png", "fig08_cond_samples.png"), ("reports/gates/P4/contrast.png", "fig09_contrast.png"),
                     ("reports/gates/P3_cos_dense/nearest_train_cosine.png", "fig10_nearest_train.png"), ("reports/assets/tile_contact_sheet.png", "fig11_tiles.png")):
        p = ROOT / src
        if p.exists():
            shutil.copy(p, FIG / dst)
        note(dst, p.exists())
    for i, dst in ((1, "fig13_showcase_stairs.png"), (5, "fig14_showcase_platforms.png"), (7, "fig15_showcase_coins.png")):
        sp = ROOT / "app/showcase" / f"showcase_{i:02d}.png"
        if sp.exists():
            shutil.copy(sp, FIG / dst)
        note(dst, sp.exists())
    for p in sorted((ROOT / "reports/gates/P5").glob("render_*.png"))[:4]:
        shutil.copy(p, FIG / f"fig12_{p.stem}.png")
        note(p.stem, True)


if __name__ == "__main__":
    for f in (fig_pipeline, fig_training_curves, fig_p3_tradeoff, fig_val_buckets, fig_checker, fig_cfg, copy_sheets):
        try:
            f()
        except Exception as e:
            skipped.append(f"{f.__name__}: {e}")
    print("figures written:", ", ".join(done))
    if skipped:
        print("skipped:", "; ".join(skipped), file=sys.stderr)
