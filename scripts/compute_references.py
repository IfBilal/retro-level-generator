"""Measure the real-data reference values every metric is judged against (doc 22 §3, ledger F22; pre-registration).

Explain-me:
  What: on REAL windows only (no model exists yet) computes: floor-validity on real dev windows, broken structures in all
        real windows, novelty_ref and copy_rate_ref of dev (stride 8) vs train, tile-JS of the real dev set and of two
        degenerate generators (all-sky, uniform noise).  The G-P3 copy-rate threshold is copy_rate_ref + 0.05.
  Out: reports/references/REFERENCES.json (committed BEFORE any model sample exists, so the thresholds cannot be tuned).
  Uses only train/dev levels (never test).
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from text2mario.data.dataset import MarioWindows
from text2mario.eval import structure as S


def main():
    train, dev = MarioWindows("train"), MarioWindows("dev")
    tr = train.grids.numpy()
    dv_rows = dev.eval_rows(8)
    dv = dev.grids[dv_rows].numpy()
    dv_all = dev.grids.numpy()
    nov = S.novelty(dv, tr)
    nov_all = S.novelty(dv_all, tr)
    real_all = np.concatenate([tr, dv_all])
    rng = np.random.default_rng(0)
    sky = np.zeros_like(dv)
    noise = rng.integers(0, 13, dv.shape).astype(np.uint8)
    pipes_total = sum(S.broken_counts(g)[2] for g in real_all[::1])
    broken_total = sum(S.broken_counts(g)[0] + S.broken_counts(g)[1] for g in real_all)
    ref = {
        "n_dev_windows_stride8": len(dv), "n_train_windows": len(tr),
        "floor_valid_real_dev": S.floor_valid_rate(dv_all), "floorless_share_real_dev": S.floorless_share(dv_all),
        "broken_structures_in_real_train_dev": broken_total, "pipe_components_in_real_train_dev": pipes_total,
        "novelty_ref_median_stride8": nov["novelty_med"], "novelty_mean": nov["mean"], "novelty_p10": nov["p10"], "novelty_p90": nov["p90"],
        "novelty_ref_median_all_stride1": nov_all["novelty_med"],
        "copy_rate_ref": S.copy_rates(nov["nn"])["copy_rate"], "copy_rate_strict_ref": S.copy_rates(nov["nn"])["copy_rate_strict"],
        "exact_copy_rate_ref": S.copy_rates(nov["nn"])["exact_copy_rate"],
        "tile_js_real_dev_vs_train": S.tile_js(dv_all, tr), "tile_js_all_sky": S.tile_js(sky, tr), "tile_js_uniform_noise": S.tile_js(noise, tr),
    }
    ref["G-P3_copy_rate_threshold"] = round(ref["copy_rate_ref"] + 0.05, 4)
    out = ROOT / "reports/references/REFERENCES.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(ref, indent=2) + "\n")
    print(json.dumps(ref, indent=2))


if __name__ == "__main__":
    main()
