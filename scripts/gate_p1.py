"""Gate G-P1 (data): facts re-verified, split leakage-safe, renders correct, round trips exact (docs 09, 10, 24).

Explain-me:
  What: re-reads the audit result, checks the locked split (disjoint levels, families in train, leakage <= 3 %),
        renders 10 random training windows (floor must be at the bottom, pipes upright, '<' left of '>'),
        and checks grid -> one-hot -> argmax is the identity.
  In/Out: -> reports/gates/P1/RESULT.json + 10 PNG renders + a contact sheet.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from text2mario.data import splits
from text2mario.data import tiles as T
from text2mario.data.dataset import MarioWindows
from text2mario.render.renderer import grid_to_image, grid_to_tensor, sheet, tensor_to_grid
from text2mario.utils.gates import gate_dir, write_result

checks = []


def check(name, ok, detail=""):
    checks.append({"name": name, "pass": bool(ok), "detail": detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")


def main():
    audit = json.loads((ROOT / "reports/eda/AUDIT.json").read_text())
    check("audit: all hard facts F1-F17/A-series pass", audit["passed"], f"({len(audit['checks'])} checks)")
    S = splits.load(verify=True)
    check("split matches LOCK", True)
    tr, dv, te = (set(S["splits"][k]) for k in ("train", "dev", "test"))
    check("no level in two splits; union = 42 levels", not (tr & dv or tr & te or dv & te) and len(tr | dv | te) == 42)
    check("family rule: every family member is in train", all(set(f["levels"]) <= tr for f in S["families"]))
    lk = S["leakage"]
    check("test windows within Hamming<=2 of train <= 3 %", lk["test"]["le2"] <= 0.03, f"= {lk['test']['le2']:.2%}")
    check("dev windows within Hamming<=2 of train <= 3 %", lk["dev"]["le2"] <= 0.03, f"= {lk['dev']['le2']:.2%}")
    pytest = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_splits.py", "tests/test_detector.py", "tests/test_data.py"],
                            cwd=ROOT, capture_output=True, text=True)
    check("pytest: splits, detector, data tests", pytest.returncode == 0, pytest.stdout.strip().splitlines()[-1])

    d = MarioWindows("train")
    rng = np.random.default_rng(1)
    rows = torch.as_tensor(rng.choice(len(d), 10, replace=False))
    out = gate_dir("P1")
    grids = d.grids[rows].cpu().numpy()
    for k, g in enumerate(grids):
        grid_to_image(g, 16).save(out / f"render_{k:02d}.png")
    sheet(list(grids), cols=5, tile_px=10).save(out / "contact_sheet.png")
    solid_bottom = np.mean([np.isin(g[15], list(T.SOLID)).mean() for g in grids])
    solid_top = np.mean([np.isin(g[0], list(T.SOLID)).mean() for g in grids])
    check("orientation: floor at the bottom, sky at the top", solid_bottom > solid_top, f"(bottom solid {solid_bottom:.2f} > top {solid_top:.2f})")
    ok_lip = all(not ((g[:, :-1] == T.PIPE_TR) & (g[:, 1:] == T.PIPE_TL)).any() for g in grids)
    check("pipe tops: '<' is left of '>' (never '>' then '<')", ok_lip)
    x = d.onehot(rows)
    check("round trip grid -> one-hot -> argmax is the identity", (tensor_to_grid(x) == grids).all())
    all_rows = torch.arange(len(d))
    check("round trip on all train windows", (tensor_to_grid(d.onehot(all_rows)) == d.grids.cpu().numpy()).all())
    check("x0 values are exactly {-1,+1}; 12 channels at -1 per cell", set(x.unique().tolist()) == {-1.0, 1.0} and bool((x.sum(1) == -11).all()))
    check("helper grid_to_tensor agrees with dataset.onehot", torch.equal(grid_to_tensor(grids), x))
    passed = all(c["pass"] for c in checks)
    write_result("P1", passed, {"test_le2": lk["test"]["le2"], "dev_le2": lk["dev"]["le2"], "train_windows": S["window_counts"]["train"]}, checks)
    print("G-P1", "PASS" if passed else "FAIL")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
