"""Compare the model with the non-learning baselines on the DEV prompts: python scripts/compare_baselines.py

Explain-me:
  What: scores the constant generators and the retrieval baseline on P-dev (train windows only as the library) and prints them next to the model's dev numbers at
        the frozen guidance (reports/dev_eval/C-main_0018000_p_dev_w5.json is w = 5; the w* = 7.5 row comes from the P4 gate).  Dev prompts only: the test prompts are
        used once by scripts/eval_test.py.
  In/Out: -> reports/dev_eval/baselines.json.  Previous: eval/baselines.py.  Next: report section 5.2.  Try: python scripts/compare_baselines.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from text2mario.utils.env import bootstrap_env

bootstrap_env()
from text2mario.data.dataset import MarioWindows
from text2mario.eval.baselines import evaluate_baselines
from text2mario.eval.devloop import load_prompts


def main():
    prompts = [r["prompt"] for r in load_prompts("p_dev")]
    tr = MarioWindows("train")
    base = evaluate_baselines(prompts, tr.captions, tr.grids.numpy())
    model = {}
    p4 = ROOT / "reports/gates/P4/RESULT.json"
    if p4.exists():
        d = json.loads(p4.read_text())["diagnostics"]
        model = {"c_score": d["at_w_star"]["c_score"], "exact_match": d["at_w_star"]["exact_match"], "broken_rate": d["at_w_star"]["broken_rate"],
                 "floor_valid": d["at_w_star"]["floor_valid"], "copy_rate": d["at_w_star"]["copy_rate"], "novelty_med": d["at_w_star"]["novelty_med"],
                 "w": d["best_w"]}
    out = {"n_prompts": len(prompts), "baselines": base, "model_at_w_star": model}
    (ROOT / "reports/dev_eval/baselines.json").write_text(json.dumps(out, indent=1))
    for k, v in {**base, "MODEL (dev, w*)": model}.items():
        print(f"{k:24s} " + "  ".join(f"{m}={v[m]:.3f}" for m in ("c_score", "exact_match", "broken_rate", "copy_rate", "novelty_med") if m in v))


if __name__ == "__main__":
    main()
