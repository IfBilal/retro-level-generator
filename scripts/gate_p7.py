"""Gate G-P7: app latency and offline behaviour, MEASURED first, then pinned (doc 24 §G-P7, CLAUDE.md rule 14).

Explain-me:
  What: reads reports/app/latency.json (scripts/bench_app.py), checks the provisional targets (112 columns in fast mode <= 10 s on the GPU,
        cold start < 20 s, CPU fast mode 112 columns < 45 s) and verifies that the game page references no external resource.  It also writes
        PINNED regression thresholds = measured median x 1.25.  A provisional target that is missed is reported as a FAIL, not adjusted.
  In/Out: reports/app/latency.json -> reports/gates/P7/RESULT.json.  Previous: bench_app.py.  Next: the report quotes it.
  Try: python scripts/bench_app.py --ckpt app/model --skip-full --cpu && python scripts/gate_p7.py
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from text2mario.utils.gates import write_result


def main():
    lat = json.loads((ROOT / "reports/app/latency.json").read_text())
    checks = [{"id": "P7.01", "name": "GPU fast mode, 112 columns, median s (provisional target)", "value": lat.get("cuda_112_fast_p50"), "op": "<=", "threshold": 10.0,
               "pass": lat.get("cuda_112_fast_p50", 1e9) <= 10.0},
              {"id": "P7.02", "name": "cold start s", "value": lat.get("cold_start_s"), "op": "<", "threshold": 20.0, "pass": lat.get("cold_start_s", 1e9) < 20.0},
              {"id": "P7.03", "name": "CPU fast mode, 112 columns, median s", "value": lat.get("cpu_112_p50"), "op": "<", "threshold": 45.0, "pass": lat.get("cpu_112_p50", 1e9) < 45.0}]
    try:
        from text2mario.render.game_html import build_game_html
        lvl = json.loads(next((ROOT / "app/showcase").glob("showcase_00.level.json")).read_text())
        html = build_game_html(lvl)
        ext = re.findall(r'(?:src|href)=["\']https?://', html) + re.findall(r"fetch\(\s*[\"']https?://", html)
        checks.append({"id": "P7.04", "name": "game page references no external resource (offline)", "value": len(ext), "op": "==", "threshold": 0, "pass": not ext})
    except Exception as e:  # missing showcase or assets: report it, do not hide it
        checks.append({"id": "P7.04", "name": "offline check", "value": str(e), "pass": False})
    pinned = {k: round(v * 1.25, 1) for k, v in lat.items() if isinstance(v, (int, float)) and k.endswith(("p50", "cold_start_s"))}
    write_result("P7", all(c["pass"] for c in checks), {"latency": lat}, checks, pinned_regression_thresholds_median_x1_25=pinned,
                 note="provisional targets are reported as measured; pinned values are for future regression checks only")
    for c in checks:
        print(c["id"], c["value"], "PASS" if c["pass"] else "FAIL")


if __name__ == "__main__":
    main()
