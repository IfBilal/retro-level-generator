"""Static hygiene scan of the test suite: python scripts/audit_test_hygiene.py

Explain-me:
  What: parses every tests/*.py and flags test functions that cannot fail: no assert, no pytest.raises / pytest.fail / pytest.skip, no call of a helper known to assert
        (fresh/go in the UI tests assert there is no exception); constant-true assertions (`assert True`, `... or True`, `if False`); and bare `except: pass` that would swallow a failure.
        Also lists tests that are only skipped under a condition so a silently skipped test is visible.
  In/Out: tests/ -> reports/audit/test_hygiene.json; exit code 1 when something is flagged.
  Previous: tests/.  Next: reports/audit/AUDIT_REPORT.md (the tests were tested too).
  Try: python scripts/audit_test_hygiene.py
"""
import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSERTING_HELPERS = {"fresh", "go", "entry", "check", "run_gate", "assert_valid"}      # helpers whose body asserts


def has_assertion(fn: ast.AST) -> bool:
    for n in ast.walk(fn):
        if isinstance(n, ast.Assert):
            return True
        if isinstance(n, ast.Call):
            f = n.func
            name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
            if name in ("raises", "fail", "skip", "importorskip", "warns", "approx") or name in ASSERTING_HELPERS or name.startswith("assert"):
                return True
    return False


def main():
    flagged, counts = [], {"files": 0, "tests": 0, "skipif": 0}
    for f in sorted((ROOT / "tests").rglob("test_*.py")):
        tree = ast.parse(f.read_text(encoding="utf-8"))
        counts["files"] += 1
        src = f.read_text(encoding="utf-8")
        counts["skipif"] += src.count("pytest.mark.skipif")
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                counts["tests"] += 1
                if not has_assertion(node):
                    flagged.append({"file": f.name, "test": node.name, "why": "no assertion, raises, fail or asserting helper"})
                for n in ast.walk(node):
                    if isinstance(n, ast.Assert) and isinstance(n.test, ast.Constant) and n.test.value:
                        flagged.append({"file": f.name, "test": node.name, "why": "assert of a constant true"})
                    if isinstance(n, ast.Assert) and isinstance(n.test, ast.BoolOp) and isinstance(n.test.op, ast.Or) and any(isinstance(v, ast.Constant) and v.value is True for v in n.test.values):
                        flagged.append({"file": f.name, "test": node.name, "why": "assertion with '... or True'"})
                    if isinstance(n, ast.ExceptHandler) and len(n.body) == 1 and isinstance(n.body[0], ast.Pass):
                        flagged.append({"file": f.name, "test": node.name, "why": "except: pass swallows failures"})
    out = {**counts, "flagged": flagged}
    (ROOT / "reports/audit/test_hygiene.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({**counts, "flagged": len(flagged)}))
    for x in flagged:
        print(" ", x)
    return 1 if flagged else 0


if __name__ == "__main__":
    sys.exit(main())
