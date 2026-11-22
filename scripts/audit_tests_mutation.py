"""Audit the TESTS: break the code on purpose and require the tests to notice.  python scripts/audit_tests_mutation.py [--only ID,ID] [--list]

Explain-me:
  What: a hand-written mutation test.  Each entry replaces ONE decision in the new code (a comparison, a guard, a constant, a whole condition) with a plausible bug, runs the tests that are meant
        to protect that code (pytest -x), and records KILLED (a test failed: good) or SURVIVED (all tests passed: the tests do not protect that decision).  The file is restored after every
        mutation, and `git diff` is checked at the end.  Survivors are then either covered by a new test or explained as equivalent mutants.
  In/Out: -> reports/audit/test_mutation.json + test_mutation.md.  Previous: tests/.  Next: reports/audit/AUDIT_REPORT.md (section 'the tests were tested too').
  Try: python scripts/audit_tests_mutation.py --only A1,A2   (a few mutations, about 30 s each)
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FAST = ["tests/test_adherence_plan.py", "tests/test_adherence_properties.py", "tests/test_count_audit.py", "tests/test_normalizer_want.py",
        "tests/test_service_paths.py", "tests/test_full_level.py", "tests/test_level_schema.py", "tests/test_normalizer.py"]
UI = ["tests/test_app_ui.py"]
GRAPH = ["tests/test_cuda_graph.py"]
ADH = "src/text2mario/generate/adherence.py"
CNT = "src/text2mario/eval/count_audit.py"
FL = "src/text2mario/generate/full_level.py"
NRM = "src/text2mario/text/normalizer.py"
SVC = "src/text2mario/app/service.py"
GH = "src/text2mario/render/game_html.py"
GR = "src/text2mario/diffusion/graph.py"
APP = "app/streamlit_app.py"

# (id, file, old, new, tests, what the bug would be)
M = [
    ("A1", ADH, "starts = [0 if firsts[s] == 0 else firsts[s] * STRIDE + (WINDOW - STRIDE) for s in order]", "starts = [0 if firsts[s] == 0 else firsts[s] * STRIDE for s in order]", FAST, "scene regions start 8 columns too early"),
    ("A2", ADH, "    return lo + (hi - lo) // 2\n", "    return hi\n", FAST, "a vague quantity is planned at its top instead of its middle"),
    ("A3", ADH, "if hi - lo > 20 or _A[\"range_target\"] == \"low\":", "if hi - lo > 200 or _A[\"range_target\"] == \"low\":", FAST, "an open-ended range (many) is planned at 54 objects"),
    ("A4", ADH, "f |= set().union(*(wanted[j] for j in range(len(wanted)) if j != i)) - wanted[i]", "f |= set().union(*(wanted[j] for j in range(len(wanted)) if j != i))", FAST, "a topic asked for in this scene is also forbidden here"),
    ("A5", ADH, "if _A[\"exclusive_scenes\"] and len(norm.scenes) > 1:", "if False:", FAST, "'then' is not exclusive"),
    ("A6", ADH, "if asked in wanted[i] and other not in wanted[i]:", "if False:", FAST, "a stray staircase of the other direction is allowed"),
    ("A7", ADH, "\"implicit\": t not in sc.forbid}", "\"implicit\": False}", FAST, "implied prohibitions are reported as asked for"),
    ("A8", ADH, "    if bad:\n        topics =", "    if False:\n        topics =", FAST, "a self-contradicting prompt is not refused"),
    ("A9", ADH, "if len(norm.scenes) > windows:", "if len(norm.scenes) > windows + 5:", FAST, "more scenes than windows is not refused"),
    ("A10", ADH, "        if n > cap:\n", "        if n > cap * 100:\n", FAST, "impossible counts are not refused"),
    ("A11", ADH, "int(_A[\"capacity_per_16_cols\"].get(t, 99) * width / WINDOW)", "int(_A[\"capacity_per_16_cols\"].get(t, 99) * 48 / WINDOW)", FAST, "capacity ignores the level width"),
    ("A12", ADH, "cum = math.floor((i + 1) * tgt / m + 0.5) if k < last_k else tgt", "cum = math.floor((i + 1) * tgt / m) if k < last_k else tgt", FAST, "a single object lands at the end of the scene"),
    ("A13", ADH, "done = sum(1 for a in have[t] if a >= region_start)", "done = sum(1 for a in have[t] if True)", FAST, "objects of earlier scenes count towards this scene's quota"),
    ("A14", ADH, "new = max(0, min(cap, cum - done))", "new = max(0, min(cap, cum))", FAST, "already-placed objects are requested again (the original doubling bug)"),
    ("A15", ADH, "cap = max(int(_A[\"max_new_per_window\"].get(t, 1)), math.ceil(tgt / m))", "cap = int(_A[\"max_new_per_window\"].get(t, 1))", FAST, "eight coins in a one-window level become impossible"),
    ("A16", ADH, "base.insert(0, str(_A[\"default_floor\"]))", "pass", FAST, "windows lose their default floor phrase"),
    ("A17", ADH, "if t in scene.forbid or t not in scene.want:", "if t not in scene.want:", FAST, "a forbidden topic can still be requested in the caption"),
    ("A18", ADH, "c0 = 0 if k == 0 else STRIDE\n    a = anchors(win)", "c0 = 0\n    a = anchors(win)", FAST, "the known half is counted as new"),
    ("A19", ADH, "miss += abs(n_new - plan.get(t, 0))", "miss += max(0, n_new - plan.get(t, 0))", FAST, "a window with too FEW objects counts as exact"),
    ("A20", ADH, "        if t in forbid:\n            miss += n_new", "        if t in forbid:\n            miss += 0", FAST, "forbidden objects are not penalised"),
    ("A21", ADH, "\"met\": c[\"lo\"] <= len(pos) <= c[\"hi\"]", "\"met\": c[\"lo\"] <= len(pos)", FAST, "too many objects are accepted"),
    ("A22", ADH, "pos = [x for x in a[c[\"topic\"]] if lo_c <= x < hi_c]", "pos = [x for x in a[c[\"topic\"]] if lo_c <= x]", FAST, "objects right of a scene region are counted in it"),
    ("A23", ADH, "\"status\": \"met\" if not unmet else \"unmet\"", "\"status\": \"met\"", FAST, "unmet is never reported"),
    ("A24", ADH, "return owner(row[\"anchors\"][row[\"hi\"]])", "return owner(row[\"anchors\"][0])", FAST, "the wrong window is redrawn for a surplus"),
    ("A25", ADH, "min(n_windows - 1, (col - (WINDOW - STRIDE)) // STRIDE)", "min(n_windows - 1, col // STRIDE)", FAST, "column to window ownership is off by one window"),
    ("A26", ADH, "if (win_plans[k] or {}).get(row[\"topic\"], 0) > sum(", "if (win_plans[k] or {}).get(row[\"topic\"], 0) < sum(", FAST, "the plan-guided shortfall rule is inverted"),
    ("A27", ADH, "    return ks[-1]\n", "    return ks[0]\n", FAST, "the shortfall fallback picks the wrong window"),
    ("C1", CNT, "if height[c] < 3:", "if height[c] < 2:", FAST, "2-high stubs count as towers"),
    ("C2", CNT, "if c2 - c + 1 <= 2 and left", "if c2 - c + 1 <= 3 and left", FAST, "3-wide blocks count as towers"),
    ("C3", CNT, "left <= low - 2 and right <= low - 2", "left <= low - 1 and right <= low - 1", FAST, "a step of one counts as a tower edge"),
    ("C4", CNT, "BLOCKS = frozenset(T.SOLID) - frozenset(T.PIPE) - frozenset(T.CANNON)", "BLOCKS = frozenset(T.SOLID)", FAST, "pipes and cannons count as towers"),
    ("C5", CNT, "min(p, c - h)", "c - h", FAST, "the Wilson interval can exclude its own estimate"),
    ("C6", CNT, "k * stride + (window - stride)", "k * stride", FAST, "the audit's scene ownership differs from the generator's"),
    ("C7", CNT, "good = None not in firsts and seq == sorted(seq) and len(set(seq)) == len(seq)", "good = True", FAST, "ordering is never checked"),
    ("C8", CNT, "    if len(starts) != len(expect):", "    if False:", FAST, "a scene-count mismatch is not handled"),
    ("F1", FL, " and adh_draws < adh_draw_cap and not out_of_time)", " and not out_of_time)", FAST, "the draw cap does not stop whole-level resampling"),
    ("F2", FL, "            play_done = True ", "            play_done = False ", FAST, "counts stop being fixed after the route budget"),
    ("F3", FL, "\"attempts_total\": int(min(total, max_total_attempts))", "\"attempts_total\": int(total)", FAST, "attempts_total can exceed the ledger cap and invalidate the level"),
    ("F4", FL, "if miss == 0 or adh_draws >= adh_draw_cap or", "if miss == 0 or", FAST, "the draw cap does not stop window redraws"),
    ("F5", FL, "adh_verify = use_adh and verify and bool(adherence._A[\"verify\"])", "adh_verify = use_adh and bool(adherence._A[\"verify\"])", FAST, "verify=False still redraws windows"),
    ("F6", FL, "    if adherence.enabled():\n        adherence.check_feasible(norm, width)", "    if False:\n        adherence.check_feasible(norm, width)", FAST, "impossible requests are not refused in generate()"),
    ("F7", FL, "before = _assemble(windows[:k], STRIDE * (k - 1) + WINDOW) if k > 0 else None", "before = None", FAST, "the planner forgets what is already in the level"),
    ("F8", FL, "if best is None or miss < best_miss:", "if best is None or miss > best_miss:", FAST, "the WORST candidate window is kept"),
    ("F9", FL, "j = int(np.argmin(misses))", "j = 0", FAST, "the first candidate is kept instead of the best"),
    ("N1", NRM, "sc.want[t] = list(BUCKET_RANGE[bucket(v)]) if vague.get(t) else [v, v]", "sc.want[t] = [v, v]", FAST, "vague words become exact numbers"),
    ("N2", NRM, "            sc.caption = NEUTRAL_CAPTION", "            sc.caption = \"\"", FAST, "a bare prohibition has no scene again"),
    ("N3", NRM, "sc.warnings.append(f\"CONTRADICTION:{t}\")", "pass", FAST, "contradictions are not detected"),
    ("N4", NRM, "and val > 2\n", "and val > 5\n", FAST, "'a few' keeps no range"),
    ("S1", SVC, "re.search(r\"(?<![\\w])-\\s?\\d\", t)", "re.search(r\"(?<![\\w])-\\s?\\d{9}\", t)", FAST, "negative numbers are accepted"),
    ("S2", SVC, "guarded = self.profile == \"cloud\" or self.device != \"cuda\"", "guarded = self.profile == \"cloud\"", FAST, "a CPU server has no time guard"),
    ("G1", GH, ".replace(\"<!--\", \"<\\\\!--\")", "", FAST, "'<!--' can break the game page"),
    ("G2", GH, ".replace(\"</\", \"<\\\\/\")", "", FAST, "'</script>' can break out of the page"),
    ("H1", "src/text2mario/render/schema.py", "if a.get(\"status\") != \"adherence\" and a.get(\"attempt\", 1) > 5:", "if a.get(\"attempt\", 1) > 5:", FAST, "count resamples are held to the route limit of 5 per window"),
    ("H2", "src/text2mario/render/schema.py", "if adh.get(\"attempts\", 0) > ADHERENCE_MAX_ATTEMPTS:", "if False:", FAST, "the adherence budget is not enforced by the schema"),
    ("H3", "src/text2mario/render/schema.py", "if adh and adh.get(\"judged_on\") != \"raw\":", "if False:", FAST, "adherence could be judged on the repaired display grid"),
    ("H4", SVC, "    if len(t) > MAX_PROMPT_CHARS:", "    if len(t) > MAX_PROMPT_CHARS + 1000:", FAST, "over-long prompts are accepted"),
    ("H5", SVC, "if any(ord(c) < 32 and c not in \"\\n\\t\" for c in t):", "if False:", FAST, "control characters are accepted"),
    ("H6", "src/text2mario/generate/sanitizer.py", "                    g[r, c] = T.SKY", "                    g[r, c] = T.GROUND", FAST, "the display repair ADDS ground instead of deleting broken parts"),
    ("H7", FL, "forb_k = forb_sets[scene_of[k]] if use_adh else None", "forb_k = forb_sets[0] if use_adh else None", FAST, "a later scene uses the first scene's prohibitions"),
    ("H8", FL, "adherence.has_constraints(scene_k, forb_k))", "adherence.has_constraints(scene_k))", FAST, "scenes whose only constraint is an implied prohibition are not checked"),
    ("H9", FL, "            adh_draws += n - k  ", "            adh_draws += 0  ", FAST, "whole-level resamples do not count towards the draw cap"),
    ("H10", SVC, "    if code.startswith(\"CLAMPED:\"):", "    if False:", FAST, "clamp warnings are shown as raw codes"),
    ("H11", GH, ".replace(chr(0x2028), \"\\\\u2028\")", "", FAST, "a U+2028 in a prompt breaks the game script"),
    ("P1", GR, "enabled=ac, cache_enabled=False)", "enabled=ac, cache_enabled=True)", GRAPH, "the CUDA graph reads freed autocast weights"),
    ("P2", GR, "        with self._lock:\n", "        if True:\n", GRAPH, "two sessions corrupt each other's graph buffers"),
    ("U1", APP, "    st.session_state[\"seed\"] = random.randint(0, 10**9)", "    pass", UI, "Dice does nothing"),
    ("U2", APP, "    if dropped:\n", "    if False:\n", UI, "a redeploy into a running process breaks the app again"),
    ("U3", APP, "                st.session_state.result_prompt = st.session_state.prompt\n", "                pass\n", UI, "an old level is not labelled as belonging to an earlier sentence"),
    ("U4", APP, "    if a[\"status\"] == \"met\":", "    if True:", UI, "unmet counts are shown as a success"),
]


# mutants that cannot change behaviour, with the proof (reported separately from survivors)
EQUIVALENT = {"H11": "json.dumps(..., ensure_ascii=True) already writes U+2028 as the ASCII text \u2028, so this replace never has a character to replace "
                     "(tests/test_level_schema.py::test_json_is_ascii_so_line_separators_never_reach_the_script proves it)"}


def run(tests):
    env = {**os.environ, "PYTHONPATH": f"{ROOT / 'src'}{os.pathsep}{ROOT / 'tests'}", "PYTHONDONTWRITEBYTECODE": "1"}
    try:
        p = subprocess.run([sys.executable, "-B", "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider", "--hypothesis-seed=0", *tests], cwd=ROOT, env=env,
                           capture_output=True, text=True, timeout=240, check=False)
    except subprocess.TimeoutExpired:                      # a mutation that creates an infinite loop is caught by the timeout (a hang is a failure)
        kill = ("Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*pytest*--hypothesis-seed=0*' } | "
                "ForEach-Object { Stop-Process -Id $_.ProcessId -Force }")
        subprocess.run(["powershell", "-NoProfile", "-Command", kill], capture_output=True, text=True, check=False)
        return 1, "TIMEOUT: the mutated code hangs (an infinite loop), the test run was killed after 240 s"
    return p.returncode, (p.stdout.strip().splitlines() or [""])[-1][:150]


BACKUP = ROOT / "reports/audit/_mutation_backup"


def heal():
    """If an earlier run was killed in the middle of a mutation, put the original files back before doing anything else."""
    if BACKUP.exists():
        for f in BACKUP.rglob("*"):
            if f.is_file():
                (ROOT / f.relative_to(BACKUP)).write_bytes(f.read_bytes())
                print("restored", f.relative_to(BACKUP))
        for f in sorted(BACKUP.rglob("*"), reverse=True):
            f.unlink() if f.is_file() else f.rmdir()
        BACKUP.rmdir()


def main():
    heal()
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    todo = [m for m in M if not a.only or m[0] in a.only.split(",")]
    if a.list:
        for m in todo:
            print(m[0], m[1], "-", m[5])
        return 0
    for name, tests in (("fast", FAST), ("ui", UI), ("graph", GRAPH)):          # a failing BASELINE would make every mutant look killed: check it first
        if any(m[4] is tests for m in todo):
            code, last = run(tests)
            if code != 0:
                print(f"baseline group '{name}' FAILS on the unmutated code ({last}); fix the tests before auditing them")
                return 2
            print(f"baseline group '{name}' passes: {last}", flush=True)
    results = []
    for mid, rel, old, new, tests, bug in todo:
        path = ROOT / rel
        original = path.read_bytes()
        (BACKUP / rel).parent.mkdir(parents=True, exist_ok=True)
        (BACKUP / rel).write_bytes(original)                       # survives a forced kill; heal() restores it on the next start
        crlf = b"\r\n" in original
        text = original.decode("utf-8").replace("\r\n", "\n")          # match patterns regardless of the file's line endings
        if text.count(old) != 1:
            results.append({"id": mid, "file": rel, "bug": bug, "outcome": f"BAD MUTATION (pattern found {text.count(old)} times)"})
            print(mid, "BAD MUTATION", flush=True)
            continue
        t0 = time.time()
        try:
            mutated = text.replace(old, new)
            path.write_bytes((mutated.replace("\n", "\r\n") if crlf else mutated).encode("utf-8"))
            code, last = run(tests)
        finally:
            path.write_bytes(original)
            (BACKUP / rel).unlink()
        outcome = "KILLED" if code == 1 else "KILLED (crash)" if code not in (0, 5) else "SURVIVED" if code == 0 else "NO TESTS"
        results.append({"id": mid, "file": rel, "bug": bug, "outcome": outcome, "seconds": round(time.time() - t0, 1), "pytest": last})
        print(f"{mid:4s} {outcome:15s} {bug}  [{last}]", flush=True)
    heal()
    diff = subprocess.run(["git", "diff", "--stat", "--", "src", "app"], cwd=ROOT, capture_output=True, text=True, check=False).stdout.strip()
    killed = sum(1 for r in results if r["outcome"].startswith("KILLED"))
    surv = [r for r in results if (r["outcome"] in ("SURVIVED", "NO TESTS") or r["outcome"].startswith("BAD")) and r["id"] not in EQUIVALENT]
    equiv = [r["id"] for r in results if r["outcome"] == "SURVIVED" and r["id"] in EQUIVALENT]
    out = {"mutations": len(results), "killed": killed, "survived": [r["id"] for r in surv], "equivalent_mutants": {i: EQUIVALENT[i] for i in equiv}, "source_untouched_after_run": diff == "", "results": results}
    (ROOT / "reports/audit/test_mutation.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    md = ["# Mutation audit of the tests", "",
          (f"{killed}/{len(results)} injected bugs were caught by the tests; survivors: {', '.join(r['id'] for r in surv) or 'none'}; proven-equivalent mutants: {', '.join(equiv) or 'none'}.  "
           f"Source tree identical afterwards: {diff == ''}."), "",
          "| id | file | the bug that was injected | outcome |", "|---|---|---|---|"]
    md += [f"| {r['id']} | `{r['file'].split('/')[-1]}` | {r['bug']} | {r['outcome']} |" for r in results]
    (ROOT / "reports/audit/test_mutation.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\n{killed}/{len(results)} killed; survivors: {[r['id'] for r in surv]}; source untouched: {diff == ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
