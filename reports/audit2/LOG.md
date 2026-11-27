# audit2 log (what was tried, what failed, what was reverted)
- 27 Nov: `that_assets/` is not on this machine; battery re-run on current main instead. Most of U1-U14 were already closed by audit 1; reopened items: U3 (references counted as objects), U5 (flat/empty ran unconditioned), U7 (positions dropped silently), H8 (display != raw).
- Wrote a hand-edit of the normaliser through shell heredocs; `\b` turned into a backspace character in two regexes (found by a failing test, fixed). Lesson: edit regexes with the Edit tool.
- The old test `test_verify_loop_respects_budgets` used the junk prompt "x"; now refused by design, test changed to a real prompt.
- No subagents were spawned (cost); the adversary round is run by a separate fresh-seed audit instead.
- Not run: H5 retraining (see MODEL_RCA.md).
