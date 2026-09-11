"""Validity gate: prove an arm on its first chunk of episodes before the rest spend money.

The discipline this encodes (ruling 2026-08-10): validate on a short snap of data first.
Smoke tests of 3-6 episodes caught most of campaign 1's bugs, but they were manual and I
skipped them once — so this runs automatically alongside the campaign. Each check is a
tripwire from a REAL bug we shipped: if one fires, the gate writes the HALT file and the
campaign stops instead of burning a night on poisoned numbers.

    python tools/validity_gate.py --out experiments/runs_main2 [--min 60] [--halt]
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import sys
import time
from pathlib import Path

CHECKED_MARKER = ".validity_checked"
ALLOW_SHIFTS = False


def _load(out_dir: Path, arm: str) -> list[dict]:
    out = []
    for f in glob.glob(str(out_dir / arm / "*.json")):
        try:
            out.append(json.loads(Path(f).read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            pass
    return out


def gate_arm(arm: str, ts: list[dict]) -> list[str]:
    """Hard failures only. Every rule is a bug we actually shipped, cited inline."""
    fails: list[str] = []
    n = len(ts)

    # bug #7: a config failure once wrote 500 ERROR trajectories that looked like results
    errors = [t for t in ts if t.get("outcome") == "ERROR"]
    if len(errors) > 0.1 * n:
        fails.append(f"{arm}: {len(errors)}/{n} ERROR outcomes "
                     f"(first: {str(errors[0].get('error'))[:90]})")

    # bug #1: axis B leaked into an axis-A campaign (p_shift 0.25 -> 345/400 shifted).
    # Only meaningful when shifts are unexpected: pass --allow-shifts for axis-B runs.
    if not ALLOW_SHIFTS:
        shifted = sum(1 for t in ts if any(r.get("shift") for r in t["turns"]))
        if shifted:
            fails.append(f"{arm}: {shifted}/{n} episodes contain a SHIFT in an axis-A run")

    # bug #8: exec races were recorded as legitimate outcomes
    exec_failed = sum(1 for t in ts for note in t.get("experiment", {}).get("agent_notes", ())
                      if "exec_failed" in note)
    if exec_failed:
        fails.append(f"{arm}: {exec_failed} literal_exec_failed notes (thread race?)")

    per_ep_props = (sum(1 for t in ts for r in t["turns"]
                        if (r.get("action") or {}).get("kind") == "PROPOSE") / n) if n else 0

    if arm == "LIT":
        # bug #2: the floor must be honest. Under the 2026-08-10 semantics, falsify no
        # longer implies an unsatisfiable literal (irrelevant_information narrows WITHIN
        # the truth; a detour substitution lands elsewhere; a dropped doubted premise
        # widens) -- so the leak test is scoped to DEAD-END literals (lit_card == 0),
        # where a literal actor genuinely cannot win without reading hidden material.
        dead_s = dead_n = 0
        for t in ts:
            sig = (t.get("header") or {}).get("signature") or {}
            if (t.get("experiment", {}).get("governing_kind") == "falsify"
                    and sig.get("lit_card") == 0):
                dead_n += 1
                dead_s += t["outcome"] == "SUCCESS"
        if dead_n >= 5 and dead_s > 0:
            fails.append(f"{arm}: {dead_s}/{dead_n} successes on dead-end falsify samples "
                         f"(lit_card=0) -- a literal floor cannot win an unsatisfiable "
                         f"premise; it is reading something it should not")
        by = collections.defaultdict(lambda: [0, 0])
        for t in ts:
            k = t.get("experiment", {}).get("governing_kind")
            by[k][1] += 1
            by[k][0] += t["outcome"] == "SUCCESS"
        n_s, n_n = by.get("noise", (0, 0))
        w_s, w_n = by.get("withhold", (0, 0))
        if n_n >= 10 and w_n >= 10 and (n_s / n_n) <= (w_s / w_n):
            fails.append(f"{arm}: noise ({n_s}/{n_n}) not above withhold ({w_s}/{w_n}) -- "
                         f"literal execution is likely failing")
    else:
        # bug #3: arms that never reach the buy step measure nothing
        if n >= 30 and per_ep_props < 0.5:
            fails.append(f"{arm}: {per_ep_props:.2f} proposals/episode -- the arm is not "
                         f"reaching the buy step (no stopping rule firing?)")
        # bug #9: buy-shaped Operations swallowed by the search box. After the fix these are
        # rewritten to PROPOSE with via_operation=True; a raw ACT that still LOOKS like a buy
        # means the rewrite is not firing.
        import re as _re
        swallowed = sum(1 for t in ts for r in t["turns"]
                        if (r.get("action") or {}).get("kind") == "ACT"
                        and _re.match(r"\s*(buy|purchase)\b",
                                      str((r.get("action") or {}).get("command") or ""), _re.I))
        if swallowed:
            fails.append(f"{arm}: {swallowed} buy-shaped ACT turns swallowed by the search box")

        # bug (format): chronic malformed output means the model is not following the format
        malformed = sum(1 for t in ts for r in t["turns"]
                        if (r.get("action") or {}).get("kind") == "MALFORMED")
        turns = sum(len(t["turns"]) for t in ts) or 1
        if malformed / turns > 0.10:
            fails.append(f"{arm}: {malformed}/{turns} malformed turns (>10%)")
        # bug #6: the broken-record loop -- identical consecutive user replies.
        # The mute-user floor (L0) answers every ask with the same placeholder BY
        # DESIGN, so the placeholder is exempt; the check still guards real dialogue.
        MUTE = "(the user does not respond)"
        loops = 0
        for t in ts:
            replies = [r.get("reply") for r in t["turns"]
                       if r.get("reply") and r.get("reply") != MUTE]
            loops += any(a == b for a, b in zip(replies, replies[1:], strict=False))
        if n >= 30 and loops > 0.3 * n:
            fails.append(f"{arm}: {loops}/{n} episodes with identical consecutive user "
                         f"replies (broken-record regression)")
    return fails


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments/runs_main2")
    ap.add_argument("--min", type=int, default=60, help="episodes before an arm is judged")
    ap.add_argument("--halt", action="store_true", help="write HALT on hard failure")
    ap.add_argument("--allow-shifts", action="store_true",
                    help="axis-B run: intent shifts are expected, skip the leak tripwire")
    args = ap.parse_args()
    global ALLOW_SHIFTS
    ALLOW_SHIFTS = args.allow_shifts
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = Path(__file__).resolve().parents[1] / out_dir

    any_fail = False
    for arm_dir in sorted(out_dir.iterdir()) if out_dir.exists() else []:
        if not arm_dir.is_dir():
            continue
        arm = arm_dir.name
        marker = arm_dir / CHECKED_MARKER
        if marker.exists():
            continue
        ts = _load(out_dir, arm)
        if len(ts) < args.min:
            print(f"  {arm}: {len(ts)} episodes, waiting for {args.min}")
            continue
        fails = gate_arm(arm, ts)
        if fails:
            any_fail = True
            for f in fails:
                print(f"GATE FAIL  {f}")
            if args.halt:
                (out_dir / "HALT").write_text(json.dumps(
                    {"reason": "validity gate failed", "fails": fails,
                     "at": int(time.time())}, indent=1), encoding="utf-8")
                print(f"HALT written to {out_dir / 'HALT'}")
        else:
            print(f"GATE PASS  {arm} on {len(ts)} episodes")
            marker.write_text(str(int(time.time())), encoding="utf-8")
    return 1 if any_fail else 0


if __name__ == "__main__":
    sys.exit(main())
