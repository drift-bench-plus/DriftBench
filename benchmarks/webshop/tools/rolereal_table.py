#!/usr/bin/env python
"""Assemble the role-realism table: 3 benchmarks x 2 metrics x 3 seeds.

Reads judgments_v2/judgments.json under experiments/rolereal/<cell>/ and prints
per-seed rows plus mean +-std per benchmark, for both judges, their agreement,
and the strict both-judges-correct score. Dummy baselines: identification 20%
(5-way), consistency 50% (balanced two-way).
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1] / "pipeline" / "src"))
from intent_graph.runtime.judges import summarize_judgments  # noqa: E402

CELLS = {
    "WebShop": ["ws_A0_r1", "ws_A0_r2", "ws_A0_r3"],
    "Retail": ["rt_A0_s1", "rt_A0_s2", "rt_A0_s3"],
    "Airline": ["ar_A0_s1", "ar_A0_s2", "ar_A0_s3"],
}


def one(cell: str) -> dict | None:
    f = ROOT / "experiments/rolereal" / cell / "judgments_v2/judgments.json"
    if not f.exists():
        return None
    J = json.loads(f.read_text())
    s = summarize_judgments(J)
    ident, cons = J["identification"], J["consistency"]
    r = {
        "ident_a": s["identification"]["judge_a"],
        "ident_b": s["identification"]["judge_b"],
        "ident_agree": s["identification"]["agreement"],
        "ident_strict": sum(1 for j in ident
                            if j["judge_a"]["guess"] == j["judge_b"]["guess"]
                            == j["persona"]) / len(ident),
        "ident_n": len(ident),
        "cons_a": s["consistency"]["judge_a"],
        "cons_b": s["consistency"]["judge_b"],
        "cons_valid": ",".join(s["consistency"]["valid_judges"]),
        "cons_agree": sum(1 for j in cons
                          if j["judge_a"]["guess_same"] == j["judge_b"]["guess_same"])
                      / len(cons),
        "cons_strict": sum(1 for j in cons
                           if j["judge_a"]["guess_same"] == j["judge_b"]["guess_same"]
                           == j["truth_same"]) / len(cons),
        "cons_n": len(cons),
        "by_persona": s["identification"].get("by_persona", {}),
    }
    return r


def fmt(vals):
    if not vals:
        return "—"
    if len(vals) == 1:
        return f"{100 * vals[0]:.1f}"
    return f"{100 * st.mean(vals):.1f} ±{100 * st.stdev(vals):.1f}"


def main() -> int:
    print("| Bench | seed | Ident judge_a | Ident judge_b | Ident strict | agree "
          "| Cons judge_a | Cons judge_b | Cons strict | agree | n_id/n_pair |")
    print("|---" * 11 + "|")
    summary = {}
    for bench, cells in CELLS.items():
        rows = [(c, one(c)) for c in cells]
        for c, r in rows:
            if r is None:
                print(f"| {bench} | {c} | (pending) |" + " |" * 9)
                continue
            print(f"| {bench} | {c} | {100*r['ident_a']:.1f} | {100*r['ident_b']:.1f} "
                  f"| {100*r['ident_strict']:.1f} | {100*r['ident_agree']:.1f} "
                  f"| {100*r['cons_a']:.1f} | {100*r['cons_b']:.1f} "
                  f"| {100*r['cons_strict']:.1f} | {100*r['cons_agree']:.1f} "
                  f"| {r['ident_n']}/{r['cons_n']} |")
        done = [r for _, r in rows if r]
        if done:
            summary[bench] = {k: fmt([r[k] for r in done])
                              for k in ("ident_a", "ident_b", "ident_strict",
                                        "cons_a", "cons_b", "cons_strict")}
    print("\n== mean ±std over seeds (dummy: ident 20.0, cons 50.0) ==")
    print("| Bench | Ident judge_a | Ident judge_b | Ident strict | Cons judge_a "
          "| Cons judge_b | Cons strict |")
    print("|---" * 7 + "|")
    for bench, s in summary.items():
        print(f"| {bench} | {s['ident_a']} | {s['ident_b']} | {s['ident_strict']} "
              f"| {s['cons_a']} | {s['cons_b']} | {s['cons_strict']} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
