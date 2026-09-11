#!/usr/bin/env python
"""GRIP v3 report — Score-based Grounding (ruling 2026-08-22).

Rulings implemented here:
  * "the score will be the success from now on": G's headline is SCORE, the WebShop
    partial reward of the episode's FINAL adjudicated submission (accepted = 1.0).
    Binary success is not reported.
  * "earned and inferred should be calculated from the score": they are a score-MASS
    decomposition over hidden-intent episodes --
        earned   = mean(score x 1[episode recovered >=1 hidden slot by asking])
        inferred = mean(score x 1[it recovered none])
    so earned + inferred = mean score over hidden episodes. Same for PostShift:
    mean SCORE over episodes whose goal actually moved.
  * "keep double digits": fractions carry 4 decimals internally; the table prints x100
    with two real decimals -- computed, never padded.

Reads each episode once; reuses grip.score_v2 for the mechanical extraction (aim,
recovery, staleness, reaction) and adds the score channel.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1] / "pipeline" / "src"))

from intent_graph.runtime import grip  # noqa: E402

UND = "undefined"


def final_score(rec: dict) -> float:
    """WebShop reward of the last adjudicated proposal (accepted = 1.0; none = 0)."""
    last = 0.0
    for t in rec.get("turns") or []:
        acc = t.get("acceptance") if isinstance(t, dict) else None
        if isinstance(acc, dict):
            rw = acc.get("reward")
            if isinstance(rw, (int, float)):
                last = float(rw)
            elif acc.get("ok"):
                last = 1.0
            else:
                m = re.search(r"reward=([0-9.]+)", str(acc.get("reason") or ""))
                last = float(m.group(1)) if m else 0.0
    return last


def collect(run_dir: Path) -> list[dict]:
    rows = []
    for f in sorted(run_dir.glob("*.json")):
        try:
            rec = json.loads(f.read_text())
        except Exception:
            continue
        if rec.get("error") or not (rec.get("turns") or []):
            continue                       # husks are refilled, never scored
        row = grip.score_v2(rec)
        row["score"] = final_score(rec)
        rows.append(row)
    return rows


def aggregate(rows: list[dict]) -> dict:
    n = len(rows)
    if not n:
        return {"n": 0}

    def _m(vals, nd=4):
        vals = [v for v in vals if v not in (UND, None) and v != "undefined"]
        return round(sum(vals) / len(vals), nd) if vals else None

    hidden = [r for r in rows if r["hidden"]]
    asked = [r for r in rows if r["aim"] not in (UND, "undefined", None)]
    shifted = [r for r in rows if r["had_moving_shift"]]
    stale_dom = [r for r in rows if r["stale"] not in (UND, "undefined")]
    react = [r["reaction"] for r in rows if r["reaction"] not in (UND, "undefined")]

    def _rec(r):
        rv = r.get("recovery")
        return isinstance(rv, (int, float)) and rv > 0

    return {
        "n": n,
        "score": _m([r["score"] for r in rows]),
        "earned": (_m([r["score"] if _rec(r) else 0.0 for r in hidden]) if hidden else None),
        "inferred": (_m([0.0 if _rec(r) else r["score"] for r in hidden]) if hidden else None),
        "n_hidden": len(hidden),
        "aim": _m([r["aim"] for r in asked]),
        "n_asked": len(asked),
        "recovery": _m([r["recovery"] for r in hidden]),
        "patience_left": _m([r["patience_left"] for r in rows]),
        "staleness": (_m([1.0 if r["stale"] else 0.0 for r in stale_dom]) if stale_dom else None),
        "reaction": _m(react, nd=2),
        "post_shift_score": _m([r["score"] for r in shifted]),
        "n_shifted": len(shifted),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    run = Path(a.run)
    rows = collect(run)
    agg = aggregate(rows)
    print(json.dumps(agg))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
