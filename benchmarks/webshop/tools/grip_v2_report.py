"""GRIP v2 report: personas x the ten metrics, recomputed entirely from disk.

    python tools/grip_v2_report.py --run experiments/runs_grip/B0 [--judge]

The run directory is the record: trajectories (one JSON per episode, self-contained),
judgments/judgments.json (every judge call, stored), and meta.json (commit, config hash,
models, manifest). Any future metric is arithmetic over these files; only the R judges
ever call a model, and only once -- their outputs are part of the record.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1] / "pipeline" / "src"))

from intent_graph.runtime import grip  # noqa: E402

PERSONAS = ["avoidant", "dependent", "intuitive", "rational", "spontaneous"]
UND = "--"


def fmt(v):
    if v in (None, "undefined"):
        return UND
    if isinstance(v, bool):
        return "1.00" if v else "0.00"
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="arm directory with trajectory JSONs")
    ap.add_argument("--judge", action="store_true",
                    help="run the R judges now (writes judgments.json); otherwise reuse")
    ap.add_argument("--per-persona", type=int, default=20)
    ap.add_argument("--pairs", type=int, default=100)
    args = ap.parse_args()

    run = Path(args.run)
    if not run.is_absolute():
        run = ROOT / run
    rows = []
    for f in sorted(run.glob("*.json")):
        t = json.loads(f.read_text(encoding="utf-8"))
        if t.get("outcome") == "ERROR":
            continue
        rows.append(grip.score_v2(t))

    jfile = run / "judgments" / "judgments.json"
    rsum = None
    if args.judge:
        from intent_graph.runtime.cli import load_runtime, make_llm
        cfg = load_runtime(None)
        # judge_b (ROLE_SELECT) must NOT run on doubao-pro: its TPM quota cannot carry
        # transcript-length prompts (ModelAccountTpmRateLimitExceeded killed two passes).
        # turbo keeps the cross-family pairing against deepseek.
        cfg["llm"]["model"] = "doubao-seed-2-1-turbo-260628"
        llm = make_llm(cfg, fake=False)
        from intent_graph.runtime.judges import run_judges
        rsum = run_judges(run, run / "judgments", llm,
                          per_persona=args.per_persona, n_pairs=args.pairs)
    elif jfile.exists():
        from intent_graph.runtime.judges import summarize_judgments
        rsum = summarize_judgments(json.loads(jfile.read_text(encoding="utf-8")))

    cols = ["Success", "Earned", "Inferred", "Identification", "Consistency",
            "Aim", "Recovery", "Patience", "Staleness", "Reaction"]
    print("| Persona | n | " + " | ".join(cols) + " |")
    print("|" + "---|" * (len(cols) + 2))

    def ident_for(p):
        if not rsum or not rsum.get("identification"):
            return UND, UND
        bp = rsum["identification"].get("by_persona", {})
        ident = fmt(bp.get(p)) if p else fmt(
            (rsum["identification"]["judge_a"] + rsum["identification"]["judge_b"]) / 2)
        cons = UND
        if rsum.get("consistency"):
            if p is None:
                cons = fmt((rsum["consistency"]["judge_a"]
                            + rsum["consistency"]["judge_b"]) / 2)
            else:
                cp = rsum["consistency"].get("by_persona", {}).get(p)
                cons = fmt(cp["acc"]) if cp else UND
        return ident, cons

    for p in PERSONAS + [None]:
        sel = [r for r in rows if p is None or r["persona"] == p]
        if not sel:
            continue
        rep = grip.report_v2(sel)
        ident, cons = ident_for(p)
        label = p or "OVERALL"
        print(f"| {label} | {rep['n']} | "
              f"{fmt(rep['success'])} | {fmt(rep['earned'])} | {fmt(rep['inferred'])} | "
              f"{ident} | {cons} | "
              f"{fmt(rep['aim'])} | {fmt(rep['recovery'])} | {fmt(rep['patience_left'])} | "
              f"{fmt(rep['staleness'])} | {fmt(rep['reaction'])} |")

    for p in [None]:
        rep = grip.report_v2(rows)
        print(f"\nconditioning n's (OVERALL): hidden={rep['n_hidden']} asked={rep['n_asked']} "
              f"stale-domain={rep['n_stale_dom']} reaction={rep['n_reaction']} "
              f"shifted={rep['n_shifted']} post-shift-success={fmt(rep['post_shift_success'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
