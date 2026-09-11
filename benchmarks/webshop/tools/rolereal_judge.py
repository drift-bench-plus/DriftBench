#!/usr/bin/env python
"""Role-realism judging (ruling 2026-08-27): run the GRIP v2 R judges with a
cross-provider pair in which BOTH judges are non-degenerate on both metrics.

judge_a (ROLE_JUDGE, agent side)  -> gateway/gemini-3.5-flash via the model gateway
judge_b (ROLE_SELECT, user side)  -> doubao-seed-2-1-turbo-260628 (ark)

The deepseek judge_a of grip_v2_report answered DIFFERENT on 200/200 consistency
pairs (the failure mode already documented in judges.py); swapping the family keeps
the protocol identical -- run_judges is reused verbatim -- while giving the author's
"two judges must agree" requirement two valid judges on both metrics.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1] / "pipeline" / "src"))
sys.path.insert(0, str(ROOT / "tools"))

GATEWAY = "https://YOUR-GATEWAY.example/v1"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default=None, help="judgment dir (default RUN/judgments_v2)")
    ap.add_argument("--out-name", default="judgments_v2",
                    help="judgment dir name under RUN when --out is not given")
    ap.add_argument("--per-persona", type=int, default=50)
    ap.add_argument("--pairs", type=int, default=200)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--judge-a", default="gateway/gemini-3.5-flash",
                    help="agent-side judge model (ROLE_JUDGE)")
    ap.add_argument("--judge-b", default="doubao-seed-2-1-turbo-260628",
                    help="user-side judge model (ROLE_SELECT)")
    args = ap.parse_args()

    run = Path(args.run)
    if not run.is_absolute():
        run = ROOT / run
    out = Path(args.out) if args.out else run / args.out_name

    # Two independent agent-side clients, one per judge model; a tiny router
    # presents them to run_judges as a single llm (ROLE_JUDGE -> judge_a,
    # ROLE_SELECT -> judge_b). Any ark/gateway pair works.
    def make_judge_llm(model: str):
        from intent_graph.runtime.cli import load_runtime, make_llm
        cfg = load_runtime(None)
        cfg["llm"]["agent_model"] = model
        if model.startswith("gateway/"):
            cfg["llm"]["agent_base_url"] = GATEWAY
            cfg["llm"]["agent_protocol"] = "chat"
            cfg["llm"]["agent_api_key_env"] = "GPT_GATEWAY_KEY"
        cfg["llm"].pop("agent_model_fallback", None)
        return make_llm(cfg, fake=False)
    from intent_graph.runtime.llm import ROLE_JUDGE, ROLE_SELECT

    class RolePair:
        def __init__(self, a, b):
            self._a, self._b = a, b

        def complete(self, prompt, role=None, **kw):
            target = self._a if role == ROLE_JUDGE else self._b
            return target.complete(prompt, role=ROLE_JUDGE, **kw)

    llm = RolePair(make_judge_llm(args.judge_a), make_judge_llm(args.judge_b))

    from intent_graph.runtime.judges import run_judges
    rsum = run_judges(run, out, llm, per_persona=args.per_persona,
                      n_pairs=args.pairs, threads=args.threads)
    print(json.dumps(rsum, indent=1))

    # strict both-agree scoring (the headline the author asked for)
    J = json.loads((out / "judgments.json").read_text())
    ident, cons = J["identification"], J["consistency"]
    strict = {}
    if ident:
        strict["ident_both_correct"] = round(sum(
            1 for j in ident
            if j["judge_a"]["guess"] == j["judge_b"]["guess"] == j["persona"]
        ) / len(ident), 4)
        strict["ident_agreement"] = round(sum(
            1 for j in ident if j["judge_a"]["guess"] == j["judge_b"]["guess"]
        ) / len(ident), 4)
    if cons:
        strict["cons_both_correct"] = round(sum(
            1 for j in cons
            if j["judge_a"]["guess_same"] == j["judge_b"]["guess_same"] == j["truth_same"]
        ) / len(cons), 4)
        strict["cons_agreement"] = round(sum(
            1 for j in cons
            if j["judge_a"]["guess_same"] == j["judge_b"]["guess_same"]
        ) / len(cons), 4)
    print("STRICT " + json.dumps(strict))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
