"""Pragmatic-collision audit for the AIRLINE corpus (see collision_audit.py for the
contract; this is the airline extension kept as its own script so the frozen
telecom/retail audit artifact that ships in data/tau2-dataset is never overwritten).

Fail-closed dual extraction: e1 = doubao-seed-2-1-pro (ROLE_SELECT), e2 =
doubao-seed-2-1-turbo (ROLE_JUDGE) -- cross-MODEL, same family, per the 2026-08-18
allocation; the datasheet must record this difference from the retail/telecom corpus's
doubao-vs-deepseek split. Either extractor reading a domain-real assertion beyond the
true conditions rejects the sample.
"""
import json
import os
import sys
import threading
from pathlib import Path

WS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, WS + "/../../pipeline/src")
os.environ["TAU2_SRC"] = WS + "/tau2-bench/src"
os.environ["TAU2_DATA"] = WS + "/tau2-bench/data/tau2"
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
_envf = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env")
if "ARK_API_KEY" not in os.environ and os.path.exists(_envf):
    os.environ["ARK_API_KEY"] = [l.split("=", 1)[1].strip() for l in open(_envf)
                                 if l.startswith("ARK_API_KEY=")][0]
from loguru import logger as _l          # noqa: E402
_l.remove()

from intent_tree.runtime.genverify import _first_json             # noqa: E402
from intent_tree.runtime.llm import LLMClient, ROLE_JUDGE, ROLE_SELECT  # noqa: E402
from perturb_run_airline import LLM_CFG                           # noqa: E402

SCHEMA = ('Reply with ONLY a JSON object: {"collides": true or false, '
          '"evidence": "<the colliding phrase, or empty>"}')


def airline_prompt(query: str, true_requests: list[str]) -> str:
    return (
        "A passenger message to an airline's customer service is below. The passenger's "
        "REAL requests are exactly these:\n  - " + "\n  - ".join(true_requests) +
        "\n\nThe message deliberately also contains irrelevant chatter (trip stories, "
        "compliments, preferences about food or scenery, mentions of other people...). "
        "That is fine. Your ONLY job: check whether the message ALSO asks for or asserts "
        "any OTHER airline operation beyond the real ones -- another booking, "
        "cancellation, flight or cabin change, baggage or passenger change, a refund or "
        "compensation claim, or a claim about a different reservation or flight that "
        "reads as a request.\n\n"
        "Merely mentioning an airport, a past trip, or a travel companion is NOT a "
        "request. Asking the airline to DO something about it IS.\n\n"
        "Message:\n\"\"\"\n" + query + "\n\"\"\"\n\n" + SCHEMA)


def describe_true(conds) -> list[str]:
    by_action: dict[str, list[str]] = {}
    for s, _o, v in conds:
        body = str(s)
        if body.startswith("arg:"):
            _a, idx, tool, key = body.split(":", 3)
            by_action.setdefault(f"{idx}:{tool}", []).append(f"{key}={v}")
        elif body.startswith("also:"):
            _a, idx, tool = body.split(":", 2)
            by_action.setdefault(f"{idx}:{tool}", []).append(f"args={v}")
    return [f"{k.split(':', 1)[1]} with {', '.join(vv)}"
            for k, vv in sorted(by_action.items())]


_local = threading.local()


def check(job):
    r, prompt = job
    llm = getattr(_local, "llm", None)
    if llm is None:
        llm = LLMClient({"llm": LLM_CFG})
        _local.llm = llm
    verdicts = {}
    for tag, role in (("e1", ROLE_SELECT), ("e2", ROLE_JUDGE)):
        try:
            obj = _first_json(llm.complete(prompt, role=role)) or {}
        except Exception as e:
            obj = {"collides": True, "evidence": f"extractor_error:{type(e).__name__}"}
        verdicts[tag] = {"collides": bool(obj.get("collides")),
                         "evidence": str(obj.get("evidence") or "")[:200]}
    collides = verdicts["e1"]["collides"] or verdicts["e2"]["collides"]
    return {"domain": "airline", "tree": r["tree"], "strategy": r["strategy"],
            "source": r.get("_source"), "collides": collides, "verdicts": verdicts}


def main() -> None:
    # row-exhaustive over every admitted pragmatic row in every airline artifact,
    # winners and superseded alike -- newest-run-first order must mirror
    # export_dataset.py's airline file list exactly
    FILES = ["artifacts/airline_v3.json"]   # v3 = the 40-tree selected forest; earlier
    # runs are keyed to pre-rebuild tree ids and must not be mixed in
    jobs = []
    for f in FILES:
        p = Path(WS + "/" + f)
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        for r in d["results"]:
            if r["status"] != "admitted":
                continue
            if (r.get("signature") or {}).get("verdict") != "ok_pragmatic_extra":
                continue
            conds = r.get("root_conditions") or []
            jobs.append((dict(r, _source=f),
                         airline_prompt(r["query"], describe_true(conds))))
    print(f"{len(jobs)} pragmatic-extra samples to audit", flush=True)

    import concurrent.futures as cf
    rows = []
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        for i, out in enumerate(ex.map(check, jobs), 1):
            rows.append(out)
            if i % 20 == 0:
                print(f"  {i}/{len(jobs)} | collisions so far "
                      f"{sum(r['collides'] for r in rows)}", flush=True)

    summary = {"audited": len(rows),
               "collisions": sum(r["collides"] for r in rows),
               "by_domain": {"airline": {"audited": len(rows),
                                         "collisions": sum(r["collides"] for r in rows)}},
               "only_one_extractor": sum(1 for r in rows if r["collides"] and not
                                         (r["verdicts"]["e1"]["collides"]
                                          and r["verdicts"]["e2"]["collides"]))}
    Path(WS + "/artifacts/airline_collision_audit.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=1))

    # APPLY, fail-closed: every row of a flagged (tree, strategy) key -- winner and
    # superseded alike -- flips admitted -> rejected_collision in the artifact files, so
    # export_dataset.py's priority walk kills the key with no fallback (the retail
    # release-hardening rule, 2026-08-13).
    flagged = {(r["tree"], r["strategy"]) for r in rows if r["collides"]}
    changed = 0
    for f in FILES:
        p = Path(WS + "/" + f)
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        for r in d["results"]:
            if (r["tree"], r["strategy"]) in flagged and r["status"] == "admitted":
                r["status"] = "rejected_collision"
                r["collision_evidence"] = next(
                    (x["verdicts"]["e1"]["evidence"] or x["verdicts"]["e2"]["evidence"]
                     for x in rows if x["collides"]
                     and (x["tree"], x["strategy"]) == (r["tree"], r["strategy"])), "")
                changed += 1
        p.write_text(json.dumps(d, indent=1))
    print(f"applied: {changed} rows -> rejected_collision across {len(flagged)} keys")
    print(json.dumps(summary, indent=1))
    for r in [r for r in rows if r["collides"]][:6]:
        ev = r["verdicts"]["e1"]["evidence"] or r["verdicts"]["e2"]["evidence"]
        print(f"  COLLIDES {r['tree'][:8]} {r['strategy']}: {ev[:120]}")


if __name__ == "__main__":
    main()
