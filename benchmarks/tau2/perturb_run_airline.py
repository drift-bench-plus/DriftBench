"""Run the v2 generate -> extract -> admit pipeline over the airline tree corpus.

One cell = (tree, strategy). Cells are independent, so they run on a thread pool; each
worker owns its own tau2 session because a tau2 environment is stateful.
"""
import argparse
import json
import os
import sys
import threading
import time
from collections import Counter
from pathlib import Path

WS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, WS + "/../../pipeline/src")
os.environ["TAU2_SRC"] = WS + "/tau2-bench/src"
os.environ["TAU2_DATA"] = WS + "/tau2-bench/data/tau2"
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
if "ARK_API_KEY" not in os.environ:
    _envf = os.path.join(WS, "..", ".env")
    if os.path.exists(_envf):
        os.environ["ARK_API_KEY"] = [l.split("=", 1)[1].strip() for l in
                                     open(_envf)
                                     if l.startswith("ARK_API_KEY=")][0]

import airline_compat  # noqa: F401  (tree->graph rename shim)
from loguru import logger as _loguru      # noqa: E402
_loguru.remove()

from intent_tree import storage                                   # noqa: E402
from intent_tree.adapters.tau2_airline import Tau2AirlineAdapter  # noqa: E402
from intent_tree.executors.tau2_inproc import Tau2Executor        # noqa: E402
from intent_tree.runtime import genverify as gv                   # noqa: E402
from intent_tree.runtime import strategies as st                  # noqa: E402
from intent_tree.runtime.llm import LLMClient                     # noqa: E402

LLM_CFG = {
    "base_url": "https://ark.cn-beijing.volces.com/api/v3",
    # SESSION MODEL ALLOCATION (ruling 2026-08-18): the retail campaign is finished and
    # its pair is released to the airline build -- 2.1-pro generates/extracts (user
    # side), 2.1-turbo is the second extractor and later the agent under test. Note for
    # the datasheet: dual extraction is cross-MODEL (pro vs turbo) but same-family,
    # unlike the retail/telecom corpus's doubao-vs-deepseek split.
    "model": "doubao-seed-2-1-pro-260628",           # generation + extraction (user side)
    "agent_model": "doubao-seed-2-1-turbo-260628",   # second extractor / agent under test
    "agent_model_fallback": None,
    "api_key_env": "ARK_API_KEY",
    "max_output_tokens": 4000,
    "disable_thinking": True,                        # both models, per the allocation
    "temperature_render": 0.7, "temperature_phrase": 0.3, "temperature_select": 0.0,
    "temperature_agent_act": 0.7, "temperature_agent_audit": 0.0,
    "temperature_agent_sample": 1.0, "temperature_agent_judge": 0.0,
    "reasoning_effort": "", "timeout_s": 90, "sdk_max_retries": 2,
    "cache_dir": WS + "/artifacts/llm_cache", "prompt_version": "tau2v1", "offline": False,
}

ALLOWED_MODELS = {"doubao-seed-2-1-pro-260628", "doubao-seed-2-1-turbo-260628"}
for _m in (LLM_CFG["model"], LLM_CFG["agent_model"]):
    assert _m in ALLOWED_MODELS, f"model {_m!r} violates the airline build allocation"

_local = threading.local()


def session_for(adapter, executor):
    s = getattr(_local, "session", None)
    if s is None:
        s = executor.open(adapter.env_spec(adapter.ENV_KEY))
        s.__enter__()
        _local.session = s
    return s


def applicable(strategy, tree, adapter) -> tuple[bool, str]:
    conds = tree.root.conditions
    if len(conds) < strategy.needs_conditions:
        return False, "too_few_conditions"
    if strategy.needs_ordered_slot and not any(
            st.slot_is_ordered(s, o, v, adapter) for s, o, v in conds):
        return False, "no_ordered_slot"
    return True, "ok"


def run_cell(args):
    tree, strategy, adapter, executor, llm, attempts = args
    ok, why = applicable(strategy, tree, adapter)
    if not ok:
        return {"tree": tree.tree_id, "strategy": strategy.id, "status": "skipped",
                "verdict": why}
    sess = session_for(adapter, executor)
    trace: list = []
    t0 = time.time()
    try:
        payload, _verdict = gv.build_v2_mask(
            tree, strategy, llm=llm, adapter=adapter, executor=executor, session=sess,
            context=adapter.render_context(tree.root.base), dual_extract=True,
            attempts=attempts, trace=trace)
        return {"tree": tree.tree_id, "strategy": strategy.id, "status": "admitted",
                "query": payload["query"], "mask": payload["mask"],
                "signature": payload.get("signature"), "attempts": len(trace) + 1,
                "sec": round(time.time() - t0, 1),
                "root_conditions": [list(c) for c in tree.root.conditions],
                "writes": tree.root.base.get("writes")}
    except gv.Rejected as e:
        return {"tree": tree.tree_id, "strategy": strategy.id, "status": "rejected",
                "verdict": e.verdict, "attempts": len(trace),
                "sec": round(time.time() - t0, 1),
                "last_text": (trace[-1]["text"] if trace else None)}
    except Exception as e:
        return {"tree": tree.tree_id, "strategy": strategy.id, "status": "error",
                "verdict": f"{type(e).__name__}: {str(e)[:200]}",
                "sec": round(time.time() - t0, 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trees", type=int, default=0, help="0 = all")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--attempts", type=int, default=4)
    ap.add_argument("--out", default="artifacts/airline_samples_v2.json")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--strategies", default="", help="comma list; empty = all supported")
    ap.add_argument("--trees-dir", default="artifacts/trees/tau2_airline",
                    help="tree corpus to perturb")
    a = ap.parse_args()

    adapter, executor = Tau2AirlineAdapter(), Tau2Executor()
    llm = LLMClient({"llm": LLM_CFG})
    trees = sorted(storage.iter_trees(Path(WS + "/" + a.trees_dir)),
                   key=lambda t: t.tree_id)
    if a.trees:
        trees = trees[:a.trees]
    strategies = [s for s in st.STRATEGIES if s.supported]
    if a.strategies:
        want = set(a.strategies.split(","))
        strategies = [s for s in strategies if s.id in want]

    jobs = [(t, s, adapter, executor, llm, a.attempts) for t in trees for s in strategies]
    print(f"[{a.tag}] {len(trees)} trees x {len(strategies)} strategies = {len(jobs)} cells,"
          f" {a.workers} workers", flush=True)

    import concurrent.futures as cf
    results, t0 = [], time.time()
    with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
        for i, r in enumerate(ex.map(run_cell, jobs), 1):
            results.append(r)
            if i % 25 == 0:
                adm = sum(1 for x in results if x["status"] == "admitted")
                print(f"  {i}/{len(jobs)} cells | admitted {adm} "
                      f"({adm/i:.0%}) | {time.time()-t0:.0f}s", flush=True)

    by_strategy = {}
    for s in strategies:
        sub = [r for r in results if r["strategy"] == s.id]
        adm = [r for r in sub if r["status"] == "admitted"]
        by_strategy[s.id] = {
            "cells": len(sub), "admitted": len(adm),
            "rate": round(len(adm) / max(len(sub), 1), 3),
            "verdicts": dict(Counter(r.get("verdict") for r in sub
                                     if r["status"] != "admitted")),
        }
    summary = {
        "tag": a.tag, "trees": len(trees), "strategies": len(strategies),
        "cells": len(jobs), "wall_s": round(time.time() - t0, 1),
        "admitted": sum(1 for r in results if r["status"] == "admitted"),
        "rejected": sum(1 for r in results if r["status"] == "rejected"),
        "errors": sum(1 for r in results if r["status"] == "error"),
        "skipped": sum(1 for r in results if r["status"] == "skipped"),
        "by_strategy": by_strategy,
        "llm_usage": getattr(llm, "_usage", {}),
    }
    Path(WS + "/artifacts").mkdir(exist_ok=True)
    with open(WS + "/" + a.out, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=1)
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
