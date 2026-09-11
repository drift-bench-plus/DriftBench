"""Run the v2 generate -> extract -> admit pipeline over the telecom tree corpus.

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

from loguru import logger as _loguru      # noqa: E402
_loguru.remove()

from intent_tree import storage                                   # noqa: E402
from intent_tree.adapters.tau2_telecom import Tau2TelecomAdapter  # noqa: E402
from intent_tree.executors.tau2_inproc import Tau2Executor        # noqa: E402
from intent_tree.runtime import genverify as gv                   # noqa: E402
from intent_tree.runtime import strategies as st                  # noqa: E402
from intent_tree.runtime.llm import LLMClient                     # noqa: E402

# SESSION MODEL ALLOCATION (ruling 2026-08-16): the parallel session owns
# MODEL ALLOCATION, 2026-08-19.
# Retail runs on doubao-2-0-pro (user/generation) + deepseek-v4-pro (agent). WebShop runs
# on 2-1-pro + deepseek-v4-flash and Airline on 2-1-pro + 2-1-turbo. The allocations are
# disjoint ON PURPOSE so the sessions never contend for one quota.
#
# CORRECTED 2026-08-19: this block previously pinned 2-1-pro/2-1-turbo and asserted a
# "2.1-only" allocation. That is the WebShop/Airline pair -- running retail on it would
# have competed with the concurrent WebShop session for exactly the quota the split
# exists to protect. The old guard is replaced by one that enforces THIS allocation.
LLM_CFG = {
    "base_url": "https://ark.cn-beijing.volces.com/api/v3",
    "model": "doubao-seed-2-0-pro-260215",           # user sim, generation, extraction
    "agent_model": "deepseek-v4-pro-260425",         # the agent under test
    "agent_model_fallback": None,
    # Two identities address each model: the public name and a dedicated endpoint with its
    # own TPM allowance. Requests alternate between them, so usable throughput is roughly
    # doubled. Cache keys use the canonical name only (llm.py), so the rotation cannot
    # fragment the cache.
    # ENDPOINT-ONLY WIRE (ruling 2026-08-20): the public names doubao-seed-2-0-pro /
    # deepseek-v4-pro are in use by OTHER agents right now; this session may only send
    # the dedicated endpoints. The canonical names above stay as CACHE keys and labels;
    # nothing but ep-* ever reaches the API.
    # DUAL IDENTITIES RESTORED (ruling 2026-08-21 evening: other agents released the
    # public names). Wire rotates both; cache keys stay canonical.
    "model_pool": ["doubao-seed-2-0-pro-260215", "ep-YOUR-ENDPOINT-A"],
    "agent_model_pool": ["deepseek-v4-pro-260425", "ep-YOUR-ENDPOINT-B"],
    "api_key_env": "ARK_API_KEY",
    # PACING (2026-08-19). Without it, 96 workers finish a turn together and fire 96
    # requests in the same instant: the per-SECOND gate trips even though the per-minute
    # quota has room. Measured live at width 96 x 2 processes: 184 rate-limit retries and
    # 15 episodes in 15 minutes. The token bucket is shared across all worker threads in
    # a process, so the same volume goes out smoothly instead of in waves.
    "max_rps": 7.0,
    # reasoning tokens count against this budget on ark; a high ceiling costs nothing
    "max_output_tokens": 4000,
    "disable_thinking": True,                        # both models, per the author
    "temperature_render": 0.7, "temperature_phrase": 0.3, "temperature_select": 0.0,
    "temperature_agent_act": 0.7, "temperature_agent_audit": 0.0,
    "temperature_agent_sample": 1.0, "temperature_agent_judge": 0.0,
    "reasoning_effort": "", "timeout_s": 90, "sdk_max_retries": 2,
    "cache_dir": WS + "/artifacts/llm_cache", "prompt_version": "tau2v1", "offline": False,
}
assert LLM_CFG["model"] == "doubao-seed-2-0-pro-260215", (
    f"retail user/generation model {LLM_CFG['model']!r} is off-allocation")
assert LLM_CFG["agent_model"] == "deepseek-v4-pro-260425", (
    f"retail agent model {LLM_CFG['agent_model']!r} is off-allocation")
for _m in (LLM_CFG["model"], LLM_CFG["agent_model"]):
    # hard stop on the concurrent session's models: 2-1-* belongs to WebShop/Airline
    assert "2-1-" not in _m, f"model {_m!r} belongs to the WebShop/Airline allocation"

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
                "manager": tree.root.base["manager"]}
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
    ap.add_argument("--out", default="artifacts/telecom_samples_v2.json")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--strategies", default="", help="comma list; empty = all supported")
    a = ap.parse_args()

    adapter, executor = Tau2TelecomAdapter(), Tau2Executor()
    llm = LLMClient({"llm": LLM_CFG})
    trees = sorted(storage.iter_trees(Path(WS + "/artifacts/trees/tau2_telecom")),
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
