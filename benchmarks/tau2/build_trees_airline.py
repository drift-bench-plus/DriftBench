"""Build the airline intent-tree forest (no LLM anywhere -- hard project rule).

Config is the retail v3 config VERBATIM: it participates in tree_id via config_hash, and
the cross-domain convention is one config unless a change is deliberate.

Writes artifacts/trees/tau2_airline/<tree_id>.json + artifacts/airline_tree_report.json.
"""

import json
import os
import sys
from pathlib import Path

WS = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("TAU2_SRC", WS + "/tau2-bench/src")
os.environ.setdefault("TAU2_DATA", WS + "/tau2-bench/data/tau2")
sys.path.insert(0, WS + "/../../pipeline/src")
sys.path.insert(0, os.environ["TAU2_SRC"])

from intent_tree import storage  # noqa: E402
from intent_tree.adapters.tau2_airline import Tau2AirlineAdapter  # noqa: E402
from intent_tree.engine import RunStats, generate  # noqa: E402
from intent_tree.executors.tau2_inproc import Tau2Executor  # noqa: E402

CFG = {
    "branching": 8,
    "balanced_branching": True,
    "require_full_quota": False,
    "min_gt_moved_edges": 1,
    "depth": 1,
    "seed": 7,
}

OUT = Path(WS) / "artifacts/trees/tau2_airline"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    adapter = Tau2AirlineAdapter()
    executor = Tau2Executor()
    stats = RunStats()
    trees = []
    for tree in generate(adapter, executor, CFG, stats=stats):
        storage.write_tree(OUT, tree)
        trees.append(tree)
        ops = {}
        for e in tree.edges:
            ops[e.operator.value] = ops.get(e.operator.value, 0) + 1
        print(f"  {tree.tree_id}  root={tree.root.source_record:>3s}  "
              f"edges={len(tree.edges)}  {ops}")
    report = {"config": CFG, "trees": len(trees), "stats": stats.to_dict()}
    with open(WS + "/artifacts/airline_tree_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    print(f"\n{len(trees)} trees -> {OUT}")
    print(json.dumps(stats.to_dict(), indent=1))


if __name__ == "__main__":
    main()
