"""Build the EXPANDED airline forest: every branch intent is re-rooted as its own tree.

Why this exists (ruling 2026-08-18, "we need more perturbation... at least 200+ records"):
perturbation samples derive from tree ROOTS -- one cell per (root, fault type) -- so sample
volume scales with the number of ROOTS, not with branch width. The base forest has 24 roots
(tau2 ships 26 write-bearing airline tasks, two pairs of which are byte-identical), which
caps the corpus at 240 cells. The re-rooting census (expand_census_airline.py) measured that
all 143 branch intents validate and support a tree of their own, so re-rooting multiplies the
root count using material that never leaves the base tasks' own write-sets and the shipped
database.

Legitimacy is identical to the base forest's: every expanded root is an execution-verified,
policy-gated intent (is_valid_intent applies the airline policy predicates at build time),
its ground truth is a reproducible db hash, and no LLM is involved anywhere.

PROVENANCE / CLUSTERING. Each expanded root's record_id is "<base_task_id>+<intent8>", so
root.source_record carries the base task it descends from. Everything derived from one base
task is ONE CLUSTER, and the eval/learn split must be blocked on the cluster (splitting on
tree id would put near-duplicate sibling intents on both sides of the split). The cluster id
is `source_record.split("+")[0]`.

Output: artifacts/trees/tau2_airline_x/ + artifacts/airline_x_tree_report.json.
Promote with --promote (archives the base forest to tau2_airline_base24) only after the
battery and the policy gate are clean on the new forest.
"""

import argparse
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

WS = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("TAU2_SRC", WS + "/tau2-bench/src")
os.environ.setdefault("TAU2_DATA", WS + "/tau2-bench/data/tau2")
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
sys.path.insert(0, WS + "/../../pipeline/src")
sys.path.insert(0, os.environ["TAU2_SRC"])
import airline_compat  # noqa: F401  (tree->graph rename shim)
from loguru import logger as _l          # noqa: E402
_l.remove()

from intent_tree import storage                                    # noqa: E402
from intent_tree.adapters.tau2_airline import Tau2AirlineAdapter   # noqa: E402
from intent_tree.engine import RunStats, generate                  # noqa: E402
from intent_tree.executors.tau2_inproc import Tau2Executor         # noqa: E402
from intent_tree.models import Seed                                # noqa: E402

CFG = {                      # identical to the base forest: config participates in tree_id
    "branching": 8,
    "balanced_branching": True,
    "require_full_quota": False,
    "min_gt_moved_edges": 1,
    "depth": 1,
    "seed": 7,
}

BASE = Path(WS) / "artifacts/trees/tau2_airline"
OUT = Path(WS) / "artifacts/trees/tau2_airline_x"


def build_seed_pool(adapter) -> list[Seed]:
    """Base task seeds + every distinct branch intent, de-duplicated by (base, conditions)."""
    base_seeds = adapter.load()
    pool: dict[tuple, Seed] = {}
    for s in base_seeds:
        pool[(json.dumps(sorted(s.base.items()), default=str), s.conditions)] = s

    n_new = 0
    for tree in sorted(storage.iter_trees(BASE), key=lambda t: t.tree_id):
        cluster = tree.root.source_record or tree.tree_id[:8]
        for node in tree.children:
            conds = tuple(tuple(c) for c in node.conditions)
            key = (json.dumps(sorted(dict(node.base).items()), default=str), conds)
            if key in pool:
                continue        # a pivot child IS another real intent: already a root
            pool[key] = Seed(
                record_id=f"{cluster}+{node.intent_id[:8]}",
                env_key=adapter.ENV_KEY,
                base=dict(node.base),
                conditions=conds,
                shipped_answer=None,     # validate() executes; it never reads this
                meta={"expanded": True, "cluster": cluster,
                      "from_tree": tree.tree_id, "from_intent": node.intent_id},
            )
            n_new += 1
    print(f"seed pool: {len(base_seeds)} base + {n_new} re-rooted = {len(pool)} distinct",
          flush=True)
    return list(pool.values())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--promote", action="store_true",
                    help="after building, archive the base forest and install this one "
                         "as artifacts/trees/tau2_airline (run the battery + gate FIRST)")
    a = ap.parse_args()

    adapter, executor = Tau2AirlineAdapter(), Tau2Executor()
    pool = build_seed_pool(adapter)

    class ExpandedAdapter(Tau2AirlineAdapter):
        """Same adapter; only the seed source differs."""
        def load(self):
            return pool

    OUT.mkdir(parents=True, exist_ok=True)
    stats = RunStats()
    n, edges, ops, clusters = 0, 0, Counter(), Counter()
    for tree in generate(ExpandedAdapter(), executor, CFG, stats=stats):
        storage.write_tree(OUT, tree)
        n += 1
        edges += len(tree.edges)
        for e in tree.edges:
            ops[e.operator.value] += 1
        clusters[str(tree.root.source_record).split("+")[0]] += 1
        if n % 20 == 0:
            print(f"  {n} trees, {edges} edges", flush=True)

    report = {"config": CFG, "trees": n, "edges": edges,
              "per_operator": dict(ops),
              "clusters": len(clusters),
              "trees_per_cluster": {"min": min(clusters.values()) if clusters else 0,
                                    "max": max(clusters.values()) if clusters else 0,
                                    "mean": round(n / max(len(clusters), 1), 2)},
              "cluster_sizes": dict(clusters.most_common()),
              "projected_cells": n * 10,
              "stats": stats.to_dict()}
    Path(WS + "/artifacts/airline_x_tree_report.json").write_text(json.dumps(report, indent=1))
    print(f"\n{n} trees / {edges} edges / {len(clusters)} clusters -> {OUT}")
    print(f"projected perturbation cells: {n * 10}")
    print(json.dumps({k: report[k] for k in
                      ("per_operator", "clusters", "trees_per_cluster")}, indent=1))

    if a.promote:
        arch = Path(WS) / "artifacts/trees/tau2_airline_base24"
        if not arch.exists():
            shutil.copytree(BASE, arch)
        shutil.rmtree(BASE)
        shutil.copytree(OUT, BASE)
        print(f"promoted: base forest archived at {arch}, expanded forest installed at {BASE}")


if __name__ == "__main__":
    main()
