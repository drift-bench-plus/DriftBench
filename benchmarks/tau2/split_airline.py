"""Cluster-blocked eval/learn split for the airline corpus.

The retail rule was "sha1(tree_id) mod 100 < 30 = learn". That rule is UNSAFE on the
expanded airline forest: re-rooted trees are sibling intents of the base task they came
from, differing by one condition, so splitting on tree id would put near-duplicates on both
sides -- an arm that learned an askbook entry on one sibling would be evaluated on its twin,
and the eval number would be contaminated.

So the split blocks on the CLUSTER: every tree whose root descends from the same base tau2
task goes to the same side. Cluster id = root.source_record.split("+")[0] (build_trees_
airline_x.py writes "<base_task>+<intent8>" for re-rooted roots; base roots carry the bare
task id). Assignment is deterministic: sha1(cluster) mod 100 < 30 -> learn.

Writes artifacts/airline_samples_eval.json and artifacts/airline_samples_learn.json in the
runner's sample-artifact format, plus a manifest recording the split for the datasheet.
"""

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

WS = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("TAU2_SRC", WS + "/tau2-bench/src")
os.environ.setdefault("TAU2_DATA", WS + "/tau2-bench/data/tau2")
sys.path.insert(0, WS + "/../../pipeline/src")
import airline_compat  # noqa: F401  (tree->graph rename shim)
from loguru import logger as _l          # noqa: E402
_l.remove()

from intent_tree import storage          # noqa: E402

LEARN_PCT = 30


def cluster_of(tree) -> str:
    rec = str(tree.root.source_record or tree.tree_id)
    return rec.split("+")[0]


def is_learn(cluster: str) -> bool:
    h = hashlib.sha1(cluster.encode("utf-8")).hexdigest()
    return int(h, 16) % 100 < LEARN_PCT


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default="artifacts/airline_samples_v2.json",
                    help="comma-separated sample artifacts, newest-run first")
    ap.add_argument("--trees", default="artifacts/trees/tau2_airline")
    a = ap.parse_args()

    trees = {t.tree_id: t for t in storage.iter_trees(Path(WS + "/" + a.trees))}
    cluster_by_tree = {tid: cluster_of(t) for tid, t in trees.items()}

    # merge the artifacts exactly as export does: first decisive row per (tree, strategy)
    decided, rows = {}, []
    for f in a.samples.split(","):
        p = Path(WS + "/" + f.strip())
        if not p.exists():
            continue
        for r in json.loads(p.read_text())["results"]:
            key = (r["tree"], r["strategy"])
            if key in decided:
                continue
            if r["status"] == "admitted":
                decided[key] = True
                rows.append(r)
            elif r["status"] == "rejected_collision":
                decided[key] = False

    missing = {r["tree"] for r in rows} - set(cluster_by_tree)
    if missing:
        raise SystemExit(f"{len(missing)} sample tree_ids are not in {a.trees} "
                         f"(e.g. {sorted(missing)[:3]}) -- samples and trees are out of "
                         f"step; re-export against the current forest.")

    learn, evalr = [], []
    for r in rows:
        (learn if is_learn(cluster_by_tree[r["tree"]]) else evalr).append(r)

    for name, sub in (("learn", learn), ("eval", evalr)):
        clusters = {cluster_by_tree[r["tree"]] for r in sub}
        summary = {"split": name, "rule": f"sha1(cluster) mod 100 < {LEARN_PCT} = learn",
                   "samples": len(sub), "clusters": len(clusters),
                   "trees": len({r["tree"] for r in sub}),
                   "by_strategy": dict(Counter(r["strategy"] for r in sub))}
        Path(WS + f"/artifacts/airline_samples_{name}.json").write_text(
            json.dumps({"summary": summary, "results": sub}, indent=1))
        print(json.dumps(summary, indent=1))

    overlap = ({cluster_by_tree[r["tree"]] for r in learn}
               & {cluster_by_tree[r["tree"]] for r in evalr})
    print(f"\ncluster overlap between splits: {len(overlap)} (must be 0)")
    if overlap:
        raise SystemExit("SPLIT LEAK: a cluster appears on both sides")


if __name__ == "__main__":
    main()
