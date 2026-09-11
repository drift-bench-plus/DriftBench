"""Cross-cluster leakage: is the scrape-query cluster the right candidate pool?

The plan required this measurement before trusting a cluster-scoped pool, and it was never
built -- which matters, because it is exactly the check that would have caught ground truth
being computed over 0.8% of each cluster.

The question it answers: if we widened the pool beyond the cluster, would products from
*other* clusters also satisfy the intent? If many do, the cluster boundary is doing work the
intent should be doing, and ground truth is an artifact of scoping.
"""

import random

import pytest

from intent_graph.cli import build, load_config
from intent_graph.executors.webshop_inproc import _best_options

pytestmark = pytest.mark.data

LEAKAGE_RED_LINE = 0.02      # plan: >2% and the pools need widening
GOALS = 20
WIDER_MULTIPLE = 10


@pytest.fixture(scope="module")
def env():
    adapter, executor = build("webshop", load_config())
    executor._load()
    return adapter, executor


def _qualifies(ex, product, goal):
    price = product["price"]
    if goal["price_upper"] and price > goal["price_upper"]:
        return False
    options = _best_options(product["options"], goal["goal_options"])
    return ex._reward(product, goal, price, options) >= 0.999


def test_cross_cluster_leakage_is_below_the_red_line(env):
    """Sample test-split goals, score them against a pool ~10x wider than their cluster,
    and count qualifying products that come from a DIFFERENT cluster."""
    adapter, ex = env
    seeds = [s for s in adapter.load() if s.root_eligible]
    rng = random.Random(0)
    picked = rng.sample(seeds, min(GOALS, len(seeds)))

    index = {k: v["query"] for k, v in _index(ex).items()}
    all_clusters = sorted(set(index.values()))

    leaked = checked = 0
    per_goal = []
    for seed in picked:
        cluster = seed.base["cluster"]
        own = ex.load_cluster(cluster)
        goal = adapter.compile(seed.base, seed.conditions)["goal"]

        # a wider pool: this cluster plus other clusters until ~10x the size
        others, target = [], len(own) * WIDER_MULTIPLE
        for other in all_clusters:
            if other == cluster or len(others) >= target:
                continue
            try:
                others.extend(ex.load_cluster(other)[:400])
            except FileNotFoundError:
                continue
            if len(others) >= target:
                break

        hits_out = sum(1 for a in others if a in ex.products and _qualifies(ex, ex.products[a], goal))
        # reload: scoring other clusters may have evicted this one
        own = ex.load_cluster(cluster)
        hits_in = sum(1 for a in own if _qualifies(ex, ex.products[a], goal))
        leaked += hits_out
        checked += len(others)
        per_goal.append((cluster, hits_in, hits_out, len(others)))

    rate = leaked / max(checked, 1)
    detail = "\n".join(f"  {c}: in={i} out={o} of {n} foreign products"
                       for c, i, o, n in per_goal)
    assert rate <= LEAKAGE_RED_LINE, (
        f"cross-cluster leakage {rate:.4%} exceeds the {LEAKAGE_RED_LINE:.0%} red line; "
        f"the cluster boundary is carrying intent that the conditions should carry.\n{detail}"
    )


def _index(ex):
    import json
    return json.loads((ex.clusters_dir / "_index.json").read_text(encoding="utf-8"))
