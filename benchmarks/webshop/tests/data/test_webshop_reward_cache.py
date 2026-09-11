"""The two memoizations sit directly in the ground-truth path — prove they change nothing.

The `webshop_inproc` docstring has always claimed "there is a unit test that asserts cached
and uncached rewards agree". There was not one. It matters more now: `load_cluster` evicts
least-recently-used clusters and prunes `_type_cache` alongside them, so a wrong eviction
would silently corrupt ground truth rather than crash.
"""

import importlib
import random
import sys

import pytest

from intent_graph.cli import build, load_config
from intent_graph.executors.webshop_inproc import _best_options

pytestmark = pytest.mark.data

CLUSTER = "area rugs"
SAMPLE = 60


@pytest.fixture(scope="module")
def ex():
    _, executor = build("webshop", load_config())
    executor._load()
    return executor


@pytest.fixture(scope="module")
def uncached(ex):
    """A fresh copy of WebShop's goal module: original nlp, original get_type_reward."""
    sys.path.insert(0, str(ex.repo))
    import web_agent_site.engine.goal as gm
    return importlib.reload(gm).get_reward


def _goal(ex, pool):
    seed = ex.products[pool[0]]
    return {"asin": seed["asin"], "query": CLUSTER, "name": seed["name"],
            "product_category": seed["product_category"], "category": "garden",
            "attributes": ["machine washable"], "price_upper": 100.0, "goal_options": {}}


def _rewards(ex, goal, sample):
    out = []
    for asin in sample:
        p = ex.products[asin]
        out.append(ex._reward(p, goal, p["price"],
                              _best_options(p["options"], goal["goal_options"])))
    return out


def _disagreements(ex, uncached, goal, sample):
    bad = []
    for asin in sample:
        p = ex.products[asin]
        options = _best_options(p["options"], goal["goal_options"])
        a = ex._reward(p, goal, p["price"], options)
        b = uncached(p, goal, p["price"], options)
        if abs(a - b) > 1e-12:
            bad.append((asin, a, b))
    return bad


def test_cached_and_uncached_rewards_agree(ex, uncached):
    pool = ex.load_cluster(CLUSTER)
    goal = _goal(ex, pool)
    sample = random.Random(0).sample(pool, min(SAMPLE, len(pool)))
    assert _disagreements(ex, uncached, goal, sample) == []


def test_rewards_survive_cluster_eviction(ex, uncached):
    """Reloading a cluster after eviction must reproduce identical rewards."""
    pool = ex.load_cluster(CLUSTER)
    goal = _goal(ex, pool)
    sample = random.Random(0).sample(pool, min(SAMPLE, len(pool)))
    before = _rewards(ex, goal, sample)

    for other in ("bar stools", "accent chairs", "baker's racks"):
        ex.load_cluster(other)
    assert CLUSTER not in ex.by_query, "eviction should have dropped the cluster"

    ex.load_cluster(CLUSTER)
    assert _rewards(ex, goal, sample) == before
    assert _disagreements(ex, uncached, goal, sample) == []
