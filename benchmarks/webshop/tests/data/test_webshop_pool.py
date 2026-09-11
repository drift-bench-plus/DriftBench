"""The candidate pool must be the whole scrape-query cluster.

Regression guard for the defect this replaced: the executor loaded
`human_ins_products_full.jsonl` (the human-instruction subset) and treated it as the
cluster, so every ground truth was computed over ~0.8% of the products that actually share
the query -- 9 of 6,955 for baker's racks. Nothing failed loudly; the GT was simply thin.
"""

import json

import pytest

from intent_graph.cli import build, load_config
from intent_graph.executors.webshop_inproc import _slug

pytestmark = pytest.mark.data


@pytest.fixture(scope="module")
def ex():
    _, executor = build("webshop", load_config())
    executor._load()
    return executor


def _index(executor):
    return json.loads((executor.clusters_dir / "_index.json").read_text(encoding="utf-8"))


def test_pool_matches_the_catalog_cluster_size(ex):
    """Not "a lot of products" -- exactly the cluster, per the shard index."""
    index = _index(ex)
    for cluster in ("baker's racks", "area rugs", "bar stools"):
        expected = index[_slug(cluster)]["count"]
        assert len(ex.load_cluster(cluster)) == expected, cluster


def test_pool_is_far_larger_than_the_human_subset(ex):
    """Pins the actual defect: the subset had 9 baker's racks, the cluster has thousands."""
    assert len(ex.load_cluster("baker's racks")) > 1000


def test_every_pool_product_carries_the_fields_reward_needs(ex):
    pool = ex.load_cluster("area rugs")
    for asin in (pool[0], pool[len(pool) // 2], pool[-1]):
        p = ex.products[asin]
        for field in ("Title", "Description", "BulletPoints", "Attributes", "options", "price"):
            assert field in p, (asin, field)
        assert p["Attributes"], asin


def test_missing_shard_fails_loudly_rather_than_falling_back(ex):
    """A silent fallback to a smaller pool is the bug; absence must raise."""
    with pytest.raises(FileNotFoundError):
        ex.load_cluster("no such cluster exists anywhere")


def test_eviction_keeps_memory_bounded(ex):
    index = _index(ex)
    big = sorted(index.values(), key=lambda v: -v["count"])[:4]
    for entry in big:
        ex.load_cluster(entry["query"])
    assert len(ex._order) <= ex.max_resident_clusters
    resident = {a for c in ex._order for a in ex.by_query[c]}
    assert set(ex.products) == resident, "evicted clusters must not leak product records"
