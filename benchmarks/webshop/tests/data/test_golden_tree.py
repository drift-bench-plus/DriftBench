"""Regenerating the golden cluster must reproduce the committed fixture byte for byte.

The gap this closes: determinism was only asserted by generating twice inside one process and
comparing ids. That catches randomness but not a change in what a graph *contains* -- reorder
conditions, or change what goes into a ground-truth record, and the ids move consistently
while every test still passes.

Not hypothetical. Graph content changed twice on 2026-08-07 (the cluster-pool fix, then the
balanced-quota fix); both were noticed by reading output, and two graph sets had to be
quarantined afterwards because they still looked current.

If this test fails and the change was intended, run `python tools/make_golden.py` and commit
the fixture diff in the same commit as the code change, so the change is visible in review.
"""

import json

import pytest

from intent_graph.golden import GOLDEN_CLUSTER, golden_path, regenerate
from intent_graph.ids import canonical_dumps

pytestmark = pytest.mark.data


@pytest.fixture(scope="module")
def committed():
    return json.loads(golden_path().read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def fresh():
    return regenerate()


def test_golden_fixture_exists_and_is_not_vacuous(committed):
    """An empty fixture would pass every comparison while proving nothing."""
    assert committed["cluster"] == GOLDEN_CLUSTER
    assert committed["graphs"], "fixture has no graphs"
    for graph in committed["graphs"]:
        assert len(graph["children"]) == 8
        assert len(graph["edges"]) == 8


def test_regeneration_matches_the_fixture_byte_for_byte(committed, fresh):
    if canonical_dumps(fresh) == canonical_dumps(committed):
        return
    # a readable failure: name what moved rather than dumping 85KB of JSON
    old = {t["graph_id"]: t for t in committed["graphs"]}
    new = {t["graph_id"]: t for t in fresh["graphs"]}
    msgs = []
    if committed.get("config_fingerprint") != fresh.get("config_fingerprint"):
        msgs.append(f"config changed: {committed.get('config_fingerprint')} -> "
                    f"{fresh.get('config_fingerprint')}")
    if set(old) != set(new):
        msgs.append(f"graph ids changed: only-committed={sorted(set(old) - set(new))} "
                    f"only-fresh={sorted(set(new) - set(old))}")
    for tid in sorted(set(old) & set(new)):
        for field in ("root", "children", "edges", "env_spec"):
            if canonical_dumps(old[tid][field]) != canonical_dumps(new[tid][field]):
                msgs.append(f"{tid}: {field} differs")
    raise AssertionError(
        "generated graphs no longer match tests/golden/webshop_cluster.json:\n  "
        + "\n  ".join(msgs or ["content differs but no field-level diff was located"])
        + "\nIf intended: python tools/make_golden.py, and commit the diff with the change."
    )


def test_ground_truth_content_is_pinned_not_just_hashes(committed):
    """A hash-only fixture would not catch a change in what a GT record contains."""
    for graph in committed["graphs"]:
        gt = graph["root"]["ground_truth"]
        assert gt["kind"] == "purchaseset"
        assert gt["value"], "ground truth values must be pinned, not just the hash"
        for raw in gt["value"]:
            asin, options = json.loads(raw)
            assert isinstance(asin, str) and asin
            assert isinstance(options, list)


def test_every_operator_appears_exactly_twice(committed):
    """The 2/2/2/2 guarantee is part of what the fixture pins."""
    from collections import Counter
    for graph in committed["graphs"]:
        counts = Counter(e["operator"] for e in graph["edges"])
        assert counts == {"REFINEMENT": 2, "RELAXATION": 2,
                          "SUBSTITUTION": 2, "PIVOT": 2}, counts
