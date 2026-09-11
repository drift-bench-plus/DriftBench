"""Ground-truth canonicalization and model round-tripping."""

import pytest

from intent_graph.models import Edge, GroundTruth, Node, Operator, Provenance, Graph, sort_conditions


def test_rowset_order_independent():
    a = GroundTruth.rowset([["x", 1], ["y", 2]])
    b = GroundTruth.rowset([["y", 2], ["x", 1]])
    assert a.hash == b.hash


def test_rowset_distinguishes_content():
    assert GroundTruth.rowset([["x"]]).hash != GroundTruth.rowset([["y"]]).hash


def test_purchaseset_option_order_independent():
    a = GroundTruth.purchaseset([("A1", {"color": "red", "size": "s"})])
    b = GroundTruth.purchaseset([("A1", {"size": "s", "color": "red"})])
    assert a.hash == b.hash


def test_scalar_strips_whitespace():
    assert GroundTruth.scalar(" 5\n").hash == GroundTruth.scalar("5").hash


def test_emptiness_by_kind():
    assert GroundTruth.rowset([]).is_empty
    assert GroundTruth.purchaseset([]).is_empty
    assert GroundTruth.scalar("").is_empty
    assert not GroundTruth.scalar("0").is_empty          # "0" is an answer, not an absence
    assert not GroundTruth.statehash("deadbeef").is_empty  # a state always exists


def test_cardinality():
    assert GroundTruth.rowset([["a"], ["b"]]).cardinality() == 2
    assert GroundTruth.scalar("7").cardinality() == 1


def test_sort_conditions_is_canonical():
    a = sort_conditions([("b", "=", 1), ("a", "=", 2)])
    b = sort_conditions([("a", "=", 2), ("b", "=", 1)])
    assert a == b


def _node(**kw):
    defaults = dict(adapter="t", adapter_version="1", env_id="e", base={"f": 1},
                    conditions=[("a", "=", 1)], recipe={"r": 1},
                    ground_truth=GroundTruth.rowset([["x"]]))
    return Node.build(**{**defaults, **kw})


def test_node_roundtrip():
    n = _node()
    assert Node.from_dict(n.to_dict()) == n


def test_node_bound_slots_derived_from_conditions():
    n = _node(conditions=[("a", "=", 1), ("b", "=", 2)])
    assert n.bound_slots == ("a", "b")


def test_tree_roundtrip_and_ids_stable():
    root = _node(conditions=[("a", "=", 1)])
    child = _node(conditions=[("a", "=", 1), ("b", "=", 2)])
    edge = Edge(root.intent_id, child.intent_id, Operator.REFINEMENT,
                {"added": [["b", "=", 2]]}, Provenance.SYNTHETIC, True)
    t = Graph.build(adapter="t", adapter_version="1", env_id="e", env_spec={"env_id": "e"},
                   root=root, children=[child], edges=[edge], config={"seed": 1}, cfg_hash="h")
    assert Graph.from_dict(t.to_dict()) == t


def test_tree_rejects_mixed_environments():
    root = _node(base={"f": 1, "env_id": "e"})
    stray = _node(base={"f": 2, "env_id": "OTHER"}, conditions=[("a", "=", 2)])
    t = Graph.build(adapter="t", adapter_version="1", env_id="e", env_spec={"env_id": "e"},
                   root=root, children=[stray], edges=[], config={}, cfg_hash="h")
    with pytest.raises(ValueError, match="spans environments"):
        t.assert_single_environment()
