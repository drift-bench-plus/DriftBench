"""Consistency-battery checks, including the semantics they exist to protect."""

from intent_graph import battery
from intent_graph.models import Edge, GroundTruth, Node, Operator, Provenance, Graph


def _node(conds, gt, **kw):
    return Node.build(adapter="t", adapter_version="1", env_id="e", base={"f": 1},
                      conditions=conds, recipe={"r": conds}, ground_truth=gt, **kw)


def _tree(root, children, edges):
    return Graph.build(adapter="t", adapter_version="1", env_id="e", env_spec={"env_id": "e"},
                      root=root, children=children, edges=edges, config={}, cfg_hash="h")


def test_clean_tree_produces_no_findings():
    root = _node((("a", "=", 1),), GroundTruth.rowset([["x"], ["y"]]), is_seed=True)
    seed = _node((("a", "=", 1), ("b", "=", 2)), GroundTruth.rowset([["x"]]))
    edge = Edge(root.intent_id, seed.intent_id, Operator.REFINEMENT,
                {"added": [["b", "=", 2]]}, Provenance.REAL, True)
    assert battery.check_structure(_tree(root, [seed], [edge])) == []


def test_flags_empty_ground_truth():
    root = _node((("a", "=", 1),), GroundTruth.rowset([]), is_seed=True)
    seed = _node((("a", "=", 1), ("b", "=", 2)), GroundTruth.rowset([["x"]]))
    edge = Edge(root.intent_id, seed.intent_id, Operator.REFINEMENT, {}, Provenance.REAL, True)
    checks = {f.check for f in battery.check_structure(_tree(root, [seed], [edge]))}
    assert "non_empty" in checks


def test_flags_tree_where_nothing_moves():
    gt = GroundTruth.rowset([["x"]])
    root = _node((("a", "=", 1),), gt, is_seed=True)
    seed = _node((("a", "=", 1), ("b", "=", 2)), gt)
    edge = Edge(root.intent_id, seed.intent_id, Operator.REFINEMENT, {}, Provenance.REAL, False)
    checks = {f.check for f in battery.check_structure(_tree(root, [seed], [edge]))}
    assert "discriminative" in checks


def test_flags_refinement_that_is_not_a_subset():
    root = _node((("a", "=", 1),), GroundTruth.rowset([["x"]]), is_seed=True)
    seed = _node((("a", "=", 1), ("b", "=", 2)), GroundTruth.rowset([["y"]]))
    edge = Edge(root.intent_id, seed.intent_id, Operator.REFINEMENT, {}, Provenance.REAL, True)
    findings = battery.check_structure(_tree(root, [seed], [edge]))
    assert any(f.check == "monotonic" for f in findings)


def test_monotonicity_skipped_for_computed_answers():
    """An aggregate (COUNT/SUM) changes value without any subset relation holding."""
    root = _node((("a", "=", 1),), GroundTruth.rowset([["5"]]), is_seed=True,
                 gt_extensional=False)
    seed = _node((("a", "=", 1), ("b", "=", 2)), GroundTruth.rowset([["3"]]),
                 gt_extensional=False)
    edge = Edge(root.intent_id, seed.intent_id, Operator.REFINEMENT, {}, Provenance.REAL, True)
    assert not [f for f in battery.check_structure(_tree(root, [seed], [edge]))
                if f.check == "monotonic"]


def test_purchaseset_monotonicity_compares_products_not_option_tuples():
    """Regression: relaxing an option requirement changes the correct option selection.

    The same product still qualifies, but the answer tuple differs, so comparing full
    (product, options) tuples would wrongly report a broken relaxation.
    """
    strict = GroundTruth.purchaseset([("A1", {"flavor": "lemon"})])
    relaxed = GroundTruth.purchaseset([("A1", {}), ("A2", {})])
    assert relaxed.monotonic_view() >= strict.monotonic_view()

    root = _node((("attr:x", "=", "x"), ("option:0", "=", "lemon")), strict, is_seed=True)
    seed = _node((("attr:x", "=", "x"),), relaxed)
    edge = Edge(root.intent_id, seed.intent_id, Operator.RELAXATION, {}, Provenance.REAL, True)
    findings = battery.check_structure(_tree(root, [seed], [edge]))
    assert not [f for f in findings if f.check == "monotonic"]


def test_flags_dangling_edge():
    root = _node((("a", "=", 1),), GroundTruth.rowset([["x"]]), is_seed=True)
    seed = _node((("a", "=", 1), ("b", "=", 2)), GroundTruth.rowset([["x"]]))
    edge = Edge(root.intent_id, "nonexistent", Operator.REFINEMENT, {}, Provenance.REAL, True)
    checks = {f.check for f in battery.check_structure(_tree(root, [seed], [edge]))}
    assert "dangling_edge" in checks


def test_flags_root_that_is_not_the_seed():
    root = _node((("a", "=", 1),), GroundTruth.rowset([["x"]]))          # is_seed=False
    child = _node((("a", "=", 1), ("b", "=", 2)), GroundTruth.rowset([["x"]]))
    checks = {f.check for f in battery.check_structure(_tree(root, [child], []))}
    assert "one_seed" in checks


def test_flags_a_child_that_duplicates_the_seed():
    root = _node((("a", "=", 1),), GroundTruth.rowset([["x"]]), is_seed=True)
    child = _node((("a", "=", 1), ("b", "=", 2)), GroundTruth.rowset([["x"]]), is_seed=True)
    checks = {f.check for f in battery.check_structure(_tree(root, [child], []))}
    assert "one_seed" in checks
