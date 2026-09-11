"""End-to-end engine behaviour, exercised through the synthetic toybench adapter."""

import pytest

from intent_graph.adapters.toybench import ToyAdapter, ToyExecutor
from intent_graph.engine import (
    RunStats,
    enumerate_refinements,
    enumerate_relaxations,
    enumerate_substitutions,
    generate,
    soft_slots,
)
from intent_graph.models import Operator, Provenance, Seed

CONFIG = {"seed": 42, "depth": 1, "branching": 8, "min_gt_moved_edges": 1,
          # toybench cannot fill a 2/2/2/2 quota; the quota itself is
          # covered in test_classify_gate_rank.py
          "require_full_quota": False}


@pytest.fixture
def run():
    adapter, executor = ToyAdapter(), ToyExecutor()
    stats = RunStats()
    graphs = list(generate(adapter, executor, CONFIG, stats=stats))
    return graphs, stats


def test_validation_filters_bad_seeds(run):
    _, stats = run
    # t_bad_answer (wrong shipped answer), t_bad_empty (unsatisfiable), t_unparsed (no conditions)
    assert stats.seeds_loaded == 9
    assert stats.seeds_parseable == 8
    assert stats.seeds_validated == 6
    assert stats.validation_failures["mismatch"] == 3


def test_the_seed_is_the_root(run):
    """The user's true intent is the seed; scoring a vaguer intersection instead would
    let an agent that never recovered the withheld details still score full marks."""
    graphs, _ = run
    assert graphs
    for t in graphs:
        assert t.root.is_seed and t.root.source_record
        assert not [c for c in t.children if c.is_seed]


def test_single_environment_invariant_holds(run):
    graphs, _ = run
    for t in graphs:
        t.assert_single_environment()
        assert len({t.env_id}) == 1


def test_no_node_has_empty_ground_truth(run):
    graphs, _ = run
    for t in graphs:
        for node in (t.root, *t.children):
            assert not node.ground_truth.is_empty


def test_every_tree_has_at_least_one_moved_edge(run):
    graphs, _ = run
    for t in graphs:
        assert any(e.gt_moved for e in t.edges), "D7: a wholly unmoved graph is vacuous"


def test_unmoved_edges_are_kept_not_dropped(run):
    """D7: redundant-condition branches survive, labelled gt_moved=false."""
    graphs, stats = run
    assert stats.gt_moved.get("unmoved", 0) > 0
    assert any(not e.gt_moved for t in graphs for e in t.edges)


def test_real_branches_are_retrieved_and_preferred(run):
    graphs, stats = run
    assert stats.provenance[Provenance.REAL.value] > 0
    # toybench ships explicit refinement/relaxation/substitution/pivot siblings
    ops = {e.operator for t in graphs for e in t.edges if e.provenance is Provenance.REAL}
    assert Operator.PIVOT in ops


def test_pivots_stay_inside_the_environment(run):
    """D6: a pivot changes the goal, never the world."""
    graphs, _ = run
    for t in graphs:
        for e in t.edges:
            if e.operator is Operator.PIVOT:
                dst = next(c for c in t.children if c.intent_id == e.dst)
                assert dst.base != t.root.base       # different frame
                assert t.env_id == t.env_spec["env_id"]  # same environment


def test_branching_cap_respected():
    cfg = {**CONFIG, "branching": 3}
    graphs = list(generate(ToyAdapter(), ToyExecutor(), cfg))
    for t in graphs:
        assert len(t.children) <= 3


def test_determinism_same_config_same_ids():
    a = list(generate(ToyAdapter(), ToyExecutor(), CONFIG))
    b = list(generate(ToyAdapter(), ToyExecutor(), CONFIG))
    assert [t.graph_id for t in a] == [t.graph_id for t in b]
    assert [n.intent_id for t in a for n in t.children] == \
           [n.intent_id for t in b for n in t.children]


def test_config_change_changes_tree_ids():
    a = list(generate(ToyAdapter(), ToyExecutor(), CONFIG))
    b = list(generate(ToyAdapter(), ToyExecutor(), {**CONFIG, "seed": 7}))
    assert {t.graph_id for t in a}.isdisjoint({t.graph_id for t in b})


def test_depth_greater_than_one_is_explicitly_unimplemented():
    with pytest.raises(NotImplementedError, match="depth"):
        list(generate(ToyAdapter(), ToyExecutor(), {**CONFIG, "depth": 2}))


# ------------------------------------------------------------- enumeration units
def _seed(conds):
    return Seed("s", "toyenv", {"f": "shop"}, tuple(conds))


def test_enumerate_refinement_requires_some_but_not_all():
    witness = [{"color": "red", "in_stock": "yes"}, {"color": "blue", "in_stock": "yes"}]
    cands = enumerate_refinements(_seed([("size", "=", "small")]), witness, cap=10)
    added = {tuple(c.delta["added"][0]) for c in cands}
    assert ("color", "=", "red") in added      # held by some
    assert ("in_stock", "=", "yes") not in added  # held by all -> would not narrow


def test_enumerate_refinement_needs_at_least_two_witnesses():
    assert enumerate_refinements(_seed([("a", "=", 1)]), [{"color": "red"}], cap=10) == []


def test_enumerate_relaxation_never_empties_the_intent():
    assert enumerate_relaxations(_seed([("a", "=", 1)]), cap=10) == []
    assert len(enumerate_relaxations(_seed([("a", "=", 1), ("b", "=", 2)]), cap=10)) == 2


def test_enumerate_substitution_uses_environment_values():
    cands = enumerate_substitutions(
        _seed([("color", "=", "red")]), {"color": ["red", "blue", "green"]}, cap=10
    )
    values = {c.conditions[0][2] for c in cands}
    assert values == {"blue", "green"}  # never the value it already has


# ---------------------------------------------------- soft slots on the seed/root
def test_soft_slots_name_what_the_siblings_disagree_about():
    from intent_graph.models import Candidate

    seed = _seed([("color", "=", "red"), ("size", "=", "small")])
    kids = [
        Candidate(conditions=(("color", "=", "red"),), base=seed.base,
                  operator=Operator.RELAXATION, delta={}, provenance=Provenance.REAL),
        Candidate(conditions=(("color", "=", "red"), ("size", "=", "large")), base=seed.base,
                  operator=Operator.SUBSTITUTION, delta={}, provenance=Provenance.REAL),
    ]
    # both branches move `size`; nothing touches `color`
    assert soft_slots(seed, kids) == ("size",)


def test_soft_slots_ignore_pivots():
    """A pivot is a different goal, not disagreement about this one."""
    from intent_graph.models import Candidate

    seed = _seed([("color", "=", "red")])
    kids = [Candidate(conditions=(("material", "=", "linen"),), base={"f": "outlet"},
                      operator=Operator.PIVOT, delta={}, provenance=Provenance.REAL)]
    assert soft_slots(seed, kids) == ()
