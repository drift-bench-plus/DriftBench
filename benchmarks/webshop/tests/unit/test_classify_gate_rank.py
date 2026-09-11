"""Operator classification (retrieval), admission gate, and ranking."""


from collections import Counter

from intent_graph.classify import classify, delta
from intent_graph.gate import admit, magnitude, quota_for, rank_and_cap
from intent_graph.models import Candidate, GroundTruth, Operator, Provenance

FRAME = {"f": "shop"}
OTHER = {"f": "outlet"}

A = (("color", "=", "red"), ("size", "=", "small"))


def test_classify_refinement():
    b = (*A, ("material", "=", "cotton"))
    assert classify(A, b, FRAME, FRAME) is Operator.REFINEMENT


def test_classify_relaxation():
    b = (("color", "=", "red"),)
    assert classify(A, b, FRAME, FRAME) is Operator.RELAXATION


def test_classify_substitution():
    b = (("color", "=", "blue"), ("size", "=", "small"))
    assert classify(A, b, FRAME, FRAME) is Operator.SUBSTITUTION


def test_classify_pivot_is_frame_change():
    assert classify(A, A, FRAME, OTHER) is Operator.PIVOT


def test_classify_identical_is_not_a_branch():
    assert classify(A, A, FRAME, FRAME) is None


def test_classify_unrelated_swap_is_not_a_branch():
    # same size, but one condition traded for a different slot: neither containment nor
    # substitution -- must not be silently mislabelled
    b = (("color", "=", "red"), ("material", "=", "wool"))
    assert classify(A, b, FRAME, FRAME) is None


def test_delta_records_the_edit():
    b = (*A, ("material", "=", "cotton"))
    assert delta(A, b, Operator.REFINEMENT) == {"added": [["material", "=", "cotton"]]}
    assert delta(A, (A[0],), Operator.RELAXATION) == {"removed": [["size", "=", "small"]]}
    sub = (("color", "=", "blue"), ("size", "=", "small"))
    assert delta(A, sub, Operator.SUBSTITUTION) == {"changed": [["color", "=", "red", "blue"]]}


# ------------------------------------------------------------------ admission gate
def _cand(op=Operator.REFINEMENT, prov=Provenance.SYNTHETIC):
    return Candidate(conditions=A, base=FRAME, operator=op, delta={}, provenance=prov)


_OPS = (Operator.REFINEMENT, Operator.RELAXATION, Operator.SUBSTITUTION,
        Operator.PIVOT)
PARENT = GroundTruth.rowset([["a"], ["b"]])


def test_gate_rejects_empty_gt():
    res = admit(_cand(), GroundTruth.rowset([]), PARENT)
    assert not res.admitted and res.reason == "unfulfillable_empty_gt"


def test_gate_rejects_execution_error():
    res = admit(_cand(), None, PARENT, error="OperationalError")
    assert not res.admitted and res.reason.startswith("execution_error")


def test_gate_admits_and_labels_moved():
    res = admit(_cand(), GroundTruth.rowset([["a"]]), PARENT)
    assert res.admitted and res.gt_moved


def test_gate_admits_unmoved_branch_without_rejecting_it():
    """D7: an intent change whose answer happens not to move is kept, just labelled."""
    res = admit(_cand(), GroundTruth.rowset([["a"], ["b"]]), PARENT)
    assert res.admitted and not res.gt_moved


def test_gate_honours_adapter_supplied_equality():
    """north-star #2: the benchmark's own comparison wins over our hashing."""
    near = GroundTruth.rowset([["4.109"]])
    parent = GroundTruth.rowset([["4.11"]])

    def tolerant(a, b):
        return abs(float(a.value[0].strip('[]"')) - float(b.value[0].strip('[]"'))) < 1e-2

    assert admit(_cand(), near, parent).gt_moved is True
    assert admit(_cand(), near, parent, gt_equal=tolerant).gt_moved is False


# ------------------------------------------------------------------------ ranking
def test_magnitude_per_kind():
    a, b = GroundTruth.rowset([["x"]]), GroundTruth.rowset([["y"]])
    assert magnitude(a, b) == 1.0
    assert magnitude(a, a) == 0.0
    assert magnitude(GroundTruth.statehash("a"), GroundTruth.statehash("b")) == 1.0
    assert magnitude(GroundTruth.scalar("5"), GroundTruth.scalar("5")) == 0.0


def test_rank_prefers_real_then_moved():
    gt_moved = GroundTruth.rowset([["z"]])
    real = admit(_cand(prov=Provenance.REAL), gt_moved, PARENT)
    synth = admit(_cand(prov=Provenance.SYNTHETIC), gt_moved, PARENT)
    unmoved = admit(_cand(prov=Provenance.SYNTHETIC), PARENT, PARENT)
    picked = rank_and_cap([unmoved, synth, real], PARENT, 3)
    assert picked[0].candidate.provenance is Provenance.REAL
    assert picked[-1].gt_moved is False


def test_balanced_quota_gives_two_of_each_at_branching_eight():
    """N=8 means 2 refinements, 2 relaxations, 2 substitutions, 2 pivots -- exactly."""
    results = []
    for op in (Operator.REFINEMENT, Operator.RELAXATION, Operator.SUBSTITUTION,
               Operator.PIVOT):
        results += [admit(_cand(op=op), GroundTruth.rowset([[f"{op.value}{i}"]]), PARENT)
                    for i in range(6)]
    picked = rank_and_cap(results, PARENT, 8)
    counts = Counter(r.candidate.operator for r in picked)
    assert len(picked) == 8
    assert all(counts[op] == 2 for op in _OPS), counts


def test_balanced_quota_does_not_backfill_a_short_category():
    """20 substitutions must NOT become 8 branches: the other three quotas are unfillable,
    and backfilling would report a balanced layer that is nothing of the kind."""
    results = [admit(_cand(op=Operator.SUBSTITUTION), GroundTruth.rowset([[f"r{i}"]]), PARENT)
               for i in range(20)]
    shortfall = {}
    picked = rank_and_cap(results, PARENT, 8, shortfall=shortfall)
    assert len(picked) == 2
    assert {r.candidate.operator for r in picked} == {Operator.SUBSTITUTION}
    assert shortfall == {"REFINEMENT": 2, "RELAXATION": 2, "PIVOT": 2}


def test_unbalanced_mode_still_caps_at_branching():
    results = [admit(_cand(), GroundTruth.rowset([[f"r{i}"]]), PARENT) for i in range(20)]
    assert len(rank_and_cap(results, PARENT, 8, balanced=False)) == 8


def test_branching_not_divisible_by_four_falls_back_and_respects_the_cap():
    """A quota is impossible at N=3; rounding down would emit 4 branches for a cap of 3."""
    assert quota_for(3) is None and quota_for(8) == 2 and quota_for(4) == 1
    results = []
    for op in _OPS:
        results += [admit(_cand(op=op), GroundTruth.rowset([[f"{op.value}{i}"]]), PARENT)
                    for i in range(3)]
    assert len(rank_and_cap(results, PARENT, 3)) == 3


def test_rank_surfaces_a_lone_pivot_among_many_substitutions():
    """The reason the quota exists: 20 substitutions must not crowd out the single pivot."""
    subs = [admit(_cand(op=Operator.SUBSTITUTION), GroundTruth.rowset([[f"s{i}"]]), PARENT)
            for i in range(20)]
    pivot = admit(_cand(op=Operator.PIVOT), GroundTruth.rowset([["p"]]), PARENT)
    picked = rank_and_cap([*subs, pivot], PARENT, 8)
    assert Operator.PIVOT in {r.candidate.operator for r in picked}


def test_rank_is_deterministic():
    results = [admit(_cand(), GroundTruth.rowset([[f"r{i}"]]), PARENT) for i in range(12)]
    first = [r.ground_truth.hash for r in rank_and_cap(results, PARENT, 8)]
    second = [r.ground_truth.hash for r in rank_and_cap(list(reversed(results)), PARENT, 8)]
    assert first == second
