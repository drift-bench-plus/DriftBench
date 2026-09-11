"""Admission gate and ranking (plan D7, D9).

One hard rule and one label:

  HARD   the branch's ground truth must be non-empty -- an intent nobody can satisfy is
         not a task.
  LABEL  gt_moved records whether the answer actually changed.  It is NOT a filter: a
         user changing their mind in a way that happens not to change the right answer is
         realistic, and training on only answer-moving shifts would teach an agent to
         always change course when the user speaks.  Ranking prefers moved branches, and
         a graph needs at least one of them to be worth emitting.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .models import Candidate, GroundTruth, Operator, Provenance

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class GateResult:
    candidate: Candidate
    ground_truth: GroundTruth | None
    admitted: bool
    reason: str
    gt_moved: bool = False


def default_gt_equal(a: GroundTruth, b: GroundTruth) -> bool:
    return a.hash == b.hash


def admit(
    candidate: Candidate,
    gt: GroundTruth | None,
    parent_gt: GroundTruth,
    *,
    gt_equal=default_gt_equal,
    error: str | None = None,
) -> GateResult:
    """Decide whether a candidate becomes a branch.

    ``gt_equal`` lets an adapter supply the benchmark's own comparison (e.g. DBBench's
    ``compare_results`` with its 1e-2 float tolerance) rather than raw hash equality.
    """
    if error is not None:
        return GateResult(candidate, None, False, f"execution_error: {error}")
    if gt is None:
        return GateResult(candidate, None, False, "no_ground_truth")
    if gt.is_empty:
        return GateResult(candidate, gt, False, "unfulfillable_empty_gt")
    moved = not gt_equal(gt, parent_gt)
    return GateResult(candidate, gt, True, "ok", gt_moved=moved)


def magnitude(gt: GroundTruth, parent_gt: GroundTruth) -> float:
    """How far the answer moved, per ground-truth kind (D9 tiebreaker).

    Set-valued kinds get Jaccard distance; kinds without an interior (a scalar, a state
    hash) can only be same-or-different.
    """
    if gt.kind != parent_gt.kind:
        return 1.0
    if gt.kind in ("rowset", "purchaseset"):
        a, b = set(gt.value), set(parent_gt.value)
        union = a | b
        return len(a ^ b) / len(union) if union else 0.0
    return 0.0 if gt.hash == parent_gt.hash else 1.0


_OPERATOR_ORDER = {
    Operator.REFINEMENT: 0,
    Operator.RELAXATION: 1,
    Operator.SUBSTITUTION: 2,
    Operator.PIVOT: 3,
}


def quota_for(branching: int) -> int | None:
    """Branches per operator, or None when a balanced layer is impossible.

    N=8 means 2 refinements, 2 relaxations, 2 substitutions, 2 pivots. A balanced layer is
    what makes the runtime's category-probability vector meaningful: if the graph held 6
    substitutions and 1 pivot, asking for "20% pivots" could not be honoured, and the
    realised operator mix would be dictated by whatever the corpus offered rather than by the
    hyperparameter.

    Returns None unless ``branching`` divides evenly by the four operators. Rounding down
    instead would break the cap -- at N=3 a quota of 1 across four operators yields 4
    branches -- and rounding to a ragged split (2/2/1/1) would quietly reintroduce the
    imbalance the quota exists to remove.
    """
    n = len(_OPERATOR_ORDER)
    if branching < n or branching % n:
        return None
    return branching // n


def rank_and_cap(results: list[GateResult], parent_gt: GroundTruth, branching: int,
                 *, balanced: bool = True, shortfall: dict | None = None) -> list[GateResult]:
    """Deterministic selection of branches (D9).

    Order within an operator: real records first, then answer-moving branches, then larger
    answer movement, then a stable hash tiebreak.

    With ``balanced`` (the default) each operator contributes exactly ``quota_for(branching)``
    branches and no more -- a category that cannot fill its quota is recorded in ``shortfall``
    and leaves the graph smaller, rather than being silently backfilled by whichever operator
    happens to be abundant. Backfilling is what the old round-robin did, and it produced
    layers like 4 substitutions + 2 pivots + 2 refinements + 0 relaxations that still
    reported as "8 branches".
    """
    admitted = [r for r in results if r.admitted]

    def sort_key(r: GateResult):
        return (
            0 if r.candidate.provenance is Provenance.REAL else 1,
            0 if r.gt_moved else 1,
            _OPERATOR_ORDER[r.candidate.operator],
            -magnitude(r.ground_truth, parent_gt),
            r.ground_truth.hash,
        )

    ordered = sorted(admitted, key=sort_key)
    buckets: dict[Operator, list[GateResult]] = {}
    for r in ordered:
        buckets.setdefault(r.candidate.operator, []).append(r)

    quota = quota_for(branching) if balanced else None
    if quota is None:
        if balanced:
            log.warning("branching=%d is not divisible by %d operators; falling back to "
                        "round-robin (no balanced quota possible)",
                        branching, len(_OPERATOR_ORDER))
        picked: list[GateResult] = []
        while len(picked) < branching and any(buckets.values()):
            for op in sorted(buckets, key=lambda o: _OPERATOR_ORDER[o]):
                if len(picked) >= branching:
                    break
                if buckets[op]:
                    picked.append(buckets[op].pop(0))
        return sorted(picked, key=sort_key)

    picked = []
    for op in sorted(_OPERATOR_ORDER, key=lambda o: _OPERATOR_ORDER[o]):
        available = buckets.get(op, [])
        picked.extend(available[:quota])
        if shortfall is not None and len(available) < quota:
            shortfall[op.value] = shortfall.get(op.value, 0) + (quota - len(available))
    return sorted(picked, key=sort_key)


def rejection_summary(results: list[GateResult]) -> dict[str, int]:
    """Reasons a candidate did not become a branch, keeping the exception type.

    The type matters: "execution_error" alone hides whether the pipeline hit a data
    quirk it should tolerate or a bug it should not.
    """
    out: dict[str, int] = {}
    for r in results:
        if not r.admitted:
            out[r.reason] = out.get(r.reason, 0) + 1
    return out
