"""Operator classification by condition-set algebra.

This is the retrieval half of the pipeline (plan D10): given two records that live in the
same environment, decide mechanically whether one is a branch of the other, and which
kind.  No model, no judgement -- just set containment.

  cond(B) strictly contains cond(A)      -> B refines A
  cond(B) strictly contained in cond(A)  -> B relaxes A
  same slots and ops, some value differs -> B substitutes into A
  different base frame (same environment) -> B is a pivot from A
"""

from __future__ import annotations

from .models import Condition, Operator


def _slots_ops(conds: tuple[Condition, ...]) -> frozenset[tuple[str, str]]:
    return frozenset((s, o) for s, o, _ in conds)


def classify(
    a_conds: tuple[Condition, ...],
    b_conds: tuple[Condition, ...],
    a_base: dict,
    b_base: dict,
) -> Operator | None:
    """Relation of B to A, or None if B is not a usable branch of A.

    Callers must have already established that A and B share an environment; a pivot is
    a change of *goal*, never a change of world (D6).
    """
    if a_base != b_base:
        # Same environment, different frame -> the goal itself moved.
        return Operator.PIVOT

    sa, sb = set(a_conds), set(b_conds)
    if sa == sb:
        return None  # identical intent, not a branch
    if sb > sa:
        return Operator.REFINEMENT
    if sb < sa:
        return Operator.RELAXATION
    if len(sa) == len(sb) and _slots_ops(a_conds) == _slots_ops(b_conds):
        # same shape, at least one value differs (sa != sb was established above)
        return Operator.SUBSTITUTION
    return None  # overlapping but unrelated -- e.g. one condition swapped for another slot


def delta(
    a_conds: tuple[Condition, ...],
    b_conds: tuple[Condition, ...],
    operator: Operator,
    b_base: dict | None = None,
) -> dict:
    """Machine-readable description of the edit, for the edge record."""
    sa, sb = set(a_conds), set(b_conds)
    if operator is Operator.REFINEMENT:
        return {"added": sorted(list(c) for c in sb - sa)}
    if operator is Operator.RELAXATION:
        return {"removed": sorted(list(c) for c in sa - sb)}
    if operator is Operator.SUBSTITUTION:
        old = {(s, o): v for s, o, v in sa - sb}
        new = {(s, o): v for s, o, v in sb - sa}
        return {
            "changed": sorted(
                [list(k) + [old[k], new[k]] for k in old.keys() & new.keys()],
                key=str,
            )
        }
    return {"pivot_to": b_base or {}}


def classify_pairs(seeds) -> dict[str, list[tuple[str, Operator]]]:
    """All ordered branch relations among parseable seeds sharing one environment.

    Returns ``{seed.record_id: [(other_record_id, operator), ...]}``.
    """
    out: dict[str, list[tuple[str, Operator]]] = {s.record_id: [] for s in seeds}
    parseable = [s for s in seeds if s.parseable]
    for a in parseable:
        for b in parseable:
            if a.record_id == b.record_id:
                continue
            op = classify(a.conditions, b.conditions, a.base, b.base)
            if op is not None:
                out[a.record_id].append((b.record_id, op))
    return out
