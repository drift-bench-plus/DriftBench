"""Does the perturbation actually do what its strategy claims?

Drift-Bench cannot answer this: it records which strategy was *requested*, never what
changed, so fidelity is unauditable.  Because our mask is structured, the literal reading is
a real intent — so we can execute it and check the answer moved the way the strategy
implies.  Withholding information must widen the answer; a false premise must make it
unsatisfiable; noise must leave it alone.

The subtlety that v1 of the plan got wrong: "widen" and "empty" only mean what you expect
for ground truth that *is* a set of matching objects.  Three of our four ground-truth kinds
are computed values, where a false premise shows up as *the value you get when nothing
matches* — a count of zero, or a table hash identical to the untouched table.  Hence one
rule per kind, and an explicit NOT_CHECKABLE verdict rather than a silent pass.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from ..models import GroundTruth
from . import strategies as st
from .perturb import PerturbationSpec

log = logging.getLogger(__name__)

OK = "ok"
NOT_CHECKABLE = "not_checkable"


def _as_set(gt: GroundTruth) -> set | None:
    return gt.monotonic_view()


def _zero_aggregate(value: Any) -> bool:
    """Is this the answer an aggregate gives when nothing matches?"""
    s = str(value).strip().strip('[]"')
    return s in {"0", "0.0", "", "None", "null", "NULL"}


def classify_signature(spec: PerturbationSpec, true_gt: GroundTruth, lit_gt: GroundTruth,
                       *, extensional: bool, pristine_hash: str | None = None) -> dict:
    """Compare the true answer with the answer the literal reading gives."""
    kind = true_gt.kind
    moved = lit_gt.hash != true_gt.hash
    # A composite mask carries several mask kinds; one rule has to govern. See
    # PerturbationSpec.dominant_kind: falsify > withhold/mark > noise.
    governing = spec.dominant_kind()
    detail = {"kind": kind, "extensional": extensional, "moved": moved,
              "true_card": true_gt.cardinality(), "lit_card": lit_gt.cardinality(),
              "governing_kind": governing, "mask_kinds": list(spec.kinds or (spec.mask_kind,))}

    # ---- noise / oblique: the goal is untouched, so the answer must not move ----
    if governing == st.NOISE:
        return {"ok": not moved, "verdict": OK if not moved else "noise_changed_answer", **detail}

    # ---- withheld / marked: less is stated, so the answer must widen ----
    if governing == st.WITHHOLD:
        if kind in ("rowset", "purchaseset") and extensional:
            a, b = _as_set(true_gt), _as_set(lit_gt)
            if a is None or b is None:
                return {"ok": True, "verdict": NOT_CHECKABLE, **detail}
            if b >= a and b != a:
                return {"ok": True, "verdict": OK, **detail}
            if b == a:
                return {"ok": False, "verdict": "withhold_did_not_widen", **detail}
            return {"ok": False, "verdict": "withhold_lost_answers", **detail}
        # computed answers (aggregates, shell scalars, state hashes): no subset relation
        # exists, so the observable requirement is simply that the answer changed
        return {"ok": moved, "verdict": OK if moved else "withhold_did_not_change_answer",
                **detail}

    # ---- falsified: the stated condition does not hold, so nothing should satisfy it ----
    if governing == st.FALSIFY:
        if kind in ("rowset", "purchaseset") and extensional:
            return {"ok": lit_gt.is_empty,
                    "verdict": OK if lit_gt.is_empty else "false_premise_still_satisfiable",
                    **detail}
        if kind == "rowset":            # an aggregate over no rows
            empty = lit_gt.is_empty or all(_zero_aggregate(v) for v in lit_gt.value)
            return {"ok": empty,
                    "verdict": OK if empty else "false_premise_aggregate_nonzero", **detail}
        if kind == "scalar":            # e.g. `... | wc -l` prints "0", which is not is_empty
            empty = lit_gt.is_empty or _zero_aggregate(lit_gt.value)
            return {"ok": empty, "verdict": OK if empty else "false_premise_scalar_nonzero",
                    **detail}
        if kind == "statehash":         # the statement matched nothing: state is untouched
            if pristine_hash is None:
                return {"ok": True, "verdict": NOT_CHECKABLE, **detail}
            same = str(lit_gt.value) == str(pristine_hash)
            return {"ok": same, "verdict": OK if same else "false_premise_changed_state",
                    "pristine": pristine_hash, **detail}

    return {"ok": True, "verdict": NOT_CHECKABLE, **detail}


def make_verifier(adapter, session, node, *, pristine_hash: str | None = None
                  ) -> Callable[[PerturbationSpec], dict]:
    """A closure the perturbation pipeline calls to test a candidate mask.

    Executing the literal reading is the whole point: it is the same machinery that produced
    the node's ground truth, so nothing here is judged or inferred.
    """
    true_gt = node.ground_truth
    extensional = bool(getattr(node, "gt_extensional", True))

    def verify(spec: PerturbationSpec) -> dict:
        reading = spec.literal_readings[0] if spec.literal_readings else ()
        if not reading and spec.dominant_kind() == st.WITHHOLD:
            # everything was withheld: the query would constrain nothing
            return {"ok": False, "verdict": "empty_literal_reading"}
        try:
            recipe = adapter.compile(node.base, reading)
            mutating = adapter.is_mutating(recipe)
            lit_gt = adapter.execute(recipe, session) if not mutating else _fresh(
                adapter, session, recipe)
        except Exception as exc:
            return {"ok": False, "verdict": f"literal_execution_error:{type(exc).__name__}"}
        out = classify_signature(spec, true_gt, lit_gt, extensional=extensional,
                                 pristine_hash=pristine_hash)
        out["literal_gt_hash"] = lit_gt.hash
        return out

    return verify


def _fresh(adapter, session, recipe):
    session.rematerialize()
    return adapter.execute(recipe, session)


def pristine_state_hash(adapter, session) -> str | None:
    """The state hash of an untouched environment (write tasks only)."""
    fn = getattr(adapter, "_state_hash", None)
    if fn is None:
        return None
    try:
        session.rematerialize()
        return fn(session)
    except Exception as exc:  # pragma: no cover - environment-specific
        log.debug("pristine hash unavailable: %s", exc)
        return None
