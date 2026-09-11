"""Acceptance: the executable "okay".

"The user says okay" must be a verifier call, not an opinion.  The proposal is accepted iff
it satisfies the *current* node, decided by the benchmark's own comparison -- so the
simulator's mood cannot decide whether a task was solved, and a correct-but-stale answer is
rejected after the intent moves.

This module is a thin dispatcher.  The real logic lives on each adapter as an optional
``accepts()`` method, probed with ``getattr`` the way ``gt_equal`` and ``gt_extensional``
already are (adding it to the Protocol would break toybench).
"""

from __future__ import annotations

import json
import logging
from typing import Any

from ..models import GroundTruth, Node

log = logging.getLogger(__name__)


def default_parse_proposal(raw: Any) -> Any:
    """Decode an agent's submitted answer when the adapter has no opinion.

    Agents submit text (``Final Answer: ["i01"]``), while ground truth holds structured
    rows.  Without this, a correct answer is compared as a string against a decoded row and
    always rejected -- which shows up as the oracle agent being unable to win.
    """
    if not isinstance(raw, str):
        return raw
    text = raw.strip()
    try:
        value = json.loads(text)
        return value if isinstance(value, list) else [value]
    except Exception:
        parts = [t.strip().strip("'\"") for t in text.split(",") if t.strip()]
        return parts or [text]


class NotAcceptable(RuntimeError):
    """The adapter cannot evaluate proposals for this node kind."""


def accepts(adapter, proposal: Any, node: Node, session, *, executor=None) -> tuple[bool, str]:
    """Does ``proposal`` satisfy ``node``? Returns ``(ok, reason)``."""
    fn = getattr(adapter, "accepts", None)
    if fn is None:
        return generic_accepts(adapter, proposal, node, session)
    return fn(proposal, node, session, executor=executor)


def generic_accepts(adapter, proposal: Any, node: Node, session) -> tuple[bool, str]:
    """Fallback for adapters without a bespoke rule (used by toybench).

    Read-only kinds only: anything requiring execution of the agent's own statement must be
    implemented by the adapter, which knows how to do that without side effects.
    """
    gt: GroundTruth = node.ground_truth
    if gt.kind == "rowset":
        # the answer is the COMPLETE result set: flatten both sides and compare as sets,
        # matching how the SQL benchmarks compare answers
        want = {str(v) for row in gt.rows() for v in row}
        got = {str(v) for v in (proposal if isinstance(proposal, (list, tuple)) else [proposal])}
        ok = want == got
        return ok, "row_set_match" if ok else "row_set_mismatch"
    if gt.kind in ("purchaseset", "scalar"):
        ok = gt.contains(proposal)
        return ok, "in_ground_truth" if ok else "not_in_ground_truth"
    raise NotAcceptable(f"{adapter.name}: no acceptance rule for gt kind {gt.kind!r}")
