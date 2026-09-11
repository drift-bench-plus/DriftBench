"""Axis B: moving the hidden intent, before the user speaks.

Graphs are depth 1, so repeated shifts need *sibling* transitions (root -> A, then A -> B).
Those are derived rather than stored: ``classify()`` already relates any two condition sets
in one environment, so the full transition graph is a pairwise pass over the graph's nodes.

Why the category is sampled *before* the edge: pivots are 47.5% of all legal transitions
(measured over 400 WebShop graphs, 33,768 ordered pairs), so uniform sampling over edges
would drown the other three operators.  The category-probability vector decouples how often
an operator fires from how many edges happen to have that shape.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from ..classify import classify
from ..classify import delta as edge_delta
from ..gate import default_gt_equal
from ..models import Operator, Provenance, Graph

log = logging.getLogger(__name__)
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9'\-]+")


@dataclass(frozen=True, slots=True)
class SiblingEdge:
    src: str
    dst: str
    operator: Operator
    delta: dict
    gt_moved: bool
    provenance: Provenance

    def to_dict(self) -> dict:
        return {"src": self.src, "dst": self.dst, "operator": self.operator.value,
                "delta": self.delta, "gt_moved": self.gt_moved,
                "provenance": self.provenance.value}


def transition_graph(graph: Graph, adapter=None) -> dict[str, dict[Operator, list[SiblingEdge]]]:
    """Every legal move between the graph's nodes, grouped by operator.

    ``gt_moved`` has to be recomputed here: ``classify`` returns only the operator, and the
    graph's stored ``gt_moved`` flags describe root-to-child edges, not sibling ones.
    """
    gt_equal = getattr(adapter, "gt_equal", default_gt_equal) if adapter else default_gt_equal
    nodes = {n.intent_id: n for n in (graph.root, *graph.children)}
    real_records = {n.intent_id for n in nodes.values() if n.source_record}

    # Built under its own name: the adapter hooks below take ``graph`` and must receive
    # the intent Graph itself (certificate stores key by ``graph.graph_id``), never the
    # half-built transition dict.
    out: dict[str, dict[Operator, list[SiblingEdge]]] = {nid: {} for nid in nodes}
    for src_id, src in nodes.items():
        for dst_id, dst in nodes.items():
            if src_id == dst_id:
                continue
            op = classify(src.conditions, dst.conditions, src.base, dst.base)
            if op is None:
                # tau2: an adapter may relate two goals the condition algebra cannot
                cls = getattr(adapter, "classify_shift", None) if adapter else None
                op = cls(src, dst) if cls is not None else None
            if op is None:
                continue
            # An adapter may require a shift edge to carry an EXECUTABLE certificate:
            # proof that the destination goal is non-vacuous and achievable in the world
            # the shift would create (whether the old answer still satisfies it -- moved --
            # is a label on the edge, not a gate). Without the hook every classified edge
            # is legal (the original behaviour).
            ok_fn = getattr(adapter, "shift_edge_ok", None) if adapter else None
            if ok_fn is not None and not ok_fn(graph, src, dst, op):
                continue
            default_moved = not gt_equal(src.ground_truth, dst.ground_truth)
            moved_fn = getattr(adapter, "shift_gt_moved", None) if adapter else None
            out[src_id].setdefault(op, []).append(SiblingEdge(
                src=src_id, dst=dst_id, operator=op,
                delta=edge_delta(src.conditions, dst.conditions, op, dst.base),
                # moved may come from EXECUTION (an adapter certificate) rather than from
                # GT-signature comparison: two repairs can differ as signatures while the
                # old one still satisfies the new goal, and P metrics condition on this.
                gt_moved=(moved_fn(graph, src, dst, default_moved)
                          if moved_fn is not None else default_moved),
                provenance=Provenance.REAL if dst_id in real_records else Provenance.SYNTHETIC,
            ))
    return out


class Traversal:
    """Decides whether and where the intent moves, from a seeded RNG."""

    def __init__(self, graph: Graph, config: dict, rng, adapter=None) -> None:
        rt = config["runtime"]
        self.adapter = adapter
        self.graph = transition_graph(graph, adapter)
        self.rng = rng
        self.p_shift = float(rt.get("p_shift", 0.0))
        # How many user-facing actions may roll for a shift. None = every one (the
        # original semantics, kept so earlier results remain reproducible); an int makes
        # exposure arm-independent -- see maybe_shift for the measurement that forced it.
        w = rt.get("shift_roll_window", None)
        self.window = None if w in (None, "", 0) else int(w)
        self.max_shifts = int(rt.get("max_shifts", 1))
        self.allow_revisit = bool(rt.get("allow_revisit", False))
        self.on_empty = str(rt.get("on_empty_category", "resample"))
        self.forbid_after_mutation = str(rt.get("shift_after_mutation", "forbid")) == "forbid"
        # Axis-B arm: hold the shift until the agent has proposed at least once. With
        # p_shift alone the shift can fire on turn 1, before the agent has committed to
        # anything -- so there is no stale answer to submit and STALE cannot be measured at
        # all. This is what makes a clean axis-B arm possible rather than incidental.
        self.after_first_proposal = bool(rt.get("shift_after_first_proposal", False))
        probs = rt.get("category_probs") or {}
        self.category_probs = {Operator(k): float(v) for k, v in probs.items() if float(v) > 0}
        self.shifts_used = 0
        self.visited: set[str] = set()

    # ------------------------------------------------------------------ policy
    def may_shift(self, current: str, *, mutated: bool, proposed: bool = True) -> bool:
        if self.shifts_used >= self.max_shifts:
            return False
        if mutated and self.forbid_after_mutation:
            return False
        if self.after_first_proposal and not proposed:
            return False
        return any(self._available(current).values())

    def _available(self, current: str) -> dict[Operator, list[SiblingEdge]]:
        """Usable edges per category, INCLUDING categories with none.

        Empty categories are kept deliberately. Filtering them out here is what made
        `on_empty_category` dead code: sampling could then never land on an empty category,
        so `skip` and `resample` behaved identically and the config knob did nothing. The
        vector is over categories, so the draw has to be able to land on an empty one.
        """
        out: dict[Operator, list[SiblingEdge]] = {}
        for op in self.category_probs:
            edges = self.graph.get(current, {}).get(op, [])
            out[op] = [e for e in edges if self.allow_revisit or e.dst not in self.visited]
        return out

    def maybe_shift(self, current: str, *, mutated: bool,
                    proposed: bool = True, action_index: int | None = None
                    ) -> SiblingEdge | None:
        """Roll for a shift; on success return the chosen edge (caller updates state).

        ``action_index`` is the 1-based count of user-facing actions so far. When
        ``runtime.shift_roll_window`` is set, only the first N of them may roll.

        WHY THE WINDOW EXISTS (measured 2026-08-12). The roll fires on every ASK/PROPOSE, so
        the number of rolls -- and therefore the chance an episode ever shifts -- is set by how
        much the agent interacts. Measured on one shard: exposure was 82% for the plain
        baseline, 78% and 70% for two clarification arms. Within each stratum the arms scored
        identically (shifted 29.3 / 30.8 / 28.6; unshifted 77.8 / 90.9 / 80.0), and the raw
        Success gap closed to the decimal from the mix alone: 0.82x29.3 + 0.18x77.8 = 38.0
        against 0.70x28.6 + 0.30x80.0 = 44.0. So a +6 point "improvement" was entirely the
        reward for being terse. Success then measures brevity rather than intent recovery,
        which is the opposite of what axis B is for. Capping the rolls at a fixed count makes
        exposure arm-independent, because every arm reaches the first few user actions.
        """
        if not self.may_shift(current, mutated=mutated, proposed=proposed):
            return None
        window = self.window
        if window is not None and action_index is not None and action_index > window:
            return None
        if self.rng.random() >= self.p_shift:
            return None
        return self.choose(current)

    def scheduled_fire(self, current: str, *, mutated: bool, proposed: bool = True,
                       forced_categories: tuple[Operator, ...] | None = None
                       ) -> SiblingEdge | None:
        """SCHEDULED mode (ruling 2026-08-14): the shift decision was drawn at episode
        start, so this call carries no roll and no window -- it fires the pending shift
        now if the gate allows, and the caller keeps the pending flag alive when the
        gate refuses (the shift waits for the next user exchange).

        ``forced_categories`` is the agent-caused channel (ruling 2026-08-19): the type was
        decided by what the agent said, so the draw is restricted to those categories --
        uniform among them, because deciding the SPECIFIC destination from language would
        be less principled than deciding the kind. Random within the kind, as always."""
        if not self.may_shift(current, mutated=mutated, proposed=proposed):
            return None
        return self.choose(current, forced_categories=forced_categories)

    def choose(self, current: str,
               forced_categories: tuple[Operator, ...] | None = None) -> SiblingEdge | None:
        """Category first, then an edge within it.

        The category is drawn from the probability vector over ALL configured categories, so
        a draw can land on one with no legal edge. `on_empty_category` decides what then:
        `resample` renormalises over the rest and draws again, `skip` means no shift happens
        this turn -- which is a real behavioural difference, since skipping preserves the
        vector's marginals while resampling redistributes them.

        With ``forced_categories`` the vector is replaced by a uniform draw over the forced
        set (agent-caused shifts: the evidence names the KIND of change, never the edge).
        """
        available = self._available(current)
        if not any(available.values()):
            return None
        if forced_categories:
            weights = {op: 1.0 for op in forced_categories if available.get(op)}
            if not weights:
                return None
        else:
            weights = {op: self.category_probs[op] for op in available}
        # Keep renormalising until the categories are EXHAUSTED, not twice.
        # `resample` is documented as "renormalise over the rest and draw again", but this
        # loop used to run at most twice, so with four categories it could only ever rule
        # out two of them. If the one populated category happened to be drawn third or
        # fourth, the shift was abandoned even though a legal edge was sitting right there.
        # Measured on tau2 retail: 32% of draws that HAD a legal edge returned nothing.
        # WebShop is unaffected -- with all four categories populated the first draw always
        # lands on a real edge.
        while weights:
            ops = sorted(weights, key=lambda o: o.value)
            picked = self.rng.choices(ops, weights=[weights[o] for o in ops], k=1)[0]
            edges = available.get(picked)
            if edges:
                edge = self.rng.choice(sorted(edges, key=lambda e: e.dst))
                self.shifts_used += 1
                self.visited.add(edge.dst)
                return edge
            if self.on_empty != "resample":
                return None
            # renormalise over what is left, rather than silently biasing the vector
            weights.pop(picked, None)
        return None

    def mark_visited(self, node_id: str) -> None:
        self.visited.add(node_id)

    # -------------------------------------------------- agent-caused shifts
    def trigger_categories(self, current: str, question_words: set[str]
                           ) -> tuple[tuple[Operator, ...], tuple[str, ...]]:
        """Which shift KINDS does this question license, and which slots matched?

        Matching is by SLOT, never by destination value (ruling 2026-08-19: "it's hardly
        possible to match the value. I recommend just matching the category"): the words a
        user could use for a requirement are matched against the agent's question, and a
        matching edge contributes its OPERATOR to the candidate set. PIVOT never triggers
        from evidence -- a pivot is the user's own change of goal, not a reaction
        (settled 2026-08-19; measured: pivot destinations barely appear in agent text,
        and when they do it is noise).
        """
        cats: set[Operator] = set()
        slots: set[str] = set()
        for op, edges in self._available(current).items():
            if op is Operator.PIVOT:
                continue
            for e in edges:
                for slot, value in _delta_slots(e):
                    if self._slot_words(slot, value) & question_words:
                        cats.add(op)
                        slots.add(slot)
        return tuple(sorted(cats, key=lambda o: o.value)), tuple(sorted(slots))

    def _slot_words(self, slot: str, value) -> set[str]:
        """Plain words a user or agent would use for this requirement.

        The adapter may override via ``slot_match_words`` (WebShop's ``slot_phrase`` is
        deliberately generic -- "a product feature you require" -- so it cannot serve here).
        Default: the slot name's tail plus the CURRENT value's words. The current value
        identifies the topic ("red" -> the colour option); it is not destination matching.
        """
        hook = getattr(self.adapter, "slot_match_words", None)
        if hook is not None:
            try:
                return {w.lower() for w in hook(slot, value)}
            except Exception:  # pragma: no cover - adapter-specific
                pass
        words = _words(slot.split(":")[-1]) | _words(value)
        # IDENTIFIERS ARE NOT TOPICS (2026-08-19). A requirement is named in words, never
        # in an id. Without this filter tau2 leaks order and reservation numbers into the
        # match set -- 'w5199551', 'hkeg34', 'hat039' -- so an agent that merely quotes the
        # order it is working on would trigger an intent shift. That is noise, and it fires
        # hardest for the most careful agents, which is exactly backwards.
        # Cost measured on 120 WebShop graphs: 33 words dropped, 30 of 721 edge-deltas lose
        # all their words; the casualties are SKU fragments ('multi-13124', 'solid3-navy').
        return {w for w in words if len(w) >= 3 and not any(c.isdigit() for c in w)}


def _words(text) -> set[str]:
    return {w.lower() for w in _WORD_RE.findall(str(text or ""))}


def _delta_slots(edge: SiblingEdge):
    """(slot, current-side value) pairs an edge's delta touches. For a SUBSTITUTION the
    OLD value is the current side; for a REFINEMENT the added value is the only handle
    (on WebShop an attribute's name IS its value, and that is how an agent would name it)."""
    d = edge.delta or {}
    for c in d.get("added", []) or []:
        if isinstance(c, (list, tuple)) and len(c) >= 3:
            yield c[0], c[2]
    for c in d.get("removed", []) or []:
        if isinstance(c, (list, tuple)) and len(c) >= 3:
            yield c[0], c[2]
    for ch in d.get("changed", []) or []:
        if isinstance(ch, (list, tuple)) and len(ch) >= 4:
            yield ch[0], ch[2]          # old value: the requirement as the user holds it


# --------------------------------------------------------------- announcements
def _say(adapter, slot: str) -> str:
    """Human phrasing for a slot, so an announcement never prints an internal identifier."""
    fn = getattr(adapter, "slot_phrase", None)
    if fn is not None:
        try:
            return fn(slot)
        except Exception:  # pragma: no cover - adapter-specific
            pass
    return slot.split(":")[-1]


def announce(edge: SiblingEdge, adapter=None) -> str:
    """What the user says when their intent moves.

    Templated by default.  The PIVOT case needs care: its delta is ``{"pivot_to": base}``
    with no old/new condition, and on WebShop that base carries the target ASIN and the
    full product name -- announcing it verbatim would hand the agent the answer.
    """
    d = edge.delta or {}
    if edge.operator is Operator.REFINEMENT:
        added = ", ".join(_fmt(c, adapter) for c in d.get("added", []))
        return f"Also, it needs to be {added}." if added else "There's one more requirement."
    if edge.operator is Operator.RELAXATION:
        removed = d.get("removed", [])
        if removed:
            what = ", ".join(_say(adapter, c[0]) for c in removed)
            return f"Actually, {what} doesn't matter after all."
        return "Actually, one of those requirements doesn't matter."
    if edge.operator is Operator.SUBSTITUTION:
        changes = d.get("changed", [])
        if changes:
            old, new = changes[0][2], changes[0][3]
            return f"Actually, make it {new} instead of {old}."
        return "Actually, I need to change one of those details."
    # PIVOT
    describe = getattr(adapter, "describe_pivot", None)
    if describe is not None:
        return f"Forget that — {describe(d.get('pivot_to') or {})}"
    return "Forget that — I need something different, let me explain."


def _fmt(cond, adapter=None) -> str:
    """A refinement states a NEW value, so naming it is intended -- but the identifier is
    not. WebShop attribute names are their own values, which is why this prints the value."""
    slot, _op, value = cond[0], cond[1], cond[2]
    if slot.startswith("attr:"):
        return str(value)
    return f"{_say(adapter, slot)} {value}"
