"""Trajectory -> numbers. Every metric is a set or count over the log; nothing is judged.

Two definitions worth reading before using them:

* **intent recovery** is computable only because the state machine records every reveal, so
  "did the agent get the withheld detail out of the user before committing" is a set
  operation rather than an interpretation of its words.
* **over-reaction** needs an episode with at least one *rejected* proposal: a correct
  proposal ends the episode, so there is no "pending" proposal to abandon.  It is defined as
  "proposal n satisfied the pre-shift node, a non-moving shift happened, and proposal n+1
  differs" -- the counterpart to premature action, and the reason non-moving edges are kept.

The agent's ``Predicted user question`` is logged but never scored: grading free text needs
a judge, which the design forbids.
"""

from __future__ import annotations

from dataclasses import dataclass, field

UNDEFINED = None


@dataclass
class EpisodeMetrics:
    graph_id: str
    adapter: str
    persona: str
    strategy_id: str | None
    outcome: str
    success: bool
    turns: int
    patience_spent: int
    hidden_slots: tuple[str, ...] = ()
    recovered_slots: tuple[str, ...] = ()
    intent_recovery: float | None = None
    premature_action: bool | None = None
    n_shifts: int = 0
    shift_moved_answer: bool | None = None
    over_reaction: bool | None = None
    post_shift_latency: int | None = None
    n_proposals: int = 0
    n_malformed: int = 0
    provenance: str | None = None
    is_write_graph: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = dict(vars(self))
        d["hidden_slots"] = list(self.hidden_slots)
        d["recovered_slots"] = list(self.recovered_slots)
        return d


def score(traj) -> EpisodeMetrics:
    """Compute every metric from a trajectory dict (or Trajectory)."""
    data = traj if isinstance(traj, dict) else traj.to_dict()
    head, turns = data["header"], data["turns"]
    spec = head.get("perturbation") or {}
    hidden = tuple(spec.get("hidden_slots") or ())
    is_write = bool(head.get("is_write_graph"))
    outcome = data["outcome"]

    revealed_before_first_proposal: set[str] = set()
    revealed_all: set[str] = set()
    first_proposal_turn = None
    mutating_before_recovery = False
    shifts = []
    proposals = []
    malformed = 0

    for rec in turns:
        kind = (rec.get("action") or {}).get("kind")
        if rec.get("shift"):
            shifts.append({"turn": rec["turn"], **rec["shift"]})

        if kind == "ASK":
            for r in rec.get("reveals") or []:
                if not r.get("declined"):
                    revealed_all.add(r["slot"])
                    if first_proposal_turn is None:
                        revealed_before_first_proposal.add(r["slot"])
        elif kind == "ACT":
            if rec.get("act_is_mutating") and not set(hidden) <= revealed_all:
                mutating_before_recovery = True
        elif kind == "PROPOSE":
            acc = rec.get("acceptance") or {}
            proposals.append({"turn": rec["turn"], "node": rec.get("node"),
                              "ok": bool(acc.get("ok")),
                              "raw": (acc.get("proposal") or "")})
            if first_proposal_turn is None:
                first_proposal_turn = rec["turn"]
        elif kind == "MALFORMED":
            malformed += 1

    recovery = (len(revealed_before_first_proposal & set(hidden)) / len(hidden)
                if hidden else UNDEFINED)

    # --- over-reaction: needs a rejected proposal straddling a non-moving shift ---
    over = UNDEFINED
    if shifts and len(proposals) >= 2:
        unmoved = [s for s in shifts if s.get("gt_moved") is False]
        if unmoved:
            t = unmoved[0]["turn"]
            before = [p for p in proposals if p["turn"] < t]
            after = [p for p in proposals if p["turn"] > t]
            if before and after:
                over = before[-1]["raw"] != after[0]["raw"]

    # --- how long after a shift did the agent land a valid answer? ---
    latency = UNDEFINED
    if shifts:
        t = shifts[0]["turn"]
        ok_after = [p["turn"] for p in proposals if p["turn"] >= t and p["ok"]]
        latency = (ok_after[0] - t) if ok_after else UNDEFINED

    provenance = shifts[0].get("provenance") if shifts else UNDEFINED
    moved = shifts[0].get("gt_moved") if shifts else UNDEFINED
    patience = 0
    if turns:
        patience = (turns[0].get("patience_before", 0) - turns[-1].get("patience_after", 0))

    return EpisodeMetrics(
        # Legacy trajectories key this as "tree_id".
        graph_id=head.get("graph_id") or head.get("tree_id", ""),
        adapter=head.get("adapter", ""),
        persona=head.get("persona", ""), strategy_id=head.get("strategy_id"),
        outcome=outcome, success=(outcome == "SUCCESS"), turns=len(turns),
        patience_spent=patience,
        hidden_slots=hidden, recovered_slots=tuple(sorted(revealed_before_first_proposal)),
        intent_recovery=recovery,
        # mutation is the task on a write graph, so "premature" is not meaningful there
        premature_action=(UNDEFINED if is_write else mutating_before_recovery),
        n_shifts=len(shifts), shift_moved_answer=moved, over_reaction=over,
        post_shift_latency=latency, n_proposals=len(proposals), n_malformed=malformed,
        provenance=provenance, is_write_graph=is_write,
    )


def aggregate(metrics: list[EpisodeMetrics]) -> dict:
    """Summary, reported separately for shifted / unshifted and REAL / SYNTHETIC."""
    if not metrics:
        return {"episodes": 0}

    def rate(items, pred):
        vals = [pred(m) for m in items if pred(m) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    def block(items):
        if not items:
            return {"episodes": 0}
        return {
            "episodes": len(items),
            "success_rate": rate(items, lambda m: m.success),
            "mean_turns": round(sum(m.turns for m in items) / len(items), 2),
            "intent_recovery": rate(items, lambda m: m.intent_recovery),
            "premature_action_rate": rate(items, lambda m: m.premature_action),
            "over_reaction_rate": rate(items, lambda m: m.over_reaction),
            "outcomes": {o: sum(1 for m in items if m.outcome == o)
                         for o in sorted({m.outcome for m in items})},
        }

    shifted = [m for m in metrics if m.n_shifts > 0]
    return {
        "all": block(metrics),
        "by_adapter": {a: block([m for m in metrics if m.adapter == a])
                       for a in sorted({m.adapter for m in metrics})},
        "by_persona": {p: block([m for m in metrics if m.persona == p])
                       for p in sorted({m.persona for m in metrics})},
        "by_strategy": {s: block([m for m in metrics if m.strategy_id == s])
                        for s in sorted({m.strategy_id or "?" for m in metrics})},
        "shifted": block(shifted),
        "unshifted": block([m for m in metrics if m.n_shifts == 0]),
        "shift_moved": block([m for m in shifted if m.shift_moved_answer]),
        "shift_unmoved": block([m for m in shifted if m.shift_moved_answer is False]),
        "by_provenance": {p: block([m for m in shifted if m.provenance == p])
                          for p in sorted({m.provenance or "?" for m in shifted})},
    }
