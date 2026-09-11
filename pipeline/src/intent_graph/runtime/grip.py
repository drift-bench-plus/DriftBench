"""GRIP: does the agent have a grip on what the user actually wants?

Four dimensions, all computed from the episode log plus the structured mask:

  G  Grounding    -- when it succeeded, had it extracted the intent, or inferred it?
  R  Recovery     -- how much of the hidden intent did it get out of the user?
  I  Inquiry      -- was the asking aimed, and did it know when NOT to ask?
  P  Persistence  -- when the intent moved, did it track without over-reacting?

Two design commitments worth stating, because both are easy to get wrong.

**Grounding labels, it does not gate.** A success where the agent never extracted the hidden
requirement is filed as INFERRED, not as a failure. Guessing what a user meant without being
told is a real skill and a protocol that punished it would train agents to interrogate instead
of to infer. The label exists so the trade is visible, not to dock anyone.

**Extraction excludes what the user volunteered.** Personas hand over slots unasked; crediting
the agent for those would inflate every Recovery number. This is why `Reveal.volunteered`
exists.

Nothing here calls a language model. The one model-mediated input is which slot a question was
about, decided by the simulated user during the episode (`SimulatedUser.select_slots`) and
already fixed in the log by the time GRIP reads it -- so Inquiry inherits that dependency while
G, R and P do not.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field
from typing import Any

UNDEFINED = None

# Grounding buckets. Exactly one per episode; the five shares sum to 1.
EARNED = "EARNED"                        # succeeded having extracted the whole hidden intent
INFERRED = "INFERRED"                    # succeeded without extracting it -- inference, not luck
RECOVERED_NOT_SOLVED = "RECOVERED_NOT_SOLVED"   # got the intent out, still failed the task
PREMATURE = "PREMATURE"                  # failed, and committed before asking anything
INCOMPLETE = "INCOMPLETE"                # failed, asked, never got the whole intent
# Its own bucket, not folded into EARNED. With nothing hidden, `hidden <= extracted` is
# trivially true, so these episodes would land in EARNED and make that share mean two
# different things at once -- "extracted everything hidden" and "there was nothing to
# extract". The EARNED share would then move with the strategy mix rather than with the
# agent. Restraint is the measure for this subset.
NO_HIDDEN_INTENT = "NO_HIDDEN_INTENT"
# A PIVOT shift moves the intent to a different product category; the trajectory records the
# edge but not the destination node's condition set, so the post-shift target is unknowable
# from the log alone. Rather than score such episodes against a stale target, they are
# excluded from the G/R denominators and counted -- reported, never silently dropped.
UNSCORED_PIVOT = "UNSCORED_PIVOT"
BUCKETS = (EARNED, INFERRED, RECOVERED_NOT_SOLVED, PREMATURE, INCOMPLETE, NO_HIDDEN_INTENT,
           UNSCORED_PIVOT)


@dataclass
class GripEpisode:
    """One episode's GRIP record."""

    graph_id: str
    persona: str
    strategy_id: str | None
    governing_kind: str | None
    outcome: str
    success: bool

    # G
    bucket: str = INCOMPLETE
    # R
    hidden: tuple[str, ...] = ()
    extracted_before_commit: tuple[str, ...] = ()
    extracted_ever: tuple[str, ...] = ()
    recovery: float | None = None
    volunteered_slots: tuple[str, ...] = ()
    # I
    n_asks: int = 0
    n_aimed_asks: int = 0
    n_unmapped_asks: int = 0
    aim: float | None = None
    restraint: bool | None = None
    patience_spent: int = 0
    # P
    n_shifts: int = 0
    shift_moved: bool | None = None
    stale: bool | None = None
    over_reaction: bool | None = None
    post_shift_latency: int | None = None
    # WHY the intent moved, not just that it did (2026-08-19). A shift is either SCHEDULED
    # (a patience threshold was crossed) or TRIGGERED (the agent's question named a live
    # requirement and the suggestibility roll passed). The episode writes
    # `record["shift"]["trigger"]`; until now nothing read it, so the two channels were
    # indistinguishable in every reported number -- which is exactly the distinction the
    # second layer exists to create.
    n_shifts_triggered: int = 0
    trigger_slots: tuple[str, ...] = ()
    trigger_categories: tuple[str, ...] = ()

    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = dict(vars(self))
        for k in ("hidden", "extracted_before_commit", "extracted_ever", "volunteered_slots",
                  "trigger_slots", "trigger_categories"):
            d[k] = list(d[k])
        return d


def score(traj, *, sample: dict | None = None) -> GripEpisode:
    """Compute GRIP for one trajectory.

    ``sample`` is the exported task instance, used for `governing_kind` and -- since the
    2026-08-21 any-channel ruling -- for the hidden slots' TRUE VALUES, which let world
    observations count as discovery (see below). Without it, discovery credit falls back
    to the user-spoken channel alone, exactly as before.
    """
    data = traj if isinstance(traj, dict) else traj.to_dict()
    head, turns = data["header"], data["turns"]
    spec = head.get("perturbation") or {}
    hidden = set(spec.get("hidden_slots") or ())
    success = data["outcome"] == "SUCCESS"
    governing = spec.get("governing_kind") or ((sample or {}).get("mask") or {}).get(
        "governing_kind")

    # ANY-CHANNEL DISCOVERY (the author's ruling, 2026-08-21): a hidden slot counts as
    # discovered when its true value reached the agent through the USER'S WORDS (reveal
    # records, as always) or through a WORLD OBSERVATION -- reading the true value out of
    # a tool result is discovering the intent nonetheless. One guard: an observation that
    # merely echoes a value the agent itself already asserted (it appeared in one of the
    # agent's own earlier or same-turn commands/proposals) is an echo, not discovery.
    _conds = {}
    for c in ((sample or {}).get("hidden_intent") or {}).get("conditions", []) or              (head.get("hidden_intent") or {}).get("conditions", []):
        if len(c) == 3 and c[0]:
            _conds[c[0]] = c[2]

    def _value_strings(v) -> list[str]:
        import json as _json
        if isinstance(v, (list, tuple)):
            out = []
            for x in v:
                out += _value_strings(x)
            return out
        if isinstance(v, str) and v.startswith("["):
            try:
                return _value_strings(_json.loads(v))
            except Exception:
                pass
        sv = str(v).strip()
        return [sv.lower()] if len(sv) >= 3 else []

    _true_vals = {sl: _value_strings(v) for sl, v in _conds.items()}
    _agent_said = ""                     # cumulative lower-cased agent command text

    extracted: set[str] = set()          # revealed, not declined, NOT volunteered
    volunteered: set[str] = set()
    reveal_turns: list[tuple[int, str]] = []      # (turn, slot) for every extraction
    ask_events: list[tuple[int, list[str]]] = []  # (turn, mapped slots) for every ASK
    extracted_at_commit: set[str] | None = None
    final_commit_turn: int | None = None
    asks = aimed = unmapped = 0
    proposals: list[dict] = []
    shifts: list[dict] = []

    for rec in turns:
        kind = (rec.get("action") or {}).get("kind")
        act = rec.get("action") or {}
        _cmd = f"{act.get('command') or ''} {act.get('proposal_raw') or ''} {act.get('raw') or ''}".lower()
        obs = rec.get("observation")
        if obs and _true_vals:
            low = str(obs).lower()
            for sl, vals in _true_vals.items():
                for v in vals:
                    if v in low and v not in _agent_said and v not in _cmd:
                        # the world spoke the true value before the agent ever wrote it
                        extracted.add(sl)
                        reveal_turns.append((rec["turn"], sl))
                        break
        _agent_said += " " + _cmd
        # A submission flushes every crossed-but-unfired shift, so ONE turn can carry
        # several hops. `episode._fire_scheduled` keeps them in order: `shifts_earlier`
        # holds the earlier hops and the singular `shift` key always holds the LAST one.
        # Reading only the singular key (the behaviour until 2026-08-19) silently dropped
        # the intermediate hops, which undercounted `n_shifts` and -- worse -- left
        # `target_at()` composing an incomplete edit list, so a slot added by a middle hop
        # and untouched by the last one never entered the target at all.
        for earlier in (rec.get("shifts_earlier") or ()):
            shifts.append({"turn": rec["turn"], **earlier})
        if rec.get("shift"):
            shifts.append({"turn": rec["turn"], **rec["shift"]})

        if kind == "ASK":
            asks += 1
            reveals = rec.get("reveals") or []
            if not reveals:
                unmapped += 1
            # aim is computed AFTER the loop, against the target as it stood at this turn --
            # scoring it here against the original mask double-charged intent shifts (a
            # post-shift re-ask of a stale slot read as "already extracted").
            ask_events.append((rec["turn"], [r["slot"] for r in reveals]))
            for r in reveals:
                if r.get("declined"):
                    continue
                if r.get("volunteered"):
                    volunteered.add(r["slot"])
                else:
                    extracted.add(r["slot"])
                    reveal_turns.append((rec["turn"], r["slot"]))
        elif kind == "PROPOSE":
            # user v2: a rejected proposal draws a reaction that may STATE requirements.
            # Rejection-sourced reveals ARE credited to Recovery (the author's ruling, upheld
            # 2026-08-13 against an earlier exclusion: "Rejection-sourced reveals is
            # acceptable, because rejection cost patience more" -- and for the regular B0,
            # whose ONLY information channel is the rejection, excluding them pinned
            # Recovery at a structural 0.0: an episode where the rejection reveals the
            # whole intent and the next buy wins must score 1). They never enter Aim:
            # a proposal is not a question, so it joins no ask_events.
            for r in rec.get("reveals") or []:
                if r.get("declined"):
                    continue
                if r.get("volunteered"):
                    volunteered.add(r["slot"])
                else:
                    extracted.add(r["slot"])
                    reveal_turns.append((rec["turn"], r["slot"]))
            acc = rec.get("acceptance") or {}
            proposals.append({"turn": rec["turn"], "ok": bool(acc.get("ok")),
                              "raw": acc.get("proposal") or ""})
            # Extraction state at the FINAL proposal -- updated on every proposal rather than
            # frozen at the first (the author's ruling, 2026-08-13). The first-proposal freeze
            # punished any strategy that uses a cheap early proposal and learns from the
            # rejection: measured on 2,700 hidden-intent episodes per arm, B2b's Recovery
            # read 47.19 under the freeze against 57.85 at the final commit (a 10.7-point
            # pure timing artefact), while B0, whose asks all precede its first buy, moved
            # only 66.78 -> 67.44. The commit that matters is the adjudicated one.
            extracted_at_commit = set(extracted)
            final_commit_turn = rec["turn"]

    # ---- the moving target -------------------------------------------------
    # The intent is TIME-INDEXED. Every shift edits the user's requirements -- whether or
    # not the winning answer moved -- because the user answers from the current node either
    # way. The target composes every shift in order: ADDED slots join, CHANGED slots make
    # pre-shift reveals stale, REMOVED slots leave.
    #
    # A PIVOT replaces the goal outright, so it does not edit the requirement set -- it
    # REPLACES it with the destination's. Episodes written from 2026-08-19 record
    # `dst_slots` on every shift, so a pivot is fully scoreable: the target becomes the new
    # node's requirements, none of which the agent has been told. Older logs have no
    # `dst_slots`; for those the target really is unknowable after a pivot, and the episode
    # is set aside exactly as before.
    events: list[tuple] = []          # ("edit", turn, add, rem, chg, after) | ("pivot", turn, slots)
    for sh in shifts:
        d = sh.get("delta") or {}
        if "pivot_to" in d:
            slots = sh.get("dst_slots")
            events.append(("pivot", sh["turn"], set(slots) if slots is not None else None))
        else:
            events.append(("edit", sh["turn"],
                           {c[0] for c in d.get("added", [])},
                           {c[0] for c in d.get("removed", [])},
                           {c[0] for c in d.get("changed", [])},
                           bool(sh.get("after_reply"))))
    # kept for the reveal-invalidation pass below
    shift_edits = [(e[1], e[2], e[3], e[4], e[5]) for e in events if e[0] == "edit"]
    # a pivot whose destination we cannot read still blinds the scorer
    blind_turn = next((t for k, t, s in
                       ((e[0], e[1], e[2]) for e in events if e[0] == "pivot")
                       if s is None), None)
    last_pivot_turn = max((e[1] for e in events if e[0] == "pivot"), default=None)

    def target_at(t: int) -> set[str] | None:
        """The requirements the agent still owes, as of turn t.

        None only when an unreadable pivot has fired (a pre-2026-08-19 log)."""
        if blind_turn is not None and t >= blind_turn:
            return None
        tgt = set(hidden)
        for ev in events:
            if ev[1] > t:
                break
            if ev[0] == "pivot":
                # the goal was replaced. Nothing the user said about the old goal is owed
                # any more, and nothing about the new one has been said yet.
                tgt = set(ev[2])
            else:
                _k, _v, add, rem, chg, _after = ev
                tgt = (tgt - rem) | add | chg
        return tgt

    def valid_at(t: int | None, *, before: int | None = None) -> set[str]:
        """Slots whose extraction is CURRENT at time t: a reveal is invalidated by any
        later shift (within the horizon) that changed or (re)introduced that slot.

        A pivot does NOT blanket-invalidate (removed 2026-08-21, ruling: "I don't think I
        have this rule" -- the blanket cut shipped with the 8/19 pivots-scoreable fix
        without a ruling, and it alone discarded 123 final-target reveals on one A2 cell).
        A pivot re-anchors the TARGET; a pre-pivot reveal simply stops mattering when its
        slot leaves the target, and keeps counting when the new goal owes the same slot.
        Known limit: same-named slots whose required VALUE changed across the pivot are
        credited by name; making that exact needs `dst_conditions` on future shift
        records.

        A reveal only counts while its slot is CURRENTLY OWED: an answer about a slot the
        goal no longer contains is not extraction of anything (this keeps the abandoned
        -goal guarantee the old blanket cut provided, without discarding still-owed
        slots)."""
        owed = target_at(t if t is not None else 10**9)
        out = set()
        for u, slot in reveal_turns:
            if owed is not None and slot not in owed:
                continue
            if t is not None and u > t:
                continue
            if before is not None and u >= before:
                continue
            horizon = [e for e in shift_edits if t is None or e[0] <= t]
            # a shift that fired AFTER the reply (scheduled mode) makes that same turn's
            # reveals stale too: the answer reflected the OLD intent
            if any((u < v or (u == v and after)) and (slot in add or slot in chg)
                   for v, add, rem, chg, after in horizon):
                continue
            out.add(slot)
        return out

    # ---- I: aim against the target AS IT STOOD when each question was asked --
    aimed = 0
    asks_scoreable = asks
    for t, slots in ask_events:
        tgt = target_at(t)
        if tgt is None:                       # post-pivot: unscorable, not unaimed
            asks_scoreable -= 1
            continue
        prior = valid_at(t, before=t)
        if any(sl in tgt and sl not in prior for sl in slots):
            aimed += 1

    end_turn = final_commit_turn if final_commit_turn is not None else (
        turns[-1]["turn"] if turns else 0)
    tgt_final = target_at(end_turn)
    pivot = tgt_final is None
    target = tgt_final or set()

    at_commit = valid_at(final_commit_turn)
    full_at_commit = bool(target) and target <= at_commit
    full_ever = bool(target) and target <= valid_at(None)

    # ---- G -----------------------------------------------------------------
    if pivot:
        bucket = UNSCORED_PIVOT
    elif not target:
        bucket = NO_HIDDEN_INTENT
    elif success:
        bucket = EARNED if full_at_commit else INFERRED
    elif full_ever:
        bucket = RECOVERED_NOT_SOLVED
    elif proposals and asks == 0:
        bucket = PREMATURE
    else:
        bucket = INCOMPLETE

    # ---- R -----------------------------------------------------------------
    # TASK-LEVEL (the author's ruling, 2026-08-13): 1.0 iff the WHOLE re-anchored target was
    # validly revealed by the final commit, else 0.0; the report's mean is then the share
    # of tasks whose true intent was revealed. (Previously: per-episode fraction of the
    # ORIGINAL hidden slots -- both differences were ruled bugs.)
    recovery = (1.0 if full_at_commit else 0.0) if (target and not pivot) else UNDEFINED

    # ---- I -----------------------------------------------------------------
    # Undefined, never 0.0, for an agent that never asked: a passive agent must not be able to
    # appear merely imprecise.
    aim = (aimed / asks_scoreable) if asks_scoreable else UNDEFINED
    # Restraint is only meaningful where there was nothing to ask about.
    restraint = (asks == 0) if not hidden else UNDEFINED
    patience = 0
    if turns:
        patience = turns[0].get("patience_before", 0) - turns[-1].get("patience_after", 0)

    # ---- P -----------------------------------------------------------------
    stale = over = latency = UNDEFINED
    if shifts:
        first = shifts[0]
        t = first["turn"]
        before = [p for p in proposals if p["turn"] < t]
        after = [p for p in proposals if p["turn"] >= t]
        if before and after:
            same = after[0]["raw"] == before[-1]["raw"]
            if first.get("gt_moved"):
                # it re-submitted the answer the shift invalidated
                stale = same
            elif first.get("gt_moved") is False:
                # the shift did NOT move the answer, so changing course discarded a valid one
                over = not same
        ok_after = [p["turn"] for p in proposals if p["turn"] >= t and p["ok"]]
        latency = (ok_after[0] - t) if ok_after else UNDEFINED

    # ---- which channel fired each shift ------------------------------------
    triggered = [sh for sh in shifts if sh.get("trigger")]
    trig_slots: set[str] = set()
    trig_cats: set[str] = set()
    for sh in triggered:
        tg = sh.get("trigger") or {}
        trig_slots.update(tg.get("slots") or ())
        trig_cats.update(tg.get("categories") or ())

    return GripEpisode(
        # Legacy trajectories (pre-2026-08-19 rename) key this as "tree_id".
        graph_id=head.get("graph_id") or head.get("tree_id", ""),
        persona=head.get("persona", ""),
        strategy_id=head.get("strategy_id"), governing_kind=governing,
        outcome=data["outcome"], success=success,
        bucket=bucket,
        hidden=tuple(sorted(target)) if not pivot else (),
        extracted_before_commit=tuple(sorted(at_commit)),
        extracted_ever=tuple(sorted(extracted)),
        recovery=recovery, volunteered_slots=tuple(sorted(volunteered)),
        n_asks=asks, n_aimed_asks=aimed, n_unmapped_asks=unmapped,
        aim=aim, restraint=restraint, patience_spent=patience,
        n_shifts=len(shifts), shift_moved=(shifts[0].get("gt_moved") if shifts else UNDEFINED),
        stale=stale, over_reaction=over, post_shift_latency=latency,
        n_shifts_triggered=len(triggered),
        trigger_slots=tuple(sorted(trig_slots)),
        trigger_categories=tuple(sorted(trig_cats)),
    )


def _mean(values):
    vals = [v for v in values if v is not None]
    return (sum(vals) / len(vals)) if vals else UNDEFINED


def _rate(values):
    vals = [v for v in values if v is not None]
    return (sum(1 for v in vals if v) / len(vals)) if vals else UNDEFINED


def report(episodes: list[GripEpisode]) -> dict:
    """The standard GRIP table.

    Reported as a vector, never as one composite. A composite hides which of the four
    dimensions broke, which is the only thing the protocol is for.
    """
    if not episodes:
        return {"n": 0}

    n = len(episodes)
    buckets = collections.Counter(e.bucket for e in episodes)
    with_hidden = [e for e in episodes if e.hidden]
    without_hidden = [e for e in episodes if not e.hidden]
    shifted = [e for e in episodes if e.n_shifts]

    out: dict[str, Any] = {
        "n": n,
        # G -- shares over ALL episodes, summing to 1
        "grounding": {b: round(buckets[b] / n, 4) for b in BUCKETS},
        # ...and over the subset where there was actually an intent to recover, which is the
        # interpretable view: it does not move when the strategy mix changes.
        "grounding_where_hidden": (
            {b: round(sum(1 for e in with_hidden if e.bucket == b) / len(with_hidden), 4)
             for b in BUCKETS if b != NO_HIDDEN_INTENT} if with_hidden else {}),
        "success_rate": round(sum(1 for e in episodes if e.success) / n, 4),
        "success_rate_where_hidden": (
            round(sum(1 for e in with_hidden if e.success) / len(with_hidden), 4)
            if with_hidden else UNDEFINED),
        # R
        "recovery": {
            "mean": _round(_mean(e.recovery for e in with_hidden)),
            "full": _round(_rate(bool(e.hidden) and set(e.hidden) <= set(e.extracted_before_commit)
                                 for e in with_hidden)),
            "n": len(with_hidden),
        },
        # I
        "inquiry": {
            "aim": _round(_mean(e.aim for e in episodes)),
            "restraint": _round(_rate(e.restraint for e in without_hidden)),
            "mean_asks": _round(_mean(float(e.n_asks) for e in episodes)),
            "silent_episodes": _round(sum(1 for e in episodes if e.n_asks == 0) / n),
            "unmapped_asks": _round(_mean(float(e.n_unmapped_asks) for e in episodes)),
            "patience_per_slot": _round(_patience_per_slot(with_hidden)),
        },
        # P
        "persistence": {
            "n_shifted": len(shifted),
            "stale": _round(_rate(e.stale for e in shifted)),
            "over_reaction": _round(_rate(e.over_reaction for e in shifted)),
            "latency": _round(_mean(float(e.post_shift_latency) for e in shifted
                                    if e.post_shift_latency is not None)),
            # The two shift channels, reported separately. `triggered_share` is over
            # SHIFTS, not over episodes: an episode may carry one of each. An arm that
            # never names a requirement in its questions should read ~0 here, and an arm
            # that interrogates a suggestible persona should read high -- if both read the
            # same, the trigger is not doing anything and the second layer is decorative.
            "n_shifts_total": sum(e.n_shifts for e in episodes),
            "n_shifts_triggered": sum(e.n_shifts_triggered for e in episodes),
            "triggered_share": _round(
                (sum(e.n_shifts_triggered for e in episodes) / sum(e.n_shifts for e in episodes))
                if sum(e.n_shifts for e in episodes) else UNDEFINED),
            "episodes_with_trigger": _round(
                sum(1 for e in episodes if e.n_shifts_triggered) / n),
            "mean_shifts_per_episode": _round(_mean(float(e.n_shifts) for e in episodes)),
            "trigger_categories": dict(collections.Counter(
                c for e in episodes for c in e.trigger_categories).most_common()),
        },
    }
    # Mandatory strata: an aggregate that hides these is not interpretable.
    out["by_governing_kind"] = _strata(episodes, lambda e: e.governing_kind or "?")
    out["by_persona"] = _strata(episodes, lambda e: e.persona or "?")
    out["by_strategy"] = _strata(episodes, lambda e: e.strategy_id or "?")
    return out


def _patience_per_slot(episodes):
    slots = sum(len(set(e.hidden) & set(e.extracted_before_commit)) for e in episodes)
    spent = sum(e.patience_spent for e in episodes)
    return (spent / slots) if slots else UNDEFINED


def _strata(episodes, key):
    groups: dict[str, list] = collections.defaultdict(list)
    for e in episodes:
        groups[key(e)].append(e)
    return {k: {"n": len(v),
                "success_rate": round(sum(1 for e in v if e.success) / len(v), 4),
                "earned": round(sum(1 for e in v if e.bucket == EARNED) / len(v), 4),
                "inferred": round(sum(1 for e in v if e.bucket == INFERRED) / len(v), 4),
                "recovery": _round(_mean(e.recovery for e in v)),
                "aim": _round(_mean(e.aim for e in v))}
            for k, v in sorted(groups.items())}


def _round(v):
    return round(v, 4) if isinstance(v, (int, float)) else v


__all__ = ["score", "report", "GripEpisode", "BUCKETS", "EARNED", "INFERRED",
           "RECOVERED_NOT_SOLVED", "PREMATURE", "INCOMPLETE", "NO_HIDDEN_INTENT"]


# ============================================================== GRIP v2 (2026-08-11)
# The eleven-metric protocol as finalized with ruling:
#   G: success, earned, inferred          R: identification, consistency (judge-based,
#   I: aim, recovery, patience_left          computed in runtime/judges.py from stored
#   P: staleness, reaction,                  transcripts -- never inside episodes)
#      post_shift_success (adopted into P 2026-08-12: success on the ~74% of episodes
#      whose answer was moved by a shift -- the unconditional companion to staleness,
#      which conditions on failures, and reaction, which conditions on successes)
# Everything here is arithmetic over stored trajectories; nothing calls a model.

def score_v2(traj) -> dict:
    """The per-episode quantities the v2 report aggregates."""
    e = score(traj)                      # reuse the mechanical v1 extraction
    data = traj if isinstance(traj, dict) else traj.to_dict()
    turns = data["turns"]

    patience_left = UNDEFINED
    for rec in reversed(turns):
        if "patience_after" in rec:
            patience_left = rec["patience_after"]
            break

    # P -- defined only around the LAST answer-moving shift
    moving = [r for r in turns if (r.get("shift") or {}).get("gt_moved")]
    stale = reaction = UNDEFINED
    post_shift_success = UNDEFINED
    if moving:
        last = moving[-1]["turn"]
        post_shift_success = e.success
        if not e.success:
            for rec in turns:
                chk = rec.get("stale_check")
                if chk is not None and "valid_under_old" in chk:
                    stale = bool(chk["valid_under_old"])
        else:
            accepted = [r["turn"] for r in turns
                        if (r.get("acceptance") or {}).get("ok") and r["turn"] >= last]
            if accepted:
                reaction = accepted[0] - last
    return {
        "persona": e.persona, "strategy_id": e.strategy_id, "outcome": e.outcome,
        "success": e.success, "bucket": e.bucket, "hidden": bool(e.hidden),
        "aim": e.aim, "recovery": e.recovery,
        "patience_left": patience_left,
        "had_moving_shift": bool(moving),
        "stale": stale, "reaction": reaction, "post_shift_success": post_shift_success,
    }


def report_v2(rows: list[dict]) -> dict:
    """Aggregate score_v2 rows into the ten metrics. Undefined is never zero: every
    ratio conditions on its enabling event and reports that n."""
    n = len(rows)
    if not n:
        return {"n": 0}
    hidden = [r for r in rows if r["hidden"]]
    won_hidden = [r for r in hidden if r["success"]]
    asked = [r for r in rows if r["aim"] is not UNDEFINED and r["aim"] != "undefined"]
    pat = [r["patience_left"] for r in rows
           if r["patience_left"] not in (UNDEFINED, "undefined")]
    stale_dom = [r for r in rows if r["stale"] not in (UNDEFINED, "undefined")]
    react = sorted(r["reaction"] for r in rows
                   if r["reaction"] not in (UNDEFINED, "undefined"))
    shifted = [r for r in rows if r["had_moving_shift"]]

    def _m(vals):
        vals = [v for v in vals if v not in (UNDEFINED, "undefined") and v is not None]
        return round(sum(vals) / len(vals), 4) if vals else UNDEFINED

    return {
        "n": n,
        # G
        "success": round(sum(r["success"] for r in rows) / n, 4),
        "earned": (round(sum(1 for r in hidden if r["bucket"] == EARNED) / len(hidden), 4)
                   if hidden else UNDEFINED),
        "inferred": (round(sum(1 for r in won_hidden if r["bucket"] == INFERRED)
                           / len(hidden), 4) if hidden else UNDEFINED),
        "n_hidden": len(hidden),
        # I
        "aim": _m([r["aim"] for r in asked]),
        "n_asked": len(asked),
        "recovery": _m([r["recovery"] for r in hidden]),
        "patience_left": _m(pat),
        # P
        "staleness": (_m([1.0 if r["stale"] else 0.0 for r in stale_dom])
                      if stale_dom else UNDEFINED),
        "n_stale_dom": len(stale_dom),
        # mean per the author (2026-08-11); the median stays available in the record
        "reaction": (round(sum(react) / len(react), 2) if react else UNDEFINED),
        "reaction_median": (react[len(react) // 2] if react else UNDEFINED),
        "n_reaction": len(react),
        "post_shift_success": _m([r["post_shift_success"] for r in shifted]),
        "n_shifted": len(shifted),
    }
