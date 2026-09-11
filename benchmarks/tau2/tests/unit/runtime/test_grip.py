"""GRIP: the four dimensions, and the ablation signatures that justify them.

The protocol's claim is diagnostic: a bad score should NAME the fault. These tests build the
four archetypal agents as synthetic trajectories and assert each produces a distinct signature.
If two archetypes ever score the same, the protocol has stopped doing its job.
"""

import pytest

from intent_graph.runtime.grip import (
    BUCKETS,
    EARNED,
    INCOMPLETE,
    INFERRED,
    NO_HIDDEN_INTENT,
    PREMATURE,
    RECOVERED_NOT_SOLVED,
    report,
    score,
)

HIDDEN = ["attr:wool", "option:0"]


def traj(turns, *, outcome="SUCCESS", hidden=HIDDEN, governing="withhold"):
    return {
        "header": {"graph_id": "t1", "persona": "rational", "strategy_id": "s",
                   "perturbation": {"hidden_slots": list(hidden),
                                    "governing_kind": governing}},
        "turns": turns, "outcome": outcome, "final_node": "n", "error": None,
    }


def ask(turn, slots, *, declined=False, volunteered=False, patience=(10, 9)):
    return {"turn": turn, "action": {"kind": "ASK"}, "patience_before": patience[0],
            "patience_after": patience[1],
            "reveals": [{"slot": s, "value": "v", "was_repeat": False,
                         "declined": declined, "volunteered": volunteered} for s in slots]}


def propose(turn, raw, ok, *, patience=(9, 9)):
    return {"turn": turn, "action": {"kind": "PROPOSE"}, "patience_before": patience[0],
            "patience_after": patience[1],
            "acceptance": {"proposal": raw, "ok": ok, "reason": "r"}}


def shift(rec, *, moved=True):
    rec = dict(rec)
    rec["shift"] = {"operator": "SUBSTITUTION", "gt_moved": moved, "dst": "n2",
                    "provenance": "REAL", "announcement": "changed my mind"}
    return rec


# --------------------------------------------------------------- G buckets
def test_earned_requires_the_whole_intent_before_committing():
    e = score(traj([ask(1, ["attr:wool"]), ask(2, ["option:0"]), propose(3, "A", True)]))
    assert e.bucket == EARNED
    assert e.recovery == 1.0


def test_success_without_extraction_is_labelled_inferred_not_penalised():
    """Guessing an unstated requirement is a skill. It gets a LABEL, not a failure -- a
    protocol that punished it would train agents to interrogate instead of infer."""
    e = score(traj([propose(1, "A", True)]))
    assert e.bucket == INFERRED
    assert e.success is True
    assert e.recovery == 0.0


def test_partial_extraction_before_commit_is_still_inferred():
    e = score(traj([ask(1, ["attr:wool"]), propose(2, "A", True)]))
    assert e.bucket == INFERRED
    # Recovery is TASK-LEVEL (the author's ruling 2026-08-13): the true intent either was revealed
    # in full or it was not. One of two hidden slots is not a half-revealed intent.
    assert e.recovery == 0.0


def test_recovered_but_failed_is_its_own_bucket():
    """The bucket that changes what the research program works on: the intent job is DONE and
    the task is the blocker. Merging it with INCOMPLETE would send us after comprehension."""
    e = score(traj([ask(1, HIDDEN), propose(2, "A", False)], outcome="TURNS_EXCEEDED"))
    assert e.bucket == RECOVERED_NOT_SOLVED


def test_premature_is_failing_after_committing_with_no_question():
    e = score(traj([propose(1, "A", False)], outcome="EXHAUSTED"))
    assert e.bucket == PREMATURE


def test_incomplete_is_asked_but_never_got_it_all():
    e = score(traj([ask(1, ["attr:wool"]), propose(2, "A", False)], outcome="TURNS_EXCEEDED"))
    assert e.bucket == INCOMPLETE


def test_buckets_are_exhaustive_and_shares_sum_to_one():
    eps = [
        score(traj([ask(1, HIDDEN), propose(2, "A", True)])),
        score(traj([propose(1, "A", True)])),
        score(traj([ask(1, HIDDEN), propose(2, "A", False)], outcome="TURNS_EXCEEDED")),
        score(traj([propose(1, "A", False)], outcome="EXHAUSTED")),
        score(traj([ask(1, ["attr:wool"]), propose(2, "A", False)], outcome="TURNS_EXCEEDED")),
    ]
    r = report(eps)
    assert set(r["grounding"]) == set(BUCKETS)
    assert abs(sum(r["grounding"].values()) - 1.0) < 1e-9


# ------------------------------------------------------------------- R
def test_volunteered_slots_do_not_count_as_recovered():
    """A chatty persona handing over a slot is not the agent recovering it. Crediting these
    would inflate every recovery number, which is why Reveal.volunteered exists."""
    e = score(traj([ask(1, ["attr:wool"]),
                    ask(2, ["option:0"], volunteered=True),
                    propose(3, "A", True)]))
    # task-level: the volunteered slot leaves the intent NOT fully agent-revealed -> 0.0
    assert e.recovery == 0.0
    assert e.volunteered_slots == ("option:0",)
    assert e.bucket == INFERRED, "credit for a volunteered slot must not buy EARNED"


def test_declined_reveals_do_not_count_as_recovered():
    e = score(traj([ask(1, HIDDEN, declined=True), propose(2, "A", True)]))
    assert e.recovery == 0.0


def test_recovery_is_undefined_when_nothing_was_hidden():
    e = score(traj([propose(1, "A", True)], hidden=[], governing="noise"))
    assert e.recovery is None


def test_nothing_hidden_gets_its_own_bucket_and_never_counts_as_earned():
    """Otherwise `hidden <= extracted` is trivially true, EARNED means two different things,
    and its share moves with the strategy mix rather than with the agent."""
    e = score(traj([propose(1, "A", True)], hidden=[], governing="noise"))
    assert e.bucket == NO_HIDDEN_INTENT
    assert e.success is True


# ------------------------------------------------------------------- I
def test_aim_counts_asks_that_targeted_a_hidden_slot():
    e = score(traj([ask(1, ["attr:wool"]), ask(2, ["price_upper"]),
                    ask(3, ["option:0"]), propose(4, "A", True)]))
    assert e.n_asks == 3 and e.n_aimed_asks == 2
    assert e.aim == pytest.approx(2 / 3)


def test_a_refused_but_well_aimed_ask_still_counts_as_aimed():
    """Aim must measure aim, not the persona's mood."""
    e = score(traj([ask(1, ["attr:wool"], declined=True), propose(2, "A", False)],
                   outcome="TURNS_EXCEEDED"))
    assert e.n_aimed_asks == 1 and e.aim == 1.0


def test_aim_is_undefined_for_an_agent_that_never_asks():
    """Not 0.0 -- a passive agent must not be able to look merely imprecise."""
    assert score(traj([propose(1, "A", True)])).aim is None


def test_restraint_is_only_defined_when_nothing_was_hidden():
    silent = score(traj([propose(1, "A", True)], hidden=[], governing="noise"))
    chatty = score(traj([ask(1, ["attr:wool"]), propose(2, "A", True)], hidden=[],
                        governing="noise"))
    assert silent.restraint is True
    assert chatty.restraint is False
    assert score(traj([propose(1, "A", True)])).restraint is None


# ------------------------------------------------------------------- P
def test_stale_is_resubmitting_the_answer_a_moved_shift_invalidated():
    e = score(traj([propose(1, "A", False),
                    shift(propose(2, "A", False), moved=True)], outcome="TURNS_EXCEEDED"))
    assert e.stale is True and e.over_reaction is None


def test_tracking_a_moved_shift_is_not_stale():
    e = score(traj([propose(1, "A", False),
                    shift(propose(2, "B", True), moved=True)]))
    assert e.stale is False


def test_over_reaction_is_abandoning_a_still_valid_answer():
    """The mirror failure, and why unmoved shifts are kept in the data: measuring only
    staleness would reward an agent that panics at every user utterance."""
    e = score(traj([propose(1, "A", False),
                    shift(propose(2, "B", False), moved=False)], outcome="TURNS_EXCEEDED"))
    assert e.over_reaction is True and e.stale is None


def test_holding_a_still_valid_answer_is_not_over_reaction():
    e = score(traj([propose(1, "A", False),
                    shift(propose(2, "A", True), moved=False)]))
    assert e.over_reaction is False


# ---------------------------------------------------- ablation signatures
def _archetype(name):
    """Four agents, each failing differently."""
    if name == "passive":          # never asks; wins only where nothing was hidden
        return [score(traj([propose(1, "A", True)], hidden=[], governing="noise")),
                score(traj([propose(1, "A", False)], outcome="EXHAUSTED"))]
    if name == "shotgun":          # extracts everything, asks about everything
        return [score(traj([ask(1, ["attr:wool"]), ask(2, ["price_upper"]),
                            ask(3, ["attr:other"]), ask(4, ["option:0"]),
                            propose(5, "A", True)]))]
    if name == "inferrer":         # succeeds without extracting
        return [score(traj([propose(1, "A", True)]))]
    if name == "stubborn":         # aligns once, then stops listening
        return [score(traj([ask(1, HIDDEN), propose(2, "A", False),
                            shift(propose(3, "A", False), moved=True)],
                           outcome="TURNS_EXCEEDED"))]
    raise AssertionError(name)


def test_each_archetype_has_a_distinct_signature():
    passive, shotgun, inferrer, stubborn = (report(_archetype(n)) for n in
                                            ("passive", "shotgun", "inferrer", "stubborn"))
    # passive: no asks at all, and its only win came from a nothing-hidden sample
    assert passive["inquiry"]["silent_episodes"] == 1.0
    assert passive["inquiry"]["restraint"] == 1.0
    assert passive["grounding"][EARNED] == 0.0
    assert passive["grounding"][NO_HIDDEN_INTENT] == 0.5
    # shotgun: full recovery, poor aim
    assert shotgun["recovery"]["mean"] == 1.0
    assert shotgun["inquiry"]["aim"] == pytest.approx(0.5)
    # inferrer: succeeds, zero recovery, all INFERRED
    assert inferrer["success_rate"] == 1.0
    assert inferrer["grounding"][INFERRED] == 1.0
    assert inferrer["recovery"]["mean"] == 0.0
    # stubborn: perfect recovery, but stale after the shift
    assert stubborn["recovery"]["mean"] == 1.0
    assert stubborn["persistence"]["stale"] == 1.0


def test_report_never_collapses_to_one_composite_number():
    """A composite hides which dimension broke, which is the only thing GRIP is for."""
    r = report(_archetype("shotgun"))
    for dim in ("grounding", "recovery", "inquiry", "persistence"):
        assert dim in r
    assert not any(k in r for k in ("grip_score", "score", "overall"))


def test_report_stratifies_by_the_dimensions_that_change_interpretation():
    r = report(_archetype("passive") + _archetype("shotgun"))
    for k in ("by_governing_kind", "by_persona", "by_strategy"):
        assert k in r and r[k]


# ------------------------------------------- post-shift re-anchoring (ruling 2026-08-13)
def test_recovery_reanchors_to_the_shifted_intent():
    """After an answer-moving shift, "true intent revealed" means the NEW intent: a slot whose
    value the shift changed must be re-revealed; pre-shift knowledge of it is stale."""
    turns = [ask(1, HIDDEN), propose(2, "A", False)]
    turns[1]["shift"] = {"gt_moved": True,
                         "delta": {"changed": [["attr:wool", "=", "wool", "cotton"]]}}
    turns.append(ask(3, ["option:0"]))
    turns.append(propose(4, "A", True))
    e = score(traj(turns))
    # attr:wool was revealed BEFORE the shift that changed it -> stale -> intent not revealed
    assert e.recovery == 0.0
    assert e.bucket == INFERRED

    # ...but re-asking it AFTER the shift restores full recovery
    turns2 = [ask(1, HIDDEN), propose(2, "A", False)]
    turns2[1]["shift"] = {"gt_moved": True,
                          "delta": {"changed": [["attr:wool", "=", "wool", "cotton"]]}}
    turns2.append(ask(3, ["attr:wool"]))
    turns2.append(propose(4, "A", True))
    e2 = score(traj(turns2))
    assert e2.recovery == 1.0
    assert e2.bucket == EARNED


def test_shift_added_requirement_joins_the_target():
    """A REFINEMENT shift adds a requirement the opening query never contained; the intent is
    only fully revealed once that new slot is extracted too."""
    turns = [ask(1, HIDDEN), propose(2, "A", False)]
    turns[1]["shift"] = {"gt_moved": True,
                         "delta": {"added": [["attr:vegan", "=", "vegan"]]}}
    turns.append(propose(3, "A", True))
    e = score(traj(turns))
    assert e.recovery == 0.0            # knew the original hidden set, not the addition
    turns2 = list(turns[:2]) + [ask(3, ["attr:vegan"]), propose(4, "A", True)]
    e2 = score(traj(turns2))
    assert e2.recovery == 1.0


def test_pivot_shift_is_unscored_not_mislabelled():
    """A PIVOT's destination conditions are not in the log, so the episode is excluded from
    the G/R denominators and counted, never scored against the stale target."""
    from intent_graph.runtime.grip import UNSCORED_PIVOT
    turns = [ask(1, HIDDEN), propose(2, "A", False)]
    turns[1]["shift"] = {"gt_moved": True, "delta": {"pivot_to": {"category": "shoes"}}}
    turns.append(propose(3, "A", True))
    e = score(traj(turns))
    assert e.bucket == UNSCORED_PIVOT
    assert e.recovery is None
    assert e.hidden == ()


def test_learning_after_a_rejected_buy_counts_toward_recovery():
    """The final-commit rule (ruling 2026-08-13): extraction is snapshotted at the LAST
    proposal, not frozen at the first. An arm that buys early, gets rejected, and then
    learns the hidden requirement must be credited -- the adjudicated commit is what counts.
    Under the old first-proposal freeze this scored recovery 0.0 / INFERRED."""
    turns = [propose(1, "A", False),          # cheap early buy, rejected
             ask(2, HIDDEN),                  # learns the full hidden intent afterwards
             propose(3, "B", True)]           # final, adjudicated commit
    e = score(traj(turns))
    assert e.recovery == 1.0
    assert e.bucket == EARNED


def test_non_gt_moved_shift_still_reanchors():
    """Any shift edits the user's conditions -- the user answers from the new node whether or
    not the winning product moved -- so a RELAXATION that removes the hidden requirement
    shrinks the target even with gt_moved False."""
    turns = [ask(1, ["attr:wool"]), propose(2, "A", False)]
    turns[1]["shift"] = {"gt_moved": False,
                         "delta": {"removed": [["option:0", "=", "large"]]}}
    turns.append(propose(3, "A", True))
    e = score(traj(turns))
    # original hidden = {attr:wool, option:0}; option:0 was removed by the shift ->
    # target = {attr:wool}, which WAS revealed -> fully recovered
    assert e.recovery == 1.0
    assert e.bucket == EARNED


def test_aim_credits_the_post_shift_reask():
    """Re-asking a slot the shift invalidated is the CORRECT behaviour and must count as
    aimed; under the old anchoring it read as 'already extracted' and was penalised."""
    turns = [ask(1, HIDDEN), propose(2, "A", False)]
    turns[1]["shift"] = {"gt_moved": True,
                         "delta": {"changed": [["attr:wool", "=", "wool", "cotton"]]}}
    turns.append(ask(3, ["attr:wool"]))     # the re-ask
    turns.append(propose(4, "B", True))
    e = score(traj(turns))
    assert e.n_asks == 2
    assert e.aim == 1.0                     # both asks aimed: first at hidden, re-ask at stale


def test_asks_after_a_pivot_leave_the_aim_denominator():
    turns = [ask(1, HIDDEN), propose(2, "A", False)]
    turns[1]["shift"] = {"gt_moved": True, "delta": {"pivot_to": {"category": "shoes"}}}
    turns.append(ask(3, ["attr:wool"]))     # unscorable: target unknown after the pivot
    turns.append(propose(4, "B", True))
    e = score(traj(turns))
    assert e.aim == 1.0                     # 1 aimed of 1 SCOREABLE ask, not 1 of 2


# ------------------------------------------------- rejection-sourced recovery credit
def rejected_with_reveals(turn, slots, *, patience=(8, 4)):
    """A rejected proposal whose reaction states requirements (user v2)."""
    rec = propose(turn, "wrong-item", False, patience=patience)
    rec["reveals"] = [{"slot": s, "value": "v", "was_repeat": False, "declined": False,
                      "volunteered": False, "source": "rejection"} for s in slots]
    return rec


def test_rejection_reveals_credit_recovery():
    """the author's scenario (2026-08-13): the rejection reveals the WHOLE true intent and the
    second proposal gets it right -- Recovery must be 1.0 and the win is EARNED. This is
    the regular B0's only information channel; excluding it pinned Recovery at 0.0."""
    e = score(traj([propose(1, "wrong", False) | {"reveals": [
        {"slot": s, "value": "v", "was_repeat": False, "declined": False,
         "volunteered": False, "source": "rejection"} for s in HIDDEN]},
        propose(2, "right", True)]))
    assert e.recovery == 1.0
    assert e.bucket == EARNED


def test_partial_rejection_reveal_is_not_full_recovery():
    """One slot from the rejection, the other never stated: task-level Recovery stays 0."""
    e = score(traj([rejected_with_reveals(1, ["attr:wool"]),
                    propose(2, "right", True)]))
    assert e.recovery == 0.0
    assert e.bucket == INFERRED


# ------------------------------------------------- the two shift channels
# Added 2026-08-19 after a code read found that the episode was writing two things GRIP
# never read: the earlier hops of a multi-hop flush, and the agent-caused trigger record.
def _hop(operator, dst, *, added=(), removed=(), changed=(), moved=True, trigger=None,
         after_reply=False):
    d = {}
    if added:
        d["added"] = [[s, "=", "v"] for s in added]
    if removed:
        d["removed"] = [[s, "=", "v"] for s in removed]
    if changed:
        d["changed"] = [[s, "=", "old", "new"] for s in changed]
    hop = {"operator": operator, "gt_moved": moved, "dst": dst,
           "provenance": "REAL", "delta": d}
    if trigger is not None:
        hop["trigger"] = trigger
    if after_reply:
        hop["after_reply"] = True
    return hop


def test_multi_hop_flush_counts_every_hop_not_just_the_last():
    """A submission fires every due shift. The singular `shift` key holds only the LAST
    hop; the earlier ones live in `shifts_earlier`. Reading only the singular key made
    n_shifts undercount."""
    rec = propose(3, "A", False)
    rec["shifts_earlier"] = [_hop("REFINEMENT", "n2", added=["attr:extra"])]
    rec["shift"] = _hop("RELAXATION", "n3", removed=["option:0"])
    e = score(traj([ask(1, ["attr:wool"]), ask(2, ["option:0"]), rec], outcome="EXHAUSTED"))
    assert e.n_shifts == 2, "the earlier hop was dropped"


def test_multi_hop_flush_composes_the_intermediate_edit_into_the_target():
    """The real damage: a slot ADDED by a middle hop and untouched by the last one has to
    join the target. With only the last hop read, it never did."""
    rec = propose(3, "A", False)
    rec["shifts_earlier"] = [_hop("REFINEMENT", "n2", added=["attr:middle"])]
    rec["shift"] = _hop("RELAXATION", "n3", removed=["option:0"])
    e = score(traj([ask(1, ["attr:wool"]), ask(2, ["option:0"]), rec], outcome="EXHAUSTED"))
    assert "attr:middle" in e.hidden, "middle hop's added slot never entered the target"
    assert "option:0" not in e.hidden, "last hop's removal was not applied"


def test_scheduled_shift_is_not_counted_as_agent_caused():
    e = score(traj([ask(1, ["attr:wool"]), shift(ask(2, ["option:0"])),
                    propose(3, "A", True)]))
    assert e.n_shifts == 1
    assert e.n_shifts_triggered == 0
    assert e.trigger_categories == ()


def test_agent_caused_shift_is_recorded_with_its_slots_and_categories():
    rec = ask(2, ["option:0"])
    rec["shift"] = _hop("REFINEMENT", "n2", added=["attr:new"], after_reply=True,
                        trigger={"kind": "agent_ask", "slots": ["attr:wool"],
                                 "categories": ["REFINEMENT", "RELAXATION"]})
    e = score(traj([ask(1, ["attr:wool"]), rec, propose(3, "A", True)]))
    assert e.n_shifts == 1
    assert e.n_shifts_triggered == 1
    assert e.trigger_slots == ("attr:wool",)
    assert e.trigger_categories == ("REFINEMENT", "RELAXATION")


def test_report_separates_the_two_channels():
    sched = traj([ask(1, ["attr:wool"]), shift(ask(2, ["option:0"])), propose(3, "A", True)])
    trig_rec = ask(2, ["option:0"])
    trig_rec["shift"] = _hop("REFINEMENT", "n2", added=["attr:new"],
                             trigger={"kind": "agent_ask", "slots": ["attr:wool"],
                                      "categories": ["REFINEMENT"]})
    trig = traj([ask(1, ["attr:wool"]), trig_rec, propose(3, "A", True)])
    r = report([score(sched), score(trig)])
    p = r["persistence"]
    assert p["n_shifts_total"] == 2
    assert p["n_shifts_triggered"] == 1
    assert p["triggered_share"] == 0.5
    assert p["episodes_with_trigger"] == 0.5
    assert p["trigger_categories"] == {"REFINEMENT": 1}


# ------------------------------------------------- pivots are scoreable
# A PIVOT replaces the goal outright. The scorer used to give up on those episodes because
# the log did not say what the goal became -- so 21% of WebShop shifts and 55% of retail
# shifts silently left the Grounding and Recovery denominators. The destination node is
# known at episode time, so it is now written down as `dst_slots`.
def _pivot(turn_rec, dst_slots, *, moved=True):
    rec = dict(turn_rec)
    rec["shift"] = {"operator": "PIVOT", "gt_moved": moved, "dst": "n9",
                    "provenance": "REAL", "delta": {"pivot_to": {"cat": "other"}}}
    if dst_slots is not None:
        rec["shift"]["dst_slots"] = list(dst_slots)
    return rec


def test_a_pivot_replaces_the_target_rather_than_voiding_it():
    """After a pivot the user wants something else entirely. Everything they said before is
    about an abandoned goal, and nothing about the new goal has been said yet. So the target
    becomes the destination's requirements -- all of them still owed."""
    e = score(traj([ask(1, ["attr:wool"]),
                    _pivot(ask(2, ["option:0"]), ["attr:steel", "option:9"]),
                    propose(3, "A", False)], outcome="EXHAUSTED"))
    assert e.bucket != "UNSCORED_PIVOT", "the pivot was still treated as unscoreable"
    assert set(e.hidden) == {"attr:steel", "option:9"}, e.hidden
    # answers about the abandoned goal must not count as recovery of the new one
    assert "attr:wool" not in e.extracted_before_commit
    assert e.recovery == 0.0


def test_a_pivot_we_cannot_read_is_still_set_aside():
    """Backward compatibility: logs written before dst_slots existed carry no destination,
    so the target genuinely is unknowable and the episode must still be set aside rather
    than scored against a stale goal."""
    e = score(traj([ask(1, ["attr:wool"]),
                    _pivot(ask(2, ["option:0"]), None),
                    propose(3, "A", False)], outcome="EXHAUSTED"))
    assert e.bucket == "UNSCORED_PIVOT"


def test_a_pivot_still_counts_as_a_shift():
    e = score(traj([ask(1, ["attr:wool"]),
                    _pivot(ask(2, ["option:0"]), ["attr:steel"]),
                    propose(3, "A", True)]))
    assert e.n_shifts == 1


def test_pivot_keeps_reveals_of_slots_the_new_goal_still_owes():
    """2026-08-21 ruling (ruling: the blanket pivot cut was never his rule): a pre-pivot
    reveal survives a pivot iff the destination still owes that slot; answers about
    abandoned slots stop counting because the slot leaves the target, not because a
    pivot voids speech wholesale."""
    e = score(traj([ask(1, ["attr:wool"]),
                    _pivot(ask(2, ["option:0"]), ["attr:wool", "option:9"]),
                    propose(3, "A", False)], outcome="EXHAUSTED"))
    assert "attr:wool" in e.extracted_before_commit      # still owed -> still counts
    assert e.recovery == 0.0                             # option:9 never revealed
