"""GRIP v2: the ten-metric scorer is pure arithmetic over stored trajectories."""

from intent_graph.runtime import grip


def _traj(outcome, turns, hidden=("price_upper",), persona="rational"):
    return {
        "header": {"graph_id": "t", "persona": persona, "strategy_id": "s",
                   "perturbation": {"hidden_slots": list(hidden)}},
        "turns": turns,
        "outcome": outcome,
    }


def test_v2_earned_patience_and_no_shift_undefined():
    turns = [
        {"turn": 1, "action": {"kind": "ASK"}, "patience_before": 13, "patience_after": 12,
         "reveals": [{"slot": "price_upper", "value": "30", "declined": False,
                      "volunteered": False}]},
        {"turn": 2, "action": {"kind": "PROPOSE"}, "acceptance": {"ok": True},
         "patience_before": 12, "patience_after": 12},
    ]
    row = grip.score_v2(_traj("SUCCESS", turns))
    assert row["success"] and row["bucket"] == grip.EARNED
    assert row["patience_left"] == 12
    assert row["stale"] is grip.UNDEFINED and row["reaction"] is grip.UNDEFINED
    rep = grip.report_v2([row])
    assert rep["success"] == 1.0 and rep["earned"] == 1.0
    assert rep["staleness"] is grip.UNDEFINED and rep["reaction"] is grip.UNDEFINED


def test_v2_staleness_from_recorded_executable_check():
    turns = [
        {"turn": 1, "action": {"kind": "PROPOSE"}, "acceptance": {"ok": False},
         "patience_before": 13, "patience_after": 11},
        {"turn": 2, "action": {"kind": "PROPOSE"}, "patience_before": 11,
         "patience_after": 9, "acceptance": {"ok": False},
         "shift": {"src": "a", "dst": "b", "gt_moved": True}},
        {"turn": 3, "forced_final": True, "action": {"kind": "PROPOSE"},
         "acceptance": {"ok": False},
         "stale_check": {"old_node": "a", "valid_under_old": True}},
    ]
    row = grip.score_v2(_traj("EXHAUSTED", turns))
    assert row["had_moving_shift"] and row["stale"] is True
    rep = grip.report_v2([row])
    assert rep["staleness"] == 1.0 and rep["n_stale_dom"] == 1


def test_v2_reaction_is_turns_from_last_moving_shift_to_acceptance():
    turns = [
        {"turn": 1, "action": {"kind": "ASK"}, "patience_before": 13, "patience_after": 12,
         "reveals": []},
        {"turn": 2, "action": {"kind": "PROPOSE"}, "acceptance": {"ok": False},
         "patience_before": 12, "patience_after": 10,
         "shift": {"src": "a", "dst": "b", "gt_moved": True}},
        {"turn": 5, "action": {"kind": "PROPOSE"}, "acceptance": {"ok": True},
         "patience_before": 10, "patience_after": 10},
    ]
    row = grip.score_v2(_traj("SUCCESS", turns))
    assert row["reaction"] == 3 and row["post_shift_success"] is True
    rep = grip.report_v2([row])
    assert rep["reaction"] == 3 and rep["n_shifted"] == 1


def test_v2_report_conditions_never_zero_fill():
    rows = [grip.score_v2(_traj("SUCCESS", [
        {"turn": 1, "action": {"kind": "PROPOSE"}, "acceptance": {"ok": True},
         "patience_before": 13, "patience_after": 13}], hidden=()))]
    rep = grip.report_v2(rows)
    assert rep["earned"] is grip.UNDEFINED    # nothing hidden
    assert rep["aim"] is grip.UNDEFINED       # never asked
    assert rep["staleness"] is grip.UNDEFINED # no shift
