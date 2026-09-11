"""The patience channel: OFF by default, and only ever a cost signal.

Rationale in LLMAgent.patience_note -- the measured cause of ask-budget saturation is that
the cost of asking was invisible to the agent (it saw only actions-left out of 40).
"""

from __future__ import annotations

from intent_graph.runtime.agents import ARMS


def _agent(arm: str, rt: dict):
    cls = ARMS[arm]
    a = cls.__new__(cls)
    a.rt, a.notes, a.state = rt, [], {}
    return a


def test_off_by_default_emits_nothing():
    a = _agent("B0", {})
    a.patience_left = 7
    assert a.patience_note() == ""


def test_absent_attribute_is_safe_even_when_enabled():
    a = _agent("B0", {"show_patience": True})
    assert a.patience_note() == ""          # loop has not published it yet


def test_enabled_reports_remaining_and_both_prices():
    a = _agent("B0", {"show_patience": True, "cost_ask": 2, "cost_reject": 2})
    a.patience_left = 9
    note = a.patience_note()
    assert "9 left" in note
    assert "costs 2" in note                # ask price
    assert "rejected purchase costs 2" in note


def test_warns_when_no_room_left():
    a = _agent("B0", {"show_patience": True, "cost_ask": 3, "cost_reject": 2})
    a.patience_left = 3
    assert "no room for another question" in a.patience_note()
    a.patience_left = 12
    assert "no room" not in a.patience_note()


def test_note_carries_no_intent_information():
    """It must be a pure cost signal: no slot names, no values, no ground truth."""
    a = _agent("A6", {"show_patience": True, "cost_ask": 1, "cost_reject": 2})
    a.patience_left = 5
    note = a.patience_note().lower()
    for leak in ("slot", "hidden", "intent", "answer", "correct", "asin"):
        assert leak not in note
