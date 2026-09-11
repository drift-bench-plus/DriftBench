"""The state block must never reach the simulated user inside the question text.

Regression test for a measured defect: the guidance tells each stateful arm to end its
reply with 'LABEL: {json}', and the action slice ran to the end of the reply, so the raw
JSON travelled inside action.question. Measured on stored runs before the fix: 9.6% of
A1-v3 asks, 9.5% of A2, 23.2% of A3-v2 and 47.4% of A5 carried the block, and those asks
failed user.select_slots 14-33% of the time -- one patience point each, no information.
"""

from __future__ import annotations

import pytest

from intent_graph.runtime.agents import ARMS


def _arm(name: str):
    """An arm instance without __init__ (which wants a live llm/config)."""
    cls = ARMS[name]
    a = cls.__new__(cls)
    a.state, a.notes = {}, []
    return a


def test_nested_state_block_stripped_and_parsed():
    a = _arm("LEG1")   # ledger arm (A1 is now the memory arm)
    out = a._absorb(
        "Action: Clarify\nWhat is your budget?\n\n"
        'LEDGER: {"slots": {"color": "grey", "budget": "UNK"}, "confirmed": false}')
    assert "LEDGER" not in out
    assert out.endswith("What is your budget?")
    # brace-matched, so the NESTED object survives -- a non-greedy regex truncates here
    assert a.state == {"slots": {"color": "grey", "budget": "UNK"}, "confirmed": False}
    assert not a.notes


def test_state_block_before_action_also_stripped():
    a = _arm("A5")
    out = a._absorb('AMBIGUITY: {"type": "VAGUE", "settled": []}\n'
                    "Action: Clarify\nHow much can you spend?")
    assert out == "Action: Clarify\nHow much can you spend?"
    assert a.state["type"] == "VAGUE"


def test_repeated_blocks_all_stripped_last_wins():
    a = _arm("A6")
    out = a._absorb('GRAPH: {"nodes": {"a": {"belief": 0.1}}}\n'
                    "Action: Clarify\nWhat size?\n"
                    'GRAPH: {"nodes": {"b": {"belief": 0.9}}}')
    assert "GRAPH" not in out
    assert out.endswith("What size?")
    assert list(a.state["nodes"]) == ["b"]


def test_no_block_is_unchanged():
    a = _arm("A1")
    out = a._absorb("Action: Operation\n```\nsearch[grey socks]\n```")
    assert out.startswith("Action: Operation")
    assert "search[grey socks]" in out
    assert not a.notes


def test_malformed_state_is_recorded_and_still_stripped():
    a = _arm("LEG3")   # intent-record arm (A3 is now the self-evolving arm)
    out = a._absorb('Action: Clarify\nWhich one?\nRECORD: {"broken": }')
    assert "RECORD" not in out
    assert any("state_parse_failed" in n for n in a.notes)


@pytest.mark.parametrize("arm", ["A1", "A2", "A3", "LEG1", "LEG2", "LEG3", "A4", "A5", "A6"])
def test_every_stateful_arm_strips_its_own_label(arm):
    a = _arm(arm)
    label = ARMS[arm].state_label
    out = a._absorb(f'Action: Clarify\nAny colour preference?\n{label}: {{"x": 1}}')
    assert label not in out
    assert out.endswith("Any colour preference?")


def test_truncated_state_block_is_stripped_not_leaked():
    """The reply can hit max_output_tokens mid-JSON, leaving braces unbalanced.

    Measured on B2 (belief graph), whose state is verbose: 8 open braces against 7 closed.
    Before this case was handled the fragment leaked into the user-visible question.
    """
    a = _arm("B2")
    truncated = ('Action: Clarify\nWhat waist size?\n'
                 'GRAPH: {"nodes": {"waist": {"value": "31", "belief": 0.1, '
                 '"importance": 0.9}, "price_cap": {"value":')
    out = a._absorb(truncated)
    assert "GRAPH" not in out
    assert "belief" not in out
    assert out.endswith("What waist size?")
    assert any("state_parse_failed" in n for n in a.notes)


def test_canonical_and_legacy_arm_names_resolve_to_one_class():
    """A = ours, B1/B2 = published baselines; the old keys must keep working."""
    assert ARMS["A"] is ARMS["LEG1"]
    assert ARMS["B1"] is ARMS["A5"]
    assert ARMS["B2"] is ARMS["A6"]


# ---------------------------------------------- each method held to its own resolution rule
def _armed(name: str, state: dict):
    a = _arm(name)
    a.state = state
    return a


def test_b2_forbids_buying_from_an_unsettled_graph():
    """B2's rule is "ask iff the largest gap reaches 0.4"; such a graph is not settled, so
    buying from it is the move the rule exists to prevent. Same threshold, other branch."""
    a = _armed("B2", {"nodes": {}, "gap": 0.8, "decision": "ask"})
    assert "not resolved" in (a.rule_forbids_commit() or "")
    assert a.rule_forbids_ask() is None            # asking IS allowed at 0.8
    b = _armed("B2", {"nodes": {}, "gap": 0.1, "decision": "act"})
    assert b.rule_forbids_commit() is None         # settled: buying is the rule's own answer
    assert "settled" in (b.rule_forbids_ask() or "")


def test_b3_forbids_buying_from_a_pool_that_still_splits():
    a = _armed("B3", {"pool": ["X", "Y"], "split": "colour", "decision": "ask"})
    assert "still splits" in (a.rule_forbids_commit() or "")
    b = _armed("B3", {"pool": ["X"], "split": None, "decision": "act"})
    assert b.rule_forbids_commit() is None


def test_b1_has_no_purchase_gate_but_is_held_to_its_evidence_rule():
    """AT-CoT says nothing about committing, so it gets NO purchase gate -- inventing one
    would be a contribution on its behalf. It DOES state an evidence rule for asking (a
    type must be quotable, a settled type is never re-opened), and that rule is enforced
    like every other stated arithmetic in this suite (2026-08-14: in the prompt alone it
    rode the ask cap in 86% of episodes)."""
    a = _armed("B1", {"type": "VAGUE", "evidence": "cheap", "settled": []})
    assert a.rule_forbids_commit() is None          # no invented purchase gate
    assert ARMS["B1"].enforce_own_rule is True      # ask-side rule IS enforced
    # first typed question is allowed; a second, after one is settled, is not
    assert a.rule_forbids_ask() is None
    b = _armed("B1", {"type": "MISSING", "evidence": "no budget", "settled": ["VAGUE"]})
    assert b.rule_forbids_ask() is not None
    # and a NONE classification with nothing settled simply acts
    c = _armed("B1", {"type": "NONE", "evidence": "", "settled": []})
    assert c.rule_forbids_ask() is None


def test_missing_or_junk_state_never_blocks():
    for state in ({}, {"gap": "nonsense"}, {"nodes": {}}):
        assert _armed("B2", state).rule_forbids_commit() is None
        assert _armed("B2", state).rule_forbids_ask() is None


def test_question_count_does_not_release_a_method_commit_gate():
    """A removed question cap must not silently survive as a stateful-arm release valve."""
    class _T:
        def __init__(self, asks):
            self.turns = [{"role": "agent", "content": "Action: Clarify\nStrategy: Ask_Parameter"}
                          for _ in range(asks)]

    a = _arm("B2")
    a.state = {"nodes": {}, "gap": 0.9, "decision": "ask"}
    a.rt = {"max_turns": 100}
    a.llm = None                                     # must not be reached
    a.build_prompt = lambda t: "unused"
    # The method's own gate holds regardless of how many questions were asked.
    assert a.rule_forbids_commit() is not None
    assert a.must_commit(_T(2)) is False
    assert a.must_commit(_T(50)) is False


def test_b3_assumed_attributes_block_the_purchase():
    """The analogue of B2's winning fix, in B3's own representation.

    ProductAgent summarises the request's attributes, so provenance is available to it: an
    attribute the agent ASSUMED is one its own query invented, and pool agreement on it is the
    query's doing rather than the shopper's preference. Such a pool is not resolved.
    """
    a = _arm("B3")
    a.state = {"pool": ["X"], "stated": ["black"], "assumed": ["size 10"], "split": None,
               "decision": "act"}
    assert "still assuming" in (a.rule_forbids_commit() or "")
    b = _arm("B3")
    b.state = {"pool": ["X"], "stated": ["black", "size 10"], "assumed": [], "split": None,
               "decision": "act"}
    assert b.rule_forbids_commit() is None
