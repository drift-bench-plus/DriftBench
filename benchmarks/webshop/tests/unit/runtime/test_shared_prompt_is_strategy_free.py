"""The shared framing must not teach clarification.

Every arm's system prompt is PROMPT_FORMAT + SHOP_DOMAIN + that arm's own guidance, so
anything in the first two is given to B0 -- the arm whose defining property is having NO
strategy. Before 2026-08-12 the shared text carried a five-way clarification menu WITH
DEFINITIONS, an instruction to ask when the request "is ambiguous, incomplete, or looks
wrong", and the patience cost model. That is a clarification playbook, and it pre-installs
most of AT-CoT's (B1's) contribution for every arm.
"""

from __future__ import annotations

from intent_graph.runtime.agent_api import CLARIFY_STRATEGIES, PROMPT_FORMAT, parse_action
from intent_graph.runtime.agents import SHOP_DOMAIN, ARMS

SHARED = PROMPT_FORMAT + SHOP_DOMAIN


def test_no_when_to_ask_instruction():
    low = SHARED.lower()
    for teach in ("if the request is ambiguous", "incomplete, or looks wrong",
                  "ask the user:"):
        assert teach not in low, teach


def test_no_strategy_definitions():
    """The tokens must survive (the parser demands them); their meanings must not."""
    low = SHARED.lower()
    for definition in ("report an objective problem", "ask for a missing detail",
                       "offer options and ask the user to choose",
                       "point out the problem and suggest an alternative",
                       "confirm before an irreversible action"):
        assert definition not in low, definition


def test_strategy_tokens_are_gone_from_the_menu():
    """the author 2026-08-13: 'Don't use those 5 strategy. Just a asking skill.' The token
    names were a standing list of reasons to message the user; none may appear in the
    format any arm sees."""
    for token in CLARIFY_STRATEGIES:
        assert token not in PROMPT_FORMAT, token


def test_a_clarify_parses_with_and_without_a_strategy_line():
    """Strategy is optional metadata now: bare Content parses, a legacy Strategy line is
    recorded but never required and never judged."""
    bare = parse_action("Action: Clarify\nContent: What size?")
    assert bare.kind.value == "ASK" and bare.question == "What size?"
    assert bare.strategy is None and not bare.problems
    legacy = parse_action("Action: Clarify\nStrategy: Ask_Parameter\nContent: What size?")
    assert legacy.kind.value == "ASK" and legacy.strategy == "Ask_Parameter"
    assert not legacy.problems


def test_no_cost_model_in_shared_text():
    low = SHARED.lower()
    for econ in ("costs the shopper's patience", "unnecessary question",
                 "never buying anything", "spends the shopper's patience"):
        assert econ not in low, econ


def test_b0_has_no_guidance_and_no_coaching():
    assert ARMS["B0"].guidance == ""
    assert ARMS["B0"].coach_on_budget is False


def test_no_arm_states_the_cost_arithmetic():
    """the author's ruling (2026-08-13): NO arm knows the economy -- not the baselines (they
    never did) and, since the need-driven-asking redesign, not ours either. The agent
    must not be told the budget or what actions cost; asking is a skill used on need,
    not an arithmetic played against a known price list."""
    for arm in ("B1", "B2", "A", "A1v2", "A1v3", "A1nomem"):
        low = (ARMS[arm].guidance or "").lower()
        assert "patience" not in low, arm
        for phrase in ("costs 1", "costs 2", "costs 4", "costs double"):
            assert phrase not in low, (arm, phrase)
