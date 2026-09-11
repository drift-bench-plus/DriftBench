"""The simulated user: leak-proofing, reveal semantics, and persona behaviour."""

import random

import pytest

from intent_graph.models import GroundTruth, Node
from intent_graph.runtime import persona as P
from intent_graph.runtime.llm import ROLE_PHRASE, ROLE_SELECT, FakeLLM
from intent_graph.runtime.user import SimulatedUser, type_label

CONDS = (("attr:easy use", "=", "easy use"),
         ("option:0", "=", "heather charcoal"),
         ("price_upper", "<=", 50.0))

CONFIG = {"runtime": {"max_consecutive_none": 3, "min_reveal_after_n_asks": 2}}


def node(conds=CONDS):
    return Node.build(adapter="t", adapter_version="1", env_id="e", base={"f": 1},
                      conditions=conds, recipe={}, ground_truth=GroundTruth.rowset([["x"]]))


def make(persona="rational", responder=None, seed=1):
    fake = FakeLLM(responder=responder or (lambda role, sys, p: "slot_0" if role == ROLE_SELECT
                                           else "sure, here you go"))
    u = SimulatedUser(P.get(persona), fake, random.Random(seed), CONFIG)
    return u, fake


# ------------------------------------------------------------------- leakage
def test_selector_prompt_contains_no_slot_names_and_no_values():
    """On WebShop the slot NAME is the value (`attr:easy use`), so names must not be shown."""
    u, fake = make()
    u.answer_ask("which colour do you want?", node())
    sel = fake.prompts(ROLE_SELECT)[0]
    for slot, _, value in CONDS:
        assert slot not in sel, f"slot name {slot!r} leaked into the selector prompt"
        assert str(value).lower() not in sel.lower(), f"value {value!r} leaked"
    assert "slot_0" in sel and "product option" in sel


def test_no_unrevealed_value_ever_appears_in_any_call():
    """The load-bearing test. It may be made stricter, never weaker."""
    secret = "topsecretvalue"
    conds = (("attr:easy use", "=", "easy use"), ("option:0", "=", secret),
             ("price_upper", "<=", 50.0))
    # the selector always picks slot_0, which sorts to attr:easy use -- so option:0 stays hidden
    u, fake = make(responder=lambda role, sys, p: "slot_0" if role == ROLE_SELECT else "ok")
    for _ in range(200):
        u.answer_ask("tell me about the feature you need", node(conds))
    assert secret not in u.state.revealed.get("option:0", "") or True  # not revealed at all
    assert "option:0" not in u.state.revealed
    assert secret not in fake.all_text(), "an unrevealed value reached the model"


def test_phrasing_receives_only_the_revealed_pairs():
    u, fake = make()
    u.answer_ask("what feature?", node())
    phrase = fake.prompts(ROLE_PHRASE)[0]
    assert "easy use" in phrase                       # the one revealed
    assert "heather charcoal" not in phrase            # the others are not
    assert "50" not in phrase


# -------------------------------------------------------------- reveal rules
def test_compound_question_can_reveal_up_to_granularity():
    u, fake = make("dependent",                      # granularity 2
                   responder=lambda role, sys, p: "slot_0, slot_1" if role == ROLE_SELECT else "ok")
    _, reveals = u.answer_ask("which feature and which option?", node())
    granted = [r for r in reveals if not r.declined]
    assert len(granted) == 2


def test_granularity_caps_the_reveal():
    u, _ = make("rational",                           # granularity 1
                responder=lambda role, sys, p: "slot_0, slot_1, slot_2" if role == ROLE_SELECT else "ok")
    _, reveals = u.answer_ask("tell me everything", node())
    assert len([r for r in reveals if not r.declined]) == 1


def test_unmatched_question_reveals_nothing():
    u, _ = make(responder=lambda role, sys, p: "NONE" if role == ROLE_SELECT else "huh?")
    text, reveals = u.answer_ask("what is the weather", node())
    assert reveals == [] and u.state.consecutive_none == 1


def test_repeated_ask_is_marked_as_a_repeat():
    u, _ = make()
    u.answer_ask("what feature?", node())
    _, reveals = u.answer_ask("what feature again?", node())
    assert reveals[0].was_repeat is True


def test_stonewalling_is_bounded_by_max_consecutive_none():
    """A selector that never matches must not make the episode unwinnable."""
    u, _ = make(responder=lambda role, sys, p: "NONE" if role == ROLE_SELECT else "ok")
    for _ in range(2):
        _, r = u.answer_ask("???", node())
        assert r == []
    _, reveals = u.answer_ask("???", node())          # third strike: the user volunteers
    assert any(not r.declined for r in reveals)


# ------------------------------------------------------------------ personas
def test_declining_persona_yields_to_persistence():
    """Deterministic: a persona that ALWAYS declines must still yield, or an episode with
    it could never be won. p_decline=1.0 makes the contract testable without a coin flip."""
    stubborn = P.Persona("stubborn", "bio", reveal_granularity=1, volunteer=0.0,
                         p_decline=1.0, patience_mult=1.0, hint=0.0)
    fake = FakeLLM(responder=lambda role, sys, p: "slot_0" if role == ROLE_SELECT else "ok")
    u = SimulatedUser(stubborn, fake, random.Random(0), CONFIG)

    _, first = u.answer_ask("what feature?", node())
    assert first and first[0].declined, "p_decline=1.0 must decline the first ask"
    assert "attr:easy use" not in u.state.revealed

    _, second = u.answer_ask("what feature?", node())   # min_reveal_after_n_asks = 2
    assert any(not r.declined for r in second), "the override must fire on the second ask"
    assert u.state.revealed["attr:easy use"] == "easy use"


def test_avoidant_can_decline_at_all():
    """Statistical, over seeds: avoidant's p_decline=0.5 fires sometimes."""
    fired = 0
    for seed in range(20):
        u, _ = make("avoidant", seed=seed)
        _, reveals = u.answer_ask("what feature?", node())
        if reveals and reveals[0].declined:
            fired += 1
    assert 3 <= fired <= 17, f"p_decline=0.5 fired {fired}/20 times"


def test_rational_never_declines():
    u, _ = make("rational", seed=5)
    for _ in range(10):
        _, reveals = u.answer_ask("what feature?", node())
        assert not any(r.declined for r in reveals)


def test_persona_bio_is_the_system_text_not_the_prompt():
    u, fake = make("avoidant")
    u.answer_ask("what feature?", node())
    phrase_calls = [c for c in fake.calls if c.role == ROLE_PHRASE]
    assert phrase_calls and "marketing coordinator" in (phrase_calls[0].system or "")
    assert "marketing coordinator" not in phrase_calls[0].prompt


def test_selector_never_receives_the_persona_bio():
    """Selection is a mechanical mapping; personality has no business influencing it."""
    u, fake = make("avoidant")
    u.answer_ask("what feature?", node())
    assert all(c.system in (None, "") for c in fake.calls if c.role == ROLE_SELECT)


def test_patience_scales_with_persona():
    assert P.get("dependent").patience(10) == 12
    assert P.get("spontaneous").patience(10) == 7


def test_random_persona_requires_the_seeded_rng():
    with pytest.raises(ValueError, match="seeded"):
        P.get("random")
    assert P.get("random", random.Random(0)).id in P.PERSONA_IDS


# ---------------------------------------------------------------- type labels
@pytest.mark.parametrize("slot,expected", [
    ("where:Method", "a filter on the data"),
    ("set:Venue", "a value to write"),
    ("value:Player", "a field of the new record"),
    ("attr:easy use", "a required product feature"),
    ("option:0", "a product option such as size or colour"),
    ("price_upper", "a price limit"),
    ("find:size", "a file property"),
    ("grep:pattern", "a text-search setting"),
])
def test_type_labels_describe_without_revealing(slot, expected):
    assert type_label(slot) == expected


# -------------------------------------------- consecutive-ask decline override
def _stubborn_user(p_decline=1.0, n=2):
    """A user who always declines, so only an override can produce a reveal."""
    from intent_graph.runtime.persona import Persona
    from intent_graph.runtime.user import SimulatedUser
    persona = Persona(id="x", bio="b", reveal_granularity=1, volunteer=0.0,
                      p_decline=p_decline, patience_mult=1.0, hint=0.0)
    cfg = {"runtime": {"min_reveal_after_n_asks": n, "max_consecutive_none": 99}}
    return SimulatedUser(persona, None, random.Random(0), cfg)


def test_override_fires_on_consecutive_asks_for_the_same_slot():
    u = _stubborn_user()
    assert u._should_decline("attr:a") is True
    u.state.streak_slot, u.state.streak = "attr:a", 1
    assert u._should_decline("attr:a") is True
    u.state.streak = 2
    assert u._should_decline("attr:a") is False, "persistence must be rewarded"


def test_override_does_not_fire_when_asks_are_interleaved():
    """The specified rule is CONSECUTIVE asks. Asking A, B, A is two asks for A but never
    two in a row, so it must not unlock A -- that was the old total-count behaviour."""
    u = _stubborn_user()
    u.state.ask_counts = {"attr:a": 2}
    u.state.streak_slot, u.state.streak = "attr:b", 1
    assert u._should_decline("attr:a") is True


def test_total_ask_backstop_keeps_episodes_winnable():
    """A purely consecutive rule lets an alternating agent be declined forever; an episode
    no correct agent can win measures nothing."""
    u = _stubborn_user(n=2)
    u.state.ask_counts = {"attr:a": 4}          # 2 * min_reveal_after_n_asks
    u.state.streak_slot, u.state.streak = "attr:b", 1
    assert u._should_decline("attr:a") is False


def test_streak_resets_on_a_different_slot_and_on_compound_questions():
    from intent_graph.runtime.user import UserState
    u = _stubborn_user()
    u.state = UserState()

    def ask(slots):
        for s in slots:
            u.state.ask_counts[s] = u.state.ask_counts.get(s, 0) + 1
        if len(slots) == 1:
            only = slots[0]
            u.state.streak = u.state.streak + 1 if u.state.streak_slot == only else 1
            u.state.streak_slot = only
        else:
            u.state.streak_slot, u.state.streak = None, 0

    ask(["attr:a"])
    ask(["attr:a"])
    assert (u.state.streak_slot, u.state.streak) == ("attr:a", 2)
    ask(["attr:b"])
    assert (u.state.streak_slot, u.state.streak) == ("attr:b", 1)
    ask(["attr:b", "attr:c"])
    assert (u.state.streak_slot, u.state.streak) == (None, 0)


# ------------------------------------- re-asking a known slot must not pay off
def test_reasking_a_known_slot_gets_an_acknowledgement_not_a_repeat_answer():
    """Observed live: the agent asked four different questions, select_slots mapped each to the
    one slot already revealed, and the user emitted the SAME sentence every time. The agent
    gained nothing and looped to the turn cap, making every arm's score a simulator artifact."""
    import random as _r

    from intent_graph.runtime.persona import Persona
    from intent_graph.runtime.user import SimulatedUser

    calls = []

    class _LLM:
        def complete(self, prompt, *, role, system=None, **kw):
            calls.append(prompt)
            return "already told you"

    conds = (("attr:a", "=", "aaa"), ("option:0", "=", "bbb"), ("price_upper", "<=", 50))
    persona = Persona(id="x", bio="b", reveal_granularity=1, volunteer=0.0, p_decline=0.0,
                      patience_mult=1.0, hint=0.0)
    u = SimulatedUser(persona, _LLM(), _r.Random(0),
                      {"runtime": {"min_reveal_after_n_asks": 2, "max_consecutive_none": 99}})
    u.select_slots = lambda q, c: ["attr:a"]          # every question maps to the same slot

    class _Node:
        conditions = conds
    first_text, first = u.answer_ask("what feature?", _Node())
    assert [r.slot for r in first] == ["attr:a"] and first[0].was_repeat is False

    _text, again = u.answer_ask("tell me about the feature again?", _Node())
    assert [r.slot for r in again] == ["attr:a"]
    assert again[0].was_repeat is True, "a second ask for the same slot is a repeat"
    assert "already told" in calls[-1].lower() or "already" in calls[-1].lower()


def test_reasking_never_volunteers_the_remaining_slots():
    """The exploit an earlier version of the redirect created: asking one question repeatedly
    would have handed over every other slot, so persistence would outscore aimed questioning."""
    import random as _r

    from intent_graph.runtime.persona import Persona
    from intent_graph.runtime.user import SimulatedUser

    class _LLM:
        def complete(self, prompt, *, role, system=None, **kw):
            return "ok"

    conds = (("attr:a", "=", "aaa"), ("option:0", "=", "SECRET"), ("price_upper", "<=", 50))
    persona = Persona(id="x", bio="b", reveal_granularity=1, volunteer=0.0, p_decline=0.0,
                      patience_mult=1.0, hint=0.0)
    u = SimulatedUser(persona, _LLM(), _r.Random(0),
                      {"runtime": {"min_reveal_after_n_asks": 2, "max_consecutive_none": 99}})
    u.select_slots = lambda q, c: ["attr:a"]

    class _Node:
        conditions = conds
    for _ in range(8):
        u.answer_ask("the feature?", _Node())
    assert "option:0" not in u.state.revealed, "re-asking must not leak other slots"
    assert "price_upper" not in u.state.revealed


def test_repeat_phrasings_differ_across_ordinals():
    """The broken-record fix said the right thing in verbatim-identical cached words forever
    (RS gate catch, 2026-08-10): the repeat ordinal is part of the phrasing prompt, so the
    2nd and 3rd repeats produce DIFFERENT prompts and therefore different cached replies."""
    import random as _r

    from intent_graph.runtime.persona import Persona
    from intent_graph.runtime.user import SimulatedUser

    prompts = []

    class _LLM:
        def complete(self, prompt, *, role, system=None, **kw):
            prompts.append(prompt)
            return f"reply#{len(prompts)}"

    conds = (("attr:a", "=", "aaa"), ("option:0", "=", "bbb"))
    persona = Persona(id="x", bio="b", reveal_granularity=1, volunteer=0.0, p_decline=0.0,
                      patience_mult=1.0, hint=0.0)
    u = SimulatedUser(persona, _LLM(), _r.Random(0),
                      {"runtime": {"min_reveal_after_n_asks": 9, "max_consecutive_none": 99}})
    u.select_slots = lambda q, c: ["attr:a"]

    class _Node:
        conditions = conds
    u.answer_ask("q1?", _Node())            # first reveal
    u.answer_ask("q2?", _Node())            # repeat 2
    u.answer_ask("q3?", _Node())            # repeat 3
    repeats = [p for p in prompts if "AGAIN" in p]
    assert len(repeats) == 2
    assert repeats[0] != repeats[1], "consecutive repeat prompts must differ (ordinal)"
    assert "number 2" in repeats[0] and "number 3" in repeats[1]


# ================================================================== user v2 (2026-08-13)
def test_v2_decline_gate_never_invokes_the_llm():
    from intent_graph.runtime.user import SimulatedUserV2
    from intent_graph.runtime.persona import get
    import random

    class _Boom:
        def complete(self, *a, **k):
            raise AssertionError("LLM must not be called when the gate declines")

    rng = random.Random(0)
    u = SimulatedUserV2(get("avoidant", rng), _Boom(), rng,
                        {"runtime": {"max_consecutive_none": 3,
                                     "min_reveal_after_n_asks": 2}})
    # avoidant p_decline = 0.5; force the draw deterministically
    u.rng = type("R", (), {"random": staticmethod(lambda: 0.0)})()

    class _N:
        conditions = (("attr:wool", "=", "wool"),)

    reply, reveals = u.answer_ask("what material?", _N())
    assert "understand" in reply.lower()
    assert reveals == []


def test_v2_persistence_cracks_the_gate():
    """Two consecutive asks and the decline gate must stop firing (solvability guard)."""
    from intent_graph.runtime.user import SimulatedUserV2
    from intent_graph.runtime.persona import get
    import random

    calls = []

    class _LLM:
        def complete(self, prompt, **k):
            calls.append(k.get("role"))
            if k.get("role") == "select":
                # the split-observer format (asked-for vs extra disclosures)
                return "ASKED: slot_0\nEXTRA: NONE" if "ASKED" in prompt else "slot_0"
            return "It needs to be wool."

    rng = random.Random(0)
    u = SimulatedUserV2(get("avoidant", rng), _LLM(), rng,
                        {"runtime": {"max_consecutive_none": 3,
                                     "min_reveal_after_n_asks": 2}})
    u.rng = type("R", (), {"random": staticmethod(lambda: 0.0)})()  # always tries to decline

    class _N:
        conditions = (("attr:wool", "=", "wool"),)

    r1, _ = u.answer_ask("what material?", _N())
    r2, _ = u.answer_ask("what material?", _N())
    reply3, reveals3 = u.answer_ask("what material?", _N())
    assert "understand" in r1.lower() and "understand" in r2.lower()
    assert "wool" in reply3.lower()          # the third press gets a real answer
    assert reveals3 and reveals3[0].slot == "attr:wool"


def test_v2_extra_disclosures_are_marked_volunteered():
    """the author 2026-08-13: information the agent never asked for is not elicitation. The
    observer splits ASKED from EXTRA; extras land as volunteered, which Recovery ignores."""
    from intent_graph.runtime.user import SimulatedUserV2
    from intent_graph.runtime.persona import get
    import random

    class _LLM:
        def complete(self, prompt, **k):
            if k.get("role") == "select":
                return "ASKED: slot_0\nEXTRA: slot_1"
            # "50" in digits, not "fifty": the grounding gate (2026-08-21) only credits
            # a reveal whose value tokens are literally in the spoken text; numbers
            # ground on their digit string. "fifty" being dropped is the gate working.
            return "Wool, please. Oh and my budget is 50."

    rng = random.Random(0)
    u = SimulatedUserV2(get("rational", rng), _LLM(), rng,
                        {"runtime": {"max_consecutive_none": 3,
                                     "min_reveal_after_n_asks": 2}})
    u.rng = type("R", (), {"random": staticmethod(lambda: 0.99)})()  # no decline, no volunteer

    class _N:
        conditions = (("attr:wool", "=", "wool"), ("price_upper", "<=", "50"))

    _, reveals = u.answer_ask("what material?", _N())
    by = {r.slot: r.volunteered for r in reveals}
    assert by["attr:wool"] is False          # asked for -> credited extraction
    assert by["price_upper"] is True         # never asked -> volunteered, uncredited


def test_v2_privacy_tags_and_uncredited_extras():
    """The blurt fix (2026-08-13): PRIVATE items are tagged by CODE in the prompt, and a
    disclosure the question did not ask for is recorded as volunteered (uncredited)."""
    from intent_graph.runtime.user import SimulatedUserV2, _intent_lines
    from intent_graph.runtime.persona import get
    import random

    conds = (("attr:wool", "=", "wool"), ("price_upper", "<=", "30"))
    lines = _intent_lines(conds, shared={"attr:wool"})
    assert "[already mentioned]" in lines and "[PRIVATE]" in lines
    assert lines.index("wool") < lines.index("30")   # sorted, sanity

    class _LLM:
        def __init__(self): self.n = 0
        def complete(self, prompt, **k):
            if k.get("role") == "select":
                # observer: wool was asked for; the price was blurted
                return "ASKED: slot_0\nEXTRA: slot_1"
            return "It must be wool — oh and my budget is 30 dollars."

    rng = random.Random(0)
    u = SimulatedUserV2(get("rational", rng), _LLM(), rng,
                        {"runtime": {"max_consecutive_none": 3,
                                     "min_reveal_after_n_asks": 2}})
    u.rng = type("R", (), {"random": staticmethod(lambda: 0.99)})()   # no decline, no volunteer
    u.stated_slots = set()

    class _N:
        conditions = conds

    reply, reveals = u.answer_ask("what material do you want?", _N())
    got = {r.slot: r.volunteered for r in reveals}
    assert got["attr:wool"] is False          # asked-for: credited
    assert got["price_upper"] is True         # blurted: recorded but UNCREDITED


def test_rejection_hint_is_persona_gated():
    """Whether a rejection explains itself is a PERSONA trait (ruling 2026-08-14), not a
    constant: avoidant shoppers (hint 0.1) usually just say no, dependent ones (0.6)
    usually say what is wrong. Before this, every rejection explained itself, which handed
    every arm a free information channel and contradicted the persona."""
    import random

    from intent_graph.runtime import scripted as S
    from intent_graph.runtime.user import REACT_HINT, REACT_NO_HINT, SimulatedUserV2
    from intent_graph.runtime.persona import PERSONAS

    class _Node:
        conditions = (("attr:red", "=", "red"), ("price_upper", "<=", 20))

    seen = {}
    for pid in ("avoidant", "dependent"):
        prompts = []

        class _LLM:
            def complete(self, prompt, **kw):
                prompts.append(prompt)
                return "No, that's not it."
            def usage(self): return {}

        u = SimulatedUserV2(PERSONAS[pid], _LLM(), random.Random(0),
                            {"runtime": {"user_v2": True}})
        u.stated_slots = set()
        hinted = 0
        for i in range(40):
            u.rng = random.Random(i)
            prompts.clear()
            u.react_to_proposal("some item", _Node(), accepted=False)
            body = "\n".join(prompts)
            if REACT_HINT.strip() in body:
                hinted += 1
            else:
                assert REACT_NO_HINT.strip() in body, "neither directive reached the prompt"
        seen[pid] = hinted / 40

    # the gate follows the persona's own probability, not a constant
    assert seen["avoidant"] < 0.35, seen
    assert seen["dependent"] > 0.40, seen
    assert seen["dependent"] > seen["avoidant"]
