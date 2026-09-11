"""Composing k>=2 misalignments into one query.

One flaw per sentence is the normal case; stacking them is the difficulty dial. What must
not happen is a composite that silently isn't one -- two flaws hitting the same slot, a
literal reading that disagrees with the stated query, or a signature rule that checks only
one of the flaws.
"""

import random

import pytest

from intent_graph.runtime import strategies as st
from intent_graph.runtime.perturb import (
    PerturbationSpec,
    build_mask,
    compose_masks,
    derive_literal_reading,
)

CONDS = (("attr:wool", "=", "wool"), ("attr:red", "=", "red"),
         ("option:0", "=", "large"), ("price_upper", "<=", 50.0))

WITHHOLD = st.BY_ID["insufficient_information"]
FALSIFY = st.BY_ID["factual_error"]
NOISE = st.BY_ID["irrelevant_information"]
MARK = st.BY_ID["referential_ambiguity"]


def _rng():
    return random.Random(7)


# --------------------------------------------------------------- merging
def _withhold_avoiding(slot):
    """A withhold mask that does not touch `slot`, found by seed search rather than skipped:
    a conditional skip here would hide a regression in composition itself."""
    for seed in range(200):
        m = build_mask(CONDS, WITHHOLD, random.Random(seed), min_retained=1, withhold_k_max=1)
        if slot not in m.withheld:
            return m
    raise AssertionError(f"no withhold mask avoids {slot}")


def test_composite_merges_withhold_and_falsify():
    a = _withhold_avoiding("attr:red")
    b = build_mask(CONDS, FALSIFY, _rng(), bogus=("attr:red", "purple"))
    c = compose_masks([a, b], CONDS)
    assert c.is_composite and c.components == (a.strategy_id, b.strategy_id)
    assert set(c.withheld) == set(a.withheld)
    assert c.falsified == (("attr:red", "purple"),)
    assert set(c.kinds) == {st.WITHHOLD, st.FALSIFY}


def test_composite_refuses_to_misstate_a_slot_twice():
    """Withholding and falsifying one slot would tell the renderer to omit it AND assert
    a value for it."""
    a = build_mask(CONDS, WITHHOLD, _rng(), min_retained=1, withhold_k_max=1)
    slot = a.withheld[0]
    b = build_mask(CONDS, FALSIFY, _rng(), bogus=(slot, "bogus"))
    with pytest.raises(ValueError, match="misstated twice"):
        compose_masks([a, b], CONDS)


def test_composite_respects_min_retained():
    two = (("a", "=", 1), ("b", "=", 2))
    a = build_mask(two, WITHHOLD, _rng(), min_retained=1, withhold_k_max=1)
    other = "b" if a.withheld[0] == "a" else "a"
    b = build_mask(two, FALSIFY, _rng(), bogus=(other, "x"))
    with pytest.raises(ValueError, match="min_retained"):
        compose_masks([a, b], two, min_retained=1)


def test_literal_reading_is_derived_from_the_merged_mask():
    """Not intersected from the parts: the merged reading is missing one condition AND
    wrong about another, and only the merged parts know both."""
    a = build_mask(CONDS, WITHHOLD, _rng(), min_retained=1, withhold_k_max=1)
    target = next(s for s, _, _ in CONDS if s not in a.withheld)
    b = build_mask(CONDS, FALSIFY, _rng(), bogus=(target, "WRONG"))
    c = compose_masks([a, b], CONDS)
    reading = dict((s, v) for s, _, v in c.literal_readings[0])
    for slot in a.withheld:
        assert slot not in reading
    assert reading[target] == "WRONG"


def test_retained_excludes_everything_misstated_by_any_component():
    a = build_mask(CONDS, WITHHOLD, _rng(), min_retained=1, withhold_k_max=1)
    target = next(s for s, _, _ in CONDS if s not in a.withheld)
    b = build_mask(CONDS, FALSIFY, _rng(), bogus=(target, "WRONG"))
    c = compose_masks([a, b], CONDS)
    assert not set(c.retained) & (set(a.withheld) | {target})


def test_noise_composes_without_consuming_a_slot():
    """Noise adds distracting text; it hides nothing, so it can always be layered on."""
    a = build_mask(CONDS, WITHHOLD, _rng(), min_retained=1, withhold_k_max=1)
    n = build_mask(CONDS, NOISE, _rng())
    c = compose_masks([a, n], CONDS)
    assert c.noise is True
    assert set(c.withheld) == set(a.withheld)
    assert c.dominant_kind() == st.WITHHOLD


def test_single_spec_composes_to_itself():
    a = build_mask(CONDS, WITHHOLD, _rng(), min_retained=1, withhold_k_max=1)
    assert compose_masks([a], CONDS) is a


# ------------------------------------------------------------- dominance
@pytest.mark.parametrize("kinds,expected", [
    ((st.NOISE,), st.NOISE),
    ((st.OBLIQUE,), st.NOISE),
    ((st.WITHHOLD,), st.WITHHOLD),
    ((st.MARK,), st.WITHHOLD),
    ((st.WITHHOLD, st.NOISE), st.WITHHOLD),
    ((st.FALSIFY, st.NOISE), st.FALSIFY),
    ((st.FALSIFY, st.WITHHOLD), st.FALSIFY),
    ((st.FALSIFY, st.WITHHOLD, st.MARK, st.NOISE), st.FALSIFY),
])
def test_dominant_kind_ordering(kinds, expected):
    """A false premise makes the reading unsatisfiable whatever else is hidden, so it
    governs; withholding only widens, so it beats pure noise."""
    spec = PerturbationSpec(strategy_id="x", family="f", mask_kind="composite", kinds=kinds)
    assert spec.dominant_kind() == expected


def test_derive_literal_reading_drops_withheld_and_marked_keeps_falsified():
    parts = {"withheld": ("attr:wool",), "marked": (("option:0", "deictic"),),
             "falsified": (("attr:red", "purple"),)}
    reading = derive_literal_reading(CONDS, parts)
    slots = {s: v for s, _, v in reading}
    assert "attr:wool" not in slots and "option:0" not in slots
    assert slots["attr:red"] == "purple"
    assert slots["price_upper"] == 50.0


# ------------------------------------------------------------- sampling
def test_sample_strategies_returns_k_and_prefers_distinct_kinds():
    got = st.sample_strategies(CONDS, random.Random(3), k=2, min_retained=1)
    assert len(got) == 2
    assert len({s.id for s in got}) == 2
    assert len({s.mask_kind for s in got}) == 2, "two flaws should be two KINDS of flaw"


def test_sample_strategies_k1_matches_single_sampling():
    got = st.sample_strategies(CONDS, random.Random(11), k=1, min_retained=1)
    want = st.sample_strategy(CONDS, random.Random(11), min_retained=1)
    assert [s.id for s in got] == [want.id]


def test_sample_strategies_degrades_when_conditions_cannot_support_k():
    """Each hiding flaw consumes a condition and min_retained must survive, so a two-slot
    intent cannot carry three hiding flaws. Fewer beats crashing."""
    two = (("a", "=", 1), ("b", "=", 2))
    got = st.sample_strategies(two, random.Random(5), k=4, min_retained=1)
    hiding = [s for s in got if s.mask_kind not in (st.NOISE, st.OBLIQUE)]
    assert len(hiding) <= 1


def test_sample_strategies_is_deterministic_for_a_seed():
    a = [s.id for s in st.sample_strategies(CONDS, random.Random(9), k=3)]
    b = [s.id for s in st.sample_strategies(CONDS, random.Random(9), k=3)]
    assert a == b
