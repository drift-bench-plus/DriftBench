"""Mask construction, literal readings, fidelity, and signature classification."""

import random

import pytest

from intent_graph.models import GroundTruth
from intent_graph.runtime import strategies as st
from intent_graph.runtime.llm import FakeLLM
from intent_graph.runtime.perturb import (
    PerturbationSpec,
    Unperturbable,
    bogus_candidates,
    build_mask,
    build_render_prompt,
    check_fidelity,
    derive_literal_reading,
    internal_syntax_leaks,
    perturb,
    template_render,
)
from intent_graph.runtime.signature import classify_signature

CONDS = (("attr:machine wash", "=", "machine wash"),
         ("option:0", "=", "heather charcoal"),
         ("price_upper", "<=", 50.0))

CONFIG = {"runtime": {"min_retained_conditions": 1, "withhold_k_max": 2,
                      "falsify_max_attempts": 5, "render_max_attempts": 3}}


def rng(seed=7):
    return random.Random(seed)


# --------------------------------------------------------------- applicability
def test_one_condition_node_cannot_withhold():
    one = (("attr:x", "=", "x"),)
    assert not st.applicable("insufficient_information", one)
    assert st.applicable("irrelevant_information", one)      # noise always works
    assert st.applicable("factual_error", one)


def test_zero_condition_node_supports_only_content_free_strategies():
    ids = {s.id for s in st.applicable_strategies(())}
    assert ids == {"irrelevant_information", "contextual_irrelevance", "indirect_intent"}


def test_vagueness_requires_an_ordered_slot():
    unordered = (("attr:a", "=", "a"), ("attr:b", "=", "b"))
    assert not st.applicable("vagueness_subjectivity", unordered)
    assert st.applicable("vagueness_subjectivity", CONDS)     # price_upper is ordered


def test_tool_capability_mismatch_stays_unsupported():
    """Dropped rather than faked for WebShop: the flaw is about choosing the wrong INSTRUMENT
    ("use the weather API to query the table") and WebShop offers exactly one (search+click),
    so there is nothing to mismatch. Its natural home is dbbench."""
    assert not st.applicable("tool_capability_mismatch", CONDS)
    assert not st.BY_ID["tool_capability_mismatch"].supported


def test_the_two_ambiguity_flaws_are_supported_as_mark_kinds():
    """They leave a slot MENTIONED BUT UNBOUND, which is exactly MARK -- the same mask,
    literal reading and signature rule as referential_ambiguity, differing only in wording."""
    for sid, kind in (("lexical_ambiguity", "polysemous"),
                      ("syntactic_ambiguity", "attachment")):
        s = st.BY_ID[sid]
        assert s.supported and s.mask_kind == st.MARK
        assert s.marker_kind == kind
        assert st.applicable(sid, CONDS), sid


def test_every_marker_kind_has_a_render_instruction():
    """A marker kind with no instruction would silently render as an unmarked query."""
    from intent_graph.runtime.perturb import _DEICTIC, marker_kind_for
    for s in st.SUPPORTED:
        if s.mask_kind == st.MARK:
            assert marker_kind_for(s) in _DEICTIC, s.id


def test_expression_family_now_has_four_strategies():
    """The reason for adding them: expression was down to 2 of 4 families' worth of coverage."""
    expr = [s for s in st.SUPPORTED if s.family == "expression"]
    assert len(expr) == 4, [s.id for s in expr]


def test_family_first_sampling_balances_families():
    counts = {}
    r = rng()
    for _ in range(4000):
        s = st.sample_strategy(CONDS, r)
        counts[s.family] = counts.get(s.family, 0) + 1
    # four applicable families here; uniform-over-strategies would skew expression 4:2:2:2
    for fam, n in counts.items():
        assert 0.15 < n / 4000 < 0.35, (fam, n)


# ------------------------------------------------------------------ masks
def test_withhold_respects_min_retained_and_k_max():
    for seed in range(30):
        spec = build_mask(CONDS, st.BY_ID["insufficient_information"], rng(seed),
                          min_retained=1, withhold_k_max=2)
        assert 1 <= len(spec.withheld) <= 2
        assert len(spec.retained) >= 1


def test_withhold_impossible_raises():
    with pytest.raises(ValueError, match="nothing can be withheld"):
        build_mask((("a", "=", 1),), st.BY_ID["insufficient_information"], rng())


def test_literal_reading_drops_withheld_and_substitutes_falsified():
    parts = {"withheld": ("option:0",), "falsified": (("attr:machine wash", "BOGUS"),)}
    reading = derive_literal_reading(CONDS, parts)
    slots = {s: v for s, _, v in reading}
    assert "option:0" not in slots                       # withheld -> absent
    assert slots["attr:machine wash"] == "BOGUS"         # falsified -> bogus value
    assert slots["price_upper"] == 50.0                  # untouched


def test_marked_slot_leaves_the_reading_unconstrained():
    spec = build_mask(CONDS, st.BY_ID["referential_ambiguity"], rng(),
                      marker=("option:0", "deictic"))
    reading_slots = {s for s, _, _ in spec.literal_readings[0]}
    assert "option:0" not in reading_slots               # mentioned but unbound
    assert spec.hidden_slots == ("option:0",)


def test_noise_and_oblique_leave_the_reading_identical():
    for sid in ("irrelevant_information", "indirect_intent"):
        spec = build_mask(CONDS, st.BY_ID[sid], rng())
        assert spec.literal_readings[0] == CONDS
        assert spec.hidden_slots == ()


def test_bogus_candidates_never_include_the_true_value():
    doms = {"attr:machine wash": ["machine wash", "hand wash"], "option:0": ["black", "red"]}
    cands = bogus_candidates("attr:machine wash", "machine wash", CONDS, doms, rng(), limit=5)
    assert cands and all(str(c) != "machine wash" for c in cands)


# --------------------------------------------------------------- fidelity
def _spec(**kw):
    base = dict(strategy_id="insufficient_information", family="parameter",
                mask_kind=st.WITHHOLD, retained=("attr:machine wash",),
                withheld=("option:0",), literal_readings=((),))
    base.update(kw)
    return PerturbationSpec(**base)


def test_fidelity_accepts_a_compliant_query():
    ok, notes = check_fidelity("I want something machine wash please", _spec(), CONDS)
    assert ok, notes


def test_fidelity_rejects_a_leaked_withheld_value():
    ok, notes = check_fidelity("machine wash in heather charcoal please", _spec(), CONDS)
    assert not ok and any(n.startswith("withheld_leaked") for n in notes)


def test_fidelity_rejects_a_missing_retained_value():
    ok, notes = check_fidelity("just get me anything at all", _spec(), CONDS)
    assert not ok and any(n.startswith("retained_missing") for n in notes)


def test_fidelity_skips_values_shorter_than_three_chars():
    """("find:type","=","f") must not make the letter 'f' forbidden in English."""
    conds = (("find:type", "=", "f"), ("find:mtime", "<", "7"))
    spec = _spec(retained=("find:mtime",), withheld=("find:type",))
    ok, notes = check_fidelity("how many files were modified in the last 7 days", spec, conds)
    assert ok and any(n.startswith("skipped_short_withheld") for n in notes)


def test_fidelity_rejects_too_short():
    ok, notes = check_fidelity("machine wash", _spec(), CONDS)
    assert not ok and "too_short" in notes


def test_fidelity_requires_the_bogus_value_and_forbids_the_true_one():
    spec = _spec(mask_kind=st.FALSIFY, withheld=(), retained=("option:0", "price_upper"),
                 falsified=(("attr:machine wash", "dry clean only"),))
    ok, _ = check_fidelity("get me a dry clean only one in heather charcoal under 50", spec, CONDS)
    assert ok
    bad, notes = check_fidelity("get me a machine wash one in heather charcoal under 50",
                                spec, CONDS)
    assert not bad and any("falsified" in n for n in notes)


# ------------------------------------------------------------------ rendering
def test_render_prompt_never_contains_the_base():
    """WebShop's base holds the target asin and product name — leaking it gives the answer."""
    spec = build_mask(CONDS, st.BY_ID["insufficient_information"], rng())
    prompt = build_render_prompt(spec, CONDS, st.BY_ID["insufficient_information"],
                                 surface="shopping")
    assert "B09HX5CD2D" not in prompt and "asin" not in prompt.lower()
    for slot in spec.withheld:
        val = dict((s, v) for s, _, v in CONDS)[slot]
        assert f'"{val}"' not in prompt      # the withheld value is never quoted as a target


def test_render_prompt_states_every_instruction():
    spec = build_mask(CONDS, st.BY_ID["insufficient_information"], rng(),
                      min_retained=1, withhold_k_max=1)
    p = build_render_prompt(spec, CONDS, st.BY_ID["insufficient_information"], surface="s")
    assert "do NOT mention" in p and "say plainly" in p
    # no internal identifier may reach the prompt: whatever is in it gets copied into the query
    assert "attr:" not in p and "price_upper" not in p and "option:0" not in p


def test_perturb_retries_then_falls_back_to_template():
    """A model that keeps leaking gets three tries, then a model-free rendering."""
    leaky = FakeLLM(responder=lambda *a: "machine wash in heather charcoal under 50 dollars")
    spec_strategy = st.BY_ID["insufficient_information"]
    res = perturb(node=None, conditions=CONDS, strategy=spec_strategy, rng=rng(),
                  llm=leaky, surface="s", config=CONFIG)
    assert "template_fallback" in res.fidelity_notes
    ok, _ = check_fidelity(res.query, res.spec, CONDS)
    assert ok


def _compliant_responder(conditions):
    """A stand-in for a model that follows the mask: it states exactly the values the
    prompt asked it to state, and nothing else."""
    values = {s: str(v) for s, _, v in conditions}

    def respond(role, system, prompt):
        # the prompt now names requirements in words, not as `slot = "value"`, so key off the
        # quoted value it was told to state
        stated = [v for v in values.values() if 'you want' in prompt and f'"{v}"' in prompt]
        return "I would like something " + ", ".join(stated or ["please help"]) + " thanks"
    return respond


@pytest.mark.parametrize("seed", range(8))
def test_perturb_accepts_a_compliant_rendering_on_the_first_try(seed):
    good = FakeLLM(responder=_compliant_responder(CONDS))
    res = perturb(node=None, conditions=CONDS, strategy=st.BY_ID["insufficient_information"],
                  rng=random.Random(seed), llm=good, surface="s", config=CONFIG)
    assert res.attempts == 1, res.fidelity_notes
    assert "template_fallback" not in res.fidelity_notes


def test_perturb_raises_when_the_signature_never_passes():
    llm = FakeLLM(default="anything at all here")
    with pytest.raises(Unperturbable, match="signature check"):
        perturb(node=None, conditions=CONDS, strategy=st.BY_ID["insufficient_information"],
                rng=rng(), llm=llm, surface="s", config=CONFIG,
                verify=lambda spec: {"ok": False, "verdict": "nope"})


def test_template_render_is_always_fidelity_clean():
    for sid in ("insufficient_information", "irrelevant_information", "referential_ambiguity"):
        s = st.BY_ID[sid]
        spec = build_mask(CONDS, s, rng(), marker=("option:0", "deictic")
                          if s.mask_kind == st.MARK else None)
        ok, notes = check_fidelity(template_render(spec, CONDS), spec, CONDS)
        assert ok, (sid, notes)


# ------------------------------------------------------- signature semantics
ROWSET_TRUE = GroundTruth.rowset([["a"], ["b"]])


def _sig(spec_kind, lit, *, extensional=True, true_gt=ROWSET_TRUE, pristine=None):
    spec = _spec(mask_kind=spec_kind)
    return classify_signature(spec, true_gt, lit, extensional=extensional,
                              pristine_hash=pristine)


def test_withhold_must_widen_an_extensional_answer():
    assert _sig(st.WITHHOLD, GroundTruth.rowset([["a"], ["b"], ["c"]]))["ok"]
    assert not _sig(st.WITHHOLD, ROWSET_TRUE)["ok"]                  # unchanged
    assert not _sig(st.WITHHOLD, GroundTruth.rowset([["a"]]))["ok"]  # narrowed


def test_withhold_on_a_computed_answer_only_needs_to_move():
    """An aggregate has no subset relation: `SELECT COUNT(*)` just changes value."""
    r = _sig(st.WITHHOLD, GroundTruth.rowset([["9"]]), extensional=False,
             true_gt=GroundTruth.rowset([["5"]]))
    assert r["ok"]
    assert not _sig(st.WITHHOLD, GroundTruth.rowset([["5"]]), extensional=False,
                    true_gt=GroundTruth.rowset([["5"]]))["ok"]


def test_noise_must_not_move_the_answer():
    assert _sig(st.NOISE, ROWSET_TRUE)["ok"]
    assert not _sig(st.NOISE, GroundTruth.rowset([["a"]]))["ok"]


def test_false_premise_must_empty_an_extensional_answer():
    assert _sig(st.FALSIFY, GroundTruth.rowset([]))["ok"]
    assert not _sig(st.FALSIFY, ROWSET_TRUE)["ok"]


def test_false_premise_on_a_scalar_accepts_zero_not_only_empty():
    """`find ... | wc -l` prints "0" when nothing matches — that is not is_empty."""
    r = _sig(st.FALSIFY, GroundTruth.scalar("0"), extensional=False,
             true_gt=GroundTruth.scalar("9"))
    assert r["ok"]
    assert not _sig(st.FALSIFY, GroundTruth.scalar("9"), extensional=False,
                    true_gt=GroundTruth.scalar("9"))["ok"]


def test_false_premise_on_a_statehash_compares_against_pristine():
    """A write that matched nothing leaves the table exactly as it was."""
    true_gt = GroundTruth.statehash("changed")
    assert _sig(st.FALSIFY, GroundTruth.statehash("pristine"), extensional=False,
                true_gt=true_gt, pristine="pristine")["ok"]
    assert not _sig(st.FALSIFY, GroundTruth.statehash("somethingelse"), extensional=False,
                    true_gt=true_gt, pristine="pristine")["ok"]


def test_statehash_without_a_pristine_reference_is_not_checkable():
    r = _sig(st.FALSIFY, GroundTruth.statehash("x"), extensional=False,
             true_gt=GroundTruth.statehash("y"))
    assert r["ok"] and r["verdict"] == "not_checkable"


# --------------------------------------------- no internal syntax in user text
LEAKY = [
    'attr:high fructose = "high fructose", option:0 = "citrus", price_upper = "120"',
    "I need where price_upper is 50",
    "give me slot_2 please",
    'set item_count = "4"',
]
CLEAN = [
    "I need some high fructose citrus tonic water under 120 dollars.",
    "Looking for a machine washable small shirt, ideally under fifty dollars.",
    "Can you find me something in size small? My budget is about 50.",
]


@pytest.mark.parametrize("q", LEAKY)
def test_internal_syntax_is_detected(q):
    """A query carrying field names is not a user utterance, so it is not usable data.
    check_fidelity used to accept these because every required VALUE was present."""
    assert internal_syntax_leaks(q), q


@pytest.mark.parametrize("q", CLEAN)
def test_natural_queries_are_not_flagged(q):
    assert internal_syntax_leaks(q) == [], q


def test_fidelity_rejects_a_query_that_leaks_identifiers():
    spec = build_mask(CONDS, st.BY_ID["insufficient_information"], rng(), withhold_k_max=1)
    stated = " ".join(f'{s} = "{v}"' for s, _, v in CONDS if s in spec.retained)
    ok, notes = check_fidelity(f"please find {stated}", spec, CONDS)
    assert not ok
    assert any(n.startswith("internal_syntax_leaked") for n in notes), notes


def test_render_prompt_never_quotes_a_withheld_value_even_via_its_slot_name():
    """The regression this pins: describing `attr:machine wash` as 'the feature "machine
    wash"' put the withheld value in the prompt -- as part of the instruction to omit it."""
    for seed in range(12):
        spec = build_mask(CONDS, st.BY_ID["insufficient_information"], random.Random(seed),
                          withhold_k_max=2)
        p = build_render_prompt(spec, CONDS, [st.BY_ID["insufficient_information"]],
                                surface="s")
        for slot in spec.withheld:
            value = next(str(v) for s, _, v in CONDS if s == slot)
            assert f'"{value}"' not in p, (slot, value)
            assert slot not in p, slot


def test_default_slot_phrase_never_returns_an_identifier():
    from intent_graph.runtime.perturb import default_slot_phrase
    for slot in ("price_upper", "attr:high fructose", "option:0", "where:round",
                 "set:name", "find:size", "grep:pattern"):
        phrase = default_slot_phrase(slot)
        assert ":" not in phrase and "_" not in phrase, (slot, phrase)
        assert not internal_syntax_leaks(phrase), (slot, phrase)


def test_attr_phrase_is_value_free():
    """On WebShop the slot name is the value, so the phrase must not contain it."""
    from intent_graph.adapters.webshop import WebShopAdapter
    from intent_graph.runtime.perturb import default_slot_phrase
    for fn in (default_slot_phrase, WebShopAdapter.slot_phrase):
        assert "high fructose" not in fn("attr:high fructose")


def test_render_context_is_included_but_the_base_is_not():
    """Topical context stops the model inventing a product type (a tonic-water intent came
    out as "stainless steel drinking straws"), which is a falsehood the mask never records."""
    spec = build_mask(CONDS, st.BY_ID["insufficient_information"], rng(), withhold_k_max=1)
    p = build_render_prompt(spec, CONDS, [st.BY_ID["insufficient_information"]],
                            surface="s", context='The user is shopping in the "tonic water" section.')
    assert "tonic water" in p
    assert "Invent nothing" in p


def test_fidelity_rejects_the_model_reasoning_aloud():
    """Observed live: a render returned ~1,000 words of "Wait no, the instruction says claim
    the required feature is..." and fidelity PASSED it, because every required value appeared
    somewhere in the monologue."""
    spec = build_mask(CONDS, st.BY_ID["insufficient_information"], rng(), withhold_k_max=1)
    stated = " ".join(str(v) for s_, _, v in CONDS if s_ in spec.retained)
    monologue = (f"I need {stated}. Wait no, wait the instruction says claim the required "
                 "feature is deliberately wrong, that is the factual error, wait let me "
                 "phrase this naturally, wait no ") * 6
    ok, notes = check_fidelity(monologue, spec, CONDS)
    assert not ok
    assert any(n.startswith("meta_commentary") for n in notes), notes
    assert any(n.startswith("too_long") for n in notes), notes


def test_fidelity_accepts_a_normal_length_query():
    spec = build_mask(CONDS, st.BY_ID["insufficient_information"], rng(), withhold_k_max=1)
    stated = ", ".join(str(v) for s_, _, v in CONDS if s_ in spec.retained)
    ok, notes = check_fidelity(f"Hi, I am looking for something with {stated}. Can you help?",
                               spec, CONDS)
    assert ok, notes


def test_webshop_render_context_names_the_cluster_and_nothing_else():
    from intent_graph.adapters.webshop import WebShopAdapter
    base = {"asin": "B09HX5CD2D", "cluster": "alcoholic beverages",
            "name": "Secret Product Name 12-Pack", "product_category": "Grocery > X > Y"}
    ctx = WebShopAdapter.render_context(base)
    assert "alcoholic beverages" in ctx
    for leak in ("B09HX5CD2D", "Secret Product Name", "Grocery"):
        assert leak not in ctx, leak


# ------------------------------------------------- the bogus-value ladder
def test_falsify_prefers_slots_whose_falsification_can_be_verified():
    """On WebShop an option: slot can never carry a false premise -- the reward
    nearest-matches whatever the product stocks, so something always satisfies the claim.
    Falsifying one randomly chosen slot is why FALSIFY managed only 4 of 14 graphs."""
    from intent_graph.runtime.perturb import _falsify_rank
    assert _falsify_rank("attr:wool") < _falsify_rank("price_upper")
    assert _falsify_rank("price_upper") < _falsify_rank("option:0")
    assert _falsify_rank("where:round") < _falsify_rank("option:2")


def test_falsify_candidates_cover_more_than_one_slot():
    """The regression: a single slot choice meant a whole graph was lost whenever that slot
    happened to be unfalsifiable."""
    from intent_graph.runtime.perturb import _candidate_specs
    specs = _candidate_specs(CONDS, st.BY_ID["factual_error"], rng(), min_retained=1,
                             k_max=2, falsify_tries=10,
                             domains={s: ["a", "b", "c"] for s, _, _ in CONDS})
    slots = {slot for sp in specs for slot, _ in sp.falsified}
    assert len(slots) > 1, f"only falsified {slots}"


def test_ladder_is_ordered_not_shuffled_away():
    """Foreign values come first (plausible AND unlikely to hold); shuffling the whole ladder
    and truncating discarded the reliable candidates at random."""
    from intent_graph.runtime.perturb import bogus_candidates
    foreign = [f"foreign {i}" for i in range(12)]
    got = bogus_candidates("attr:wool", "wool", CONDS,
                           {"attr:wool": ["local a", "local b"]}, rng(),
                           limit=6, foreign=foreign)
    assert any(str(g).startswith("foreign") for g in got)


def test_ladder_always_reserves_room_for_a_guaranteed_mutation():
    """Priority order alone was not enough: the foreign rung could fill the whole budget, so
    the guaranteed-unsatisfiable rung became unreachable again."""
    from intent_graph.runtime.perturb import bogus_candidates
    foreign = [f"foreign {i}" for i in range(30)]
    for limit in (3, 5, 10):
        got = [str(g) for g in bogus_candidates(
            "attr:high fructose", "high fructose", CONDS, {}, rng(),
            limit=limit, foreign=foreign)]
        assert any(g.startswith("high fructose") for g in got), (limit, got)
        assert len(got) <= limit


def test_ladder_never_offers_the_true_value():
    from intent_graph.runtime.perturb import bogus_candidates
    got = bogus_candidates("attr:wool", "wool", CONDS,
                           {"attr:wool": ["wool", "cotton"]}, rng(), limit=8,
                           foreign=["wool", "silk"])
    assert "wool" not in [str(g) for g in got]


def test_speakability_filter_rejects_junk_attribute_tokens():
    """A false premise only tests something if it is believable: "the product feature '1'"
    reads as corrupted text, not as a person who is mistaken."""
    from intent_graph.adapters.webshop import _is_speakable_attribute as ok
    for junk in ("1", "#1", "5x3fu", "10 inches", "16.9 fl oz", "x", "", "DUMMY_ATTR"):
        assert not ok(junk), junk
    for good in ("machine washable", "stainless steel", "gluten free", "clinically proven"):
        assert ok(good), good
