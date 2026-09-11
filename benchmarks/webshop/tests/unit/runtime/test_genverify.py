"""Pipeline v2: generate-then-verify (docs/pipeline-v2.md).

Everything here uses scripted fakes -- the tests pin the ADMISSION machinery, which is the
part that turns free generation back into a verified dataset.
"""

import pytest

from intent_graph.runtime.genverify import (
    Rejected,
    admit,
    check_contract,
    check_text,
    extractions_agree,
    parse_extraction,
)

CONDS = (("attr:machine washable", "=", "machine washable"),
         ("option:0", "=", "light gray"), ("price_upper", "<=", 50))


def ext(**kw):
    base = {"substituted": [], "presupposed": [], "withheld": [], "marked": [],
            "ambiguous": [], "extras": [], "noise": False, "oblique": False}
    base.update(kw)
    return base


class _FakeLLM:
    """Scripted completions, in call order."""
    def __init__(self, responses):
        self.responses = list(responses)

    def complete(self, prompt, role=None, model=None):
        return self.responses.pop(0)


# ------------------------------------------------------------------ contract
def test_contract_requires_exactly_the_requested_class():
    check_contract("factual_error",
                   ext(substituted=[["attr:machine washable", "machine washable", "hand wash only"]]))
    with pytest.raises(Rejected, match="missing_substituted"):
        check_contract("factual_error", ext())
    with pytest.raises(Rejected, match="extra_presupposed"):
        check_contract("factual_error",
                       ext(substituted=[["a", "b", "c"]],
                           presupposed=[{"text": "x", "condition": None}]))


def test_contract_catches_multi_flaw_generation():
    """The live prototype's FE generation stacked three flaws; this is the net that catches
    it -- Drift-bench has no equivalent, which is the whole point of v2."""
    messy = ext(substituted=[["a", "b", "c"]], withheld=["price_upper"])
    with pytest.raises(Rejected, match="extra_withheld"):
        check_contract("factual_error", messy)


def test_fp_contract_is_additive_not_substitutive():
    """The v1 conflation, structurally impossible now: FP must present as an ADDED condition,
    never as a substitution."""
    with pytest.raises(Rejected, match="extra_substituted"):
        check_contract("false_presupposition",
                       ext(substituted=[["a", "b", "c"]],
                           presupposed=[{"text": "x", "condition": None}]))
    check_contract("false_presupposition",
                   ext(presupposed=[{"text": "waterproof", "condition": ["attr:waterproof", "=", "waterproof"]}]))


# ------------------------------------------------------------------ text checks
def test_text_rejects_asserted_value_missing_from_text():
    e = ext(substituted=[["attr:machine washable", "machine washable", "dry clean only"]])
    with pytest.raises(Rejected, match="asserted_value_absent"):
        check_text(CONDS, e, "I want light gray window coverings under 50 dollars.")
    check_text(CONDS, e, "I want dry clean only light gray coverings under 50.")


def test_text_rejects_withheld_value_still_present():
    e = ext(withheld=["option:0"])
    with pytest.raises(Rejected, match="withheld_value_present"):
        check_text(CONDS, e, "I want machine washable light gray coverings.")


# ------------------------------------------------------------------ admission
class _GT:
    def __init__(self, values, kind="purchaseset"):
        self.value = list(values)
        self.kind = kind
        self.hash = "h:" + "|".join(map(str, sorted(values)))
    @property
    def is_empty(self):
        return not self.value
    def cardinality(self):
        return len(self.value)
    def monotonic_view(self):
        return set(self.value)


class _Node:
    def __init__(self, gt):
        self.base = {"cluster": "c"}
        self.conditions = CONDS
        self.ground_truth = gt


class _Adapter:
    """Executes readings against a scripted table keyed by the condition values."""
    def __init__(self, table):
        self.table = table
    def compile(self, base, conds):
        return tuple(sorted((str(s), str(v)) for s, _o, v in conds))
    def execute(self, recipe, session):
        return _GT(self.table.get(recipe, []))


def _key(conds):
    return tuple(sorted((str(s), str(v)) for s, _o, v in conds))


def test_admit_fe_requires_consequential_substitution():
    """2026-08-10 (>=10 floor): the substituted literal must denote a DIFFERENT set than
    the truth -- a dead end (empty) is the special case, a detour (non-empty, different)
    also admits; only a substitution the catalog cannot distinguish is rejected."""
    truth = _GT(["p1", "p2"])
    sub = [["attr:machine washable", "machine washable", "self cleaning"]]
    lit_key = _key((("attr:machine washable", "=", "self cleaning"),) + CONDS[1:])
    dead_end = admit("factual_error", ext(substituted=sub), CONDS, _Node(truth),
                     _Adapter({lit_key: []}), None)
    assert dead_end["signature"]["lit_card"] == 0
    detour = admit("factual_error", ext(substituted=sub), CONDS, _Node(truth),
                   _Adapter({lit_key: ["p9"]}), None)
    assert detour["signature"]["lit_card"] == 1
    with pytest.raises(Rejected, match="substitution_without_consequence"):
        admit("factual_error", ext(substituted=sub), CONDS, _Node(truth),
              _Adapter({lit_key: ["p1", "p2"]}), None)


def test_admit_fp_premise_must_have_consequence():
    """New FP semantics (ruling 2026-08-10): the text assumes a true requirement
    unsatisfiable and settles for a fallback; following the premise must lead somewhere
    other than the truth (the truth's own satisfiability refutes the premise)."""
    truth = _GT(["p1", "p2"])
    pre = [{"text": "since you probably don't have light gray", "condition": None,
            "about": ["option:0", "light gray"], "alternative": ["option:0", "beige"]}]
    k_alt = _key((CONDS[0], ("option:0", "=", "beige"), CONDS[2]))
    out = admit("false_presupposition", ext(presupposed=pre), CONDS, _Node(truth),
                _Adapter({k_alt: ["p9"]}), None)
    assert out["signature"]["lit_card"] == 1
    assert out["signature"]["true_card"] == 2      # non-empty == the premise is false
    with pytest.raises(Rejected, match="without_consequence"):
        admit("false_presupposition", ext(presupposed=pre), CONDS, _Node(truth),
              _Adapter({k_alt: ["p1", "p2"]}), None)


def test_admit_fp_without_fallback_drops_the_doubted_slot():
    truth = _GT(["p1", "p2"])
    pre = [{"text": "as light gray seems unavailable", "condition": None,
            "about": ["option:0", "light gray"], "alternative": None}]
    k_drop = _key((CONDS[0], CONDS[2]))
    out = admit("false_presupposition", ext(presupposed=pre), CONDS, _Node(truth),
                _Adapter({k_drop: ["p1", "p2", "p3"]}), None)
    assert out["signature"]["lit_card"] == 3


def test_admit_fp_requires_a_premise_about_the_true_intent():
    with pytest.raises(Rejected, match="no_presupposition"):
        admit("false_presupposition",
              ext(presupposed=[{"text": "as we discussed", "condition": None}]),
              CONDS, _Node(_GT(["p1"])), _Adapter({}), None)


def test_unexecutable_reading_costs_one_attempt_not_the_cell():
    """A malformed extracted value crashing adapter.compile must surface as Rejected
    (one retry burned), never as a raw ValueError (cell forfeited as v2:error:...)."""
    class _Boom:
        def compile(self, base, conds):
            raise ValueError("could not convert string to float: 'around 35 dollars'")
        def execute(self, recipe, session):
            raise AssertionError("unreachable")
    sub = [["price_upper", "50", "around 35 dollars"]]
    with pytest.raises(Rejected, match="reading_not_executable:ValueError"):
        admit("factual_error", ext(substituted=sub), CONDS, _Node(_GT(["p1"])), _Boom(), None)


def test_admit_extras_must_strictly_narrow():
    """irrelevant_information (ruling 2026-08-10): an extra detail that provably does not
    matter -- heeding it must narrow the candidates (possibly to zero), never widen or
    leave them unchanged."""
    truth = _GT(["p1", "p2"])
    xs = [{"text": "with a free calibration card",
           "condition": ["option:calibration card", "=", "calibration card"]}]
    k_x = _key(CONDS + (("option:calibration card", "=", "calibration card"),))
    out = admit("irrelevant_information", ext(extras=xs), CONDS, _Node(truth),
                _Adapter({k_x: ["p1"]}), None)
    assert out["signature"]["lit_card"] == 1
    out0 = admit("irrelevant_information", ext(extras=xs), CONDS, _Node(truth),
                 _Adapter({k_x: []}), None)        # nonexistent add-on: narrows to zero
    assert out0["signature"]["lit_card"] == 0
    with pytest.raises(Rejected, match="extra_did_not_narrow"):
        admit("irrelevant_information", ext(extras=xs), CONDS, _Node(truth),
              _Adapter({k_x: ["p1", "p2"]}), None)
    with pytest.raises(Rejected, match="extra_not_compilable"):
        admit("irrelevant_information",
              ext(extras=[{"text": "nicely giftwrapped", "condition": None}]),
              CONDS, _Node(truth), _Adapter({}), None)


def test_admit_ambiguous_requires_consequential_disagreement():
    """The new AMBIG rule: both readings execute, sets differ, one contains the truth."""
    truth = _GT(["p1", "p2"])
    amb = [{"surface": "light", "placements": [["option:0", "light gray"],
                                              ["option:0", "lightweight"]]}]
    k_true = _key(CONDS)
    k_decoy = _key((CONDS[0], ("option:0", "=", "lightweight"), CONDS[2]))
    ok_ad = _Adapter({k_true: ["p1", "p2"], k_decoy: ["p9"]})
    out = admit("lexical_ambiguity", ext(ambiguous=amb), CONDS, _Node(truth), ok_ad, None)
    assert out["signature"]["reading_cards"] == [2, 1]
    same_ad = _Adapter({k_true: ["p1", "p2"], k_decoy: ["p1", "p2"]})
    with pytest.raises(Rejected, match="readings_agree"):
        admit("lexical_ambiguity", ext(ambiguous=amb), CONDS, _Node(truth), same_ad, None)
    neither = _Adapter({k_true: ["p8"], k_decoy: ["p9"]})
    with pytest.raises(Rejected, match="no_reading_matches_truth"):
        admit("lexical_ambiguity", ext(ambiguous=amb), CONDS, _Node(truth), neither, None)
    dead_end = _Adapter({k_true: ["p1", "p2"], k_decoy: []})
    # a dead-end second sense is admissible (2026-08-10: an empty-reading rejection bound
    # only lexical and starved it; the agent must still resolve which sense is meant)
    out = admit("lexical_ambiguity", ext(ambiguous=amb), CONDS, _Node(truth), dead_end, None)
    assert out["signature"]["reading_cards"] == [2, 0]


def test_admit_noise_requires_untouched_answer():
    truth = _GT(["p1"])
    ok = admit("contextual_irrelevance", ext(noise=True), CONDS, _Node(truth),
               _Adapter({_key(CONDS): ["p1"]}), None)
    assert ok["signature"]["true_card"] == 1
    with pytest.raises(Rejected, match="noise_changed_answer"):
        admit("contextual_irrelevance", ext(noise=True), CONDS, _Node(truth),
              _Adapter({_key(CONDS): ["p1", "p2"]}), None)


# ------------------------------------------------------------------ agreement
def test_extractor_agreement_keys_on_classes_and_slots_not_wording():
    a = ext(substituted=[["option:0", "light gray", "misty ash"]])
    b = ext(substituted=[["option:0", "light gray", "the misty-ash shade"]])
    assert extractions_agree(a, b)
    c = ext(withheld=["option:0"])
    assert not extractions_agree(a, c)


def test_parse_extraction_maps_fault_type_keys():
    """The diff form is keyed by the fault-type names (compound keys where several types
    share one structural realization); parsing tolerates any pipe arrangement."""
    raw = ('Here you go:\n{"factual_error": [["a","b","c"]],'
           ' "insufficient_information": ["price_upper"], "contextual_irrelevance": false}')
    got = parse_extraction(raw)
    assert got["substituted"] == [["a", "b", "c"]]
    assert got["withheld"] == ["price_upper"]
    assert got["noise"] is False and got["oblique"] is False and got["extras"] == []
    assert parse_extraction("not json at all") is None


def test_parse_extraction_accepts_compound_key_variants():
    for key in ("false_presupposition | irrelevant_information",
                "false_presupposition|irrelevant_information",
                "irrelevant_information"):
        got = parse_extraction('{"%s": [{"text": "x", "condition": null}]}' % key)
        assert got["presupposed"] == [{"text": "x", "condition": None}], key
    for key in ("referential_ambiguity | vagueness_subjectivity | lexical_ambiguity | "
                "syntactic_ambiguity", "lexical_ambiguity"):
        got = parse_extraction('{"%s": [["option:0", "light"]]}' % key)
        assert got["marked"] == [["option:0", "unresolved"]], key


def test_text_checks_fp_premise_names_the_doubted_value():
    pre = [{"text": "since you probably don't have light gray", "condition": None,
            "about": ["option:0", "light gray"], "alternative": ["option:0", "beige"]}]
    check_text(CONDS, ext(presupposed=pre),
               "Since you probably don't have light gray, a beige set could work, "
               "machine washable and under 50.")
    with pytest.raises(Rejected, match="doubted_value_absent"):
        check_text(CONDS, ext(presupposed=pre),
                   "Since you probably don't have that shade, beige could work, "
                   "machine washable and under 50.")


def test_text_checks_extra_detail_must_appear():
    xs = [{"text": "with a free calibration card", "condition": None}]
    with pytest.raises(Rejected, match="extra_detail_absent"):
        check_text(CONDS, ext(extras=xs),
                   "machine washable light gray coverings under 50")


def test_norequest_blacklist_and_conveyed_values_are_mechanical():
    from intent_graph.runtime.genverify import _extract_norequest
    with pytest.raises(Rejected, match="request_language_present"):
        _extract_norequest(CONDS, "I'm looking for machine washable light gray under 50.",
                           _FakeLLM([]), dual_extract=False)
    with pytest.raises(Rejected, match="requirement_not_conveyed"):
        _extract_norequest(CONDS, "I heard machine washable coverings hold up well.",
                           _FakeLLM([]), dual_extract=False)
    got = _extract_norequest(
        CONDS, "I heard machine washable light gray coverings around 50 dollars hold up "
               "well in damp bathrooms.",
        _FakeLLM(['{"request_stated": false, "values_conveyed": [], "other_changes": []}']),
        dual_extract=False)
    assert got["oblique"] is True


def test_contract_rejects_accidental_noise_alongside_content_flaw():
    """k=1: off-topic padding next to a factual error is a second flaw, not flavor."""
    with pytest.raises(Rejected, match="extra_noise"):
        check_contract("factual_error",
                       ext(substituted=[["a", "b", "c"]], noise=True))


def test_malformed_substitution_costs_one_attempt_not_the_cell():
    """A 2-element substituted item must raise Rejected (caught, retried), never a bare
    ValueError (which forfeits the cell's remaining attempts as v2:error:ValueError)."""
    from intent_graph.runtime.genverify import _apply
    with pytest.raises(Rejected, match="extraction_malformed_substitution"):
        _apply(CONDS, substituted=[["option:0", "light gray"]])


def test_text_rejects_unresolved_value_stated_outright():
    """Mechanical guard: if the supposedly-unresolved requirement's exact value appears in
    the text, no extractor opinion can admit it. Syntactic attachment is exempt -- there
    the value is stated by definition."""
    amb = [{"surface": "light", "placements": [["option:0", "light gray"],
                                              ["option:0", "lightweight"]]}]
    with pytest.raises(Rejected, match="unresolved_value_present"):
        check_text(CONDS, ext(ambiguous=amb),
                   "I need a light option, machine washable, light gray, under 50.")
    check_text(CONDS, ext(ambiguous=amb),
               "I need a light option, machine washable, under 50.")
    check_text(CONDS, ext(ambiguous=amb),
               "machine washable in light gray for under 50, as discussed",
               allow_stated_ambiguous=True)


def test_offtopic_ratio_and_conveyed_requirements_are_mechanical():
    from intent_graph.runtime.genverify import _extract_offtopic
    import json as _json
    ok = '{"off_topic_passage": %s, "request_complete": true, "other_changes": []}'
    span = ("My neighbor spent the whole weekend repainting his vintage bicycle frame a "
            "deep forest green and would not stop talking about brake cables, saddle "
            "leather, and the merits of steel over carbon.")
    long_request = ("I need machine washable light gray coverings under 50 for the "
                    "guest room, the hallway, the study, both bathrooms, the kitchen "
                    "window over the sink, and the two odd-sized dormers upstairs, "
                    "each measured twice to be safe.")
    with pytest.raises(Rejected, match="off_topic_shorter_than_request"):
        _extract_offtopic(CONDS, span + " " + long_request,
                          _FakeLLM([ok % _json.dumps(span)]), dual_extract=False)
    missing_value = span + " I need machine washable coverings under 50."
    with pytest.raises(Rejected, match="requirement_not_conveyed"):
        _extract_offtopic(CONDS, missing_value,
                          _FakeLLM([ok % _json.dumps(span)]), dual_extract=False)


def test_contextual_gets_the_raised_word_cap():
    """220 words for contextual_irrelevance, 130 for everyone else."""
    long_text = " ".join(["word"] * 150)
    with pytest.raises(Rejected, match="too_long"):
        check_text(CONDS, ext(noise=True), long_text)
    check_text((), ext(noise=True), long_text, max_words=220)


def test_syntactic_extractor_handles_stated_values_and_plausibility_veto():
    """The dedicated attachment form: every value stated is EXPECTED; rejection needs
    either no unclear attachment, slot disagreement, or BOTH annotators denying
    plausibility (single denial tolerated -- the author 2026-08-10)."""
    from intent_graph.runtime.genverify import _extract_syntactic
    ok = ('{"ambiguous_phrase": "under 50", "target_slot": "price_upper",'
          ' "attachments": ["the coverings", "the installation service"],'
          ' "both_attachments_plausible": %s, "other_changes": []}')
    got = _extract_syntactic(CONDS, "irrelevant", _FakeLLM([ok % "true", ok % "false"]),
                             dual_extract=True)
    assert got["ambiguous"][0]["placements"] == [["price_upper", "50"],
                                                 ["price_upper", None]]
    with pytest.raises(Rejected, match="implausible_reading"):
        _extract_syntactic(CONDS, "irrelevant", _FakeLLM([ok % "false", ok % "false"]),
                           dual_extract=True)
    none = ('{"ambiguous_phrase": null, "target_slot": null, "attachments": [],'
            ' "both_attachments_plausible": false, "other_changes": []}')
    with pytest.raises(Rejected, match="no_unclear_attachment"):
        _extract_syntactic(CONDS, "irrelevant", _FakeLLM([none]), dual_extract=False)
    other_slot = ok.replace("price_upper", "option:0")
    with pytest.raises(Rejected, match="extractor_disagreement"):
        _extract_syntactic(CONDS, "irrelevant",
                           _FakeLLM([ok % "true", other_slot % "true"]),
                           dual_extract=True)


def test_starved_types_get_deeper_retry_budget():
    from intent_graph.runtime.genverify import EXTRA_ATTEMPTS
    assert EXTRA_ATTEMPTS == {"factual_error": 8, "lexical_ambiguity": 8,
                              "syntactic_ambiguity": 8}


def test_offtopic_span_checks_are_mechanical():
    from intent_graph.runtime.genverify import _extract_offtopic
    span = ("My neighbor spent the whole weekend repainting his vintage bicycle frame a "
            "deep forest green and would not stop talking about brake cables, saddle "
            "leather, and the merits of steel over carbon.")
    text = span + " I need machine washable light gray coverings under 50."
    ok = '{"off_topic_passage": %s, "request_complete": true, "other_changes": []}'
    import json as _json
    got = _extract_offtopic(CONDS, text, _FakeLLM([ok % _json.dumps(span)]),
                            dual_extract=False)
    assert got["noise"] is True and got["offtopic_span"] == span
    with pytest.raises(Rejected, match="off_topic_too_short"):
        _extract_offtopic(CONDS, text, _FakeLLM([ok % '"a short aside"']),
                          dual_extract=False)
    leaky = span + " Also I keep thinking about light gray paint."
    with pytest.raises(Rejected, match="off_topic_mentions_requirement"):
        _extract_offtopic(CONDS, leaky + " I need machine washable coverings under 50.",
                          _FakeLLM([ok % _json.dumps(leaky)]), dual_extract=False)


def test_withheld_short_values_are_testable_and_reject_when_present():
    """The polarity bug: '30' was 'too short to test', reported PRESENT, and every
    price-withholding sample died. Word boundaries make short values safe to test."""
    e = ext(withheld=["price_upper"])
    with pytest.raises(Rejected, match="withheld_value_present"):
        check_text(CONDS, e, "machine washable light gray, budget is 50 dollars")
    check_text(CONDS, e, "machine washable light gray coverings please")   # price absent: ok


def test_agreement_tolerates_marked_vs_ambiguous_on_the_same_slot():
    """A genuinely fuzzy boundary between models; both mean 'mentioned but unresolved'."""
    a = ext(ambiguous=[{"surface": "light", "placements": [["option:0", "light gray"],
                                                          ["option:0", "lightweight"]]}])
    b = ext(marked=[["option:0", "vague"]])
    assert extractions_agree(a, b)
    c = ext(withheld=["option:0"])
    assert not extractions_agree(a, c), "unbound vs absent is a REAL disagreement"


def test_normalizer_drops_phantom_product_type_presupposition():
    """A sentence needs a noun; "a shirt" is not an added requirement."""
    from intent_graph.runtime.genverify import normalize_extraction
    e = ext(substituted=[["attr:machine washable", "machine washable", "dry clean"]],
            presupposed=[{"text": "a shirt", "condition": ["attr:product type", "=", "shirt"]}])
    n = normalize_extraction(e, CONDS)
    assert n["presupposed"] == []
    assert n["substituted"][0][0] == "attr:machine washable"


def test_normalizer_canonicalizes_renamed_slots_by_true_value():
    """One extractor said attr:material for the substitution another called
    attr:polyester cotton -- same flaw, cosmetic name, and it broke agreement."""
    from intent_graph.runtime.genverify import extractions_agree, normalize_extraction
    a = ext(substituted=[["attr:machine washable", "machine washable", "dry clean"]])
    b = ext(substituted=[["attr:care method", "machine washable", "dry clean"]])
    nb = normalize_extraction(b, CONDS)
    assert nb["substituted"][0][0] == "attr:machine washable"
    assert extractions_agree(normalize_extraction(a, CONDS), nb)


# ---------------------------------------------------- extractor shape guards
# The extractors are language models. Asked for [["cabin", "economy"], ...] a model will
# sometimes return ["cabin", ...]. Indexing a bare string does NOT raise -- "cabin"[0] is
# "c" -- so an unguarded row silently became a one-letter slot name. One sibling site did
# raise, and the worker's generic handler then tallied AttributeError beside genuine
# rejections, so a software failure was indistinguishable from a considered one.
def test_a_bare_string_never_becomes_a_one_letter_slot():
    from intent_graph.runtime import genverify as gv
    ext = {"substituted": ["cabin", ["cabin", "economy"]],
           "marked": ["baggage", ["baggage"]],
           "withheld": [], "presupposed": [], "ambiguous": []}
    rows = gv._rows(ext["substituted"])
    assert rows == [["cabin", "economy"]], rows
    assert all(len(str(r[0])) > 1 for r in rows), "a single character survived as a slot"
    assert gv._rows(ext["marked"]) == [["baggage"]]


def test_a_bare_string_in_an_object_slot_is_dropped_not_crashed():
    from intent_graph.runtime import genverify as gv
    # `ambiguous` rows must be objects; a bare string used to reach .get() and raise
    assert gv._objs(["oops", {"placements": [["cabin", "economy"]]}]) == \
        [{"placements": [["cabin", "economy"]]}]
    assert gv._objs(None) == []
    assert gv._rows(None) == []


def test_shape_guards_keep_well_formed_rows_untouched():
    from intent_graph.runtime import genverify as gv
    good = [["a", 1], ("b", 2)]
    assert gv._rows(good) == good
    objs = [{"x": 1}, {"y": 2}]
    assert gv._objs(objs) == objs


def test_extra_details_accepts_a_bare_string_as_the_detail():
    """A model that returns the detail as a plain string has made a formatting slip, not
    omitted the detail. Coerce it rather than crashing on `.get`."""
    from intent_graph.runtime import genverify as gv
    import inspect
    src = inspect.getsource(gv._extract_extras)
    assert 'isinstance(e, str)' in src, "the bare-string coercion is gone"
    assert 'e = {"text": e, "condition": None}' in src
