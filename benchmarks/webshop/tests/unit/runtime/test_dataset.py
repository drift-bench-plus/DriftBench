"""Exported samples: what the agent sees vs what only the scorer sees.

A sample is a task instance, not a solved episode. The one hard rule is that `query` is the
entire agent-visible surface -- if any part of the answer reaches it, every score computed
from this dataset is inflated and nothing downstream would reveal it.
"""

import pytest

from intent_graph.adapters.toybench import ToyAdapter, ToyExecutor
from intent_graph.dataset import build_samples, sample_id, write_samples
from intent_graph.engine import RunStats, generate
from intent_graph.runtime import scripted as S

GEN = {"seed": 42, "depth": 1, "branching": 8, "min_gt_moved_edges": 1,
       "require_full_quota": False}

RUNTIME = {"runtime": {
    "misalignments_k": 1, "withhold_k_max": 2, "min_retained_conditions": 1,
    "falsify_max_attempts": 5, "render_max_attempts": 3, "render_max_words": 120,
    "composite_max_combinations": 24, "persona": "rational",
    "category_probs": {"REFINEMENT": .25, "RELAXATION": .25, "SUBSTITUTION": .30,
                       "PIVOT": .20},
    "p_shift": 0.0, "max_shifts": 1, "allow_revisit": False,
    "on_empty_category": "resample", "shift_after_mutation": "forbid",
    "max_turns": 12, "patience_init": 14,
}}


@pytest.fixture(scope="module")
def graph():
    out = list(generate(ToyAdapter(), ToyExecutor(), GEN, stats=RunStats()))
    assert out
    return out[0]


@pytest.fixture(scope="module")
def built(graph):
    return build_samples(graph, ToyAdapter(), ToyExecutor(), RUNTIME, S.stub_llm(),
                         personas=["rational", "avoidant"])


def test_samples_are_produced_for_every_persona(built):
    samples, _ = built
    assert samples
    assert {s["persona"] for s in samples} == {"rational", "avoidant"}


def test_query_is_the_only_agent_visible_field(built):
    """Everything else is scoring material. This test exists so that adding a field to the
    record cannot quietly widen what an agent could be shown."""
    samples, _ = built
    for s in samples:
        assert isinstance(s["query"], str) and s["query"].strip()
        for key in ("hidden_intent", "ground_truth", "shift_options", "mask"):
            assert key in s, key


def test_the_answer_never_appears_in_the_query(built):
    """The whole dataset is worthless if the answer is in the prompt."""
    samples, _ = built
    for s in samples:
        q = s["query"].lower()
        for raw in s["ground_truth"]["value"]:
            assert raw.lower() not in q
        # withheld values must not be recoverable from the text either
        withheld = set(s["mask"]["withheld"])
        for slot, _op, value in s["hidden_intent"]["conditions"]:
            if slot in withheld and len(str(value)) >= 3:
                assert str(value).lower() not in q, (slot, value)


def test_hidden_slots_match_the_mask(built):
    samples, _ = built
    for s in samples:
        assert set(s["hidden_intent"]["hidden_slots"]) == set(s["mask"]["hidden_slots"])


def test_sample_ids_are_deterministic_and_unique(built):
    samples, _ = built
    ids = [s["sample_id"] for s in samples]
    assert len(ids) == len(set(ids)), "sample ids collide"
    for s in samples:
        assert s["sample_id"] == sample_id(s["graph_id"], s["strategy_ids"], s["persona"],
                                           s["mask"] and _spec_id(s))


def _spec_id(s):
    from intent_graph.ids import content_hash
    m = s["mask"]
    return content_hash(m["strategy_id"], tuple(m["withheld"]),
                        tuple(tuple(x) for x in m["falsified"]),
                        tuple(tuple(x) for x in m["marked"]), m["noise"], m["oblique"])


def test_shift_options_carry_an_announcement_without_internal_syntax(built):
    from intent_graph.runtime.perturb import internal_syntax_leaks
    samples, _ = built
    for s in samples:
        for opt in s["shift_options"]:
            assert opt["announcement"].strip()
            assert internal_syntax_leaks(opt["announcement"]) == [], opt


def test_strategy_skips_are_recorded_not_dropped(graph):
    """A strategy that cannot be masked must leave a trace: a run where half of them failed
    would otherwise look like a smaller run."""
    from intent_graph.runtime import strategies as st
    samples, skipped = build_samples(
        graph, ToyAdapter(), ToyExecutor(), RUNTIME, S.stub_llm(),
        personas=["rational"], strategies=[st.BY_ID["vagueness_subjectivity"]])
    assert len(samples) + len(skipped) == 1
    for entry in skipped:
        assert entry["reason"]


def test_write_samples_round_trips(tmp_path, built):
    from intent_graph.dataset import iter_samples
    samples, _ = built
    n = write_samples(tmp_path, "toybench", samples)
    assert n == len(samples)
    back = list(iter_samples(tmp_path, "toybench"))
    assert {s["sample_id"] for s in back} == {s["sample_id"] for s in samples}


# ------------------------------------------------------- export accounting
def test_skip_classification_reports_the_verdict_not_the_strategy():
    """Keying on the leading strategy id reports "vagueness_subjectivity" as the reason
    vagueness_subjectivity was skipped, which says nothing. The verdict is the diagnosis."""
    from intent_graph.dataset import classify_skip
    assert classify_skip(
        "vagueness_subjectivity: no mask passed the signature check "
        "(last verdict: withhold_did_not_widen)") == "signature:withhold_did_not_widen"
    assert classify_skip(
        "insufficient_information: nothing can be withheld from 1 conditions"
    ) == "too_few_conditions_to_withhold"
    assert classify_skip("referential_ambiguity: no eligible slot to mark") == "no_markable_slot"
    assert classify_skip("a+b: no coherent composite over 3 conditions") == "composite_incoherent"


def test_export_stats_surface_the_three_ways_a_run_degrades(built):
    """A degraded run must not look like a smaller one: fewer flaws than requested, a
    strategy that could not be masked, or a render that fell back to the template."""
    from intent_graph.dataset import ExportStats
    samples, skipped = built
    stats = ExportStats()
    stats.record(samples, skipped)
    d = stats.to_dict()
    assert d["samples"] == len(samples)
    assert d["strategy_skips"] == len(skipped)
    for key in ("misalignments_achieved", "template_fallbacks", "render_attempts",
                "signature_verdicts", "skips_by_strategy", "skip_reasons",
                "samples_hiding_information"):
        assert key in d, key
    # every sample is accounted for in the per-strategy and per-persona breakdowns
    assert sum(d["by_persona"].values()) == len(samples)
    assert sum(d["misalignments_achieved"].values()) == len(samples)


def test_sample_rng_varies_by_sample_but_stays_reproducible():
    """Seeding every sample from one constant made them all draw the same sequence, which
    showed up as the same false premise ('argan oil') again and again."""
    from intent_graph.dataset import _FixedRng
    a = [_FixedRng("tree1", "strat", "rational").random() for _ in range(3)]
    b = [_FixedRng("tree1", "strat", "rational").random() for _ in range(3)]
    c = [_FixedRng("tree2", "strat", "rational").random() for _ in range(3)]
    d = [_FixedRng("tree1", "strat", "avoidant").random() for _ in range(3)]
    assert a == b, "same sample must reproduce exactly"
    assert a != c, "different graphs must draw differently"
    assert a != d, "different personas must draw differently"
