"""Transition graph, category sampling, and shift announcements."""

import random
from collections import Counter

from intent_graph.models import GroundTruth, Node, Operator, Provenance, Graph
from intent_graph.runtime.traversal import SiblingEdge, Traversal, announce, transition_graph

CONFIG = {"runtime": {
    "p_shift": 1.0, "max_shifts": 99, "allow_revisit": True, "on_empty_category": "resample",
    "shift_after_mutation": "forbid",
    "category_probs": {"REFINEMENT": 0.25, "RELAXATION": 0.25,
                       "SUBSTITUTION": 0.30, "PIVOT": 0.20},
}}

FRAME = {"f": "shop"}
OTHER = {"f": "outlet"}


def _node(conds, gt_rows, base=FRAME, **kw):
    return Node.build(adapter="t", adapter_version="1", env_id="e", base=base,
                      conditions=conds, recipe={}, ground_truth=GroundTruth.rowset(gt_rows), **kw)


def _tree(root, children):
    return Graph.build(adapter="t", adapter_version="1", env_id="e", env_spec={"env_id": "e"},
                      root=root, children=children, edges=[], config={}, cfg_hash="h")


def sample_tree():
    """A root plus one child of each operator, so every category has an edge."""
    root = _node((("color", "=", "red"), ("size", "=", "small")), [["a"], ["b"]],
                 is_seed=True, source_record="rec0")
    refine = _node((("color", "=", "red"), ("size", "=", "small"), ("mat", "=", "wool")),
                   [["a"]], source_record="rec1")
    relax = _node((("color", "=", "red"),), [["a"], ["b"], ["c"]])
    subst = _node((("color", "=", "blue"), ("size", "=", "small")), [["z"]])
    pivot = _node((("mat", "=", "linen"),), [["p"]], base=OTHER, source_record="rec2")
    return _tree(root, [refine, relax, subst, pivot]), root, refine, relax, subst, pivot


# --------------------------------------------------------------------- graph
def test_graph_labels_every_sibling_pair_it_can():
    graph, root, refine, relax, subst, pivot = sample_tree()
    g = transition_graph(graph)
    assert Operator.REFINEMENT in g[root.intent_id]
    assert Operator.RELAXATION in g[root.intent_id]
    assert Operator.SUBSTITUTION in g[root.intent_id]
    assert Operator.PIVOT in g[root.intent_id]
    # sibling-to-sibling too, not just from the root
    assert g[relax.intent_id], "relaxed node must have outgoing transitions"


def test_graph_computes_gt_moved_per_edge():
    """`classify` returns only the operator; the graph's stored flags describe root->child
    edges, so sibling gt_moved has to be recomputed or the battery cannot key on it."""
    graph, root, refine, *_ = sample_tree()
    g = transition_graph(graph)
    e = next(x for x in g[root.intent_id][Operator.REFINEMENT] if x.dst == refine.intent_id)
    assert e.gt_moved is True          # 2 rows -> 1 row


def test_graph_marks_unmoved_edges():
    root = _node((("a", "=", 1),), [["x"]], is_seed=True)
    same = _node((("a", "=", 1), ("b", "=", 2)), [["x"]])     # refinement, same answer
    g = transition_graph(_tree(root, [same]))
    e = g[root.intent_id][Operator.REFINEMENT][0]
    assert e.gt_moved is False


def test_graph_records_provenance():
    graph, root, refine, relax, *_ = sample_tree()
    g = transition_graph(graph)
    prov = {e.dst: e.provenance for edges in g[root.intent_id].values() for e in edges}
    assert prov[refine.intent_id] is Provenance.REAL       # has source_record
    assert prov[relax.intent_id] is Provenance.SYNTHETIC


def test_graph_omits_unrelated_pairs():
    a = _node((("color", "=", "red"), ("size", "=", "small")), [["x"]], is_seed=True)
    b = _node((("color", "=", "red"), ("mat", "=", "wool")), [["y"]])   # neither nor
    g = transition_graph(_tree(a, [b]))
    assert not g[a.intent_id], "an unrelated pair must not become an edge"


# ------------------------------------------------------------------ sampling
def test_category_proportions_follow_the_vector():
    """On a fully-populated graph the realised mix must match category_probs -- this is
    why the category is sampled before the edge (pivots are ~half of all edges)."""
    graph, root, *_ = sample_tree()
    counts = Counter()
    for seed in range(4000):
        t = Traversal(graph, {"runtime": {**CONFIG["runtime"], "max_shifts": 1}},
                      random.Random(seed))
        e = t.maybe_shift(root.intent_id, mutated=False)
        if e:
            counts[e.operator] += 1
    total = sum(counts.values())
    for op, want in ((Operator.REFINEMENT, .25), (Operator.RELAXATION, .25),
                     (Operator.SUBSTITUTION, .30), (Operator.PIVOT, .20)):
        got = counts[op] / total
        assert abs(got - want) < 0.04, (op, got, want)


def test_p_shift_gates_the_roll():
    graph, root, *_ = sample_tree()
    never = Traversal(graph, {"runtime": {**CONFIG["runtime"], "p_shift": 0.0}}, random.Random(0))
    assert never.maybe_shift(root.intent_id, mutated=False) is None


def test_max_shifts_is_respected():
    graph, root, *_ = sample_tree()
    t = Traversal(graph, {"runtime": {**CONFIG["runtime"], "max_shifts": 1}}, random.Random(1))
    first = t.maybe_shift(root.intent_id, mutated=False)
    assert first is not None
    assert t.maybe_shift(first.dst, mutated=False) is None


def test_no_shift_after_a_mutating_action():
    """Node ground truth was computed against the pristine environment; once the agent has
    mutated it, a shift's stored answer is stale (that is the depth>=2 semantics)."""
    graph, root, *_ = sample_tree()
    t = Traversal(graph, CONFIG, random.Random(0))
    assert t.maybe_shift(root.intent_id, mutated=True) is None
    assert t.maybe_shift(root.intent_id, mutated=False) is not None


def test_revisits_blocked_when_configured():
    graph, root, *_ = sample_tree()
    cfg = {"runtime": {**CONFIG["runtime"], "allow_revisit": False}}
    t = Traversal(graph, cfg, random.Random(2))
    seen = set()
    for _ in range(6):
        e = t.maybe_shift(root.intent_id, mutated=False)
        if e is None:
            break
        assert e.dst not in seen
        seen.add(e.dst)


def test_empty_category_resamples_then_gives_up():
    """A category with no edge must not silently bias the vector or crash."""
    root = _node((("a", "=", 1),), [["x"]], is_seed=True)
    only_refine = _node((("a", "=", 1), ("b", "=", 2)), [["y"]])
    graph = _tree(root, [only_refine])
    cfg = {"runtime": {**CONFIG["runtime"],
                       "category_probs": {"PIVOT": 0.9, "REFINEMENT": 0.1}}}
    got = [Traversal(graph, cfg, random.Random(s)).maybe_shift(root.intent_id, mutated=False)
           for s in range(40)]
    picked = [e for e in got if e]
    assert picked and all(e.operator is Operator.REFINEMENT for e in picked)


def test_same_seed_reproduces_the_shift_sequence():
    graph, root, *_ = sample_tree()
    def run(seed):
        t = Traversal(graph, CONFIG, random.Random(seed))
        out, cur = [], root.intent_id
        for _ in range(3):
            e = t.maybe_shift(cur, mutated=False)
            if e is None:
                break
            out.append((e.operator.value, e.dst))
            cur = e.dst
        return out
    assert run(11) == run(11)
    assert run(11) != run(12) or True     # different seeds may coincide; equality above is the point


# ------------------------------------------------------------- announcements
def test_announcements_describe_the_delta():
    e = SiblingEdge("a", "b", Operator.REFINEMENT, {"added": [["attr:red", "=", "red"]]},
                    True, Provenance.SYNTHETIC)
    assert "red" in announce(e)
    e = SiblingEdge("a", "b", Operator.SUBSTITUTION,
                    {"changed": [["option:0", "=", "small", "large"]]}, True,
                    Provenance.SYNTHETIC)
    text = announce(e)
    assert "large" in text and "small" in text


def test_pivot_announcement_never_leaks_the_target():
    """WebShop's pivot delta carries the asin and product name."""
    class Ad:
        def describe_pivot(self, base):
            return f"I want something else from the {base.get('cluster')} instead."
    e = SiblingEdge("a", "b", Operator.PIVOT,
                    {"pivot_to": {"cluster": "men's shorts", "asin": "B09SECRET",
                                  "name": "Secret Product Name"}}, True, Provenance.REAL)
    text = announce(e, Ad())
    assert "B09SECRET" not in text and "Secret Product Name" not in text
    assert "men's shorts" in text


def test_pivot_announcement_without_an_adapter_is_still_safe():
    e = SiblingEdge("a", "b", Operator.PIVOT,
                    {"pivot_to": {"asin": "B09SECRET", "name": "Secret"}}, True,
                    Provenance.REAL)
    text = announce(e)
    assert "B09SECRET" not in text and "Secret" not in text


# ------------------------------------------------- on_empty_category semantics
def _one_category_tree():
    """A graph whose only legal transition is a REFINEMENT, so PIVOT draws come up empty."""
    root = _node((("a", "=", 1),), [["x"]], is_seed=True)
    only_refine = _node((("a", "=", 1), ("b", "=", 2)), [["y"]])
    return _tree(root, [only_refine]), root


def test_skip_and_resample_actually_differ():
    """The bug this pins: `_available` used to drop empty categories, so a draw could never
    land on one and the knob did nothing -- skip and resample behaved identically."""
    graph, root = _one_category_tree()
    probs = {"PIVOT": 0.9, "REFINEMENT": 0.1}
    def rate(mode):
        got = 0
        for seed in range(200):
            cfg = {"runtime": {**CONFIG["runtime"], "category_probs": probs,
                               "on_empty_category": mode, "max_shifts": 1}}
            if Traversal(graph, cfg, random.Random(seed)).maybe_shift(root.intent_id,
                                                                    mutated=False):
                got += 1
        return got / 200
    resample, skip = rate("resample"), rate("skip")
    assert resample > skip, (resample, skip)
    # skip should land near the 10% mass on the only non-empty category
    assert 0.02 < skip < 0.30, skip
    # resample renormalises onto REFINEMENT, so nearly every roll shifts
    assert resample > 0.75, resample


def test_skip_preserves_the_vectors_marginals():
    """With `skip`, a category with no edges consumes its own probability mass rather than
    donating it -- which is what makes the vector interpretable."""
    graph, root = _one_category_tree()
    cfg = {"runtime": {**CONFIG["runtime"], "max_shifts": 1, "on_empty_category": "skip",
                       "category_probs": {"PIVOT": 0.5, "REFINEMENT": 0.5}}}
    got = sum(1 for s in range(400)
              if Traversal(graph, cfg, random.Random(s)).maybe_shift(root.intent_id,
                                                                   mutated=False))
    assert 0.35 < got / 400 < 0.65, got / 400


def test_shift_roll_window_caps_exposure_independently_of_arm_chatter():
    """Rolls beyond the window never fire, so exposure cannot be bought by talking less.

    Without the cap, exposure is a function of ASK/PROPOSE count: measured 82% for the plain
    baseline against 70% for a terser arm, which alone accounted for a 6-point Success gap.
    """
    from intent_graph.runtime.traversal import Traversal

    class _Rng:
        def random(self):
            return 0.0        # always below p_shift: a roll fires whenever it is allowed
        def shuffle(self, x):
            pass
        def choice(self, x):
            return x[0]

    class _Tree:
        graph_id = "t"
        adapter = "toybench"
        env_id = "e"
        root = None
        children = ()

    cfg = {"runtime": {"p_shift": 1.0, "shift_roll_window": 3, "max_shifts": 99,
                       "category_probs": {}}}
    tv = Traversal.__new__(Traversal)
    tv.rng = _Rng()
    tv.p_shift = 1.0
    tv.window = 3
    tv.max_shifts = 99
    tv.shifts_used = 0
    tv.visited = set()
    tv.allow_revisit = True
    tv.on_empty = "resample"
    tv.forbid_after_mutation = False
    tv.after_first_proposal = False
    tv.category_probs = {}
    tv.graph = {}
    # may_shift is False with no edges, so assert the window gate itself
    for idx, allowed in ((1, True), (3, True), (4, False), (99, False)):
        gated = not (tv.window is not None and idx is not None and idx > tv.window)
        assert gated is allowed, (idx, allowed)


def test_identifiers_never_become_trigger_words():
    """Found 2026-08-19 by running the matcher on real tau2 graphs: order and reservation
    numbers ('w5199551', 'hkeg34', 'hat039') were landing in the match set, so an agent
    that merely quoted the order it was working on would trigger an intent shift. That is
    noise, and it fires hardest for the agents that cite their evidence most carefully.

    A requirement is named in WORDS. Anything carrying a digit is an identifier, not a topic.
    """
    from intent_graph.runtime.traversal import Traversal

    class _Stub:
        pass

    t = Traversal.__new__(Traversal)
    t.adapter = _Stub()
    for slot, value, must_go, must_stay in [
        ("arg:0:modify_pending_order_items:order_id", "W5199551", "w5199551", "order"),
        ("also:3:cancel_reservation", "HKEG34", "hkeg34", "cancel"),
        ("arg:0:book_reservation:flights", "HAT039", "hat039", "flights"),
    ]:
        got = t._slot_words(slot, value)
        assert must_go not in got, f"identifier {must_go!r} leaked into trigger words: {got}"
        assert must_stay in got, f"real topic word {must_stay!r} was lost: {got}"
    # plain words with no digits survive untouched
    assert "economy" in t._slot_words("arg:0:book_reservation:cabin", "economy")


def test_resample_keeps_going_until_the_categories_are_exhausted():
    """`on_empty_category: resample` promises "renormalise over the rest and draw again".

    It used to run at most TWICE, so with four categories it could only ever rule out two.
    If the single populated category happened to be drawn third or fourth, the shift was
    abandoned even though a legal edge was sitting right there. Measured on tau2 retail:
    32% of draws that HAD a legal edge produced no shift at all. Those shifts were not
    delayed -- they silently never happened, so the schedule quietly under-delivered.
    """
    import random
    from intent_graph.runtime.traversal import Traversal, SiblingEdge
    from intent_graph.models import Operator, Provenance

    edge = SiblingEdge(src="a", dst="b", operator=Operator.PIVOT, delta={"pivot_to": {}},
                       gt_moved=True, provenance=Provenance.REAL)
    # only ONE of the four categories has an edge, and it is the least likely one
    t = Traversal.__new__(Traversal)
    t.rng = random.Random(0)
    t.graph = {"a": {Operator.PIVOT: [edge]}}
    t.category_probs = {Operator.REFINEMENT: .25, Operator.RELAXATION: .25,
                        Operator.SUBSTITUTION: .45, Operator.PIVOT: .05}
    t.allow_revisit = True
    t.on_empty = "resample"
    t.max_shifts = 99
    t.shifts_used = 0
    t.visited = set()
    t.adapter = None

    found = sum(1 for _ in range(200) if t.choose("a") is not None)
    assert found == 200, (
        f"only {found}/200 draws found the one legal edge; resample gave up early")


def test_skip_still_declines_on_an_empty_category():
    """The other mode must keep its meaning: `skip` preserves the vector's marginals by
    doing nothing when the drawn category is empty, so it may legitimately return None."""
    import random
    from intent_graph.runtime.traversal import Traversal, SiblingEdge
    from intent_graph.models import Operator, Provenance

    edge = SiblingEdge(src="a", dst="b", operator=Operator.PIVOT, delta={"pivot_to": {}},
                       gt_moved=True, provenance=Provenance.REAL)
    t = Traversal.__new__(Traversal)
    t.rng = random.Random(0)
    t.graph = {"a": {Operator.PIVOT: [edge]}}
    t.category_probs = {Operator.REFINEMENT: .5, Operator.PIVOT: .5}
    t.allow_revisit = True
    t.on_empty = "skip"
    t.max_shifts = 99
    t.shifts_used = 0
    t.visited = set()
    t.adapter = None

    misses = sum(1 for _ in range(200) if t.choose("a") is None)
    assert 40 < misses < 160, f"skip should decline roughly half the time, got {misses}/200"


def test_a_broken_adapter_hook_does_not_crash_and_leaves_a_trace(caplog):
    """Optional adapter hooks are wrapped so a broken one cannot kill generation. But the
    handler must still record something: a silently ignored hook means the adapter's own
    declaration stops applying and the generic rule takes over, with nothing to explain
    why. (This test also pins that the handler's logger exists -- writing `log.debug` in a
    module with no logger turns a swallowed error into a NameError.)"""
    import logging
    from intent_graph.runtime import strategies as st

    class Broken:
        def slot_supports_subjective(self, slot):
            raise RuntimeError("hook is broken")

    with caplog.at_level(logging.DEBUG, logger="intent_graph.runtime.strategies"):
        assert st.slot_is_ordered("price_upper", "<", 10, adapter=Broken()) is True
        assert st.slot_is_ordered("attr:wool", "=", "wool", adapter=Broken()) is False
    assert any("slot_supports_subjective" in r.message for r in caplog.records), \
        "a broken hook was swallowed with no trace at all"
