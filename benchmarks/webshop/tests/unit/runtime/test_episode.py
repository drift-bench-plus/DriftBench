"""End-to-end episode behaviour on toybench: no data, no docker, no network."""


import pytest

from intent_graph.adapters.toybench import ToyAdapter, ToyExecutor
from intent_graph.engine import RunStats, generate
from intent_graph.runtime import metrics as M
from intent_graph.runtime import scripted as S
from intent_graph.runtime.agent_api import ActionKind, parse_action
from intent_graph.runtime.episode import Episode, Outcome
from intent_graph.runtime.user import type_label

GEN = {"seed": 42, "depth": 1, "branching": 8, "min_gt_moved_edges": 1,
       # toybench has too few candidates to fill 2/2/2/2; these suites
       # exercise the runtime, not graph yield. See test_gate for the quota.
       "require_full_quota": False}

RUNTIME = {
    "runtime": {
        "graphs_dir": "artifacts", "episode_seed": 7, "max_turns": 12,
        "act_consumes_turn": False,
        "p_shift": 0.0, "max_shifts": 1, "allow_revisit": False,
        "on_empty_category": "resample", "shift_after_mutation": "forbid",
        "axis_b_on_write_trees": False, "perturb_shift_announcements": False,
        "category_probs": {"REFINEMENT": .25, "RELAXATION": .25,
                           "SUBSTITUTION": .30, "PIVOT": .20},
        "strategy_probs": {}, "withhold_k_max": 2, "min_retained_conditions": 1,
        "falsify_max_attempts": 5, "render_max_attempts": 3,
        "persona": "rational", "max_consecutive_none": 3, "min_reveal_after_n_asks": 2,
        "patience_init": 14, "cost_ask": 1, "cost_reject": 2, "cost_malformed": 1,
        "cost_act": 0,
    }
}


@pytest.fixture(scope="module")
def graphs():
    out = list(generate(ToyAdapter(), ToyExecutor(), GEN, stats=RunStats()))
    assert out, "toybench must produce graphs"
    return out


def cfg(**over):
    c = {"runtime": dict(RUNTIME["runtime"])}
    c["runtime"].update(over)
    return c


def hidden_slot_ids(graph, spec_hidden):
    """Map hidden slot names to the opaque ids the user will offer."""
    ordered = [s for s, _, _ in sorted(graph.root.conditions)]
    return [f"slot_{ordered.index(s)}" for s in spec_hidden if s in ordered]


def play(graph, agent_factory, **over):
    """Run one episode, wiring the scripted user stub.

    The episode is built AFTER the probe: a forwarding wrapper would proxy only `act` and
    swallow `observe_node`, which silently turns the oracle into the intent-ignorer.
    """
    from intent_graph.runtime.episode import Episode
    adapter, executor = ToyAdapter(), ToyExecutor()

    probe_traj = Episode(graph=graph, adapter=adapter, executor=executor, config=cfg(**over),
                         llm=S.stub_llm(), agent=S.NoOp(graph=graph)).run()
    spec = (probe_traj.header.get("perturbation") or {}).get("hidden_slots") or []
    ids = hidden_slot_ids(graph, spec)

    ep = Episode(graph=graph, adapter=adapter, executor=executor, config=cfg(**over),
                 llm=S.stub_llm(), agent=agent_factory(graph, ids))
    return ep.run(), ids, probe_traj


# ------------------------------------------------------------------ the loop
def test_oracle_always_succeeds(graphs):
    """The single most important test: if a perfectly-behaved agent cannot win, the
    environment is unwinnable and every other number is meaningless."""
    for graph in graphs:
        traj, ids, _ = play(graph, lambda t, i: S.Oracle(graph=t, hidden_slot_ids=i))
        assert traj.outcome == Outcome.SUCCESS.value, (graph.graph_id, traj.error, traj.turns)


def test_oracle_succeeds_under_every_persona(graphs):
    """A persona that makes the task unwinnable is a misconfiguration, not a difficulty."""
    from intent_graph.runtime.persona import PERSONA_IDS
    graph = graphs[0]
    for pid in PERSONA_IDS:
        traj, _, _ = play(graph, lambda t, i: S.Oracle(graph=t, hidden_slot_ids=i),
                          persona=pid)
        assert traj.outcome == Outcome.SUCCESS.value, (pid, traj.outcome, traj.error)


def test_oracle_recovers_every_hidden_slot(graphs):
    graph = graphs[0]
    traj, ids, _ = play(graph, lambda t, i: S.Oracle(graph=t, hidden_slot_ids=i))
    m = M.score(traj)
    if m.hidden_slots:
        assert m.intent_recovery == 1.0, (m.hidden_slots, m.recovered_slots)
    assert m.premature_action is False


def test_no_op_never_succeeds(graphs):
    traj, _, _ = play(graphs[0], lambda t, i: S.NoOp(graph=t))
    assert traj.outcome in (Outcome.EXHAUSTED.value, Outcome.TURNS_EXCEEDED.value)
    assert M.score(traj).success is False


def test_second_valid_member_is_accepted(graphs):
    """Acceptance is set membership; a verifier that admits only one answer is wrong."""
    multi = [t for t in graphs if t.root.ground_truth.cardinality() > 1]
    if not multi:
        pytest.skip("no toybench graph has a multi-member answer set")
    traj, _, _ = play(multi[0], lambda t, i: S.SecondValid(graph=t))
    assert traj.outcome == Outcome.SUCCESS.value


def test_malformed_turns_cost_patience_and_are_counted(graphs):
    traj, _, _ = play(graphs[0], lambda t, i: S.NoOp(graph=t))
    m = M.score(traj)
    assert m.n_malformed >= 1
    assert m.patience_spent >= 1


def test_episode_is_replayable(graphs):
    """Same graph + same seed + same stub => identical turn-by-turn trajectory."""
    a, _, _ = play(graphs[0], lambda t, i: S.Oracle(graph=t, hidden_slot_ids=i))
    b, _, _ = play(graphs[0], lambda t, i: S.Oracle(graph=t, hidden_slot_ids=i))
    assert [t["action"]["kind"] for t in a.turns] == [t["action"]["kind"] for t in b.turns]
    assert a.outcome == b.outcome and a.final_node == b.final_node


def test_header_records_everything_needed_to_replay(graphs):
    traj, _, _ = play(graphs[0], lambda t, i: S.Oracle(graph=t, hidden_slot_ids=i))
    h = traj.header
    for key in ("graph_id", "adapter", "env_id", "config_hash", "episode_seed", "persona",
                "perturbation", "strategy_id", "query"):
        assert key in h, f"missing header field {key}"


def test_the_agent_never_sees_a_hidden_value(graphs):
    """The whole point of the perturbation: withheld values must not reach the transcript
    until the user reveals them."""
    graph = graphs[0]
    traj, _, probe = play(graph, lambda t, i: S.NoOp(graph=t))   # NoOp never asks
    spec = probe.header.get("perturbation") or {}
    values = {s: str(v) for s, _, v in graph.root.conditions}
    query = probe.header.get("query", "")
    for slot in spec.get("withheld", []):
        val = values.get(slot, "")
        if len(val) >= 3:
            assert val.lower() not in query.lower(), f"{slot} leaked into the opening query"


# ----------------------------------------------------------------- axis B
def test_stale_answer_is_rejected_exactly_when_it_stops_being_valid(graphs):
    """`gt_moved` is about the answer SET; it does not imply a specific stale answer is now
    wrong.  A RELAXATION widens the set, so the old answer stays valid — and rejecting it
    would punish a correct agent.  The real invariant is membership in the NEW node."""
    from intent_graph.runtime.accept import accepts as accepts_fn
    from intent_graph.runtime.accept import default_parse_proposal
    checked = 0
    for graph in graphs:
        traj, _, _ = play(graph, lambda t, i: S.IntentIgnorer(graph=t),
                          p_shift=1.0, max_shifts=1)
        m = M.score(traj)
        if not m.n_shifts:
            continue
        final = {n.intent_id: n for n in (graph.root, *graph.children)}[traj.final_node]
        stale = default_parse_proposal(S._gt_member(graph.root))
        still_valid, _ = accepts_fn(ToyAdapter(), stale, final, None)
        checked += 1
        if still_valid:
            assert traj.outcome == Outcome.SUCCESS.value, (
                "the stale answer still satisfies the new intent, so it must be accepted")
        else:
            assert traj.outcome != Outcome.SUCCESS.value, (
                "the stale answer no longer satisfies the new intent, so it must be rejected")
    assert checked, "no episode produced a shift; cannot assert the property"


def test_shift_that_does_not_move_the_answer_still_accepts_it(graphs):
    """The counterweight: a non-moving change must not punish a correct answer."""
    found = False
    for graph in graphs:
        for seed in range(6):
            traj, _, _ = play(graph, lambda t, i: S.IntentIgnorer(graph=t),
                              p_shift=1.0, max_shifts=1, episode_seed=seed)
            m = M.score(traj)
            if m.n_shifts and m.shift_moved_answer is False:
                found = True
                assert traj.outcome == Outcome.SUCCESS.value, (
                    "a non-moving shift wrongly invalidated a correct answer")
    if not found:
        pytest.skip("no unmoved shift occurred in this sample")


def test_shifts_are_logged_with_their_category(graphs):
    for graph in graphs:
        traj, _, _ = play(graph, lambda t, i: S.Oracle(graph=t, hidden_slot_ids=i), p_shift=1.0)
        for rec in traj.turns:
            if rec.get("shift"):
                assert rec["shift"]["operator"] in ("REFINEMENT", "RELAXATION",
                                                    "SUBSTITUTION", "PIVOT")
                assert "gt_moved" in rec["shift"]
                return
    pytest.skip("no shift occurred")


# ------------------------------------------------------------ action parsing
@pytest.mark.parametrize("raw,kind", [
    ("Action: Clarify\nStrategy: Ask_Parameter\nContent: which size?\n", ActionKind.ASK),
    ("Action: Clarify\nStrategy: Ask_Parameter\nContent: which size?", ActionKind.ASK),
    ("Act: clarify\nStrategy: Ask_Parameter\nContent: which?\n", ActionKind.ASK),
    ("Action: Clarify\nContent: no strategy here\n", ActionKind.ASK),
    ("Action: Answer\nFinal Answer: 42\n", ActionKind.PROPOSE),
    ("Action: Answer\n", ActionKind.MALFORMED),
    ("Action: Operation\n```\nSELECT 1\n```\n", ActionKind.ACT),
    ("```\nls -la\n```", ActionKind.ACT),
    ("just chatting", ActionKind.MALFORMED),
])
def test_action_parsing(raw, kind):
    assert parse_action(raw).kind is kind


def test_clarify_on_the_last_line_is_not_missed():
    """Drift-Bench's regex needs a trailing newline; we accept end-of-string too."""
    a = parse_action("Action: Clarify\nStrategy: Ask_Parameter\nContent: which one?")
    assert a.kind is ActionKind.ASK and a.question == "which one?"


def test_predicted_question_is_captured_but_not_scored():
    a = parse_action("Action: Answer\nPredicted user question: the red one\nFinal Answer: x\n")
    assert a.predicted_question == "the red one"
    m = M.score({"header": {"perturbation": {}}, "turns": [], "outcome": "SUCCESS"})
    assert not hasattr(m, "predicted_question_score")


def test_candidates_are_parsed():
    a = parse_action('Action: Clarify\nStrategy: Disambiguate\nContent: which?\n'
                     'Candidates: ["a", "b"]\n')
    assert a.candidates == ("a", "b")


def test_any_strategy_text_is_tolerated_metadata():
    """The strategy taxonomy is gone (ruling 2026-08-13): whatever an arm writes there is
    recorded verbatim, never validated, never a problem."""
    a = parse_action("Action: Clarify\nStrategy: Telepathy\nContent: hm?\n")
    assert a.kind is ActionKind.ASK and a.strategy == "Telepathy" and not a.problems


# ----------------------------------------------------------------- metrics
def test_aggregate_splits_by_the_dimensions_that_matter(graphs):
    ms = []
    for graph in graphs[:4]:
        traj, _, _ = play(graph, lambda t, i: S.Oracle(graph=t, hidden_slot_ids=i))
        ms.append(M.score(traj))
    agg = M.aggregate(ms)
    for key in ("all", "by_adapter", "by_persona", "by_strategy", "shifted", "unshifted"):
        assert key in agg
    assert agg["all"]["episodes"] == len(ms)


def test_type_label_is_used_rather_than_the_slot_name():
    assert type_label("attr:easy use") == "a required product feature"


def test_unimplemented_shift_perturbation_flag_refuses_rather_than_lying(graphs):
    """The flag was declared in config and read nowhere, so setting it silently did nothing.
    Templating the announcement while config claims it is perturbed is the worse failure."""
    graph = graphs[0]
    with pytest.raises(NotImplementedError, match="perturb_shift_announcements"):
        Episode(graph=graph, adapter=ToyAdapter(), executor=ToyExecutor(),
                config=cfg(perturb_shift_announcements=True), llm=S.stub_llm(),
                agent=S.NoOp(graph=graph))


# ------------------------------------------------------- patience visibility channel
def test_loop_publishes_patience_to_the_agent(graphs):
    """The loop must publish the patience meter every turn, whatever the arm does with it.

    Without this the agent sees only actions-remaining and asking is free from its side --
    the measured cause of ask-budget saturation (0.0-2.7% silent episodes across five arms).
    """
    seen: list[int] = []

    class Recorder(S.NoOp):
        def act(self, transcript):
            seen.append(getattr(self, "patience_left", None))
            return super().act(transcript)

    traj, _, _ = play(graphs[0], lambda t, i: Recorder(graph=t))
    assert seen, "agent was never called"
    # floats since the deduction-side economy (2026-08-13): costs scale by persona, budgets don't
    assert all(isinstance(p, (int, float)) for p in seen), seen
    assert seen[0] == 14, seen                      # nominal budget, identical for every persona
    assert all(b >= a for b, a in zip(seen, seen[1:])), seen   # monotone non-increasing


def test_patience_line_reaches_the_prompt_only_when_enabled(graphs):
    """End-to-end: the flag, the loop's published value, and the rendered prompt."""
    from intent_graph.runtime.agents import DirectAgent

    prompts: list[str] = []

    class Spy(DirectAgent):
        def build_prompt(self, transcript):
            p = super().build_prompt(transcript)
            prompts.append(p)
            return p

        def act(self, transcript):          # never call the network in a unit test
            self.build_prompt(transcript)
            return "Action: Operation\n```\nsearch[thing]\n```"

    for enabled in (False, True):
        prompts.clear()
        play(graphs[0], lambda t, i: Spy(llm=S.stub_llm(), config=cfg(show_patience=enabled)),
             show_patience=enabled)
        assert prompts, "prompt was never built"
        hit = any("shopper's patience" in p for p in prompts)
        assert hit is enabled, (enabled, prompts[0][-400:])


# ---------------------------------------------------- affordability (ruling 2026-08-13)
def test_patience_never_goes_negative(graphs):
    """An action costing more than the remaining budget cannot be bought: the episode ends
    through forced-final (unaffordable ASK) or the proposal becomes the adjudicated last
    shot (unaffordable PROPOSE). patience_after must never be negative on any turn."""
    for cost_ask, cost_reject in ((2, 4), (3, 5)):
        traj, _, _ = play(graphs[0], lambda t, i: S.Oracle(graph=t, hidden_slot_ids=i),
                          patience_init=5, cost_ask=cost_ask, cost_reject=cost_reject)
        for rec in traj.turns:
            if "patience_after" in rec:
                assert rec["patience_after"] >= 0, (cost_ask, cost_reject, rec)


def test_unaffordable_proposal_is_the_final_shot(graphs):
    """At patience below cost_reject, a proposal is adjudicated once and ends the episode
    either way -- a correct one still wins, a wrong one exhausts without a negative charge."""
    from intent_graph.runtime.episode import Outcome
    traj, _, _ = play(graphs[0], lambda t, i: S.Oracle(graph=t, hidden_slot_ids=i),
                      patience_init=1, cost_ask=2, cost_reject=4)
    assert traj.outcome in (Outcome.SUCCESS.value, Outcome.EXHAUSTED.value)
    for rec in traj.turns:
        if "patience_after" in rec:
            assert rec["patience_after"] >= 0


class _StubbornProposer(S.ScriptedAgent):
    """Proposes the same wrong item every turn -- the present/reject/revise loop with no
    revision, which pins the economy arithmetic exactly."""

    name = "stubborn_proposer"

    def plan(self, transcript, turn):
        return S.answer_block("definitely-not-what-anyone-wants")


def test_multiplier_scales_the_budget_once(graphs):
    """Settled economy (ruling 2026-08-13): the multiplier applies ONCE to the total
    budget, EXACTLY -- no rounding -- and prices are flat. Spontaneous at init 8 holds
    8*0.7 = 5.6: one rejection is payable (5.6-4 = 1.6), and the second proposal --
    unaffordable at 1.6 < 4 -- is adjudicated as the forced final submission."""
    traj, _, _ = play(graphs[0], lambda t, i: _StubbornProposer(graph=t),
                      persona="spontaneous", patience_init=8, cost_ask=2, cost_reject=4)
    props = [r for r in traj.turns if "acceptance" in r]
    assert len(props) == 2, [r.get("acceptance") for r in traj.turns]
    first, second = props
    assert not first["acceptance"].get("final_unaffordable_reject")
    assert first["patience_after"] == pytest.approx(1.6), first
    assert second["acceptance"].get("final_unaffordable_reject") is True
    assert traj.outcome == Outcome.EXHAUSTED.value
    for rec in traj.turns:
        if "patience_after" in rec:
            assert rec["patience_after"] >= 0


def test_budget_is_exact_not_rounded(graphs):
    """Dependent at init 8 holds exactly 9.6 (ruling: "you should not round, the budget
    should be a float"): the first ask lands on 9.6-2 = 7.6, never on round(9.6)-2 = 8."""
    traj, _, _ = play(graphs[0], lambda t, i: S.Oracle(graph=t, hidden_slot_ids=i),
                      persona="dependent", patience_init=8, cost_ask=2, cost_reject=4)
    asks = [r for r in traj.turns
            if (r.get("action") or {}).get("kind") == "ASK" and "patience_after" in r]
    assert asks, "oracle asked nothing -- test setup broken"
    assert asks[0]["patience_after"] == pytest.approx(7.6)


def test_b0_menu_has_no_user_channel():
    """B0 is a regular agent: Clarify is absent from its action menu entirely, while an
    arm built to ask (A0) keeps the full grammar."""
    from intent_graph.runtime.agents import DirectAgent, FreeFormAgent
    b0 = DirectAgent(llm=None, config={"runtime": {}}).system()
    a0 = FreeFormAgent(llm=None, config={"runtime": {}}).system()
    assert "Clarify" not in b0
    assert "Action: Operation" in b0 and "Action: Answer" in b0
    assert "Action: Clarify" in a0


def test_b0_fallback_never_asks():
    """The parse-failure fallback must stay inside the arm's action space: for B0 (no user
    channel) it is a catalogue search, never the harness-injected safe question."""
    from intent_graph.runtime.agents import DirectAgent, FreeFormAgent

    class _Garbage:
        def complete(self, *a, **k): return "utter nonsense with no action"
        def usage(self): return {}

    class _T:
        turns = [{"role": "user", "content": "buy me a red mug under 20 dollars"}]

    b0 = DirectAgent(llm=_Garbage(), config={"runtime": {}})
    out = b0.act(_T())
    assert "Clarify" not in out and "search[" in out
    a0 = FreeFormAgent(llm=_Garbage(), config={"runtime": {}})
    assert "Clarify" in a0.act(_T())


def test_a1v3_forces_ask_before_proposal():
    """A1v3's mechanism: a buy without a fresh ask since the last proposal is re-prompted,
    and if the model insists, converted into the ledger's most valuable question. With a
    fresh ask on the transcript, the buy passes through untouched."""
    from intent_graph.runtime.agents import ForcedAskLedgerAgent

    class _AlwaysBuys:
        def complete(self, *a, **k):
            return ("Action: Answer\nPredicted user question: a mug\n"
                    "Final Answer: buy B00X {}\nLEDGER: {\"slots\": "
                    "{\"color\": {\"value\": \"UNK\", \"confidence\": 0}}, \"confirmed\": false}")
        def usage(self): return {}

    class _T:
        turns = [{"role": "user", "content": "buy me a mug"}]

    arm = ForcedAskLedgerAgent(llm=_AlwaysBuys(), config={"runtime": {}})
    out = arm.act(_T())
    assert "Action: Clarify" in out and "color" in out, out

    class _Asked(_T):
        turns = [{"role": "user", "content": "buy me a mug"},
                 {"role": "agent", "content": "Action: Clarify\nStrategy: Ask_Parameter\n"
                                              "Content: what color?"},
                 {"role": "user", "content": "red"}]

    out2 = ForcedAskLedgerAgent(llm=_AlwaysBuys(), config={"runtime": {}}).act(_Asked())
    assert "Final Answer" in out2, out2


def test_scheduled_shift_semantics(graphs):
    """Scheduled shifts: decided at episode start, and fired at a defined point in the
    exchange. WHERE that point is depends on what the agent did, and the split is the whole
    semantics (ruling 2026-08-19, revised):

      * QUESTION -- the user answers first, then the shift fires. The answer therefore
        reflects the OLD intent, and carries `after_reply` so the scorer knows to treat
        that turn's reveals as stale. You cannot answer using a requirement you have not
        thought of yet.
      * PROPOSAL -- every due shift fires FIRST, then the proposal is adjudicated against
        the goal as it now stands. So `after_reply` is absent. This holds for a
        mid-conversation proposal and for the forced final answer alike: there is no
        reaction/submission split on the proposal side. An answer that was right before the
        goal moved is refused for being stale, which is what makes staleness measurable.

    The schedule remains a property of the episode, never of the agent."""
    # an asking agent absorbs the shift AFTER its answer (after_reply marks it)
    traj, _, _ = play(graphs[0], lambda t, i: S.Oracle(graph=t, hidden_slot_ids=i),
                      shift_scheduled=True, p_shift=1.0, max_shifts=1)
    ask_shifts = [u for u in traj.turns
                  if (u.get("action") or {}).get("kind") == "ASK" and u.get("shift")]
    prop_shifts = [u for u in traj.turns
                   if (u.get("action") or {}).get("kind") == "PROPOSE" and u.get("shift")]
    assert ask_shifts or prop_shifts, "p=1.0 scheduled shift never fired"
    for u in ask_shifts:
        assert u["shift"].get("after_reply") is True, "a reply must be given under the old goal"
    for u in prop_shifts:
        # a proposal is judged against the CURRENT goal, so the shift fired BEFORE it
        assert not u["shift"].get("after_reply"), "a proposal must face the moved goal"

    # a propose-only agent never gets a reply, so every shift it meets fires before an
    # adjudication and none of them are marked after_reply
    traj2, _, _ = play(graphs[0], lambda t, i: _StubbornProposer(graph=t),
                       shift_scheduled=True, p_shift=1.0, max_shifts=1)
    shifts2 = [u for u in traj2.turns if u.get("shift")]
    assert shifts2 and not any(u["shift"].get("after_reply") for u in shifts2)

    # the schedule is deterministic per episode seed
    traj3, _, _ = play(graphs[0], lambda t, i: S.Oracle(graph=t, hidden_slot_ids=i),
                       shift_scheduled=True, p_shift=1.0, max_shifts=1)
    assert [u["turn"] for u in traj.turns if u.get("shift")] == \
           [u["turn"] for u in traj3.turns if u.get("shift")]


def test_grip_invalidates_same_turn_reveal_when_shift_fired_after_reply():
    """Under scheduled shifts the answer on the shift turn reflects the OLD intent, so
    the scorer must treat that turn's reveals as stale for moved slots."""
    from intent_graph.runtime import grip

    def traj(after_reply):
        return {"header": {"persona": "rational", "strategy_id": "s", "query": "q",
                           "perturbation": {"hidden_slots": ["s1"],
                                            "governing_kind": "withhold"}},
                "outcome": "EXHAUSTED",
                "turns": [
                    {"turn": 1, "action": {"kind": "ASK", "question": "?"},
                     "reveals": [{"slot": "s1", "value": "old", "declined": False,
                                  "volunteered": False}],
                     "shift": {"gt_moved": True, "after_reply": after_reply,
                               "delta": {"changed": [["s1", "=", "new"]],
                                         "added": [], "removed": []}}},
                    {"turn": 2, "action": {"kind": "PROPOSE"},
                     "acceptance": {"ok": False, "reason": "reward=0.0"}},
                ]}
    stale = grip.score_v2(traj(True))
    fresh = grip.score_v2(traj(False))
    assert stale["recovery"] == 0.0      # the answer described the OLD value
    assert fresh["recovery"] == 1.0      # legacy semantics: reply already reflected NEW


# ------------------------------------------------- persona shift schedule (2026-08-19)
def _test_persona(monkeypatch, pid, placements, sugg, mult=1.0):
    """Register a throwaway persona so a test can pin the schedule exactly."""
    from intent_graph.runtime import persona as P
    p = P.Persona(id=pid, bio="test", reveal_granularity=1, volunteer=0.0,
                  p_decline=0.0, patience_mult=mult, hint=0.0,
                  shift_placements=placements, suggestibility=sugg)
    monkeypatch.setitem(P.PERSONAS, pid, p)
    return p


class _AskerNaming(S.ScriptedAgent):
    """Asks a question whose text NAMES a real slot word, then proposes garbage forever.
    The question deliberately avoids the opaque slot-id protocol: the trigger matcher
    reads natural words, and 'material'/'color' are toybench slot names."""
    name = "asker-naming"

    def __init__(self, graph, word, asks=1, **kw):
        super().__init__(graph=graph, **kw)
        self.word, self.asks = word, asks

    def plan(self, transcript, turn):
        if turn <= self.asks:
            return ("Action: Clarify\nStrategy: Ask_Parameter\n"
                    f"Content: Should the {self.word} matter for this?\n")
        return S.answer_block("definitely-not-what-anyone-wants")


def _slot_word_for(graph):
    """A word from some legal non-pivot edge's delta off the root."""
    from intent_graph.runtime.traversal import Traversal, _delta_slots, _words
    from intent_graph.models import Operator
    trav = Traversal(graph, {"runtime": dict(RUNTIME["runtime"])}, None)
    for op, edges in trav.graph[graph.root.intent_id].items():
        if op is Operator.PIVOT:
            continue
        for e in edges:
            for slot, value in _delta_slots(e):
                for w in sorted(_words(slot.split(":")[-1])):
                    if len(w) >= 3:
                        return w
    return None


def test_persona_thresholds_fire_on_patience_crossings(graphs, monkeypatch):
    """Layer 1+2 (ruling 2026-08-19): the persona's placements x initial patience are the
    thresholds; a shift fires at the first user exchange after patience drops below one."""
    _test_persona(monkeypatch, "two-shifts", (0.66, 0.33), sugg=0.0)
    traj, _, _ = play(graphs[0], lambda t, i: _StubbornProposer(graph=t),
                      persona="two-shifts", patience_init=14, cost_reject=2,
                      shift_scheduled=True)
    shifts = [u for u in traj.turns if u.get("shift")]
    init = 14.0
    thr = [0.66 * init, 0.33 * init]
    # every fired shift sits at the first exchange whose post-cost patience is below its
    # threshold, in order
    assert len(shifts) == 2, [u.get("patience_after") for u in traj.turns]
    assert shifts[0]["patience_after"] < thr[0] and shifts[0]["patience_before"] >= thr[0] - 2
    assert shifts[1]["patience_after"] < thr[1]
    for u in shifts:
        # this agent only proposes, so every shift fires BEFORE an adjudication
        assert not u["shift"].get("after_reply")
        assert "trigger" not in u["shift"]          # sugg=0: no agent-caused shifts


def test_suggestibility_zero_never_triggers(graphs, monkeypatch):
    _test_persona(monkeypatch, "immovable", (0.5,), sugg=0.0)
    w = _slot_word_for(graphs[0])
    assert w, "toybench must expose a slot word"
    traj, _, _ = play(graphs[0], lambda t, i: _AskerNaming(graph=t, word=w, asks=3),
                      persona="immovable", patience_init=14, cost_ask=1, cost_reject=2,
                      shift_scheduled=True)
    for u in traj.turns:
        if u.get("shift"):
            assert "trigger" not in u["shift"]


def test_suggestibility_one_triggers_and_spends_the_budget(graphs, monkeypatch):
    """Layer 3 (ruling 2026-08-19): a triggering ask fires the shift NOW; the LAST scheduled
    placement is dropped; total shifts never exceed the persona's count."""
    _test_persona(monkeypatch, "eager", (1.0, 0.1), sugg=1.0)
    w = _slot_word_for(graphs[0])
    traj, _, _ = play(graphs[0], lambda t, i: _AskerNaming(graph=t, word=w, asks=1),
                      persona="eager", patience_init=14, cost_ask=1, cost_reject=2,
                      shift_scheduled=True, max_turns=14)
    shifts = [u for u in traj.turns if u.get("shift")]
    triggered = [u for u in shifts if u["shift"].get("trigger")]
    assert triggered, "sugg=1.0 + a slot-naming ask must trigger"
    trig = triggered[0]["shift"]["trigger"]
    assert trig["kind"] == "agent_ask" and trig["slots"], trig
    # budget minus one: at most the persona's count in total, ever
    assert len(shifts) <= 2
    # the trigger replaced the LAST placement (0.1): after the triggered shift and the
    # 1.0-threshold scheduled one, nothing else may fire
    assert len([u for u in shifts if not u["shift"].get("trigger")]) <= 1


def test_submission_fires_due_shift_before_adjudication(graphs, monkeypatch):
    """A SUBMISSION is scored against the intent as it stands NOW: a threshold crossed by
    a non-exchange drop (malformed) must fire before the final adjudication."""
    _test_persona(monkeypatch, "late-shift", (0.5,), sugg=0.0, mult=1.0)

    class _GarbageThenPropose(S.ScriptedAgent):
        name = "garbage-then-propose"
        def plan(self, transcript, turn):
            if turn == 1:
                return "utter nonsense the parser cannot read"
            return S.answer_block("definitely-not-what-anyone-wants")

    # init=2.4: malformed (cost 1) -> 1.4 > 1.2 (no cross); second malformed 0.4 < 1.2 due;
    # then PROPOSE with patience 0.4 < cost_reject -> submission -> flush fires the shift
    class _TwoGarbageThenPropose(S.ScriptedAgent):
        name = "garbage2-then-propose"
        def plan(self, transcript, turn):
            if turn <= 2:
                return "utter nonsense the parser cannot read"
            return S.answer_block("definitely-not-what-anyone-wants")

    traj, _, _ = play(graphs[0], lambda t, i: _TwoGarbageThenPropose(graph=t),
                      persona="late-shift", patience_init=2.4, cost_malformed=1,
                      cost_reject=2, shift_scheduled=True)
    start = traj.turns[0]["node"]
    shifts = [u for u in traj.turns if u.get("shift")]
    assert shifts, "the due shift must fire at the submission"
    last = traj.turns[-1]
    assert last.get("shift") or shifts, traj.turns
    assert traj.final_node != start, "submission must be adjudicated against the NEW intent"


# ===================================================================== the
# agent-trigger channel, proved rather than asserted (ruling 2026-08-19:
# "I need to be sure of this: the agent trigger actually works, and the
# intent actually shifts as expected").
def _shift_records(traj):
    """Every shift hop the episode logged, in order (including flushed ones)."""
    out = []
    for rec in traj.turns:
        for e in (rec.get("shifts_earlier") or ()):
            out.append((rec["turn"], e))
        if rec.get("shift"):
            out.append((rec["turn"], rec["shift"]))
    return out


def test_trigger_fires_and_the_node_actually_changes(graphs, monkeypatch):
    """The whole claim in one test: the agent names a live requirement, a shift fires,
    it is tagged as agent-caused, and the episode's CURRENT NODE is genuinely different
    afterwards. A logged edge that did not move `current` would be theatre."""
    _test_persona(monkeypatch, "trig-moves", (1.0,), sugg=1.0)
    g = graphs[0]
    w = _slot_word_for(g)
    assert w, "toybench must expose a matchable slot word"
    traj, _, _ = play(g, lambda t, i: _AskerNaming(graph=t, word=w, asks=1),
                      persona="trig-moves", shift_scheduled=True,
                      shift_schedule="persona", p_shift=1.0)
    shifts = _shift_records(traj)
    assert shifts, "no shift fired at suggestibility 1.0 on a matching question"
    turn, sh = shifts[0]
    assert sh.get("trigger"), f"shift was not tagged agent-caused: {sh}"
    assert sh["trigger"]["kind"] == "agent_ask"
    assert sh["dst"] != g.root.intent_id, "destination is the node we were already on"
    # the node recorded on the NEXT turn must be the destination -- proof `current` moved
    later = [r for r in traj.turns if r["turn"] > turn]
    if later:
        assert later[0]["node"] == sh["dst"], (
            f"turn {turn} logged a shift to {sh['dst']} but turn {later[0]['turn']} "
            f"still ran on {later[0]['node']}")


def test_trigger_destination_is_inside_the_matched_categories(graphs, monkeypatch):
    """"Random within the KIND, and only the kinds the words licensed." If the operator
    that fires is outside the matched set, the type restriction is not real."""
    _test_persona(monkeypatch, "trig-cat", (1.0,), sugg=1.0)
    checked = 0
    for g in graphs:
        w = _slot_word_for(g)
        if not w:
            continue
        traj, _, _ = play(g, lambda t, i: _AskerNaming(graph=t, word=w, asks=1),
                          persona="trig-cat", shift_scheduled=True,
                          shift_schedule="persona", p_shift=1.0)
        for _turn, sh in _shift_records(traj):
            tg = sh.get("trigger")
            if not tg:
                continue
            checked += 1
            assert sh["operator"] in tg["categories"], (
                f"fired {sh['operator']} but the question only licensed {tg['categories']}")
            assert "PIVOT" not in tg["categories"], "a pivot must never be agent-triggered"
            assert tg["slots"], "trigger recorded no matched slot"
    assert checked, "no triggered shift was produced across any graph"


def test_an_unrelated_question_does_not_trigger(graphs, monkeypatch):
    """Negative control. Same persona, same suggestibility 1.0 -- only the WORDS differ.
    Without this, a trigger that fires on everything would pass every other test here."""
    _test_persona(monkeypatch, "trig-neg", (1.0,), sugg=1.0)
    g = graphs[0]
    traj, _, _ = play(g, lambda t, i: _AskerNaming(graph=t, word="zzzqqxnonsense", asks=1),
                      persona="trig-neg", shift_scheduled=True,
                      shift_schedule="persona", p_shift=1.0)
    triggered = [sh for _t, sh in _shift_records(traj) if sh.get("trigger")]
    assert not triggered, f"a word matching nothing still triggered a shift: {triggered}"


def test_total_shifts_never_exceed_the_persona_count(graphs, monkeypatch):
    """The invariant that makes the design fair: an agent that asks constantly, at maximum
    suggestibility, must still face no more shifts than a silent one. Budget-minus-one is
    only meaningful if this holds."""
    _test_persona(monkeypatch, "trig-cap", (1.0, 0.5), sugg=1.0)
    for g in graphs:
        w = _slot_word_for(g)
        if not w:
            continue
        traj, _, _ = play(g, lambda t, i: _AskerNaming(graph=t, word=w, asks=8),
                          persona="trig-cap", shift_scheduled=True,
                          shift_schedule="persona", p_shift=1.0)
        n = len(_shift_records(traj))
        assert n <= 2, f"persona declares 2 placements but {n} shifts fired on {g.graph_id}"


def test_the_schedule_is_deterministic_for_a_seed(graphs, monkeypatch):
    """Same graph, same seed, same agent -> same shifts. Axis B must be replayable or no
    paired comparison between arms means anything."""
    _test_persona(monkeypatch, "trig-det", (1.0, 0.5), sugg=1.0)
    g = graphs[0]
    w = _slot_word_for(g)
    runs = []
    for _ in range(2):
        traj, _, _ = play(g, lambda t, i: _AskerNaming(graph=t, word=w, asks=4),
                          persona="trig-det", shift_scheduled=True,
                          shift_schedule="persona", p_shift=1.0)
        runs.append([(t, s["dst"], s["operator"], bool(s.get("trigger")))
                     for t, s in _shift_records(traj)])
    assert runs[0] == runs[1], f"non-deterministic shifts:\n{runs[0]}\n{runs[1]}"


def test_the_shift_cap_is_bound_to_the_persona_count(graphs, monkeypatch):
    """WHAT ACTUALLY ENFORCES "clarity never adds shifts".

    Mutation-tested 2026-08-19: deleting the budget-minus-one `shift_thresholds.pop()`
    changed NOTHING -- 6/6 graphs produced byte-identical shift sequences. The cap is
    really enforced by `traversal.max_shifts`, which episode.py binds to the persona's
    placement count at episode start, and `may_shift` refuses past it. The pop is
    defence-in-depth, not the guard.

    So this test targets the guard itself: if a future change ever lets `max_shifts`
    drift away from the persona count, the fairness invariant silently dies and every
    other shift test still passes. That is exactly the failure this pins down.
    """
    from intent_graph.runtime.traversal import Traversal
    from intent_graph.runtime import persona as P
    for placements in ((1.0,), (1.0, 0.5), (0.66, 0.33)):
        pid = f"cap{len(placements)}-{placements[0]}"
        _test_persona(monkeypatch, pid, placements, sugg=1.0)
        g = graphs[0]
        w = _slot_word_for(g)
        traj, _, _ = play(g, lambda t, i: _AskerNaming(graph=t, word=w, asks=8),
                          persona=pid, shift_scheduled=True,
                          shift_schedule="persona", p_shift=1.0)
        n = len(_shift_records(traj))
        assert n <= len(placements), (
            f"persona declares {len(placements)} placements, {n} shifts fired")
        # and the binding itself, directly
        pers = P.PERSONAS[pid]
        assert len(pers.shift_thresholds(14.0)) == len(placements)


def test_a_proposal_is_judged_against_the_moved_goal(graphs, monkeypatch):
    """THE RULE (ruling 2026-08-19, revised): a proposal is compared with the intent as it
    stands NOW, not the one the user held when the agent started composing it.

    Concretely: if a shift is due when a proposal arrives, the shift fires first and the
    proposal is adjudicated against the destination node. The proof is that the turn's
    recorded node is the shift's destination -- if the old ordering came back, the turn
    would be recorded on the source node instead."""
    _test_persona(monkeypatch, "prop-first", (1.0,), sugg=0.0)
    traj, _, _ = play(graphs[0], lambda t, i: _StubbornProposer(graph=t),
                      persona="prop-first", shift_scheduled=True, p_shift=1.0)
    fired = [u for u in traj.turns if u.get("shift")]
    assert fired, "no shift fired"
    for u in fired:
        if (u.get("action") or {}).get("kind") != "PROPOSE":
            continue
        assert not u["shift"].get("after_reply")
        # the acceptance check on this turn ran against the DESTINATION
        assert u["node"] == u["shift"]["dst"], (
            f"turn recorded on {u['node']} but the shift moved to {u['shift']['dst']}: "
            "the proposal was judged against the stale goal")
