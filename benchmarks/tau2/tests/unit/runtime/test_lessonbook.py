"""Lessonbook invariants, including every 2026-08-20 review repro that survived
adversarial verification. If one of these regresses, an A3 result is not reportable."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[5] / "pipeline" / "src"))
from intent_graph.runtime import lessonbook as L                     # noqa: E402

TRAJ = {"header": {"sample_id": "0213288d7b__false_presupposition",
                   "query": "cancel order #W111222 placed by mistake"},
        "outcome": "SUCCESS",
        "turns": [
            {"action": {"kind": "ACT"}, "observation": "..."},
            {"action": {"kind": "ASK", "question": "Which order should I cancel?"},
             "reply": "Order #W111222, and my user id is noah_brown_6181."},
            {"action": {"kind": "PROPOSE"}, "acceptance": {"ok": False},
             "reply": "No wait, also return the shoes from #W999."},
            {"action": {"kind": "PROPOSE"}, "acceptance": {"ok": True}, "reply": "great"}]}

GOOD = {"when": "the request already names the target <ORDER_ID> and the desired change",
        "do": "act without asking; a question here spends patience and changes nothing",
        "avoid": "auditing details the user already stated", "evidence": 6, "utility": "high",
        "confidence": 0.8}


class StubLLM:
    def __init__(self, payload):
        self.payload, self.prompt = payload, None

    def complete(self, prompt, **kw):
        self.prompt = prompt
        return self.payload


# ---------------------------------------------------------------- review criticals
def test_fault_strategy_label_never_reaches_reflection_prompt():
    llm = StubLLM(json.dumps({"lessons": []}))
    L.summarize(llm, prior=[], digests=[L.episode_digest(TRAJ)], profile="tau2_retail")
    assert "false_presupposition" not in llm.prompt
    assert "sample_id" not in llm.prompt


def test_entity_lexicon_rejects_names_cities_products():
    lex = {"tokens": frozenset({"yusuf", "rossi", "philadelphia"}),
           "phrases": ("yusuf rossi", "ergoflex standing desk")}
    bad = {"when": "a customer like yusuf rossi asks for an exchange",
           "do": "verify identity first", "avoid": "", "evidence": 2, "utility": "high",
           "confidence": 0.7}
    assert L.validate_lesson(bad, grams=set(), lexicon=lex).startswith("names_entity")
    bad2 = {"when": "the request mentions the ergoflex standing desk",
            "do": "confirm the variant", "avoid": "", "evidence": 1, "utility": "high",
            "confidence": 0.7}
    assert L.validate_lesson(bad2, grams=set(), lexicon=lex).startswith("names_entity")
    assert L.validate_lesson(GOOD, grams=set(), lexicon=lex) is None


def test_redaction_leaves_no_digit_even_glued():
    t = L.redact("user noah_brown_6181 zip80279 said item9612497925 costs $12.5, W999x")
    assert not any(c.isdigit() for c in t), t


def test_digest_carries_did_markers_for_act_turns():
    d = L.episode_digest(TRAJ)
    assert {"did": "ACT"} in d["sequence"]
    assert d["asks"] == 1 and d["rejected"] == 1


# ---------------------------------------------------------------- review majors
def test_evidence_bool_rejected():
    les = dict(GOOD, evidence=True)
    assert L.validate_lesson(les, grams=set()) == "bad:evidence"


def test_over_cap_lessons_counted():
    many = [dict(GOOD, when=GOOD["when"] + f" variant {chr(97 + i)}") for i in range(12)]
    llm = StubLLM(json.dumps({"lessons": many}))
    res = L.summarize(llm, prior=[], digests=[L.episode_digest(TRAJ)], profile="tau2_retail")
    assert len(res["lessons"]) == L.MAX_LESSONS
    assert res["dropped_by_validator"].get("over_cap") == 4


def test_no_json_keeps_prior():
    llm = StubLLM("sorry, nothing structured today")
    res = L.summarize(llm, prior=[GOOD], digests=[L.episode_digest(TRAJ)],
                      profile="tau2_retail")
    assert res["parse_error"] == "no_json_object"
    kept = [{k: v for k, v in l.items() if k != "id"} for l in res["lessons"]]
    assert kept == [GOOD]                    # prior survives, id-annotated


def test_binding_covers_gen_size_and_samples_sha():
    snap = L.make_snapshot(lessons_result={"lessons": [dict(GOOD, id="L1")], "dropped_by_validator": {}},
                           profile="tau2_retail", run_id="a3v2_r2/A3_dependent", arm="A3",
                           persona="dependent", seed=1, generation=1, parent=None,
                           source_episode_ids=["x"], n_seen=1,
                           gen_size=25, samples_sha="abc123")
    kw = dict(profile="tau2_retail", run_id="a3v2_r2/A3_dependent", arm="A3",
              persona="dependent", seed=1, gen_size=25, samples_sha="abc123")
    L.check_binding(snap, **kw)
    with pytest.raises(L.BindingError):
        L.check_binding(snap, **{**kw, "gen_size": 50})
    with pytest.raises(L.BindingError):
        L.check_binding(snap, **{**kw, "samples_sha": "zzz"})
    with pytest.raises(L.BindingError):
        L.check_binding(snap, **{**kw, "run_id": "a3v2_r1/A3_dependent"})


def test_legacy_askbook_rejected():
    with pytest.raises(L.BindingError):
        L.check_binding({"version": 3, "profile": "tau2_retail"}, profile="tau2_retail",
                        run_id="x", arm="A3", persona="d", seed=1)


def test_build_entity_lexicon_from_db_shape():
    db = {"users": [{"name": {"first_name": "Yusuf", "last_name": "Rossi"},
                     "address": {"address1": "123 Elm Grove Lane", "city": "Philadelphia"}}],
          "products": [{"name": "ErgoFlex Standing Desk"}]}
    p = Path("/tmp/_lb_test_db.json")
    p.write_text(json.dumps(db))
    lex = L.build_entity_lexicon(p)
    assert "yusuf" in lex["tokens"] and "philadelphia" in lex["tokens"]
    assert "ergoflex" not in lex["tokens"]      # products ban as phrases only
    assert "yusuf rossi" in lex["phrases"] and "elm grove lane" in lex["phrases"]


def test_credit_labels_answer_used_and_grounded():
    traj = {"header": {"sample_id": "t2__x", "query": "exchange my keyboard in order #W1000001"},
            "outcome": "SUCCESS",
            "turns": [
                {"action": {"kind": "ACT", "command": 'get_order_details(order_id="#W1000001")'},
                 "observation": '{"items": [{"item_id": "7706410293"}]}'},
                {"action": {"kind": "ASK", "question": "Which variant do you want?"},
                 "reply": "The blue one, item 9025753381 please."},
                {"action": {"kind": "PROPOSE",
                            "command": 'exchange_delivered_order_items(order_id="#W1000001", '
                                       'item_ids=["7706410293"], new_item_ids=["9025753381"])'},
                 "acceptance": {"ok": True}, "reply": "great"},
            ]}
    d = L.episode_digest(traj)
    ask = next(s for s in d["sequence"] if "ask" in s)
    assert ask["answer_used"] is True          # 9025753381 from the reply entered the command
    prop = next(s for s in d["sequence"] if s.get("proposed"))
    assert prop["grounded"] is True            # every arg traces to query/reply/observation
    # now a guessed proposal: new_item_id from nowhere
    traj["turns"][2]["action"]["command"] = ('exchange_delivered_order_items(order_id="#W1000001", '
                                             'item_ids=["7706410293"], new_item_ids=["1111111111"])')
    d2 = L.episode_digest(traj)
    prop2 = next(s for s in d2["sequence"] if s.get("proposed"))
    assert prop2["grounded"] is False
    # unused answer: reply content never appears later
    traj["turns"][1]["reply"] = "Hmm honestly whatever is fine with me dear."
    traj["turns"][2]["action"]["command"] = ('exchange_delivered_order_items(order_id="#W1000001", '
                                             'item_ids=["7706410293"], new_item_ids=["7706410293"])')
    d3 = L.episode_digest(traj)
    ask3 = next(s for s in d3["sequence"] if "ask" in s)
    assert ask3["answer_used"] is False


def test_credit_labels_never_read_acceptance_reason():
    traj = {"header": {"sample_id": "t3__x", "query": "cancel order #W2000002"},
            "outcome": "EXHAUSTED",
            "turns": [{"action": {"kind": "PROPOSE", "proposal_raw": "done"},
                       "acceptance": {"ok": False,
                                      "reason": "golden_actions_missing:secret_tool_name"},
                       "reply": "that is not right"}]}
    d = L.episode_digest(traj)
    assert "secret_tool_name" not in json.dumps(d)


def test_confidence_required_and_stability_guard():
    # missing confidence rejected
    assert L.validate_lesson({k: v for k, v in GOOD.items() if k != "confidence"},
                             grams=set()) == "bad:confidence"
    assert L.validate_lesson(dict(GOOD, confidence=True), grams=set()) == "bad:confidence"
    assert L.validate_lesson(dict(GOOD, confidence=1.4), grams=set()) == "bad:confidence"
    # a high-confidence prior lesson silently dropped by the model is re-inserted
    prior = [dict(GOOD, id="L1", confidence=0.9),
             dict(GOOD, when=GOOD["when"] + " variant b", id="L2", confidence=0.3)]
    llm = StubLLM(json.dumps({"lessons": [
        dict(GOOD, when="a completely new situation arises mid task", confidence=0.5)]}))
    res = L.summarize(llm, prior=prior, digests=[L.episode_digest(TRAJ)],
                      profile="tau2_retail")
    ids = {l["id"] for l in res["lessons"]}
    assert "L1" in ids                                    # high-conf survivor
    assert "L2" not in ids                                # low-conf may be dropped
    assert res["dropped_by_validator"].get("stability_reinserted") == 1


def test_phase_gate_renders_book_only_after_first_submission():
    import sys as _s
    from intent_graph.runtime.agents import SelfEvolvingAgent
    from intent_graph.runtime import askbook
    a = SelfEvolvingAgent.__new__(SelfEvolvingAgent)
    a._profile = askbook.get_profile("tau2_retail")
    from intent_graph.runtime import lessonbook as LB
    a._lessonbook = LB
    a._book = {"lessons": [dict(GOOD, id="L1")], "n_episodes_seen": 10, "generation": 2}
    a.rt = {}                                   # default gate: reject-only

    class T:
        def __init__(self, turns): self.turns = turns
    calls = []
    class Base:
        def build_prompt(self, transcript): return "SCAFFOLD\nYour single action now:"
    # monkey-style: bind the parent implementation
    import types
    orig = SelfEvolvingAgent.__mro__[1].build_prompt
    SelfEvolvingAgent.__mro__[1].build_prompt = Base().build_prompt.__func__ if hasattr(Base().build_prompt, "__func__") else (lambda self, tr: "SCAFFOLD\nYour single action now:")
    try:
        pre = a.build_prompt(T([{"role": "user", "content": "hi"}]))
        assert "WHAT THIS RUN'S EARLIER EPISODES TAUGHT" not in pre       # clean phase = A2 text
        post = a.build_prompt(T([{"role": "agent", "content": "Action: Answer\nFinal Answer: done"},
                                 {"role": "user", "content": "no, that's wrong"}]))
        assert "WHAT THIS RUN'S EARLIER EPISODES TAUGHT" in post          # recovery phase = book
        assert post.index("EPISODES TAUGHT") < post.index("Your single action now:")
    finally:
        SelfEvolvingAgent.__mro__[1].build_prompt = orig


def test_failed_act_calls_surface_error_text_in_digest():
    traj = {"header": {"sample_id": "t4__x", "query": "cancel order #W3000003"},
            "outcome": "EXHAUSTED",
            "turns": [
                {"action": {"kind": "ACT", "command": 'cancel_pending_order(order_id="#W3000003", reason="No longer needed")'},
                 "observation": "error: ValueError: Invalid reason: must be one of ['no longer needed', 'ordered by mistake']"},
                {"action": {"kind": "PROPOSE", "proposal_raw": "done"},
                 "acceptance": {"ok": False}, "reply": "it is not cancelled"}]}
    d = L.episode_digest(traj)
    act = next(s for s in d["sequence"] if s.get("did") == "ACT")
    assert "error" in act and "invalid reason" in act["error"].lower()
    assert "3000003" not in act["error"]          # redaction still applies


def test_reflection_json_survives_reasoning_prose_around_it():
    """With thinking enabled a model prefixes its answer with reasoning that may
    contain braces. The old greedy `{.*}` extractor swallowed that prose and the
    reflection failed (measured 2026-08-25: JSONDecodeError at char 1313, book
    silently empty for a whole cell). Balanced-block scanning reads it correctly."""
    import json
    from intent_graph.runtime.lessonbook import _json_candidates
    noisy = ('Thinking: the agent {maybe} should ask earlier.\n'
             '{"lessons": [{"id": "L1", "when": "a", "do": "b"}]}')
    got = None
    for c in _json_candidates(noisy):
        try:
            o = json.loads(c)
        except Exception:
            continue
        if isinstance(o, dict) and "lessons" in o:
            got = o
            break
    assert got and got["lessons"][0]["id"] == "L1"


def test_json_candidates_returns_nothing_without_braces():
    from intent_graph.runtime.lessonbook import _json_candidates
    assert _json_candidates("no json here at all") == []
