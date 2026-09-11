"""The final arm ladder (B0 -> A0 -> A1 -> A2 -> A3) and A3's cross-task ask knowledge.

Each arm must add exactly one mechanism to the previous one, and A3's knowledge must be
learnable only from what the agent itself could observe.
"""

import json

import pytest

from intent_graph.runtime import askbook
from intent_graph.runtime.agents import ARMS


def _sys(arm: str) -> str:
    return ARMS[arm](llm=None, config={"runtime": {}}).system()


# ------------------------------------------------------------------ the ladder
def test_b0_cannot_speak_and_the_rest_can():
    assert "Action: Clarify" not in _sys("B0")
    for arm in ("A0", "A1", "A2", "A3"):
        assert "Action: Clarify" in _sys(arm), arm


def test_each_rung_adds_its_mechanism():
    a0, a1, a2, a3 = (_sys(a) for a in ("A0", "A1", "A2", "A3"))
    # A1 adds memory
    assert "MEMORY" not in a0 and "KEEP A MEMORY" in a1
    # A2 adds planning and verification on top of memory
    assert "KEEP A MEMORY" in a2
    assert "PLAN THE BUDGET" not in a1 and "PLAN THE BUDGET" in a2
    assert "VERIFY BEFORE YOU COMMIT" not in a1 and "VERIFY BEFORE YOU COMMIT" in a2
    # A3 adds cross-task knowledge on top of A2
    assert "PLAN THE BUDGET" in a3
    assert "USE THE PLAYBOOK" not in a2 and "USE THE PLAYBOOK" in a3


def test_no_arm_is_told_the_economy_in_its_own_guidance():
    """The patience meter reaches A2/A3 through the environment channel
    (runtime.show_patience), never as a hard-coded price list in the prompt."""
    for arm in ("B0", "A0", "A1", "A2", "A3"):
        low = (ARMS[arm].guidance or "").lower()
        for phrase in ("costs 2", "costs 4", "patience_init", "10 points"):
            assert phrase not in low, (arm, phrase)


def test_patience_meter_reaches_a2_only_when_the_environment_shows_it():
    cfg_off = {"runtime": {}}
    cfg_on = {"runtime": {"show_patience": True, "cost_ask": 2, "cost_reject": 4}}
    off = ARMS["A2"](llm=None, config=cfg_off)
    on = ARMS["A2"](llm=None, config=cfg_on)
    off.patience_left = on.patience_left = 7.2
    assert off.patience_note() == ""
    note = on.patience_note()
    assert "7.2" in note and "costs 2" in note and "costs 4" in note


def test_a3_prompt_carries_the_playbook_when_a_book_exists(tmp_path):
    kb = askbook.distill([
        {"sig": askbook.query_signature("a red mug"), "dim": "budget",
         "informative": True, "won": True, "question": "What can you spend?"}
        for _ in range(6)
    ])
    path = tmp_path / "kb.json"
    askbook.save(kb, path)

    class _T:
        turns = [{"role": "user", "content": "a red mug"}]

    with_book = ARMS["A3"](llm=None, config={"runtime": {"askbook_path": str(path)}})
    without = ARMS["A3"](llm=None, config={"runtime": {}})
    assert "WHAT EXPERIENCE SAYS TO ASK ABOUT" in with_book.build_prompt(_T())
    assert "WHAT EXPERIENCE SAYS TO ASK ABOUT" not in without.build_prompt(_T())


# ------------------------------------------------------------------ askbook
def test_observable_view_hides_everything_privileged():
    traj = {"header": {"query": "a mug", "perturbation": {"hidden_slots": ["price_upper"]}},
            "outcome": "SUCCESS",
            "turns": [{"turn": 1, "action": {"kind": "ASK", "question": "budget?"},
                       "reply": "under 20 dollars",
                       "reveals": [{"slot": "price_upper", "value": 20}]}]}
    view = askbook.observable_view(traj)
    blob = json.dumps(view)
    assert "price_upper" not in blob and "reveals" not in blob
    assert view["turns"][0]["question"] == "budget?"
    assert view["turns"][0]["reply"] == "under 20 dollars"


def test_declines_do_not_count_as_information():
    assert askbook._informative("I need it under 30 dollars, that's firm")
    assert not askbook._informative("Oh, either one works for me, I'm not picky")
    assert not askbook._informative("hm")


def test_playbook_ranks_what_paid_off_and_warns_off_what_did_not():
    recs = []
    sig = askbook.query_signature("a rug for the living room")
    for _ in range(8):
        recs.append({"sig": sig, "dim": "budget", "informative": True, "won": True,
                     "question": "What's your budget?"})
    for _ in range(8):
        recs.append({"sig": sig, "dim": "purpose", "informative": False, "won": False,
                     "question": "What room is it for?"})
    book = askbook.render_playbook(askbook.distill(recs), "a rug for the living room")
    assert "budget: usually pays off" in book
    assert "Do not spend questions on: purpose" in book
    assert "What's your budget?" in book


def test_distill_dir_reads_finished_episodes_only(tmp_path):
    good = {"header": {"query": "a mug"}, "outcome": "SUCCESS",
            "turns": [{"action": {"kind": "ASK", "question": "What size?"},
                       "reply": "the 12 oz one please"}]}
    husk = {"header": {"query": "a mug"}, "outcome": "ERROR", "turns": []}
    (tmp_path / "a.json").write_text(json.dumps(good), encoding="utf-8")
    (tmp_path / "b.json").write_text(json.dumps(husk), encoding="utf-8")
    kb = askbook.distill_dir([tmp_path])
    assert kb["n_records"] == 1
    assert kb["global"]["option"]["hit"] == 1


@pytest.mark.parametrize("q,dim", [
    ("What is your budget for this?", "budget"),
    ("Which colour would you like?", "option"),
    ("How many do you need?", "quantity"),
    ("What material should it be?", "feature"),
])
def test_question_classification(q, dim):
    assert askbook.classify_question(q) == dim


def test_stale_check_is_written_on_both_final_paths():
    """Staleness must not depend on WHICH exhaustion path fired: the unaffordable final
    shot records the same stale check the forced-final path does."""
    import inspect

    from intent_graph.runtime import episode as ep
    assert "def _record_stale_check" in inspect.getsource(ep.Episode)
    assert inspect.getsource(ep.Episode._play).count("_record_stale_check") == 2


def test_forced_final_repairs_an_unbuyable_answer_but_never_picks_the_product():
    """The harness restates the required form and gives ONE repair attempt: an apology in
    the Final Answer slot is a format defect (24% of B0's dev episodes), not a decision.
    But the repair only fixes FORM -- an agent that still will not name a product keeps
    its zero, and the harness never substitutes a product of its own."""
    import re as _re

    from intent_graph.runtime.episode import Episode

    class _Adapter:
        """Same contract as the webshop adapter: a purchase parses to a tuple, anything
        else parses to None (which is what later scores 'unparseable_proposal')."""

        def parse_proposal(self, raw):
            m = _re.match(r"\s*(?:buy|purchase)\s+([A-Za-z0-9]{6,})", str(raw or ""),
                          _re.IGNORECASE)
            return (m.group(1), {}) if m else None

    ep = Episode.__new__(Episode)
    ep.adapter = _Adapter()

    class _Agent:
        def __init__(self, reply): self.reply, self.calls = reply, 0
        def act(self, _t):
            self.calls += 1
            return self.reply

    class _T:
        def __init__(self): self.msgs = []
        def env(self, m): self.msgs.append(m)

    # refuses again -> keeps the zero, no substitution
    ep.agent = _Agent("Action: Answer\nFinal Answer: I still cannot find one.")
    rec = {}
    out, repaired = ep._repair_final("I cannot find a suitable product.", _T(), rec)
    assert repaired is False and out == "I cannot find a suitable product."
    assert "final_format_repair" not in rec

    # names a product on the retry -> accepted as the final purchase
    ep.agent = _Agent('Final Answer: buy B01ABCDEFG {"color": "grey"}')
    rec = {}
    out, repaired = ep._repair_final("I cannot find a suitable product.", _T(), rec)
    assert repaired is True and "B01ABCDEFG" in out and rec["final_format_repair"] is True

    # a well-formed answer is never touched (no wasted call)
    agent = ep.agent = _Agent("unused")
    out, repaired = ep._repair_final('buy B0XYZ12345 {"size": "s"}', _T(), {})
    assert repaired is False and agent.calls == 0


def test_style_learning_recommends_the_better_question_form():
    recs = []
    sig = askbook.query_signature("a lamp for the bedroom under 40 dollars")
    for _ in range(10):
        recs.append({"sig": sig, "dim": "option", "style": "audit", "informative": True,
                     "won": True, "question": "So far I have: a brass lamp. Anything off?"})
    for _ in range(10):
        recs.append({"sig": sig, "dim": "option", "style": "targeted", "informative": False,
                     "won": False, "question": "Which shade of brass?"})
    kb = askbook.distill(recs)
    book = askbook.render_playbook(kb, "a lamp for the bedroom under 40 dollars")
    assert "Style that works better: AUDIT" in book


def test_style_classification():
    assert askbook.classify_style("So far I have: twin, grey, under $500. Anything off?") == "audit"
    assert askbook.classify_style("What size do you need?") == "targeted"


def test_b6_calibrate_then_act_contract():
    """B6: can ask, carries the calculus, hard-codes no prices (they arrive through the
    meter), and renders the meter when the environment shows it."""
    from intent_graph.runtime.agents import ARMS
    b6 = ARMS["B6"](llm=None, config={"runtime": {"show_patience": True,
                                                  "cost_ask": 2, "cost_reject": 4}})
    sysp = b6.system()
    assert "Action: Clarify" in sysp and "CALIBRATE, THEN ACT" in sysp
    low = (ARMS["B6"].guidance or "").lower()
    for phrase in ("costs 2", "costs 4", "patience_init"):
        assert phrase not in low
    b6.patience_left = 6.4
    note = b6.patience_note()
    assert "6.4" in note and "costs 2" in note and "costs 4" in note


def test_b5_bed_contract():
    """B5: computed EIG report reaches the prompt when a search observation exists, and
    the stopping rule lives in the guidance."""
    from intent_graph.runtime.agents import ARMS
    b5 = ARMS["B5"](llm=None, config={"runtime": {}})
    assert "VERDICT" in b5.system() and "Action: Clarify" in b5.system()

    class _T:
        turns = [
            {"role": "user", "content": "a rug"},
            {"role": "environment", "content":
                "2 products match most of those words; best 2:\n"
                "  B0AAAAAAAA | $10.00 | match 90% | Rug One\n"
                "      features: washable\n"
                "      options: {'color': ['navy', 'grey']}\n"
                "  B0BBBBBBBB | $12.00 | match 85% | Rug Two\n"
                "      features: soft\n"
                "      options: {'color': ['navy']}\n"},
        ]
    p = b5.build_prompt(_T())
    assert "EXPERIMENT DESIGN OVER THE" in p and "VERDICT" in p   # computed verdict injected
    # 'color' splits and the shopper never said it -> the design selects it
    assert "option:color" in p

    class _Said:
        turns = [{"role": "user", "content":
                  "a navy color rug please, washable and soft"},
                 _T.turns[1]]
    said_p = b5.build_prompt(_Said())
    # stated dimension netted out -> low gain -> the verdict says stop
    assert "STOP asking" in said_p

    class _Empty:
        turns = [{"role": "user", "content": "a rug"}]
    assert "EXPERIMENT DESIGN" not in b5.build_prompt(_Empty())


def test_b4_sage_contract():
    """B4: stateful UNCERTAINTY block round-trips, guidance carries the EVPI gate with its
    known costs, and the state never leaks into the returned action."""
    from intent_graph.runtime.agents import ARMS
    b4 = ARMS["B4"](llm=None, config={"runtime": {}})
    assert "EVPI-GATED" in b4.system() and "Action: Clarify" in b4.system()
    out = b4._absorb('Action: Clarify\nContent: what colour?\n'
                     'UNCERTAINTY: {"params": {"colour": {"value": null, '
                     '"source": "unknown", "confidence": 0.1}}}')
    assert "UNCERTAINTY" not in out and out.startswith("Action: Clarify")
    assert b4.state["params"]["colour"]["confidence"] == 0.1
    assert "UNCERTAINTY" in b4.build_prompt(type("T", (), {"turns": []})())


def test_rate_limiter_paces_requests_and_is_shared():
    """The token bucket smooths the aggregate rate across threads; 0 disables it."""
    import threading
    import time

    from intent_graph.runtime.llm import _RateLimiter

    off = _RateLimiter(0)
    t0 = time.monotonic()
    for _ in range(50):
        off.acquire()
    assert time.monotonic() - t0 < 0.05          # disabled: no pacing at all

    lim = _RateLimiter(200)                       # 200/s -> 5ms apart
    t0 = time.monotonic()
    threads = [threading.Thread(target=lambda: [lim.acquire() for _ in range(10)])
               for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.monotonic() - t0
    assert 0.12 < elapsed < 0.5, elapsed          # 40 calls / 200 per sec ~= 0.2s


def test_computed_split_gate_blocks_only_when_nothing_unstated_splits():
    """B2/B3/B5 all define their gate over the CANDIDATE SET; reading it from the model's
    own claim let them ride the question cap (74%, 99%, 62% of episodes). The computed
    version blocks only when no UNSTATED attribute still separates the survivors."""
    from intent_graph.runtime.agents import unstated_split_remains

    obs = ("2 products match most of those words; best 2:\n"
           "  B0AAAAAAAA | $10.00 | match 90% | Rug One\n"
           "      features: washable\n"
           "      options: {'color': ['navy', 'grey']}\n"
           "  B0BBBBBBBB | $12.00 | match 85% | Rug Two\n"
           "      features: soft\n"
           "      options: {'color': ['navy']}\n")

    class _T:
        def __init__(self, said):
            self.turns = [{"role": "user", "content": said},
                          {"role": "environment", "content": obs}]

    alive, best, who = unstated_split_remains(_T("a rug"))
    assert alive and best >= 0.5 and who                    # colour/features still split

    alive2, _b, _w = unstated_split_remains(_T("a navy color washable soft rug"))
    assert not alive2                                       # all splitters already stated

    assert unstated_split_remains(None)[0] is True          # no evidence -> never block
