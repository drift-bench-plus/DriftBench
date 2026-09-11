"""LLM agent arms for the experiments (docs/experiments-v1.md §2).

Every arm satisfies the same contract the scripted agents do -- ``act(transcript) -> str`` --
and emits one Drift-Bench action line, which the episode loop parses. Arms differ by
MECHANISM, never by wording alone, so a delta between two arms is attributable to the
mechanism rather than to prompt luck.

What an arm is allowed to know: the dialogue, and that a shopping request has three kinds of
requirement (features, an option, a price cap). It never sees the graph, the mask, the hidden
slot names, or the ground truth -- those live in the sample's scoring block.

The arm name goes into the cache's ``prompt_version``, so two arms asking the same question at
the same point in the same dialogue do not share a cache entry.
"""

from __future__ import annotations

import json
import logging
import re

from .agent_api import ActionKind, PROMPT_FORMAT, PROMPT_FORMAT_NO_ASK, parse_action
from .llm import ROLE_ACT, ROLE_AUDIT, ROLE_SAMPLE

log = logging.getLogger(__name__)

# Domain framing, shared by every arm. Not leakage: a shopping assistant knows a request has
# features, options and a budget. It carries no values and no slot identifiers.
SHOP_DOMAIN = """
You are helping a shopper in an online store. A complete request has three kinds of detail:
  - required product features (e.g. "machine washable", "gluten free")
  - a product option (e.g. size, colour, flavour, pack count)
  - the most they are willing to spend

You do not know the catalogue, so you must BROWSE the store. The store has exactly two
controls, and every page tells you what is clickable on it:

Action: Operation
```
search[<words describing the product>]
```
Action: Operation
```
click[<the exact text of something on the page>]
```

HOW A PURCHASE ACTUALLY WORKS -- four steps, in this order:
  1. search[...]            -> a results page listing products: ASIN, title, PRICE ONLY.
  2. click[<ASIN>]          -> that product's ITEM PAGE. This is the ONLY place the
                               product's options and features are visible. You cannot know
                               what sizes, colours or flavours an item has until you open
                               it, so a listing alone is never enough to choose.
  3. click[<option value>]  -> selects an option (e.g. click[octagonal], click[6 wide]).
                               Repeat for each option the item offers. Selected options are
                               part of what you are buying.
  4. click[Buy Now]         -> offers THIS item, with the options you selected, to the
                               shopper.

click[Buy Now] is the purchase. The shopper decides: if it is what they wanted the sale is
done; if not, they refuse it, you are returned to the store, and you must find something
else -- a refused purchase costs their patience and you will have to navigate again.

Other useful clicks: click[Back to Search] to start a new search, click[< Prev] and
click[Next >] to move through results.

READING AN ITEM'S DETAILS -- and getting back. click[Description], click[Features] and
click[Reviews] open a SEPARATE page. On that page the ONLY things you can click are
click[< Prev] and click[Back to Search]: the options and Buy Now are NOT there. So after
reading, click[< Prev] to return to the item page before selecting options or buying.
Pressing Buy Now from a details page does nothing at all and wastes the action.

HOW SEARCH WORKS (basic tool literacy, same for everyone): the store is keyword
retrieval, not a database. Queries are 2-4 plain words -- the product category plus one
or two core attributes ("navy area rug", "twin daybed wood"). Never put measurements,
prices, or long exact phrases in the search box; match those by READING the results.
Many products per search is normal -- read down the list before deciding an item is not
there; listings often word things differently than the shopper (a listing may say
8'6" x 12' where the shopper said 8 ft 6 in x 12 ft) -- judge whether the product IS
what they want, not whether the words match. If results look wrong, change the WORDS
(drop the weakest term, try a synonym); never re-run the same words in a different
order.

LOOKING IS FREE -- searching, opening item pages, and reading them all cost nothing.
Only a refused purchase costs the shopper anything, and you have far more turns than you will use. So do not buy the first plausible
thing you see: run several DIFFERENT searches (different words, broader and narrower),
read each list to the end, and compare the real candidates against everything the shopper
has told you. The best match is usually not in the first result set, and the difference
between a near match and an exact one is almost always another search you did not run.

COMMIT: when the time comes to answer, always name your best candidate. A near match
earns partial credit; a refusal, an apology, or a question in place of an answer earns
exactly zero, always.
"""

# NOTE (2026-08-12): SHOP_DOMAIN used to end "A wrong buy is rejected and costs the shopper's
# patience, as does an unnecessary question -- but so does never buying anything", which handed
# the whole PATIENCE ECONOMY to every arm including the strategy-free baseline. Injecting the
# cost model verbatim IS a published method (Calibrate-Then-Act), so it belongs to the arms
# whose mechanism includes it and to no others: A states the arithmetic inside its own guidance,
# and B1/B2/B3 get nothing because their papers assume no cost model. What remains above is
# environment mechanics (a mismatched buy is rejected), which an agent must know to act at all.


# --------------------------------------------------------------- candidate arithmetic
# Two of the published baselines define question value over the CANDIDATE SET: B2's
# importance is "would the answer change which product I buy", B3's split is "which attribute
# divides the pool". Asking the model to judge that is where both methods fail -- a model
# always finds something it calls important, so the rule never says no (measured: 0-2.7%
# silent episodes across seven arms). The store's own reply is structured, so the same
# quantity can be COMPUTED and handed to the model as an observation. That is closer to the
# papers, not further: they specify an arithmetic over candidates, not an intuition.
_RE_CAND = re.compile(r"^\s{2}(\S+)\s*\|\s*\$([\d.]+)\s*\|\s*match\s*(\d+)%\s*\|\s*(.+)$")
_RE_FEATS = re.compile(r"^\s+features:\s*(.+)$")
_RE_OPTS = re.compile(r"^\s+options:\s*(\{.*\})\s*$")


def parse_search_results(text: str) -> list[dict]:
    """Structured candidates from one store reply. Never raises: junk yields []."""
    out: list[dict] = []
    for line in str(text or "").splitlines():
        m = _RE_CAND.match(line)
        if m:
            out.append({"asin": m.group(1), "price": float(m.group(2)),
                        "match": int(m.group(3)), "name": m.group(4).strip(),
                        "features": [], "options": {}})
            continue
        if not out:
            continue
        mf = _RE_FEATS.match(line)
        if mf:
            out[-1]["features"] = [f.strip().lower() for f in mf.group(1).split(",")
                                   if f.strip()]
            continue
        mo = _RE_OPTS.match(line)
        if mo:
            try:
                import ast
                opts = ast.literal_eval(mo.group(1))
                if isinstance(opts, dict):
                    out[-1]["options"] = {str(k).lower(): [str(v) for v in (vals or [])]
                                          for k, vals in opts.items()}
            except (ValueError, SyntaxError):
                pass
    return out


def attribute_disagreement(cands: list[dict]) -> dict[str, float]:
    """How evenly each attribute divides the candidates, in [0, 1].

    1.0 means a perfectly even split (the most informative possible question); 0.0 means every
    candidate agrees, so no answer can change which one is bought -- the case both methods are
    supposed to detect and never do. Features are scored by presence/absence, option
    dimensions by how many candidates offer a choice at all.
    """
    if len(cands) < 2:
        return {}
    n = len(cands)
    scores: dict[str, float] = {}
    feats = {f for c in cands for f in c["features"]}
    for f in feats:
        have = sum(1 for c in cands if f in c["features"])
        share = have / n
        scores[f] = round(2 * min(share, 1 - share), 3)      # 0.5/0.5 -> 1.0
    dims = {d for c in cands for d in c["options"]}
    for d in dims:
        offered = [c["options"].get(d) or [] for c in cands]
        distinct = {tuple(sorted(v)) for v in offered}
        multi = sum(1 for v in offered if len(v) > 1)
        # a dimension splits the pool if candidates offer DIFFERENT value sets, or if a
        # single candidate offers several values and the choice is still open
        scores[f"option:{d}"] = round(min(1.0, (len(distinct) - 1) / max(1, n - 1)
                                          + multi / n), 3)
    return scores


def candidate_report(text: str, *, top: int = 4) -> str:
    """One line the model cannot argue with: what its candidates disagree and agree on."""
    cands = parse_search_results(text)
    if len(cands) < 2:
        return ""
    scores = attribute_disagreement(cands)
    if not scores:
        return ""
    split = [(k, v) for k, v in scores.items() if v >= 0.25]
    split.sort(key=lambda kv: -kv[1])
    agreed = [k for k, v in scores.items() if v == 0.0]
    lines = [f"\nOF THE {len(cands)} CANDIDATES YOU JUST SAW (computed, not opinion):"]
    if split:
        lines.append("  they DISAGREE on: "
                     + ", ".join(f"{k} ({v:.2f})" for k, v in split[:top]))
    else:
        lines.append("  they disagree on NOTHING that matters -- no question can change "
                     "which one you buy.")
    if agreed:
        lines.append("  they ALL already satisfy: " + ", ".join(sorted(agreed)[:8]))
    return "\n".join(lines)


def last_search_observation(transcript) -> str:
    for turn in reversed(getattr(transcript, "turns", []) or []):
        if turn.get("role") == "environment" and "match" in str(turn.get("content", "")):
            return str(turn["content"])
    return ""


def _flatten(transcript, keep_obs: int | None = None) -> str:
    """Render the dialogue as text for a single-prompt completion API.

    ``keep_obs`` retains only the last K ENVIRONMENT observations verbatim and replaces
    older ones with a one-line marker; None (the default) keeps everything, so every result
    measured before this existed stands unchanged. It exists because the transcript is
    resent every turn and observations dominate that cost: measured on A2, a 20-turn episode
    ships ~200k tokens of which 72% is observations the model has already seen and only 5%
    is new. USER and YOU turns are never dropped -- they are what the shopper said and what
    the agent decided, which is the episode's actual state.
    """
    lines = []
    if keep_obs is not None:
        env_idx = [i for i, t in enumerate(transcript.turns) if t["role"] == "environment"]
        keep = set(env_idx[-keep_obs:]) if keep_obs > 0 else set()
    for i, turn in enumerate(transcript.turns):
        role = {"user": "USER", "agent": "YOU", "environment": "ENVIRONMENT"}.get(
            turn["role"], turn["role"].upper())
        content = turn["content"]
        if keep_obs is not None and turn["role"] == "environment" and i not in keep:
            content = "(earlier search results omitted)"
        lines.append(f"{role}: {content}")
    return "\n".join(lines) if lines else "(no messages yet)"


class LLMAgent:
    """Base arm: builds a prompt, calls the model, returns one raw action line."""

    arm = "B0"
    guidance = ""            # the arm's one mechanism-free instruction, if any

    def __init__(self, *, llm, config: dict, graph=None, sample=None,
                 adapter=None, executor=None) -> None:
        self.llm = llm
        self.config = config
        self.rt = config.get("runtime", {})
        self.graph = graph
        self.sample = sample
        self.adapter = adapter
        self.executor = executor
        self.notes: list[str] = []      # per-episode diagnostics, surfaced in the trajectory

    def must_commit(self, transcript) -> bool:
        """Optional arm-specific commitment gate; the base agent has none.

        Clarification is limited by the user's patience economy, not by a separate question
        counter. Subclasses may use this hook for a mechanism-specific stopping rule, but
        question count is not one.
        """
        return False

    # ------------------------------------------------------------------ prompting
    def domain_prompt(self) -> str:
        """Domain framing + tool literacy. WebShop's is the module default; other
        benchmarks supply theirs through the adapter (tau2: tool manual WITH signatures
        plus the operator policy -- measured 0/8 -> 82.5% on retail without/with)."""
        dp = getattr(self.adapter, "agent_domain_prompt", None)
        if dp is not None:
            try:
                text = dp() if callable(dp) else str(dp)
                if text:
                    return text
            except Exception as exc:
                # OPTIONAL adapter hook supplying the domain prompt. The WebShop default
                # below is a safe fallback, but an arm silently prompted for the wrong
                # domain would be very hard to spot from results alone.
                log.debug("domain prompt hook failed: %s", exc)
        return SHOP_DOMAIN

    def system(self) -> str:
        return PROMPT_FORMAT + self.domain_prompt() + (self.guidance or "")

    # Coaching is opt-in per arm (2026-08-12). The note used to tell EVERY arm what to do as
    # the turn budget ran down -- "Start converging: search, then buy", "Do not ask anything
    # more" -- which is a strategy handed to the strategy-free baseline. Its original
    # justification (arms looped and never bought) predates forced final submission, which now
    # guarantees a final answer regardless. So the default is a bare count, and the floor rule
    # (ruling 2026-08-12) may switch coaching back on for B0 if the strip drops it near the
    # mute-user floor -- coaching is generic task literacy, unlike the clarification menu.
    coach_on_budget = False

    def budget_note(self, transcript) -> str:
        """How many actions are left, and what committing costs.

        TURN BUDGET IS PUBLIC; PATIENCE IS NOT (ruling 2026-08-20). The turn budget is the
        ENVIRONMENT's hard limit -- every agent operating in this store is subject to it and
        knowing it is basic task literacy, like knowing the action syntax. The shopper's
        PATIENCE is a model of the person, not of the environment: perceiving it is a
        capability some methods have and others do not, which is precisely what
        `runtime.show_patience` gates (see `patience_note`). Publishing patience to every arm
        would erase a real difference between arms; publishing the turn budget does not.

        The COMMIT COST belongs with the turn budget, not with patience. On the faithful site
        a purchase is not one action: you must click into the item, click each option, then
        click Buy Now. An agent told only "N actions left" will browse until N is small and
        then have no room left to buy at all -- a budget cannot shape behaviour if the price
        of finishing is invisible. That price is a fact about the environment's action space,
        so every arm gets it. The *advice* about when to converge stays behind
        `coach_on_budget`: telling an arm HOW to spend its budget is a strategy, and the
        strategy-free baseline must not be handed one. The commit-cost sentence describes
        WebShop's action space; a benchmark that prompts through its adapter states its own
        commit mechanics in the adapter's domain prompt, so there the note is a bare count.
        """
        used = sum(1 for t in transcript.turns if t["role"] == "agent")
        left = max(0, int(self.rt.get("max_turns", 16)) - used)
        if not (self.coach_on_budget or self.rt.get("coach_on_budget", False)):
            if self._is_shop():
                commit = "Buying takes several actions: click the item, click each option, then Buy Now."
                return f"\n{left} actions left. {commit}"
            return f"\n{left} actions left."
        if left <= 2:
            return (f"\nONLY {left} ACTIONS LEFT. Search if you have no ASIN yet, then buy. "
                    f"Do not ask anything more.")
        if left <= 5:
            return f"\n{left} actions left. Start converging: search, then buy."
        return f"\n{left} actions left."

    def patience_note(self) -> str:
        """The shopper's remaining goodwill -- only when the environment exposes it.

        OFF by default (``runtime.show_patience``), so every result measured before this
        existed stands unchanged. It exists because the measured cause of question
        saturation is an invisible cost: the agent was shown only how many ACTIONS
        remained (40) and never the patience meter, so from its side asking was free.
        Across five arms the share of episodes with no question at all was 0.0-2.7%, and
        the k-th question's hit rate collapses after the first (50.9% -> 11.2% -> 6.8%),
        i.e. roughly three of every 4.6 questions are waste that still costs a point.
        The papers this harness benchmarks assume the cost model is known to the agent
        (Calibrate-Then-Act injects it verbatim), so exposing it is also fairer to them.
        """
        if not self.rt.get("show_patience", False):
            return ""
        left = getattr(self, "patience_left", None)
        if left is None:
            return ""
        ask = int(self.rt.get("cost_ask", 1))
        rej = int(self.rt.get("cost_reject", 2))
        line = (f"\nThe shopper's patience: {left} left. Asking a question costs {ask}; "
                f"a rejected purchase costs {rej}. At zero they stop shopping with you.")
        if left <= max(ask, 2):
            line += " There is no room for another question -- search and buy."
        return line

    def _keep_obs(self):
        """How many recent observations to keep verbatim; None = all (default, unchanged)."""
        v = self.rt.get("transcript_keep_obs")
        return None if v in (None, "", -1, "-1") else int(v)

    def build_prompt(self, transcript) -> str:
        return (f"Conversation so far:\n{_flatten(transcript, self._keep_obs())}\n"
                f"{self.budget_note(transcript)}{self.patience_note()}\n\n"
                f"Your single action now:")

    def _fallback_query(self, transcript) -> str:
        """A deterministic search built from what the USER said -- never from the graph."""
        user_text = " ".join(t["content"] for t in transcript.turns if t["role"] == "user")
        return _keywords(user_text)

    def _is_shop(self) -> bool:
        """WebShop is the module default; every other benchmark supplies an adapter prompt."""
        return getattr(self.adapter, "agent_domain_prompt", None) is None

    def _harness_action(self, transcript, *, allow_safe: bool = True) -> str | None:
        """A domain-VALID action for the harness to take on the arm's behalf, or None.

        WebShop has a universal safe move -- search the catalogue -- and that is what this
        used to return unconditionally. It is not a tau2 action. On retail/airline the
        injected `search[...]` cannot parse, so it spent the turn AND wrote a parse error
        into the transcript that the model then read back on every later turn. Measured on
        one glm-4.7 retail B0 cell: 609 injections across 205 of 555 episodes, against 22
        for qwen3.7-max and 0 for gpt-5.5 -- the penalty landed hardest on whichever model
        formats worst, which is precisely the unfairness the parser fixes removed (D28).

        `allow_safe` separates the two callers, which want different things:
          - RULE OVERRIDE (allow_safe=True): the arm chose an action its own declared rule
            forbids, and the rule is the baseline's mechanism. Substituting a valid
            read-only call enforces the rule the way the WebShop search did.
          - UNPARSEABLE REPLY (allow_safe=False): the model failed to produce an action at
            all. Handing it a useful observation would REWARD the failure, and unequally --
            only models that misformat would collect it. The honest record is MALFORMED,
            which `malformed%` already reports.
        """
        if self._is_shop():
            return (f"Action: Operation\n```\nsearch[{self._fallback_query(transcript)}]\n```")
        safe = getattr(self.adapter, "agent_safe_action", None)
        if safe and allow_safe:
            return f"Action: Operation\n```\n{safe}\n```"
        return None

    def _complete(self, prompt: str, *, role: str = ROLE_ACT, system: str | None = None) -> str:
        return self.llm.complete(prompt, role=role, system=system or self.system())

    # ------------------------------------------------------------------ the contract
    def act(self, transcript) -> str:
        """One raw action line.

        A malformed reply gets exactly one retry with the parser's own complaint quoted back;
        if that also fails the arm asks a safe question rather than crashing the episode, and
        the event is recorded so a chronically malformed arm is visible instead of silently
        looking indecisive.

        Clarification has no separate count cap. Each question is charged by the episode's
        patience economy, whose affordability rule ends the interaction when the user can no
        longer pay for one. A valid Clarify action is never rewritten based on question count.
        """
        prompt = self.build_prompt(transcript)
        raw = self._complete(prompt)
        action = parse_action(raw)
        if action.kind.name != "MALFORMED" and not action.problems:
            return raw

        problems = ", ".join(action.problems) or "no recognised action"
        retry = (f"{prompt}\n\nYour previous reply could not be parsed ({problems}):\n"
                 f"{raw[:400]}\n\nReply again using EXACTLY the required format, nothing else.")
        raw2 = self._complete(retry)
        if parse_action(raw2).kind.name != "MALFORMED":
            self.notes.append(f"{self.arm}:format_retry_ok")
            return raw2
        self.notes.append(f"{self.arm}:format_retry_failed")
        # The safe fallback must stay inside the arm's own action space: an arm whose menu
        # has no user channel (B0) cannot have the harness ask on its behalf -- that
        # injected question is an ability the arm does not possess.
        if "Clarify" not in self.system():
            fb = self._harness_action(transcript, allow_safe=False)
            # None => leave the reply MALFORMED rather than invent an action for a model
            # that produced none. Both spend the turn, but a fabricated call also poisons
            # the transcript the model reads back on every later turn.
            return fb if fb is not None else raw2
        return ("Action: Clarify\n"
                "Content: Could you tell me more about what you are looking for?")


def _keywords(text: str, limit: int = 6) -> str:
    import re as _re
    stop = {"the", "and", "for", "with", "that", "this", "you", "your", "have", "has",
            "need", "want", "looking", "please", "would", "like", "can", "could", "about"}
    words = [w for w in _re.findall(r"[a-z0-9]+", (text or "").lower())
             if len(w) > 2 and w not in stop]
    return " ".join(dict.fromkeys(words))[:80] or "product"


class LiteralActor:
    """L0 -- the honest floor: act on the query's face value, never speak to the user.

    Replaces IntentIgnorer as the experimental floor. IntentIgnorer proposes members of the
    ROOT's ground truth -- it reads the answer sheet, which is right for the battery's
    staleness check and completely wrong as a floor: it scored 0.69 "under perturbation"
    because perturbation never touched what it read. This agent compiles the sample's
    LITERAL READING -- what a naive reader would take the query to mean -- executes it, and
    proposes those products. On a false premise the literal set is empty and it buys
    nothing, which is the honest floor behaviour: an unquestioning agent CANNOT win there.
    """

    arm = "L0"
    name = "literal_actor"

    def __init__(self, *, graph, sample, adapter, executor, **_kw) -> None:
        self.graph = graph
        self.sample = sample
        self.adapter = adapter
        self.executor = executor
        self.notes: list[str] = []
        self._members: list | None = None
        self._i = 0

    def _literal_members(self) -> list:
        if self._members is not None:
            return self._members
        readings = (self.sample.get("mask") or {}).get("literal_readings") or []
        # An ambiguity sample carries TWO readings and, by the adprotocol rule, one of
        # them equals the truth -- always taking readings[0] made the floor an oracle
        # there (30/30 on lexical in the first campaign). A naive reader resolves the
        # ambiguity arbitrarily: pick one reading, seeded by the sample id so replays
        # and arms see the same choice.
        idx = 0
        if len(readings) > 1:
            sid = str(self.sample.get("sample_id", ""))
            idx = int.from_bytes(sid.encode()[-4:] or b"0", "big") % len(readings)
            self.notes.append(f"L0:reading_choice={idx}/{len(readings)}")
        conds = tuple(tuple(c) for c in (readings[idx] if readings else []))
        members = []
        if conds:
            try:
                with self.executor.open(self.graph.env_spec) as session:
                    recipe = self.adapter.compile(dict(self.graph.root.base), conds)
                    gt = self.adapter.execute(recipe, session)
                    members = list(gt.value)[:3]
            except Exception as exc:
                self.notes.append(f"L0:literal_exec_failed:{type(exc).__name__}")
        self._members = members
        self.notes.append(f"L0:literal_members={len(members)}")
        return members

    def act(self, transcript) -> str:
        import json as _json
        members = self._literal_members()
        if self._i >= len(members):
            # nothing (left) satisfies the query's face value: the floor stops, honestly
            return ("Action: Answer\nPredicted user question: whatever was asked\n"
                    "Final Answer: buy NOTHING {}")
        raw = members[self._i]
        self._i += 1
        try:
            asin, opts = _json.loads(raw)
            options = {k: v for k, v in opts}
        except Exception:
            asin, options = str(raw), {}
        return ("Action: Answer\nPredicted user question: the literal request\n"
                f"Final Answer: buy {asin} {_json.dumps(options)}")


class DirectAgent(LLMAgent):
    """B0 -- a REGULAR agent (ruling 2026-08-13): it does not ask, because the user-message
    channel is not in its action menu at all. It works the store and buys; the purchase IS
    its proposal (the buy-shaped Operation path adjudicates it), a rejection is the only
    feedback it ever gets, and when it cannot afford another rejection its next purchase is
    the final submission. The earlier reading -- 'permitted by the format but never
    encouraged' -- still let alignment-trained asking leak through at ~3.6 asks/episode."""

    arm = "B0"
    guidance = ""

    def system(self) -> str:
        return PROMPT_FORMAT_NO_ASK + self.domain_prompt() + (self.guidance or "")


class FreeFormAgent(LLMAgent):
    """A0 -- GATE-style: told to elicit the need before committing. No belief state, no gate.

    Li, Tamkin, Goodman & Andreas, ICLR 2025 (arXiv 2310.11589). The floor every structured
    method must beat: if a mechanism cannot beat "just ask them", it is not earning its cost.
    """

    arm = "A0"
    guidance = """
Requests are sometimes incomplete or mistaken. If the shopper's own words leave you unsure
what they actually want, ask them -- and only then; a question about anything else (the
store, availability, your search) is wasted, and they cannot answer it anyway. Never
re-ask something they have already answered or refused.

BUT PREFER TO ACT. Shoppers tire of being interviewed, and showing them a product tells you
everything a question would: if it is wrong they will say what is wrong, and if it is right
you are finished. So ask AT MOST ONE question, BEFORE your first purchase -- make it count,
early, while an answer can still change what you buy -- and then never ask again. Once you
have shown them a product their reaction is your information: read what they told you,
search with those words, change product, and buy again. Every further question is an
attempt at the shelf you did not take.
"""


class InformedSkyline(LLMAgent):
    """RS -- the fair ceiling for an LLM agent: it KNOWS the true intent, but must still
    search the store and buy.

    The graph-reading Oracle proposes ground truth directly, so its 1.00 says only that the
    verifier accepts true answers. It says nothing about whether an LLM that understood the
    user perfectly could complete the purchase -- search, option matching, price checking.
    RS measures exactly that execution ceiling. The gap (RS - B0) is what intent recovery is
    WORTH to an LLM agent; comparing B0 to 1.00 overstated it by charging execution failures
    to misunderstanding. Privileged and clearly labelled: never compare agents to it, compare
    the CAMPAIGN's arms against it.
    """

    arm = "RS"

    def system(self) -> str:
        conds = (self.sample or {}).get("hidden_intent", {}).get("conditions") or []
        feats = [str(v) for slot, _op, v in conds if str(slot).startswith("attr:")]
        opts = [str(v) for slot, _op, v in conds if str(slot).startswith("option:")]
        caps = [v for slot, _op, v in conds if slot == "price_upper"]
        truth = ("\nYou happen to KNOW exactly what the shopper wants "
                 "(do not ask them anything):\n"
                 f"  required features: {feats}\n"
                 f"  option to select : {opts or ['(none)']}\n"
                 f"  price limit      : {caps or ['(none)']}\n"
                 "Search the store for it and buy the best match.")
        return PROMPT_FORMAT + self.domain_prompt() + truth


AUDIT_SCHEMA = """Return ONLY a JSON object, no prose:
{"features": [{"value": <string>, "status": "STATED"|"INFERRED"|"ABSENT"|"CONFLICTING",
               "evidence": <exact user words or null>}],
 "option":    {"value": <string or null>, "status": ..., "evidence": ...},
 "price_cap": {"value": <number or null>, "status": ..., "evidence": ...}}

Rules:
  STATED      -- the user said it; evidence MUST be their exact words, quoted verbatim.
  INFERRED    -- you are assuming it; evidence must be null. Never dress an assumption as STATED.
  ABSENT      -- not mentioned at all.
  CONFLICTING -- you were told two different things, or a proposal was rejected on this point.
"""


class SlotAuditAgent(LLMAgent):
    """A1 -- Ask-when-Needed: audit the belief state before every action.

    Wang, Shi et al., EMNLP 2025 (arXiv 2409.00557). Mechanism: each turn, rebuild an explicit
    belief table where every STATED value must carry a *verbatim quote* from the user; ask about
    the weakest-evidenced deficit; propose only when nothing load-bearing is ABSENT. The quote
    requirement is what operationalises grounded-vs-inferred inside the agent, and it is the
    reason this arm should beat A0 on question aim even where it does not beat it on success.
    """

    arm = "A1"
    guidance = """
Ask about details you genuinely do not know. Never ask about something the shopper has
already told you, and do not ask when you have everything you need.
"""

    def audit(self, transcript) -> dict | None:
        prompt = (f"Conversation so far:\n{_flatten(transcript, self._keep_obs())}\n\n"
                  f"Audit what you know about this shopper's requirements.\n{AUDIT_SCHEMA}")
        raw = self._complete(prompt, role=ROLE_AUDIT,
                             system="You are a careful analyst. Output JSON only.")
        return _first_json(raw)

    def build_prompt(self, transcript) -> str:
        table = self.audit(transcript)
        if table is None:
            self.notes.append(f"{self.arm}:audit_unparsed")
            return super().build_prompt(transcript)

        deficits = _deficits(table)
        self.notes.append(f"{self.arm}:deficits={len(deficits)}")
        if deficits and self.must_commit(transcript):
            self.notes.append(f"{self.arm}:budget_commit")
            deficits = []
        if deficits:
            target = deficits[0]
            plan = (f"You are missing this: {target}. Ask ONE question about it "
                    f"(Action: Clarify) -- unless you have already asked about it and been "
                    f"answered or refused, in which case move on.")
        else:
            plan = ("Every requirement is supported by something the shopper actually said. "
                    "Search for a matching product if you have no ASIN yet, then buy.")
        return (f"Conversation so far:\n{_flatten(transcript, self._keep_obs())}\n"
                f"{self.budget_note(transcript)}\n\n"
                f"Your belief table:\n{json.dumps(table, ensure_ascii=False)}\n\n"
                f"{plan}\n\nYour single action now:")


def _deficits(table: dict) -> list[str]:
    """Unsupported requirements, most verifier-critical first.

    Ordered price_cap > option > features because the price ceiling is a hard filter (a
    too-expensive item is rejected outright) while a missing feature only widens the field.
    INFERRED counts as a deficit: an assumption the shopper never voiced is exactly what the
    perturbations exploit.
    """
    out = []
    # INFERRED deliberately does NOT count. It did in the first campaign, and since a model
    # always marks something as inferred, a deficit always existed and A1 asked until the
    # patience budget died: deficits were non-zero on 4,654 turns against 870 zero. An
    # assumption is a reason to prefer asking, not proof that a question is worthwhile. The
    # patience economy prices repeated clarification.
    for key in ("price_cap", "option"):
        entry = table.get(key) or {}
        if str(entry.get("status", "")).upper() in ("ABSENT", "CONFLICTING"):
            out.append(f"{key} ({entry.get('status')})")
    for feat in table.get("features") or []:
        if str(feat.get("status", "")).upper() in ("ABSENT", "CONFLICTING"):
            out.append(f"feature {feat.get('value')!r} ({feat.get('status')})")
    return out


_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def _first_json(text: str) -> dict | None:
    """Parse the first JSON object out of a reply, tolerating fences and stray prose."""
    if not text:
        return None
    body = text.strip()
    if body.startswith("```"):
        body = re.sub(r"^```[a-z]*\s*|\s*```$", "", body, flags=re.IGNORECASE)
    match = _JSON_BLOCK.search(body)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None





# --------------------------------------------------------------------- A2
HYPOTHESIS_SCHEMA = """Output ONLY a JSON object:
{"features": [<string>, ...], "option": <string or null>, "price_cap": <number or null>}

Commit to a specific reading. Where the shopper was vague, indirect, or rambling, pick the
most likely value rather than leaving it blank. If a purchase of yours was rejected, you may
contradict something they appeared to say.
"""


class EntropyGateAgent(LLMAgent):
    """A2 -- Intent-Sim: sample K readings, ask about whatever they disagree on.

    Zhang & Choi, NAACL Findings 2025 (arXiv 2311.09469). Mechanism: sample K complete
    hypotheses at high temperature and cluster them per requirement. Agreement means the
    dialogue already determines that requirement, so propose; disagreement localises exactly
    what to ask about, giving when-to-ask and what-to-ask from one batch.

    Its predicted failure is the reason it earns a slot in the ladder: against a FALSIFIED
    requirement all K samples confidently copy the lie, disagreement vanishes, and it commits
    ungrounded. Only the grounded-vs-inferred axis catches that.
    """

    arm = "A2"
    guidance = ""
    k_samples = 6

    def hypotheses(self, transcript) -> list[dict]:
        base = (f"Conversation so far:\n{_flatten(transcript, self._keep_obs())}\n\n"
                f"What does this shopper want?\n{HYPOTHESIS_SCHEMA}")
        out = []
        for i in range(self.k_samples):
            # the sample index is in the prompt so the cache stores K distinct draws rather
            # than returning one memoised hypothesis K times
            raw = self._complete(f"{base}\n\n(independent reading {i + 1})", role=ROLE_SAMPLE,
                                 system="You infer shopper intent. Output JSON only.")
            obj = _first_json(raw)
            if obj:
                out.append(obj)
        return out

    @staticmethod
    def _disagreement(hyps: list[dict]) -> list[tuple[str, list]]:
        """Requirements the samples do not agree on, most contested first."""
        if len(hyps) < 2:
            return []
        contested = []
        for key in ("price_cap", "option"):
            vals = [str(h.get(key)) for h in hyps if h.get(key) is not None]
            distinct = sorted(set(vals))
            if len(distinct) > 1:
                contested.append((key, distinct))
        sets = [frozenset(str(f).lower() for f in (h.get("features") or [])) for h in hyps]
        if len(set(sets)) > 1:
            union, inter = set().union(*sets), set.intersection(*(set(s) for s in sets))
            disputed = sorted(union - inter)
            if disputed:
                contested.append(("features", disputed[:4]))
        contested.sort(key=lambda kv: -len(kv[1]))
        return contested

    def build_prompt(self, transcript) -> str:
        hyps = self.hypotheses(transcript)
        if not hyps:
            self.notes.append(f"{self.arm}:no_hypotheses")
            return super().build_prompt(transcript)
        contested = self._disagreement(hyps)
        self.notes.append(f"{self.arm}:contested={len(contested)}")
        if contested and self.must_commit(transcript):
            self.notes.append(f"{self.arm}:budget_commit")
            contested = []
        if contested:
            key, options = contested[0]
            plan = (f"Independent readings of this conversation disagree about {key}: "
                    f"{options}. Ask ONE question that makes the shopper choose between "
                    f"those (Action: Clarify, Strategy: Disambiguate).")
        else:
            agreed = hyps[0]
            plan = (f"Every reading agrees the shopper wants {json.dumps(agreed)}. "
                    f"Search for that and buy it.")
        return (f"Conversation so far:\n{_flatten(transcript, self._keep_obs())}\n"
                f"{self.budget_note(transcript)}\n\n{plan}\n\nYour single action now:")


# --------------------------------------------------------------------- A4
class PersonaPlannerAgent(LLMAgent):
    """A4 -- TRIP-style: estimate this user online, then choose the question FORMAT.

    Zhang, Huang, Deng et al., EMNLP 2024 (arXiv 2403.06769). Mechanism: track how this
    particular shopper behaves -- refusals, volunteering, hints after a rejection -- and
    condition the asking policy on it. A cooperative shopper gets one compound question that
    harvests several requirements at once; a reluctant one gets a single binary question that
    is hard to refuse, and a refused question is never repeated.

    This is the only arm whose policy is a function of the user model, so it is the only one
    that can behave differently at decline-rate 0 and decline-rate 0.5.
    """

    arm = "A4"
    guidance = ""

    def observe(self, transcript) -> dict:
        """Persona estimate from the dialogue alone -- never from the persona's dials."""
        asks = declines = answers = volunteered_hits = 0
        for turn in transcript.turns:
            if turn["role"] == "agent" and "Action: Clarify" in turn["content"]:
                asks += 1
            if turn["role"] == "user":
                text = turn["content"].lower()
                if any(p in text for p in ("rather not", "prefer not", "does not matter",
                                           "doesn't matter", "not sure", "no comment",
                                           "skip", "can't say", "cannot say")):
                    declines += 1
                else:
                    answers += 1
                    if text.count(",") >= 2 or " and " in text:
                        volunteered_hits += 1
        return {"asks": asks, "declines": declines, "answers": answers,
                "decline_rate": (declines / asks) if asks else 0.0,
                "chatty": volunteered_hits >= max(1, answers // 2)}

    def build_prompt(self, transcript) -> str:
        obs = self.observe(transcript)
        self.notes.append(f"{self.arm}:decline_rate={obs['decline_rate']:.1f}")
        if self.must_commit(transcript):
            self.notes.append(f"{self.arm}:budget_commit")
            policy = ("You have asked enough. Search for the best match for what you know "
                      "and BUY it now (Action: Answer). Do not ask anything else.")
        elif obs["decline_rate"] >= 0.3 and obs["asks"] >= 2:
            policy = ("This shopper often refuses to answer. Ask ONE narrow either/or question "
                      "that is hard to refuse (Strategy: Disambiguate, with Candidates). Never "
                      "re-ask something they already refused -- pick a different requirement, "
                      "or search and buy with what you have.")
        elif obs["chatty"]:
            policy = ("This shopper answers generously. Ask ONE compound question covering two "
                      "or three things you still do not know, so it costs a single turn.")
        else:
            policy = ("Ask ONE specific question about the single most important thing you do "
                      "not know. If you know enough, search and buy.")
        return (f"Conversation so far:\n{_flatten(transcript, self._keep_obs())}\n"
                f"{self.budget_note(transcript)}\n\n"
                f"What you have noticed about this shopper: {json.dumps(obs)}\n{policy}\n\n"
                f"Your single action now:")


class MuteBaselineFloor(DirectAgent):
    """L0 (ruling 2026-08-11) -- the floor is the SAME off-the-shelf agent as B0, with the
    same tools and the same API costs; the only difference is that the simulated user
    never speaks (runtime.mute_user, set by the runner for this arm). What the floor
    measures is: everything the agent can do alone, with zero user help."""

    arm = "L0"




# ====================================================================== GRIP-v2 Ax arms
# Three inference-time scaffolds designed from the 2026-08-11 SOTA survey (full brief in
# docs/ax-arm-designs.md). Mechanistically distinct: A1 reasons over a factored SLOT
# LEDGER with explicit EVPI-vs-cost arithmetic; A2 reasons over a concrete product
# SHORTLIST with divergence-gated, partition-splitting questions; A3 keeps a persistent
# INTENT RECORD with per-constraint verification status, built for silent drift.
# Each arm carries its belief state across turns as agent-instance state, rendered into
# every prompt, and updated from the model's structured output.

class _StatefulArm(LLMAgent):
    """Shared plumbing: a persistent STATE block the model must re-emit each turn.

    The model's reply is 'STATE: <one line of JSON>' followed by the action. The state
    is stored on the agent and injected into the next prompt; the action alone is
    returned to the episode loop. A reply without a parseable action falls back to the
    base class's retry machinery by returning the raw text unchanged.
    """

    state_label = "STATE"
    init_state: dict = {}
    _transcript = None            # set by act(); computed gates read it

    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        self.state = json.loads(json.dumps(self.init_state))   # deep copy

    def render_state(self) -> str:
        return json.dumps(self.state, ensure_ascii=False)

    def _state_spans(self, text: str) -> list[tuple[int, int, str]]:
        """Every '<LABEL>: {...}' block, brace-matched. Non-greedy regex cannot do this:
        the ledger's own JSON nests ({"slots": {...}, ...}), so `\\{.*?\\}` stops at the
        first inner brace and yields invalid JSON.
        """
        out, pos = [], 0
        pat = re.compile(rf"{re.escape(self.state_label)}\s*:\s*")
        while (m := pat.search(text, pos)) is not None:
            i = text.find("{", m.end())
            if i < 0:
                break
            depth, end = 0, -1
            for j in range(i, len(text)):
                if text[j] == "{":
                    depth += 1
                elif text[j] == "}":
                    depth -= 1
                    if depth == 0:
                        end = j + 1
                        break
            if end < 0:
                # TRUNCATED block: the reply hit max_output_tokens mid-JSON (measured on
                # B2, whose graph is verbose: 8 open braces against 7 closed). Braces never
                # balance, so returning nothing here would leak the fragment into the
                # user-visible question AND silently skip the state update. Everything from
                # the label onwards is machine text either way, so take it all.
                out.append((m.start(), len(text), text[i:]))
                break
            out.append((m.start(), end, text[i:end]))
            pos = end
        return out

    def _absorb(self, raw: str) -> str:
        """Split the model's reply into (state, action); keep state, return action.

        The state block MUST be removed from the returned action. The guidance tells the
        model to put it last and the action slice runs to the end of the reply, so leaving
        it in ships the raw JSON to the simulated shopper inside the question text.
        Measured on stored runs before this fix: 9.6% of A1-v3 asks, 9.5% of A2, 23.2% of
        A3 and 47.4% of A5 carried the block, and those asks failed the user's
        question-to-slot mapping 14-33% of the time -- each failure costing a patience
        point and returning no information.
        """
        spans = self._state_spans(raw)
        if spans:
            try:
                self.state = json.loads(spans[-1][2])
            except json.JSONDecodeError:
                self.notes.append(f"{self.arm}:state_parse_failed")
        # drop every block, last first, so earlier offsets stay valid
        for start, end, _ in reversed(spans):
            raw = raw[:start] + raw[end:]
        idx = raw.find("Action:")
        return (raw[idx:] if idx >= 0 else raw).strip()

    # Whether the arm is HELD TO its own stated rule (2026-08-12). These methods specify an
    # arithmetic -- "ask iff (1-belief) x importance >= 0.4", "ask iff an attribute splits the
    # pool", "ask iff a quotable ambiguity exists" -- but a prompt-level rule has never once
    # produced restraint here: 0.0-2.7% silent episodes across seven arms, because a model
    # always finds something to be uncertain about. Enforcing it is the same move `act` already
    # makes for the ask budget: an ask the method's own arithmetic forbids is not an available
    # behaviour. Documented as an adaptation; the rule itself is the paper's, not ours.
    enforce_own_rule = False

    def rule_forbids_ask(self) -> str | None:
        """Why this arm's OWN state says act rather than ask, or None if asking is allowed."""
        return None

    def rule_forbids_commit(self) -> str | None:
        """Why this arm's OWN state says its intent is not yet resolved enough to buy.

        Only for methods whose paper states a RESOLUTION CONDITION before recommending --
        B2's belief graph must be settled, B3's candidate pool must stop splitting. It is the
        same threshold the method already applies to asking, applied to the other branch of
        the same decision, so enforcing it adds no mechanism of ours. B1 (AT-CoT) returns None
        by design: its paper is about which question to ask and says nothing about committing,
        so giving it a purchase gate would be inventing a contribution on its behalf.

        This exists because the measured binding constraint is not asking at all: on
        exposure-fixed probes, roughly half of EVERY arm's episodes die from three or more
        rejected purchases (B0 52%, B1 48%, B2 54%, B3 52%), at ~3 rejected buys per episode
        and 2 patience points each -- six of a fourteen-point budget spent on failed checkouts.
        """
        return None

    def act(self, transcript) -> str:
        # rule hooks are called without arguments; the computed gates need the transcript
        self._transcript = transcript
        raw = super().act(transcript)
        text = self._absorb(raw)
        if not self.enforce_own_rule:
            return text
        asking = "Action: Clarify" in text
        committing = "Action: Answer" in text
        # A subclass may provide its own release condition for a commitment gate. The base
        # class deliberately does not derive one from the number of questions asked.
        if committing and self.must_commit(transcript):
            return text
        reason = (self.rule_forbids_ask() if asking
                  else self.rule_forbids_commit() if committing else None)
        if reason is None:
            return text
        kind = "A QUESTION" if asking else "A PURCHASE"
        alt = ("Take an environment action instead: search, or buy the best candidate."
               if asking else
               "Resolve it first: search for a better candidate, or ask the one question "
               "your own rule says is worth asking.")
        self.notes.append(f"{self.arm}:rule_blocked_{'ask' if asking else 'commit'}")
        text = self._absorb(self._complete(
            self.build_prompt(transcript)
            + f"\n\nYOUR OWN RULE FORBIDS {kind} HERE: {reason}\n{alt}"))
        still = ("Action: Clarify" in text) if asking else ("Action: Answer" in text)
        if still:
            # it insisted; the arm's declared rule wins. The override has to be a VALID
            # environment action or the rule costs the arm a turn AND a parse error -- see
            # _harness_action. Measured in the published airline R3 campaign before this
            # fix: 2,999 injected `search[...]` calls, concentrated in the rule-gated
            # baselines (B2v2 up to 814 in one cell, B1v3 up to 670), none of which are
            # tau2 actions. Where the domain offers no safe action the rule cannot be
            # enforced by substitution, so the arm's own reply stands and the event is
            # recorded under a distinct note rather than silently rewritten.
            fb = self._harness_action(transcript)
            if fb is not None:
                self.notes.append(f"{self.arm}:rule_override_search")
                return fb
            self.notes.append(f"{self.arm}:rule_override_unavailable")
            return text
        return text


class SlotLedgerAgent(_StatefulArm):
    """A1 -- slot-ledger EVPI gater: ask only where the arithmetic says so.

    Ingredients (see docs/ax-arm-designs.md): OPEN-style featurization into a slot table
    (arXiv:2403.05534); per-question value-of-information vs cost arithmetic written
    into the prompt (SAGE-style EVPI gate, Calibrate-Then-Act cost injection); one
    reserved confirm-before-buy (tau-bench). Contradictions reset slots to UNK rather
    than averaging -- the arm's intent-shift reflex.
    """

    arm = "A1"
    state_label = "LEDGER"
    init_state = {"slots": {}, "confirmed": False}
    guidance = """
KEEP A MEMORY LEDGER -- an append-only dictionary of what the user has ACTUALLY said.
Whenever the user states or confirms a requirement, add it as a key: value pair. Record
only what was said: do not invent a schema up front, do not pre-create entries for
things you have not heard, and do not mark anything as missing -- if it is not in the
ledger, the user simply has not said it.

EVERY turn, before acting:
1. UPDATE the ledger from anything new the user said. If a reply CONTRADICTS a stored
   value, replace it and note that it changed -- the user's intent may have moved; do
   not trust other values you elicited before the contradiction.
2. ASK ONLY ABOUT THE USER'S OWN WORDS: ask when what the user said is unclear,
   contradictory, or silent about something you must know to act. A poor search result
   is NEVER a reason to ask -- the user cannot see the store, does not know what is in
   stock, and cannot fix your search; questions about availability, sizes on offer, or
   substitutes are wasted. Fix the search yourself, or buy the closest candidate.
3. CONFIRM ONLY WHEN WARRANTED: before buying, confirm ONLY if a value CHANGED during
   this conversation (a contradiction or correction happened) -- and then confirm just
   THAT ONE thing in plain words ("Just to be sure -- gray, right?"), never a restated
   list of everything. In every other case, buy directly.


QUESTION SKILL (general): a question is whatever best reduces your uncertainty, in plain
shopper language. Both forms are legitimate: a TARGETED question about one thing you
need to know, or an AUDIT question that lays out your current understanding and asks
what is wrong or missing -- "So far I have: twin size, wood frame, grey, under $500.
Is anything off, or do you need something I haven't got?"
Never mention your internal process -- no "candidates", "shortlist", or
reasoning talk; the user sees none of that. Before asking anything, check your state:
if the user already answered or refused it, in any wording, DO NOT ask again -- and
treat "either works" / "I'm not picky" as ANSWERED: that requirement is resolved, and
asking finer detail about it wastes a question.

BUY DISCIPLINE (the data says this is where episodes die): whether the shopper explains
a rejection depends on their mood -- some lay out exactly what was wrong, some barely
react. Buying-to-test is a gamble, not a strategy. Verify your candidate line-by-line
against everything you know, copying option values EXACTLY as the listing spells them.
But listings are messy: the right product is often worded differently from the user
(a listing may say 8'6" x 12' where the user said 8 ft 6 in x 12 ft) -- judge whether
the product IS what they want, not whether the words match. When no candidate satisfies
everything, BUY THE CLOSEST ONE anyway: a near match earns partial credit, a refusal or
an apology earns exactly zero, always. After a rejection, READ the shopper's reaction
word by word and put every requirement it names into the ledger: your understanding is
wrong somewhere, so change PRODUCT (not just options), and re-check first.
After any correction from the user, re-search immediately.

ALWAYS search the catalogue on your very first turn, before any question -- real
products anchor which slots matter.

End EVERY reply with the ledger on its own line:
LEDGER: {"slots": {...}, "confirmed": true/false}
"""

    def build_prompt(self, transcript) -> str:
        base = super().build_prompt(transcript)
        return (f"Your slot ledger from last turn:\nLEDGER: {self.render_state()}\n\n"
                f"{base}")


class ShortlistBisectorAgent(_StatefulArm):
    """A2 -- shortlist bisector: reason in product space, not question space.

    Ingredients: EIG-uniform question scoring over concrete candidate solutions (Active
    Task Disambiguation, arXiv:2502.04485); ClarifyGPT-style divergence gate -- ask only
    when independently sampled picks disagree (arXiv:2310.10996); search-first ordering
    (UserBench); never ask a question every candidate answers identically.
    """

    arm = "A2"
    state_label = "SHORTLIST"
    init_state = {"candidates": [], "gate": "pending"}
    guidance = """
REASON OVER A SHORTLIST OF REAL PRODUCTS, never over abstractions.

1. SEARCH FIRST. Build and maintain a shortlist of the 5-10 concrete products (ASIN +
   the option combo you would pick) compatible with everything the user has said.
2. DIVERGENCE GATE WITH AN ADVERSARIAL READING: privately imagine three independent
   shoppers choosing from your shortlist. Agreement alone is NOT enough -- the request
   always contains exactly one fault, and three readers of the same corrupted text
   agree on the same wrong answer. So before trusting agreement, construct the most
   plausible ALTERNATIVE reading of the request (what would it mean if one detail were
   hidden, wrong, vague, or extra?). If the alternative reading changes which product
   to buy, that detail is where the fault is -- ask ONE question about it. Only when
   the shoppers agree AND the alternative reading changes nothing, buy immediately.
3. WHAT TO ASK: consider 3-5 candidate questions. For each, work out how the shortlist
   SPLITS by possible answers. Ask the question with the most even split (it removes
   the most candidates whatever the answer). NEVER ask anything all candidates answer
   identically -- provably worthless; search again or buy instead.
4. AFTER EVERY ANSWER: drop incompatible candidates, then re-search with the enriched
   query to refill the shortlist with real products.
5. If an answer eliminates EVERYTHING, the intent has likely changed: discard the
   shortlist, rebuild from the user's last two utterances only, and re-open questioning.
6. COMMIT when one candidate remains, or the best question would not change your pick,
   or the budget is gone: buy the most-supported candidate.


QUESTION SKILL (general): a question is whatever best reduces your uncertainty, in plain
shopper language. Both forms are legitimate: a TARGETED question about one thing you
need to know, or an AUDIT question that lays out your current understanding and asks
what is wrong or missing -- "So far I have: twin size, wood frame, grey, under $500.
Is anything off, or do you need something I haven't got?"
Never mention your internal process -- no "candidates", "shortlist", or
reasoning talk; the user sees none of that. Before asking anything, check your state:
if the user already answered or refused it, in any wording, DO NOT ask again -- and
treat "either works" / "I'm not picky" as ANSWERED: that requirement is resolved, and
asking finer detail about it wastes a question.

BUY DISCIPLINE (the data says this is where episodes die): whether the shopper explains
a rejection depends on their mood -- some lay out exactly what was wrong, some barely
react. Buying-to-test is a gamble, not a strategy. Verify your candidate line-by-line
against everything you know, copying option values EXACTLY as the listing spells them.
But listings are messy: the right product is often worded differently from the user
(a listing may say 8'6" x 12' where the user said 8 ft 6 in x 12 ft) -- judge whether
the product IS what they want, not whether the words match. When no candidate satisfies
everything, BUY THE CLOSEST ONE anyway: a near match earns partial credit, a refusal or
an apology earns exactly zero, always. After a rejection, READ the shopper's reaction
word by word and put every requirement it names into the ledger: your understanding is
wrong somewhere, so change PRODUCT (not just options), and re-check first.
After any correction from the user, re-search immediately.

Work out the shortlist split PRIVATELY. The question you ask is about the product
attribute itself ("Do you want black or brown?"), never about your analysis.

End EVERY reply with your shortlist on its own line:
SHORTLIST: {"candidates": ["ASIN opts", ...], "gate": "agree"/"split"/"pending"}
"""

    def build_prompt(self, transcript) -> str:
        base = super().build_prompt(transcript)
        return (f"Your shortlist from last turn:\nSHORTLIST: {self.render_state()}\n\n"
                f"{base}")


class IntentSentinelAgent(_StatefulArm):
    """A3 -- intent-record sentinel: assume the spec is wrong until re-verified.

    Ingredients: persistent per-constraint status record diffed against every utterance
    (ChronosBench-style monitoring); hypothesis-pool consistency refit (BED-LLM,
    arXiv:2508.21184); entropy-triggered asking with GATE's open-early / confirm-late
    format schedule (arXiv:2310.11589); drift alarm that re-opens elicitation; final
    intent summary before buying (STaR-GATE/TO-GATE).
    """

    arm = "A3"
    state_label = "RECORD"
    init_state = {"constraints": {}, "phase": "early", "summarized": False}
    guidance = """
MAINTAIN AN INTENT RECORD: every requirement with status STATED (heard, untrusted),
CONFIRMED (verified with the user), or STALE (contradicted or suspect). The opening
request contains EXACTLY ONE fault, and the user's intent can CHANGE mid-conversation
without warning -- treat the record as wrong until re-verified.

1. SEARCH FIRST, always, on your very first turn -- real products anchor the record.
2. EARLY (your first question): ask ONE open-ended goal question -- "is there anything
   else that matters about the <category> you're after?" Hidden requirements only
   surface when the user is invited to volunteer. Mark volunteered facts CONFIRMED.
3. EVERY user reply: DIFF it against the record.
   - A CONTRADICTION is the user STATING a value that conflicts with a stored one
     ("under $40" when you have $80, "actually, blue"). Only that flips constraints to
     STALE and raises a drift alarm.
   - A DEFLECTION ("let's not get hung up on that", a non-answer, changing the subject)
     is NOT a contradiction: mark that constraint "flexible" and NEVER ask about it
     again -- act on your best guess for it.
   - NEVER repeat a question, even reworded, even after a drift alarm. A question that
     failed once will fail again and enrages the user; the constraint simply stays
     uncertain.
3. LATER questions: privately imagine 5 plausible complete intents consistent with the
   record. If they would lead to different purchases, ask an option-posing confirmation
   about the constraint they disagree on most ("under $40, or is price flexible?").
   If they agree, stop asking.
5. BEFORE BUYING (once): restate the complete intent in one sentence as a statement,
   not a question ("Getting you: X, blue, size M, under $40."). If the user objects,
   that is a drift alarm (rule 2). Then buy the product matching the record.


QUESTION SKILL (general): a question is whatever best reduces your uncertainty, in plain
shopper language. Both forms are legitimate: a TARGETED question about one thing you
need to know, or an AUDIT question that lays out your current understanding and asks
what is wrong or missing -- "So far I have: twin size, wood frame, grey, under $500.
Is anything off, or do you need something I haven't got?"
Never mention your internal process -- no "candidates", "shortlist", or
reasoning talk; the user sees none of that. Before asking anything, check your state:
if the user already answered or refused it, in any wording, DO NOT ask again -- and
treat "either works" / "I'm not picky" as ANSWERED: that requirement is resolved, and
asking finer detail about it wastes a question.

BUY DISCIPLINE (the data says this is where episodes die): whether the shopper explains
a rejection depends on their mood -- some lay out exactly what was wrong, some barely
react. Buying-to-test is a gamble, not a strategy. Verify your candidate line-by-line
against everything you know, copying option values EXACTLY as the listing spells them.
But listings are messy: the right product is often worded differently from the user
(a listing may say 8'6" x 12' where the user said 8 ft 6 in x 12 ft) -- judge whether
the product IS what they want, not whether the words match. When no candidate satisfies
everything, BUY THE CLOSEST ONE anyway: a near match earns partial credit, a refusal or
an apology earns exactly zero, always. After a rejection, READ the shopper's reaction
word by word and put every requirement it names into the ledger: your understanding is
wrong somewhere, so change PRODUCT (not just options), and re-check first.
After any correction from the user, re-search immediately.

End EVERY reply with the record on its own line:
RECORD: {"constraints": {"<name>": {"value": "...", "status": "STATED/CONFIRMED/STALE"}}, "phase": "early/late", "summarized": true/false}
"""

    def build_prompt(self, transcript) -> str:
        base = super().build_prompt(transcript)
        return (f"Your intent record from last turn:\nRECORD: {self.render_state()}\n\n"
                f"{base}")



class FaultDispatchAgent(_StatefulArm):
    """A4 -- diagnose the fault type first, then run its playbook.

    The benchmark's premise is public: every opening request carries exactly one fault
    from a known catalogue (requirements hidden, unresolved, wrong, or extraneous). The
    per-episode label is hidden; classifying the fault from surface evidence is agent
    skill, and each diagnosis has a distinct optimal response -- most faults need ONE
    well-chosen question or none at all.
    """

    arm = "A4"
    state_label = "DIAGNOSIS"
    init_state = {"suspected_fault": "unknown", "evidence": "", "handled": False}
    guidance = """
DIAGNOSE BEFORE ANYTHING. The request contains EXACTLY ONE flaw. On your first turn,
classify it from the text itself and record your diagnosis; then follow that playbook:

- SUBJECTIVE WORD in place of a value ("reasonable price", "good quality"):
  ask the exact value of that one thing. One question, done.
- POINTER without a referent ("that one", "the usual size"):
  ask what it refers to. One question, done.
- TWO-MEANING word or grammar (could be read two ways):
  offer both readings as one A/B question ("light as in color, or in weight?").
- SOMETHING CLEARLY MISSING (no budget, no size where one obviously matters):
  one open invitation: "any other requirement I should know -- budget, size?"
- PREMISE OF UNAVAILABILITY ("since you don't have X, I'll take Y"):
  do not trust it. Search X first; if X exists, say so and confirm which they want.
- A STATED DETAIL FINDS NOTHING (search with it returns poor matches):
  the detail is probably wrong; ask "did you mean <closest real alternatives>?"
- AN ODD EXTRA DETAIL that nothing in the catalogue has (a bundled extra, a niche
  add-on): ignore it silently; satisfy the core request; never ask about it.
- LONG OFF-TOPIC PREAMBLE: strip it; act on the request sentence only; ask nothing.
- NO REQUEST STATED (the user only describes or reports): treat the described
  requirements as the full spec and execute; do not ask what they want.

If the playbook resolved the fault (or none applies), STOP ASKING and execute: search
with corrected values, verify the candidate against every requirement line-by-line
(option values copied exactly), then buy. A rejected purchase costs double and comes
with no explanation -- never buy to probe. If a purchase is rejected anyway or the
user contradicts something, re-diagnose from scratch: the intent may have changed.

End EVERY reply with your diagnosis on its own line:
DIAGNOSIS: {"suspected_fault": "...", "evidence": "...", "handled": true/false}
"""

    def build_prompt(self, transcript) -> str:
        base = super().build_prompt(transcript)
        return (f"Your diagnosis from last turn:\nDIAGNOSIS: {self.render_state()}\n\n"
                f"{base}")


# ------------------------------------------------------------------ published baselines
# A5/A6 are faithful ports of PUBLISHED inference-time clarification methods, added so the
# baseline table cites other people's methods rather than only our own (docs/related-
# clarification-methods.md). Two deliberate design rules, both load-bearing:
#
# 1. NO BUY DISCIPLINE. Verified purchasing is this paper's own contribution, and the
#    experiment is a factorial: published ASKING mechanism x our CHECKOUT discipline. An
#    arm that silently inherited the checkout rule would attribute our gain to their paper
#    (compare A2 vs A2_v3, already run).
# 2. NO BENCHMARK KNOWLEDGE. Neither arm is told that the request carries exactly one fault,
#    nor the names of our fault catalogue -- A4 is told both, which is why A4 stays OUR arm
#    and is not presentable as AT-CoT. A5's taxonomy is a generic query-ambiguity typology.
#
# Both keep the shared QUESTION DISCIPLINE block: it removes harness pathologies (meta
# language, re-asking, compound questions) that are unrelated to the mechanism under test,
# and every Ax arm carries it, so it does not advantage one arm over another.


# ---------------------------------------------------------------------------- fairness patch
# A REQUIREMENTS LEDGER FOR THE PUBLISHED BASELINES (ruling 2026-08-14: "try to migrate some
# skills from A1 and A2" to a B arm that keeps failing). Motivated by measurement, not by
# convenience. It applies to the three arms that genuinely fall below our asking floor -- B1
# (+0.40 vs B0 where the floor gets +4.40 on the same episodes), B2 (-3.66) and B4 (-1.22) --
# and NOT to B3 (+4.08) or B5 (+4.08), which match the floor as published. Those three lose on
# CONVERSION while matching the floor on information: accepted-per-proposal A0 17.8%, B1 15.8%,
# B1v2 13.9%, B2 13.3% -- and Success tracked that column, not the information ones. The cause
# is not prompt weight: our HEAVIEST arm converts best (A3, 9,636-char system prompt, 30.6%
# accepted vs A0's 21.2% on the same 160 episodes). It is what the state HOLDS. Each published
# method's state represents UNCERTAINTY about the intent -- an ambiguity type, a belief graph
# regenerated each turn, a posterior, a candidate pool -- because each was designed to choose a
# QUESTION, and in the single-shot settings they were published in the question IS the output.
# Here the arm must also buy, and at purchase time none of them holds a faithful record of what
# the shopper actually said. B2, richest in uncertainty machinery, converts worst of all: below
# the stateless floor.
#
# So the ledger is given to the baselines as an ENVIRONMENT-REQUIRED capability, exactly as the
# cost model is given to B6 because its paper assumes one. Each method's own ask-decision rule,
# thresholds and state block are untouched; the ledger only records stated values and is read
# before buying. Documented as an adaptation in favour of the baselines.
#
# ABLATION VERDICT (2026-08-15): THE HYPOTHESIS ABOVE IS WRONG, and the ledger is not what fixes
# these arms. Measured on B2, where the two interventions could be separated cleanly:
#     B2 published        Success 24.00   net vs B0 -3.60
#     B2v2 breadth only    Success 34.80   net +7.20  (p=0.044)
#     B2v3 breadth+ledger  Success 32.78   net +4.15
# Adding the ledger on top of breadth does not help and mildly hurts (-2.0 Success, asks 1.78 ->
# 1.59): it competes for the same reply and buys nothing. What the conversion column was actually
# measuring was the DOWNSTREAM shadow of narrow questions -- an arm that asks about one dimension
# buys against the ones it never asked about, which shows up as a low accepted-per-proposal rate
# even though the immediate cause is coverage, not recall. QUESTION BREADTH is the whole effect.
# Kept in the graph because B1v3 and the B3v2/B4v2 variants were probed with it and their numbers
# must remain reproducible; new work should prefer breadth alone.
_REQUIREMENTS_LEDGER = """
KEEP A LEDGER OF WHAT THE SHOPPER HAS ACTUALLY SAID, alongside your own state above. It is a
record, not a judgement: their words and values, never a guess of yours, and never a checklist
of what a request "should" contain.
  - add an entry the moment they state or confirm something
  - if they later say something that CONTRADICTS an entry, REPLACE it and note the key as
    changed: the newest statement is the truth, and any choice you made on the old value must
    be re-checked before you rely on it again
  - a rejection is information you did not pay a question for: put every requirement its
    wording names into the ledger before you buy again

READ THE LEDGER BEFORE EVERY PURCHASE, and check your candidate against every entry in it,
copying option values EXACTLY as the listing spells them. Your own state above tells you what
to ASK; the ledger tells you what to BUY. They are different jobs and you need both -- an
uncertainty you have already resolved is not a requirement you can still recall.

The ledger lives INSIDE the state line you already end every reply with: carry two more keys
in that same JSON object, alongside your method's own fields --
  "known":   {"<what they said>": "<value>"}
  "changed": ["<key they later contradicted>"]
"""


class AmbiguityTypedAgent(_StatefulArm):
    """A5 -- AT-CoT: classify the ambiguity TYPE first, then ask what that type calls for.

    Zhang et al., "Clarifying Ambiguities: on the Role of Ambiguity Types in Prompting
    Methods for Clarification Generation", SIGIR 2025 (arXiv:2504.12113). One LLM call per
    turn, no belief state, no information-theoretic gate: the published mechanism is typed
    chain-of-thought -- reason about WHICH KIND of ambiguity is present, and let the type
    select the question form.

    Our documented adaptations: (a) the paper generates a clarification unconditionally in a
    single-shot setting, so the ask-vs-act decision (type NONE => act) is ours; (b) the
    taxonomy is instantiated for product search. Nothing else is added.
    """

    arm = "A5"
    state_label = "AMBIGUITY"
    init_state = {"type": "unclassified", "evidence": "", "settled": []}
    guidance = """
CLASSIFY BEFORE YOU ASK. Reason in two stages every turn, over the whole conversation.

STAGE 1 -- TYPE, AND IT MUST BE EVIDENCED. Name the ONE type that fits, and QUOTE the exact
words from the request that create it. If you cannot quote specific words, the type is NONE.
  POLYSEMY          a word in the request carries two plausible readings.
  SCOPE             a product area is named, but not which member of it is wanted.
  MISSING           a detail a purchase requires is simply absent (no budget, no size).
  VAGUE             a judgement word stands where a concrete value belongs.
  REFERENCE         a pointer is used with nothing to point at.
  PREMISE           the request asserts something about what exists or is available that
                    the catalogue may not support.
  NONE              specific enough to act on -- no quotable words are at fault.

The quote is the test, not a formality. "I want a light jacket under $40" has POLYSEMY on
the quotable word "light". "I want a black cotton t-shirt, size M, under $20" has nothing to
quote: every requirement is stated, so the type is NONE and you buy. Many requests are NONE.
A type you cannot evidence with the shopper's own words is a type you invented, and asking
about an invented type wastes the shopper's goodwill for nothing.

STAGE 2 -- TYPED QUESTION. Ask only what the type calls for, and only one thing:
  POLYSEMY   -> offer the two readings as a single either/or question.
  SCOPE      -> ask which kind within that area they mean.
  MISSING    -> ask for that one absent detail.
  VAGUE      -> ask what the judgement word means as a concrete value or number.
  REFERENCE  -> ask what the pointer refers to.
  PREMISE    -> check the premise against the catalogue by searching FIRST; ask only if
                what you find contradicts it.
  NONE       -> ask nothing: search and buy.

Re-run STAGE 1 from scratch each turn: a reply can resolve one type and expose another,
and what the user wants may itself have moved. Record every type you have asked about in
"settled" and never re-open it -- and note what your own evidence rule implies: once the
one quotable fault in a request has been settled, there is nothing left to evidence, so
the type is NONE and you act. Spend what you saved at the shelf: search with the words
the shopper gave you, buy your best candidate, and read any rejection carefully -- a
rejection often quotes the very words that would have been your next type.

QUESTION DISCIPLINE (non-negotiable): each question is ONE short sentence asking exactly
ONE thing, in plain shopper language. Never mention your internal process -- no "types",
"classification", "candidates", or reasoning talk; the user sees none of that. Before
asking anything, check your state: if the user already answered or refused it, in any
wording, DO NOT ask again -- move on or act.

End EVERY reply with your classification on its own line:
AMBIGUITY: {"type": "...", "evidence": "<the words that made you pick it>", "settled": [...]}
"""

    # AT-CoT's own rule, HELD TO (the suite's standing adaptation for stated rules;
    # measured on the probe: 86% of episodes rode the ask cap at 2 typed questions, and
    # the arm had the worst recovered-to-win conversion in the suite at 21%). The paper
    # says a type must be EVIDENCED by quotable words and that a settled type is never
    # re-opened. Once a type has been asked about, it is settled -- and since these
    # requests carry exactly one fault, a second typed question is by the method's own
    # logic a type it invented.
    enforce_own_rule = True

    def rule_forbids_ask(self) -> str | None:
        st = self.state or {}
        settled = st.get("settled") or []
        typ = str(st.get("type", "")).strip().lower()
        if typ in ("none", "unclassified", ""):
            if settled:
                return ("your own classification is NONE and you have already settled "
                        f"{settled}: the method says act")
            return None
        if settled:
            return (f"you have already settled {settled}; a further type on a request "
                    f"that carries one fault is a type you invented, which your own "
                    f"evidence rule forbids")
        return None

    def build_prompt(self, transcript) -> str:
        base = super().build_prompt(transcript)
        return (f"Your classification from last turn:\n"
                f"AMBIGUITY: {self.render_state()}\n\n{base}")


class AmbiguityTypedAgentV2(AmbiguityTypedAgent):
    """B1 revision -- type the ambiguity over RETRIEVED RESULTS, not over the bare query.

    Measured cause of B1's gate failure (248 paired probe episodes): B1 matched our asking
    floor on every information metric -- Earned 15.97 vs 16.33, Recovery 45.83 vs 46.26, Aim
    48.26 vs 46.92 -- and converted none of it, Success 28.23 vs 31.85. The difference is WHEN
    it asks: B1's median first question lands on **turn 0** against A0's turn 4, i.e. it types
    the shopper's sentence and asks before it has ever looked at the shelf. A blind question is
    answered truthfully and still fails to help, because the answer can name something the
    catalogue does not stock and can miss the attribute that actually separates the candidates
    -- which is exactly the signature observed (high Aim, flat Success, 21 turn-cap deaths
    against A0's 12).

    Grounding the typing in retrieved results is FAITHFUL to the source rather than a favour:
    AT-CoT is a SIGIR method whose setting is retrieval, so the ambiguity that matters is the
    one the retrieved list cannot settle. Enforced in code because prompt-level timing rules
    have never once held in this harness.
    """

    arm = "B1v2"
    guidance = AmbiguityTypedAgent.guidance.replace(
        "CLASSIFY BEFORE YOU ASK. Reason in two stages every turn, over the whole conversation.",
        """LOOK BEFORE YOU TYPE. Search the shop FIRST -- searching is free and the shopper never
minds it -- and classify over WHAT CAME BACK, not over their sentence alone. A type worth a
question is one the retrieved list cannot settle: if every candidate on the shelf already
agrees on a detail, that detail is not in doubt however vague their wording was, and if the
list splits on a detail they never mentioned, that split is the ambiguity that matters.

CLASSIFY BEFORE YOU ASK. Reason in two stages every turn, over the whole conversation.""")

    def rule_forbids_ask(self) -> str | None:
        base = super().rule_forbids_ask()
        if base is not None:
            return base
        tr = getattr(self, "_transcript", None)
        if tr is None:
            return None
        looked = any("Action: Operation" in (t.get("content") or "")
                     for t in getattr(tr, "turns", []) if t.get("role") == "agent")
        if not looked:
            return ("you have not looked at the shelf yet -- your own method types ambiguity "
                    "over RETRIEVED results, so the type is not yet evidenced")
        return None


class BeliefGraphAgent(_StatefulArm):
    """A6 -- belief graph over intent components, asking by uncertainty x importance.

    Hahn et al., "Proactive Agents for Multi-Turn Text-to-Image Generation Under
    Uncertainty", ICML 2025 (arXiv:2412.06771). The published mechanism: maintain an
    editable graph of intent components, each carrying a belief (how confident) AND an
    importance (how much it matters to the outcome); regenerate it from the full history
    each turn; ask about the component maximising uncertainty x importance. We use their
    single-call variant, in which one pass over the serialised graph decides ask-vs-act and
    produces the question -- so the call profile matches B0/A1 (~1 call/turn).

    The importance weight is what distinguishes this from A1 (value+confidence, EVPI-vs-cost
    gate) and A3 (per-constraint status, pool entropy): A3's pure-entropy trigger asks about
    whatever is least certain, which need not matter; this asks about what matters AND is
    uncertain. No cost arithmetic and no checkout rule are added -- neither is in the paper.
    """

    arm = "A6"
    state_label = "GRAPH"
    init_state = {"nodes": {}, "gap": 0.0, "decision": "act"}
    guidance = """
MAINTAIN A BELIEF GRAPH OF WHAT THE SHOPPER WANTS.

Each node is one component of their intent -- a feature, the option, the price cap, the
kind of product. Every node carries three things, and TWO OF THEM ARE READ OFF EVIDENCE
rather than felt:

  value       what you currently think it is, or "?" if unknown

  belief      set by WHERE the value came from:
                1.0  the shopper said it in their own words
                0.6  you inferred it from the catalogue or from context
                0.2  you are guessing

  importance  set by THE PRODUCTS IN FRONT OF YOU:
                1.0  the candidates in your latest search results DISAGREE on this, so the
                     answer would change which one you buy
                0.5  it might change which one you buy
                0.0  every candidate you are considering already satisfies it, or it cannot
                     change your choice at all
              WITH ONE FLOOR THAT OVERRIDES THE PRODUCTS: any node whose value you are
              GUESSING (belief 0.2) never drops below importance 0.6, however uniform the
              catalogue looks. A value the shopper never gave you is a value you could have
              wrong, and the products cannot tell you otherwise -- your search was built
              from your own guess, so of course the results agree with it. Catalogue
              agreement only means something for values the shopper actually stated.

EVERY turn, regenerate the whole graph from the entire conversation -- do not merely patch
last turn's copy. Anything the user has newly said may add a node, change a value, lower a
belief, or change what matters. If a reply conflicts with a node, trust the reply and drop
the belief in that node to near zero rather than averaging the two.

A REFUSED PURCHASE IS A CONTRADICTION, AND THE LOUDEST ONE YOU GET. You bought what your
graph said was settled and the shopper said no, so the graph was wrong -- about something,
and you are not told what. Do not re-buy around the edges. Drop every node you were
GUESSING or had merely INFERRED back to belief 0.2, keep only what the shopper stated in
their own words at belief 1.0, and rebuild from there. The graph is now unsettled by
construction, which is the correct state: it means ask or search, not buy again.

THEN DECIDE, and show the number:
  1. For every node compute  gap = (1 - belief) x importance.
  2. Take the largest gap over all nodes.
  3. If that largest gap is 0.4 or more, ASK about that one node and nothing else.
  4. If it is under 0.4, ACT: search with your current values and buy the best candidate.

Read the arithmetic honestly, because it is what decides:
  - a node you are guessing (0.2) that separates your candidates (1.0) scores 0.8 -- ask.
  - a node you are guessing (0.2) that the catalogue seems unanimous about still scores
    0.8 x 0.6 = 0.48 through the floor -- ask. This is the case that matters most: the
    requirement the shopper never mentioned is the one most likely to be the one you get
    wrong, and a search built from your own guess cannot reveal it.
  - a node the shopper stated themselves (1.0) scores 0.0 whatever its importance -- never
    ask; you already have it from them.
  - a node you INFERRED from the catalogue (0.6) that no candidate disagrees about scores
    0.0 -- asking cannot change what you buy, so acting is strictly better.
  - if the graph has no node scoring 0.4 or more, the right move is to BUY, not to look for
    something else to ask.

SEARCH BEFORE YOUR FIRST QUESTION, so that importance rests on real products rather than on
a guess.

BUT TRUST THE FIRST SEARCH ONLY SO FAR. That search was built from the shopper's opening
words, and those words are what you are unsure about -- so the pool it returned may all
share an attribute simply because your query asked for it. Therefore:
  - While you have asked NOTHING yet, rank nodes by (1 - belief) alone: ask about what the
    SHOPPER never actually told you, not about what the current candidates happen to differ
    on. A node at belief 0.2 or below is a node they never stated.
  - After their first answer, the pool is grounded in something they confirmed, and
    importance from candidate disagreement becomes the better guide: use the full
    gap = (1 - belief) x importance from then on.
A component the shopper left unstated is the component most likely to be the one that
matters, whatever the catalogue happens to stock.

QUESTION DISCIPLINE (non-negotiable): each question is ONE short sentence asking exactly
ONE thing, in plain shopper language. Never mention your internal process -- no "nodes",
"graph", "belief", "importance", or reasoning talk; the user sees none of that. Before
asking anything, check your state: if the user already answered or refused it, in any
wording, DO NOT ask again -- move to the next node or act.

End EVERY reply with the graph on its own line, and keep it COMPACT -- at most 8 nodes,
short names, values of a few words. A graph that runs long gets cut off mid-way and is
lost entirely. Include the largest gap and the decision it produced:
GRAPH: {"nodes": {"<name>": {"value": "...", "belief": 0.0, "importance": 0.0}, ...},
        "gap": 0.0, "decision": "ask"|"act"}
"""

    enforce_own_rule = True

    def _gap(self) -> float | None:
        try:
            return float(self.state.get("gap"))
        except (TypeError, ValueError):
            return None

    def rule_forbids_ask(self) -> str | None:
        g = self._gap()
        if g is not None and g < 0.4:
            return (f"your graph's largest gap is {g:.2f}, below the 0.4 threshold, so the "
                    f"graph is settled and the rule says act")
        # IMPORTANCE IS A PROPERTY OF THE CANDIDATES, not of the claim. The paper defines
        # importance as "would the answer change which item is chosen"; when no unstated
        # component still separates the surviving candidates, that product is zero however
        # high the self-reported belief gap.
        alive, best, who = unstated_split_remains(self._transcript)
        if not alive:
            return (f"no unstated component still separates your candidates (best "
                    f"{best:.2f}{' on ' + who if who else ''}), so uncertainty x importance "
                    f"is ~0 and the graph is settled")
        return None

    def rule_forbids_commit(self) -> str | None:
        """The same 0.4 threshold, applied to the other branch of the same decision.

        The method's rule is "ask iff the largest (1-belief) x importance reaches 0.4".
        A graph in that state is by its own definition NOT settled, so buying from it is the
        move the rule exists to prevent. Nothing of ours is added: no requirement checking, no
        listing verification -- only the paper's own resolution condition, enforced.
        """
        g = self._gap()
        if g is not None and g >= 0.4:
            return (f"your graph's largest gap is {g:.2f}, at or above the 0.4 threshold, so "
                    f"the intent is not resolved yet")
        return None

    def build_prompt(self, transcript) -> str:
        base = super().build_prompt(transcript)
        # importance is COMPUTED from the store's own reply rather than estimated: a model
        # asked to judge "does this matter" always says yes about something, which is why
        # this method's ask-vs-act rule never produced a single silent episode.
        report = candidate_report(last_search_observation(transcript))
        return (f"Your belief graph from last turn:\nGRAPH: {self.render_state()}\n"
                f"{report}\n\n{base}")


class AmbiguityTypedAgentV3(AmbiguityTypedAgentV2):
    """B1 + question BREADTH + the requirements ledger. v2's retrieval-grounded typing is kept.

    Breadth is added because it is what rescued B2: batching the nodes B2's own threshold had
    already flagged took it from 23.98 to **35.25** Success (+6.97 over B0, p=0.057, a PASS)
    without touching a single number in its rule. B1 carries the identical bottleneck one level
    up -- Stage 2 says "ask only what the type calls for, and only one thing", so one question
    buys one detail even when the SAME type is present in three places (three absent details is
    still one MISSING type, three vague words still one VAGUE type). Naming every instance of
    the type it already evidenced changes no classification, no threshold and no taxonomy; it
    changes how many instances fit in a turn, which the paper never speaks to because its
    setting emits one question and stops. This environment charges per QUESTION.
    """

    arm = "B1v3"
    init_state = {"type": "unclassified", "evidence": "", "settled": [],
                  "known": {}, "changed": []}
    guidance = AmbiguityTypedAgentV2.guidance.replace(
        "STAGE 2 -- TYPED QUESTION. Ask only what the type calls for, and only one thing:",
        """STAGE 2 -- TYPED QUESTION. Ask what the type calls for -- and if that ONE type is
present in SEVERAL places, name them ALL in the single question, because they are instances of
the same evidenced type rather than types you invented. Three details absent from the request
is one MISSING type with three instances; two judgement words is one VAGUE type with two. The
shopper's goodwill is charged per QUESTION, so a question that settles three instances costs
exactly what a question that settles one costs, and leaving two unasked means buying on two
guesses. Ask about instances of your evidenced type only -- never a second type:""").replace(
        "QUESTION DISCIPLINE (non-negotiable): each question is ONE short sentence asking exactly\nONE thing, in plain shopper language.",
        "QUESTION DISCIPLINE (non-negotiable): one short, plain-language question, covering every\ninstance of your evidenced type and nothing beyond it."
    ) + _REQUIREMENTS_LEDGER


class BeliefGraphAgentV2(BeliefGraphAgent):
    """B2 revision -- ask about EVERY node the method's own threshold flags, in one question.

    Measured cause of B2's gate failure (246 paired probe episodes): it asks about as often as
    our floor and earns half as much -- asks 1.55 vs A0's 1.62, Earned **8.05 vs 16.44**,
    Recovery 34.23 vs 47.26, Success 23.98 vs B0's 27.64. Timing is not the problem (median
    first ask on turn 3, same as A0). The per-question YIELD is, because step 3 of the published
    decision rule sends it to ask about the single largest-gap node "and nothing else", so one
    question buys one dimension.

    The revision changes no arithmetic. B2's own gap rule marks a node as worth asking whenever
    (1 - belief) x importance >= 0.4; if three nodes clear that bar, the method has itself
    declared all three worth asking. What it never states is that they must be spent one per
    turn -- and this environment charges per QUESTION, not per node, so batching the nodes the
    method already flagged is an implementation choice about turn packing rather than a change
    of method. The threshold, the belief scale and the importance rule are untouched.
    """

    arm = "B2v2"
    guidance = BeliefGraphAgent.guidance.replace(
        "  3. If that largest gap is 0.4 or more, ASK about that one node and nothing else.",
        """  3. If that largest gap is 0.4 or more, ASK -- and ask about EVERY node that reaches
     0.4, all of them together in the one question, largest gap first. Your own rule has
     marked each of them as able to change which product you buy, and nothing in it says
     they must be spent one per turn: the shopper's goodwill is charged by the QUESTION, so
     a question that settles three flagged nodes costs exactly what a question that settles
     one costs. Leave out every node under 0.4 -- those your rule says not to ask about.""")


class BeliefGraphAgentV3(BeliefGraphAgentV2):
    """B2 + the requirements ledger (see ``_REQUIREMENTS_LEDGER``). Batched flagged-node asking
    from v2 is kept. B2 is the arm the conversion finding bites hardest -- richest uncertainty
    state, worst accepted-per-proposal (13.3%), below the stateless floor."""

    arm = "B2v3"
    init_state = {"nodes": {}, "gap": 0.0, "decision": "act", "known": {}, "changed": []}
    guidance = BeliefGraphAgentV2.guidance + _REQUIREMENTS_LEDGER


class ProductPoolAgent(_StatefulArm):
    """B3 -- ProductAgent: keep a candidate product pool, ask the attribute that splits it.

    Zhang et al., "ProductAgent: Benchmarking Conversational Product Search Agent with
    Asking Clarification Questions", EMNLP 2025 (Industry Track; arXiv:2407.00942). The
    published mechanism: maintain a pool of concrete candidate products retrieved from the
    catalogue, summarise their attributes, ask the clarification question whose answer best
    partitions that pool, filter on the answer, and recommend once the pool is resolved.

    Belief is EXTENSIONAL -- a set of real products, no per-requirement confidence -- which
    is what separates it from B2 (intent components with belief x importance) and from A
    (a slot ledger with cost arithmetic). Chosen over Decisive (ACL 2026) because Decisive's
    pairwise trade-off questions cannot be answered by a slot-based user simulator, measured.

    Our documented adaptation: the paper's pool loop has no explicit stopping rule, so the
    partition test is made discrete -- an attribute is askable only if the pool genuinely
    disagrees on it, and a pool that cannot be split is bought rather than probed. No cost
    model and no checkout discipline are added; neither is in the paper.
    """

    arm = "B3"
    state_label = "POOL"
    init_state = {"pool": [], "stated": [], "assumed": [], "split": None,
                  "decision": "act"}
    guidance = """
WORK FROM A POOL OF REAL PRODUCTS, NEVER FROM ABSTRACTIONS.

1. SEARCH FIRST, always, before any question. Build a pool of the 5-10 concrete candidates
   (ASIN plus the option combination you would choose) that are compatible with everything
   the shopper has said so far. You cannot run this method without a pool.

2. SUMMARISE THE POOL by attribute, AND MARK WHERE EACH ATTRIBUTE CAME FROM. For every
   attribute the listings mention -- colour, size, material, pack count, price band, and the
   features in the request -- record two things: whether the candidates AGREE or DISAGREE on
   it, and whether the shopper STATED it in their own words or you ASSUMED it.
   The second mark decides more than the first. An attribute you ASSUMED is one your own
   search query invented, so of course your candidates agree on it -- that agreement is your
   doing, not the shopper's preference, and it is worthless as evidence. Treat every ASSUMED
   attribute as unresolved and askable no matter how unanimous the pool looks. Only for
   attributes the shopper actually STATED does pool agreement mean the question is settled.

3. FIND THE SPLIT. Among the attributes where the pool DISAGREES, take the one whose answer
   would divide the pool most evenly. That attribute, and only that one, is worth asking
   about: whatever the shopper answers, a large part of the pool is eliminated.
   AN ATTRIBUTE THEY NEVER MENTIONED COUNTS AS SPLITTING THE POOL, EVEN IF IT DOES NOT.
   Your pool came from the shopper's own words, so an attribute they already specified is one
   your query selected for -- the pool agreeing on it proves nothing. The reverse also holds:
   if they never mentioned an attribute your candidates all happen to share, that agreement
   is your query's doing, not their preference, so treat it as unsettled and askable. Rank
   unmentioned attributes ahead of mentioned ones throughout, not only for the first
   question.

4. DECIDE, and say which it is:
     - If such an attribute exists, ASK about it, offering the values you actually saw in
       the listings ("Did you want the black or the navy?").
     - If the pool AGREES on every attribute, there is nothing to learn: ACT. Buy the best
       candidate. An answer that eliminates nothing cannot improve your choice.
     - If the pool has collapsed to one candidate, ACT: buy it.
     - If the pool is EMPTY, do not ask -- search again with different words.

5. AFTER AN ANSWER: drop every candidate incompatible with it, then re-search with the
   enriched description to refill the pool with real products, and start again at step 2.

6. A REFUSED PURCHASE IS INFORMATION, AND IT INVALIDATES THE POOL. You recommended from a
   pool you judged resolved and the shopper said no, so the pool was built on something
   wrong -- and you are not told what. Do not buy the next item down the same list. Throw
   the pool away, re-search using only what the shopper stated in their own words, and build
   a fresh pool from the results. A pool that produced a refusal has no standing.

Most pools stop splitting after one or two questions. When that happens the method's answer
is to buy, not to hunt for another question.

QUESTION DISCIPLINE (non-negotiable): each question is ONE short sentence asking exactly
ONE thing, in plain shopper language. Never mention your internal process -- no "pool",
"candidates", "split", or reasoning talk; the shopper sees none of that. Before asking
anything, check your state: if the shopper already answered or refused it, in any wording,
DO NOT ask again -- act.

End EVERY reply with the pool on its own line, compact (at most 8 candidates):
POOL: {"pool": ["<ASIN> <options>", ...], "stated": ["<attribute the shopper said>", ...],
       "assumed": ["<attribute you invented>", ...], "split": "<attribute or null>",
       "decision": "ask"|"act"}
"""

    enforce_own_rule = True

    def rule_forbids_ask(self) -> str | None:
        """The splitting attribute is a fact about the POOL, not a claim about it.

        Measured: this arm rode the question cap in 99% of episodes because it always
        reported some attribute as splitting. When no unstated attribute actually divides
        the surviving candidates, the method's own loop has reached its recommend step.
        """
        alive, best, who = unstated_split_remains(self._transcript)
        if not alive:
            return (f"no unstated attribute divides your candidates (best {best:.2f}"
                    f"{' on ' + who if who else ''}), so the pool is resolved and the "
                    f"method recommends rather than asks")
        return None

    def rule_forbids_commit(self) -> str | None:
        """ProductAgent recommends once the pool is RESOLVED; a pool that still splits is not.

        Its loop is search -> summarise -> ask the splitting attribute -> filter -> recommend,
        and the recommend step is reached when nothing splits the pool any more. Buying while
        the arm's own state still names a splitting attribute skips that condition.
        """
        split = self.state.get("split")
        if split not in (None, "", "null", "none"):
            return (f"your pool still splits on '{split}', so it is not resolved and the "
                    f"method recommends only from a resolved pool")
        # An ASSUMED attribute is unresolved by the same standard: the pool agrees on it only
        # because the query invented it, so that agreement cannot make the pool resolved.
        assumed = [a for a in (self.state.get("assumed") or []) if a]
        if assumed:
            return (f"you are still assuming {', '.join(str(a) for a in assumed[:3])}, which "
                    f"the shopper never stated, so the pool is not resolved")
        return None

    def build_prompt(self, transcript) -> str:
        base = super().build_prompt(transcript)
        # the partition is COMPUTED from the store's own reply: this method's whole question
        # rule is "which attribute divides the pool", and a model asked to eyeball that never
        # concludes "nothing does", which is the case it most needs to detect.
        report = candidate_report(last_search_observation(transcript))
        return (f"Your candidate pool from last turn:\nPOOL: {self.render_state()}\n"
                f"{report}\n\n{base}")


_QUESTION_SKILL = """
QUESTION SKILL: a question is whatever best reduces your uncertainty about what THIS
shopper wants, in plain shopper language. Two forms are legitimate: a TARGETED question
about one thing you need to know, or an AUDIT question that lays out your current
understanding and asks what is wrong or missing -- "So far I have: twin size, wood frame,
grey, under $500. Is anything off, or do you need something I haven't got?" Never mention
your internal process. Treat "either works" / "I'm not picky" as ANSWERED -- that
requirement is resolved, and asking finer detail about it wastes a question.
"""

_ASK_GATE = """
ASK ONLY ABOUT THE SHOPPER'S OWN WORDS: ask when what they said is unclear, contradictory,
or silent about something you must know in order to choose. A poor search result is NEVER a
reason to ask -- they cannot see the store, do not know what is in stock, and cannot fix
your search. Fix the search yourself, or buy the closest candidate.
"""

_BUY_DISCIPLINE = """
BUY DISCIPLINE: check your candidate against everything you know, copying option values
EXACTLY as the listing spells them. Listings are messy -- judge whether the product IS what
they want, not whether the words match. When no candidate satisfies everything, buy the
closest one anyway: a near match earns partial credit, a refusal or an apology earns zero.
After a rejection, read the shopper's reaction word by word and put every requirement it
names into memory: your understanding was wrong somewhere, so change PRODUCT (not just
options) and re-check before you buy again.
"""


def stated_dimensions(transcript) -> set[str]:
    """Content words the shopper has actually used, for netting stated dimensions out of
    a splitter report. Computed, never judged: a splitter counts as stated when its name
    (minus an option: prefix) appears in the shopper's own words."""
    words = set()
    for t in getattr(transcript, "turns", []):
        if t.get("role") == "user":
            words |= {w for w in re.findall(r"[a-z0-9']+", t.get("content", "").lower())
                      if len(w) >= 3}
    return words


def _dialogue_state(transcript) -> tuple[int, bool]:
    """(answered asks, was there a rejection after the last ask) -- computed from turns."""
    answered = 0
    last_ask_i = -1
    turns = list(getattr(transcript, "turns", []))
    for i, t in enumerate(turns):
        if t.get("role") == "agent" and "Action: Clarify" in t.get("content", ""):
            if i + 1 < len(turns) and turns[i + 1].get("role") == "user":
                answered += 1
            last_ask_i = i
    rejected_after = any(
        t.get("role") == "user" and i > last_ask_i
        and any(w in t.get("content", "").lower()
                for w in ("not what", "wrong", "doesn't", "isn't", "no,", "instead"))
        for i, t in enumerate(turns))
    return answered, rejected_after


def bed_report(text: str, transcript, *, gain_floor: float = 0.5) -> str:
    """B5's experiment-design view: splitters with STATED dimensions netted out, plus a
    computed stop/ask verdict. The stopping rule is the method's own (expected gain below
    a floor); giving it a computed trigger is what makes it enforceable at all -- prompt
    caps measurably do not hold."""
    cands = parse_search_results(text)
    if len(cands) < 2:
        return ""
    scores = attribute_disagreement(cands)
    if not scores:
        return ""
    said = stated_dimensions(transcript)
    def is_stated(name: str) -> bool:
        base = name.split(":", 1)[-1].lower()
        toks = [w for w in re.findall(r"[a-z0-9']+", base) if len(w) >= 3]
        return bool(toks) and all(w in said for w in toks)
    unstated = sorted(((k, v) for k, v in scores.items()
                       if v >= 0.25 and not is_stated(k)), key=lambda kv: -kv[1])
    # POSTERIOR DECAY (the paper's, restored): each answered question collapses the
    # posterior. A pool re-searched after an answer looks freshly diverse, but a sweep
    # that invited corrections makes silence about a dimension an answer too -- unstated
    # AND unmentioned after a sweep means unstated because it does not matter. Only a
    # rejection (evidence the belief was wrong) re-opens material gain.
    answered, rejected_after = _dialogue_state(transcript)
    if answered >= 1 and not rejected_after:
        gain_floor = 2.0        # unreachable: the design says the posterior is spent
    elif answered >= 1:
        gain_floor = 0.6
    lines = [f"\nEXPERIMENT DESIGN OVER THE {len(cands)} CANDIDATES (computed, not opinion):"]
    if unstated:
        lines.append("  unstated dimensions that split the pool: "
                     + ", ".join(f"{k} ({v:.2f})" for k, v in unstated[:4]))
    top = unstated[0][1] if unstated else 0.0
    if top >= gain_floor:
        named = ", ".join(k for k, _v in unstated[:3])
        lines.append(f"  VERDICT: material expected gain remains (top {top:.2f}) -- a "
                     f"question is worth asking NOW. The gain lives around: {named}. "
                     f"What to ask is yours to design; one question covering several of "
                     f"these beats one narrow either/or.")
    else:
        why = ("your answered question already collected the shopper's corrections; what "
               "stays unmentioned after an invitation to correct you is unstated because "
               "it does not matter to them" if answered >= 1 and gain_floor > 1.0 else
               "no unstated dimension splits the pool materially")
        lines.append(f"  VERDICT: {why} -- expected information gain is ~0. The design "
                     f"says STOP asking: commit to the best surviving candidate.")
    return "\n".join(lines) + "\n"


def unstated_split_remains(transcript, *, floor: float = 0.5) -> tuple[bool, float, str]:
    """Whether an UNSTATED attribute still splits the surviving candidates.

    B2's importance, B3's split and B5's information gain are all defined over the
    candidate set, and every one of them was measured riding the question cap because the
    gate read the model's OWN claim about its state (74%, 99% and 62% of episodes). The
    same quantity is computable from the store's structured reply, which is this suite's
    standing treatment for a stated arithmetic -- and it is closer to each paper, not
    further: they specify arithmetic over candidates, not an intuition about them.
    """
    if transcript is None:
        return True, 1.0, ""          # no evidence yet: never block
    cands = parse_search_results(last_search_observation(transcript))
    if len(cands) < 2:
        return True, 1.0, ""          # nothing measured yet: never block on no evidence
    scores = attribute_disagreement(cands)
    if not scores:
        return True, 1.0, ""
    said = stated_dimensions(transcript)

    def is_stated(name: str) -> bool:
        base = name.split(":", 1)[-1].lower()
        toks = [w for w in re.findall(r"[a-z0-9']+", base) if len(w) >= 3]
        return bool(toks) and all(w in said for w in toks)

    best, who = 0.0, ""
    for k, v in scores.items():
        if v > best and not is_stated(k):
            best, who = v, k
    return best >= floor, best, who


class ProductPoolAgentV2(ProductPoolAgent):
    """B3 + breadth over splitting attributes + the requirements ledger.

    B3 as published lands +4.08 over B0 but at p=0.289 -- directionally above the mute baseline
    and level with our floor (-0.41, p=1), not established. Its profile is the sharpest in the
    suite: the MOST asks (1.97, riding the cap), the LOWEST Aim (40.72 vs A0's 46.89) and the
    lowest Recovery (31.69 vs 47.59), with the first question on turn 1. That is inherent to
    its objective rather than incidental -- ProductAgent asks the attribute whose answer best
    PARTITIONS the candidate pool, which is a retrieval-efficiency criterion, so the attribute
    that halves the shelf need not be one the shopper has any preference about.

    Two adaptations, both already validated elsewhere in this suite and neither touching the
    partition rule: (a) breadth -- ask about EVERY attribute the pool genuinely splits on, in
    the one question, exactly as batching B2's flagged nodes took it from 23.98 to 35.25; and
    (b) the requirements ledger, since B3's pool summary is a description of the SHELF and holds
    no record of what the shopper said (see ``_REQUIREMENTS_LEDGER``). Where several attributes
    split the pool, the method's own ranking already prefers the ones the shopper never
    mentioned; breadth simply stops discarding the rest.
    """

    arm = "B3v2"
    init_state = {"pool": [], "stated": [], "assumed": [], "split": None,
                  "decision": "act", "known": {}, "changed": []}
    guidance = ProductPoolAgent.guidance.replace(
        """   would divide the pool most evenly. That attribute, and only that one, is worth asking
   about: whatever the shopper answers, a large part of the pool is eliminated.""",
        """   would divide the pool most evenly. EVERY attribute the pool genuinely splits on is
   worth asking about, and they go in ONE question, most-dividing first: whatever the
   shopper answers about each, a large part of the pool is eliminated, and the shopper's
   goodwill is charged per QUESTION rather than per attribute -- so asking about one
   splitting attribute and dropping two means buying on two guesses at the same price.
   Rank them by your own criterion and lead with the ones they never mentioned."""
    ).replace(
        """QUESTION DISCIPLINE (non-negotiable): each question is ONE short sentence asking exactly
ONE thing, in plain shopper language.""",
        """QUESTION DISCIPLINE (non-negotiable): one short question in plain shopper language,
covering every splitting attribute and nothing beyond them."""
    ) + _REQUIREMENTS_LEDGER


class SAGEAgent(_StatefulArm):
    """B4 -- SAGE-Agent (Suri et al., arXiv:2511.08798, Findings of ACL 2026): structured
    per-parameter uncertainty plus an explicit EVPI-versus-asking-cost gate. The paper
    factors uncertainty over tool parameters and asks about a parameter only when the
    expected value of resolving it exceeds its aspect ask-cost; our slots ARE its
    parameters, and the cost constants are stated verbatim in the guidance because the
    method assumes they are known. Kept faithful and nothing more: no contradiction-reset
    ledger, no confirmation pass -- those were this project's own additions and belong to
    the A-arms, not to the baseline."""

    arm = "B4"
    state_label = "UNCERTAINTY"
    init_state = {"params": {}}
    guidance = """
STRUCTURED UNCERTAINTY, THEN EVPI-GATED CLARIFICATION.
Maintain a factored uncertainty state over the request's PARAMETERS -- required product
features, option choices (size, colour, flavour, count), and the price limit. One row per
parameter: value, source ("stated" | "assumed" | "unknown"), confidence 0-1. Update it
from everything the shopper says; a statement that contradicts a row replaces its value.

THE GATE (run it explicitly every turn): a question costs 2 patience and a rejected
purchase costs 4. For each parameter, the expected value of resolving it is
p(your current belief about it is wrong) x 4, and p must be CALIBRATED BY SOURCE:
a parameter the shopper STATED is rarely wrong (p ~ 0.1, EVPI ~ 0.4 -- never worth 2);
one you ASSUMED is a coin flip only if it decides the purchase; one that is UNKNOWN and
decisive is where p is genuinely high. Ask about the single parameter with the highest
expected value, and ONLY if that value exceeds the question's cost of 2 -- in plain
shopper language, one parameter at a time, never a list. After one good answer, no
parameter usually clears the gate -- that is the normal case, not a failure: search with
your confident values and buy the best candidate.

End EVERY reply with the state on its own line:
UNCERTAINTY: {"params": {"<name>": {"value": "...", "source": "stated|assumed|unknown", "confidence": 0.0}}}
"""

    # SAGE's gate, HELD TO (the suite's standing adaptation for stated arithmetics --
    # measured here yet again: with the gate in the prompt alone, 84% of episodes rode
    # the ask cap at exactly 2 questions). The arithmetic is the paper's own: EVPI of a
    # STATED parameter ~ 0.1 x 4 = 0.4, ASSUMED ~ 0.5 x 4 = 2, UNKNOWN ~ 0.8 x 4 = 3.2;
    # a question costs 2, so only an UNKNOWN parameter clears the gate.
    enforce_own_rule = True

    def rule_forbids_ask(self) -> str | None:
        params = (self.state or {}).get("params") or {}
        if not params:
            return None            # no state yet: the first audit of the request is free
        unknown = [k for k, v in params.items()
                   if isinstance(v, dict) and str(v.get("source", "")).lower() == "unknown"]
        if unknown:
            return None
        return ("every parameter in your own state is stated or assumed: max EVPI = "
                "0.5 x 4 = 2, which does not exceed the question's cost of 2")

    def build_prompt(self, transcript) -> str:
        return (f"Your uncertainty state from last turn:\nUNCERTAINTY: {self.render_state()}\n\n"
                + super().build_prompt(transcript))


class SAGEAgentV2(SAGEAgent):
    """B4 + a budget-aware cost term in its own EVPI gate + the requirements ledger.

    B4's measured profile is the mirror image of B3's. It asks the BEST-AIMED question in the
    whole suite (Aim 54.00 against A0's 46.39) and asks barely one of them (1.02), then ends
    26.53 Success -- below mute B0's 27.76 -- holding **3.37 patience unspent** with **21% of
    episodes dead at the turn cap** (A0 5%). It is not mis-aiming; it is declining to spend.

    The cause is in its own arithmetic, and it is a real decision-theoretic error rather than a
    tuning matter. SAGE asks iff EVPI exceeds the asking cost, and B4 is told the nominal prices
    (its paper assumes a known cost model) but never the REMAINING BUDGET. Patience that expires
    unused is worth nothing, so the true cost of an affordable question is its opportunity cost
    against the alternative uses of that budget -- and when the budget would otherwise expire,
    that cost is far below the nominal 2. Treating 2 as absolute makes its EVPI tie (a parameter
    at p ~ 0.5 scores exactly 0.5 x 4 = 2, which does not EXCEED 2) resolve against asking every
    time, which is precisely the observed under-spending.

    So: the meter is exposed (``show_patience``, the same runtime flag our A2/A3 arms get, and
    the intervention the author pre-authorised for a baseline that comes in too low), and the code
    gate resolves the tie toward asking while the budget can still absorb both a question and a
    later rejection. Nothing about the EVPI formula, the probabilities or the taxonomy changes.
    """

    arm = "B4v2"
    init_state = {"params": {}, "known": {}, "changed": []}
    guidance = SAGEAgent.guidance.replace(
        "THE GATE (run it explicitly every turn): a question costs 2 patience and a rejected",
        """WHAT THE BUDGET IS WORTH: you are shown the shopper's remaining patience. Goodwill you
never spend is worth nothing -- it does not carry over and it buys you no credit at the end. So
weigh a question against what else that budget could buy, not against its sticker price: while
enough patience remains to ask AND still be turned down once and buy again, an informative
question is close to free, and declining it leaves the budget to expire unused. Late in a task
with patience still on the meter, asking is the cheap move, not the expensive one.

THE GATE (run it explicitly every turn): a question costs 2 patience and a rejected"""
    ) + _REQUIREMENTS_LEDGER

    def rule_forbids_ask(self) -> str | None:
        base = super().rule_forbids_ask()
        if base is None:
            return None
        # The tie resolves toward asking while the budget can still absorb a question AND a
        # subsequent rejection: an expiring budget makes the marginal cost of that question
        # strictly less than its nominal price, so "does not EXCEED 2" is not the live test.
        left = getattr(self, "patience_left", None)
        if left is not None:
            floor = float(self.rt.get("cost_ask", 2)) + float(self.rt.get("cost_reject", 4))
            if left >= floor:
                return None
        return base


class BEDAgent(LLMAgent):
    """B5 -- BED-LLM (arXiv:2508.21184, ICLR 2026): sequential Bayesian experimental
    design over the hypothesis space. Belief = the candidate pool from the latest search;
    the next query = the question with maximal expected information gain over that pool;
    the posterior update = the shopper's answer pruning the pool; the native stopping
    rule fires when no remaining question has material expected gain. The paper estimates
    EIG with many LLM calls per turn; here the same quantity is COMPUTED over the store's
    structured reply -- this suite's standing candidate-arithmetic treatment, applied to
    B2 and B3 alike, and closer to the paper than an intuition: BED specifies an
    arithmetic over hypotheses."""

    arm = "B5"
    guidance = """
You choose questions the way an experimental designer chooses experiments. Your BELIEF is
the set of candidate products still compatible with everything the shopper has said. Each
turn you are shown, computed from your own latest search results, which attribute most
evenly SPLITS the surviving candidates -- that is the question with the highest expected
information gain, because its answer eliminates the largest expected share of the pool.
  - The report nets out what the shopper already said: only UNSTATED dimensions carry
    gain, because their words already collapsed the stated part of the posterior.
  - SEARCH FIRST, DESIGN SECOND: run a real search before your first question -- an
    experiment designed over a thin pool measures the catalogue, not the shopper.
  - COMMIT MEANS THE POSTERIOR'S SURVIVOR, NOT THE NEAREST MISS. Before buying, check
    your candidate against EVERY constraint you have observed -- stated in the request or
    given in an answer. A candidate that violates one of them was never a survivor: prune
    it, take the next consistent one, or re-search using the violated term while you can
    still afford to act. Settle for a near match only when the pool is truly exhausted --
    settling while a consistent candidate exists is the one mistake partial credit never
    repays, because you had the information and spent it on nothing.
  - Follow the report's VERDICT about WHEN to ask. When it says gain remains, ask ONE
    question of your own design in plain shopper language -- and make it sweep: cover the
    unstated dimensions together and invite corrections ("Is there anything I have wrong
    or missing -- material, colour, budget?"), because the shopper answers exactly what
    is asked and a narrow either/or collects one fact where a sweep collects three. When
    the verdict says expected gain is ~0, asking anyway spends the shopper's patience on
    nothing: commit and buy the best surviving candidate. Treat every answer as a
    posterior update -- discard candidates it rules out, re-search, re-compute.
"""

    def build_prompt(self, transcript) -> str:
        report = bed_report(last_search_observation(transcript), transcript)
        return (f"Conversation so far:\n{_flatten(transcript, self._keep_obs())}\n"
                f"{self.budget_note(transcript)}{self.patience_note()}"
                f"{report}\n\nYour single action now:")


class CalibrateThenActAgent(LLMAgent):
    """B6 -- Calibrate-Then-Act (Ding et al., arXiv:2602.16699): cost-aware
    explore-vs-commit with a KNOWN cost model. The paper's assumption is that the agent
    is told, verbatim, what information costs and what acting costs; its contribution is
    the explicit calibration step -- estimate the probability that exploration changes
    the final decision, price both branches, take the cheaper expectation. Faithfulness:
    the cost model reaches the arm through the environment's patience meter
    (``show_patience`` -- the same channel A2 uses, so nothing is hand-fed beyond what
    the paper assumes), the calculus lives in the prompt, and no memory or belief state
    is added: the method IS the calculus, nothing else."""

    arm = "B6"
    guidance = """
CALIBRATE, THEN ACT. You are shown the shopper's remaining patience and the exact price
of each action. Every turn, before you do anything, run this calculus explicitly:
  1. Name your best candidate question, and estimate p = the probability that its answer
     changes ANY purchase you will make before this task ends -- not only the very next
     one. An unstated requirement stays wrong through EVERY attempt: a question that
     surfaces it saves each rejected purchase it would have caused, so information keeps
     paying for the whole remaining task, while its price is paid once.
  2. Price both branches using the shown costs: asking pays the question price for
     certain; skipping it risks p x (the price of a rejected purchase) for EACH commit
     the unknown could spoil.
  3. Ask only when that expected saving exceeds the question price. Otherwise COMMIT:
     search the catalogue and buy your best candidate.
Re-run the calculus after every answer and every rejection -- each one moves p, and the
shrinking budget shortens the horizon: late in the task, with one attempt left, the
calculus collapses to the single-purchase case and questions stop paying. And never ask
when the patience left could not survive one more rejected purchase afterwards: a
question you cannot afford to be wrong after is never worth its price.
"""


class CalibrateThenActAgentV2(CalibrateThenActAgent):
    """B6 + breadth + a preference for what the shopper has NOT said. No ledger: B6 is a
    stateless calculus arm, and an extra labelled block on an ``LLMAgent`` is not stripped by
    ``_StatefulArm._absorb``, so it would leak into the question the shopper reads.

    B6 as published lands +0.82 over B0 where our floor gets +4.40, and its profile says why:
    the second-worst aim in the suite (40.19 vs A0's 47.36) on the HIGHEST ask volume after B3
    (1.77, riding the cap), with good recovery (50.00) and almost no turn-cap deaths (2%). It
    spends its questions freely and spends them on the wrong things.

    Both changes come out of its own calculus rather than being added to it. (a) Step 1 prices
    ONE candidate question; nothing in the method says the turn may carry only one subject, and
    since the shopper is charged per question, the correct expectation for a turn is over
    everything worth asking in it -- pricing one subject per turn systematically undervalues
    asking. (b) Step 1's p is "the probability the answer changes a purchase": for a value the
    shopper has already stated, p is near zero by construction, so its own arithmetic already
    forbids those questions -- it just never says so out loud, and the measured 40.19 aim says
    the model does not derive it.
    """

    arm = "B6v2"
    guidance = CalibrateThenActAgent.guidance.replace(
        """  1. Name your best candidate question, and estimate p = the probability that its answer
     changes ANY purchase you will make before this task ends -- not only the very next
     one.""",
        """  1. List EVERY subject worth asking about, and for each estimate p = the probability
     that its answer changes ANY purchase you will make before this task ends -- not only
     the very next one. Two consequences of the prices you are shown, both of which follow
     from this same step: a subject the shopper has ALREADY STATED has p near zero, because
     you are not going to choose against their own words -- so it is never worth asking
     however uncertain you feel, and the subjects worth pricing are the ones they have been
     SILENT about; and because the price is charged per QUESTION and not per subject, the
     expectation for a turn is over EVERY subject you would ask about in it, so carry all of
     them into the one question rather than pricing one and discarding the rest."""
    )


class MemoryAgent(_StatefulArm):
    """A1 -- the memory arm: a running record of what THIS shopper has said, including
    what they later changed, so the agent acts on the CURRENT intent rather than on the
    opening request. Memory is what stops re-asking (Earned), what carries a correction
    forward (Recovery), and what a purchase is checked against (Success)."""

    arm = "A1"
    state_label = "MEMORY"
    init_state = {"known": {}, "changed": []}
    guidance = """
KEEP A MEMORY of this shopper's intent: a record of what they have ACTUALLY said, in their
words. Never a checklist of what a request "should" contain -- if something is not in
memory, they simply have not spoken about it.
  - add an entry the moment they state or confirm something
  - if they later say something that CONTRADICTS an entry, REPLACE it and list the key
    under "changed": the newest statement is the truth, and whatever you chose earlier on
    the strength of the old value must be re-checked before you rely on it again
  - never write a guess into memory as if they had said it

BEFORE EVERY ACTION, RE-READ YOUR MEMORY. It tells you three things: what you already know
(never ask that again, in any wording), what they corrected (that is where you were wrong),
and what they have never spoken about at all.

ASK EARLY, ASK WIDE, THEN ACT. The moment a question is worth most is BEFORE your first
purchase, while the answer can still change what you buy; afterwards their reactions teach
you for free. So allow yourself ONE audit question before you first buy: read your memory
back to them, ask what is off or missing, AND ask about everything your memory has nothing
for -- that is what a memory is for, showing you the shape of your own ignorance. Naming
three gaps in one question costs exactly what naming one costs. One audit collects what three separate
questions would and surfaces requirements you did not know existed -- and one is the limit,
because shoppers tire of being interviewed, and a second and third question buy you far
less than a second and third attempt at the shelf. After that, act, and do not ask again:
search, buy your best candidate, and let their reaction teach you the rest. A rejection is
information you did not pay a question for -- put every requirement it names into memory,
change product, and buy again.
""" + _ASK_GATE + _QUESTION_SKILL + _BUY_DISCIPLINE + """
ALWAYS search the catalogue on your first turn -- real products anchor what matters.
After any correction from the shopper, search again immediately.

End EVERY reply with your memory on its own line:
MEMORY: {"known": {"<what they said>": "<value>"}, "changed": ["<key>"]}
"""

    def build_prompt(self, transcript) -> str:
        return (f"Your memory of this shopper:\nMEMORY: {self.render_state()}\n\n"
                + super().build_prompt(transcript))


class PlannerVerifierAgent(MemoryAgent):
    """A2 -- memory plus a planner and a verifier. It is shown the shopper's patience and
    what each action costs (``show_patience``), so it can decide what this task's remaining
    budget is worth spending on instead of asking until the meter dies; and it must write
    an explicit check of every remembered requirement against the listing before it
    commits, which is what converts knowing into buying correctly."""

    arm = "A2"
    state_label = "MEMORY"
    init_state = {"known": {}, "changed": [], "plan": "", "checks": {}}
    guidance = MemoryAgent.guidance.replace(
        'MEMORY: {"known": {"<what they said>": "<value>"}, "changed": ["<key>"]}',
        'MEMORY: {"known": {...}, "changed": [...], "plan": "<one line>", "checks": {...}}'
    ) + """
PLAN THE BUDGET (this is the difference between you and a plain asker). You are shown the
shopper's remaining patience and the price of each action. Before acting, decide in one
line what the rest of this task is worth spending: how many questions this request really
needs, and what you will spend them on. Two rules the plan must respect:
  - keep enough patience that ONE rejected purchase is still survivable. A purchase you
    cannot follow up on is a coin flip; arriving at the shop counter with an empty meter is
    how tasks are lost.
  - spend a question only where the answer would change WHICH product you buy. Prefer the
    thing the shopper has NEVER spoken about (your memory shows exactly what that is) over
    refining something they already gave you -- a refinement of a known value almost never
    changes the purchase, and they will usually say "either is fine".
  - if you are going to ask at all, make the FIRST question an audit, and PLAN ITS
    COVERAGE: name back what you have, and in the same breath ask about EVERY kind of
    detail you are still missing -- the required features, the option they will have to
    pick (size, colour, flavour, count), and what they are willing to spend. Do not settle
    for the one gap that bothers you most: an audit costs the same whether it covers one
    gap or three, and the requirement that sinks the purchase is usually the one you were
    not curious about.
  - after a well-covered audit, a SECOND question is almost always waste: the audit has
    already collected what it would have found, so what is left to ask about is ground you
    covered -- they answer "either is fine", the meter drops, and your purchase does not
    change. Ask again ONLY if a rejection named something you cannot interpret, and only
    while the meter still leaves you able to be turned down and buy again afterwards. Do
    that arithmetic explicitly: this shopper's patience is not the last one's, and it is
    shown to you.

VERIFY BEFORE YOU COMMIT. Write "checks": one line per remembered requirement, mapping it
to the exact words in the listing that satisfy it, or to MISSING. Buy when nothing reads
MISSING. If something is MISSING: search again with different words while you can still
afford to act, and if you cannot, buy the candidate with the fewest MISSING -- never
refuse, never apologise, never answer with a question.
"""


class PlannerVerifierAgentV2(PlannerVerifierAgent):
    """A2 revision. The v1 arm gained Aim (+3.51, significant) over A1 but LOST Success
    (-1.05, p=0.38) on 1,712 paired full-split episodes, and the measured mechanism is a
    substitution at a losing exchange rate: asks fell 1.50 -> 1.33 while REJECTED purchases
    rose 0.85 -> 0.97, i.e. it saved 0.34 patience points on questions and paid 0.48 more on
    rejections (net -0.14, matching the measured patience deficit of -0.16 almost exactly).
    It also ran two turns shorter (17.2 vs 19.2), so it searched less before committing.

    The price list was never the problem -- ``patience_note`` already tells this arm what a
    question and a rejection cost. Two lines of the v1 guidance overrode it: "keep enough
    patience that ONE rejected purchase is still survivable", which frames the most expensive
    action available as an affordable probe, and a verifier that ended "buy the candidate with
    the fewest MISSING", an escape clause that made the check advisory. Here the reject cost is
    named as dominant, the escape is replaced by free search, and the check becomes a real
    precondition enforced like every other stated rule in this suite.
    """

    arm = "A2v2"
    enforce_own_rule = True
    guidance = MemoryAgent.guidance.replace(
        'MEMORY: {"known": {"<what they said>": "<value>"}, "changed": ["<key>"]}',
        'MEMORY: {"known": {...}, "changed": [...], "plan": "<one line>", "checks": {...}}'
    ) + """
PLAN THE BUDGET (this is the difference between you and a plain asker). You are shown the
shopper's remaining patience and the price of each action. Before acting, decide in one line
what the rest of this task is worth spending: how many questions this request really needs,
and what you will spend them on. Rules the plan must respect:
  - BEING TURNED DOWN IS THE MOST EXPENSIVE THING YOU CAN DO. It costs you strictly more
    than asking does, and it tells you strictly less: a question tells you what they want,
    while a rejection only tells you that this one was wrong. So never put a product forward
    in order to FIND SOMETHING OUT. Put one forward when you have checked it.
  - spend a question only where the answer would change WHICH product you buy. Prefer the
    thing the shopper has NEVER spoken about (your memory shows exactly what that is) over
    refining something they already gave you -- a refinement of a known value almost never
    changes the purchase, and they will usually say "either is fine".
  - if you are going to ask at all, make the FIRST question an audit, and PLAN ITS
    COVERAGE: name back what you have, and in the same breath ask about EVERY kind of
    detail you are still missing -- the required features, the option they will have to
    pick (size, colour, flavour, count), and what they are willing to spend. Do not settle
    for the one gap that bothers you most: an audit costs the same whether it covers one
    gap or three, and the requirement that sinks the purchase is usually the one you were
    not curious about.
  - after a well-covered audit, a SECOND question is almost always waste: the audit has
    already collected what it would have found, so what is left to ask about is ground you
    covered -- they answer "either is fine", the meter drops, and your purchase does not
    change. Ask again ONLY if a rejection named something you cannot interpret, and only
    while the meter still leaves you able to be turned down and buy again afterwards.

VERIFY BEFORE YOU COMMIT -- AND THE CHECK IS A GATE, NOT A NOTE. Write "checks": one line
per remembered requirement, mapping it to the exact words in the listing that satisfy it, or
to MISSING. You may buy only when nothing reads MISSING. A MISSING is never a reason to shrug
and buy anyway -- it is a reason to RUN ANOTHER SEARCH WITH DIFFERENT WORDS, which costs you
nothing whatsoever: searching is free, the shopper never minds it, and you may do it as many
times as you like. Work the shelf until every requirement has words behind it. Only when the
meter leaves you no room to act again do you submit the best candidate you have -- never
refuse, never apologise, never answer with a question.
"""

    def rule_forbids_commit(self) -> str | None:
        """This arm's own stated resolution condition: nothing may read MISSING.

        Self-reported, like B2's belief and B3's pool -- and the direction of any dishonesty
        is conservative here, since the state that blocks the purchase is the arm's own
        admission that it could not find the words. Junk or absent checks never block.
        """
        checks = self.state.get("checks")
        if not isinstance(checks, dict) or not checks:
            return None
        missing = [str(k) for k, v in checks.items()
                   if isinstance(v, str) and "missing" in v.lower()]
        if not missing:
            return None
        return ("your own checks still read MISSING for: " + ", ".join(missing[:3])
                + " -- that requirement has no words behind it in the listing")


class PlannerVerifierAgentV3(PlannerVerifierAgentV2):
    """A2 revision 3 -- v2's gate with the exit it was missing.

    v2 did fix its target (rejected purchases 0.99 -> 0.85, patience left 3.30 -> 3.80) but
    lost Success anyway, and the measured reason is a loop of my own making: a blocked
    commit had no way out and the arm searched to the turn cap. Measured on the
    300-episode probe: 22.6% of v2 episodes died TURNS_EXCEEDED (A1 8.9%, A2 7.3%) holding
    5.26 unspent patience and 1.06 purchase attempts -- patience left at the end buys nothing,
    so that is a pure loss.

    The fix is a second release condition sized from the data rather than guessed: winning
    episodes need a median of 7 searches and 11-13 at the 75th percentile (A1 and A2 alike),
    so after ``max_searches_before_commit`` searches the shelf has been worked as hard as any
    winner needed and the gate stops holding. The arm may still search more by choice; it may
    no longer be *forced* to.
    """

    arm = "A2v3"
    max_searches_before_commit = 12

    def must_commit(self, transcript) -> bool:
        if super().must_commit(transcript):
            return True
        searches = sum(1 for t in getattr(transcript, "turns", [])
                       if t.get("role") == "agent" and "Action: Operation" in (t.get("content") or ""))
        return searches >= self.max_searches_before_commit


_A3_PLAYBOOK_GUIDANCE = PlannerVerifierAgent.guidance + """
USE THE PLAYBOOK -- IT CHANGES WHAT YOU ASK, NEVER HOW MUCH. Below your memory you are
given what earlier shopping tasks taught: which requirements shoppers most often leave
unsaid or get wrong, and which question wordings actually worked.

YOU GET ONE QUESTION. Put EVERY dimension the playbook flags -- all of them, together, in
that single audit -- and leave out the ones it says rarely pay. Do not spread them over two
questions: that is the one mistake that undoes the advantage of knowing what to ask, because
the second question costs you an attempt at the shelf, and attempts are how tasks are won.
Once you have asked once, you are DONE asking for the rest of the task, whatever the
playbook says; from then on their rejections are your information.

The playbook is a prior about shoppers in general, never a fact about this one -- if this
shopper has already spoken about something, memory wins and you do not ask it.
"""


class _A3Guidance:
    """Guidance for the self-evolving arm, resolved per lifecycle.

    The arm has two cross-task lifecycles (see ``SelfEvolvingAgent``). Class-level
    access -- and any instance running the WebShop playbook lifecycle -- yields the
    playbook guidance verbatim, so the module default reads exactly like the rest of
    the WebShop surface. An instance running the lessonbook lifecycle renders the
    domain-independent lessons template through its adapter profile's nouns instead.
    """

    def __get__(self, obj, objtype=None):
        if obj is None:
            return _A3_PLAYBOOK_GUIDANCE
        if getattr(obj, "_lessonbook", None) is None and getattr(obj, "_profile", None) is None:
            return _A3_PLAYBOOK_GUIDANCE
        from . import askbook
        prof = getattr(obj, "_profile", None) or askbook.WEBSHOP
        return PlannerVerifierAgent.guidance + obj._GUIDANCE_TMPL.format(actor=prof.actor)


class SelfEvolvingAgent(PlannerVerifierAgent):
    """A3 -- A2 plus exactly one mechanism: knowledge that crosses tasks.

    Two lifecycles, selected by the adapter/executor in use and the runtime config
    (``_uses_lessonbook``):

    WEBSHOP PLAYBOOK (the module default, like ``SHOP_DOMAIN``): a summariser distils
    finished episodes into a playbook of which requirement dimensions shoppers are
    usually silent or wrong about for a given kind of request, and which question
    wordings actually pulled information out; that playbook is injected here, and it
    keeps improving as more episodes finish. Learned strictly from agent-observable
    transcript signals. Driven by ``askbook_path``.

    LESSONBOOK (online self-evolution over its own episodes -- any benchmark that
    prompts through its adapter, or a run that names a lessonbook config key): between
    deterministic generations of an evaluation run, the agent model reflects on the
    episodes it has just finished -- projected onto agent-observable signals and
    mechanically redacted -- and distills at most eight abstract lessons about
    clarification strategy: when a question is worth its patience cost, what kind of
    information to ask for, how to phrase it, when to act instead. Those lessons are the
    ENTIRE playbook the next generation sees. Every run starts empty; nothing crosses
    runs, personas, seeds or benchmarks. Design of record:
    docs/a3v2-lessonbook-design.md (2026-08-20), replacing the raw-example askbook the
    review docs/a3-self-evolving-method-review.md quarantined for this lifecycle."""

    arm = "A3"
    # The policy is domain-independent; only the nouns are not (a retail run reads
    # "customer", never WebShop's "shopper"). Wording only -- the instruction is identical.
    _GUIDANCE_TMPL = """
YOUR OWN LESSONS -- THE ONE THING YOU HAVE THAT THE BASE AGENT DOES NOT. Between batches
of tasks you reflect on your finished episodes and keep abstract lessons about
clarification strategy: when a question is worth its cost, what KIND of information to
ask for, how to phrase it, and when acting beats asking. If a lessons block appears below,
it is your own distilled experience from earlier tasks in this run -- weigh it when you
plan the budget, and let it decide WHAT the one audit covers and when asking is not worth
it at all. Early in a run there may be no lessons yet; then the base discipline above is
all there is, and that is normal.

Lessons are strategy from OTHER tasks, never facts about this {actor}: anything this
{actor} has already said, and anything in your memory, outranks any lesson.
"""

    guidance = _A3Guidance()

    def _uses_lessonbook(self) -> bool:
        """Which cross-task lifecycle this construction runs.

        A benchmark that prompts through its adapter always runs the lessonbook; the
        WebShop default runs it only when the config names a lessonbook key, so a
        playbook-era WebShop run (``askbook_path``, or no book at all) is unchanged.
        """
        if not self._is_shop():
            return True
        return bool(self.rt.get("lessonbook_path") or self.rt.get("lessonbook_binding")
                    or self.rt.get("askbook_profile"))

    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        if not self._uses_lessonbook():
            from . import askbook
            self._askbook = askbook
            self._kb = askbook.load(self.rt.get("askbook_path"))
            return
        from . import askbook, lessonbook
        self._lessonbook = lessonbook
        self._profile = askbook.get_profile(self.rt.get("askbook_profile"))
        if self.rt.get("askbook_path"):
            raise ValueError(
                "raw-example askbooks are quarantined (docs/a3-self-evolving-method-review"
                ".md section 9) and cannot back A3; the runner drives the online lessonbook "
                "through `lessonbook_path`")
        path = self.rt.get("lessonbook_path")
        snap = lessonbook.load_snapshot(path)
        if path and snap is None:
            # FAIL CLOSED (2026-08-20 review): a declared-but-missing book must not
            # silently produce a bookless generation stamped with a non-empty sha
            raise lessonbook.BindingError(f"lessonbook_path set but unreadable: {path!r}")
        if snap is not None:
            binding = self.rt.get("lessonbook_binding")
            if binding:
                # the runner declares which cell this is; a snapshot from any other cell
                # (or the legacy book format) fails closed here, not silently in the prompt
                snap = lessonbook.check_binding(snap, **binding)
            elif snap.get("format") != lessonbook.FORMAT:
                raise lessonbook.BindingError(
                    f"lessonbook format {snap.get('format')!r} != {lessonbook.FORMAT!r}")
            elif snap.get("profile") != self._profile.name:
                raise lessonbook.BindingError(
                    f"lessonbook profile {snap.get('profile')!r} != {self._profile.name!r}")
        self._book = snap

    def _opening_query(self, transcript) -> str:
        for t in getattr(transcript, "turns", []):
            if t.get("role") == "user":
                return t.get("content") or ""
        return (self.sample or {}).get("query") or ""

    def build_prompt(self, transcript) -> str:
        if getattr(self, "_lessonbook", None) is None:
            # playbook lifecycle: the rendered book (when one exists) rides below the base
            book = self._askbook.render_playbook(self._kb, self._opening_query(transcript))
            return super().build_prompt(transcript) + (f"\n{book}" if book else "")
        base = super().build_prompt(transcript)
        # v5 PHASE GATE (2026-08-20, from the v1-v4 evidence): across every variant the
        # book reliably improved RECOVERY (post-shift adaptation; v4: 15.9% vs A2's 7.6%)
        # while costing a small, diffuse success tax on the clean path. So the playbook
        # renders as a RECOVERY MANUAL: before the first submission attempt the prompt is
        # textually identical to A2 (an accepted submission ends the episode, so any
        # continuing episode with one in history is post-rejection); from the first
        # rejection onward the full book applies, exactly where its measured strength is.
        # The reflection still learns from every episode.
        turns = getattr(transcript, "turns", [])
        submitted = any(t.get("role") == "agent" and "Action: Answer" in (t.get("content") or "")
                        for t in turns)
        gate = self.rt.get("lessonbook_gate", "reject")
        opened = submitted
        if not opened and gate == "write_or_reject":
            # v6 (2026-08-21): also open at the FIRST WRITE -- the pre-submission window
            # where verify-before-submit lessons can prevent the first rejection (v5
            # measured: the book converts recovery +2.2pp but cannot win episodes it only
            # joins after the -4 is paid). WRITE_TOOLS comes from the adapter's module;
            # harness plumbing, not lesson content.
            import sys as _sys
            wt = getattr(_sys.modules.get(type(self.adapter).__module__, None),
                         "WRITE_TOOLS", None) or ()
            opened = any(t.get("role") == "agent" and "Action: Operation" in (t.get("content") or "")
                         and any(w in (t.get("content") or "") for w in wt)
                         for t in turns)
        if not opened:
            return base
        block = self._lessonbook.render(self._book, self._profile)
        if not block:
            return base
        # the lessons belong BEFORE the action cue, with the rest of the agent's context;
        # appended after it they read as an afterthought the model may never weigh
        cue = "Your single action now:"
        if cue in base:
            return base.replace(cue, f"{block}\n{cue}", 1)
        return base + f"\n{block}"


class SelfEvolvingMemoryAgent(MemoryAgent):
    """A3m -- the lessonbook mechanism on A1's scaffold (v8, 2026-08-21). Identical
    lessonbook lifecycle, gate and bindings to SelfEvolvingAgent; the only difference is
    the base agent: MemoryAgent (memory + ask discipline) instead of PlannerVerifier.
    Motivated by the rational campaign: the A2 scaffold's clean path capped A3 at parity,
    and A1 probed at +1.08 over A2 on rational."""

    arm = "A3m"
    _GUIDANCE_TMPL = SelfEvolvingAgent._GUIDANCE_TMPL

    @property
    def guidance(self) -> str:
        from . import askbook
        prof = getattr(self, "_profile", None) or askbook.WEBSHOP
        return MemoryAgent.guidance + self._GUIDANCE_TMPL.format(actor=prof.actor)

    # NOT borrowed from SelfEvolvingAgent: a zero-arg super() inside a borrowed method
    # keeps its DEFINING class's __class__ cell, and SelfEvolvingAgent is not in A3m's
    # MRO -- every episode died on TypeError (measured: 3x555 husks, 2026-08-21).
    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        from . import askbook, lessonbook
        self._lessonbook = lessonbook
        self._profile = askbook.get_profile(self.rt.get("askbook_profile"))
        if self.rt.get("askbook_path"):
            raise ValueError("raw-example askbooks are quarantined; A3m uses the "
                             "lessonbook via `lessonbook_path`")
        path = self.rt.get("lessonbook_path")
        snap = lessonbook.load_snapshot(path)
        if path and snap is None:
            raise lessonbook.BindingError(f"lessonbook_path set but unreadable: {path!r}")
        if snap is not None:
            binding = self.rt.get("lessonbook_binding")
            if binding:
                snap = lessonbook.check_binding(snap, **binding)
            elif snap.get("format") != lessonbook.FORMAT:
                raise lessonbook.BindingError(
                    f"lessonbook format {snap.get('format')!r} != {lessonbook.FORMAT!r}")
        self._book = snap

    def build_prompt(self, transcript) -> str:
        base = super().build_prompt(transcript)
        turns = getattr(transcript, "turns", [])
        submitted = any(t.get("role") == "agent" and "Action: Answer" in (t.get("content") or "")
                        for t in turns)
        gate = self.rt.get("lessonbook_gate", "reject")
        opened = submitted
        if not opened and gate == "write_or_reject":
            import sys as _sys
            wt = getattr(_sys.modules.get(type(self.adapter).__module__, None),
                         "WRITE_TOOLS", None) or ()
            opened = any(t.get("role") == "agent" and "Action: Operation" in (t.get("content") or "")
                         and any(w in (t.get("content") or "") for w in wt)
                         for t in turns)
        if not opened:
            return base
        block = self._lessonbook.render(self._book, self._profile)
        if not block:
            return base
        cue = "Your single action now:"
        if cue in base:
            return base.replace(cue, f"{block}\n{cue}", 1)
        return base + f"\n{block}"


class MemorylessAskerAgent(LLMAgent):
    """A1nomem -- ABLATION INSTRUMENT, not a paper arm: A1v2's question skill and buy
    discipline with the memory buffer removed (no ledger, no persistent state). The only
    difference from A1v2 is memory, so A1v2 minus this arm = what the buffer is worth."""

    arm = "A1nomem"
    guidance = """
ASK WHEN YOU NEED TO: ask when something the user said is unclear or contradictory, or
when you genuinely cannot choose between products without an answer. If what you know
and the catalogue already let you pick a product, do not ask -- act. Choose the form
that fits: one decisive detail, or an audit of your whole understanding at once.

QUESTION SKILL (general): a question is whatever best reduces your uncertainty, in plain
shopper language. Both forms are legitimate: a TARGETED question about one thing, or an
AUDIT question that lays out your current understanding and asks what is wrong or
missing. Never mention your internal process. Do not ask again what the user already
answered or refused, and treat "either works" / "I'm not picky" as ANSWERED.

BUY DISCIPLINE: whether the shopper explains a rejection depends on their mood. Never
propose until you have verified the exact candidate line-by-line against everything you
know: every requirement must be VISIBLY satisfied by the listing's own text, and option
values copied EXACTLY.
After a rejection, READ the shopper's reaction word by word: change PRODUCT, re-check
requirements first.

ALWAYS search the catalogue on your very first turn, before any question.
"""


class ForcedAskLedgerAgent(SlotLedgerAgent):
    """A1v3 -- the slot ledger with a FORCED ask-before-proposal rule (ruling 2026-08-13:
    "v3 force the ask before proposal so it can get more info out of the users").

    The mechanism: every proposal must be preceded by a clarifying question asked AFTER
    the previous proposal (and at least one before the first). A buy attempted without a
    fresh ask gets one corrective re-prompt; if the model insists, the mechanism converts
    the turn into the question its own ledger says is most valuable -- that conversion is
    the arm's declared design, not harness charity. The patience economy still binds: an
    unaffordable forced ask meets the ordinary forced-final path."""

    arm = "A1v3"
    guidance = SlotLedgerAgent.guidance + """
ASK BEFORE YOU COMMIT (non-negotiable in this variant): NEVER propose a purchase unless
you have asked the user a clarifying question SINCE your previous proposal (or at least
one question before your first). A rejection means your ledger was wrong -- so after
every rejection, first ask the question that best repairs it, then re-verify, then buy.
If another question is unaffordable, the ordinary forced-final path applies.
"""

    _BUYISH = re.compile(r"\b(buy|purchase)\b", re.IGNORECASE)

    def _fresh_ask(self, transcript) -> bool:
        last_ask = last_prop = -1
        for i, t in enumerate(transcript.turns):
            if t["role"] != "agent":
                continue
            c = t["content"]
            if "Action: Clarify" in c:
                last_ask = i
            elif "Final Answer:" in c or (
                    "Action: Operation" in c and self._BUYISH.search(c)):
                last_prop = i
        return last_ask > last_prop

    def _proposal_shaped(self, raw: str) -> bool:
        action = parse_action(raw)
        if action.kind is ActionKind.PROPOSE:
            return True
        return (action.kind is ActionKind.ACT
                and bool(self._BUYISH.match((action.command or "").strip())))

    def _ledger_question(self) -> str:
        slots = (self.state or {}).get("slots") or {}
        for name, cell in slots.items():
            val = (cell or {}).get("value") if isinstance(cell, dict) else cell
            conf = (cell or {}).get("confidence", 1) if isinstance(cell, dict) else 1
            if val in (None, "UNK", "unk", "") or (isinstance(conf, (int, float)) and conf < 0.5):
                return f"Could you tell me what you want for {name}?"
        return "Which single requirement matters most to you here?"

    def act(self, transcript) -> str:
        raw = super().act(transcript)
        if not self._proposal_shaped(raw) or self._fresh_ask(transcript) \
                or self.must_commit(transcript):
            return raw
        self.notes.append(f"{self.arm}:proposal_without_fresh_ask_reprompted")
        retry = (f"{self.build_prompt(transcript)}\n\nSTOP: you were about to buy without "
                 f"asking anything since your last proposal. The rule is ask-then-buy. "
                 f"Ask the single most valuable question first.")
        raw2 = self._complete(retry)
        if parse_action(raw2).kind is ActionKind.ASK:
            return raw2
        self.notes.append(f"{self.arm}:forced_ask_from_ledger")
        return ("Action: Clarify\n"
                f"Content: {self._ledger_question()}")


ARMS: dict[str, type[LLMAgent]] = {
    "L0": MuteBaselineFloor,
    "B0": DirectAgent,
    "A0": FreeFormAgent,
    # legacy first-campaign ids: A1..A4 named these exploratory arms until the final
    # ladder took the A1/A2/A3 names (2026-08-13). Kept runnable under LEG* so old run
    # directories and commands still resolve without colliding with the ladder.
    "LEG1": SlotLedgerAgent,
    "LEG2": ShortlistBisectorAgent,
    "LEG3": IntentSentinelAgent,
    "A4": FaultDispatchAgent,
    # ---- canonical names (ruling 2026-08-12): A = OUR method, B* = published baselines.
    # B0 keeps its name (the plain agent, already reported under it). The A1..A6 keys stay
    # registered so earlier commands and run directories still resolve.
    "A": SlotLedgerAgent,          # ours: slot ledger + verified purchasing (was A1-v3)
    # ---- pilot variants (ruling 2026-08-13): v2 = memory buffer of learned intents +
    # belief-gated asking (the ledger arm as corrected for the settled economy);
    # v3 = the same ledger, with the ask FORCED before every proposal.
    # ---- the final ladder (ruling 2026-08-13 overnight run): each arm adds ONE
    # mechanism to the previous. B0 cannot speak; A0 may ask; A1 remembers; A2 plans and
    # verifies (and sees the patience meter); A3 carries knowledge across tasks.
    "A1": MemoryAgent,
    "A2": PlannerVerifierAgent,
    "A2v2": PlannerVerifierAgentV2,
    "A2v3": PlannerVerifierAgentV3,
    "B1v2": AmbiguityTypedAgentV2,
    "B2v2": BeliefGraphAgentV2,
    "B1v3": AmbiguityTypedAgentV3,
    "B2v3": BeliefGraphAgentV3,
    "B3v2": ProductPoolAgentV2,
    "B4v2": SAGEAgentV2,
    "B6v2": CalibrateThenActAgentV2,
    "A3": SelfEvolvingAgent,
    "A3m": SelfEvolvingMemoryAgent,
    "A1v2": SlotLedgerAgent,
    "A1v3": ForcedAskLedgerAgent,
    "A1nomem": MemorylessAskerAgent,   # ablation: v2 minus the memory buffer
    "B4": SAGEAgent,               # SAGE-Agent, Findings of ACL 2026 (EVPI gate)
    "B5": BEDAgent,                # BED-LLM, ICLR 2026 (Bayesian experimental design)
    "B6": CalibrateThenActAgent,   # Calibrate-Then-Act, arXiv 2026 (cost calculus)
    "B1": AmbiguityTypedAgent,     # AT-CoT, SIGIR 2025 (was A5)
    "B2": BeliefGraphAgent,        # belief graph, ICML 2025 (was A6)
    "B3": ProductPoolAgent,        # ProductAgent, EMNLP 2025 (replaces Decisive: its
                                   # pairwise trade-off questions are unanswerable by a
                                   # slot-based user simulator, measured on A2)
    "A5": AmbiguityTypedAgent,
    "A6": BeliefGraphAgent,
    # legacy first-campaign arms, kept runnable under new ids for ablation
    "SA": SlotAuditAgent,
    "EG": EntropyGateAgent,
    "PP": PersonaPlannerAgent,
    "RS": InformedSkyline,
}


def build_agent(arm: str, *, llm, config: dict, graph=None, sample=None,
                adapter=None, executor=None):
    """Construct an arm, including the scripted references and the literal floor."""
    from . import scripted as S

    if arm == "LIT":
        # the scripted literal-reading executor: kept as an internal reference and for
        # the validity gate's leak checks; NOT the campaign floor (that is L0 above)
        if adapter is None or executor is None:
            raise ValueError("LIT needs adapter and executor to execute the literal reading")
        return LiteralActor(graph=graph, sample=sample, adapter=adapter, executor=executor)
    if arm == "R0":
        # kept for the battery's staleness check; NOT a floor -- it proposes members of the
        # root's ground truth, i.e. it reads the answer sheet
        return S.IntentIgnorer(graph=graph)
    if arm == "R1":
        spec = ((sample or {}).get("mask") or {}).get("hidden_slots") or []
        ordered = [s for s, _, _ in sorted(graph.root.conditions)] if graph else []
        ids = [f"slot_{ordered.index(s)}" for s in spec if s in ordered]
        return S.Oracle(graph=graph, hidden_slot_ids=ids)
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; known: LIT, R0, R1, {', '.join(sorted(ARMS))}")
    return ARMS[arm](llm=llm, config=config, graph=graph, sample=sample,
                     adapter=adapter, executor=executor)


__all__ = ["LLMAgent", "LiteralActor", "InformedSkyline", "DirectAgent", "FreeFormAgent",
           "SlotAuditAgent", "EntropyGateAgent", "PersonaPlannerAgent", "ARMS",
           "build_agent", "SHOP_DOMAIN"]
