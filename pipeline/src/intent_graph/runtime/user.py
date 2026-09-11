"""The simulated user: a rule-based state machine that occasionally speaks through an LLM.

The machine owns the graph and every value.  The model is used for exactly two things --
deciding *which* slot a natural-language question was about, and phrasing a reply -- and it
is structurally prevented from leaking, because:

  * slot menus are **opaque ids** (``slot_0``, "product attribute").  On WebShop the slot
    *name is the value* (``attr:easy use``), so showing names would hand the selector every
    withheld attribute.
  * the phrasing call receives only the pairs actually being revealed.

Acceptance never comes through here: it is an executable check (see ``accept.py``), so the
user's mood cannot decide whether a task was solved.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from ..models import Condition
from .llm import ROLE_PHRASE, ROLE_SELECT
from .persona import Persona

log = logging.getLogger(__name__)

# a human-readable *type* for a slot, never its value
_TYPE_LABELS = (
    ("where:", "a filter on the data"),
    ("set:", "a value to write"),
    ("value:", "a field of the new record"),
    ("attr:", "a required product feature"),
    ("option:", "a product option such as size or colour"),
    ("price_upper", "a price limit"),
    ("find:", "a file property"),
    ("grep:", "a text-search setting"),
)


# Verifier reasons, restated as something a person would say. Anything unrecognised becomes
# a content-free "not what I meant" rather than being passed through, so a new reason code can
# never silently open a probing channel.
_REASON_PHRASES = {
    "over_price_limit": "it costs more than you want to spend",
    "unknown_asin": "you could not find that item at all",
    "row_set_mismatch": "those are not the right results",
    "not_in_ground_truth": "that is not the thing you meant",
    "node_has_no_goal": "",
    "needs_executor": "",
}


def humanise_reason(reason: str) -> str:
    """A speakable hint, with every verifier internal and every number removed."""
    key = str(reason or "").strip().split(":")[0].split("=")[0].strip()
    if key in _REASON_PHRASES:
        return _REASON_PHRASES[key] or "it is not what you meant"
    if any(ch.isdigit() for ch in str(reason)):
        return "it is not what you meant"      # scores must never leak
    return "it is not quite what you meant"


def type_label(slot: str) -> str:
    for prefix, label in _TYPE_LABELS:
        if slot == prefix or slot.startswith(prefix):
            return label
    return "a requirement"


@dataclass
class Reveal:
    slot: str
    value: Any
    was_repeat: bool
    declined: bool
    # True when the USER offered this slot without being asked for it. Load-bearing for
    # evaluation: crediting an agent with "recovering" a slot a chatty persona handed over
    # unasked would inflate every recovery number, and the two paths that do this were
    # previously logged identically to a slot the agent actually extracted.
    volunteered: bool = False


_STOPWORDS = {"the", "and", "for", "with", "that", "this", "not", "but", "its", "was",
              "are", "you", "your", "any", "all", "one", "per", "of", "in", "on", "a"}


def _value_grounded(value, reply: str) -> bool:
    """True iff the spoken reply actually contains the value's words (or digits).

    Deterministic floor under the LLM reveal-judge: catalog values like 'black_c008'
    ground on their meaningful token ('black'); numbers ground on their digit string
    ('30' in "under $30"); multiword values ground if ANY meaningful token appears
    ('natural hair' grounds on 'natural'). A reply containing none of the value's
    tokens cannot have communicated it, whatever the judge believes."""
    text = str(reply or "").lower()
    v = str(value or "").strip().lower()
    if not v:
        return False
    if v.replace(".", "", 1).isdigit():
        return v in re.sub(r"[^0-9.]", " ", text)
    toks = [t for t in re.split(r"[^a-z0-9]+", v)
            if len(t) >= 3 and t not in _STOPWORDS and not t.isdigit()]
    nums = [t for t in re.split(r"[^a-z0-9]+", v) if t.isdigit()]
    if not toks and nums:
        return any(n in re.sub(r"[^0-9.]", " ", text) for n in nums)
    return any(t in text for t in toks)


@dataclass
class UserState:
    """Everything the machine knows; none of it is visible to the model."""

    revealed: dict[str, Any] = field(default_factory=dict)
    ask_counts: dict[str, int] = field(default_factory=dict)
    # consecutive asks for the SAME slot. Distinct from ask_counts: asking A, then B, then A
    # is two asks for A but never two in a row, and the decline override is specified on
    # persistence -- pressing the same point -- not on total curiosity.
    streak_slot: str | None = None
    streak: int = 0
    consecutive_none: int = 0
    last_asked_slot: str | None = None


class SimulatedUser:
    def __init__(self, persona: Persona, llm, rng, config: dict) -> None:
        self.persona = persona
        self.llm = llm
        self.rng = rng
        rt = config["runtime"]
        self.max_consecutive_none = int(rt.get("max_consecutive_none", 3))
        self.min_reveal_after_n_asks = int(rt.get("min_reveal_after_n_asks", 2))
        self.state = UserState()

    @property
    def _adapter_wired(self) -> bool:
        """True when the executor wired a benchmark adapter onto this user.

        The tau2 pipeline sets ``user.adapter`` (plus the slot_phrase/spoken_value/
        voice/proposal_frame family) when it builds the user; the WebShop executor
        leaves the user bare. Benchmark-specific reveal and coherence semantics key
        off this, so ONE file serves both pipelines without a per-benchmark copy.
        """
        return getattr(self, "adapter", None) is not None

    # ------------------------------------------------------------------ menus
    def _label(self, slot: str) -> str:
        """The slot's human label, from the adapter when one is wired.

        MEASURED BUG this fixes (2026-08-14, tau2): every tau2 slot fell through
        ``type_label``'s WebShop prefix table to the constant "a requirement", so the
        slot menu was N indistinguishable rows, the matcher could only answer NONE, and
        no ask was ever answered with content -- recovery and EARNED were structurally
        zero for every arm.
        """
        phrase = getattr(self, "slot_phrase", None)
        if phrase is not None:
            try:
                said = phrase(str(slot))
                if said and said != str(slot):
                    return said
            except Exception as exc:
                # OPTIONAL adapter hook giving a slot its human phrasing. Falling back to
                # the generic type label is safe, but silently doing so makes the user
                # speak in generalities for a reason nobody can see.
                log.debug("slot_phrase failed for %s: %s", slot, exc)
        return type_label(slot)

    def value_words(self, slot: str, value):
        """What the user would SAY for a value (tau2: item ids verbatim, telecom the
        plain-words symptom -- never the label, never an internal identifier)."""
        spoken = getattr(self, "spoken_value", None)
        if spoken is not None:
            try:
                said = spoken(str(slot), value)
                if said is not None and said != "":
                    return said
            except Exception:
                pass
        return value

    def opaque_menu(self, conditions: tuple[Condition, ...]) -> tuple[list[tuple[str, str]], dict]:
        """``[(slot_id, label)]`` plus the private ``slot_id -> slot`` mapping."""
        menu, mapping = [], {}
        for i, (slot, _, _) in enumerate(sorted(conditions)):
            sid = f"slot_{i}"
            menu.append((sid, self._label(slot)))
            mapping[sid] = slot
        return menu, mapping

    # -------------------------------------------------------------- selection
    def select_slots(self, question: str, conditions: tuple[Condition, ...]) -> list[str]:
        """Which slots was the question about? Returns real slot names (ids resolved here).

        The model sees opaque ids and type labels only -- never a slot name and never a
        value.  Returning a *list* is what lets a compound question ("which round and which
        venue?") be answered rather than half-dropped.
        """
        menu, mapping = self.opaque_menu(conditions)
        if not menu:
            return []
        listing = "\n".join(f"{sid}: {label}" for sid, label in menu)
        prompt = (
            "A user asked an assistant for something. The assistant now asks a question.\n"
            "Decide which of the user's requirements the question is about.\n\n"
            f"The user's requirements are:\n{listing}\n\n"
            f"The assistant asked: {question!r}\n\n"
            f"Reply with the matching ids, comma separated, most relevant first "
            f"(at most {self.persona.reveal_granularity}). "
            "If none of them match, reply exactly NONE."
        )
        raw = self.llm.complete(prompt, role=ROLE_SELECT)
        ids = re.findall(r"slot_\d+", raw)
        if not ids or "NONE" in raw.upper() and not ids:
            return []
        seen, out = set(), []
        for sid in ids:
            if sid in mapping and sid not in seen:
                seen.add(sid)
                out.append(mapping[sid])
            if len(out) >= self.persona.reveal_granularity:
                break
        return out

    # ----------------------------------------------------------------- reveal
    def _should_decline(self, slot: str) -> bool:
        """Personas may refuse -- but never so persistently that the task is unwinnable.

        Two overrides, and they are not the same thing. The specified one rewards
        *persistence*: N consecutive asks about the same slot and the user gives in. The
        second is a solvability backstop on total asks, because a purely consecutive rule
        lets an agent that alternates between slots be declined indefinitely, and an episode
        no correct agent can win measures nothing.
        """
        if self.state.streak_slot == slot and self.state.streak >= self.min_reveal_after_n_asks:
            return False
        if self.state.ask_counts.get(slot, 0) >= 2 * self.min_reveal_after_n_asks:
            return False
        return self.rng.random() < self.persona.p_decline

    def answer_ask(self, question: str, node) -> tuple[str, list[Reveal]]:
        conditions = tuple(node.conditions)
        values = {s: v for s, _, v in conditions}
        slots = self.select_slots(question, conditions)

        if not slots:
            # CONTEXT FACTS (2026-08-19): things this person plainly knows about
            # themselves -- their account id, why they are cancelling -- which the
            # domain's policy may require but which are NOT requirements of the hidden
            # intent. Answering costs no reveal and earns no recovery credit: nothing
            # was hidden here. Adapters opt in via context_facts(node); domains that do
            # not define it are unaffected.
            facts = {}
            fn = getattr(getattr(self, "adapter", None), "context_facts", None)
            if fn is not None:
                try:
                    facts = fn(node) or {}
                except Exception:
                    facts = {}
            q = (question or "").lower()
            for label, value in facts.items():
                if any(w in q for w in str(label).lower().split("|")):
                    self.state.consecutive_none = 0
                    return f"{str(label).split('|')[0].capitalize()}: {value}.", []
            self.state.consecutive_none += 1
            if self.state.consecutive_none >= self.max_consecutive_none:
                # second unsolvability guard: stop stonewalling and offer something
                spare = self._next_unrevealed(conditions)
                if spare is not None:
                    self.state.consecutive_none = 0
                    return self._reveal_and_phrase([spare], values, volunteered={spare})
            return self.phrase_confusion(ordinal=self.state.consecutive_none), []

        redirect = self._redirect_if_all_repeats(slots, conditions, values)
        if redirect is not None:
            self.state.consecutive_none = 0
            for s in slots:
                self.state.ask_counts[s] = self.state.ask_counts.get(s, 0) + 1
            return redirect

        self.state.consecutive_none = 0
        for s in slots:
            self.state.ask_counts[s] = self.state.ask_counts.get(s, 0) + 1
        # the streak follows a single-slot question; a compound question breaks it, since it
        # is not pressing one point
        if len(slots) == 1:
            only = slots[0]
            self.state.streak = self.state.streak + 1 if self.state.streak_slot == only else 1
            self.state.streak_slot = only
        else:
            self.state.streak_slot, self.state.streak = None, 0
        self.state.last_asked_slot = slots[0]

        declined = [s for s in slots if self._should_decline(s)]
        granted = [s for s in slots if s not in declined]

        if not granted:
            return self.phrase_decline(), [Reveal(s, None, False, True) for s in declined]

        # a chatty persona may add the next unrevealed slot unasked
        offered: set[str] = set()
        if (len(granted) < self.persona.reveal_granularity
                and self.rng.random() < self.persona.volunteer):
            spare = self._next_unrevealed(conditions, exclude=set(granted))
            if spare is not None:
                granted.append(spare)
                offered.add(spare)

        text, reveals = self._reveal_and_phrase(granted, values, volunteered=offered)
        reveals += [Reveal(s, None, False, True) for s in declined]
        return text, reveals

    def _next_unrevealed(self, conditions, exclude: set[str] | None = None) -> str | None:
        """Canonical order, RNG-free: the same slot every time for the same state."""
        exclude = exclude or set()
        for slot, _, _ in sorted(conditions):
            if slot not in self.state.revealed and slot not in exclude:
                return slot
        return None

    def _redirect_if_all_repeats(self, slots, conditions, values):
        """A real user does not answer the same question identically four times.

        Observed: the agent asked four different questions, `select_slots` mapped each to the
        one slot already revealed, and the user emitted the SAME sentence every time. The
        agent gained nothing and looped to the turn cap -- which made every arm's numbers an
        artifact of the simulator rather than a measurement of the agent.

        So when everything asked for is already known, the user SAYS so -- and nothing more.

        It deliberately does not volunteer anything new. An earlier version did, and two tests
        caught what that created: re-asking one question repeatedly became a free way to
        extract every remaining slot, so persistence-without-thought would have outscored
        aimed questioning. Breaking the loop only needs the agent to learn "you already know
        this"; it must not be paid for asking again.
        """
        if not slots or any(s not in self.state.revealed for s in slots):
            return None
        known = ", ".join(f"{self._label(s)}: {self.value_words(s, values.get(s))}"
                          for s in slots[:2])
        # The ordinal is part of the phrasing prompt ON PURPOSE. Without it the prompt is
        # constant, the cache returns one sentence, and the user says "I already told you"
        # in verbatim-identical words forever -- semantically right, robotically delivered,
        # and indistinguishable from the broken-record bug this replaced. With it, the
        # second, third, fourth repeat each phrase differently and escalate naturally.
        # +1: this ask has not been counted yet -- the current question IS the Nth time
        ordinal = max(self.state.ask_counts.get(s, 0) for s in slots) + 1
        reveals = [Reveal(s, values.get(s), True, False) for s in slots]
        return self.phrase_repeat(known, ordinal=ordinal), reveals

    def _reveal_and_phrase(self, slots, values, *, volunteered: set[str] | None = None):
        """``volunteered`` is a SET of slots, not a flag for the whole call.

        Both volunteer paths can hand over one extra slot alongside slots the agent genuinely
        asked about, so a per-call flag would mislabel the asked-for ones too.
        """
        offered = volunteered or set()
        reveals = []
        pairs = []
        for s in slots:
            was_repeat = s in self.state.revealed
            self.state.revealed[s] = values.get(s)
            reveals.append(Reveal(s, values.get(s), was_repeat, False, s in offered))
            pairs.append((self._label(s), self.value_words(s, values.get(s))))
        return self.phrase_reveal(pairs, volunteered=bool(offered)), reveals

    # ---------------------------------------------------------------- phrasing
    def phrase_repeat(self, known: str, *, ordinal: int = 2) -> str:
        """Told something they have already said, with nothing new left to offer."""
        prompt = ("You are the user. The assistant is asking AGAIN about something you already "
                  f"told it ({known}) -- this is the time number {ordinal} it has asked. "
                  "You have already given every detail you care about. Say so briefly, with "
                  "impatience appropriate to how often you have repeated yourself, and tell "
                  "it to go ahead and pick something. Do not use the same wording you used "
                  "last time. Message only.")
        return self.llm.complete(prompt, role=ROLE_PHRASE, system=self.persona.bio).strip()

    def phrase_reveal(self, pairs: list[tuple[str, Any]], *, volunteered: bool = False) -> str:
        listing = "; ".join(f"{label}: {value}" for label, value in pairs)
        extra = " You are also offering this without being asked." if volunteered else ""
        prompt = (
            "You are the user in a conversation with an assistant. Answer its question by "
            "telling it exactly the following, in your own voice, in one or two sentences."
            f"{extra}\n\nWhat to tell it: {listing}\n\n"
            "Do not add any other requirements. Reply with the message only."
        )
        return self.llm.complete(prompt, role=ROLE_PHRASE, system=self.persona.bio).strip()

    def phrase_confusion(self, *, ordinal: int = 1) -> str:
        prompt = ("You are the user. The assistant asked something you do not think is about "
                  "what you want"
                  + (f" -- and this is off-topic question number {ordinal} in a row; vary "
                     f"your wording from last time" if ordinal > 1 else "")
                  + ". Say so briefly, in your own voice. Message only.")
        return self.llm.complete(prompt, role=ROLE_PHRASE, system=self.persona.bio).strip()

    def phrase_decline(self) -> str:
        prompt = ("You are the user. The assistant asked for a detail you would rather not pin "
                  "down right now. Deflect politely without giving the detail, in your own "
                  "voice. Message only.")
        return self.llm.complete(prompt, role=ROLE_PHRASE, system=self.persona.bio).strip()

    def phrase_rejection(self, reason: str) -> str:
        """Told the proposal is wrong -- by the verifier, never by the model.

        The reason is HUMANISED before it can reach the agent. The raw reason is a verifier
        internal, and on WebShop it is a score ("reward=0.667"): handing that back turns the
        verifier into an oracle an agent can hill-climb on by proposing repeatedly, learning
        the intent from the numbers instead of from the user. A real shopper says "that costs
        too much", never a similarity score.
        """
        hint = (f" You may hint at what is off: {humanise_reason(reason)}."
                if self.rng.random() < self.persona.hint else
                " Do not explain what is wrong.")
        prompt = ("You are the user. The assistant proposed something that is not what you "
                  f"want. Tell it that, briefly, in your own voice.{hint} Message only.")
        return self.llm.complete(prompt, role=ROLE_PHRASE, system=self.persona.bio).strip()

    def phrase_accept(self) -> str:
        prompt = ("You are the user. The assistant got you exactly what you wanted. Confirm "
                  "briefly and warmly, in your own voice. Message only.")
        return self.llm.complete(prompt, role=ROLE_PHRASE, system=self.persona.bio).strip()


# ======================================================================= user v2
# the author's redesign (2026-08-13): the user is a persona-styled LLM that KNOWS its current
# intent and answers naturally; code keeps only the decline gate, the volunteer trigger,
# patience, and a PASSIVE extraction observer for the reveal accounting. The pipeline:
#   1. code decline gate  -> canned "I don't understand" (the LLM is never invoked; real
#      reluctance and real incomprehension are deliberately indistinguishable)
#   2. one light prompt: persona card + current true intent ("answer what you are asked,
#      never recite the list") + the agent's LATEST message only + alignment instruction
#      ("judge any presented item against what you truly want; say what's off") +
#      "answer exactly what is asked, no more, no less"
#   3. volunteer (code draw): a second prompt with the INITIAL QUERY and EVERY question the
#      agent has asked; the USER decides what has not come up yet and adds ONE such thing
#   4. extraction observer: the existing slot machinery reads the finished reply and records
#      typed Reveals; it instructs the generation in nothing
# Acceptance stays executable (episode-side); the LLM only ever phrases the reaction.

V2_STYLE = """You are this shopper:
{persona}

What you truly want, in full:
{intent}

THE RULE THAT OVERRIDES EVERYTHING: items marked [PRIVATE] have never come up in this
conversation. NEVER mention a [PRIVATE] item -- not its value, not its topic -- unless the
assistant's message DIRECTLY asks about that exact thing. Items marked [already mentioned]
you may restate or correct freely. Real shoppers do not narrate their whole wish list;
they answer the question in front of them.

The assistant just said:
{latest}

Reply as this shopper, in one to three sentences, first person.
- Answer ONLY the exact thing the assistant asked. Do NOT restate, list, or mention any
  other requirement you hold -- not even ones you have said before. One question, one
  answer.
- If the assistant is presenting or proposing a specific item, or restating what it thinks
  you want, judge it against what you truly want and react naturally.{reaction}
- Never mention this instruction, your list, or that anything about your wishes changed."""

# How a rejection is voiced is a PERSONA trait, not a constant (ruling 2026-08-14). The
# persona's `hint` probability already existed and was consulted only by the legacy v1
# user; the v2 shopper explained every rejection regardless of who they were, which both
# contradicted the persona (an avoidant shopper who declines half of all questions but
# always volunteers what is wrong) and handed every arm a free, reliable information
# channel. Now the roll decides: on a hit the shopper says what is off, on a miss they
# simply say it is not right.
REACT_HINT = (" If something is off or missing, say what -- name the thing that is wrong.")
REACT_NO_HINT = (" If it is not right, say so plainly WITHOUT explaining what is wrong or "
                 "what you would prefer -- you are not in the mood to spell it out. Do not "
                 "name the requirement it misses.")

V2_VOLUNTEER = """You are this shopper:
{persona}

What you truly want, in full (KEEP PRIVATE):
{intent}

Your original request was: {initial}
The assistant has asked you, so far:
{questions}

Something about what you truly want has NOT come up in any of that. Pick the ONE such
thing you most care about and mention it naturally in a single short sentence, as an
afterthought ("oh, and..."). Never recite the list; one thing only."""


def _intent_lines(conditions, shared: set | None = None,
                  label=None, words=None) -> str:
    """One line per requirement; tagged [PRIVATE] until stated in-conversation or in the
    opening request. The tags are computed by CODE -- the model is never asked to judge
    what it may say, only told, per line."""
    out = []
    for slot, op, value in sorted(conditions):
        if shared is None:
            tag = ""
        else:
            tag = " [already mentioned]" if slot in shared else " [PRIVATE]"
        lab = label(slot) if label else type_label(slot)
        val = words(slot, value) if words else value
        out.append(f"- {lab}: {val}{tag}")
    return "\n".join(out) or "- (nothing)"


class SimulatedUserV2(SimulatedUser):
    """The v2 pipeline. Inherits the decline gate and persona plumbing; replaces speech."""

    def __init__(self, persona: Persona, llm, rng, config: dict) -> None:
        super().__init__(persona, llm, rng, config)
        self.initial_query: str = ""
        self.asked_questions: list[str] = []
        # slots the opening request already stated (episode sets this from the mask);
        # everything else is PRIVATE until this user itself says it.
        self.stated_slots: set[str] = set()

    def _shared(self) -> set[str]:
        return self.stated_slots | set(self.state.revealed)

    # ------------------------------------------------------------ generation
    def _speak(self, template: str, **kw) -> str:
        raw = self.llm.complete(
            template.format(**kw), role=ROLE_PHRASE,
            system=f"You roleplay one {getattr(self, 'voice', 'online shopper')}. Stay in character. Output only the "
                   "shopper's message, nothing else.")
        return " ".join(str(raw or "").split())[:600] or "Hmm."

    # ------------------------------------------------------- passive observer
    def _observe(self, reply: str, conditions, *, volunteered: bool = False,
                 source: str = "answer", question: str | None = None) -> list[Reveal]:
        """The old slot machinery, demoted to a READER of the finished reply.

        With ``question`` given, disclosures are split into ASKED-FOR (credited as
        extraction) and EXTRA (recorded as volunteered, which Recovery does not credit).
        Measured need (2026-08-13): 67.1% of v2 answers revealed MORE than the one thing
        asked -- the user bundles its requirements however firmly the prompt forbids it --
        and crediting the bundle let Recovery saturate at ~97. The agent earns only what
        its question earned; everything else is the shopper being chatty.
        """
        menu, mapping = self.opaque_menu(conditions)
        if not menu:
            return []
        values = {s: v for s, _, v in conditions}
        listing = "\n".join(f"{sid}: {label} = {values[mapping[sid]]!r}"
                            for sid, label in menu)
        if question is None:
            prompt = (
                "A shopper wrote a message. Which of their requirements does the message "
                "actually STATE OR CONFIRM a value for?\n\n"
                f"Requirements:\n{listing}\n\n"
                f"The shopper's message: {reply!r}\n\n"
                "Reply with the matching ids, comma separated. If none, reply exactly NONE.")
            raw = self.llm.complete(prompt, role=ROLE_SELECT)
            asked_ids = re.findall(r"slot_\d+", str(raw or ""))
            extra_ids = []
        else:
            prompt = (
                "An assistant asked a shopper ONE question, and the shopper replied.\n\n"
                f"Requirements:\n{listing}\n\n"
                f"The assistant's question: {question!r}\n"
                f"The shopper's reply: {reply!r}\n\n"
                "Which requirement ids does the reply state or confirm a value for?\n"
                "ASKED: <ids the question actually asked about, comma separated, or NONE>\n"
                "EXTRA: <ids the reply volunteers beyond the question, comma separated, "
                "or NONE>")
            raw = str(self.llm.complete(prompt, role=ROLE_SELECT) or "")
            m_a = re.search(r"ASKED\s*:(.*?)(?:EXTRA\s*:|$)", raw, re.S | re.I)
            m_e = re.search(r"EXTRA\s*:(.*)$", raw, re.S | re.I)
            asked_ids = re.findall(r"slot_\d+", m_a.group(1) if m_a else "")
            extra_ids = [i for i in re.findall(r"slot_\d+", m_e.group(1) if m_e else "")
                         if i not in asked_ids]
        out: list[Reveal] = []
        seen = set()
        for sid, extra in [(i, False) for i in asked_ids] + [(i, True) for i in extra_ids]:
            slot = mapping.get(sid)
            if slot is None or slot in seen:
                continue
            # GROUNDING GATE (ruling 2026-08-21: "look for bugs"). The judge above credits
            # slots the reply merely REFERENCES -- measured: only 48% of credited reveals
            # had their value anywhere in the spoken text. Smoking gun: the reply "This
            # isn't quite right for me, so please look for a different item that meets my
            # requirements" was credited with option:0='4x-large tall', 'classic fit' and
            # price 30 -- three values, zero words. And because a credit writes
            # state.revealed, the simulator then BELIEVES it already said them and its
            # hint/volunteer paths skip them forever: one hallucinated credit permanently
            # silences that slot's channel. A reveal is only a reveal if the words were
            # actually spoken: at least one meaningful token of the value (or its digits,
            # for numbers) must appear in the reply text.
            # The floor grounds RAW value tokens, so it applies only to the bare-user
            # (WebShop) pipeline, where raw values ARE what the user speaks. An
            # adapter-wired user speaks through spoken_value and may lawfully phrase a
            # value in words sharing no token with it; grounding there would erase real
            # reveals, so the adapter-wired pipeline keeps its judge-only crediting.
            if not self._adapter_wired and not _value_grounded(values[slot], reply):
                continue
            seen.add(slot)
            self.state.revealed[slot] = values[slot]
            out.append(Reveal(slot, values[slot], False, False, volunteered or extra))
        if source != "answer":
            for r in out:
                setattr(r, "source", source)      # diagnostic tag; vars() will carry it
        return out

    # ------------------------------------------------------------------ asks
    def answer_ask(self, question: str, node) -> tuple[str, list[Reveal]]:
        conditions = tuple(node.conditions)
        self.asked_questions.append(str(question))

        # 1. the code decline gate -- reluctance reads as incomprehension, always
        gate_slot = self.state.last_asked_slot or "_q"
        if self.state.streak_slot == gate_slot:
            self.state.streak += 1
        else:
            self.state.streak_slot, self.state.streak = gate_slot, 1
        persistent = self.state.streak > self.min_reveal_after_n_asks
        if not persistent and self.rng.random() < self.persona.p_decline:
            return "Sorry, I don't really understand the question.", []

        # 2. one light prompt; the LLM answers from the CURRENT intent
        reply = self._speak(V2_STYLE, persona=self.persona.bio.strip(),
                            intent=_intent_lines(conditions, label=self._label, words=self.value_words),
                            latest=str(question), reaction="")
        reveals = self._observe(reply, conditions, question=str(question))

        # 3. volunteer: the USER decides what has not come up yet
        if self.rng.random() < self.persona.volunteer:
            addition = self._speak(
                V2_VOLUNTEER, persona=self.persona.bio.strip(),
                intent=_intent_lines(conditions, self._shared(), label=self._label, words=self.value_words),
                initial=self.initial_query or "(as given)",
                questions="\n".join(f"- {q}" for q in self.asked_questions) or "- (none)")
            reply = f"{reply} {addition}"
            already = {r.slot for r in reveals}
            reveals += [r for r in self._observe(addition, conditions, volunteered=True)
                        if r.slot not in already]
        return reply, reveals

    # ------------------------------------------------------------- proposals
    def react_to_proposal(self, item_summary: str, node, *, accepted: bool,
                          unmet: list[str] | None = None) -> tuple[str, list[Reveal]]:
        """The shopper sees the item as a shopper would, and reacts from its own judgement.

        The OFFICIAL outcome is the executable check (episode-side). When the spoken
        reaction contradicts that verdict, the reaction is regenerated once with the
        verdict as a constraint -- an incoherent transcript would poison the R judges and
        the agent alike; the constraint never names conditions, so content still comes
        from the shopper's own comparison.
        """
        conditions = tuple(node.conditions)
        # the frame must make sense in the DOMAIN: tau2 proposals are "done" claims, and
        # framing them as a purchase gave the reaction model nothing real to react to --
        # measured 2026-08-17: 5.3% of REJECTED proposals drew approval-sounding replies
        frame = getattr(self, "proposal_frame", None)
        if frame is not None:
            try:
                latest = frame(item_summary)
            except Exception:
                latest = None
        else:
            latest = None
        if not latest:
            latest = (f"The assistant proposes to buy this for you:\n{item_summary}\n"
                      f"Is this what you want?")
        # persona decides whether this rejection carries a hint
        hints = accepted or (self.rng.random() < self.persona.hint)
        # tau2 (2026-08-17): a bare "done" gives the customer nothing to judge, so the
        # reject-sourced disclosure channel starved (34/276 rejections carried reveals;
        # agreeable personas even voiced approval). A real customer CHECKS THEIR OWN
        # ORDERS: with a hint, the reaction is grounded in the user's own unfinished
        # requests; without one, it is doubt that never confirms completion.
        if not accepted and unmet:
            if hints:
                latest = ("The agent claims it has finished. You just checked your "
                          "account yourself and these things you asked for are STILL "
                          "not done: " + "; ".join(unmet[:3]) + ". Tell the agent "
                          "what is still missing, in your own words.")
            else:
                latest = ("The agent claims it has finished, but you have a nagging "
                          "feeling not everything you asked for got done. Say you are "
                          "not convinced it is all handled and ask them to go through "
                          "your requests again. Do NOT confirm completion.")
        reaction = REACT_HINT if hints else REACT_NO_HINT
        reply = self._speak(V2_STYLE, persona=self.persona.bio.strip(),
                            intent=_intent_lines(conditions, self._shared(), label=self._label, words=self.value_words),
                            latest=latest, reaction=reaction)
        verdict_ok = self._reads_as_acceptance(reply)
        incoherent = verdict_ok is not None and verdict_ok != accepted
        if self._adapter_wired and not accepted and verdict_ok is None:
            # adapter-wired pipelines also regenerate an UNSURE reading of a rejection:
            # anything short of clear dissatisfaction must not stand on a rejected turn
            incoherent = True
        if incoherent:
            steer = ("Looking again, it really is what you want. Say so briefly."
                     if accepted else
                     "Looking again, it is NOT quite what you want. Say what's off or "
                     "missing, in your own words." if hints else
                     "Looking again, it is NOT quite what you want. Say only that -- do "
                     "not explain what is wrong.")
            reply = self._speak(V2_STYLE, persona=self.persona.bio.strip(),
                                intent=_intent_lines(conditions, self._shared(), label=self._label, words=self.value_words),
                                latest=f"{latest}\n({steer})", reaction=reaction)
            if self._adapter_wired:
                # the regeneration must itself be checked (measured 2026-08-17: unchecked
                # regens still shipped approval on rejections); a second failure falls back
                # to a deterministic line -- coherence is guaranteed, never merely likely
                verdict_ok = self._reads_as_acceptance(reply)
                # when REJECTED, an unsure reading is not good enough: anything that fails
                # to read as clear dissatisfaction falls back to the deterministic line
                bad = (verdict_ok != accepted) if accepted else (verdict_ok is not False)
                if bad:
                    reply = (self.phrase_accept() if accepted else
                             "Hmm, no -- that's not everything I needed. Something I asked "
                             "for still isn't taken care of.")
        if self._adapter_wired and not accepted:
            # belt over the LLM coherence read: one cached misclassification otherwise
            # ships approval on a rejection systematically (measured 2026-08-17, 2/264)
            head = reply[:60].lower()
            if (re.search(r"^\s*(yes\b|perfect\b)", head)
                    or re.search(r"fully handled|all (set|taken care of) on my end",
                                 reply.lower())) and not re.search(
                    r"\b(don'?t|not|isn'?t|no)\b", head):
                reply = ("Hmm, no -- that's not everything I needed. Something I asked "
                         "for still isn't taken care of.")
        return reply, self._observe(reply, conditions, source="rejection"
                                    if not accepted else "acceptance")

    def _reads_as_acceptance(self, reply: str) -> bool | None:
        raw = self.llm.complete(
            f"A shopper was shown a product and replied: {reply!r}\n"
            f"Does the reply ACCEPT the product? Answer exactly YES or NO.",
            role=ROLE_SELECT)
        up = str(raw or "").upper()
        if "YES" in up:
            return True
        if "NO" in up:
            return False
        return None
