"""Pipeline v2: generate the flaw freely, then verify it structurally (docs/pipeline-v2.md).

Drift-bench's generation — the strategy card verbatim, the LLM writes the flawed instruction
however the strategy naturally comes out — followed by structured EXTRACTION of the diff and
mechanical ADMISSION. The mask still exists (recovered, not imposed), the literal readings
still execute, and a sample ships only after clearing three independent nets:

  1. the one-flaw contract (the extractor is a flaw-counter)
  2. string cross-checks against the text
  3. execution: each strategy's flaw must have its provable consequence

Rejection is cheap and honest; regeneration is budgeted; every failure lands in the skips
ledger with its verdict.
"""

from __future__ import annotations

import json
import logging
import re

# --------------------------------------------------------------- shape guards
# The extractors are language models, and a model that is asked for
# [["cabin", "economy"], ...] will occasionally return ["cabin", ...] instead. Indexing a
# bare string does NOT raise -- "cabin"[0] is "c" -- so an unguarded row silently becomes
# a one-letter slot name and corrupts the mask. Rows of the wrong shape are dropped, which
# turns a silent corruption (or an AttributeError tallied beside real rejections) into a
# missing row that the contract checks below catch honestly.
def _rows(items):
    """Rows that must be [slot, ...] sequences."""
    return [it for it in (items or []) if isinstance(it, (list, tuple)) and it]


def _objs(items):
    """Rows that must be objects."""
    return [it for it in (items or []) if isinstance(it, dict)]


from ..ids import content_hash
from . import strategies as st
from .agents import _first_json
from .llm import ROLE_JUDGE, ROLE_RENDER, ROLE_SELECT
from .perturb import internal_syntax_leaks, meta_commentary

log = logging.getLogger(__name__)

# extraction classes a strategy is REQUIRED to produce, and the ones it must NOT.
# Semantics per fault type follow the Drift-bench cards plus the author's 2026-08-10 corrections:
#   false_presupposition  -- the false content lives in a PRESUPPOSITION (a premise treated
#                            as a given, e.g. assuming a true requirement is unsatisfiable
#                            and settling for a fallback), not in the request itself
#   irrelevant_information -- a parameter-family flaw: extra stated details that do not
#                            matter for the need (possibly nonexistent), not padding text
#   contextual_irrelevance -- an intention-family flaw: substantial off-topic content that
#                            buries the (fully stated) request
#   indirect_intent       -- task underspecification: the requirements are conveyed but no
#                            request is ever stated (not mere politeness)
CONTRACT: dict[str, str] = {
    "factual_error": "substituted",
    "false_presupposition": "presupposed",
    "insufficient_information": "withheld",
    "irrelevant_information": "extras",
    "referential_ambiguity": "marked",
    "vagueness_subjectivity": "marked",
    "lexical_ambiguity": "ambiguous",
    "syntactic_ambiguity": "ambiguous",
    "contextual_irrelevance": "noise",
    "indirect_intent": "oblique",
}
DIFF_CLASSES = ("substituted", "presupposed", "withheld", "marked", "ambiguous", "extras")


# Sharpened per-strategy instructions, each earned by an observed failure mode in the pilot
# ledger -- never added by taste. The evidence is cited inline.
GEN_HINTS: dict[str, str] = {
    # observed: appended "from the discontinued Waverly Mills brand" (an added premise wearing
    # an FE label) and "30 Pan Am dollars" (the card's example entity bleeding into a currency);
    # 2026-08-10 regeneration: extra_presupposed still the top FE killer -- restate-exactly added
    # round-2 lesson: "does not exist" invited fictional currencies ("20 quatloos") and
    # invented brands -- additions, not substitutions. What admits is the NEAR-MISS: a
    # same-kind variant absent from the domain ("7x6ft" where 7x5ft exists).
    "factual_error": "- your flaw: pick ONE stated requirement that is NOT the price and "
                     "REPLACE its value with a near-miss variant of the same kind -- "
                     "plausible enough to exist, but not actually offered here (like "
                     "asking for a 7x6ft size where 7x5ft is what exists, or an uncommon "
                     "material blend). Do NOT invent currencies, brands, or new details; "
                     "the flawed value must sit exactly where the true value sat. Restate "
                     "every OTHER requirement exactly as it is.\n",
    # the author 2026-08-10: the false content must be PRESUPPOSED (treated as a given), not
    # merely requested. Mechanism is benchmark-portable: doubt satisfiability, settle.
    "false_presupposition": "- your flaw: build on ONE false assumption stated as a given in "
                            "a subordinate clause: assume that one of the true requirements "
                            "cannot be satisfied ('since you probably don't have ...', 'as "
                            "... seems to be unavailable'), and settle for an alternative to "
                            "it instead. Name the doubted true value inside the assumption. "
                            "Keep every other requirement accurately stated.\n",
    # the author 2026-08-10: parameter-family flaw -- an extra detail, not padding text.
    # observed (r0, 4x extra_not_compilable): soft preferences ("nicely packaged if
    # possible") cannot compile to a checkable property -- the detail must be concrete
    "irrelevant_information": "- your flaw: state EVERY true requirement accurately, and ADD "
                              "exactly one extra CONCRETE product property or add-on that "
                              "does not matter for the need -- something specific enough "
                              "that a store could filter by it (a bundled accessory, a "
                              "specific certification or feature), never a soft preference "
                              "like 'nicely packaged'. State it as part of the wish list, "
                              "not as an assumption.\n",
    # the author 2026-08-10: intention-family flaw -- enough off-topic text to genuinely mislead
    "contextual_irrelevance": "- your flaw: begin with three or four sentences about a "
                              "COMPLETELY different topic -- specific and engaging enough to "
                              "be distracting, at least as long as the request itself, and "
                              "mentioning none of the true requirements. Then state the full "
                              "true request accurately. (This flaw may exceed the usual "
                              "two-sentence limit.)\n",
    # the author 2026-08-10: task underspecification, not politeness -- no request stated at all
    "indirect_intent": "- your flaw: NEVER state a request, a desire, or a question. Describe "
                       "your situation, or repeat something you heard ('I've heard that ... "
                       "works well when ...'), mentioning every true requirement value -- "
                       "but ask for nothing: no 'I want', 'I need', 'I'm looking for', "
                       "'find', 'please', 'can you'.\n",
    # observed: generation replaced the PRODUCT with "it" instead of a requirement
    "referential_ambiguity": "- your flaw: state every requirement EXCEPT one; refer to that "
                             "one only as 'that one' / 'the usual one' / 'same as last time', "
                             "never naming its value.\n",
    # example words mirror the Drift-bench card exactly (dimension-neutral, benchmark-
    # portable): "best", "reasonable", "good" -- never dimension-evoking words
    "vagueness_subjectivity": "- your flaw: state every requirement EXCEPT one; describe that "
                              "one only with a subjective word ('best', 'reasonable', "
                              "'good'), never its exact value.\n",
    # observed: generations restated the intent with no ambiguity anywhere in them
    "lexical_ambiguity": "- your flaw: pick ONE requirement and refer to it ONLY with a word "
                         "or phrase that has TWO plausible meanings here (like 'light' -- "
                         "weight or colour?). Never state its exact value.\n",
    # the author 2026-08-10: a reading nobody could mean misleads nobody. Structural recipe
    # added after the regeneration produced flat wish-lists with no second attachment site.
    "syntactic_ambiguity": "- your flaw: mention a small companion item alongside the "
                           "product and place ONE trailing modifier after both, so it "
                           "could cover either or both ('a backdrop and a stand under "
                           "$20' -- does the cap apply to the backdrop, the stand, or "
                           "the pair?). BOTH readings must be requests a real person "
                           "could plausibly mean. Never signal which is intended.\n",
}


# The three types the author flagged as starved (2026-08-10: FE 7/20, lexical 4/20,
# syntactic 0/20) get a deeper retry budget; verification is unchanged.
EXTRA_ATTEMPTS: dict[str, int] = {
    "factual_error": 8,
    "lexical_ambiguity": 8,
    "syntactic_ambiguity": 8,
}


# The two ambiguity strategies need a stronger WRITER: turbo restated the intent with no pun
# in it, three attempts out of three, while the pro model produced genuine polysemy
# ("regular wash") on the first try in the prototype. Generation only -- extraction and
# admission are unchanged, and the deepseek second extractor keeps its independence.
# 2026-08-12: the pin applies only on the original WebShop path. An adapter that renders
# its own intent (see _adapter_renders_intent) uses the configured model for every
# strategy -- the pin existed for a measured reason on WebShop and stays scoped to it.
WRITER_MODEL: dict[str, str] = {
    "lexical_ambiguity": "doubao-seed-2-1-pro-260628",
    "syntactic_ambiguity": "doubao-seed-2-1-pro-260628",
}


class Rejected(Exception):
    """One candidate failed admission; carries the ledger verdict."""

    def __init__(self, verdict: str):
        super().__init__(verdict)
        self.verdict = verdict


# Card text shown to the GENERATOR where the 2026-08-10 revision supersedes the original
# Drift-bench wording. indirect_intent: the card's own example ("I was wondering if it's
# possible to know...") is politeness, which the revised definition rejects -- generations
# imitating it would burn the whole retry budget on request_language_present rejections.
REVISED_CARDS: dict[str, tuple[str, str]] = {
    "indirect_intent": (
        "Conveys the requirements without ever stating a request: the speaker describes "
        "a situation or reports something heard, and the task itself is left unstated.",
        "Original: 'Can you check tomorrow's weather?' -> Rewritten: 'I heard the "
        "forecast keeps flip-flopping about tomorrow.'",
    ),
}


# ------------------------------------------------------------------ stage G
def _adapter_renders_intent(adapter) -> bool:
    """Whether this adapter opts into the generalized prompt family.

    An adapter that renders its own intent block (``describe_intent``) is a ported
    domain: its slot names are internal identifiers and its speaker is not a shopper,
    so the prompts below address "the user", explain slots in plain words, and consult
    the adapter's phrasing before rejecting on string checks. Without the hook, every
    prompt literal and every check matches the original WebShop wording and logic byte
    for byte -- the two families are selected by the adapter in use, never by editions
    of this file.
    """
    return getattr(adapter, "describe_intent", None) is not None


def describe_intent(conditions, adapter=None) -> str:
    """Render the true intent for the generator.

    WebShop intents are attribute/option/price shaped and were projected through those
    three prefixes. Any other benchmark (a repair request, a database query) has different
    slot namespaces, and an unprojected intent reaches the generator EMPTY -- it is then
    asked to corrupt something it cannot see. Adapters may therefore supply their own
    rendering; the WebShop projection is kept as the default so its prompts are unchanged.
    """
    fn = getattr(adapter, "describe_intent", None)
    if fn is not None:
        return fn(conditions)
    phrase = getattr(adapter, "slot_phrase", None)
    feats = [str(v) for s, _, v in conditions if str(s).startswith("attr:")]
    opts = [str(v) for s, _, v in conditions if str(s).startswith("option:")]
    caps = [v for s, _, v in conditions if s == "price_upper"]
    if feats or opts or caps:
        return (f"  required features: {feats}\n"
                f"  option: {opts or ['(none)']}\n"
                f"  price limit: {caps or ['(none)']}")
    lines = []
    for slot, op, value in conditions:
        label = phrase(slot) if phrase else str(slot)
        lines.append(f"  - {label}" if op == "=" else f"  - {label} {op} {value}")
    return "\n".join(lines) or "  (no stated requirements)"


def generation_prompt(conditions, strategy: st.Strategy, *, context: str,
                      attempt: int, adapter=None, voice: str = "a plain shopper's voice",
                      domain_word: str = "shopping") -> str:
    hint = (getattr(adapter, "gen_hints", {}) or {}).get(strategy.id,
                                                        GEN_HINTS.get(strategy.id, ""))
    dyn = getattr(adapter, "dynamic_gen_hint", None)
    if dyn is not None:
        try:
            extra = dyn(strategy.id, conditions)
            if extra:
                hint = (hint + "\n" if hint else "") + extra
        except Exception as exc:
            # OPTIONAL adapter hook supplying an extra generation hint. Falling back to the
            # stock hint is correct; doing it silently is not, because the adapter's hint
            # would then be missing from every prompt with no sign of why.
            log.debug("dynamic hint hook failed for %s: %s", strategy.id, exc)
    # Adapter-supplied cards take precedence: the stock examples are safely out-of-domain
    # for WebShop/telecom/retail, but airline's stock cards ARE flight-booking examples,
    # and the "NEVER copy entities from the example" rule below would then instruct the
    # generator to avoid the very entities the domain requires.
    cards = getattr(adapter, "card_overrides", {}) or {}
    description, example = cards.get(
        strategy.id, REVISED_CARDS.get(strategy.id,
                                       (strategy.description, strategy.example)))
    intent_block = describe_intent(conditions, adapter)
    # the header names the speaker: the original wording for the WebShop path, the
    # domain-neutral one for adapters that render their own intent
    owner = ("The user's TRUE intent:" if _adapter_renders_intent(adapter)
             else "The shopper's TRUE intent:")
    return f"""You are an expert at creating flawed instructions that test an agent's ability to handle
ambiguous or mistaken requests through clarification.

{owner}
{intent_block}
{context}

Apply EXACTLY ONE flaw, of this type and no other:
  {strategy.name} ({strategy.id}): {description}
  Example of the flaw: {example}

Rules:
- one or two sentences, first person, {voice}
- the flaw must be present; every OTHER part of the intent stays accurately stated
- the flaw must fit THIS {domain_word} domain; NEVER copy entities from the example above
- do not reveal that anything is wrong
{hint}
(variation {attempt})
Output only the flawed instruction."""


# ------------------------------------------------------------------ stage X
# The generic diff form. Keys are the fault-type names from the taxonomy (compound keys
# where several fault types share one structural realization) -- never invented terms.
# Used for factual_error and insufficient_information; the remaining fault types have
# targeted questionnaires below.
EXTRACT_SCHEMA = """Output ONLY a JSON object with exactly these keys:
{"factual_error": [[<slot>, <true value>, <stated value>]],
 "false_presupposition | irrelevant_information": [
     {"text": <the added or assumed requirement, quoted>,
      "condition": [<slot>, "=", <value>] or null}],
 "insufficient_information": [<slot>],
 "referential_ambiguity | vagueness_subjectivity | lexical_ambiguity | syntactic_ambiguity": [
     [<slot>, <the phrase standing in for it>]],
 "contextual_irrelevance": <bool>,
 "indirect_intent": <bool>}

Definitions:
  factual_error -- a TRUE requirement stated with a DIFFERENT value in its place.
  false_presupposition | irrelevant_information -- any requirement asserted or assumed that
                 is NOT in the true intent. Use a compilable condition when it is a product
                 property; null when it is pragmatic (a prior conversation, an assumed event).
  insufficient_information -- a true requirement not mentioned and not hinted at.
  referential_ambiguity | vagueness_subjectivity | lexical_ambiguity | syntactic_ambiguity --
                 a true requirement mentioned but WITHOUT its exact value ("that one",
                 "cheap", a two-meaning word, unclear grammar).
  contextual_irrelevance -- substantial content about a completely unrelated topic.
  indirect_intent -- the text never actually asks for anything.
IMPORTANT: naming the product kind ("a shirt", "a top", "clothes", "coverings") is NOT an
added requirement -- a sentence needs a noun. Only concrete product properties count
(features, options, price, brand, condition).
Report ONLY what the flawed text actually does. Empty lists / false are expected."""


# syntactic_ambiguity is NOT here: the unbound form asks for a requirement lacking its
# exact value, and attachment ambiguity states every value -- it has its own form below
# (the 2026-08-10 regeneration killed 0/20 syntactic cells on exactly this mismatch)
UNBOUND_STRATEGIES = ("referential_ambiguity", "vagueness_subjectivity",
                      "lexical_ambiguity")

SYNTACTIC_SCHEMA = """Output ONLY a JSON object:
{"ambiguous_phrase": <the modifier whose attachment is unclear, quoted from the text, or
                      null if the sentence is structurally unambiguous>,
 "target_slot": <the requirement that modifier states, from the list>,
 "attachments": [<what it could describe #1>, <what it could describe #2>],
 "both_attachments_plausible": <false ONLY if one attachment is something no real person
                                could mean; otherwise true>,
 "other_changes": [<slot>]}
    // true requirements stated with a DIFFERENT value, dropped, or left unresolved"""


def syntactic_extraction_prompt(conditions, text: str) -> str:
    vocab = [[s, str(v)] for s, _, v in conditions]
    return (f"A shopper's TRUE intent, as (slot, value) pairs:\n{json.dumps(vocab)}\n\n"
            f"An instruction derived from it:\n{text}\n\n"
            "The sentence may contain a modifier that could GRAMMATICALLY attach to "
            "either of two different things (as in 'a hotel near the museum with free "
            "parking' -- does the parking belong to the hotel or the museum?). Every "
            "value may be stated; what is unclear is which thing one modifier "
            f"describes. Identify it.\n{SYNTACTIC_SCHEMA}")

UNBOUND_SCHEMA = """Output ONLY a JSON object:
{"target_slot": <slot name from the list, or null if every requirement is stated exactly>,
 "surface": <the word/phrase in the text that stands in for it, or null>,
 "exact_value_stated": <true if the text states that requirement's exact value anywhere>,
 "second_reading": <for a genuinely two-way expression: the OTHER plausible value it could
                    mean, else null>,
 "both_readings_plausible": <false ONLY if one of the two meanings is something no real
                             person could intend here; otherwise true>}"""


def unbound_extraction_prompt(conditions, text: str) -> str:
    """The targeted form for mentioned-but-unresolved flaws.

    The generic diff form asks two models to sort the same phenomenon into three adjacent
    bins (withheld / marked / ambiguous) and they split -- 0/3 admissions across four
    strategies on normal graphs. Whether "cheap" is a hint, a mark, or a withholding is a
    taxonomy argument; WHICH requirement is unresolved is a question models agree on.
    """
    vocab = [[s, str(v)] for s, _, v in conditions]
    return (f"A shopper's TRUE intent, as (slot, value) pairs:\n{json.dumps(vocab)}\n\n"
            f"An instruction derived from it:\n{text}\n\n"
            "Exactly one requirement may be referred to WITHOUT its exact value -- via a "
            "pointer ('that one'), a subjective word ('cheap'), a two-meaning word, or "
            f"grammar that leaves it unclear. Identify it.\n{UNBOUND_SCHEMA}")


def parse_unbound(raw: str) -> dict | None:
    obj = _first_json(raw)
    if not isinstance(obj, dict):
        return None
    return {"target_slot": obj.get("target_slot"), "surface": obj.get("surface"),
            "exact_value_stated": bool(obj.get("exact_value_stated")),
            "second_reading": obj.get("second_reading"),
            "both_readings_plausible": bool(obj.get("both_readings_plausible"))}


# ---------------------------------------------------- targeted questionnaires
# One per fault type whose structure the generic diff form cannot carry. Each includes an
# "other_changes" cross-check so the single-fault contract stays enforced.

FP_SCHEMA = """Output ONLY a JSON object:
{"presupposition": <the clause that treats something as a GIVEN, quoted from the text,
                    or null if nothing is assumed>,
 "concerns": [<slot>, <value>] or null,
    // WHICH true requirement the false assumption is about -- either one the text assumes
    // cannot be satisfied, or one it assumes has ALREADY been dealt with
 "assumed_unsatisfiable": [<slot>, <value>] or null,
    // fill only if the text assumes the requirement CANNOT be satisfied
 "fallback": [<slot>, <value>] or null,
    // the alternative value the text settles for instead, if any
 "other_changes": [<slot>]}
    // every OTHER true requirement the text drops, changes, or leaves unresolved"""


# The original single-shape form ("cannot be satisfied, settled for a fallback"), kept
# verbatim: it is what the WebShop path shows its annotators, and its answers never
# contain a "concerns" key, so the settle-only extraction below stays byte-identical.
_FP_SCHEMA_SETTLE_ONLY = """Output ONLY a JSON object:
{"presupposition": <the clause that treats something as a GIVEN, quoted from the text,
                    or null if nothing is assumed>,
 "assumed_unsatisfiable": [<slot>, <value>] or null,
    // the true requirement the text assumes cannot be satisfied
 "fallback": [<slot>, <value>] or null,
    // the alternative value the text settles for instead, if any
 "other_changes": [<slot>]}
    // every OTHER true requirement the text drops, changes, or leaves unresolved"""


FE_SCHEMA = """Output ONLY a JSON object:
{"replaced": [<slot>, <value>] or null,
    // the TRUE requirement the user failed to report
 "stated_instead": <the wrong requirement they named instead, as its identifier from the
                    domain list below if it matches one, else quoted from the text> or null,
 "other_changes": [<slot>]}
    // every OTHER true requirement the text drops, changes, or leaves unresolved"""


def fe_extraction_prompt(conditions, text: str, adapter=None) -> str:
    """Targeted questionnaire for factual_error on identifier-valued domains.

    The generic diff form asks for [slot, true value, stated value]. Where a slot's value is
    an internal identifier the two annotators cannot agree what to put in the third position
    -- one quotes the sentence, one repeats the identifier, and every cell dies as
    extractor_disagreement (measured: 91 of 114 telecom cells). Naming the two requirements
    separately, against the domain's own vocabulary, is a question both can answer the same
    way.
    """
    vocab = [[s, str(v)] for s, _, v in conditions]
    known = ""
    lister = getattr(adapter, "all_condition_values", None)
    if lister is not None:
        try:
            known = ("\nEvery requirement that exists in this domain:\n  "
                     + ", ".join(sorted(lister())) + "\n")
        except Exception:
            known = ""
    note = getattr(adapter, "extraction_note", "")
    return (f"The user's TRUE requirements, as (slot, value) pairs:\n{json.dumps(vocab)}\n"
            f"{known}{note}\n"
            f"An instruction derived from them:\n{text}\n\n"
            "The instruction may report a DIFFERENT requirement in place of a true one. A "
            "paraphrase or synonym of a true requirement is NOT that -- only a genuinely "
            "different requirement counts.\n"
            f"Identify the swap.\n{FE_SCHEMA}")


def _extract_fe(conditions, text: str, llm, *, dual_extract: bool, adapter=None) -> dict:
    """factual_error via the targeted form; returns the standard substituted diff."""
    vocab = {str(s): str(v) for s, _o, v in conditions}

    def one(role):
        obj = _first_json(llm.complete(fe_extraction_prompt(conditions, text, adapter),
                                       role=role))
        if not isinstance(obj, dict):
            raise Rejected("extraction_unparseable")
        return obj

    e1 = one(ROLE_SELECT)
    rep, stated = e1.get("replaced"), e1.get("stated_instead")
    if not rep or not stated:
        raise Rejected("contract_violation:missing_substituted")
    if e1.get("other_changes"):
        raise Rejected("contract_violation:other_requirements_touched")
    slot = _canon(rep[0], vocab, true_value=rep[1] if len(rep) > 1 else None)
    if slot not in vocab:
        raise Rejected("contract_violation:substitution_not_about_true_intent")
    if dual_extract:
        e2 = one(ROLE_JUDGE)
        r2 = e2.get("replaced") or [None]
        s2 = _canon(r2[0], vocab, true_value=r2[1] if len(r2) > 1 else None)
        if not e2.get("stated_instead") or s2 != slot or e2.get("other_changes"):
            raise Rejected("extractor_disagreement")
    ext = _empty_ext()
    ext["substituted"] = [[slot, vocab[slot], str(stated)]]
    return ext


def fp_extraction_prompt(conditions, text: str, adapter=None) -> str:
    """The false-presupposition form.

    The settle-only wording describes ONE shape of presupposition -- "requirement X seems
    unavailable, I'll settle for Y" -- which is what a shopper does. A support caller
    presupposes differently: they assert a past event or a current fact as given ("your
    team already enabled roaming last week"). Annotators asked only about the first shape
    answer "nothing is assumed" and the fault type dies (measured: 51 of 114 telecom
    cells rejected as no_presupposition on texts that plainly contained one). The
    two-shape form therefore belongs to adapters that render their own intent; the
    WebShop path keeps its original question verbatim.
    """
    vocab = [[s, str(v)] for s, _, v in conditions]
    if not _adapter_renders_intent(adapter):
        return (f"A shopper's TRUE intent, as (slot, value) pairs:\n{json.dumps(vocab)}\n\n"
                f"An instruction derived from it:\n{text}\n\n"
                "The instruction may treat a false assumption as a given -- for example, "
                "assuming a requirement cannot be satisfied and settling for something else. "
                f"Identify exactly what is assumed.\n{_FP_SCHEMA_SETTLE_ONLY}")
    note = getattr(adapter, "fp_note", "")
    return (f"The user's TRUE intent, as (slot, value) pairs:\n{json.dumps(vocab)}\n\n"
            f"An instruction derived from it:\n{text}\n\n"
            "The instruction may treat something FALSE as a given: assuming a requirement "
            "cannot be satisfied and settling for something else, OR asserting a past "
            "event or current fact that is not true. Either counts.\n"
            f"{note}"
            f"Identify exactly what is assumed.\n{FP_SCHEMA}")


IRRELEVANT_SCHEMA = """Output ONLY a JSON object:
{"extra_details": [{"text": <the added detail, quoted from the text>,
                    "condition": [<slot>, "=", <value>] or null}],
    // concrete details requested that are NOT in the true intent
 "true_requirements_intact": <true if every true requirement is still stated accurately>,
 "other_changes": [<slot>]}
    // true requirements the text drops, changes, or leaves unresolved"""


def irrelevant_extraction_prompt(conditions, text: str) -> str:
    vocab = [[s, str(v)] for s, _, v in conditions]
    return (f"A shopper's TRUE intent, as (slot, value) pairs:\n{json.dumps(vocab)}\n\n"
            f"An instruction derived from it:\n{text}\n\n"
            "The instruction may request extra details beyond the true intent. Identify "
            "them. Naming the product kind (\"a shirt\", \"a backdrop\") is NOT an extra "
            f"detail -- only concrete added properties count.\n{IRRELEVANT_SCHEMA}")


OFFTOPIC_SCHEMA = """Output ONLY a JSON object:
{"off_topic_passage": <the passage about an unrelated topic, quoted verbatim from the
                       text, or null if there is none>,
 "request_complete": <true if the text ALSO states every true requirement accurately>,
 "other_changes": [<slot>]}
    // true requirements the text drops, changes, or leaves unresolved"""


def offtopic_extraction_prompt(conditions, text: str) -> str:
    vocab = [[s, str(v)] for s, _, v in conditions]
    return (f"A shopper's TRUE intent, as (slot, value) pairs:\n{json.dumps(vocab)}\n\n"
            f"An instruction derived from it:\n{text}\n\n"
            "The instruction may contain a substantial passage about a completely "
            f"unrelated topic. Identify it.\n{OFFTOPIC_SCHEMA}")


NOREQUEST_SCHEMA = """Output ONLY a JSON object:
{"request_stated": <true if the text anywhere asks for something, states a desire, or
                    poses a question that expects action>,
 "values_conveyed": [<slot>],
    // true requirements whose values appear somewhere in the text
 "other_changes": [<slot>]}
    // true requirements stated with a DIFFERENT value or contradicted"""


def norequest_extraction_prompt(conditions, text: str) -> str:
    vocab = [[s, str(v)] for s, _, v in conditions]
    return (f"A shopper's TRUE intent, as (slot, value) pairs:\n{json.dumps(vocab)}\n\n"
            f"A text derived from it:\n{text}\n\n"
            f"Does this text actually request anything?\n{NOREQUEST_SCHEMA}")


def extraction_prompt(conditions, text: str, adapter=None) -> str:
    """The annotator's form.

    ``adapter.extraction_note`` lets a benchmark state what counts as *the same*
    requirement in its own vocabulary. Without it, annotators disagree on paraphrase:
    a telecom fault named ``airplane_mode_on`` restated as "flight mode is turned on" is
    the SAME requirement accurately conveyed, not a substituted one -- and two annotators
    splitting on that produce extractor_disagreement on every cell rather than a finding.

    WebShop slot names carry their values in plain sight, so its form needs neither
    labels nor a note and is kept verbatim; the labelled form belongs to adapters that
    render their own intent.
    """
    vocab = [[s, str(v)] for s, _, v in conditions]
    if not _adapter_renders_intent(adapter):
        return (f"A shopper's TRUE intent, as (slot, value) pairs:\n{json.dumps(vocab)}\n\n"
                f"A FLAWED instruction derived from it:\n{text}\n\n"
                f"Describe exactly how the instruction misrepresents the intent.\n{EXTRACT_SCHEMA}")
    labels = ""
    phrase = getattr(adapter, "slot_phrase", None)
    if phrase is not None:
        labels = ("\nWhat each slot means in plain words:\n"
                  + "\n".join(f"  {s}: {phrase(s)}" for s, _, _ in conditions) + "\n")
    note = getattr(adapter, "extraction_note", "")
    return (f"The user's TRUE intent, as (slot, value) pairs:\n{json.dumps(vocab)}\n"
            f"{labels}{note}\n"
            f"A FLAWED instruction derived from it:\n{text}\n\n"
            f"Describe exactly how the instruction misrepresents the intent.\n{EXTRACT_SCHEMA}")


def _empty_ext() -> dict:
    return {"substituted": [], "presupposed": [], "withheld": [], "marked": [],
            "ambiguous": [], "extras": [], "noise": False, "oblique": False}


def parse_extraction(raw: str) -> dict | None:
    """Map the fault-type-keyed diff form onto the internal classes.

    Keys are matched by the fault-type names they contain, so any arrangement of a
    compound key ("a | b", "a|b", either half alone) lands in the right class.
    """
    obj = _first_json(raw)
    if not isinstance(obj, dict):
        return None
    out = _empty_ext()
    for key, value in obj.items():
        parts = {p.strip().lower() for p in str(key).split("|")}
        if "factual_error" in parts:
            out["substituted"] = value or []
        elif parts & {"false_presupposition", "irrelevant_information"}:
            out["presupposed"] = value or []
        elif "insufficient_information" in parts:
            out["withheld"] = value or []
        elif parts & {"referential_ambiguity", "vagueness_subjectivity",
                      "lexical_ambiguity", "syntactic_ambiguity"}:
            out["marked"] = [[it[0], "unresolved"] for it in (value or [])
                             if isinstance(it, (list, tuple)) and it]
        elif "contextual_irrelevance" in parts:
            out["noise"] = bool(value)
        elif "indirect_intent" in parts:
            out["oblique"] = bool(value)
    return out


def _families(d: dict) -> dict[str, set[str]]:
    """Collapse extraction classes into flaw FAMILIES for agreement purposes.

    marked vs ambiguous is a genuinely fuzzy boundary between models -- the same "light"
    can be labelled either -- and requiring class-exact agreement killed 3/3 lexical
    candidates. Both mean "this slot is mentioned but unresolved", so they agree as a
    family; every other family stays distinct, which keeps cross-family multi-flaw
    detection strict.
    """
    fam: dict[str, set[str]] = {"substituted": set(), "presupposed": set(),
                                "withheld": set(), "unbound": set()}
    for it in d.get("substituted") or []:
        if isinstance(it, (list, tuple)) and it:
            fam["substituted"].add(str(it[0]))
    for _it in d.get("presupposed") or []:
        fam["presupposed"].add("*")          # presence, not slot: extras have no true slot
    for it in d.get("withheld") or []:
        fam["withheld"].add(str(it))
    for it in d.get("marked") or []:
        if isinstance(it, (list, tuple)) and it:
            fam["unbound"].add(str(it[0]))
    for it in _objs(d.get("ambiguous")):
        for pl in _rows(it.get("placements")):
            fam["unbound"].add(str(pl[0]))
    return fam


_CATEGORY_WORDS = ("product type", "category", "type", "kind", "item", "product")


def normalize_extraction(ext: dict, conditions) -> dict:
    """Canonicalize an extraction against the slot vocabulary before judging it.

    Two observed failure modes, both cosmetic: (1) the PHANTOM presupposition -- a sentence
    needs a noun, so "a shirt" gets reported as presupposed product-type although naming the
    category adds no requirement; (2) SLOT RENAMING -- one extractor says
    `attr:polyester cotton`, the other invents `attr:material` for the same substitution.
    Judging un-normalized extractions turned both into disagreements/violations.
    """
    vocab = {str(s): str(v).lower() for s, _o, v in conditions}
    by_value = {v: k for k, v in vocab.items()}

    def canon_slot(name, true_value=None):
        n = str(name)
        if n in vocab:
            return n
        if true_value is not None and str(true_value).lower() in by_value:
            return by_value[str(true_value).lower()]
        low = n.lower()
        for k in vocab:
            kl = k.lower()
            if low in kl or kl in low or vocab[k] in low:
                return k
        return n

    out = dict(ext)
    out["substituted"] = [[canon_slot(it[0], it[1] if len(it) > 1 else None),
                          *list(it)[1:]] for it in _rows(ext.get("substituted"))]
    out["withheld"] = [canon_slot(x) for x in (ext.get("withheld") or [])]
    out["marked"] = [[canon_slot(it[0]), *list(it)[1:]] for it in _rows(ext.get("marked"))]
    keep = []
    for pcond in ext.get("presupposed") or []:
        cond = (pcond or {}).get("condition")
        slot = str(cond[0]).lower() if cond else ""
        if any(w in slot for w in _CATEGORY_WORDS):
            continue                      # phantom: the sentence's noun, not a requirement
        if cond is None and ext.get("noise"):
            continue                      # a story inside declared noise IS the noise
        keep.append(pcond)
    out["presupposed"] = keep
    amb = []
    for a in ext.get("ambiguous") or []:
        pls = [[canon_slot(pl[0]), pl[1] if len(pl) > 1 else None]
               for pl in (a or {}).get("placements") or [] if pl]
        amb.append({**a, "placements": pls})
    out["ambiguous"] = amb
    return out


def extractions_agree(a: dict, b: dict) -> bool:
    """Agreement on flaw families and their target slots; wording differences are fine."""
    fa, fb = _families(a), _families(b)
    for k in fa:
        if bool(fa[k]) != bool(fb[k]):
            return False
    for k in ("substituted", "withheld", "unbound"):
        if fa[k] != fb[k]:
            return False
    return a["noise"] == b["noise"]


# ------------------------------------------------------------------ stage V
def check_contract(strategy_id: str, ext: dict) -> None:
    """Exactly the requested flaw class; everything else empty."""
    want = CONTRACT[strategy_id]
    for cls in DIFF_CLASSES:
        have = bool(ext.get(cls))
        if cls == want and not have:
            raise Rejected(f"contract_violation:missing_{cls}")
        if cls != want and have:
            raise Rejected(f"contract_violation:extra_{cls}")
    if want == "noise" and not ext["noise"]:
        raise Rejected("contract_violation:missing_noise")
    if want == "oblique" and not ext["oblique"]:
        raise Rejected("contract_violation:missing_oblique")
    if want != "noise" and ext["noise"]:
        # accidental off-topic padding alongside a content flaw is a second flaw (k=1)
        raise Rejected("contract_violation:extra_noise")
    if want not in ("noise", "oblique") and ext["oblique"] and strategy_id != "indirect_intent":
        pass  # oblique phrasing alongside a content flaw is tolerated; it hides nothing extra


_WORD = r"(?<!\w){}(?!\w)"


def _present(text: str, value, *, default: bool = True) -> bool:
    """Word-boundary presence, with an explicit polarity for the untestable case.

    The default matters and DIFFERS by caller. For a retained/asserted value, "cannot test"
    must count as present (lenient -- execution still applies). For a WITHHELD value the same
    default silently rejected every short value: the price "30" was "untestable", reported
    present, and every price-withholding sample died. Withheld callers pass default=False.
    """
    v = str(value).strip().lower()
    if len(v) < 2:
        return default
    return re.search(_WORD.format(re.escape(v)), text.lower()) is not None


def _value_atoms(raw) -> list:
    """The human-visible scalars inside a stored condition value.

    Stored values are canonical JSON ('"economy"', '["8926329222","5312063289"]',
    '[{"flight_number": "HAT300", "date": "2024-05-25"}]'). A person states the ATOMS, so
    presence tests must run against those, not against the JSON text. Values that are not
    JSON (telecom's plain fault ids) pass through unchanged.
    """
    if raw is None:
        return []
    try:
        v = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, ValueError):
        return [raw]
    out: list = []

    def walk(x):
        if isinstance(x, list):
            for i in x:
                walk(i)
        elif isinstance(x, dict):
            for i in x.values():
                walk(i)
        elif x is not None and not isinstance(x, bool):
            out.append(x)

    walk(v)
    return out or [raw]


def check_text(conditions, ext: dict, text: str, *, adapter=None, allow_stated_ambiguous: bool = False,
               max_words: int = 130) -> None:
    """String-level cross-checks: the extraction must describe THIS text."""
    if internal_syntax_leaks(text):
        raise Rejected("text:internal_syntax")
    if meta_commentary(text):
        raise Rejected("text:meta_commentary")
    if len(text.split()) > max_words:
        raise Rejected("text:too_long")
    values = {str(s): v for s, _, v in conditions}
    for item in ext.get("substituted") or []:
        slot, true_v, asserted = (list(item) + [None, None])[:3]
        if asserted is not None and not _present(text, asserted):
            # An identifier-valued assertion ("bad_network_preference") never appears in
            # natural text; ask the adapter what a person would call it before rejecting.
            # Only adapters that render their own intent are consulted -- their slot
            # values are internal identifiers. A plain-text domain keeps the strict
            # rejection its samples were admitted under.
            said = None
            phrase = (getattr(adapter, "slot_phrase", None)
                      if _adapter_renders_intent(adapter) else None)
            if phrase is not None:
                try:
                    said = phrase(f"fault:{asserted}") if ":" not in str(asserted) \
                        else phrase(str(asserted))
                except Exception:
                    said = None
            if not (said and said != str(asserted) and _present_loose(text, said)):
                raise Rejected("text:asserted_value_absent")
        if true_v is not None and _present(text, true_v) and str(true_v).lower() != str(asserted).lower():
            raise Rejected("text:true_value_still_present")
    for slot in ext.get("withheld") or []:
        v = values.get(str(slot))
        # Condition values are stored as CANONICAL JSON, so a string slot's value carries
        # its quotes ('"economy"') and a list slot is a JSON array. Comparing that raw form
        # against natural text can never match, which made this guard inert on every
        # JSON-valued domain (retail, airline; telecom stores plain fault ids and was
        # unaffected). Found 2026-08-18 by adversarial review of the airline build. No
        # sample had actually leaked -- the dual extractors and executable admission had
        # caught them all (verified: 0 leaks over retail 50 / telecom 59 / airline 5) --
        # but a check that cannot fire is not a check, so the value is decoded to its
        # human-visible atoms before testing.
        for atom in _value_atoms(v):
            if _present(text, atom, default=False):
                raise Rejected("text:withheld_value_present")
    for item in ext.get("presupposed") or []:
        frag = (item or {}).get("text")
        if frag and len(str(frag).split()) >= 2 and not _present_loose(text, frag):
            raise Rejected("text:presupposed_fragment_absent")
        about = (item or {}).get("about")
        if about and len(about) > 1 and not _present(text, about[1]):
            # The requirement the presupposition is about must be NAMED in the text. Where a
            # slot's value is an internal identifier (a telecom fault, `data_mode_off`) no
            # sentence can ever contain it literally, so the check consults the adapter's
            # plain-words phrasing for that slot before rejecting -- again only for
            # adapters that render their own intent; the plain-text domain rejects as it
            # always did (the doubted true value must be named inside the presupposition).
            said = None
            phrase = (getattr(adapter, "slot_phrase", None)
                      if _adapter_renders_intent(adapter) else None)
            if phrase is not None and about[0]:
                try:
                    said = phrase(str(about[0]))
                except Exception:
                    said = None
            if not (said and _present_loose(text, said)):
                raise Rejected("text:doubted_value_absent")
        alt = (item or {}).get("alternative")
        if alt and len(alt) > 1 and alt[1] and not _present(text, alt[1]):
            raise Rejected("text:fallback_value_absent")
    for item in ext.get("extras") or []:
        frag = (item or {}).get("text")
        if frag and len(str(frag).split()) >= 2 and not _present_loose(text, frag):
            raise Rejected("text:extra_detail_absent")
    # an unresolved requirement's exact value must be ABSENT from the text -- mechanical,
    # so a single extractor error cannot admit a text that states the value outright.
    # Syntactic ambiguity is exempt: the value is stated by definition; what is unclear
    # is which thing owns it.
    unresolved = [str(m[0]) for m in _rows(ext.get("marked"))]
    if not allow_stated_ambiguous:
        unresolved += [str(p[0]) for a in _objs(ext.get("ambiguous"))
                       for p in _rows(a.get("placements"))]
    for slot in unresolved:
        v = values.get(slot)
        if v is not None and _present(text, v, default=False):
            raise Rejected("text:unresolved_value_present")


def _present_loose(text: str, fragment) -> bool:
    words = [w for w in re.findall(r"[a-z0-9]+", str(fragment).lower()) if len(w) > 3]
    if not words:
        return True
    hits = sum(1 for w in words if w in text.lower())
    return hits >= max(1, len(words) // 2)


def _apply(conditions, *, substituted=(), presupposed=(), drop=()):
    conds = []
    subs = {}
    for item in substituted:
        parts = list(item)
        if len(parts) < 3:
            # a malformed extraction must cost ONE attempt (Rejected is caught and
            # retried), not the whole cell via an uncaught ValueError
            raise Rejected("extraction_malformed_substitution")
        subs[str(parts[0])] = parts[2]
    dropset = {str(s) for s in drop}
    for s, op, v in conditions:
        if str(s) in dropset:
            continue
        conds.append((s, op, subs.get(str(s), v)) if str(s) in subs else (s, op, v))
    for extra in presupposed:
        cond = (extra or {}).get("condition")
        if cond and len(cond) == 3:
            value = cond[2]
            if not isinstance(value, (str, int, float)) or isinstance(value, bool):
                # an extractor once emitted `true` as a condition value; a bool reaching
                # WebShop's reward code crashes it ('in <string>' requires string)
                raise Rejected("extraction_bad_condition_value")
            conds.append((str(cond[0]), str(cond[1]), str(value)))
    return tuple(conds)


def admit(strategy_id: str, ext: dict, conditions, node, adapter, session) -> dict:
    """Executable admission. Returns the verified readings + signature detail."""
    truth_gt = node.ground_truth

    def execute(conds):
        try:
            recipe = adapter.compile(dict(node.base), tuple(conds))
            return adapter.execute(recipe, session)
        except (ValueError, TypeError, KeyError, ZeroDivisionError) as exc:
            # a reading built from a malformed extracted value (a list where a string
            # belongs, "around 35 dollars" where a number belongs) must cost ONE attempt,
            # not the cell -- r0 of the 2026-08-10 regeneration lost 2 cells to exactly
            # this as v2:error:ValueError
            raise Rejected(f"admission:reading_not_executable:{type(exc).__name__}")

    want = CONTRACT[strategy_id]
    readings: list[tuple] = []
    detail: dict = {"rule": want, "ok": True, "verdict": "ok"}

    if want == "substituted":
        # 2026-08-10 (the author's >=10 floor): a plausible near-miss substitution often still
        # matches products, and demanding an empty set starved the type at 7/20. The
        # requirement is now CONSEQUENCE: following the wrong value must lead to a
        # different product set than the truth. lit_card is recorded, so dead-end
        # (lit_card 0) and detour (lit_card > 0) substitutions remain separable.
        lit = _apply(conditions, substituted=[tuple(x) for x in ext["substituted"]])
        gt = execute(lit)
        if gt.hash == truth_gt.hash:
            raise Rejected("admission:substitution_without_consequence")
        readings = [lit]
        detail["lit_card"] = gt.cardinality()

    elif want == "presupposed":
        # false_presupposition: the text assumes a true requirement unsatisfiable and
        # settles for a fallback. The catalog refutes the premise (the true intent,
        # including the doubted value, is satisfiable -- true_card below is its proof),
        # and following the premise must lead somewhere other than the truth.
        claim = (ext["presupposed"] or [{}])[0]
        about = claim.get("about")
        if not about:
            raise Rejected("admission:no_presupposition")
        slot = str(about[0])
        alt = claim.get("alternative")
        if alt and len(alt) > 1 and alt[1]:
            lit = _apply(conditions, substituted=[(slot, None, str(alt[1]))])
        else:
            lit = _apply(conditions, drop=[slot])
            if not lit:
                raise Rejected("admission:nothing_retained")
        gt = execute(lit)
        if gt.hash == truth_gt.hash:
            raise Rejected("admission:presupposition_without_consequence")
        readings = [lit]
        detail["lit_card"] = gt.cardinality()

    elif want == "extras":
        # irrelevant_information: heeding the extra detail must strictly narrow the
        # candidates (possibly to zero, when the detail names something nonexistent) --
        # otherwise the "irrelevant" detail provably changed nothing and the flaw is vacuous
        compilable = [e for e in ext["extras"] if (e or {}).get("condition")]
        if not compilable:
            # A closed-vocabulary domain has no way to express an irrelevant extra AS a
            # condition: a telecom intent is a set of faults, and "a 25W charging brick"
            # is not one of them, so the extra can never compile. Rejecting here would
            # zero this fault type on such domains (measured: 83 of 114 telecom cells)
            # even though the generated text is exactly the intended flaw.
            # pipeline-v2.md already contemplates admitting a pragmatic extra "admitted but
            # labelled"; that is what this does, and the label travels with the sample so
            # the executable and non-executable strata are never pooled silently.
            if not getattr(adapter, "extras_may_be_pragmatic", False):
                raise Rejected("admission:extra_not_compilable")
            if not ext["extras"]:
                raise Rejected("admission:no_extras")
            return {"readings": [],
                    "signature": {"verdict": "ok_pragmatic_extra",
                                  "executable": False,
                                  "readings": 0,
                                  "note": "extra detail is not expressible as a condition "
                                          "in this domain; string checks only"}}
        lit = _apply(conditions, presupposed=compilable)
        gt = execute(lit)
        lit_set = gt.monotonic_view()
        truth_set = truth_gt.monotonic_view()
        if lit_set is not None and truth_set is not None:
            if not (lit_set <= truth_set and lit_set != truth_set):
                raise Rejected("admission:extra_did_not_narrow")
        elif gt.hash == truth_gt.hash:
            raise Rejected("admission:extra_did_not_narrow")
        readings = [lit]
        detail["lit_card"] = gt.cardinality()

    elif want in ("withheld", "marked"):
        drop = (list(ext["withheld"]) if want == "withheld"
                else [m[0] for m in ext["marked"]])
        lit = _apply(conditions, drop=drop)
        if not lit:
            raise Rejected("admission:nothing_retained")
        gt = execute(lit)
        truth_set = truth_gt.monotonic_view()
        lit_set = gt.monotonic_view()
        if truth_set is not None and lit_set is not None:
            if not (lit_set >= truth_set and lit_set != truth_set):
                raise Rejected("admission:withhold_did_not_widen")
        elif gt.hash == truth_gt.hash:
            raise Rejected("admission:withhold_did_not_change_answer")
        readings = [lit]
        detail["lit_card"] = gt.cardinality()

    elif want == "ambiguous":
        amb = (ext["ambiguous"] or [{}])[0]
        placements = amb.get("placements") or []
        if len(placements) < 2:
            raise Rejected("admission:fewer_than_two_readings")
        sets, cards = [], []
        for slot, value in placements[:2]:
            if value is None:
                lit = _apply(conditions, drop=[slot])
            else:
                lit = _apply(conditions, substituted=[(slot, None, str(value))])
            gt = execute(lit)
            readings.append(lit)
            sets.append(gt)
            cards.append(gt.cardinality())
        if sets[0].hash == sets[1].hash:
            raise Rejected("admission:readings_agree")           # ambiguity without consequence
        if not any(g.hash == truth_gt.hash for g in sets):
            # one reading must be the truth, or the text misstates rather than ambiguates
            raise Rejected("admission:no_reading_matches_truth")
        # NOTE 2026-08-10: an earlier empty-reading rejection here bound ONLY lexical
        # (a syntactic second reading is a superset, never empty) and starved it; a
        # dead-end second sense still requires the agent to resolve WHICH sense is meant,
        # and consequence is already enforced by sets-differ + one-reading-is-truth
        detail["reading_cards"] = cards

    else:   # noise / oblique
        lit = tuple(conditions)
        gt = execute(lit)
        if gt.hash != truth_gt.hash:
            raise Rejected("admission:noise_changed_answer")
        readings = [lit]
        detail["lit_card"] = gt.cardinality()

    detail["true_card"] = truth_gt.cardinality()
    detail["readings"] = len(readings)
    return {"readings": [list(map(list, r)) for r in readings], "signature": detail}


def _extract_unbound(strategy, conditions, text, llm, *, dual_extract: bool) -> dict:
    """Targeted extraction for the unbound family, mapped back into the generic classes."""
    vocab = {str(s): str(v) for s, _o, v in conditions}
    u1 = parse_unbound(llm.complete(unbound_extraction_prompt(conditions, text),
                                    role=ROLE_SELECT))
    if u1 is None:
        raise Rejected("extraction_unparseable")
    slot = u1.get("target_slot")
    if not slot or str(slot) not in vocab:
        raise Rejected("contract_violation:no_unresolved_requirement")
    if u1.get("exact_value_stated") and strategy.id != "syntactic_ambiguity":
        # For deictic/vague/lexical flaws a stated exact value means no flaw happened. For
        # ATTACHMENT ambiguity the value is stated by definition -- what is unclear is which
        # thing owns it, so statedness is expected there, not disqualifying.
        raise Rejected("contract_violation:value_actually_stated")
    u2 = None
    if dual_extract:
        u2 = parse_unbound(llm.complete(unbound_extraction_prompt(conditions, text),
                                        role=ROLE_JUDGE))
        if u2 is None or str(u2.get("target_slot")) != str(slot):
            raise Rejected("extractor_disagreement")
        if u2.get("exact_value_stated") and strategy.id != "syntactic_ambiguity":
            # the second extractor's statedness verdict counts too -- one extractor
            # error must not admit a text that states the value outright
            raise Rejected("contract_violation:value_actually_stated")
    if strategy.id == "lexical_ambiguity":
        # a reading nobody could mean misleads nobody (ruling 2026-08-10, the "under $30
        # iPhone" case). Loosened 2026-08-10 per the author: BOTH annotators must deny
        # plausibility to reject -- a single conservative "false" no longer kills the cell
        p1 = bool(u1.get("both_readings_plausible"))
        p2 = bool(u2.get("both_readings_plausible")) if u2 is not None else p1
        if not (p1 or p2):
            raise Rejected("admission:implausible_reading")
    slot = str(slot)
    ext = _empty_ext()
    if strategy.id == "syntactic_ambiguity":
        # reading 1: the detail belongs where the truth puts it; reading 2: it belongs to
        # the other thing, i.e. the requirement is absent
        ext["ambiguous"] = [{"surface": u1.get("surface"),
                             "placements": [[slot, vocab[slot]], [slot, None]]}]
    elif strategy.id == "lexical_ambiguity":
        second = u1.get("second_reading")
        placements = [[slot, vocab[slot]]]
        placements.append([slot, str(second)] if second else [slot, None])
        ext["ambiguous"] = [{"surface": u1.get("surface"), "placements": placements}]
    else:
        kind = "vague" if strategy.id == "vagueness_subjectivity" else "deictic"
        ext["marked"] = [[slot, kind]]
    return ext


def _canon(name, vocab: dict, true_value=None) -> str:
    """Map an extractor-invented slot name onto the true vocabulary (by name, then by
    the true value it carries, then by containment)."""
    n = str(name)
    if n in vocab:
        return n
    if true_value is not None:
        for k, v in vocab.items():
            if str(true_value).strip().lower() == str(v).strip().lower():
                return k
    low = n.lower()
    for k, v in vocab.items():
        if low in k.lower() or k.lower() in low or str(v).lower() in low:
            return k
    return n


def _content_words(fragment) -> set:
    return {w for w in re.findall(r"[a-z0-9]+", str(fragment).lower()) if len(w) > 3}


def _texts_overlap(a, b) -> bool:
    wa, wb = _content_words(a), _content_words(b)
    if not wa or not wb:
        return True
    return len(wa & wb) >= max(1, min(len(wa), len(wb)) // 2)


def _span_in_text(span: str, text: str) -> bool:
    norm = lambda t: " ".join(re.findall(r"[a-z0-9]+", str(t).lower()))  # noqa: E731
    head = " ".join(norm(span).split()[:5])
    return bool(head) and head in norm(text)


def _extract_fp(conditions, text: str, llm, *, dual_extract: bool, adapter=None) -> dict:
    """false_presupposition: a premise treated as a given -- the text assumes a true
    requirement unsatisfiable and settles for a fallback. The premise is refutable by
    execution: the true intent (including the doubted value) is satisfiable."""
    vocab = {str(s): str(v) for s, _o, v in conditions}

    def one(role):
        obj = _first_json(llm.complete(fp_extraction_prompt(conditions, text, adapter), role=role))
        if not isinstance(obj, dict):
            raise Rejected("extraction_unparseable")
        return obj

    e1 = one(ROLE_SELECT)
    # A presupposition has two shapes: the requirement cannot be met (a shopper settling for
    # an alternative), or the requirement has ALREADY been met (a support caller who believes
    # the problem was fixed). Only the first was expressible before, which is why the type
    # scored 0% on both tau2 domains while its generations were correct.
    about = e1.get("assumed_unsatisfiable") or e1.get("concerns")
    if not e1.get("presupposition") or not about:
        raise Rejected("contract_violation:no_presupposition")
    if e1.get("other_changes"):
        raise Rejected("contract_violation:other_requirements_touched")
    slot = _canon(about[0], vocab, true_value=about[1] if len(about) > 1 else None)
    if slot not in vocab:
        raise Rejected("contract_violation:presupposition_not_about_true_intent")
    if dual_extract:
        e2 = one(ROLE_JUDGE)
        a2 = e2.get("assumed_unsatisfiable") or e2.get("concerns") or [None]
        s2 = _canon(a2[0], vocab, true_value=a2[1] if len(a2) > 1 else None)
        if not e2.get("presupposition") or s2 != slot or e2.get("other_changes"):
            raise Rejected("extractor_disagreement")
    fb = e1.get("fallback")
    fallback = [slot, str(fb[1])] if (fb and len(fb) > 1 and fb[1]) else None
    ext = _empty_ext()
    ext["presupposed"] = [{"text": str(e1["presupposition"]), "condition": None,
                           "about": [slot, vocab[slot]], "alternative": fallback}]
    return ext


def _extract_extras(conditions, text: str, llm, *, dual_extract: bool) -> dict:
    """irrelevant_information: extra stated details that do not matter for the need."""

    def usable(obj):
        out = []
        for e in obj.get("extra_details") or []:
            # SHAPE TOLERANCE (2026-08-19). The prompt asks for
            # {"text": ..., "condition": ...} but the model sometimes returns a bare
            # string. That is a formatting slip, not a missing detail -- the string IS
            # the extra detail -- so coerce it. Until this fix the bare string reached
            # `e.get("text")` and raised AttributeError, which the worker's generic
            # handler tallied as a verdict right beside genuine rejections. A crash and
            # a considered rejection must never be indistinguishable.
            if isinstance(e, str):
                e = {"text": e, "condition": None}
            if not isinstance(e, dict) or not e.get("text"):
                continue
            cond = e.get("condition")
            cslot = str(cond[0]).lower() if cond else ""
            if any(w in cslot for w in _CATEGORY_WORDS):
                cond = None      # phantom: the sentence's noun compiled into a condition
            out.append({"text": str(e["text"]), "condition": cond})
        return out

    def one(role):
        obj = _first_json(llm.complete(irrelevant_extraction_prompt(conditions, text),
                                       role=role))
        if not isinstance(obj, dict):
            raise Rejected("extraction_unparseable")
        return obj

    e1 = one(ROLE_SELECT)
    extras = usable(e1)
    if not extras:
        raise Rejected("contract_violation:no_extra_detail")
    if not e1.get("true_requirements_intact") or e1.get("other_changes"):
        raise Rejected("contract_violation:other_requirements_touched")
    if dual_extract:
        e2 = one(ROLE_JUDGE)
        x2 = usable(e2)
        if (not x2 or not e2.get("true_requirements_intact") or e2.get("other_changes")
                or not _texts_overlap(extras[0]["text"], x2[0]["text"])):
            raise Rejected("extractor_disagreement")
    ext = _empty_ext()
    ext["extras"] = extras
    return ext


_MIN_OFFTOPIC_WORDS = 25


def _extract_offtopic(conditions, text: str, llm, *, dual_extract: bool) -> dict:
    """contextual_irrelevance: a substantial off-topic passage burying an intact request.
    'Substantial' is mechanical: >= _MIN_OFFTOPIC_WORDS words and no shorter than the
    request it buries; 'off-topic' is mechanical too: no true value appears inside it."""

    def one(role):
        obj = _first_json(llm.complete(offtopic_extraction_prompt(conditions, text),
                                       role=role))
        if not isinstance(obj, dict):
            raise Rejected("extraction_unparseable")
        return obj

    e1 = one(ROLE_SELECT)
    span = str(e1.get("off_topic_passage") or "")
    span_words = len(span.split())
    if span_words < _MIN_OFFTOPIC_WORDS:
        raise Rejected("contract_violation:off_topic_too_short")
    if not e1.get("request_complete") or e1.get("other_changes"):
        raise Rejected("contract_violation:request_not_intact")
    if not _span_in_text(span, text):
        raise Rejected("extraction_bad_span")
    if span_words < len(text.split()) - span_words:
        raise Rejected("contract_violation:off_topic_shorter_than_request")
    for _s, _o, v in conditions:
        # mechanical request-intact check: every value present somewhere in the text,
        # and absent from the off-topic span => present in the request part
        if not _present_loose(text, v):
            raise Rejected("contract_violation:requirement_not_conveyed")
        if _present(span, v, default=False):
            raise Rejected("contract_violation:off_topic_mentions_requirement")
    if dual_extract:
        e2 = one(ROLE_JUDGE)
        span2 = str(e2.get("off_topic_passage") or "")
        if (len(span2.split()) < _MIN_OFFTOPIC_WORDS or not e2.get("request_complete")
                or e2.get("other_changes") or not _texts_overlap(span, span2)):
            raise Rejected("extractor_disagreement")
    ext = _empty_ext()
    ext["noise"] = True
    ext["offtopic_span"] = span
    return ext


_REQUEST_PHRASES = (
    "i want", "i need", "i'd like", "i would like", "i'm looking", "i am looking",
    "looking for", "searching for", "in the market for", "find", "get me", "buy",
    "purchase", "can you", "could you", "would you", "will you", "please",
    "help me", "recommend", "suggest", "show me", "share with", "wondering if",
    "any suggestions",
)


def _extract_norequest(conditions, text: str, llm, *, dual_extract: bool) -> dict:
    """indirect_intent: task underspecification -- every requirement value conveyed,
    no request ever stated. The no-request property is checked mechanically first."""
    low = re.sub(r"\s+", " ", text.lower())
    for phrase in _REQUEST_PHRASES:
        if re.search(_WORD.format(re.escape(phrase)), low):
            raise Rejected("contract_violation:request_language_present")
    for _s, _o, v in conditions:
        if not _present_loose(text, v):
            raise Rejected("contract_violation:requirement_not_conveyed")

    def one(role):
        obj = _first_json(llm.complete(norequest_extraction_prompt(conditions, text),
                                       role=role))
        if not isinstance(obj, dict):
            raise Rejected("extraction_unparseable")
        return obj

    e1 = one(ROLE_SELECT)
    if e1.get("request_stated") or e1.get("other_changes"):
        raise Rejected("contract_violation:request_stated")
    if dual_extract:
        e2 = one(ROLE_JUDGE)
        if e2.get("request_stated") or e2.get("other_changes"):
            raise Rejected("extractor_disagreement")
    ext = _empty_ext()
    ext["oblique"] = True
    return ext


def _extract_syntactic(conditions, text: str, llm, *, dual_extract: bool) -> dict:
    """syntactic_ambiguity: attachment ambiguity -- every value stated, ownership of one
    modifier unclear. Reading 1 binds the modifier to its requirement (the truth);
    reading 2 attaches it elsewhere, leaving the requirement unbound."""
    vocab = {str(s): str(v) for s, _o, v in conditions}

    def one(role):
        obj = _first_json(llm.complete(syntactic_extraction_prompt(conditions, text),
                                       role=role))
        if not isinstance(obj, dict):
            raise Rejected("extraction_unparseable")
        return obj

    e1 = one(ROLE_SELECT)
    slot = e1.get("target_slot")
    if not e1.get("ambiguous_phrase") or not slot:
        raise Rejected("contract_violation:no_unclear_attachment")
    slot = _canon(slot, vocab)
    if slot not in vocab:
        raise Rejected("contract_violation:no_unclear_attachment")
    if e1.get("other_changes"):
        raise Rejected("contract_violation:other_requirements_touched")
    p1 = bool(e1.get("both_attachments_plausible"))
    p2 = p1
    if dual_extract:
        e2 = one(ROLE_JUDGE)
        s2 = _canon(e2.get("target_slot") or "", vocab)
        if not e2.get("ambiguous_phrase") or s2 != slot or e2.get("other_changes"):
            raise Rejected("extractor_disagreement")
        p2 = bool(e2.get("both_attachments_plausible"))
    if not (p1 or p2):      # both annotators must deny plausibility to reject
        raise Rejected("admission:implausible_reading")
    ext = _empty_ext()
    ext["ambiguous"] = [{"surface": e1.get("ambiguous_phrase"),
                         "placements": [[slot, vocab[slot]], [slot, None]]}]
    return ext


# ------------------------------------------------------------------ the loop
def build_v2_mask(graph, strategy: st.Strategy, *, llm, adapter, executor, session,
                  context: str = "", dual_extract: bool = True,
                  attempts: int = 4, trace: list | None = None) -> tuple[dict, str]:
    """One admitted (mask, query) for this (graph, strategy) -- or Rejected with the last verdict.

    ``trace``, when given, collects every attempt as {text, verdict, extraction} -- including
    the REJECTED ones, which the ledger otherwise reduces to a verdict string. Needed to show
    a human what a removed low-quality generation actually looked like.
    """
    node = graph.root
    conditions = tuple(node.conditions)
    last = "no_attempts"
    attempts = max(attempts, EXTRA_ATTEMPTS.get(strategy.id, 0))
    for attempt in range(1, attempts + 1):
        ext = None      # never let a failed attempt's trace show the PREVIOUS extraction
        text = llm.complete(
            generation_prompt(conditions, strategy, context=context, attempt=attempt,
                              adapter=adapter,
                              voice=getattr(adapter, "generation_voice",
                                            "a plain shopper's voice"),
                              domain_word=getattr(adapter, "generation_domain_word",
                                                  "shopping")),
            role=ROLE_RENDER,
            # the stronger-writer pin was measured on the WebShop path; adapters that
            # render their own intent use the configured model for every strategy
            model=(None if _adapter_renders_intent(adapter)
                   else WRITER_MODEL.get(strategy.id))).strip().strip('"')
        try:
            if strategy.id in UNBOUND_STRATEGIES:
                ext = _extract_unbound(strategy, conditions, text, llm,
                                       dual_extract=dual_extract)
            elif strategy.id == "syntactic_ambiguity":
                ext = _extract_syntactic(conditions, text, llm, dual_extract=dual_extract)
            elif strategy.id == "false_presupposition":
                ext = _extract_fp(conditions, text, llm, dual_extract=dual_extract, adapter=adapter)
            elif strategy.id == "irrelevant_information":
                ext = _extract_extras(conditions, text, llm, dual_extract=dual_extract)
            elif strategy.id == "contextual_irrelevance":
                ext = _extract_offtopic(conditions, text, llm, dual_extract=dual_extract)
            elif strategy.id == "indirect_intent":
                ext = _extract_norequest(conditions, text, llm, dual_extract=dual_extract)
            elif (strategy.id == "factual_error"
                  and getattr(adapter, "fe_targeted", False)):
                ext = _extract_fe(conditions, text, llm, dual_extract=dual_extract,
                                  adapter=adapter)
            else:   # factual_error, insufficient_information: the generic diff form
                ext = parse_extraction(llm.complete(extraction_prompt(conditions, text, adapter),
                                                    role=ROLE_SELECT))
                if ext is None:
                    raise Rejected("extraction_unparseable")
                ext = normalize_extraction(ext, conditions)
                if dual_extract:
                    ext2 = parse_extraction(llm.complete(extraction_prompt(conditions, text, adapter),
                                                         role=ROLE_JUDGE))
                    if ext2 is None or not extractions_agree(
                            ext, normalize_extraction(ext2, conditions)):
                        raise Rejected("extractor_disagreement")
                check_contract(strategy.id, ext)
            check_text(conditions, ext, text, adapter=adapter,
                       allow_stated_ambiguous=(strategy.id == "syntactic_ambiguity"),
                       max_words=(220 if strategy.id == "contextual_irrelevance" else 130))
            admitted = admit(strategy.id, ext, conditions, node, adapter, session)
        except Rejected as exc:
            last = exc.verdict
            if trace is not None:
                trace.append({"attempt": attempt, "text": text, "verdict": exc.verdict,
                              "extraction": ext})
            continue

        hidden = sorted({str(x[0]) for x in ext["substituted"]}
                        | {str(s) for s in ext["withheld"]}
                        | {str(m[0]) for m in ext["marked"]}
                        # an ambiguity's target is hidden regardless of whether the second
                        # reading carries a value: the agent must resolve WHICH reading
                        # is meant, so GRIP/Oracle treat the slot as unrecovered
                        | {str(p[0]) for a in _objs(ext["ambiguous"])
                           for p in _rows(a.get("placements"))})
        mask = {
            "pipeline": "v2",
            "strategy_id": strategy.id, "family": strategy.family,
            "mask_kind": strategy.mask_kind, "kinds": [strategy.mask_kind],
            "components": [strategy.id],
            # extras counts as falsify, not noise: heeding the extra detail CHANGES the
            # literal answer (strict narrowing), while noise semantics promise the goal
            # untouched -- the user-side correction is "that detail doesn't matter"
            "governing_kind": ("falsify" if CONTRACT[strategy.id] in ("substituted",
                                                                      "presupposed", "extras")
                               else "withhold" if CONTRACT[strategy.id] in ("withheld", "marked",
                                                                            "ambiguous")
                               else "noise"),
            "substituted": ext["substituted"], "presupposed": ext["presupposed"],
            "withheld": ext["withheld"], "marked": ext["marked"],
            "ambiguous": ext["ambiguous"], "extras": ext.get("extras", []),
            "noise": ext["noise"], "oblique": ext["oblique"],
            "offtopic_span": ext.get("offtopic_span"),
            "retained": sorted({str(s) for s, _, _ in conditions} - set(hidden)),
            "hidden_slots": hidden,
            "literal_readings": admitted["readings"],
            "extractor_agreement": bool(dual_extract),
            "attempt": attempt,
        }
        if trace is not None:
            trace.append({"attempt": attempt, "text": text, "verdict": "ok",
                          "extraction": ext})
        mask["spec_id"] = content_hash(strategy.id, mask["substituted"], mask["presupposed"],
                                       mask["withheld"], mask["marked"], mask["ambiguous"],
                                       mask["extras"], mask["noise"], mask["oblique"])
        return {"mask": mask, "query": text, "signature": admitted["signature"]}, "ok"
    raise Rejected(last)


__all__ = ["build_v2_mask", "Rejected", "CONTRACT", "extractions_agree", "parse_extraction",
           "check_contract", "check_text", "admit"]
