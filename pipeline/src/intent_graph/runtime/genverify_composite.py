"""Composite perturbations: K faults from the 10-type pool woven into ONE query.

the author 2026-08-25: "randomly select 2 and 4 from the 10 fault type pool ... do the
verification all the same step by step, and the later process would be the same."
And: "it's okay if there is no true intent at all" -- the min_retained floor is waived
for these subsets; a draw may corrupt every stated detail (the runtime's pivot path
keeps such episodes scoreable).

Design:
  * the DRAW is uniform without replacement over the 10 supported types, redrawn only on
    three structural rules (below). The realized mix is reported, never hidden.
  * GENERATION is one pass: a single prompt carries all K fault cards, because a fault
    checked in isolation and then paraphrased by a later rendering pass can be silently
    repaired after its check. The final text is the only artifact the agent sees, so the
    final text is what every check runs against.
  * VERIFICATION is the single-fault machinery, applied jointly: the generic dual
    extraction (its diff form was always a flaw-counter), the specialist questionnaires
    for the types whose structure the generic form cannot carry, check_text's string
    cross-checks, and admit()'s executable consequence test PER COMPONENT. The only
    checks not applied are the single-fault EXCLUSIVITY rules ("everything else empty",
    "other_changes", "every value present") -- for a composite those are replaced by the
    composite contract, which accounts for every channel against the drawn combo.

Redraw rules (structural, not taste):
  1. detail-consuming faults (factual_error, insufficient_information, the four
     ambiguity/mark types) <= number of stated details -- each needs its OWN detail.
     No min_retained floor (ruling 2026-08-25).
  2. at most ONE unbound-family fault (referential/vagueness/lexical/syntactic) per
     combo: the extractors' family-level agreement is exactly the boundary models
     cannot attribute across two simultaneously blurred slots (the reason the
     specialist questionnaires exist), and the unbound prompt's "exactly one
     requirement" framing must stay true.
  3. false_presupposition and irrelevant_information never co-occur: the diff form
     itself merges them into one channel ("false_presupposition | irrelevant_information"),
     so a text carrying both is unattributable by construction.
"""

from __future__ import annotations

import logging

from . import strategies as st
from .agents import _first_json
from .genverify import (
    CONTRACT,
    GEN_HINTS,
    REVISED_CARDS,
    _MIN_OFFTOPIC_WORDS,
    _REQUEST_PHRASES,
    _WORD,
    Rejected,
    _adapter_renders_intent,  # selects the prompt family by the adapter in use
    _apply,
    _canon,
    _empty_ext,
    _extract_unbound,
    _present,
    _value_atoms,
    _present_loose,
    _span_in_text,
    _texts_overlap,
    admit,
    check_text,
    describe_intent,        # adapter-aware intent rendering
    fe_extraction_prompt,   # domain-aware FE question (adapter-rendered intents)
    fp_extraction_prompt,
    irrelevant_extraction_prompt,
    norequest_extraction_prompt,
    offtopic_extraction_prompt,
    syntactic_extraction_prompt,
)
from .llm import ROLE_JUDGE, ROLE_RENDER, ROLE_SELECT

import re

log = logging.getLogger(__name__)

# Two prompt families share this module: the original wording for an adapter with no
# describe_intent hook, and the generalized adapter-mediated wording for adapters that
# render their own intent. Wherever the two families diverge, the branch is selected by
# _adapter_renders_intent(adapter) -- by the adapter in use, never by a copy of this file.

# ADAPTER HOOK COVERAGE (verified 2026-08-26 against the fork's single-fault path).
# Used directly here: card_overrides, gen_hints, dynamic_gen_hint, generation_voice,
#   generation_domain_word, describe_intent, slot_phrase, spoken_value,
#   extras_may_be_pragmatic.
# Honoured transitively, because we call the fork's own prompt builders with `adapter`:
#   all_condition_values + extraction_note (inside fe_extraction_prompt), fp_note (inside
#   fp_extraction_prompt).
# Deliberately NOT honoured: `fe_targeted`. The fork consults it to decide whether a
# SINGLE-fault run uses the targeted FE questionnaire; neither tau2 adapter sets it, so
# their single-fault runs use the generic diff form. A composite cannot: the generic form
# asks two models for four simultaneous judgements and their joint agreement collapses
# (~24% measured on WebShop). Composites therefore ALWAYS use the targeted extractor,
# which is strictly the more specific instrument, never the weaker one.
REPEATABLE = ("factual_error", "irrelevant_information",
              "insufficient_information")

CONSUMING = ("factual_error", "insufficient_information", "referential_ambiguity",
             "vagueness_subjectivity", "lexical_ambiguity", "syntactic_ambiguity")
UNBOUND_FAMILY = ("referential_ambiguity", "vagueness_subjectivity",
                  "lexical_ambiguity", "syntactic_ambiguity")

# DOMAIN-VIABLE POOLS, taken from each benchmark's OWN single-fault admission rates --
# not guessed. A composite admits only if every component admits, so a pool containing a
# type the domain cannot render is a pool of guaranteed rejections.
#   airline (r2 run, 216 cells/24 trees): the three ambiguity types sit at 0.08-0.12 and
#     drag every draw containing them to near-zero; the other seven clear 0.20.
#   retail (release set, 98 trees): weakest type is lexical at 0.39, so the full pool
#     stands.
# Measured 2026-08-26; re-measure if a domain's single-fault build changes.
# A domain with no entry here (WebShop) keeps the full supported pool.
DOMAIN_POOLS: dict[str, tuple[str, ...]] = {
    "tau2_airline": ("contextual_irrelevance", "indirect_intent", "factual_error",
                     "irrelevant_information", "false_presupposition",
                     "vagueness_subjectivity", "insufficient_information"),
    # retail: same treatment, added 2026-08-27 after k=4 returned 0/400 cells on the FULL
    # ten-type pool while airline (restricted) succeeded. Its failures were diffuse across
    # exactly the excluded family -- implausible_reading, no_unclear_attachment,
    # no_unresolved_requirement, extractor_disagreement:syntactic. Retail's single-fault
    # rates are healthy for these types in ISOLATION (syntactic 0.74, referential 0.96);
    # they are unusable in COMPOSITION because their verification asks which of several
    # simultaneously-corrupted slots the writer meant.
    # RETAIL, corrected 2026-08-27 from the k=2 corpus rather than from failure
    # histograms. Of 200 admitted retail k=2 samples the fault mix is
    # irrelevant_information 183, indirect_intent 107, contextual_irrelevance 83,
    # factual_error 23, false_presupposition 4 -- and ZERO for the other five types.
    # Earlier restrictions removed irrelevant_information and indirect_intent because
    # they dominated the k=4 FAILURE histograms; they dominate those histograms because
    # they appear in nearly every draw, not because they block. The pool is therefore the
    # four types that demonstrably compose here.
    "tau2_retail_shop": ("irrelevant_information", "indirect_intent",
                         "contextual_irrelevance", "factual_error"),
}

# Domains where the k>=3 pair bans are lifted (ruling 2026-08-26): airline's viable pool is
# small enough that banning indirect x extras would leave too few k=4 combinations.
# retail joins 2026-08-27: with its pool down to five types, rule 5 (no FE x FP at
# k>=3) killed three of the five k=4 combinations and both survivors required
# vagueness_subjectivity -- which is inapplicable on trees with no subjective-capable
# slot, so every draw returned "no feasible combo".
BANS_LIFTED: frozenset = frozenset({"tau2_airline", "tau2_retail_shop"})


def _domain_key(adapter) -> str:
    return str(getattr(adapter, "ENV_KEY", "") or "").replace("_desk", "")


# ------------------------------------------------------------------- the draw
def draw_combo(rng, conditions, k: int, *, adapter=None, droppable=None,
               alts=None, max_tries: int = 200) -> list[st.Strategy] | None:
    """Uniform k-draw from the supported pool, redrawn on the three structural rules.

    Per-strategy applicability (ordered slot for vagueness, a markable non-attr slot for
    the mark types) is checked with min_retained=0 -- the floor is waived (the author
    2026-08-25), applicability is not.
    """
    dom = _domain_key(adapter)
    allowed = DOMAIN_POOLS.get(dom)
    pool = [s for s in st.SUPPORTED if allowed is None or s.id in allowed]
    if not pool:
        return None
    thin = len(pool) <= k          # too few distinct types to fill k slots
    n = len(tuple(conditions))
    for _ in range(max_tries):
        if thin:
            # allow the repeatable types to appear twice, so k slots can be filled from
            # a small pool; each repeat still gets its OWN detail via assign_slots
            base = list(pool)
            # up to TRIPLE occurrences of repeatable types (the author's definition: four
            # PERTURBATIONS, not four distinct types — three extra details + an off-topic
            # passage is a legitimate 4-perturbation message). Expands the distinct-spec
            # space, which round 2 measured as the binding ceiling (133 after 106,800
            # draws, +3 in the last 30 waves).
            extra = [x for x in pool if x.id in REPEATABLE] * 2
            combo = rng.sample(base + extra, k) if len(base) + len(extra) >= k else None
            if combo is None:
                return None
        else:
            combo = rng.sample(pool, k)
        ids = {s.id for s in combo}
        if sum(1 for s in combo if s.id in CONSUMING) > n:
            continue                                            # rule 1
        if sum(1 for s in combo if s.id in UNBOUND_FAMILY) > 1:
            continue                                            # rule 2
        if {"false_presupposition", "irrelevant_information"} <= ids:
            continue                                            # rule 3
        if k >= 3 and dom not in BANS_LIFTED:
            # rules 4-5, MEASURED not assumed (check batches 2-4, 2026-08-25): these
            # pairs block k>=3 admission almost entirely. indirect x {FP, extras}: the
            # premise/extra clause is decision- or requirement-shaped, which the
            # no-request property forbids -- the writer cannot satisfy both in one
            # message. FE x FP: two value-mutating instructions that the writer merges
            # (the substitution folds into the assumption clause). k=2 keeps the pure
            # uniform draw. The realized mix ships with the subset.
            if "indirect_intent" in ids and \
                    ids & {"false_presupposition", "irrelevant_information"}:
                continue                                        # rule 4
            if {"factual_error", "false_presupposition"} <= ids:
                continue                                        # rule 5
        if not all(st.applicable(s, conditions, min_retained=0, adapter=adapter)
                   for s in combo):
            continue
        assigned = assign_slots(rng, combo, conditions, adapter=adapter,
                                droppable=droppable, alts=alts)
        if assigned is None:
            continue
        return combo
    return None


_ID_HINTS = ("_id", "id:", "user", "reservation", "order", "payment", "certificate")


def _corruptible_rank(slot: str, value) -> int:
    """Lower = safer to falsify. Structures and identifiers make the write raise."""
    v = str(value).strip()
    if v.startswith("[") or v.startswith("{"):
        return 3                                    # JSON list/dict argument
    low = str(slot).lower()
    if any(h in low for h in _ID_HINTS):
        return 2                                    # an identifier the env resolves
    if len(v) > 40:
        return 2                                    # long opaque value
    return 0                                        # plain scalar / enum


def consequential_drops(node, conditions, adapter, session) -> set[str]:
    """Slots whose REMOVAL actually changes the outcome, tested by execution.

    Airline's compile() supplies default completions for unstated arguments (its tools
    raise on a missing required argument). So dropping `cabin` from an update rebuilds
    the reservation's current cabin and the literal reading lands on the TRUE state --
    admission then rejects it as withhold_did_not_change_answer, which is correct but
    wastes the attempt. Testing once per tree lets withholding/marking target only slots
    that carry consequence, which is why insufficient_information sat at 0.21 on airline.

    Returns every slot on failure to test, so a broken hook can only cost efficiency.
    """
    out: set[str] = set()
    try:
        truth = node.ground_truth.hash
    except Exception:
        return {str(s) for s, _o, _v in conditions}
    for slot, _op, _v in conditions:
        lit = _apply(conditions, drop=[str(slot)])
        if not lit:
            continue
        try:
            recipe = adapter.compile(dict(node.base), tuple(lit))
            gt = adapter.execute(recipe, session)
            if gt.hash != truth:
                out.add(str(slot))
        except Exception:
            # a drop the environment rejects outright is also a real consequence
            out.add(str(slot))
    return out or {str(s) for s, _o, _v in conditions}


def assign_slots(rng, combo, conditions, *, adapter=None,
                 droppable: set | None = None,
                 alts: dict | None = None) -> dict[str, str] | None:
    """Pre-assign each consuming fault its own detail (writer guidance AND the contract's
    expectation). Eligibility mirrors the single-fault rules: vagueness needs an ordered
    slot; every mark type needs a slot whose NAME does not state its value."""
    slots = [(str(s), op, v) for s, op, v in conditions]
    taken: set[str] = set()
    out: dict[str, str] = {}
    order = sorted((s for s in combo if s.id in CONSUMING),
                   key=lambda s: (s.id not in UNBOUND_FAMILY, s.id != "factual_error"))
    # with multiplicity a type may appear twice; each occurrence needs its own detail, so
    # the assignment map is keyed by occurrence index rather than by type id
    seen_ct: dict = {}
    for strat in order:                    # most-constrained first: marks, then FE
        if strat.id == "vagueness_subjectivity":
            elig = [s for s, op, v in slots
                    if s not in taken and st.slot_is_ordered(s, op, v, adapter)
                    and not st.slot_name_encodes_value(s)
                    and (droppable is None or s in droppable)]
        elif strat.id in UNBOUND_FAMILY:
            elig = [s for s, _, _ in slots
                    if s not in taken and not st.slot_name_encodes_value(s)
                    and (droppable is None or s in droppable)]
        elif strat.id == "factual_error":
            if _adapter_renders_intent(adapter):
                vals = {str(s_): v_ for s_, _o, v_ in conditions}
                cand = [s for s, _, _ in slots if s not in taken]
                # strongly prefer slots with a LEGAL alternative value: the environment
                # validates the write, so a falsified value must still be legal to produce a
                # different state rather than an exception
                with_alts = [s for s in cand if alts and alts.get(s)]
                if with_alts:
                    elig = sorted(with_alts)
                else:
                    best = min((_corruptible_rank(s, vals.get(s)) for s in cand),
                               default=0)
                    elig = sorted(s for s in cand
                                  if _corruptible_rank(s, vals.get(s)) == best)
            else:
                # near-miss substitutions admit best on strictly-compared slots; option:
                # values nearest-match, so prefer attributes and the price cap (falsify rank)
                elig = sorted((s for s, _, _ in slots if s not in taken),
                              key=lambda x: (x.startswith("option:"), x))
        else:                              # insufficient_information
            elig = [s for s, _, _ in slots
                    if s not in taken and (droppable is None or s in droppable)]
        if not elig:
            return None
        if strat.id == "factual_error" and not _adapter_renders_intent(adapter):
            pick = elig[0]                 # the sort above IS the policy; no rng drawn
        else:
            pick = rng.choice(sorted(elig))
        seen_ct[strat.id] = seen_ct.get(strat.id, 0) + 1
        key = strat.id if seen_ct[strat.id] == 1 else f"{strat.id}#{seen_ct[strat.id]}"
        out[key] = pick
        taken.add(pick)
    return out


# -------------------------------------------------------------- the generation
def composite_generation_prompt(conditions, combo, assigned: dict, *, context: str,
                                attempt: int, describe=None, adapter=None,
                                alts: dict | None = None) -> str:
    """Composite prompt built from the SAME adapter hooks the single-fault prompt uses.

    The first tau2 port skipped these and admitted 0/50 on airline. Four of them matter:
      * card_overrides -- airline's stock fault cards ARE flight-booking examples, so the
        stock "NEVER copy entities from the example" rule tells the generator to avoid the
        very entities the domain requires.
      * gen_hints / dynamic_gen_hint -- per-domain phrasing of each fault (retail's
        vagueness hint is "swap it for a nicer one", not WebShop's "cheap").
      * generation_voice / generation_domain_word -- an airline passenger, not a shopper.
      * describe_intent -- tool-argument intents render empty under WebShop's projection.

    An adapter that renders no intent of its own gets the original WebShop prompt byte
    for byte, per-flaw targeting guidance included.
    """
    generalized = _adapter_renders_intent(adapter)
    values = {str(s): v for s, _, v in conditions}

    # Precision-critical flaws first: the k=4 pilot showed the substitution dissolving
    # when factual_error sat late in a long flaw list -- the writer restated the true
    # value or paraphrased instead of asserting a crisp wrong one. Order: FE, the
    # unbound flaw, FP/extras, then the whole-message flaws (noise/indirect) last.
    _ORDER = {"factual_error": 0, "referential_ambiguity": 1, "vagueness_subjectivity": 1,
              "lexical_ambiguity": 1, "syntactic_ambiguity": 1,
              "false_presupposition": 2, "irrelevant_information": 2,
              "insufficient_information": 3,
              "contextual_irrelevance": 4, "indirect_intent": 5}
    combo = sorted(combo, key=lambda s: _ORDER.get(s.id, 9))

    cards_txt = []
    if generalized:
        intent_block = describe_intent(conditions, adapter)
        voice = getattr(adapter, "generation_voice", "a plain user's voice")
        domain_word = getattr(adapter, "generation_domain_word", "")
        adapter_hints = getattr(adapter, "gen_hints", {}) or {}
        cards = getattr(adapter, "card_overrides", {}) or {}
        spoken = getattr(adapter, "spoken_value", None)
        for i, strat in enumerate(combo, 1):
            description, example = cards.get(
                strat.id, REVISED_CARDS.get(strat.id, (strat.description, strat.example)))
            hint = adapter_hints.get(strat.id, GEN_HINTS.get(strat.id, ""))
            dyn = getattr(adapter, "dynamic_gen_hint", None)
            if dyn is not None:
                try:
                    extra = dyn(strat.id, conditions)
                    if extra:
                        hint = (hint + "\n" if hint else "") + extra
                except Exception as exc:
                    log.debug("dynamic hint hook failed for %s: %s", strat.id, exc)
            hint = hint.replace("- your flaw:", f"- how to do flaw {i}:")
            target = ""
            if strat.id in assigned:
                slot = assigned[strat.id]
                label = describe(slot) if describe else slot
                val = spoken(slot, values[slot]) if spoken else values[slot]
                target = (f"  Apply flaw {i} to this requirement if it fits naturally: "
                          f"{label} = {val}\n")
            cards_txt.append(f"Flaw {i} -- {strat.name} ({strat.id}): {description}\n"
                             f"  Example of the flaw alone: {example}\n{target}{hint}")
    else:
        feats = [str(v) for s, _, v in conditions if str(s).startswith("attr:")]
        opts = [str(v) for s, _, v in conditions if str(s).startswith("option:")]
        caps = [v for s, _, v in conditions if s == "price_upper"]
        for i, strat in enumerate(combo, 1):
            description, example = REVISED_CARDS.get(strat.id,
                                                     (strat.description, strat.example))
            hint = GEN_HINTS.get(strat.id, "").replace("- your flaw:",
                                                       f"- how to do flaw {i}:")
            target = ""
            if strat.id in assigned:
                slot = assigned[strat.id]
                phrase = describe(slot) if describe else slot
                target = (f"  APPLY flaw {i} to exactly this requirement: "
                          f"{phrase} = {values[slot]!s}\n")
                if strat.id == "factual_error" and len(combo) >= 3:
                    target += ("  State the WRONG value plainly and exactly once -- a "
                               "specific concrete value, never a paraphrase, and never "
                               f"the true value {values[slot]!r}.\n")
                    if any(s.id == "false_presupposition" for s in combo):
                        # check-3: with FP co-drawn the writer folded the substitution into
                        # the assumption clause; the two flaws target DIFFERENT requirements
                        target += ("  This is a DIFFERENT requirement from the assumption "
                                   "flaw below: state this wrong value as plain fact, not "
                                   "as an assumption or fallback.\n")
                if strat.id == "false_presupposition" and \
                        any(s.id == "indirect_intent" for s in combo):
                    # "settling for" is decision language, which the no-request flaw forbids;
                    # phrase the premise as hearsay instead
                    target += ("  Phrase the assumption as something you HEARD, never as "
                               "your decision: 'apparently the ... ones are sold out, and "
                               "people say ... works just as well' -- no 'I'll settle for' "
                               "or 'I'll take'.\n")
                if strat.id == "irrelevant_information" and \
                        any(s.id == "indirect_intent" for s in combo):
                    target += ("  Phrase the extra detail as heard preference, not a "
                               "requirement: 'folks say the ones bundled with ... are the "
                               "ones to get'.\n")
            cards_txt.append(f"Flaw {i} -- {strat.name} ({strat.id}): {description}\n"
                             f"  Example of the flaw alone: {example}\n{target}{hint}")

    k = len(combo)
    has_noise = any(s.id == "contextual_irrelevance" for s in combo)
    length = ("two to four sentences" if k <= 2 else "three to five sentences") + \
             (", plus the off-topic passage" if has_noise else "")
    # k=4 pilot: with three other flaws in the air the writer slid "I guess I'll just
    # grab..." past the hint -- a statement of intended action reads as a request. Make
    # the whole-message property explicit when indirect_intent is drawn.
    no_request_rule = ""
    if any(s.id == "indirect_intent" for s in combo):
        no_request_rule = ("- THE WHOLE MESSAGE asks for nothing and decides nothing: no "
                           "request, no question expecting action, and no statement of "
                           "what you will do, take, grab, or go with -- only your "
                           "situation and what you have heard\n")
    if generalized:
        domain_rule = (f"- the flaws must fit THIS {domain_word} domain; NEVER copy "
                       "entities from the examples\n" if domain_word else
                       "- NEVER copy entities from the examples above\n")
        return f"""You are an expert at creating flawed instructions that test an agent's ability to handle
confused, mistaken requests through clarification.

The user's TRUE intent:
{intent_block}
{context}

Write ONE message that carries ALL {k} of these flaws AT THE SAME TIME -- the way a real
person's genuinely muddled message piles several mistakes into one breath:

{chr(10).join(cards_txt)}

Rules:
- {length}, first person, {voice}; it must read as ONE natural message
- every flaw above must be present simultaneously; none may cancel or repair another
- every requirement NOT touched by a flaw stays accurately stated
{domain_rule}- do not reveal that anything is wrong
{no_request_rule}(variation {attempt})
Output only the flawed message."""
    return f"""You are an expert at creating flawed instructions that test an agent's ability to handle
confused, mistaken requests through clarification.

The shopper's TRUE intent:
  required features: {feats}
  option: {opts or ['(none)']}
  price limit: {caps or ['(none)']}
{context}

Write ONE message that carries ALL {k} of these flaws AT THE SAME TIME -- the way a real
person's genuinely muddled message piles several mistakes into one breath:

{chr(10).join(cards_txt)}

Rules:
- {length}, first person, a plain shopper's voice; it must read as ONE natural message
- every flaw above must be present simultaneously; none may cancel or repair another
- every requirement NOT named by a flaw stays accurately stated
- the flaws must fit THIS shopping domain; NEVER copy entities from the examples
- do not reveal that anything is wrong
{no_request_rule}(variation {attempt})
Output only the flawed message."""


FRAME_FAULTS = ("contextual_irrelevance", "indirect_intent")


def frame_prompt(text1: str, frame: list, *, attempt: int, adapter=None) -> str:
    """Stage 2 of two-stage rendering (k>=3): the k=4 pilots showed precision dissolving
    when four fault instructions compete in one pass -- the substitution got restated
    or paraphrased, and request language leaked past the indirect rule. The frame
    faults are TRANSFORMATIONS of a finished message (bury it, un-ask it), which the
    writer applies reliably while preserving embedded content. The final text is still
    verified on every channel, so a frame pass that repairs or drops a stage-1 fault is
    rejected, never admitted."""
    generalized = _adapter_renders_intent(adapter)
    steps = []
    if "contextual_irrelevance" in frame:
        steps.append("- ADD an opening of three or four sentences about a COMPLETELY "
                     "different topic -- specific and engaging, at least as long as the "
                     "rest of the message, mentioning none of the "
                     + ("request details" if generalized else "shopping details"))
    if "indirect_intent" in frame:
        steps.append("- REPHRASE so the message never asks for anything and never says "
                     "what you will do, take, grab, or go with: no requests, no "
                     "questions expecting action -- only your situation and what you "
                     "have heard (e.g. 'I've heard that ... works well when ...')")
    who = "user" if generalized else "shopper"
    keep_rule = ("- KEEP every stated detail, value, and mistake EXACTLY as written"
                 if generalized else
                 "- KEEP every shopping detail, value, price, and mistake EXACTLY as stated")
    voice_rule = ("- first person, in the user's own plain voice" if generalized
                  else "- first person, a plain shopper's voice")
    return f"""A {who} wrote this message:
{text1}

Rewrite it as ONE natural message, applying these changes:
{chr(10).join(steps)}

Rules:
{keep_rule} -- do not fix,
  drop, add, or alter any of them
{voice_rule}
(variation {attempt})
Output only the rewritten message."""


# ------------------------------------------- composite-aware specialist passes
# Same prompts, same parsing, same span/plausibility rules as the single-fault
# specialists; ONLY the single-fault exclusivity checks (other_changes,
# request_complete/true_requirements_intact, every-value-present) are omitted --
# the composite contract replaces them with whole-text accounting.

def _co_extract_fp(conditions, text, llm, *, dual_extract: bool, adapter=None) -> list:
    vocab = {str(s): str(v) for s, _o, v in conditions}

    def one(role):
        obj = _first_json(llm.complete(fp_extraction_prompt(conditions, text, adapter),
                                       role=role))
        if not isinstance(obj, dict):
            raise Rejected("extraction_unparseable")
        return obj

    e1 = one(ROLE_SELECT)
    about = e1.get("assumed_unsatisfiable")
    if not e1.get("presupposition") or not about:
        raise Rejected("composite:no_presupposition")
    slot = _canon(about[0], vocab, true_value=about[1] if len(about) > 1 else None)
    if slot not in vocab:
        raise Rejected("composite:presupposition_not_about_true_intent")
    if dual_extract:
        e2 = one(ROLE_JUDGE)
        a2 = e2.get("assumed_unsatisfiable") or [None]
        s2 = _canon(a2[0], vocab, true_value=a2[1] if len(a2) > 1 else None)
        if not e2.get("presupposition") or s2 != slot:
            raise Rejected("extractor_disagreement:fp")
    fb = e1.get("fallback")
    if _adapter_renders_intent(adapter):
        # Same JSON-encoding trap as factual_error: stored values carry their quotes
        # ('"economy"', '["item_1","item_2"]'), so check_text's presence test on a raw
        # fallback can never match prose -- 24 of 34 retail k=4 failures. Normalise to the
        # human-visible atom, exactly as _co_extract_fe does. Plain-text domains store the
        # human-visible value already and keep the raw fallback their samples were
        # admitted under.
        def _atom1(x):
            a = _value_atoms(x)
            return str(a[0]) if a else str(x)
        fallback = [slot, _atom1(fb[1])] if (fb and len(fb) > 1 and fb[1]) else None
    else:
        fallback = [slot, str(fb[1])] if (fb and len(fb) > 1 and fb[1]) else None
    return [{"text": str(e1["presupposition"]), "condition": None,
             "about": [slot, vocab[slot]], "alternative": fallback}]


def _co_extract_extras(conditions, text, llm, *, dual_extract: bool, adapter=None) -> list:
    def usable(obj):
        out = []
        for e in obj.get("extra_details") or []:
            if isinstance(e, str):
                e = {"text": e, "condition": None}
            if not isinstance(e, dict) or not e.get("text"):
                continue
            out.append({"text": str(e["text"]), "condition": e.get("condition")})
        return out

    def one(role):
        obj = _first_json(llm.complete(irrelevant_extraction_prompt(conditions, text),
                                       role=role))
        if not isinstance(obj, dict):
            raise Rejected("extraction_unparseable")
        return obj

    x1 = usable(one(ROLE_SELECT))
    if not x1:
        raise Rejected("composite:no_extra_detail")
    if dual_extract:
        x2 = usable(one(ROLE_JUDGE))
        if _adapter_renders_intent(adapter):
            # MULTI-EXTRA agreement (2026-08-27): with multiplicity a combo can carry TWO
            # extras, and comparing only the first of each made mere ordering differences
            # read as disagreement. Agreement = every extra one extractor found overlaps
            # some extra the other found (both directions).
            def covered(a, b):
                return all(any(_texts_overlap(e["text"], f["text"]) for f in b) for e in a)
            if not x2 or not covered(x1, x2) or not covered(x2, x1):
                raise Rejected("extractor_disagreement:extras")
        elif not x2 or not _texts_overlap(x1[0]["text"], x2[0]["text"]):
            # single-extra domains keep the original first-vs-first agreement rule
            raise Rejected("extractor_disagreement:extras")
    return x1


def _co_extract_offtopic(conditions, text, llm, *, retained: set, dual_extract: bool) -> str:
    """Span length, span-in-text, and no-RETAINED-value-in-span keep their single-fault
    strictness; the every-value-present check is dropped (other faults hide values by
    design), replaced by the contract's retained-values-present rule."""

    def one(role):
        obj = _first_json(llm.complete(offtopic_extraction_prompt(conditions, text),
                                       role=role))
        if not isinstance(obj, dict):
            raise Rejected("extraction_unparseable")
        return obj

    e1 = one(ROLE_SELECT)
    span = str(e1.get("off_topic_passage") or "")
    if len(span.split()) < _MIN_OFFTOPIC_WORDS:
        raise Rejected("composite:off_topic_too_short")
    if not _span_in_text(span, text):
        raise Rejected("extraction_bad_span")
    values = {str(s): v for s, _, v in conditions}
    for slot in retained:
        v = values.get(slot)
        if v is not None and _present(span, v, default=False):
            raise Rejected("composite:off_topic_mentions_requirement")
    if dual_extract:
        e2 = one(ROLE_JUDGE)
        span2 = str(e2.get("off_topic_passage") or "")
        if len(span2.split()) < _MIN_OFFTOPIC_WORDS or not _texts_overlap(span, span2):
            raise Rejected("extractor_disagreement:offtopic")
    return span


def _co_check_norequest(conditions, text, llm, *, dual_extract: bool) -> None:
    """indirect_intent in a composite: the no-request property is a property of the WHOLE
    text, so the mechanical phrase check and the LLM verdict both apply unchanged; only
    the every-value-conveyed check is dropped (other faults hide values by design)."""
    low = re.sub(r"\s+", " ", text.lower())
    for phrase in _REQUEST_PHRASES:
        if re.search(_WORD.format(re.escape(phrase)), low):
            raise Rejected("composite:request_language_present")

    def one(role):
        obj = _first_json(llm.complete(norequest_extraction_prompt(conditions, text),
                                       role=role))
        if not isinstance(obj, dict):
            raise Rejected("extraction_unparseable")
        return obj

    if one(ROLE_SELECT).get("request_stated"):
        raise Rejected("composite:request_stated")
    if dual_extract and one(ROLE_JUDGE).get("request_stated"):
        raise Rejected("extractor_disagreement:norequest")


def _co_extract_syntactic(conditions, text, llm, *, dual_extract: bool) -> list:
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
        raise Rejected("composite:no_unclear_attachment")
    slot = _canon(slot, vocab)
    if slot not in vocab:
        raise Rejected("composite:no_unclear_attachment")
    p1 = bool(e1.get("both_attachments_plausible"))
    p2 = p1
    if dual_extract:
        e2 = one(ROLE_JUDGE)
        s2 = _canon(e2.get("target_slot") or "", vocab)
        if not e2.get("ambiguous_phrase") or s2 != slot:
            raise Rejected("extractor_disagreement:syntactic")
        p2 = bool(e2.get("both_attachments_plausible"))
    if not (p1 or p2):
        raise Rejected("admission:implausible_reading")
    return [{"surface": e1.get("ambiguous_phrase"),
             "placements": [[slot, vocab[slot]], [slot, None]]}]


# ----------------------------------------------------- composite dual agreement
def _slotset(ext: dict, *channels) -> set[str]:
    out: set[str] = set()
    for ch in channels:
        for it in ext.get(ch) or []:
            if ch == "withheld":
                out.add(str(it))
            elif ch == "ambiguous":
                for p in (it or {}).get("placements") or []:
                    if p:
                        out.add(str(p[0]))
            elif isinstance(it, (list, tuple)) and it:
                out.add(str(it[0]))
    return out


def agree_composite(e1: dict, e2: dict, combo_ids: set) -> bool:
    """Channel-scoped dual agreement for composites.

    The k=1 rule (extractions_agree) demands per-channel identity, and the smoke run
    showed why that cannot survive k>=2: the marked/withheld/ambiguous boundary is the
    documented spot where the two models bin the same phenomenon differently (the reason
    the specialist questionnaires exist), and at k faults ANY slip on ANY channel by
    EITHER model kills the attempt -- 10 of 13 smoke combos died exactly there.

    So agreement is demanded only where the generic extraction is authoritative, and at
    the granularity that is actually decidable across models:
      * substituted slot-set (factual_error is generic-authoritative)
      * the UNION of not-exactly-stated slots (withheld+marked+ambiguous) -- WHICH bin
        a slot fell into is the fuzzy part; WHICH slots are unresolved is agreed on
      * the noise flag
    presupposed/extras agreement is delegated to the FP/extras specialists (which carry
    their own dual pass) whenever one of those types is drawn."""
    if _slotset(e1, "substituted") != _slotset(e2, "substituted"):
        return False
    if (_slotset(e1, "withheld", "marked", "ambiguous")
            != _slotset(e2, "withheld", "marked", "ambiguous")):
        return False
    if bool(e1.get("noise")) != bool(e2.get("noise")):
        return False
    if not (combo_ids & {"false_presupposition", "irrelevant_information"}):
        if bool(e1.get("presupposed")) != bool(e2.get("presupposed")):
            return False
    return True


def fold_unbound(ext: dict, combo_ids: set) -> dict:
    """When NO unbound-family fault is drawn, an extractor that binned a removed slot as
    'marked' ('vaguely mentioned') rather than 'withheld' is making the same fuzzy call
    as above. The operative definition throughout v2 is value-presence -- check_text
    verifies the withheld value is ABSENT from the text -- so fold the unbound bins into
    withheld and let that mechanical check be the judge."""
    if combo_ids & set(UNBOUND_FAMILY):
        return ext
    out = dict(ext)
    extra = sorted(_slotset(ext, "marked", "ambiguous") - set(out.get("withheld") or []))
    out["withheld"] = list(out.get("withheld") or []) + extra
    out["marked"], out["ambiguous"] = [], []
    return out


# ----------------------------------------- targeted FE + additions questionnaires
# The debug batch (2026-08-25, 5 rows, 60 attempt-dumps) showed the generic diff form
# cannot survive a multi-fault text: the FP clause reads as substitution to one model
# and presupposition to the other in nearly every pair, and the vague slot drifts in
# and out of the unresolved set. Four judgment calls at ~70% pairwise consistency give
# ~24% joint agreement -- the observed rate. The k=1 pipeline hit this same wall and
# answered with targeted questionnaires; composites do the same. The generic diff form
# is not used here at all.

FE_SCHEMA = """Output ONLY a JSON object:
{"stated_value": <the value the text states for this requirement, quoted, or null if
                  the text does not state one>,
 "same_as_true": <true if the stated value means the same thing as the true value>}"""


def fe_targeted_prompt(conditions, slot: str, text: str) -> str:
    values = {str(s): v for s, _, v in conditions}
    return (f"A shopper's TRUE intent, as (slot, value) pairs:\n"
            f"{[[str(s), str(v)] for s, _, v in conditions]}\n\n"
            f"An instruction derived from it:\n{text}\n\n"
            f"Consider ONLY this requirement: {slot} (true value: {values[slot]!r}).\n"
            "What value, if any, does the text state for it? The value may be embedded "
            "in hearsay or rambling ('I heard the triaxial ones work well') -- report "
            f"the exact value phrase the text uses.\n{FE_SCHEMA}")


def _co_extract_fe(conditions, *args, dual_extract: bool, adapter=None) -> list:
    """factual_error for a composite -- one questionnaire per prompt family.

    Called as ``(conditions, slot, text, llm)``: the targeted questionnaire above
    (fe_targeted_prompt/FE_SCHEMA), pinned to the assigned slot -- the original WebShop
    form, which reports the value stated for that one requirement.

    Called as ``(conditions, text, llm)``: the domain-aware form for adapters that render
    their own intent. Uses the single-fault fe_extraction_prompt -- domain vocabulary,
    all_condition_values, extraction_note -- so the question asked is the domain's own.
    The only thing dropped is `other_changes`, the single-fault exclusivity clause: in a
    composite the other drawn faults touch other requirements by design, so that clause
    rejects 100% of composites (measured: it was the dominant verdict on the first
    airline diagnostic). Evidence checks are untouched: both extractors must name the
    same slot, that slot must be a real requirement, and check_text still proves the
    wrong value is in the text.
    """
    if len(args) == 3:                     # (slot, text, llm): targeted, slot-pinned form
        slot, text, llm = args
        values = {str(s): v for s, _, v in conditions}

        def one(role):
            obj = _first_json(llm.complete(fe_targeted_prompt(conditions, slot, text),
                                           role=role))
            if not isinstance(obj, dict):
                raise Rejected("extraction_unparseable")
            return obj

        e1 = one(ROLE_SELECT)
        stated = e1.get("stated_value")
        if not stated or e1.get("same_as_true"):
            raise Rejected("composite:fe_not_substituted")
        if dual_extract:
            e2 = one(ROLE_JUDGE)
            s2 = e2.get("stated_value")
            if not s2 or e2.get("same_as_true"):
                raise Rejected("extractor_disagreement:fe")
            if str(s2).strip().lower() != str(stated).strip().lower() \
                    and not _texts_overlap(stated, s2):
                raise Rejected("extractor_disagreement:fe")
        return [[slot, values[slot], str(stated)]]

    text, llm = args                       # (text, llm): domain-aware form
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
        raise Rejected("composite:fe_not_substituted")
    # Stored condition values are canonical JSON ('"economy"', '["a","b"]'), so the raw
    # form never appears in prose and check_text's presence tests are meaningless against
    # it. The fork solves this with _value_atoms on the withheld branch; the substituted
    # branch needs the same treatment, and the extractor's stated_instead may come back
    # quoted too. Normalise BOTH to human-visible atoms before any presence test --
    # this was 5/5 attempts of text:asserted_value_absent on every airline FE draw.
    def _atom(x):
        a = _value_atoms(x)
        return str(a[0]) if a else str(x)
    stated = _atom(stated)
    slot = _canon(rep[0], vocab, true_value=rep[1] if len(rep) > 1 else None)
    if slot not in vocab:
        raise Rejected("composite:fe_slot_not_in_intent")
    if dual_extract:
        e2 = one(ROLE_JUDGE)
        r2 = e2.get("replaced") or [None]
        s2 = _canon(r2[0], vocab, true_value=r2[1] if len(r2) > 1 else None)
        if not e2.get("stated_instead") or s2 != slot:
            raise Rejected("extractor_disagreement:fe")
    return [[slot, None, str(stated)]]


NO_EXTRA_SCHEMA = """Output ONLY a JSON object:
{"other_requirements": [<any concrete product property the text requests or assumes
                         that is in NEITHER list, quoted>]}"""


def _co_check_no_extras(conditions, legitimate: list, text: str, llm,
                        adapter=None) -> None:
    """Polices unplanned ADDED requirements -- the one job of the retired generic sweep
    that no mechanical check covers (unplanned substitution/withholding of a retained
    slot already trips the retained-value-present check). Single-asked: a false positive
    costs one attempt, never admits a bad text."""
    owner = ("The user's" if _adapter_renders_intent(adapter) else "A shopper's")
    obj = _first_json(llm.complete(
        f"{owner} TRUE intent, as (slot, value) pairs:\n"
        f"{[[str(s), str(v)] for s, _, v in conditions]}\n\n"
        f"Also LEGITIMATELY present in the text (do not report these):\n"
        f"{legitimate}\n\n"
        f"The text:\n{text}\n\n"
        "Does the text request or assume any concrete product property (feature, option, "
        "price, brand) that appears in NEITHER list above? Naming the product kind is "
        f"not a property.\n{NO_EXTRA_SCHEMA}", role=ROLE_SELECT))
    if isinstance(obj, dict) and obj.get("other_requirements"):
        raise Rejected("composite:unplanned_added_requirement")


# ------------------------------------------------------------------- contract
def composite_contract(combo, assigned: dict, ext: dict, conditions, text: str,
                       adapter=None) -> None:
    """Every channel accounted for against the drawn combo -- the composite replacement
    for the single-fault exclusivity rules. Counts are strict and slot-exact: a text
    that corrupts an UNASSIGNED detail carries an unplanned extra fault and is rejected,
    exactly as extra_* rejections police accidental second flaws at k=1.

    Adapters that render their own intent are checked by COUNT with multiplicity (their
    thin pools may draw a repeatable type twice, and their extractors report whichever
    slot the text corrupted); the plain-text family is checked slot-exactly against the
    assignment, as its samples were admitted.
    """
    generalized = _adapter_renders_intent(adapter)
    ids = {s.id for s in combo}
    mult: dict = {}
    for s_ in combo:
        mult[s_.id] = mult.get(s_.id, 0) + 1
    values = {str(s): v for s, _, v in conditions}

    subs = [list(x) for x in ext.get("substituted") or []]
    if "factual_error" in ids:
        if generalized:
            want_n = mult["factual_error"]
            if len(subs) != want_n:
                raise Rejected(f"composite:substituted_count:{len(subs)}!={want_n}")
            if str(subs[0][0]) not in values:
                raise Rejected("composite:substituted_unknown_slot")
        else:
            if len(subs) != 1:
                raise Rejected(f"composite:substituted_count:{len(subs)}")
            if str(subs[0][0]) != assigned["factual_error"]:
                raise Rejected("composite:substituted_wrong_slot")
    elif subs:
        raise Rejected("composite:unplanned_substitution")

    withheld = [str(x) for x in ext.get("withheld") or []]
    if "insufficient_information" in ids:
        if generalized:
            real = [w for w in withheld if w in values]
            if len(real) != mult["insufficient_information"]:
                raise Rejected(f"composite:withheld_count:{len(real)}")
        else:
            if assigned["insufficient_information"] not in withheld:
                raise Rejected("composite:assigned_slot_not_withheld")
            extra_w = [w for w in withheld if w != assigned["insufficient_information"]
                       and w in values]
            if extra_w:
                raise Rejected("composite:unplanned_withholding")
    elif withheld:
        raise Rejected("composite:unplanned_withholding")

    unbound_ids = ids & set(UNBOUND_FAMILY)
    marked = {str(m[0]) for m in ext.get("marked") or []}
    amb = {str(p[0]) for a in (ext.get("ambiguous") or [])
           for p in (a.get("placements") or [])}
    unbound_slots = marked | amb
    if unbound_ids:
        if generalized:
            real = {u for u in unbound_slots if u in values}
            if len(real) != 1:
                raise Rejected(f"composite:unbound_count:{len(real)}")
        else:
            want = assigned[next(iter(unbound_ids))]
            if unbound_slots != {want}:
                raise Rejected("composite:unbound_wrong_slot")
    elif unbound_slots:
        raise Rejected("composite:unplanned_unbound")

    if "false_presupposition" in ids and not ext.get("presupposed"):
        raise Rejected("composite:missing_presupposition")
    if "irrelevant_information" in ids and not ext.get("extras"):
        raise Rejected("composite:missing_extra_detail")
    if not (ids & {"false_presupposition", "irrelevant_information"}):
        if ext.get("presupposed") or ext.get("extras"):
            raise Rejected("composite:unplanned_added_requirement")

    if bool(ext.get("noise")) != ("contextual_irrelevance" in ids):
        raise Rejected("composite:noise_mismatch")
    if "indirect_intent" in ids and not ext.get("oblique"):
        raise Rejected("composite:missing_oblique")
    # oblique phrasing alongside content flaws is tolerated when not drawn -- it hides
    # nothing extra (same tolerance as the k=1 contract).

    # channels must not overlap: one detail, one fault
    consumed = [str(s[0]) for s in subs] + withheld + sorted(unbound_slots)
    if len(consumed) != len(set(consumed)):
        raise Rejected("composite:channel_overlap")

    # every UNTOUCHED detail must still be stated: the composite is exactly its K faults.
    # For adapters that render their own intent, only where "stated" is testable by
    # string containment: a tool-argument intent carries JSON structures (a flights
    # list, a passengers list) that no natural sentence reproduces verbatim, so the check
    # is applied to scalar values only; structured ones are covered by the executable
    # admission instead.
    for slot, v in values.items():
        if slot in consumed:
            continue
        if generalized:
            sv = str(v).strip()
            if sv.startswith("[") or sv.startswith("{") or len(sv) > 40:
                continue
        if not _present_loose(text, v):
            raise Rejected("composite:retained_value_absent")


# ------------------------------------------------------------------- the loop
def build_composite_mask(graph, combo, *, llm, adapter, executor, session,
                         context: str = "", dual_extract: bool = True,
                         attempts: int = 5, trace: list | None = None,
                         droppable: set | None = None,
                         alts: dict | None = None) -> dict:
    """One admitted composite (mask, query) for this (graph, combo) -- or Rejected.

    Verification order per attempt: generic dual extraction (the flaw-counter that sees
    every channel at once), specialist passes for the drawn types that need them,
    composite contract, check_text string cross-checks, then admit() PER COMPONENT --
    each fault must carry its own executable consequence exactly as at k=1.
    """
    node = graph.root
    conditions = tuple(node.conditions)
    ids = [s.id for s in combo]
    rng_free = {s.id for s in combo if s.id not in CONSUMING}
    describe = getattr(adapter, "slot_phrase", None)
    generalized = _adapter_renders_intent(adapter)
    last = "no_attempts"

    import random as _random
    rng = _random.Random(f"{graph.graph_id}:{'+'.join(sorted(ids))}:assign")
    assigned = assign_slots(rng, combo, conditions, adapter=adapter,
                            droppable=droppable, alts=alts)
    if assigned is None:
        raise Rejected("composite:no_slot_assignment")

    has_noise = "contextual_irrelevance" in ids
    if generalized:
        # scales with the number of stated details: tau2 intents carry up to 12 tool-call
        # arguments, whose faithful restatement alone exceeds WebShop's whole budget
        max_words = (130 + 18 * max(0, len(conditions) - 4)
                     + (90 if has_noise else 0) + 25 * (len(combo) - 1))
    else:
        max_words = 130 + (90 if has_noise else 0) + 25 * (len(combo) - 1)

    frame = [i for i in ids if i in FRAME_FAULTS]
    content_combo = [s for s in combo if s.id not in FRAME_FAULTS]
    two_stage = (len(combo) >= 3 and frame and content_combo
                 and not getattr(adapter, 'extras_may_be_pragmatic', False))

    for attempt in range(1, attempts + 1):
        if two_stage:
            text1 = llm.complete(
                composite_generation_prompt(conditions, content_combo, assigned,
                                            context=context, attempt=attempt,
                                            describe=describe, adapter=adapter,
                                            alts=alts),
                role=ROLE_RENDER,
                model="doubao-seed-2-1-pro-260628").strip().strip('"')
            # Frame-retry: a leaked request phrase is a STAGE-2 fault, and discarding the
            # whole attempt throws away verified stage-1 content with it (4 of 10 check-2
            # deaths). Redo only the frame, cheaply, before committing to verification.
            text = None
            for ftry in range(1, 5):
                cand = llm.complete(frame_prompt(text1, frame, attempt=attempt * 10 + ftry,
                                                 adapter=adapter),
                                    role=ROLE_RENDER,
                                    model="doubao-seed-2-1-pro-260628").strip().strip('"')
                if "indirect_intent" in ids:
                    low = re.sub(r"\s+", " ", cand.lower())
                    if any(re.search(_WORD.format(re.escape(p)), low)
                           for p in _REQUEST_PHRASES):
                        continue
                    # pre-clear against the ACTUAL judge criterion, single-asked: the
                    # phrase list missed judge-visible requests in every check-3 indirect
                    # combo (decision language like "I'll settle for..."), and each miss
                    # burned a whole attempt's verified stage-1 content
                    probe = _first_json(llm.complete(
                        norequest_extraction_prompt(conditions, cand), role=ROLE_SELECT))
                    if isinstance(probe, dict) and probe.get("request_stated"):
                        continue
                text = cand
                break
            if text is None:
                last = "composite:request_language_present"
                if trace is not None:
                    trace.append({"attempt": attempt, "text": text1,
                                  "verdict": "composite:frame_retries_exhausted"})
                continue
        else:
            text = llm.complete(
                composite_generation_prompt(conditions, combo, assigned, context=context,
                                            attempt=attempt, describe=describe,
                                            adapter=adapter, alts=alts),
                role=ROLE_RENDER,
                model="doubao-seed-2-1-pro-260628").strip().strip('"')
        try:
            if "indirect_intent" in ids:
                _co_check_norequest(conditions, text, llm, dual_extract=dual_extract)

            # Targeted assembly: one questionnaire per drawn component, no open diff.
            ext = _empty_ext()
            ext["extras"] = []
            ext["offtopic_span"] = None
            ext["oblique"] = "indirect_intent" in ids
            ext["noise"] = False

            if "factual_error" in ids:
                if generalized:
                    # fork-native targeted FE (adapter-aware phrasing). It reports whichever
                    # slot the text substituted; the composite contract below checks that it
                    # is the slot we assigned.
                    ext["substituted"] = _co_extract_fe(
                        conditions, text, llm, dual_extract=dual_extract,
                        adapter=adapter)
                else:
                    ext["substituted"] = _co_extract_fe(
                        conditions, assigned["factual_error"], text, llm,
                        dual_extract=dual_extract)
            if "insufficient_information" in ids:
                if generalized:
                    # value-absence IS the operative definition of withheld throughout v2.
                    # With the slot no longer pinned by contract, derive it: the assigned
                    # slot if the writer honoured it, else whichever single true value the
                    # text omitted. check_text re-verifies absence mechanically below.
                    want_slot = assigned.get("insufficient_information")
                    # A SUBSTITUTED slot's true value is absent by definition -- the writer
                    # replaced it -- so absence alone would also mark it withheld, and the
                    # contract then rejects the collision its own detection created
                    # (composite:channel_overlap, 8 of 24 in the mechanical-pool test).
                    # Slots already claimed by another fault are not candidates.
                    claimed_now = {str(x[0]) for x in (ext.get("substituted") or [])}
                    absent = [str(s_) for s_, _o, v_ in conditions
                              if str(s_) not in claimed_now
                              and not _present_loose(text, v_)]
                    if want_slot and want_slot in absent:
                        ext["withheld"] = [want_slot]
                    elif len(absent) == 1:
                        ext["withheld"] = absent
                    else:
                        raise Rejected(f"composite:withheld_ambiguous:{len(absent)}")
                else:
                    # value-absence is the operative definition of withheld throughout v2,
                    # and check_text verifies it mechanically below; the questionnaire adds
                    # nothing here
                    ext["withheld"] = [assigned["insufficient_information"]]

            unbound_ids = [i for i in ids if i in UNBOUND_FAMILY]
            if unbound_ids:
                uid = unbound_ids[0]
                # The specialist's "exactly one requirement lacks its exact value" premise
                # is false once insufficient_information is also drawn: the WITHHELD slot
                # has no value in the text either, and the two extractors split over which
                # one to name (the k=4 check batch's bare extractor_disagreements). Remove
                # the withheld slot from the vocabulary the specialist sees -- it is not
                # "referred to" at all, and its absence is checked mechanically.
                if generalized:
                    # Hide EVERY slot another fault already claimed. The specialist's
                    # premise is "exactly one requirement lacks its exact value"; with a
                    # substitution and/or a withholding also present, that premise is false
                    # and the two extractors split over which slot to name -- 16 of 24
                    # failures in the retail k=4 diagnostic. Excluding the claimed slots
                    # makes the premise true again without weakening any evidence check.
                    claimed = {assigned.get("insufficient_information")}
                    claimed |= {str(x[0]) for x in (ext.get("substituted") or [])}
                    claimed |= {str(w) for w in (ext.get("withheld") or [])}
                    claimed.discard(None)
                    vis_conditions = tuple(c for c in conditions
                                           if str(c[0]) not in claimed)
                    if len(vis_conditions) < 1:
                        raise Rejected("composite:no_slot_left_for_unbound")
                else:
                    vis_conditions = tuple(
                        c for c in conditions
                        if str(c[0]) != assigned.get("insufficient_information"))
                if uid == "syntactic_ambiguity":
                    ext["ambiguous"] = _co_extract_syntactic(
                        vis_conditions, text, llm, dual_extract=dual_extract)
                else:
                    spec = _extract_unbound(st.BY_ID[uid], vis_conditions, text, llm,
                                            dual_extract=dual_extract)
                    ext["marked"], ext["ambiguous"] = spec["marked"], spec["ambiguous"]

            if "false_presupposition" in ids:
                ext["presupposed"] = _co_extract_fp(conditions, text, llm,
                                                    dual_extract=dual_extract,
                                                    adapter=adapter)
            if "irrelevant_information" in ids:
                ext["extras"] = _co_extract_extras(conditions, text, llm,
                                                   dual_extract=dual_extract,
                                                   adapter=adapter)

            if has_noise:
                consumed_now = ({str(x[0]) for x in ext["substituted"]}
                                | {str(w) for w in ext["withheld"]}
                                | {str(m[0]) for m in ext["marked"]}
                                | {str(p[0]) for a in ext["ambiguous"]
                                   for p in (a.get("placements") or [])})
                retained_now = {str(s) for s, _, _ in conditions} - consumed_now
                ext["offtopic_span"] = _co_extract_offtopic(
                    conditions, text, llm, retained=retained_now,
                    dual_extract=dual_extract)
                ext["noise"] = True

            # police unplanned ADDED requirements (the retired sweep's remaining job)
            # A domain that declares extras_may_be_pragmatic has no slot vocabulary for
            # an added detail, so ordinary conversational prose ("my cousin talked me into
            # it") reads as an added requirement to this policing question -- it fired on
            # every retail/airline draw. The mechanical checks that matter (no unplanned
            # substitution / withholding / unbound of a REAL slot) are already enforced by
            # the composite contract below.
            if not getattr(adapter, "extras_may_be_pragmatic", False):
                legit = [str(x[2]) for x in ext["substituted"]]
                legit += [str((p or {}).get("text")) for p in ext["presupposed"]]
                legit += [str((e or {}).get("text")) for e in ext["extras"]]
                _co_check_no_extras(conditions, legit, text, llm, adapter=adapter)

            composite_contract(combo, assigned, ext, conditions, text, adapter=adapter)

            check_text(conditions, ext, text, adapter=adapter,
                       allow_stated_ambiguous=("syntactic_ambiguity" in ids),
                       max_words=max_words)

            # ---- per-component executable admission: each fault has consequence alone
            component_sig = {}
            for strat in combo:
                want = CONTRACT[strat.id]
                sub = _empty_ext()
                if want == "substituted":
                    sub["substituted"] = ext["substituted"]
                elif want == "withheld":
                    sub["withheld"] = list(ext["withheld"])
                elif want == "marked":
                    sub["marked"] = ext["marked"]
                elif want == "ambiguous":
                    sub["ambiguous"] = ext["ambiguous"]
                elif want == "presupposed":
                    sub["presupposed"] = ext["presupposed"]
                elif want == "extras":
                    if getattr(adapter, "extras_may_be_pragmatic", False):
                        # domain has no slot for an added detail; the fork admits these on
                        # the string checks alone and marks them non-executable
                        component_sig[strat.id] = {"rule": "extras", "ok": True,
                                                   "verdict": "ok_pragmatic",
                                                   "executable": False}
                        continue
                    sub["extras"] = ext["extras"]
                else:                        # noise / oblique: no structural edit exists;
                    continue                 # their checks are textual and already ran
                component_sig[strat.id] = admit(strat.id, sub, conditions, node,
                                                adapter, session)["signature"]
        except Rejected as exc:
            last = exc.verdict
            if trace is not None:
                entry = {"attempt": attempt, "text": text, "verdict": exc.verdict}
                if getattr(exc, "dbg", None):
                    entry["dbg"] = exc.dbg
                trace.append(entry)
            continue

        # ---- composite literal reading: every fault applied at once ------------
        subs = [tuple(x) for x in ext["substituted"]]
        drops = list(ext["withheld"]) + [str(m[0]) for m in ext["marked"]] + \
                [str(p[0]) for a in ext["ambiguous"] for p in (a.get("placements") or [])]
        for p in ext["presupposed"]:
            alt = (p or {}).get("alternative")
            about = (p or {}).get("about")
            if alt:
                subs.append((alt[0], None, alt[1]))
            elif about:
                drops.append(str(about[0]))
        extras_c = [e for e in ext["extras"] if (e or {}).get("condition")]
        lit = _apply(conditions, substituted=subs, presupposed=extras_c, drop=drops)
        lit_card = None
        if lit:
            try:
                recipe = adapter.compile(dict(node.base), tuple(lit))
                lit_card = adapter.execute(recipe, session).cardinality()
            except Exception:
                lit_card = None              # a reading may be empty-of-content; that is
                                             # the "no true intent at all" case the author allowed

        hidden = sorted({str(x[0]) for x in ext["substituted"]}
                        | {str(s) for s in ext["withheld"]}
                        | {str(m[0]) for m in ext["marked"]}
                        | {str(p[0]) for a in ext["ambiguous"]
                           for p in (a.get("placements") or [])})
        from ..ids import content_hash
        mask = {
            "pipeline": "v2-composite",
            "strategy_id": "+".join(ids),
            "family": "+".join(sorted({s.family for s in combo})),
            "mask_kind": "composite",
            "kinds": sorted({s.mask_kind for s in combo}),
            "components": ids,
            "assigned_slots": assigned,
            "governing_kind": f"composite_k{len(combo)}",
            "substituted": ext["substituted"], "presupposed": ext["presupposed"],
            "withheld": ext["withheld"], "marked": ext["marked"],
            "ambiguous": ext["ambiguous"], "extras": ext["extras"],
            "noise": bool(ext["noise"]), "oblique": bool(ext["oblique"]),
            "offtopic_span": ext.get("offtopic_span"),
            "retained": sorted({str(s) for s, _, _ in conditions} - set(hidden)),
            "hidden_slots": hidden,
            "literal_readings": [list(map(list, lit))] if lit else [],
            "extractor_agreement": bool(dual_extract),
            "attempt": attempt,
        }
        mask["spec_id"] = content_hash("+".join(ids), mask["substituted"],
                                       mask["presupposed"], mask["withheld"],
                                       mask["marked"], mask["ambiguous"], mask["extras"],
                                       mask["noise"], mask["oblique"])
        signature = {"rule": "composite", "ok": True, "verdict": "ok",
                     "true_card": node.ground_truth.cardinality(),
                     "lit_card": lit_card,
                     "components": {k: v for k, v in component_sig.items()}}
        if trace is not None:
            trace.append({"attempt": attempt, "text": text, "verdict": "ok"})
        return {"mask": mask, "query": text, "signature": signature}
    raise Rejected(last)
