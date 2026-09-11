"""Drift-Bench's flaw taxonomy, verbatim, plus applicability rules.

Descriptions and examples are copied unchanged from
``drift-bench/dataset_generation/mia_bench_flaw_strategies.py`` because they go into the
rendering prompt -- paraphrasing them changes the data.

Eight of the eleven strategies are *mechanically maskable*: the corruption can be expressed
as a structured edit of the intent's conditions.  The other three need a generated surface
form (a polysemous word, an attachment-ambiguous rewrite, a bogus tool), which would put a
language model inside the mask and therefore inside the ground-truth path.  Those are
marked ``supported=False`` and skipped, not faked.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)

# mask kinds a strategy uses
WITHHOLD = "withhold"     # slot not mentioned at all
FALSIFY = "falsify"       # slot asserted with a value that does not hold
MARK = "mark"             # slot mentioned but left unbound ("that one", "cheap")
NOISE = "noise"           # irrelevant material added; the goal itself is untouched
OBLIQUE = "oblique"       # goal stated indirectly; nothing withheld


@dataclass(frozen=True, slots=True)
class Strategy:
    id: str
    family: str
    name: str
    description: str
    example: str
    mask_kind: str
    supported: bool = True
    needs_conditions: int = 0       # minimum conditions on the node
    needs_ordered_slot: bool = False
    # how the renderer should leave a marked slot unbound; None => derived from the strategy
    marker_kind: str | None = None
    needs_spare_condition: bool = False  # after masking, >=1 condition must remain stated


STRATEGIES: tuple[Strategy, ...] = (
    # ---------------------------------------------------------------- intention
    Strategy(
        id="indirect_intent", family="intention", name="Indirect Intent",
        description=(
            "Rewrites a direct question or request into an indirect, question-based form "
            "while preserving the core intent. The question becomes more tentative, "
            "exploratory, or questioning rather than direct."
        ),
        example=("Original: 'Can you check tomorrow's weather?' -> Rewritten: 'I was "
                 "wondering if it's possible to know what the weather might be like tomorrow?'"),
        mask_kind=OBLIQUE,
    ),
    Strategy(
        id="contextual_irrelevance", family="intention", name="Contextual Irrelevance",
        description=(
            "Adds completely unrelated or even contradicting context or content that "
            "interferes with understanding the intention of the question. The added content "
            "is about completely different topics that distract from the actual purpose of "
            "the question."
        ),
        example=("Original: 'What is the total number of medals?' -> Rewritten: 'Before "
                 "answering, I'm curious about Tesla stock. But anyway, what is the total "
                 "number of medals?'"),
        mask_kind=NOISE,
    ),
    # ------------------------------------------------------------------ premise
    Strategy(
        id="factual_error", family="premise", name="Factual Error",
        description=("Replaces correct entities with incorrect, non-existent entities "
                     "(e.g., replacing 'United Airlines' with 'Pan Am')"),
        example="Original: 'Book a United Airlines flight' -> Rewritten: 'Book a Pan Am flight'",
        mask_kind=FALSIFY, needs_conditions=1,
    ),
    Strategy(
        id="false_presupposition", family="premise", name="False Presupposition",
        description=("Adds false presupposition conditions, assuming that a non-existent "
                     "fact is true"),
        example=("Original: 'Find the winner' -> Rewritten: 'Find the winner of the 2025 "
                 "tournament' (assuming there is a 2025 tournament)"),
        mask_kind=FALSIFY, needs_conditions=1,
    ),
    Strategy(
        id="tool_capability_mismatch", family="premise", name="Tool/Capability Mismatch",
        description=("Requires using mismatched tools or capabilities (for database query "
                     "tasks, can imply using wrong table or column)"),
        example="Original: 'Query the table' -> Rewritten: 'Use the weather API to query the table'",
        # STILL UNSUPPORTED, and dropped rather than faked for WebShop: this flaw is about
    # choosing the wrong INSTRUMENT ("use the weather API to query the table"), and WebShop
    # offers exactly one (search + click), so there is nothing to mismatch. Its natural home
    # is dbbench, where "wrong table or column" is literally the example.
    mask_kind=NOISE, supported=False,
    ),
    # ---------------------------------------------------------------- parameter
    Strategy(
        id="insufficient_information", family="parameter", name="Insufficient Information",
        description=(
            "Removes key parameters, dates, numbers, or specific details from the question "
            "that would be necessary to answer it properly, making it incomplete or ambiguous"
        ),
        example=("Original: 'Find flights from SFO to JFK on March 15' -> Rewritten: "
                 "'Find flights'"),
        mask_kind=WITHHOLD, needs_conditions=2, needs_spare_condition=True,
    ),
    Strategy(
        id="irrelevant_information", family="parameter", name="Irrelevant Information",
        description=("Adds redundant, repetitive, or irrelevant information to the question "
                     "that distracts from the core task"),
        example=("Original: 'Book a ticket from SFO to JFK for Andy' -> Rewritten: 'My "
                 "colleague David just got back from LAX and is exhausted... Book me a "
                 "ticket from SFO to JFK for Andy.'"),
        mask_kind=NOISE,
    ),
    # --------------------------------------------------------------- expression
    Strategy(
        id="lexical_ambiguity", family="expression", name="Lexical Ambiguity",
        description="Uses ambiguous vocabulary (polysemy)",
        example=("Original: 'Find a bat' -> Rewritten: 'Find a bat' (bat can be an animal "
                 "or a baseball bat)"),
        # SUPPORTED: structurally this is MARK -- the slot is mentioned but left unbound.
        # Only the WORDING differs from referential_ambiguity (a polysemous word rather than
        # "that one"), and the wording is the renderer's job, which it already is for all the
        # other strategies. The mask, literal reading and signature check are unchanged.
        mask_kind=MARK, marker_kind="polysemous",
        needs_conditions=2, needs_spare_condition=True,
    ),
    Strategy(
        id="syntactic_ambiguity", family="expression", name="Syntactic Ambiguity",
        description="Uses syntactic structure that makes modification relationships unclear",
        example=("Original: 'Find a hotel near the museum' -> Rewritten: 'Find a hotel near "
                 "the museum with free parking' (does free parking modify hotel or museum?)"),
        # SUPPORTED for the same reason: an attachment-ambiguous phrasing leaves the slot
        # unbound, which is exactly MARK.
        mask_kind=MARK, marker_kind="attachment",
        needs_conditions=2, needs_spare_condition=True,
    ),
    Strategy(
        id="referential_ambiguity", family="expression", name="Referential Ambiguity",
        description="Uses unclear referential terms (e.g., 'it', 'that', 'this')",
        example="Original: 'Find the winner' -> Rewritten: 'Find that one'",
        mask_kind=MARK, needs_conditions=2, needs_spare_condition=True,
    ),
    Strategy(
        id="vagueness_subjectivity", family="expression", name="Vagueness/Subjectivity",
        description=("Uses subjective, vague descriptive words (e.g., 'best', 'reasonable', "
                     "'good')"),
        example=("Original: 'Find an Italian restaurant' -> Rewritten: 'Find a reasonable "
                 "Italian restaurant'"),
        mask_kind=MARK, needs_conditions=2, needs_spare_condition=True, needs_ordered_slot=True,
    ),
)

BY_ID = {s.id: s for s in STRATEGIES}
SUPPORTED = tuple(s for s in STRATEGIES if s.supported)
FAMILIES = ("intention", "premise", "parameter", "expression")

# slots whose domain has an order, so "cheap"/"recent" is a meaningful vague surface form
_ORDERED_PREFIXES = ("price_upper", "find:size", "find:mtime")
_ORDERED_OPS = ("<", ">", "<=", ">=")


def slot_name_encodes_value(slot: str) -> bool:
    """Would naming this slot reveal its value?

    On WebShop an attribute condition is `attr:<the attribute>`, so the identifier IS the
    value. A MARK strategy says "mention this requirement but stay vague about it" -- which
    is impossible here: mentioning it states the value outright, while the literal reading
    drops the slot, so the mask and the query would disagree about what was conveyed. Such
    slots are therefore ineligible for marking rather than silently mismarked.
    """
    return slot.startswith("attr:")


def slot_is_ordered(slot: str, op: str, value, adapter=None) -> bool:
    """Can this requirement carry a SUBJECTIVE surface ("cheap", "flaky", "the better one")?

    On WebShop the answer was numeric order: "cheap" means something only against a price.
    That is a property of WebShop's slots, not of the fault type -- Drift-bench defines
    vagueness/subjectivity as "uses subjective, vague descriptive words (best, reasonable,
    good)", which a domain can support through a quality dimension rather than a number
    ("my connection is being flaky", "swap it for a nicer one"). Adapters may therefore
    declare which of their slots take a subjective surface; without such a declaration the
    numeric rule stands, so WebShop behaviour is unchanged.
    """
    fn = getattr(adapter, "slot_supports_subjective", None)
    if fn is not None:
        try:
            if fn(slot):
                return True
        except Exception as exc:
            # OPTIONAL adapter hook. A broken one must not kill generation, but it must
            # not vanish either: without the log the adapter's declaration is silently
            # ignored and the numeric rule takes over as if none had been declared.
            log.debug("slot_supports_subjective failed for %s: %s", slot, exc)
    if slot in _ORDERED_PREFIXES or slot.startswith(_ORDERED_PREFIXES):
        return True
    if op in _ORDERED_OPS:
        return True
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def applicable(strategy: Strategy | str, conditions, *, min_retained: int = 1,
               adapter=None) -> bool:
    """Can this strategy be expressed as a mask over these conditions?"""
    s = strategy if isinstance(strategy, Strategy) else BY_ID[strategy]
    if not s.supported:
        return False
    n = len(conditions)
    if n < s.needs_conditions:
        return False
    if s.needs_spare_condition and n - min_retained < 1:
        return False
    if s.needs_ordered_slot and not any(
        slot_is_ordered(sl, op, v, adapter) for sl, op, v in conditions
    ):
        return False
    if s.mask_kind == MARK and not any(
        (not s.needs_ordered_slot or slot_is_ordered(sl, op, v))
        and not slot_name_encodes_value(sl) for sl, op, v in conditions
    ):
        # every candidate slot would reveal its own value by being named
        return False
    return True


def applicable_strategies(conditions, *, min_retained: int = 1) -> tuple[Strategy, ...]:
    return tuple(s for s in SUPPORTED
                 if applicable(s, conditions, min_retained=min_retained))


def sample_strategy(conditions, rng, *, min_retained: int = 1,
                    strategy_probs: dict | None = None) -> Strategy | None:
    """Family first, then strategy within it.

    Uniform sampling over the eleven strategies would over-weight ``expression`` 4:2:3:2;
    sampling the family first keeps the four flaw types balanced, which is the same reason
    intent-shift categories are sampled by category before edge (see traversal).
    """
    pool = applicable_strategies(conditions, min_retained=min_retained)
    if not pool:
        return None
    if strategy_probs:
        weights = [float(strategy_probs.get(s.id, 0.0)) for s in pool]
        if sum(weights) > 0:
            return rng.choices(pool, weights=weights, k=1)[0]
    families = sorted({s.family for s in pool})
    family = rng.choice(families)
    return rng.choice(sorted([s for s in pool if s.family == family], key=lambda s: s.id))


def sample_strategies(conditions, rng, *, k: int = 1, min_retained: int = 1,
                      strategy_probs: dict | None = None) -> list[Strategy]:
    """Pick ``k`` strategies to apply to the SAME query.

    One misalignment per sentence is the normal case and the default. Composing several is
    the difficulty dial: two flaws in one request is meaningfully harder than one, because
    the agent must notice that fixing one still leaves the request wrong.

    Distinct mask kinds are preferred over distinct strategies. Two strategies that both
    withhold compose into "withhold more", which is the ``withhold_k_max`` knob rather than
    a second kind of flaw; withhold + falsify is genuinely two flaws. Falls back to any
    distinct strategy, then to fewer than k, rather than failing -- a short composite is
    better than no episode.
    """
    pool = applicable_strategies(conditions, min_retained=min_retained)
    if not pool or k < 1:
        return []
    first = sample_strategy(conditions, rng, min_retained=min_retained,
                            strategy_probs=strategy_probs)
    if first is None:
        return []
    chosen = [first]
    if k == 1:
        return chosen

    # A composite must leave min_retained conditions faithfully stated, and every component
    # that hides or falsifies a slot consumes one. That caps how many can stack.
    budget = max(1, len(tuple(conditions)) - min_retained)

    def consumes(s: Strategy) -> int:
        return 0 if s.mask_kind in (NOISE, OBLIQUE) else 1

    used = sum(consumes(s) for s in chosen)
    for _ in range(k - 1):
        used_kinds = {s.mask_kind for s in chosen}
        remaining = [s for s in pool if s.id not in {c.id for c in chosen}
                     and used + consumes(s) <= budget]
        if not remaining:
            break
        fresh_kind = [s for s in remaining if s.mask_kind not in used_kinds]
        candidates = fresh_kind or remaining
        families = sorted({s.family for s in candidates})
        family = rng.choice(families)
        pick = rng.choice(sorted([s for s in candidates if s.family == family],
                                 key=lambda s: s.id))
        chosen.append(pick)
        used += consumes(pick)
    return chosen
