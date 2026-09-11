"""Axis A: turning a hidden intent into a user query that misrepresents it.

The design decision that makes this verifiable: **mask first, text second.**  The
corruption is a structured edit of the intent's conditions, computed mechanically; the
language model only renders that edit into words.  Drift-Bench does the opposite -- it asks
a model to "write a flawed instruction" and records only which strategy was *requested* --
which is why its perturbations cannot be audited.

Because the mask is structured, the *literal reading* of the query (what a naive reader
would take the conditions to be) is derived, never parsed back out of the text.  That is
what keeps the language model out of the ground-truth path.
"""

from __future__ import annotations

import itertools
import logging
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..ids import content_hash
from ..models import Condition
from . import strategies as st
from .llm import ROLE_RENDER

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PerturbationSpec:
    """What was hidden, what was falsified, and what a naive reading would imply."""

    strategy_id: str
    family: str
    mask_kind: str
    withheld: tuple[str, ...] = ()
    falsified: tuple[tuple[str, Any], ...] = ()
    marked: tuple[tuple[str, str], ...] = ()
    noise: bool = False
    oblique: bool = False
    retained: tuple[str, ...] = ()
    # plural: ambiguity families imply several readings.  Only one in v1, but the deferred
    # families cannot be added later if the type is singular.
    literal_readings: tuple[tuple[Condition, ...], ...] = ()
    # every mask kind present.  A single-strategy spec has exactly one; a composite (two or
    # more misalignments in one sentence) has several, and the signature rule has to reason
    # over the set rather than over one label.
    kinds: tuple[str, ...] = ()
    components: tuple[str, ...] = ()      # the strategy ids that were composed

    @property
    def hidden_slots(self) -> tuple[str, ...]:
        """Slots the agent must recover: withheld, marked-but-unbound, or falsified."""
        return tuple(sorted(
            set(self.withheld) | {s for s, _ in self.marked} | {s for s, _ in self.falsified}
        ))

    @property
    def is_composite(self) -> bool:
        return len(self.components) > 1

    def dominant_kind(self) -> str:
        """Which signature rule governs this mask.

        Order matters and is not arbitrary. A false premise makes the literal reading
        unsatisfiable no matter what else is hidden, so FALSIFY dominates. Withholding or
        marking only ever widens, so it dominates pure noise. Noise and oblique phrasing
        leave the goal untouched, so they only govern when nothing else is present.
        """
        kinds = set(self.kinds or (self.mask_kind,))
        if st.FALSIFY in kinds:
            return st.FALSIFY
        if kinds & {st.WITHHOLD, st.MARK}:
            return st.WITHHOLD
        return st.NOISE

    def spec_id(self) -> str:
        return content_hash(self.strategy_id, self.withheld, self.falsified, self.marked,
                            self.noise, self.oblique)

    def to_dict(self) -> dict:
        return {
            "strategy_id": self.strategy_id, "family": self.family,
            "mask_kind": self.mask_kind,
            "withheld": list(self.withheld),
            "falsified": [[s, v] for s, v in self.falsified],
            "marked": [[s, m] for s, m in self.marked],
            "noise": self.noise, "oblique": self.oblique,
            "retained": list(self.retained),
            "literal_readings": [[list(c) for c in r] for r in self.literal_readings],
            "hidden_slots": list(self.hidden_slots),
            "kinds": list(self.kinds or (self.mask_kind,)),
            "components": list(self.components),
            "governing_kind": self.dominant_kind(),
        }


# --------------------------------------------------------------------- masks
def derive_literal_reading(conditions: Sequence[Condition], spec_parts: dict) -> tuple[Condition, ...]:
    """What a naive reading of the rendered query implies.

    Withheld and marked slots are simply absent from the reading (marked slots are
    *mentioned* but left unbound, which constrains nothing).  Falsified slots appear with
    the bogus value, because the reader would believe it.
    """
    withheld = set(spec_parts.get("withheld", ()))
    marked = {s for s, _ in spec_parts.get("marked", ())}
    falsified = dict(spec_parts.get("falsified", ()))
    out: list[Condition] = []
    for slot, op, value in conditions:
        if slot in withheld or slot in marked:
            continue
        out.append((slot, op, falsified[slot]) if slot in falsified else (slot, op, value))
    return tuple(out)


def marker_kind_for(strategy: st.Strategy) -> str:
    """How this strategy leaves a marked slot unbound."""
    if strategy.marker_kind:
        return strategy.marker_kind
    return "vague" if strategy.needs_ordered_slot else "deictic"


def build_mask(conditions: Sequence[Condition], strategy: st.Strategy, rng, *,
               min_retained: int = 1, withhold_k_max: int = 2,
               bogus: tuple[str, Any] | None = None,
               marker: tuple[str, str] | None = None) -> PerturbationSpec:
    """Construct the structured corruption. Pure: no LLM, no execution."""
    conds = tuple(conditions)
    slots = [c[0] for c in conds]
    parts: dict[str, Any] = {}

    if strategy.mask_kind == st.WITHHOLD:
        upper = min(withhold_k_max, len(conds) - min_retained)
        if upper < 1:
            raise ValueError(f"{strategy.id}: nothing can be withheld from {len(conds)} conditions")
        k = rng.randint(1, upper)
        parts["withheld"] = tuple(sorted(rng.sample(slots, k)))

    elif strategy.mask_kind == st.FALSIFY:
        if bogus is None:
            raise ValueError(f"{strategy.id}: needs a bogus (slot, value); see bogus_candidates()")
        parts["falsified"] = (bogus,)

    elif strategy.mask_kind == st.MARK:
        if marker is None:
            eligible = slots
            if strategy.needs_ordered_slot:
                eligible = [s for s, op, v in conds if st.slot_is_ordered(s, op, v)]
            if not eligible or len(conds) - min_retained < 1:
                raise ValueError(f"{strategy.id}: no eligible slot to mark")
            kind = marker_kind_for(strategy)
            marker = (rng.choice(sorted(eligible)), kind)
        parts["marked"] = (marker,)

    elif strategy.mask_kind == st.NOISE:
        parts["noise"] = True

    elif strategy.mask_kind == st.OBLIQUE:
        parts["oblique"] = True

    else:  # pragma: no cover - guarded by the strategy table
        raise ValueError(f"unknown mask kind {strategy.mask_kind!r}")

    # `retained` means "stated faithfully".  A falsified slot IS mentioned, but wrongly, so
    # it must not also be listed as retained -- otherwise both the render prompt and the
    # template tell the user to state the true value AND assert the bogus one.
    misstated = (set(parts.get("withheld", ()))
                 | {s for s, _ in parts.get("marked", ())}
                 | {s for s, _ in parts.get("falsified", ())})
    retained = tuple(sorted(s for s in slots if s not in misstated))
    reading = derive_literal_reading(conds, parts)

    return PerturbationSpec(
        strategy_id=strategy.id, family=strategy.family, mask_kind=strategy.mask_kind,
        withheld=parts.get("withheld", ()), falsified=parts.get("falsified", ()),
        marked=parts.get("marked", ()), noise=parts.get("noise", False),
        oblique=parts.get("oblique", False), retained=retained,
        literal_readings=(reading,),
        kinds=(strategy.mask_kind,), components=(strategy.id,),
    )


def compose_masks(specs: Sequence[PerturbationSpec], conditions: Sequence[Condition],
                  *, min_retained: int = 1) -> PerturbationSpec:
    """Merge several single-strategy masks into one, for k>=2 misalignments per sentence.

    The literal reading is re-derived from the *merged* mask rather than intersected from
    the parts: withholding `attr:wool` and falsifying `price_upper` compose into a reading
    that is missing one condition and wrong about another, and only the merged parts know
    that. Deriving it any other way would let the stated query and the reading disagree,
    which is exactly the circularity the mask-first design exists to avoid.

    Refuses to merge overlapping misstatements. A slot that is both withheld and falsified
    is incoherent -- the render prompt would be told to omit it and to assert a value for
    it -- so the caller tries the next combination instead.
    """
    specs = [s for s in specs if s is not None]
    if not specs:
        raise ValueError("compose_masks: nothing to compose")
    if len(specs) == 1:
        return specs[0]

    conds = tuple(conditions)
    slots = [c[0] for c in conds]

    seen: set[str] = set()
    parts: dict[str, Any] = {"withheld": [], "falsified": [], "marked": []}
    for spec in specs:
        touched = (set(spec.withheld) | {s for s, _ in spec.marked}
                   | {s for s, _ in spec.falsified})
        clash = touched & seen
        if clash:
            raise ValueError(f"compose_masks: slots misstated twice: {sorted(clash)}")
        seen |= touched
        parts["withheld"].extend(spec.withheld)
        parts["falsified"].extend(spec.falsified)
        parts["marked"].extend(spec.marked)
    parts["noise"] = any(s.noise for s in specs)
    parts["oblique"] = any(s.oblique for s in specs)
    parts["withheld"] = tuple(sorted(set(parts["withheld"])))
    parts["falsified"] = tuple(parts["falsified"])
    parts["marked"] = tuple(parts["marked"])

    if len(slots) - len(seen) < min_retained:
        raise ValueError(
            f"compose_masks: {len(seen)} of {len(slots)} conditions misstated, "
            f"leaving fewer than min_retained={min_retained} stated faithfully"
        )

    retained = tuple(sorted(s for s in slots if s not in seen))
    reading = derive_literal_reading(conds, parts)
    ids = tuple(s.strategy_id for s in specs)
    kinds = tuple(sorted({k for s in specs for k in (s.kinds or (s.mask_kind,))}))
    families = sorted({s.family for s in specs})
    return PerturbationSpec(
        strategy_id="+".join(ids), family="+".join(families), mask_kind="composite",
        withheld=parts["withheld"], falsified=parts["falsified"], marked=parts["marked"],
        noise=parts["noise"], oblique=parts["oblique"], retained=retained,
        literal_readings=(reading,), kinds=kinds, components=ids,
    )


def _falsify_rank(slot: str) -> int:
    """Preference order for which slot to falsify: lower is tried first.

    A false premise is only usable if the signature check can confirm nothing satisfies it.
    On WebShop that rules `option:` slots out almost entirely -- the reward nearest-matches
    the options a product actually offers, so any requested option is satisfied by something.
    Attributes and the price ceiling are compared far more strictly.
    """
    if slot.startswith("attr:") or slot.startswith("where:"):
        return 0
    if slot == "price_upper" or slot.startswith(("find:", "grep:")):
        return 1
    return 2          # option: and anything else fuzzily matched


def bogus_candidates(slot: str, true_value: Any, conditions: Sequence[Condition],
                     domains: dict[str, list], rng, limit: int = 5,
                     foreign: Sequence[Any] = ()) -> list[Any]:
    """A ladder of values that plausibly do not hold, ordered most-plausible first.

    Never generated by a model: that would put an LLM inside the mask. Absence is not proved
    here either -- the signature check executes the falsified reading and requires it to
    return nothing, so a candidate that happens to be real is simply rejected.

    MEASURED, and the reason this is ordered rather than shuffled: WebShop's reward is fuzzy
    (`_best_options` picks the nearest available option, `get_type_reward` scores text
    similarity), so a real value from THIS environment very often still satisfies the intent.
    Shuffling the whole ladder and taking the first `limit` therefore discarded the reliable
    candidates at random, and the two FALSIFY strategies produced a sample on only 4 of 14
    graphs while every other strategy managed 13-14.

    Rung order is by likelihood of genuinely not holding, and each rung is shuffled only
    WITHIN itself, so variety survives without the reliable rungs becoming unreachable:

      1. a real value from a DIFFERENT environment -- the plan's rung 2, absent until now.
         Best of both: very unlikely to hold here, yet a real phrase, so the rendered query
         reads like a person who is simply mistaken rather than like corrupted text.
      2. a value from another slot's domain here (wrong kind of thing)
      3. another value from this slot's own domain (real, but wrong)
      4. mutations of the true value -- the guaranteed fallback, tried last because
         "high fructose_x" reads as gibberish rather than as a plausible mistake
    """
    seen = {str(true_value)}
    rungs: list[list[Any]] = []

    def rung(values):
        out = []
        for v in values:
            if v is None:
                continue
            s = str(v)
            if s and s not in seen:
                seen.add(s)
                out.append(v)
        rng.shuffle(out)
        rungs.append(out)

    rung(list(foreign)[:12])
    rung([v for other, values in sorted(domains.items()) if other != slot
          for v in list(values)[:3]])
    rung(list(domains.get(slot, []))[:6])

    s = str(true_value)
    mutations = []
    if s:
        mutations.append(s + "_x")
        if s.isdigit():
            mutations.append(str(int(s) + 7919))
        toks = s.split()
        if len(toks) > 1:
            mutations.append(" ".join(reversed(toks)))
        mutations.append(s[:-1] + chr((ord(s[-1]) - 32 + 1) % 95 + 32) if len(s) > 1
                         else s + "z")
    # Reserve the tail of the budget for mutations. Priority order alone is not enough: the
    # foreign rung can supply a dozen values and fill `limit` entirely, which put the
    # guaranteed-unsatisfiable rung out of reach again -- a different cause with the same
    # symptom. Mutations read as gibberish, so they are the last resort rather than the
    # first, but they must always BE reachable or a false premise cannot be guaranteed.
    mutations = [m for m in mutations if m and str(m) not in seen]
    reserve = min(max(1, limit // 4), len(mutations))
    head = [v for r in rungs for v in r][:max(1, limit - reserve)]
    head_strs = {str(v) for v in head}
    tail = [m for m in mutations if str(m) not in head_strs][:reserve]
    return (head + tail)[:limit]


# ------------------------------------------------------------------ rendering
_DEICTIC = {
    "deictic": 'refer to it only as "that one" / "it" / "the usual one"',
    "vague": 'describe it only qualitatively (e.g. "cheap", "recent", "small")',
    # both of these leave the slot unbound exactly as the two above do; only the surface form
    # differs, and the surface form has always been the renderer's job
    "polysemous": ('name it with a word that has TWO plausible meanings here, so which one you '
                   'mean is genuinely unclear (e.g. "light", "sharp", "fine")'),
    "attachment": ('mention it in a phrase whose grammar leaves it unclear WHICH thing the '
                   'detail applies to, without ever saying its value'),
}


def build_render_prompt(spec: PerturbationSpec, conditions: Sequence[Condition],
                        strategies, *, surface: str, describe=None, context: str = "") -> str:
    """The rendering instruction. Never includes `node.base`.

    For WebShop the base carries the target ASIN and full product name -- passing it would
    leak the answer straight into the opening query.

    Requirements are described in words, not as `slot = "value"`. Printing the internal
    identifier invites the model to copy it into the sentence verbatim -- which it does, and
    which `check_fidelity` used to accept because the *values* were all present. A query
    reading `attr:high fructose = "high fructose", option:0 = "citrus"` is not a user
    utterance, so it is not usable data.
    """
    say = describe or default_slot_phrase
    values = {s: v for s, _, v in conditions}
    lines = [
        "Write ONE short natural-language request from a user to an assistant.",
        "",
        f"Domain conventions: {surface}",
        *([context] if context else []),
        "",
        "Invent nothing: no product type, brand or requirement beyond what is listed below.",
        "",
    ]
    strats = [strategies] if hasattr(strategies, "id") else list(strategies)
    if len(strats) == 1:
        lines += [f"Flaw to apply: {strats[0].name} ({strats[0].id})",
                  f"What that means: {strats[0].description}",
                  f"Example of the flaw: {strats[0].example}", ""]
    else:
        # every flaw must land in the SAME sentence; listing them separately invites the
        # model to write one clause per flaw, which reads as a list rather than a request
        lines.append(f"Apply ALL {len(strats)} of these flaws at once, in one request:")
        for i, st_ in enumerate(strats, 1):
            lines.append(f"  {i}. {st_.name} ({st_.id}) - {st_.description}")
        lines.append("")
    lines += [
        "Follow these instructions EXACTLY:",
    ]
    for slot in spec.retained:
        # a retained slot is stated truthfully, so its value belongs in the prompt; the slot
        # is identified by its value where the value IS the identifier
        if st.slot_name_encodes_value(slot):
            lines.append(f'  - say plainly that you want this: "{values.get(slot)}"')
        else:
            lines.append(f'  - say plainly what you want for {say(slot)}: "{values.get(slot)}"')
    for slot in spec.withheld:
        lines.append(f"  - do NOT mention {say(slot)} at all, and do not hint at it")
    for slot, bogus in spec.falsified:
        lines.append(f'  - claim that {say(slot)} is "{bogus}" (this is deliberately wrong)')
    for slot, kind in spec.marked:
        lines.append(f"  - refer to {say(slot)} but {_DEICTIC[kind]}; never say which one")
    if spec.noise:
        lines.append("  - add a sentence of unrelated, distracting context before the request")
    if spec.oblique:
        lines.append("  - phrase the request indirectly and tentatively, not as a direct order")
    lines += [
        "",
        "Write it as a person would actually type it: plain prose, no field names, no",
        'key = "value" syntax, no colons or underscores copied from these instructions.',
        "Output the request text only: no preamble, no quotes, no explanation.",
    ]
    return "\n".join(lines)


_WORD_SAFE_MIN = 3

# Internal syntax that must never reach a user utterance. A query containing `attr:` or
# `price_upper = "120"` is not something a person typed, so it is not usable data -- and the
# old fidelity check accepted it, because every required VALUE was technically present.
_LEAK_PATTERNS = (
    re.compile(r"\b(?:attr|option|where|set|value|find|grep)\s*:", re.I),
    re.compile(r"\bprice_upper\b", re.I),
    re.compile(r"[A-Za-z_]+_[A-Za-z_]+\s*=\s*[\"']"),      # snake_case = "value"
    re.compile(r"\bslot_\d+\b", re.I),
)


def default_slot_phrase(slot: str) -> str:
    """Human phrasing for a slot when the adapter offers none.

    The default must not be the identifier itself: whatever appears in the render prompt is
    what the model copies into the query, and the fidelity check now rejects internal syntax.
    An adapter overrides this with `slot_phrase` when it can be more specific.
    """
    if slot == "price_upper":
        return "the most I want to spend"
    if slot.startswith("attr:"):
        # NEVER include the attribute text: on WebShop the slot name IS the value, and this
        # phrase is used for WITHHELD slots too -- quoting it would hand the model the very
        # thing it is being told to omit.
        return "a product feature you require"
    if slot.startswith("option:"):
        return "the product option (size, colour, flavour and so on)"
    if slot.startswith("where:"):
        return f'the "{slot[len("where:"):]}" it should match'
    if slot.startswith(("set:", "value:")):
        return f'the "{slot.split(":", 1)[1]}" to record'
    if slot.startswith("find:"):
        return f'the file\'s {slot[len("find:"):]}'
    if slot.startswith("grep:"):
        return f'the {slot[len("grep:"):]} of the text search'
    return slot.replace("_", " ").replace(":", " ")


def internal_syntax_leaks(query: str) -> list[str]:
    """Which internal identifiers appear in text meant to be a person speaking."""
    return [p.pattern for p in _LEAK_PATTERNS if p.search(query or "")]


# The model talking about its own task rather than performing it. Observed live: a render
# call returned 1,000 words of "Wait no, wait the instruction says claim the required
# feature is..." -- and fidelity passed it, because every required value was present
# somewhere in the monologue. A user utterance never discusses its own instructions.
_META_PATTERNS = (
    re.compile(r"\bthe instructions?\s+(?:say|says|said)\b", re.I),
    re.compile(r"\bdeliberately wrong\b", re.I),
    re.compile(r"\b(?:factual error|insufficient information|false presupposition|"
               r"referential ambiguity|contextual irrelevance|irrelevant information|"
               r"indirect intent|vagueness)\b", re.I),
    re.compile(r"\blet me (?:phrase|rephrase|re-?read|make it)\b", re.I),
    re.compile(r"\b(?:apply|applying) (?:both|all)? ?flaws?\b", re.I),
    re.compile(r"\bas an ai\b", re.I),
    re.compile(r"\bre-?read\b", re.I),
)


def meta_commentary(query: str) -> list[str]:
    """Signs the text is the model reasoning aloud instead of a user speaking."""
    return [p.pattern for p in _META_PATTERNS if p.search(query or "")]


def check_fidelity(query: str, spec: PerturbationSpec,
                   conditions: Sequence[Condition],
                   *, max_words: int = 120) -> tuple[bool, list[str]]:
    """Does the rendered text honour the mask? Mechanical: no model involved.

    Values shorter than three characters are skipped, with the skip reported: a condition
    like ``("find:type", "=", "f")`` would otherwise demand that the letter "f" not appear
    anywhere in an English sentence.
    """
    problems: list[str] = []
    q = query.lower()
    values = {s: str(v) for s, _, v in conditions}
    for pattern in internal_syntax_leaks(query):
        problems.append(f"internal_syntax_leaked:{pattern}")
    for pattern in meta_commentary(query):
        problems.append(f"meta_commentary:{pattern}")
    words = len((query or "").split())
    if words > max_words:
        # a shopper types a sentence or two; anything book-length is the model thinking aloud
        problems.append(f"too_long:{words}>{max_words}")

    def _surface_forms(value: str) -> list[str]:
        """Ways a person might write this value.

        Numeric conditions are stored as Python numbers -- WebShop's ``price_upper`` is the
        float ``50.0`` -- but nobody writes "under 50.0 dollars".  Without this, every
        numeric condition would look absent and every rendering would be rejected.
        """
        v = value.lower().strip()
        forms = {v}
        try:
            num = float(v)
        except (TypeError, ValueError):
            return [f for f in forms if f]
        if num.is_integer():
            forms.add(str(int(num)))
            forms.add(f"{int(num):,}")
        else:
            forms.add(f"{num:g}")
        forms.add(str(num))
        return [f for f in forms if f]

    def present(value: str) -> bool:
        forms = _surface_forms(value)
        if not forms:
            return True
        return any(re.search(rf"(?<!\w){re.escape(f)}(?!\w)", q) for f in forms)

    if len(query.split()) < 4:
        problems.append("too_short")

    for slot in spec.retained:
        val = values.get(slot, "")
        if len(val) < _WORD_SAFE_MIN:
            problems.append(f"skipped_short_retained:{slot}")
            continue
        if not present(val):
            problems.append(f"retained_missing:{slot}")

    for slot in spec.withheld:
        val = values.get(slot, "")
        if len(val) < _WORD_SAFE_MIN:
            problems.append(f"skipped_short_withheld:{slot}")
            continue
        if present(val):
            problems.append(f"withheld_leaked:{slot}")

    for slot, bogus in spec.falsified:
        if len(str(bogus)) >= _WORD_SAFE_MIN and not present(str(bogus)):
            problems.append(f"falsified_missing:{slot}")
        true_val = values.get(slot, "")
        if len(true_val) >= _WORD_SAFE_MIN and present(true_val):
            problems.append(f"falsified_but_true_value_present:{slot}")

    for slot, _ in spec.marked:
        val = values.get(slot, "")
        if len(val) >= _WORD_SAFE_MIN and present(val):
            problems.append(f"marked_leaked:{slot}")

    hard = [p for p in problems if not p.startswith("skipped_short")]
    return (not hard), problems


# ------------------------------------------------------------------ orchestration
@dataclass
class PerturbResult:
    spec: PerturbationSpec
    query: str
    attempts: int
    fidelity_notes: list[str] = field(default_factory=list)
    signature: dict | None = None


class Unperturbable(RuntimeError):
    """No admissible perturbation could be built for this node."""



def _candidate_specs(conditions, strategy, rng, *, min_retained: int, k_max: int,
                     falsify_tries: int, domains: dict | None,
                     foreign_domains: dict | None = None) -> list[PerturbationSpec]:
    """Every single-strategy mask worth trying for this strategy, best-effort ordered.

    Plural on purpose: one candidate per eligible slot for MARK, one per bogus value for
    FALSIFY. A single candidate would let one redundant slot doom an otherwise applicable
    strategy, which is how MARK used to fail.
    """
    out: list[PerturbationSpec] = []
    if strategy.mask_kind == st.FALSIFY:
        # Try several SLOTS, not one. Falsifying a single randomly chosen slot was the main
        # reason FALSIFY produced a sample on only 4 of 14 graphs: on WebShop an `option:`
        # slot can never carry a false premise, because `_best_options` nearest-matches
        # whatever the product offers, so *some* product always satisfies the claim. An
        # attribute or a price ceiling can genuinely fail to hold.
        ordered_slots = sorted(conditions, key=lambda c: (_falsify_rank(c[0]), c[0]))
        per_slot = max(2, falsify_tries // max(1, len(ordered_slots)))
        for slot, _, true_value in ordered_slots:
            for bogus in bogus_candidates(slot, true_value, conditions, domains or {}, rng,
                                          limit=per_slot,
                                          foreign=(foreign_domains or {}).get(slot, ())):
                out.append(build_mask(conditions, strategy, rng, min_retained=min_retained,
                                      withhold_k_max=k_max, bogus=(slot, bogus)))
    elif strategy.mask_kind == st.MARK:
        eligible = [s for s, op, v in conditions
                    if (not strategy.needs_ordered_slot or st.slot_is_ordered(s, op, v))
                    and not st.slot_name_encodes_value(s)]
        kind = marker_kind_for(strategy)
        for slot in sorted(eligible):
            try:
                out.append(build_mask(conditions, strategy, rng, min_retained=min_retained,
                                      withhold_k_max=k_max, marker=(slot, kind)))
            except ValueError:
                continue      # this slot cannot carry the marker; try the next one
    else:
        for _ in range(max(1, falsify_tries if strategy.mask_kind == st.WITHHOLD else 1)):
            try:
                out.append(build_mask(conditions, strategy, rng, min_retained=min_retained,
                                      withhold_k_max=k_max))
            except ValueError:
                break
    return out


def perturb(node, conditions, *, strategy=None, rng=None, llm=None,
            surface: str = "", config: dict | None = None,
            strategies=None, describe_slot=None, render_context: str = "",
            foreign_domains: dict | None = None,
            domains: dict[str, list] | None = None,
            verify: Callable[[PerturbationSpec], dict] | None = None,
            persona_tone: str = "") -> PerturbResult:
    """Build a mask, verify it if a verifier is supplied, then render it.

    ``verify`` is injected rather than imported so this module never depends on execution;
    it returns a dict with at least ``{"ok": bool}`` (see ``signature.make_verifier``).
    """
    rt = config["runtime"]
    min_retained = int(rt.get("min_retained_conditions", 1))
    k_max = int(rt.get("withhold_k_max", 2))
    falsify_tries = int(rt.get("falsify_max_attempts", 5))
    render_tries = int(rt.get("render_max_attempts", 3))
    max_combos = int(rt.get("composite_max_combinations", 24))
    max_words = int(rt.get("render_max_words", 120))

    strats = list(strategies) if strategies else [strategy]
    strats = [s for s in strats if s is not None]
    if not strats:
        raise ValueError("perturb: no strategy given")
    label = "+".join(s.id for s in strats)

    # ---- build a mask that survives the signature check --------------------
    per_strategy = [
        _candidate_specs(conditions, s, rng, min_retained=min_retained, k_max=k_max,
                         falsify_tries=falsify_tries, domains=domains,
                         foreign_domains=foreign_domains)
        for s in strats
    ]
    for s, cands in zip(strats, per_strategy, strict=True):
        if not cands:
            raise Unperturbable(f"{s.id}: no candidate mask")

    if len(strats) == 1:
        candidates = per_strategy[0]
    else:
        # Compose across strategies. Combinations whose misstated slots overlap are
        # incoherent and skipped rather than fixed up, so a composite either honours every
        # component exactly or is not produced at all.
        candidates = []
        skipped = 0
        for combo in itertools.islice(itertools.product(*per_strategy), max_combos):
            try:
                candidates.append(compose_masks(combo, conditions,
                                                min_retained=min_retained))
            except ValueError:
                skipped += 1
        if not candidates:
            raise Unperturbable(
                f"{label}: no coherent composite over {len(conditions)} conditions "
                f"({skipped} combinations overlapped or breached min_retained)"
            )

    if not candidates:
        raise Unperturbable(f"{label}: no candidate mask")

    spec = None
    sig = None
    for cand in candidates:
        if verify is None:
            spec, sig = cand, None
            break
        result = verify(cand)
        if result.get("ok"):
            spec, sig = cand, result
            break
        sig = result
    if spec is None:
        raise Unperturbable(
            f"{label}: no mask passed the signature check "
            f"(last verdict: {sig.get('verdict') if sig else 'n/a'})"
        )

    # ---- render, then check the text against the mask ----------------------
    prompt = build_render_prompt(spec, conditions, strats, surface=surface,
                                 describe=describe_slot, context=render_context)
    if persona_tone:
        prompt += f"\n\nWrite it in this person's voice:\n{persona_tone}"
    notes: list[str] = []
    for attempt in range(1, render_tries + 1):
        query = llm.complete(prompt, role=ROLE_RENDER).strip().strip('"')
        ok, problems = check_fidelity(query, spec, conditions, max_words=max_words)
        notes = problems
        if ok:
            return PerturbResult(spec=spec, query=query, attempts=attempt,
                                 fidelity_notes=problems, signature=sig)
        prompt_retry = prompt + (
            "\n\nYour previous attempt violated these instructions: "
            + ", ".join(p for p in problems if not p.startswith("skipped_short"))
            + ". Try again and follow them exactly."
        )
        prompt = prompt_retry

    fallback = template_render(spec, conditions, describe_slot)
    ok, problems = check_fidelity(fallback, spec, conditions, max_words=max_words)
    if not ok:
        raise Unperturbable(f"{label}: template fallback also failed: {problems}")
    log.info("perturb: fell back to template for %s (%s)", label, notes)
    return PerturbResult(spec=spec, query=fallback, attempts=render_tries + 1,
                         fidelity_notes=notes + ["template_fallback"], signature=sig)


def _clause(slot: str, value: str) -> str:
    """A speakable clause for one condition, for the model-free fallback.

    Separate from `default_slot_phrase`, which describes a slot *as a dimension* ("the most I
    want to spend") for instructions. Dropped into a sentence that reads
    "I need something with the most I want to spend 120", which is not English.
    """
    if slot == "price_upper":
        return f"under {value} dollars"
    if slot.startswith("option:"):
        return f"in {value}"
    if slot.startswith("attr:"):
        return f"that is {value}"
    if slot.startswith("where:"):
        return f"where the {slot[len('where:'):]} is {value}"
    return f"with {slot.split(':')[-1]} {value}"


def template_render(spec: PerturbationSpec, conditions: Sequence[Condition],
                    describe=None) -> str:
    """A model-free rendering, used when the LLM will not follow the mask.

    Must obey the same no-internal-syntax rule as the LLM path: this text is shown to an
    agent as a user utterance, and "where price_upper is 120" is not one. `price_upper` is
    the case that bites -- a bare identifier with no colon to strip.
    """
    values = {s: str(v) for s, _, v in conditions}
    stated = [_clause(s, values[s]) for s in spec.retained if s in values]
    parts = ["I need something"]
    if stated:
        parts = ["I need something " + ", ".join(stated)]
    for slot, bogus in spec.falsified:
        parts.append(f"and {_clause(slot, bogus)}")
    for _slot, kind in spec.marked:
        parts.append("like that one" if kind == "deictic" else "a reasonable one")
    if spec.noise:
        parts.insert(0, "By the way, unrelated, but my week has been hectic.")
    text = " ".join(parts) + "."
    return text if len(text.split()) >= 4 else text + " Please help me with this."


def surface_conventions(adapter_name: str) -> str:
    """How a user of this domain would naturally speak."""
    return {
        "dbbench": "the user is asking about rows in a single database table, in plain words",
        "osbench": "the user is asking about files and directories on their Linux machine",
        "webshop": "the user is shopping in an online store and describes what they want to buy",
        "toybench": "the user is picking an item from a small catalogue",
    }.get(adapter_name, "plain conversational English")
