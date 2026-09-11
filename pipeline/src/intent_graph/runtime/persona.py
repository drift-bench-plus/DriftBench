"""The five GDMS personas, as configuration rather than as chance.

Bios are verbatim from ``drift-bench/AgentBench/src/server/mia_bench/user_simulator.py``
(dict ``persona_descriptions``).  Two things Drift-Bench does that we deliberately do not:
it re-draws the persona with ``random.choice`` on every run, and it rotates the simulator's
model across three providers per case -- both while persona-conditioned accuracy is a
reported number.  Here the persona is an explicit config value and the model is pinned.

A persona changes **what is said**, never **what is true**: acceptance is executable and
persona-independent.  ``p_decline`` is what makes ``avoidant`` actually behave differently
from ``rational``; without it the reveal guard would guard nothing.
"""

from __future__ import annotations

from dataclasses import dataclass

RATIONAL = """You are a 35-year-old financial analyst who has always prided yourself on being methodical and analytical. You work with spreadsheets and financial data daily, and you approach every decision with careful consideration and a systematic mindset. You're not impulsive—you prefer to gather all available information and analyze it thoroughly before making any choice. When you're uncertain about something, you ask precise, targeted questions to fill in the gaps in your understanding. You're patient with explanations that provide logical reasoning, but you can become frustrated with vague or incomplete information.

In interactions, you're professional and direct. When receiving clarification requests, you respond thoughtfully and ask for the specific details you need to proceed confidently. You appreciate clear, logical explanations and provide feedback on whether the information you've received is sufficient for you to move forward."""

DEPENDENT = """You are a 28-year-old recent college graduate working as a junior accountant. While you're bright and capable, you still lack confidence in many professional situations. You tend to rely heavily on the guidance and approval of more experienced colleagues and superiors. When faced with decisions, you prefer to follow established procedures or seek advice from others rather than figure things out independently. You often ask for validation and reassurance, and you feel more comfortable when someone else takes the lead in complex or unfamiliar situations.

In interactions, you're polite and deferential. When asked for clarification, you express your uncertainty openly and seek guidance from others. You're appreciative of help and often confirm that you've understood correctly. You prefer not to make independent decisions and feel more secure when following someone else's lead."""

AVOIDANT = """You are a 52-year-old marketing coordinator who has been with the same company for over 15 years. You've seen many changes in technology and workplace practices, but you prefer to stick with what you know works. You're not enthusiastic about learning new systems and often find ways to work around changes rather than adapting to them. When asked to make decisions or provide input, you tend to be non-committal and use phrases that keep your options open. You're friendly and cooperative, but you prefer to let others take the initiative.

In interactions, you're pleasant but cautious. When receiving clarification requests, you respond vaguely and avoid committing to specific answers. You use phrases that soften your responses and leave room for flexibility. You're cooperative but prefer not to take definitive stances on unfamiliar topics."""

INTUITIVE = """You are a 42-year-old graphic designer who has worked in creative fields for over 10 years. You rely heavily on your instincts and experience when making decisions, often going with what 'feels right' rather than getting bogged down in extensive analysis. You're comfortable with ambiguity and can make quick judgments based on your accumulated knowledge and gut feelings. You prefer visual and experiential learning over detailed technical explanations.

In interactions, you're creative and instinctive. When asked for clarification, you respond quickly based on your intuition and experience. You're not patient with overly technical explanations and prefer practical, hands-on guidance. You trust your instincts and make decisions based on what feels appropriate in the moment."""

SPONTANEOUS = """You are a 31-year-old social media manager who thrives in fast-paced, dynamic environments. You're energetic and adaptable, often making quick decisions based on immediate circumstances rather than extensive planning. You enjoy trying new things and aren't afraid to take risks. You're comfortable with uncertainty and prefer action over prolonged deliberation. You learn best through doing rather than reading instructions.

In interactions, you're enthusiastic and impulsive. When receiving clarification requests, you respond quickly and energetically, often suggesting immediate courses of action. You're not patient with lengthy explanations and prefer to dive in and figure things out as you go. You appreciate straightforward, practical advice but don't like to be constrained by detailed procedures."""


@dataclass(frozen=True, slots=True)
class Persona:
    id: str
    bio: str
    reveal_granularity: int   # slots revealed per ASK
    volunteer: float          # chance of adding an unasked slot
    p_decline: float          # chance of refusing an otherwise-valid reveal
    patience_mult: float
    hint: float               # chance a rejection carries a hint
    # INTENT-SHIFT SCHEDULE (ruling 2026-08-19, settled). These are NPC traits, declared as
    # part of the environment -- not hyperparameters tuned to a result. The persona decides
    # HOW MANY shifts an episode carries and WHERE each sits by default on the patience
    # axis: threshold_i = placements[i] x initial patience, and the shift fires at the end
    # of the first user exchange with patience AT or below it (at-or-below rule,
    # the author 2026-08-21; was strictly-below). A shift the
    # episode never reaches (solved first) simply does not happen -- quick, accurate
    # solving is rewarded with a calmer user.
    shift_placements: tuple[float, ...] = (1.0,)   # fractions of the persona's own initial patience
    # Agent-caused shifts: when the agent's question names a requirement that has a legal
    # edge, roll against this. On success the shift fires NOW and the LAST remaining
    # scheduled placement is dropped ("budget minus one") -- the agent can move a shift
    # earlier or (by solving fast) avoid it, but can never face more than the persona's
    # count. How readily this person is talked into revising when the agent puts something
    # in front of them.
    suggestibility: float = 0.2

    def patience(self, base: float) -> float:
        """The multiplier applies ONCE, to the total budget, EXACTLY (ruling 2026-08-13:
        "you should not round, the budget should be a float"); action prices are flat.
        Affordability at the margin is therefore decided by the true product -- 8 x 0.8
        = 6.4 affords one rejection at price 4, not the rounded pool's arithmetic."""
        return base * self.patience_mult

    def shift_thresholds(self, base: float) -> list[float]:
        """Absolute patience thresholds, highest first (the order they are crossed)."""
        init = self.patience(base)
        return sorted((f * init for f in self.shift_placements), reverse=True)


# Shift schedules (ruling 2026-08-19): rational thought it through before speaking -- one
# early change of mind, then they stick (1 @ 1.0 = the first charged exchange). Avoidant
# postpones -- one shift; its mark moved 0.5 -> 1.0 (ruling 2026-08-22, register O7:
# the mid-budget mark made shift arrival depend on the agent's own spending -- 21%
# fired vs 62-71% for start-mark personas -- so the persona's reluctance now lives in
# decline/suggestibility alone). Intuitive reacts to what they see -- two, front-loaded.
# Dependent follows the conversation -- two, evenly spread, and by far the most suggestible.
# Spontaneous is impulsive but short-fused -- two shifts inside the smallest budget, so the
# same fractions land earliest in absolute terms.
PERSONAS: dict[str, Persona] = {
    "rational":    Persona("rational",    RATIONAL,    1, 0.10, 0.00, 1.0, 0.3, (1.0,),       0.2),
    "dependent":   Persona("dependent",   DEPENDENT,   2, 0.35, 0.00, 1.2, 0.6, (0.66, 0.33), 0.8),
    "avoidant":    Persona("avoidant",    AVOIDANT,    1, 0.00, 0.50, 0.8, 0.1, (1.0,),       0.1),
    "intuitive":   Persona("intuitive",   INTUITIVE,   1, 0.20, 0.10, 0.9, 0.3, (1.0, 0.5),   0.5),
    "spontaneous": Persona("spontaneous", SPONTANEOUS, 1, 0.30, 0.05, 0.7, 0.4, (1.0, 0.5),   0.6),
}
PERSONA_IDS = tuple(sorted(PERSONAS))


def get(name: str, rng=None) -> Persona:
    """Resolve a persona name; ``random`` requires the episode RNG so it stays replayable."""
    if name == "random":
        if rng is None:
            raise ValueError("persona='random' needs the seeded episode rng")
        return PERSONAS[rng.choice(list(PERSONA_IDS))]
    if name not in PERSONAS:
        raise KeyError(f"unknown persona {name!r}; known: {PERSONA_IDS}")
    return PERSONAS[name]
