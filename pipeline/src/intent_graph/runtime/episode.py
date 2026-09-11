"""The episode loop: where the two axes meet.

The user holds a hidden intent (which axis B may move) and says things that misrepresent it
(axis A).  The agent must interact to find out what is meant.  The loop's one hard rule is
that **acceptance is executable**: "the user says okay" means the proposal satisfies the
node current at that moment, checked by the benchmark's own comparison.  A correct-but-stale
answer is therefore rejected, which is the entire point of moving the intent.
"""

from __future__ import annotations

import json
import logging
import random
import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from ..ids import canonical_dumps, config_hash
from ..errors import EnvTransportError
from ..models import Graph
from . import accept as accept_mod
from . import persona as persona_mod
from . import perturb as perturb_mod
from . import signature as signature_mod
from . import strategies as st
from .agent_api import ActionKind, Transcript, parse_action
from . import traversal as traversal_mod
from .traversal import Traversal
from .user import SimulatedUser

log = logging.getLogger(__name__)


class Outcome(StrEnum):
    SUCCESS = "SUCCESS"
    EXHAUSTED = "EXHAUSTED"          # patience ran out
    TURNS_EXCEEDED = "TURNS_EXCEEDED"
    ERROR = "ERROR"


def _acceptance_reward(ok: bool, why: str) -> float:
    """Numeric reward of an adjudicated proposal, recorded alongside ok/reason.

    WHY (ruling 2026-08-21): acceptance is a 1.0 cliff, so Success alone cannot see that
    a more accurate near-miss is better than a blind one -- a proposal at reward 0.8 and
    one at 0.2 both scored "rejected", which erased exactly the accuracy that asking buys.
    The episode's continuous SCORE is the reward of its final adjudication (original
    WebShop reports average score alongside success for the same reason). The reward was
    always computed by the acceptance check; it was just never stored as a number.
    """
    if ok:
        return 1.0
    m = re.search(r"reward=([0-9.]+)", str(why or ""))
    return float(m.group(1)) if m else 0.0


class _Preset:
    """A perturbation replayed from an exported sample (duck-types PerturbResult)."""

    __slots__ = ("spec", "query", "signature", "fidelity_notes", "attempts")

    def __init__(self, sample: dict) -> None:
        self.spec = None                      # header takes the mask dict directly
        self.query = sample["query"]
        self.signature = sample.get("signature")
        self.fidelity_notes = list(sample.get("fidelity_notes") or ())
        self.attempts = int(sample.get("render_attempts", 1))


def _preset_perturbation(sample: dict):
    return _Preset(sample), None


@dataclass
class Trajectory:
    """Everything needed to score the episode and to replay it exactly."""

    header: dict
    turns: list[dict] = field(default_factory=list)
    outcome: str = Outcome.ERROR.value
    final_node: str | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        return {"header": self.header, "turns": self.turns, "outcome": self.outcome,
                "final_node": self.final_node, "error": self.error}


class Episode:
    def __init__(self, *, graph: Graph, adapter, executor, config: dict, llm,
                 agent, persona_name: str | None = None, seed: int | None = None,
                 preset: dict | None = None) -> None:
        self.graph = graph
        self.adapter = adapter
        self.executor = executor
        self.config = config
        self.llm = llm
        self.agent = agent
        rt = config["runtime"]
        self.rt = rt
        if bool(rt.get("perturb_shift_announcements", False)):
            # Obsolete flag: shifts are now SILENT (no announcement exists to perturb).
            # Refuse loudly if a config still sets it, rather than ignoring it.
            # Declared in the plan, never built. Perturbing an announcement needs a SECOND
            # mask whose literal reading is verified against the shifted node -- the axis-A
            # machinery over an edge delta rather than a condition set. Templating the
            # announcement while the config claims otherwise is the worse failure, so this
            # refuses at construction: a misconfiguration is not an episode outcome.
            raise NotImplementedError(
                "runtime.perturb_shift_announcements is not implemented: shift "
                "announcements are always templated (see traversal.announce). Set it to "
                "false, or implement masking over the edge delta first."
            )
        self.seed = int(rt.get("episode_seed", 0) if seed is None else seed)
        self.rng = random.Random(f"{graph.graph_id}:{self.seed}")
        self.persona = persona_mod.get(persona_name or rt.get("persona", "rational"), self.rng)
        self.max_turns = int(rt.get("max_turns", 16))
        self.nodes = {n.intent_id: n for n in (graph.root, *graph.children)}
        # An exported sample carries an already-verified perturbation (query + mask +
        # signature). Experiments replay THAT, rather than re-deriving one: the dataset the
        # verifier accepted is the dataset the agent must face, and re-deriving would silently
        # depend on cache state and RNG alignment.
        self.preset = preset

    # ------------------------------------------------------------------ setup
    def _is_write_graph(self) -> bool:
        return self.graph.root.ground_truth.kind == "statehash"

    def run(self) -> Trajectory:
        header = {
            "graph_id": self.graph.graph_id, "adapter": self.graph.adapter,
            "env_id": self.graph.env_id, "config_hash": config_hash(self.config),
            "episode_seed": self.seed, "persona": self.persona.id,
            "is_write_graph": self._is_write_graph(),
        }
        traj = Trajectory(header=header)

        try:
            with self.executor.open(self.graph.env_spec) as session:
                return self._play(session, traj)
        except Exception as exc:            # never let a broken environment look like a fail
            log.warning("episode %s errored: %s", self.graph.graph_id, exc)
            traj.outcome = Outcome.ERROR.value
            traj.error = f"{type(exc).__name__}: {exc}"
            return traj

    # ------------------------------------------------------------------- play
    def _play(self, session, traj: Trajectory) -> Trajectory:
        rt = self.rt
        current = self.graph.root

        # --- axis A: render the opening query -------------------------------
        pristine = (signature_mod.pristine_state_hash(self.adapter, session)
                    if self._is_write_graph() else None)
        if self.preset is not None:
            pert, strategy = _preset_perturbation(self.preset)
        else:
            pert, strategy = self._perturb_root(current, session, pristine)
        if pert is None:
            traj.outcome = Outcome.ERROR.value
            traj.error = strategy            # the reason, when nothing worked
            return traj
        # THE PAGE MUST SHOW THIS EPISODE'S QUERY (2026-08-20). The faithful WebShop site
        # prints an instruction on every page, drawn from its own goal list. Left alone the
        # agent is handed TWO contradictory tasks -- ours in the conversation, WebShop's on
        # screen -- and follows whichever it read last. Measured: an episode asked for
        # safety footwear searched footwear on turn 1, saw a page demanding an aqua octagonal
        # area rug, and spent the rest of its budget chasing rugs. The session cannot be told
        # at open() because the perturbed query does not exist until here, and the live
        # session connects lazily precisely so this can be set first. Sessions without an
        # on-page instruction (every non-WebShop benchmark renders the task through its
        # adapter instead) simply lack the attribute, and nothing happens here.
        if hasattr(session, "instruction"):
            session.instruction = pert.query
        traj.header["perturbation"] = (self.preset["mask"] if self.preset is not None
                                       else pert.spec.to_dict())
        if self.preset is not None:
            traj.header["strategy_id"] = "+".join(self.preset["strategy_ids"])
            traj.header["strategy_ids"] = list(self.preset["strategy_ids"])
            traj.header["misalignments"] = int(self.preset.get("misalignments", 1))
            traj.header["sample_id"] = self.preset.get("sample_id")
        else:
            # `strategy` is one Strategy for a single misalignment, or a list for a composite
            used = strategy if isinstance(strategy, list) else [strategy]
            traj.header["strategy_id"] = "+".join(s.id for s in used)
            traj.header["strategy_ids"] = [s.id for s in used]
            traj.header["misalignments"] = len(used)
        traj.header["requested_misalignments"] = max(1, int(rt.get("misalignments_k", 1)))
        traj.header["signature"] = pert.signature
        traj.header["query"] = pert.query
        traj.header["fidelity_notes"] = pert.fidelity_notes

        # --- state ----------------------------------------------------------
        if bool(rt.get("user_v2", False)):
            from .user import SimulatedUserV2
            user = SimulatedUserV2(self.persona, self.llm, self.rng, self.config)
            user.initial_query = pert.query
            mask = traj.header.get("perturbation") or {}
            hidden0 = set(mask.get("hidden_slots") or ())
            # slots the opening request already stated; everything else is PRIVATE until
            # the user itself says it (shift-added slots are private automatically: their
            # names are absent from this set)
            user.stated_slots = {sl for sl, _, _ in current.conditions} - hidden0
        else:
            user = SimulatedUser(self.persona, self.llm, self.rng, self.config)
        # The user speaks the ADAPTER's plain words for slots and values, never internal
        # identifiers ("a requirement" menus and label-as-value both measured fatal
        # 2026-08-14 -- recovery was structurally zero without these). The family is
        # wired only for adapters that declare a roleplay voice (all the tau2 benchmarks
        # do); every other adapter leaves the user BARE, keeping the user's own defaults
        # -- type_label rendering, the "online shopper" voice, and the reveal-grounding
        # floor -- which is the original WebShop behaviour unchanged. The user keys its
        # benchmark-specific reveal and coherence semantics off this wiring (see
        # user._adapter_wired), so the presence-check here is what selects the pipeline.
        if getattr(self.adapter, "roleplay_voice", None) is not None:
            user.slot_phrase = getattr(self.adapter, "slot_phrase", None)
            user.spoken_value = getattr(self.adapter, "spoken_value", None)
            user.voice = getattr(self.adapter, "roleplay_voice", None) or "online shopper"
            user.proposal_frame = getattr(self.adapter, "proposal_frame", None)
            user.adapter = self.adapter      # for context_facts (see user.answer_ask)
        self._react_session = session
        axis_b = (not self._is_write_graph()) or bool(rt.get("axis_b_on_write_graphs",
                                       rt.get("axis_b_on_write_trees", False)))
        # THE THIRD STREAM (v4 design, 2026-08-18): the destination draw must not share the
        # user's RNG, or the class realised at fire time becomes a function of how much the
        # agent talked (measured: varying prior draws changed the class in 60/60 graphs).
        traversal = Traversal(self.graph, self.config,
                              random.Random(f"{self.graph.graph_id}:{self.seed}:shift-destination"),
                              self.adapter)
        if not axis_b:
            traversal.p_shift = 0.0
        traversal.mark_visited(current.intent_id)

        # THE ECONOMY (ruling 2026-08-13, settled): the persona multiplier applies ONCE,
        # to the total budget, as an EXACT float -- 8 x 1.2 = 9.6, unrounded -- and every
        # action then charges its flat price from that pool. The affordability rule does
        # the rest: an action costing more than what remains cannot be bought (unaffordable
        # PROPOSE = the adjudicated final shot, unaffordable ASK = refused, forced final
        # submission), so patience can never go negative.
        shift_scheduled = bool(rt.get("shift_scheduled", False))
        shift_rng = random.Random(f"{self.graph.graph_id}:{self.seed}:shift-schedule")

        patience = self.persona.patience(float(rt.get("patience_init", 14)))
        # THE SHIFT SCHEDULE (ruling 2026-08-19, settled -- the "double layer" design):
        #   1) the PERSONA decides how many shifts this episode carries and where each sits
        #      by default on the PATIENCE axis (threshold = placement x initial patience);
        #      a shift fires at the end of the first user exchange after patience drops
        #      strictly below its threshold. An episode solved before a threshold is
        #      reached simply carries fewer shifts -- quick, accurate solving is rewarded.
        #   2) the AGENT can cause a shift early: a question that names a requirement with
        #      a legal edge rolls against the persona's suggestibility; on success the
        #      shift fires NOW and the LAST remaining threshold is dropped ("budget minus
        #      one"). Asking can move shifts earlier or (by solving fast) avoid them, but
        #      can never increase their number -- clarity is never penalised.
        # mode "persona" is the settled design; "coin" is the legacy 2026-08-14 draw kept
        # for replaying earlier campaigns.
        shift_mode = str(rt.get("shift_schedule", "persona"))
        if shift_scheduled and shift_mode == "persona" and axis_b:
            shift_thresholds = self.persona.shift_thresholds(float(rt.get("patience_init", 14)))
            traversal.max_shifts = len(shift_thresholds)   # persona count supersedes the cap
            shift_pending = False                          # legacy flag unused in this mode
        else:
            shift_thresholds = []
            shift_pending = shift_scheduled and (shift_rng.random() < traversal.p_shift)
        shift_due = 0          # thresholds crossed whose shift has not fired yet
        trigger_rng = random.Random(f"{self.graph.graph_id}:{self.seed}:shift-trigger")
        # SHIFT FIRE COIN (ruling 2026-08-21; ported from tau2 2026-08-22). Under the
        # at-or-below rule every episode crossed its first threshold at start, so 100% of
        # episodes shifted and PostShift degenerated into Score -- the benchmark lost its
        # static-goal population entirely. The fix: crossing a threshold CONSUMES it either
        # way, but the intent actually moves only with probability shift_fire_prob; a
        # suppressed crossing is recorded as `shift_skipped` (a shape both scorers ignore --
        # it must NOT be a shift record with gt_moved=False, which would feed over-reaction).
        # Fates are PRE-DRAWN from (graph, seed) alone -- never the live RNG stream, whose
        # position depends on the agent's own actions and would hand different methods
        # different goal scripts on the same sample. Same sample, same seed => same fates in
        # every method; the paired design stays clean. Coins independent across placements.
        # SCHEDULED shifts only: triggered shifts keep suggestibility as their own gate
        # (ruling 2026-08-22: "this doesn't replace the triggered prob") -- they consume the
        # last placement but ignore its fate.
        fire_prob = float(rt.get("shift_fire_prob", 1.0))
        fire_rng = random.Random(f"{self.graph.graph_id}:{self.seed}:shift-fire")
        shift_fates = [fire_rng.random() < fire_prob for _ in shift_thresholds]
        # ABLATION SEMANTICS (ruling 2026-08-27): fire_prob == 0.0 exactly means NO
        # SHIFTING AT ALL -- scheduled fires are all skip-fated above, and the TRIGGERED
        # path must be off too ("not only the shift_fire_prob, but also the intent will
        # not shift because of agent"). Triggers deliberately ignore fates at 0<p<1
        # (ruling 2026-08-22: "this doesn't replace the triggered prob"), so that regime
        # is untouched; only the exact-zero ablation disables them.
        # SCOPE: the exact-zero ruling was adjudicated on the WebShop-family benchmarks
        # and applies to them. The tau2 benchmarks keep the coin's original 2026-08-21
        # contract, under which the trigger channel is exempt at EVERY fire_prob (its
        # suggestibility roll is its own gate); the benchmark in use decides.
        shifting_disabled = (fire_prob == 0.0 and
                             self.graph.adapter in ("webshop", "dbbench", "osbench"))
        # WHICH CATEGORIES THIS GRAPH CAN ACTUALLY OFFER (2026-08-19). Partial graphs are
        # kept on purpose, so a graph may have no edge of some kind. `on_empty_category`
        # then decides what happens to that kind's probability, and with "resample" the mass
        # silently moves to whatever IS available. Measured on tau2 retail: a configured
        # 20% PIVOT became a realised 55%. Recording what was on offer makes the realised
        # mix auditable per episode instead of an invisible property of the corpus.
        traj.header["shift_categories_available"] = sorted(
            op.value for op, edges in traversal._available(current.intent_id).items() if edges)
        traj.header["shift_schedule"] = {
            "mode": shift_mode if shift_scheduled else "off",
            "thresholds": [round(t, 2) for t in shift_thresholds],
            # the pre-drawn goal script, auditable per episode and identical across methods
            "fire_prob": fire_prob,
            "fates": ["fire" if f else "skip" for f in shift_fates],
            "suggestibility": self.persona.suggestibility,
        }
        cost_ask = float(rt.get("cost_ask", 1))
        cost_reject = float(rt.get("cost_reject", 2))
        cost_act = float(rt.get("cost_act", 0))
        cost_malformed = float(rt.get("cost_malformed", 1))
        EPS = 1e-9
        transcript = Transcript()
        transcript.user(pert.query)
        mutated = False
        # An environment Operation may BE a proposal in an adapter's own surface syntax
        # (WebShop spells one "buy <asin> {...}").  The loop knows only PROPOSAL and
        # SUBMISSION -- the two terms that hold for every benchmark; recognising an
        # adapter's spelling is the adapter's job (ruling 2026-08-19).
        _adapter_is_proposal = getattr(self.adapter, "is_proposal_command", None)
        def _is_proposal_cmd(cmd) -> bool:
            return bool(_adapter_is_proposal(cmd)) if _adapter_is_proposal else False

        user_actions = 0      # ASK/PROPOSE count, for the shift-roll window
        proposals: list[dict] = []
        superseded_node = None      # node before the LAST answer-moving shift (P: Staleness)

        # PHASE TIMING (opt-in, runtime.phase_timing): wall-clock per phase of the loop.
        # Added because a run's wall time was ~2x the sum of its measured API latency and no
        # hypothesis (cache size, CPU, WebShop search, cluster loads) accounted for the gap.
        # Guessing was wrong three times; this measures it.
        import time as _t
        _ph = {} if self.config.get("runtime", {}).get("phase_timing") else None

        def _tick(name, t0):
            if _ph is not None:
                _ph[name] = _ph.get(name, 0.0) + (_t.perf_counter() - t0)

        for turn in range(1, self.max_turns + 1):
            # test hook: a SCRIPTED agent may be told the hidden current node, because it
            # stands in for a policy rather than being one.  A real agent has no such
            # method, so this is a no-op for anything under evaluation.
            observe = getattr(self.agent, "observe_node", None)
            if observe is not None:
                observe(current)
            # the shopper's remaining goodwill. Whether the agent is ALLOWED to see this is
            # the arm's prompt decision (LLMAgent.patience_note, gated on
            # runtime.show_patience); the loop only publishes it. Not leakage: a real
            # assistant perceives impatience, and it carries nothing about the hidden intent.
            self.agent.patience_left = patience
            _t0 = _t.perf_counter()
            raw = self.agent.act(transcript)
            _tick("agent_llm", _t0)
            action = parse_action(raw)
            transcript.agent(raw)
            record: dict[str, Any] = {
                "turn": turn, "node": current.intent_id, "action": action.to_dict(),
                "patience_before": round(patience, 2),
            }
            def _mark_due():
                """Cross thresholds at OR BELOW the current patience (ruling 2026-08-21).
                Crossed shifts become DUE and fire at the end of a user exchange, at most
                one per exchange.

                WAS `patience < threshold - EPS` ("strictly below 10"), and that single
                inequality decided the whole benchmark. Patience STARTS at the first
                threshold (10.0) and `cost_act` is 0, so nothing could become due until the
                agent spent something -- and only ASK (-2) and a refused proposal (-4)
                spend. Consequences measured over the 2026-08-21 clean run:

                  * A silent method never spent, so its FIRST proposal was judged against
                    the ORIGINAL intent: no shift had fired before the first proposal in
                    892/900 episodes, and 33% of its episodes never shifted at all.
                    B0 accepted 25.9% when judged against the original intent but only
                    6.8% against a moved one -- 231 of its ~503 wins came from that free
                    unshifted first shot.
                  * A method that asks spent immediately, so the shift fired on its first
                    question (90.6% of its first shifts landed on an ASK turn) and every
                    proposal it made was judged against a MOVED intent.

                The environment's goal trajectory was therefore a function of the agent's
                own strategy, which is a confound, not a difficulty setting. Stratifying
                the paired B0-vs-A0 comparison by whether the goal actually moved:
                neither moved -> no difference (-1.9, p=0.37, both ~88%); BOTH moved ->
                asking WINS (+2.7, p=0.017); only the asking method's goal moved -> it
                loses by 43.5. The entire published gap was the asymmetry.

                With `<=`, patience 10.0 meets threshold 10.0 at episode start, so the
                shift is due before anyone acts: every episode shifts once, every method
                faces the same trajectory, and a proposal is always judged against the
                current intent.
                """
                nonlocal shift_due
                while shift_thresholds and patience <= shift_thresholds[0] + EPS:
                    t = shift_thresholds.pop(0)
                    # FIRE COIN: the crossing consumes the threshold either way, but only
                    # a `fire` fate makes the shift due; a `skip` fate is recorded as
                    # shift_skipped and the goal stands.
                    fired = shift_fates.pop(0) if shift_fates else True
                    if fired:
                        shift_due += 1
                    else:
                        record.setdefault("shift_skipped", []).append(
                            {"threshold": round(t, 2), "turn": turn})

            def _shift_ready() -> bool:
                return shift_due > 0 if shift_mode == "persona" else shift_pending

            def _fire_scheduled(after_reply: bool, trigger: dict | None = None,
                                forced=None) -> bool:
                nonlocal current, superseded_node, shift_pending, shift_due
                if shifting_disabled:
                    return False          # ablation: no scheduled AND no triggered shifts
                if trigger is not None and not shift_thresholds and shift_due <= 0:
                    # nothing to consume: skipped placements already spent the budget,
                    # so a trigger with no placement and no due fire-unit fires nothing
                    return False
                edge = traversal.scheduled_fire(current.intent_id, mutated=mutated,
                                                proposed=bool(proposals),
                                                forced_categories=forced)
                if edge is None:
                    return False      # gate refused: the due/pending shift waits
                new_node = self.nodes[edge.dst]
                # THE WORLD MUST MOVE WITH THE INTENT, where the adapter says so.
                # Verified 2026-08-19: only tau2_telecom implements `on_shift` (a refinement
                # shift injects the new faults into the live phone). Retail and airline do
                # NOT, and correctly so -- their intents name what to book or return, and a
                # change of goal needs no change of world. The hook stays optional; a failed
                # mutation REFUSES the shift rather than leaving intent and world
                # inconsistent. `_fire_scheduled` is the single choke point for every
                # scheduled AND agent-triggered shift, so this one call covers all firing
                # sites -- which is why the persona schedule was portable at all.
                on_shift = getattr(self.adapter, "on_shift", None)
                if on_shift is not None:
                    try:
                        record["shift_world"] = on_shift(session, current, new_node)
                    except Exception as exc:
                        log.warning("on_shift failed (%s); shift refused", exc)
                        record["shift_refused"] = f"{type(exc).__name__}: {exc}"
                        return False
                if edge.gt_moved:
                    superseded_node = current
                current = new_node
                if "shift" in record:
                    # a submission flush may fire several hops in one turn; keep the full
                    # path, with the singular key always holding the LAST hop (what the
                    # scorer's after_reply invalidation reads)
                    record.setdefault("shifts_earlier", []).append(record["shift"])
                record["shift"] = edge.to_dict()
                # THE DESTINATION'S REQUIREMENTS (2026-08-19). Without this the scorer
                # cannot know what a PIVOT moved the goal TO, so it had to give up and mark
                # the episode unscoreable. The node is right here, so write its slot names
                # down. Measured cost of not doing it: 21% of WebShop shifts and 55% of
                # retail shifts were pivots, and every one of them voided Grounding and
                # Recovery for that episode.
                record["shift"]["dst_slots"] = sorted({s for s, _, _ in new_node.conditions})
                if trigger is not None:
                    record["shift"]["trigger"] = trigger
                if after_reply:
                    # the reply the agent just received reflects the OLD intent; the
                    # scorer must invalidate same-turn reveals for moved slots
                    record["shift"]["after_reply"] = True
                if shift_mode == "persona":
                    if trigger is not None:
                        # BUDGET MINUS ONE (ruling 2026-08-19): an agent-caused shift spends
                        # the LAST scheduled placement, so total count never grows.
                        # NOTE, mutation-tested 2026-08-19: this pop is REDUNDANT. The cap
                        # is actually enforced by `traversal.max_shifts`, bound to the
                        # persona's placement count ~40 lines above; deleting this pop
                        # produced byte-identical shift sequences on 6/6 graphs. Kept as
                        # defence-in-depth and because it expresses the intent, but do not
                        # mistake it for the guard -- see
                        # test_the_shift_cap_is_bound_to_the_persona_count.
                        if shift_thresholds:
                            shift_thresholds.pop()
                            if shift_fates:
                                shift_fates.pop()   # trigger discards the fate, uncoined
                        elif shift_due > 0:
                            shift_due -= 1
                    else:
                        shift_due -= 1
                else:
                    shift_pending = shift_rng.random() < traversal.p_shift
                return True

            def _fire_all_due_for_submission():
                """EVERY PROPOSAL is judged against the intent as it stands NOW (the author
                2026-08-19, revised): each crossed-but-unfired shift fires BEFORE the
                proposal is adjudicated, whether that proposal is a mid-conversation try
                or the forced final answer. There is no longer a "reaction vs submission"
                split on the proposal side -- a proposal is always measured against the
                current goal, so an answer that went stale is refused for being stale.
                Thresholds never reached do NOT fire -- solving early means fewer shifts.
                Only the user's SPOKEN reply to a question still uses the older intent:
                you cannot answer a question using a requirement you have not thought of."""
                if not shift_scheduled:
                    return
                _mark_due()
                while _shift_ready():
                    if not _fire_scheduled(after_reply=False):
                        break

            # AFFORDABILITY (ruling 2026-08-13): patience can never go negative -- an action
            # whose cost exceeds the remaining budget cannot be bought. An unaffordable ASK
            # ends the episode through the forced-final path; an unaffordable PROPOSE is
            # treated AS the final submission (the agent's last shot): adjudicated once,
            # ending the episode either way, with no charge that would cross zero.
            if action.kind is ActionKind.ASK and patience < cost_ask - EPS:
                _fire_all_due_for_submission()
                traj.turns.append(record)
                return self._forced_final(traj, current, session, proposals,
                                          Outcome.EXHAUSTED.value, transcript,
                                          superseded_node=superseded_node)
            if action.kind is ActionKind.PROPOSE and patience < cost_reject - EPS:
                # An unaffordable PROPOSE *is* the final answer. Same rule as every other
                # proposal: flush the due shifts, then judge against the current goal.
                _fire_all_due_for_submission()
                # the flush may have moved the goal; the turn was adjudicated on
                # the node as it now stands, so that is what the log must say
                record["node"] = current.intent_id
                proposal = self._parse_proposal(action.proposal_raw)
                ok, why = accept_mod.accepts(self.adapter, proposal, current, session,
                                             executor=self.executor)
                record["acceptance"] = {"proposal": str(action.proposal_raw)[:400],
                                        "ok": bool(ok), "reason": why,
                                        "reward": _acceptance_reward(bool(ok), why),
                                        "final_unaffordable_reject": True}
                proposals.append({"turn": turn, "node": current.intent_id,
                                  "raw": action.proposal_raw, "ok": bool(ok)})
                if not ok:
                    self._record_stale_check(record, proposal, superseded_node, session)
                record["patience_after"] = round(patience, 2)
                traj.turns.append(record)
                traj.outcome = (Outcome.SUCCESS.value if ok else Outcome.EXHAUSTED.value)
                traj.final_node = current.intent_id
                traj.header["proposals"] = proposals
                return traj
            # Snapshot the arm's own belief state, for diagnostics only -- it changes no
            # behaviour. Necessary because _absorb strips the state block out of the action
            # before the turn is recorded (so the raw JSON never reaches the user), which
            # also removed the only trace of WHY a method asked. Without this an arm whose
            # ask-vs-act rule is numeric cannot be debugged: you can see that it asked, but
            # not the number it claimed justified asking.
            state = getattr(self.agent, "state", None)
            if isinstance(state, dict) and state:
                try:
                    record["agent_state"] = json.loads(json.dumps(state))
                except (TypeError, ValueError):
                    record["agent_state"] = {"unserialisable": True}

            # --- axis B: the intent may move before the user replies ---------
            # Rolls happen on user-facing actions, capped at `shift_roll_window` of them. That
            # removes the largest part of the exposure confound (12pp down to ~4pp) but not all
            # of it: an arm whose purchase gate converts would-be buys into SEARCHES takes fewer
            # rolls, measured on B2 at 73.2% against B0's 77.9%. Rolling on EVERY turn instead
            # was tried and reverted -- it would have stranded three completed 1,500-episode
            # runs mid-campaign for a residual that analysis can remove instead. The residual is
            # controlled in `tools/criteria_check.py`, which re-weights each arm's per-stratum
            # success to the anchor's exposure and adjudicates on THAT rather than on raw
            # Success. Changing this back would require re-running every arm.
            if action.kind in (ActionKind.ASK, ActionKind.PROPOSE):
                user_actions += 1
                if shift_scheduled:
                    pass   # both channels fire at the END of the exchange: a reaction is
                           # judged under the intent the user held when it was made
                           # (ruling 2026-08-19); see the reply/rejection sites below.
                else:
                    roll_on_acts = bool(rt.get("shift_roll_on_acts", False))
                    edge = traversal.maybe_shift(current.intent_id, mutated=mutated,
                                                 proposed=bool(proposals),
                                                 action_index=(turn if roll_on_acts
                                                               else user_actions))
                    if edge is not None:
                        # SILENT by design (ruling 2026-08-11): no announcement is emitted.
                        new_node = self.nodes[edge.dst]
                        on_shift = getattr(self.adapter, "on_shift", None)
                        if on_shift is not None:
                            try:
                                record["shift_world"] = on_shift(session, current, new_node)
                            except Exception as exc:
                                log.warning("on_shift failed (%s); shift refused", exc)
                                record["shift_refused"] = f"{type(exc).__name__}: {exc}"
                                edge = None
                        if edge is not None:
                            if edge.gt_moved:
                                superseded_node = current
                            current = new_node
                            record["shift"] = edge.to_dict()
                            record["shift"]["dst_slots"] = sorted(
                                {s for s, _, _ in new_node.conditions})

            # --- the user replies --------------------------------------------
            if action.kind is ActionKind.ASK:
                if bool(rt.get("mute_user", False)):
                    # L0 floor (ruling 2026-08-11): the agent works the environment alone;
                    # the simulated user exists but NEVER speaks. An ask still costs a
                    # turn and patience -- it is simply never answered.
                    reply, reveals = "(the user does not respond)", []
                else:
                    _t2 = _t.perf_counter()
                    reply, reveals = user.answer_ask(action.question or "", current)
                    _tick("user_llm", _t2)
                transcript.user(reply)
                record["reveals"] = [vars(r) for r in reveals]
                record["reply"] = reply
                patience -= cost_ask
                _mark_due()
                if shift_scheduled and shift_mode == "persona" and not bool(rt.get("mute_user", False)):
                    # AGENT-CAUSED SHIFT (ruling 2026-08-19). Checked FIRST: the user's
                    # revision is a response to what the agent just said; a due scheduled
                    # shift then waits for the next exchange (one shift per exchange).
                    cats, hit_slots = traversal.trigger_categories(
                        current.intent_id, traversal_mod._words(action.question))
                    if cats and trigger_rng.random() < self.persona.suggestibility:
                        fired = _fire_scheduled(
                            after_reply=True, forced=cats,
                            trigger={"kind": "agent_ask", "slots": list(hit_slots),
                                     "categories": [c.value for c in cats]})
                        if not fired and _shift_ready():
                            _fire_scheduled(after_reply=True)
                    elif _shift_ready():
                        _fire_scheduled(after_reply=True)
                elif shift_scheduled and _shift_ready():
                    _fire_scheduled(after_reply=True)
                # test hook, mirroring observe_node: tell a scripted agent which opaque ids
                # were actually granted, so a persistent oracle knows what to re-ask
                inform = getattr(self.agent, "observe_reveal", None)
                if inform is not None:
                    ordered = [s for s, _, _ in sorted(current.conditions)]
                    granted = [f"slot_{ordered.index(r.slot)}" for r in reveals
                               if not r.declined and r.slot in ordered]
                    inform(granted)

            elif action.kind is ActionKind.PROPOSE:
                # Judged against the CURRENT goal: flush every due shift first.
                _fire_all_due_for_submission()
                # the flush may have moved the goal; the turn was adjudicated on
                # the node as it now stands, so that is what the log must say
                record["node"] = current.intent_id
                proposal = self._parse_proposal(action.proposal_raw)
                ok, why = accept_mod.accepts(self.adapter, proposal, current, session,
                                             executor=self.executor)
                record["acceptance"] = {"proposal": str(action.proposal_raw)[:400],
                                        "ok": bool(ok), "reason": why,
                                        "reward": _acceptance_reward(bool(ok), why)}
                proposals.append({"turn": turn, "node": current.intent_id,
                                  "raw": action.proposal_raw, "ok": bool(ok)})
                if ok:
                    transcript.user("" if bool(rt.get("mute_user", False))
                                    else user.phrase_accept())
                    record["patience_after"] = round(patience, 2)
                    traj.turns.append(record)
                    traj.outcome = Outcome.SUCCESS.value
                    traj.final_node = current.intent_id
                    traj.header["proposals"] = proposals
                    return traj
                if bool(rt.get("mute_user", False)):
                    transcript.user("(the user does not respond)")
                else:
                    _t3 = _t.perf_counter()
                    said = self._user_reacts_to_proposal(user, rt, record, current,
                                                         proposal, accepted=False)
                    _tick("user_llm", _t3)
                    transcript.user(said if said is not None else user.phrase_rejection(why))
                # The proposal was judged against the goal as it stands now (the flush
                # happened above). Pay the price and record any newly-crossed threshold;
                # that shift fires before the NEXT proposal, or at the next user reply.
                patience -= cost_reject
                _mark_due()

            elif action.kind is ActionKind.ACT and _is_proposal_cmd(action.command) and \
                    patience < cost_reject - EPS:
                user_actions += 1
                _fire_all_due_for_submission()
                # the flush may have moved the goal; the turn was adjudicated on
                # the node as it now stands, so that is what the log must say
                record["node"] = current.intent_id
                # unaffordable buy-shaped Operation: same rule as an unaffordable PROPOSE --
                # it becomes the final submission, adjudicated once, never charged below zero
                proposal = self._parse_proposal(action.command)
                ok, why = accept_mod.accepts(self.adapter, proposal, current, session,
                                             executor=self.executor)
                record["action"]["kind"] = "PROPOSE"
                record["action"]["proposal_raw"] = action.command
                record["action"]["via_operation"] = True
                record["acceptance"] = {"proposal": str(action.command)[:400],
                                        "ok": bool(ok), "reason": why,
                                        "reward": _acceptance_reward(bool(ok), why),
                                        "final_unaffordable_reject": True}
                proposals.append({"turn": turn, "node": current.intent_id,
                                  "raw": action.command, "ok": bool(ok)})
                if not ok:
                    self._record_stale_check(record, proposal, superseded_node, session)
                record["patience_after"] = round(patience, 2)
                traj.turns.append(record)
                traj.outcome = (Outcome.SUCCESS.value if ok else Outcome.EXHAUSTED.value)
                traj.final_node = current.intent_id
                traj.header["proposals"] = proposals
                return traj

            elif action.kind is ActionKind.ACT and _is_proposal_cmd(action.command):
                user_actions += 1   # a buy-shaped Operation is scored exactly like a
                # PROPOSE, so it gets the same rule: flush due shifts, then judge against
                # the goal as it now stands (ruling 2026-08-19, revised).
                _fire_all_due_for_submission()
                # the flush may have moved the goal; the turn was adjudicated on
                # the node as it now stands, so that is what the log must say
                record["node"] = current.intent_id
                # The buy button. Real WebShop makes buying an ENVIRONMENT action, so models
                # naturally emit `buy ...` inside Action: Operation -- and this harness was
                # routing that to the SEARCH BOX: the agent "bought" four times, got search
                # listings back, and burned to TURNS_EXCEEDED with its purchase never
                # adjudicated. An interface must accept a purchase wherever a real store
                # would, so a buy-shaped Operation is scored exactly like a proposal.
                proposal = self._parse_proposal(action.command)
                ok, why = accept_mod.accepts(self.adapter, proposal, current, session,
                                             executor=self.executor)
                record["action"]["kind"] = "PROPOSE"
                record["action"]["proposal_raw"] = action.command
                record["action"]["via_operation"] = True
                record["acceptance"] = {"proposal": str(action.command)[:400],
                                        "ok": bool(ok), "reason": why,
                                        "reward": _acceptance_reward(bool(ok), why)}
                proposals.append({"turn": turn, "node": current.intent_id,
                                  "raw": action.command, "ok": bool(ok)})
                if ok:
                    transcript.user("" if bool(rt.get("mute_user", False))
                                    else user.phrase_accept())
                    record["patience_after"] = round(patience, 2)
                    traj.turns.append(record)
                    traj.outcome = Outcome.SUCCESS.value
                    traj.final_node = current.intent_id
                    traj.header["proposals"] = proposals
                    return traj
                if bool(rt.get("mute_user", False)):
                    transcript.user("(the user does not respond)")
                else:
                    _t3 = _t.perf_counter()
                    said = self._user_reacts_to_proposal(user, rt, record, current,
                                                         proposal, accepted=False)
                    _tick("user_llm", _t3)
                    transcript.user(said if said is not None else user.phrase_rejection(why))
                # judged against the CURRENT goal (flushed above); pay and re-check
                patience -= cost_reject
                _mark_due()

            elif action.kind is ActionKind.ACT:
                is_mut = self._act_is_mutating(action.command or "")
                record["act_is_mutating"] = is_mut
                try:
                    req = ({"sql": action.command, "kind": "SELECT",
                            "table": (current.base or {}).get("table")}
                           if self.graph.adapter == "dbbench"
                           else {"command": action.command, "tool": "raw"})
                    _t1 = _t.perf_counter()
                    out = session.run(req, mutating=False)
                    _tick("env_step", _t1)
                    # 2400 chars cut the search listing mid-list; 12 results at ~350
                    # chars each need ~4200 (ruling 2026-08-13: many items per search is
                    # normal WebShop -- the agent must get a proper list to look at)
                    observation = str(out)[:4400]
                except EnvTransportError:
                    # The env is DEAD past the retry budget. Fabricating an observation
                    # here was the last leak of fake pages into scored data (the author
                    # 2026-08-21: gate v4 had 192 contaminated episodes made exactly
                    # here). Propagate: the outer wrapper records an ERROR husk, the
                    # guard counts it, the idempotent refill replaces it.
                    raise
                except Exception as exc:
                    observation = f"error: {type(exc).__name__}: {exc}"[:400]
                transcript.env(observation)
                record["observation"] = observation
                mutated = mutated or is_mut
                patience -= cost_act
                _mark_due()

            else:  # MALFORMED
                transcript.user("I didn't understand that. Could you rephrase?")
                patience -= cost_malformed
                _mark_due()

            record["patience_after"] = round(patience, 2)
            traj.turns.append(record)
            if _ph is not None:
                traj.header["phase_timing"] = {k: round(v, 2) for k, v in _ph.items()}

            if patience <= EPS:
                _fire_all_due_for_submission()
                return self._forced_final(traj, current, session, proposals,
                                          Outcome.EXHAUSTED.value, transcript,
                                          superseded_node=superseded_node)

        record = traj.turns[-1] if traj.turns else {}
        _fire_all_due_for_submission()
        return self._forced_final(traj, current, session, proposals,
                                  Outcome.TURNS_EXCEEDED.value, transcript,
                                  superseded_node=superseded_node)

    def _forced_final(self, traj: Trajectory, current, session, proposals,
                      exhaust_outcome: str, transcript, superseded_node=None) -> Trajectory:
        """Budget spent: the agent must submit a final choice, and is scored on it.

        A silent fail is not an evaluation (ruling 2026-08-11): every episode ends with an
        adjudicated answer. One action is demanded; a reply that is not a buy keeps the
        exhaustion outcome, recorded as such.
        """
        transcript.user("I'm out of time. Give me your final choice right now -- "
                        "submit the single best option you have.")
        # The wire format has to be restated HERE (2026-08-13). Measured on the dev probe:
        # 24% of B0's episodes and 3% of A1's ended with an apology or a follow-up question
        # in the Final Answer slot ("I cannot find a product matching all your
        # requirements..."), which scores exactly zero however good the agent's reasoning
        # was. That is a defect in how the harness ELICITS the last answer, identical for
        # every arm, not a property of any arm -- so the demand is made explicit, and an
        # unparseable answer gets exactly one repair attempt. An adapter may supply its own
        # required form (final_answer_format); the WebShop wording is the default.
        transcript.env(getattr(self.adapter, "final_answer_format", None) or (
                       "Final answer required now, in exactly this form:\n"
                       "Action: Answer\nPredicted user question: <what they want>\n"
                       'Final Answer: buy <ASIN> {"option name": "value"}\n'
                       "The ASIN must be one you saw in a search result. Name your best "
                       "candidate even if it is imperfect: a near match scores partial "
                       "credit, while an apology, an explanation or a question scores zero."))
        record: dict[str, Any] = {"turn": len(traj.turns) + 1,
                                  "node": current.intent_id, "forced_final": True}
        try:
            raw = self.agent.act(transcript)
            action = parse_action(raw)
            record["action"] = action.to_dict()
            # F6 (audit 2026-08-20): the forced-final turn told the agent "if any
            # requested write has not been made yet, make it FIRST", but a tool call
            # issued here was parsed and then DISCARDED -- only proposal/buy-shaped text
            # was honoured, and nothing ever executed it. 10 of 40 forced-final turns
            # issued a real write. Execute it, then adjudicate, so the instruction the
            # agent is given is one it can actually act on.
            # Gated on the adapter supplying its own final-answer form: the make-the-write-
            # FIRST instruction is part of that adapter text (see final_answer_format on
            # the tau2 adapters), so only an adapter that makes the promise gets the
            # execution; everywhere else a stray command keeps being discarded, as the
            # original WebShop loop did.
            if (getattr(self.adapter, "final_answer_format", None)
                    and action.command and not action.proposal_raw
                    and not re.match(r"\s*(buy|purchase)\b", action.command, re.IGNORECASE)):
                try:
                    obs = session.run(action.command)
                    record["observation"] = obs
                    record["forced_final_executed"] = True
                except Exception as exc:
                    record["observation"] = f"error: {type(exc).__name__}: {exc}"
            raw_prop = action.proposal_raw or (
                action.command if action.command and
                re.match(r"\s*(buy|purchase)\b", action.command, re.IGNORECASE) else None)
            raw_prop, repaired = self._repair_final(raw_prop, transcript, record)
            if raw_prop:
                if not action.proposal_raw or repaired:   # buy-shaped: score as PROPOSE
                    record["action"]["kind"] = "PROPOSE"
                    record["action"]["proposal_raw"] = raw_prop
                    record["action"]["via_operation"] = True
                proposal = self._parse_proposal(raw_prop)
                ok, why = accept_mod.accepts(self.adapter, proposal, current, session,
                                             executor=self.executor)
                record["acceptance"] = {"proposal": str(raw_prop)[:400],
                                        "ok": bool(ok), "reason": why,
                                        "reward": _acceptance_reward(bool(ok), why),
                                        **({"format_repaired": True} if repaired else {})}
                proposals.append({"turn": record["turn"], "node": current.intent_id,
                                  "raw": raw_prop, "ok": bool(ok)})
                traj.outcome = Outcome.SUCCESS.value if ok else exhaust_outcome
                if not ok and superseded_node is not None:
                    # P / Staleness: is the failed final answer exactly the answer the
                    # SUPERSEDED intent would have accepted? Decided by execution, at
                    # episode time, so the scorer stays pure log arithmetic.
                    try:
                        ok_old, _why = accept_mod.accepts(self.adapter, proposal,
                                                          superseded_node, session,
                                                          executor=self.executor)
                        record["stale_check"] = {"old_node": superseded_node.intent_id,
                                                 "valid_under_old": bool(ok_old)}
                    except Exception as exc:
                        record["stale_check"] = {"old_node": superseded_node.intent_id,
                                                 "error": type(exc).__name__}
            else:
                record["note"] = "forced_final_no_proposal"
                traj.outcome = exhaust_outcome
        except Exception as exc:   # a dying agent must not turn the episode into ERROR here
            record["note"] = f"forced_final_failed:{type(exc).__name__}"
            traj.outcome = exhaust_outcome
        traj.turns.append(record)
        traj.final_node = current.intent_id
        traj.header["proposals"] = proposals
        traj.header["forced_final"] = True
        return traj

    def _repair_final(self, raw_prop, transcript, record):
        """One repair attempt when the forced final answer is not a purchase at all.

        The agent has already been shown the required form; if what came back still cannot
        be read as a buy (an apology, an explanation, a counter-question), it is asked once
        more, plainly. Only the FORM is corrected -- the harness never chooses a product,
        so an agent that will not commit still scores zero. Returns (proposal, repaired).
        """
        def _buyable(text) -> bool:
            # `_parse_proposal` hands back the RAW STRING when the adapter could not read a
            # purchase out of it (that string is what later scores "unparseable_proposal"),
            # and a parsed structure otherwise.
            return bool(text) and not isinstance(self._parse_proposal(text), str)

        if _buyable(raw_prop):
            return raw_prop, False
        # F6b (audit 2026-08-20): this repair text was a WebShop literal shown verbatim
        # to agents in other domains -- 19 of 208 airline A0 episodes were told, on their
        # last turn, to answer with 'buy <ASIN>'. Grading ignores the text, but the
        # instruction is wrong for the domain. Use the adapter's own required form.
        transcript.env(getattr(self.adapter, "final_answer_format", None) or (
                       "That was not a purchase. Reply with ONLY this line, nothing else:\n"
                       'Final Answer: buy <ASIN> {"option name": "value"}\n'
                       "Use the best ASIN you have seen. Anything else scores zero."))
        try:
            raw2 = self.agent.act(transcript)
        except Exception:
            return raw_prop, False
        action2 = parse_action(raw2)
        cand = action2.proposal_raw or action2.command
        if not cand:
            m = re.search(r"(buy\s+\S+.*)", raw2 or "", re.IGNORECASE | re.DOTALL)
            cand = m.group(1).strip() if m else None
        if _buyable(cand):
            record["final_format_repair"] = True
            return cand, True
        return raw_prop, False

    def _record_stale_check(self, record, proposal, superseded_node, session) -> None:
        """Was this final answer still valid under the intent as it stood BEFORE the last
        answer-moving shift? Recorded on EVERY final adjudication -- the forced-final path
        and the unaffordable-proposal 'final shot' alike. Previously only the forced-final
        path wrote it, so whether Staleness was measurable depended on which of two
        exhaustion routes happened to fire, which is arbitrary.
        """
        if superseded_node is None:
            return
        try:
            ok_old, _why = accept_mod.accepts(self.adapter, proposal, superseded_node,
                                              session, executor=self.executor)
            record["stale_check"] = {"old_node": superseded_node.intent_id,
                                     "valid_under_old": bool(ok_old)}
        except Exception as exc:
            record["stale_check"] = {"old_node": superseded_node.intent_id,
                                     "error": type(exc).__name__}

    # ---------------------------------------------------------------- helpers
    def _perturb_root(self, node, session, pristine):
        """Try the applicable strategies until one produces an admissible perturbation.

        A strategy can fail legitimately -- marking a redundant slot hides nothing, so the
        signature check rejects it -- and that must not kill the episode.  The order is
        seeded, so which strategy wins is still reproducible.
        """
        rt = self.rt
        min_retained = int(rt.get("min_retained_conditions", 1))
        pool = list(st.applicable_strategies(node.conditions, min_retained=min_retained))
        if not pool:
            return None, "no applicable perturbation strategy for the root intent"

        # First choice follows the configured/family-balanced sampling; the rest are
        # fallbacks for when a mask cannot pass the signature check.
        k = max(1, int(rt.get("misalignments_k", 1)))
        combo = st.sample_strategies(node.conditions, self.rng, k=k,
                                     min_retained=min_retained,
                                     strategy_probs=rt.get("strategy_probs") or None)
        if not combo:
            return None, "no applicable perturbation strategy for the root intent"
        first = combo[0]
        # Attempt order: the requested k-way composite first, then progressively simpler
        # composites, then single strategies. A composite that cannot be made coherent must
        # degrade to a valid single misalignment rather than kill the episode -- but it is
        # recorded, so a run where k=2 silently became k=1 everywhere is visible.
        attempts: list[list] = []
        for size in range(len(combo), 0, -1):
            attempts.append(combo[:size])
        rest = [s for s in pool if s.id != first.id]
        self.rng.shuffle(rest)          # NB: shuffling `ordered[1:]` would shuffle a COPY
        # A fallback should preserve the *intent* of the sample: if we wanted a strategy that
        # hides something, try another hiding strategy before a content-free one.  Otherwise
        # content-free strategies -- which always pass the signature, since they leave the
        # answer untouched -- become a silent attractor and most episodes hide nothing.
        wanted_hiding = first.mask_kind in (st.WITHHOLD, st.FALSIFY, st.MARK)
        rest.sort(key=lambda s: (s.mask_kind in (st.NOISE, st.OBLIQUE)) == wanted_hiding)
        attempts.extend([s] for s in rest)

        domains = self._domains(node, session)
        verify = signature_mod.make_verifier(self.adapter, session, node,
                                             pristine_hash=pristine)
        reasons = []
        for group in attempts:
            label = "+".join(s.id for s in group)
            try:
                pert = perturb_mod.perturb(
                    node=node, conditions=node.conditions, strategies=group, rng=self.rng,
                    llm=self.llm, surface=perturb_mod.surface_conventions(self.graph.adapter),
                    config=self.config, domains=domains, verify=verify,
                    persona_tone=self.persona.bio.split("\n")[0],
                    describe_slot=getattr(self.adapter, "slot_phrase", None),
                    render_context=self._render_context(node),
                    foreign_domains=self._foreign_domains(node, session),
                )
                return pert, group[0] if len(group) == 1 else group
            except perturb_mod.Unperturbable as exc:
                reasons.append(f"{label}: {exc}")
        return None, "unperturbable; tried " + " | ".join(reasons)

    def _foreign_domains(self, node, session) -> dict:
        """Values from other environments, for false premises. Optional per adapter."""
        fn = getattr(self.adapter, "foreign_domains", None)
        if fn is None:
            return {}
        out = {}
        for slot, _, _ in node.conditions:
            try:
                out[slot] = list(fn(slot, node.base, session))
            except Exception:
                out[slot] = []
        return out

    def _render_context(self, node) -> str:
        """Safe topical context for the renderer -- never the answer. See the adapter."""
        fn = getattr(self.adapter, "render_context", None)
        if fn is None:
            return ""
        try:
            return fn(node.base) or ""
        except Exception:  # pragma: no cover - adapter-specific
            return ""

    @staticmethod
    def _record_attempts(traj, reasons: list[str]) -> None:
        if reasons:
            traj.header["strategy_fallbacks"] = reasons

    def _domains(self, node, session) -> dict[str, list]:
        out: dict[str, list] = {}
        for slot, _, _ in node.conditions:
            try:
                out[slot] = list(self.adapter.domains(slot, node.base, session))
            except Exception:
                out[slot] = []
        return out


    def _user_reacts_to_proposal(self, user, rt, record, current, proposal, accepted):
        """User v2: the shopper sees the item as a shopper would and reacts from its own
        judgement of fit; the OFFICIAL outcome stayed executable (ruling 2026-08-13). The
        reaction's reveals are recorded on this turn so rejections finally teach."""
        if not hasattr(user, "react_to_proposal"):
            return None
        summary = str(proposal.get("raw") or proposal)[:200] if isinstance(proposal, dict) \
            else str(proposal)[:200]
        asin = (proposal or {}).get("asin") if isinstance(proposal, dict) else None
        if asin:
            item = self.executor.get_product(asin, self.graph.env_spec.get("cluster"))
            if item:
                attrs = ", ".join(a for a in (item.get("Attributes") or [])
                                  if a != "DUMMY_ATTR")[:200]
                opts = (proposal or {}).get("options") or {}
                summary = (f"{item.get('name','')[:120]} -- ${item.get('price','?')} -- "
                           f"features: {attrs} -- chosen options: {opts}")
        unmet = None
        if not accepted:
            fn = getattr(self.adapter, "unmet_requests", None)
            if fn is not None:
                try:
                    unmet = fn(self._react_session, current)
                except Exception:
                    unmet = None
        reply, reveals = user.react_to_proposal(summary, current, accepted=accepted,
                                                unmet=unmet)
        record.setdefault("reveals", [])
        record["reveals"] += [vars(r) for r in reveals]
        record["reply"] = reply
        return reply

    def _parse_proposal(self, raw):
        fn = getattr(self.adapter, "parse_proposal", None)
        if fn is None:
            return accept_mod.default_parse_proposal(raw)
        parsed = fn(raw)
        return raw if parsed is None else parsed

    def _act_is_mutating(self, raw: str) -> bool:
        fn = getattr(self.adapter, "act_is_mutating", None)
        return True if fn is None else bool(fn(raw))


def run_episode(*, graph, adapter, executor, config, llm, agent,
                persona_name=None, seed=None) -> Trajectory:
    return Episode(graph=graph, adapter=adapter, executor=executor, config=config, llm=llm,
                   agent=agent, persona_name=persona_name, seed=seed).run()


def dumps(traj: Trajectory) -> str:
    return canonical_dumps(traj.to_dict(), indent=1)
