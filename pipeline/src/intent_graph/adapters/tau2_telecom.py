"""tau2-bench telecom adapter.

A telecom task is a customer whose phone has one or more FAULTS and who calls support to
get them fixed.  tau2 composes those tasks programmatically: each fault is a primitive
(``BaseTask``) carrying its own fault-injection functions, its own repair tool calls, and
its own environment assertions; a task is a valid multiset of faults, and its golden
action sequence is the concatenation of the per-fault repairs.

That structure is why this benchmark suits intent graphs: **a graph edge is a mechanical
edit of the fault multiset**, and the branch's ground truth is derived by re-composing the
task through tau2's own engine rather than by anyone's judgement.

Design decisions recorded here because they are load-bearing (ruling 2026-08-12):

* **Conditions are faults, not tool arguments.** A telecom intent is problem-shaped, not
  parameter-shaped: 10,927 of 13,215 golden actions are parameterless device toggles, so
  an argument-slot model would leave most intents with no conditions at all, while the
  fault multiset gives >=2 conditions on 96% of the base split.
* **Ground truth is the required repair, not the terminal state.**  Telecom fixes restore
  the device to one healthy attractor state, so 41% of edited branches end in a state
  identical to their parent's (measured in the spike).  Terminal-state hashing would
  therefore call distinct intents identical.  GT is the (sorted) repair signature plus the
  assertion specification -- what the agent must *bring about*, which differs whenever the
  fault set differs.
* **One environment.**  Every telecom task runs on the same shipped customer world; the
  faults are injected by the intent's own initialization.  So all telecom intents share an
  environment and a graph may pivot across intent *types* (mobile data / service / MMS)
  without leaving it.
* **Solo-style control.**  The environment is opened with the agent holding the union of
  agent and user tools; our simulated user speaks but never acts.
"""

from __future__ import annotations

import itertools
import json
import os
import re
import sys
from typing import Any

from ..ids import canonical_dumps
from ..models import Condition, GroundTruth, Seed

FAULT = "fault:"
_ID_RE = re.compile(r"^\[([a-z_]+)\](.*)\[PERSONA:([A-Za-z]+)\]$")

_DISTILLED_OPS = """OPERATIONS GUIDE (condensed from tau2's policy + workflow; all rules binding).

IDENTIFY FIRST: look up the customer (phone number, customer id, or full name + DOB).

REPAIR LOOP DISCIPLINE: one tool call per turn. After fixes, REBOOT the device when a fix
requires it (resuming a line ALWAYS requires reboot to restore service). Verify each fix
with the matching check_* tool before moving on. Try all relevant steps before
transfer_to_human_agents (only for out-of-scope).

NO SERVICE path (status bar shows no signal/airplane):
1 airplane mode ON -> turn OFF, recheck. 2 SIM: MISSING -> reseat, verify ACTIVE;
LOCKED (PIN/PUK) -> escalate. 3 reset APN to default + reboot, recheck.
4 line SUSPENDED -> overdue bill: collect payment (below) then lift suspension + reboot;
contract ended: NEVER lift, escalate. Else escalate.

MOBILE DATA path (speed test = no connection -> unavailable; below Excellent -> slow):
UNAVAILABLE: 1 ensure service (run NO SERVICE path). 2 user traveling -> data roaming ON
(device) AND line roaming enabled (enable at no cost). 3 mobile data toggle ON.
4 data usage EXCEEDED -> refuel data (<= 2GB max, confirm price from plan; needs user
permission) or change plan (gather plans, match requirement, apply); else escalate.
SLOW (any speed below Excellent): 1 data saver OFF. 2 network mode: prefer 5G/4G (older
2G/3G limits speed). 3 VPN active -> disconnect. Recheck speed after each; else escalate.

MMS path (can_send_mms false):
1 ensure service (NO SERVICE path). 2 ensure mobile data CONNECTIVITY (ignore speed).
3 network tech must be >= 3G (change mode if 2G). 4 wi-fi calling ON -> turn OFF.
5 messaging app needs BOTH storage and SMS permissions -> grant missing. 6 APN MMSC URL
missing -> reset APN to defaults. Recheck can_send_mms after each; else escalate.

PAYMENTS (only with user permission): verify bill is OVERDUE -> send_payment_request
(status becomes AWAITING PAYMENT; only ONE bill may be awaiting at a time) ->
check_payment_request -> make_payment -> verify bill PAID.

SUSPENSION RULES: lift only after ALL overdue bills paid; NEVER lift a contract-ended
line. After resume: reboot. Roaming for travelers is free to enable. Refuel cap: 2GB.
"""


def _ensure_tau2() -> None:
    src = os.environ.get("TAU2_SRC")
    if src and src not in sys.path:
        sys.path.insert(0, src)


def _managers() -> dict:
    _ensure_tau2()
    from tau2.domains.telecom.tasks.mms_issues import mms_issue_task_manager
    from tau2.domains.telecom.tasks.mobile_data_issues import mobile_data_task_manager
    from tau2.domains.telecom.tasks.service_issues import service_issues_task_manager

    return {"mobile_data_issue": mobile_data_task_manager,
            "service_issue": service_issues_task_manager,
            "mms_issue": mms_issue_task_manager}


class Tau2TelecomAdapter:
    name = "tau2_telecom"
    version = "1"
    executor_name = "tau2_inproc"

    ENV_KEY = "tau2_telecom_c1001"

    def __init__(self, config: dict | None = None) -> None:
        cfg = (config or {}).get("tau2_telecom", {}) if config else {}
        self.tasks_path = cfg.get(
            "tasks_path",
            os.environ.get("TAU2_DATA", "") + "/domains/telecom/tasks.json")
        self.split_path = cfg.get(
            "split_path",
            os.environ.get("TAU2_DATA", "") + "/domains/telecom/split_tasks.json")
        self.split = cfg.get("split", "base")
        self._mgrs = None
        self._universe = None      # fault name -> (manager, selection_set_index, BaseTask)

    # ------------------------------------------------------------------ engine
    @property
    def managers(self) -> dict:
        if self._mgrs is None:
            self._mgrs = _managers()
        return self._mgrs

    @property
    def universe(self) -> dict:
        """manager -> {fault name -> (selection_set_index, BaseTask)}.

        Scoped per manager on purpose: the same fault primitive appears in several
        intent types with different selection-set positions, so a flat name->entry map
        silently loses membership (and with it every refinement candidate).
        """
        if self._universe is None:
            uni: dict[str, dict] = {}
            for mname, mgr in self.managers.items():
                per = {}
                for i, ss in enumerate(mgr.selection_sets):
                    for bt in ss.tasks:
                        per[bt.name] = (i, bt)
                uni[mname] = per
            self._universe = uni
        return self._universe

    def _bt(self, name: str, manager: str | None = None):
        if manager and name in self.universe.get(manager, {}):
            return self.universe[manager][name][1]
        for per in self.universe.values():
            if name in per:
                return per[name][1]
        return None

    def all_fault_names(self) -> set:
        return {n for per in self.universe.values() for n in per}

    def fault_description(self, name: str) -> str:
        bt = self._bt(name)
        return bt.description if bt is not None else name

    def is_fixable(self, name: str) -> bool:
        bt = self._bt(name)
        return bool(bt is not None and bt.fix_funcs)

    # ------------------------------------------------------------------ load
    def load(self) -> list[Seed]:
        with open(self.tasks_path, encoding="utf-8") as f:
            tasks = json.load(f)
        keep_ids = None
        if self.split and os.path.exists(self.split_path):
            with open(self.split_path, encoding="utf-8") as f:
                splits = json.load(f)
            ids = splits.get(self.split)
            if isinstance(ids, dict):
                ids = ids.get("task_ids")
            if ids:
                keep_ids = set(map(str, ids))

        seeds: list[Seed] = []
        for t in tasks:
            tid = str(t.get("id"))
            if keep_ids is not None and tid not in keep_ids:
                continue
            m = _ID_RE.match(tid)
            if not m:
                continue
            intent_type, faults = m.group(1), [x for x in m.group(2).split("|") if x]
            known = self.all_fault_names()
            if not faults or any(f not in known for f in faults):
                continue
            conds = tuple(sorted((f"{FAULT}{f}", "=", f) for f in faults))
            seeds.append(Seed(
                record_id=tid,
                env_key=self.ENV_KEY,
                base={"kind": "telecom_repair", "manager": intent_type},
                conditions=conds,
                shipped_answer={
                    "actions": [a for a in (t.get("evaluation_criteria") or {}).get("actions") or []],
                    "env_assertions": (t.get("evaluation_criteria") or {}).get("env_assertions") or [],
                },
                meta={"n_faults": len(faults),
                      "unfixable": any(not self.is_fixable(f) for f in faults),
                      # tau2 ships a persona string inside the task id; personas live only in
                      # our evaluation-time simulator, so it is recorded and never used.
                      "tau2_persona_silenced": m.group(3)},
            ))
        return seeds

    # ------------------------------------------------------------------ recipes
    @staticmethod
    def _faults_of(conditions: tuple[Condition, ...]) -> list[str]:
        return sorted(str(v) for s, _, v in conditions if str(s).startswith(FAULT))

    def _composed(self, manager: str, faults: list[str]):
        _ensure_tau2()
        from tau2.domains.telecom.tasks.utils import ComposedTask

        bts = []
        for f in faults:
            bt = self._bt(f, manager)
            if bt is None:
                # A perturbation may name something that is not a fault in this intent
                # type; that is a rejection, never a crash.
                raise ValueError(f"unknown fault {f!r} for manager {manager!r}")
            bts.append(bt)
        bts = sorted(bts, key=lambda x: x.name)
        return ComposedTask(
            name="|".join(b.name for b in bts),
            description=", ".join(b.description for b in bts),
            composed_from=bts,
            init_funcs=[f for b in bts for f in b.init_funcs],
            fix_funcs=[f for b in bts for f in b.fix_funcs],
            extra_env_assertions=[f for b in bts for f in b.extra_env_assertions],
        )

    def _task_for(self, base: dict, conditions: tuple[Condition, ...]):
        """Build the tau2 Task through the engine's own create_task (never by hand)."""
        manager = base["manager"]
        faults = self._faults_of(conditions)
        composed = self._composed(manager, faults)
        return self.managers[manager].create_task(composed, "None")

    def compile(self, base: dict, conditions: tuple[Condition, ...]) -> Any:
        task = self._task_for(base, conditions)
        init = task.initial_state
        crit = task.evaluation_criteria
        return {
            "manager": base["manager"],
            "faults": self._faults_of(conditions),
            "initialization": {
                "initialization_data": (init.initialization_data if init else None),
                "initialization_actions": [a.model_dump() for a in (init.initialization_actions or [])]
                if init else [],
            },
            "actions": [a.model_dump() for a in (crit.actions or [])],
            "env_assertions": [a.model_dump() for a in (crit.env_assertions or [])],
        }

    # ------------------------------------------------------------------ execute
    def execute(self, recipe: Any, session) -> GroundTruth:
        """Ground truth = the repair the intent requires, executed to prove it is reachable.

        We run the recipe in a freshly materialized world (the caller passes mutating=True)
        and require the task's own assertions to hold afterwards; the stored value is the
        repair signature, which is what distinguishes one intent from another.
        """
        env = self._materialized(session, recipe)
        actions = recipe.get("actions") or []
        from ..executors.tau2_inproc import Tau2Session
        for a in actions:
            # Solo control (the ruled design): the agent holds the union of tools, so a
            # golden action tau2 attributes to the user is performed by the agent here.
            # tau2's own solo path does the same re-attribution.
            env.make_tool_call(tool_name=a["name"], requestor="assistant",
                               **Tau2Session._uncanon(a.get("arguments") or {}))
            env.sync_tools()
        vec = self._assertion_vector(env, recipe)
        sig = {
            "repair": [[a["name"], a.get("arguments") or {}] for a in actions],
            "assertions": [[a["func_name"], a.get("arguments") or {}, a.get("assert_value", True)]
                           for a in (recipe.get("env_assertions") or [])],
            "assertions_hold": vec,
        }
        return GroundTruth.statehash(canonical_dumps(sig))

    def _materialized(self, session, recipe):
        """Ensure the session's world carries THIS recipe's fault injection."""
        spec = dict(session.env_spec)
        spec["initialization"] = recipe.get("initialization") or {}
        session.env_spec = spec
        session.rematerialize()
        return session.env

    @staticmethod
    def _assertion_vector(env, recipe) -> list[bool]:
        _ensure_tau2()
        from tau2.data_model.tasks import EnvAssertion

        out = []
        for a in recipe.get("env_assertions") or []:
            out.append(bool(env.run_env_assertion(EnvAssertion(**a), raise_assertion_error=False)))
        return out

    # ------------------------------------------------------------------ validity
    def validate(self, seed: Seed, session) -> bool:
        """The composed recipe must reproduce tau2's own shipped task and satisfy it."""
        try:
            recipe = self.compile(seed.base, seed.conditions)
        except Exception:
            return False
        shipped = seed.shipped_answer or {}
        want = [(a.get("name"), canonical_dumps(a.get("arguments") or {}))
                for a in (shipped.get("actions") or [])]
        got = [(a.get("name"), canonical_dumps(a.get("arguments") or {}))
               for a in (recipe.get("actions") or [])]
        if want and want != got:
            return False
        try:
            gt = self.execute(recipe, session)
        except Exception:
            return False
        sig = json.loads(gt.value)
        return bool(sig["assertions_hold"]) and all(sig["assertions_hold"])

    def is_valid_intent(self, base: dict, conditions: tuple[Condition, ...]) -> bool:
        """tau2's own composition constraints: <=1 fault per selection set, non-empty,
        plus the manager's task validator."""
        manager = base.get("manager")
        mgr = self.managers.get(manager)
        if mgr is None:
            return False
        faults = self._faults_of(conditions)
        if not faults:
            return False
        slots: list[Any] = [None] * len(mgr.selection_sets)
        per = self.universe.get(manager, {})
        for f in faults:
            entry = per.get(f)
            if entry is None:
                return False
            idx, bt = entry
            if slots[idx] is not None:
                return False
            slots[idx] = bt
        if mgr.task_validator is not None and not mgr.task_validator(slots):
            return False
        return True

    def gt_extensional(self, recipe: Any) -> bool:
        # A repair is a single required outcome, not a matching set: subset monotonicity
        # does not apply (same stance as osbench).
        return False

    def gt_equal(self, a: GroundTruth, b: GroundTruth) -> bool:
        return a.kind == b.kind and tuple(a.value) == tuple(b.value)

    # ------------------------------------------------------------------ mining
    def witness(self, base: dict, conditions: tuple[Condition, ...], session) -> list[dict]:
        """Possible worlds in which ONE further fault is also present.

        The engine mines refinements as "a property held by some but not all witness
        objects", which for a matching-set benchmark means a product attribute. Here the
        answer is a repair, not a set, so the honest analogue is: each witness object is a
        world identical to this one except that it also carries fault X. Every candidate
        then appears in exactly one of len(candidates) worlds -- "some but not all" -- and
        the enumerator adds it as a condition, which is precisely refinement.
        """
        manager = base["manager"]
        have = set(self._faults_of(conditions))
        out = []
        for name, (idx, bt) in self.universe.get(manager, {}).items():
            if name in have:
                continue
            cand = tuple(sorted(list(conditions) + [(f"{FAULT}{name}", "=", name)]))
            if self.is_valid_intent(base, cand):
                out.append({f"{FAULT}{name}": name})
        return out

    def canonicalize_conditions(self, conditions: tuple[Condition, ...]) -> tuple[Condition, ...]:
        """Keep the slot identifier in step with its value.

        The generic substitution enumerator swaps a condition's VALUE while keeping its
        slot, which for a self-describing slot like ``fault:airplane_mode_on`` would leave
        the identifier naming a fault that is no longer present. Values are authoritative.
        """
        out = []
        for slot, op, value in conditions:
            if str(slot).startswith(FAULT):
                slot = f"{FAULT}{value}"
            out.append((slot, op, value))
        # a substitution may collide with a fault already present; dedupe deterministically
        seen, uniq = set(), []
        for c in sorted(out):
            if c[0] in seen:
                continue
            seen.add(c[0])
            uniq.append(c)
        return tuple(uniq)

    def domains(self, slot: str, base: dict, session) -> list[Any]:
        """Substitution candidates for a fault slot: other faults valid in its place."""
        if not slot.startswith(FAULT):
            return []
        manager = base["manager"]
        return sorted(self.universe.get(manager, {}))

    def foreign_domains(self, slot: str, base: dict, session) -> list[Any]:
        """Faults from the OTHER intent types -- the raw material for cross-type pivots."""
        manager = base.get("manager")
        return sorted({n for m, per in self.universe.items() if m != manager for n in per})

    # ------------------------------------------------------------------ env
    def env_spec(self, env_key: str) -> dict:
        return {"domain": "telecom", "solo_mode": True, "env_id": env_key,
                "env_key": env_key, "initialization": {}}

    def is_mutating(self, recipe: Any) -> bool:
        return True   # a repair always changes the world

    # ------------------------------------------------------------------ axis B
    shift_operators = ("REFINEMENT",)
    """Which graph edges an intent may shift along mid-episode.

    Substitution, relaxation and pivot are INCOHERENT on this domain: the intent is the
    fault set injected into the world at episode start, so those shifts would require the
    phone to retroactively have had different problems. Refinement is the one coherent
    shape -- a NEW fault appears on the live device mid-call ("now my texts stopped
    working too") -- and it is realized physically by `on_shift` below.
    """

    def on_shift(self, session, old_node, new_node) -> dict:
        """Realize an intent shift in the LIVE world.

        A shift never rewrites history: faults the new intent ADDS are injected into the
        running world (a new problem appears mid-call), and faults it DROPS stay broken --
        the user has simply stopped caring about them. Which edges are allowed at all is
        decided by the executable certificate (shift_edge_ok): the new goal must be
        non-vacuous, genuinely moved, and achievable in exactly this world.
        """
        old_faults = set(self._faults_of(old_node.conditions))
        new_faults = set(self._faults_of(new_node.conditions))
        env = getattr(session, "env", None)
        if env is None:
            raise RuntimeError("no live environment to mutate")
        applied = self._inject_delta(env, old_node, new_node)
        return {"new_faults": sorted(new_faults - old_faults),
                "abandoned_faults": sorted(old_faults - new_faults),
                "applied": applied}

    def _inject_delta(self, env, old_node, new_node) -> list:
        """Apply the new intent's extra fault injections to a live environment."""
        old_recipe = self.compile(dict(old_node.base), tuple(old_node.conditions))
        new_recipe = self.compile(dict(new_node.base), tuple(new_node.conditions))
        seen: dict[str, int] = {}
        for a in old_recipe["initialization"]["initialization_actions"]:
            k = canonical_dumps([a.get("func_name"), a.get("arguments") or {}])
            seen[k] = seen.get(k, 0) + 1
        delta = []
        for a in new_recipe["initialization"]["initialization_actions"]:
            k = canonical_dumps([a.get("func_name"), a.get("arguments") or {}])
            if seen.get(k, 0) > 0:
                seen[k] -= 1
                continue
            delta.append(a)
        _ensure_tau2()
        from tau2.data_model.tasks import EnvFunctionCall
        from ..executors.tau2_inproc import Tau2Session
        applied = []
        for a in delta:
            a = dict(a)
            a["arguments"] = Tau2Session._uncanon(a.get("arguments") or {})
            env.run_env_function_call(EnvFunctionCall(**a))
            applied.append(a.get("func_name"))
        return applied

    def classify_shift(self, src, dst):
        """Sibling moves the slot-set classifier cannot see.

        A telecom fault slot NAMES its fault, so swapping fault A for fault C changes the
        slot set and reads as "unrelated" to the generic classifier. Same manager, same
        fault count, non-identical set = SUBSTITUTION here.
        """
        from ..models import Operator
        if src.base.get("manager") != dst.base.get("manager"):
            return None
        sa = set(self._faults_of(src.conditions))
        sb = set(self._faults_of(dst.conditions))
        if len(sa) == len(sb) and sa != sb and sa & sb:
            return Operator.SUBSTITUTION
        return None

    # ------------------------------------------------------------ certification
    _cert_cache: dict | None = None

    @staticmethod
    def cert_path() -> str:
        import os as _os
        return _os.environ.get(
            "TAU2_SHIFT_CERTS",
            _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                          "..", "..", "..", "..", "artifacts",
                          "telecom_shift_certificates.json"))

    def shift_edge_ok(self, graph, src, dst, op) -> bool:
        """Only certified edges may carry a mid-episode shift.

        The certificate is precomputed by execution (tools/certify_shifts): fail-closed --
        an edge with no certificate is NOT eligible, so an uncertified corpus simply has no
        axis B rather than an unproven one.
        """
        if Tau2TelecomAdapter._cert_cache is None:
            import json as _json
            import os as _os
            path = self.cert_path()
            try:
                with open(path, encoding="utf-8") as f:
                    Tau2TelecomAdapter._cert_cache = _json.load(f)
            except OSError:
                Tau2TelecomAdapter._cert_cache = {}
        certs = Tau2TelecomAdapter._cert_cache.get(graph.graph_id) or {}
        c = certs.get(f"{src.intent_id}->{dst.intent_id}")
        return bool(c and self.cert_eligible(c, getattr(op, "value", str(op))))

    def shift_gt_moved(self, graph, src, dst, default: bool) -> bool:
        """The edge's moved label, from the certificate's execution rather than from
        GT-signature comparison: on this domain two repairs can differ as signatures while
        the old one still satisfies the new goal (relaxation), and P metrics must condition
        on what EXECUTION says."""
        if Tau2TelecomAdapter._cert_cache is None:
            self.shift_edge_ok(graph, src, dst, None)   # loads the cache
        certs = (Tau2TelecomAdapter._cert_cache or {}).get(graph.graph_id) or {}
        c = certs.get(f"{src.intent_id}->{dst.intent_id}")
        return bool(c.get("moved")) if c else default

    def certify_shift(self, session, old_node, new_node) -> dict:
        """The executable certificate for one shift edge (see the user's standard:
        the intent must be ALIGNED WITH ITS GROUND TRUTH -- proven, never assumed).

        Three executions in the exact world the shift would create:
          1. non_vacuous : the new goal's own checks FAIL before its repair
          2. moved       : the OLD intent's repair does NOT satisfy the new goal
          3. achievable  : the NEW intent's repair satisfies the new goal, with
                           abandoned faults left broken
        eligible = non_vacuous and achievable; moved is stored as the edge's label.
        """
        new_recipe = self.compile(dict(new_node.base), tuple(new_node.conditions))
        old_recipe = self.compile(dict(old_node.base), tuple(old_node.conditions))

        def shifted_world():
            spec = dict(session.env_spec)
            spec["initialization"] = old_recipe.get("initialization") or {}
            session.env_spec = spec
            session.rematerialize()
            self._inject_delta(session.env, old_node, new_node)
            return session.env

        def run_fixes(env, recipe):
            from ..executors.tau2_inproc import Tau2Session
            for a in recipe.get("actions") or []:
                try:
                    env.make_tool_call(tool_name=a["name"], requestor="assistant",
                                       **Tau2Session._uncanon(a.get("arguments") or {}))
                    env.sync_tools()
                except Exception:
                    pass    # a repair step the world refuses simply does not help

        out = {"non_vacuous": False, "moved": False, "achievable": False}
        try:
            env = shifted_world()
            out["non_vacuous"] = not all(self._assertion_vector(env, new_recipe))

            env = shifted_world()
            run_fixes(env, old_recipe)
            out["moved"] = not all(self._assertion_vector(env, new_recipe))

            env = shifted_world()
            run_fixes(env, new_recipe)
            out["achievable"] = all(self._assertion_vector(env, new_recipe))
        except Exception as e:
            out["error"] = f"{type(e).__name__}: {str(e)[:120]}"
        return out

    @staticmethod
    def cert_eligible(cert: dict, operator: str) -> bool:
        """Eligibility = the destination intent is a REAL goal that COULD pass.

        Two executable requirements (ruling 2026-08-13): the goal's own checks fail before
        its repair (it is genuinely outstanding, not already satisfied), and the goal's own
        repair passes with abandoned faults left broken (it is achievable). Whether the old
        answer still satisfies the new goal -- `moved` -- is a LABEL, never a gate: a shift
        may move the answer or not, and GRIP uses the label to decide whether the episode
        measures staleness (moved) or over-reaction (unmoved), exactly as WebShop's unmoved
        edges were kept and labelled rather than discarded.
        """
        return bool(cert.get("non_vacuous") and cert.get("achievable"))

    # ------------------------------------------------------------------ episode
    def describe_pivot(self, base: dict) -> str:
        return {"mobile_data_issue": "a problem with mobile data",
                "service_issue": "a problem with phone service",
                "mms_issue": "a problem sending or receiving picture messages"}.get(
                    base.get("manager"), "a problem with the phone")

    def slot_phrase(self, slot: str) -> str:
        if slot.startswith(FAULT):
            return self.fault_description(slot[len(FAULT):])
        return slot

    def spoken_value(self, slot: str, value) -> str:
        """A fault slot's value IS the fault; its plain-words symptom is what a customer
        says (never the internal fault id)."""
        return self.fault_description(str(value))

    def render_context(self, base: dict) -> str:
        return ("The user is a mobile customer contacting technical support because "
                "something on their phone is not working.")


    # ------------------------------------------------------------------ generation
    def describe_intent(self, conditions) -> str:
        """The user's problem, in the words tau2 itself uses for each fault."""
        lines = []
        for slot, _op, value in sorted(conditions):
            if str(slot).startswith(FAULT):
                lines.append(f"  - {self.fault_description(str(value))}")
        return "\n".join(lines) or "  (no stated problem)"

    generation_voice = "a plain mobile-phone customer's voice, describing symptoms"
    generation_domain_word = "phone support"

    # The fault vocabulary is closed (20 primitives), so an "irrelevant extra detail" can
    # never compile into a condition. Such samples are admitted on their string checks and
    # LABELLED non-executable (signature.executable=False) rather than lost.
    # FLAGGED FOR REVIEW: this is a contract loosening, scoped to one fault type.
    extras_may_be_pragmatic = True
    fe_targeted = True            # see genverify.fe_extraction_prompt

    def dynamic_gen_hint(self, strategy_id: str, conditions) -> str:
        """Per-intent guidance the static hint table cannot give.

        For factual_error the substitute problem must (a) be a REAL fault in this intent
        type and (b) compose validly with the problems that remain stated -- otherwise the
        literal reading cannot execute and admission burns the attempt. Both are decidable
        here, so the generator is handed concrete valid choices instead of being left to
        fabricate ("my network keeps dropping") or to negate a true problem, the two
        failure modes the pilot rejections showed.
        """
        if strategy_id != "factual_error":
            return ""
        base = {"kind": "telecom_repair"}
        faults = self._faults_of(tuple(conditions))
        manager = None
        for m, per in self.universe.items():
            if all(f in per for f in faults):
                manager = m
                break
        if manager is None:
            return ""
        base["manager"] = manager
        options = []
        for drop in faults:
            rest = [f for f in faults if f != drop]
            for cand, (idx, bt) in self.universe[manager].items():
                if cand in faults:
                    continue
                trial = tuple(sorted((f"{FAULT}{f}", "=", f) for f in rest + [cand]))
                if self.is_valid_intent(base, trial):
                    options.append((drop, cand, bt.description))
        if not options:
            return ""
        lines = [f"- REPLACE the problem \"{self.fault_description(d)}\" with the invented "
                 f"claim \"{desc}\" (which is false for you)"
                 for d, _c, desc in options[:4]]
        return ("- do it EXACTLY like one of these (pick one):\n  " + "\n  ".join(lines) +
                "\n- never say a true problem is absent or fine -- leave the replaced one "
                "unmentioned entirely\n"
                "- every other problem must be stated plainly and accurately")

    def all_condition_values(self) -> list:
        """Every fault that exists, so an annotator can name the wrongly-reported one."""
        return sorted(self.all_fault_names())

    def slot_supports_subjective(self, slot: str) -> bool:
        """A fault is something the customer FEELS, so it takes a subjective surface.

        "my data has been really flaky", "the signal is terrible lately" name the problem
        vaguely without saying which setting is wrong -- Drift-bench's vagueness flaw
        exactly. There is no numeric order here, which is why the WebShop rule (a price to
        be "cheap" against) rejected every telecom slot.
        """
        return str(slot).startswith(FAULT)

    fp_note = (
        "In this domain the false assumption is normally a claim that one of the problems\n"
        "has ALREADY been dealt with -- 'your team enabled roaming for me last week', 'the\n"
        "data was topped up yesterday'. Record that clause as `presupposition`, and put the\n"
        "true requirement it concerns (the problem the user wrongly believes is handled) in\n"
        "`assumed_unsatisfiable` as [slot, value]. `fallback` is null unless the user\n"
        "settles for something else instead.\n"
               "The requirement named in `concerns` is PART OF the presupposition, not a\n"
               "separate change: never also list it in `other_changes`. Leave\n"
               "`other_changes` empty unless some DIFFERENT requirement was altered.\n")

    # Per-domain generation hints. The shipped hints are shopping-specific ("NOT the
    # price... currencies, brands"), and with them the generator restated every fault
    # accurately instead of injecting the flaw. These say what each flaw MEANS when the
    # requirements are symptoms rather than product attributes.
    gen_hints = {
        "factual_error": (
            "- report a DIFFERENT problem in place of one you actually have: pick a real\n"
            "  phone problem you do NOT have (wifi calling switched off, the SIM not seated,\n"
            "  roaming disabled, a VPN interfering, data saver on) and describe THAT instead\n"
            "  of the true one, in your own words. Keep every OTHER problem accurate.\n"
            "- a synonym or paraphrase of a true problem is NOT a factual error"),
        "insufficient_information": (
            "- leave one or more problems out COMPLETELY: say nothing that hints at them,\n"
            "  not even vaguely. State the remaining problems plainly."),
        "vagueness_subjectivity": (
            "- describe ONE of the problems only through a subjective, fuzzy judgement --\n"
            "  'my data has been really flaky', 'the signal is terrible', 'things feel off'\n"
            "  -- so it is clear something is wrong with it but NOT which setting. State\n"
            "  every OTHER problem plainly and specifically."),
        "false_presupposition": (
            "- take for granted a PAST EVENT or given fact that is not true, as a\n"
            "  subordinate aside: 'since support already reset my line yesterday...',\n"
            "  'now that my plan was upgraded...'. It must NOT introduce a new symptom or\n"
            "  change any problem you report. State every real problem exactly as it is."),
    }

    extraction_note = (
        "\nHow to read this domain:\n"
        "  - Each slot is a PROBLEM the user's phone actually has. The slot value is an\n"
        "    internal identifier; the plain-words line above is what it means.\n"
        "  - A problem is STATED if its meaning is conveyed in any wording. A paraphrase or\n"
        "    synonym ('flight mode is on' for airplane_mode_on) is the same problem stated\n"
        "    accurately -- it is NOT a substituted or falsified value.\n"
        "  - A problem is OMITTED only if the instruction conveys nothing about it.\n"
        "  - Count a problem as substituted only when the instruction asserts a DIFFERENT\n"
        "    problem than the true one (e.g. says bluetooth is off when the true problem is\n"
        "    airplane mode). Record it as [slot, the slot's true value, the wrong thing the\n"
        "    user said], using the slot's own value string for the middle element.\n"
        "  - An aside about a PAST EVENT ('support already reset my line') is not a problem\n"
        "    at all: it is a presupposition, not a substituted or extra requirement.\n")

    _tool_manual_cache: str | None = None
    _policy_cache: str | None = None

    roleplay_voice = "mobile-phone customer calling tech support"

    @staticmethod
    def proposal_frame(item_summary: str) -> str:
        return ("The support agent says everything should be fixed now. Think about "
                "whether the problems you called about are actually resolved for you.\n"
                "Is everything working, as far as you can tell?")

    final_answer_format = (
        "Final answer required now, in exactly this form:\n"
        "Action: Answer\nFinal Answer: done\n"
        "The phone's state as you leave it is what gets checked. If a fix is still "
        "pending, apply it FIRST; an apology or a question scores zero.")


    def agent_domain_prompt(self) -> str:
        """tau2 telecom framing: full tool manual (both tool groups) + tau2's own solo
        policy/workflow. Measured: without the manual the agent invents tool names all
        episode; without the policy it toggles fixed settings back off (0/8 clean)."""
        return (
            "\nYou are a telecom support agent fixing a customer's phone problems. Use "
            "the tools below by writing:\n"
            "Action: Operation\n```\ntool_name(arg=\"value\", ...)\n```\n"
            "One tool call per turn; the observation follows.\n\n"
            "TOOLS:\n" + self.tool_manual() + "\n\n"
            "IMPORTANT -- you cannot chat with the customer. There is no message "
            "channel: anything you put in Action: Answer is treated as your FINAL "
            "submission, never as a question, so 'could you provide your phone number' "
            "ends the episode as a failed answer. Everything you need is discoverable "
            "with the tools (get_user_info gives the account and phone line; the "
            "check_* tools diagnose the device). Work the tools.\n\n"
            "When everything the customer needs is fixed, finish with:\n"
            "Action: Answer\nFinal Answer: done\n"
            "The device and account state are what get checked, not your words.\n\n"
            + _DISTILLED_OPS + "\n")

    def agent_policy(self) -> str:
        """tau2's OWN agent policy and troubleshooting manual for this domain.

        Not optional context: without it the agent does not know that the handset controls
        are TOGGLES, and it flips settings it has already fixed back off (measured -- the
        clean-query control arm scored 0/8 and the transcripts end with 'Mobile Data is now
        OFF' after the agent had just enabled it). tau2 gives its own agents this text, so
        using it is also what keeps our numbers comparable to the benchmark's.
        """
        if Tau2TelecomAdapter._policy_cache is not None:
            return Tau2TelecomAdapter._policy_cache
        root = os.environ.get("TAU2_DATA", "") + "/domains/telecom/"
        parts = []
        for fname in ("main_policy_solo.md", "tech_support_workflow_solo.md"):
            try:
                with open(root + fname, encoding="utf-8") as f:
                    parts.append(f.read())
            except OSError:
                continue
        Tau2TelecomAdapter._policy_cache = "\n\n".join(parts)
        return Tau2TelecomAdapter._policy_cache


    def tool_manual(self, session=None) -> str:
        """The agent's real affordance: every tool tau2 exposes in solo mode.

        Without this the agent invents plausible tool names (`get_line_state`,
        `get_account_info`) and burns the whole episode on NameErrors -- observed on every
        episode of the first telecom smoke run.
        """
        if Tau2TelecomAdapter._tool_manual_cache:
            return Tau2TelecomAdapter._tool_manual_cache
        env = getattr(session, "env", None)
        if env is None:
            _ensure_tau2()
            from tau2.registry import registry
            env = registry.get_env_constructor("telecom")(solo_mode=True)
        lines = []
        for group, tools in (("account and line (support system)", env.get_tools()),
                             ("the customer's handset (you can operate it directly)",
                              env.get_user_tools())):
            lines.append(f"  -- {group} --")
            for tool in tools:
                name = getattr(tool, "name", None) or getattr(tool, "__name__", "?")
                desc, args = "", []
                schema = getattr(tool, "openai_schema", None)
                if isinstance(schema, dict):
                    fn = schema.get("function") or {}
                    desc = " ".join(str(fn.get("description") or "").split())[:90]
                    props = ((fn.get("parameters") or {}).get("properties") or {})
                    required = set((fn.get("parameters") or {}).get("required") or [])
                    for pname, pinfo in props.items():
                        ptype = pinfo.get("type", "value")
                        args.append(f"{pname}: {ptype}" + ("" if pname in required else " (optional)"))
                # The signature is the point: without argument names the agent guesses
                # (`phone=`, `id=`, `customer_id=`) and burns the episode on TypeErrors --
                # measured on every episode of the first telecom batch.
                sig = ", ".join(args) if args else "no arguments"
                lines.append(f"  {name}({sig})" + (f" -- {desc}" if desc else ""))
        Tau2TelecomAdapter._tool_manual_cache = "\n".join(lines)
        return Tau2TelecomAdapter._tool_manual_cache

    def parse_proposal(self, raw: str) -> Any:
        """The agent declares the repair complete; the world is the answer."""
        return {"declared_done": True, "raw": raw}

    def accepts(self, proposal: Any, node, session, executor=None) -> tuple[bool, str]:
        """Executable acceptance: the CURRENT node's own assertions, on the live world.

        This is tau2's own success criterion (its ENV_ASSERTION reward basis), evaluated
        against the environment the agent has actually been working in -- so an answer that
        was right before the intent moved is rejected after it moves.
        """
        env = getattr(session, "env", None)
        if env is None:
            return False, "no_environment"
        recipe = getattr(node, "recipe", None) or {}
        assertions = recipe.get("env_assertions") or []
        if not assertions:
            return False, "node_has_no_assertions"
        try:
            vec = self._assertion_vector(env, recipe)
        except Exception as e:
            return False, f"assertion_error:{type(e).__name__}"
        if all(vec):
            return True, "all_assertions_hold"
        failed = sum(1 for v in vec if not v)
        return False, f"unresolved_faults:{failed}"

    _REPAIR_HEADS = ("toggle_", "set_", "reseat_", "refuel_", "enable_", "disable_",
                     "connect_", "disconnect_", "reset_", "resume_", "suspend_",
                     "send_payment", "make_payment", "grant_", "remove_", "pay_")

    def act_is_mutating(self, raw: str) -> bool:
        """Only REPAIR actions commit the agent to the current intent.

        Diagnostics (check_*, get_*, run_speed_test) read the world without changing it;
        counting them as mutations forbade every shift from the first turn -- measured:
        0/20 episodes shifted in the first axis-B pilot.
        """
        head = (raw or "").strip().strip("`").split("(")[0].strip()
        return head.startswith(self._REPAIR_HEADS)


from .base import register  # noqa: E402
from ..executors.tau2_inproc import Tau2Executor  # noqa: E402

register("tau2_telecom", Tau2TelecomAdapter, Tau2Executor)
