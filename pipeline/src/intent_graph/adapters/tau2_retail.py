"""tau2-bench retail adapter.

A retail task is a customer asking an agent to change something about their orders, on a
shared shop database. Unlike telecom, retail intents are PARAMETER-shaped -- the golden
actions carry the user's requirements as typed arguments (which order, which items, which
replacement variant, which payment method) -- so conditions here are argument slots, the
same shape WebShop uses. That was the ruling: the condition unit is chosen per domain
because the domains are genuinely different, and using each one's natural unit is what
keeps the fault taxonomy applicable on both.

Two facts measured in the spike shape this adapter:

* **Read actions are mechanics, not conditions.**  Replaying only the WRITE actions
  reproduced the full recipe's database hash on 114/114 tasks, so the user's requirements
  live exclusively in write-action arguments; lookups (find_user, get_order_details) are
  how an agent gets there, not what the user wants.
* **15/114 shipped recipes raise during their own golden replay** (deliberate wrong-lookup
  task design) and **11/114 leave the database unchanged**. Both are handled here rather
  than treated as broken seeds.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

from ..ids import canonical_dumps
from ..models import Condition, GroundTruth, Seed

ARG = "arg:"


def _ensure_tau2() -> None:
    src = os.environ.get("TAU2_SRC")
    if src and src not in sys.path:
        sys.path.insert(0, src)


# Write tools: the ones whose arguments carry user requirements.
WRITE_TOOLS = {
    "exchange_delivered_order_items", "return_delivered_order_items",
    "cancel_pending_order", "modify_pending_order_items",
    "modify_pending_order_address", "modify_pending_order_payment",
    "modify_user_address",
}


class Tau2RetailAdapter:

    # A read-only, no-argument call the harness may substitute when an arm's own
    # rule forbids the action it chose (see LLMAgent._harness_action). WebShop's
    # `search[...]` is not a tau2 action, so before this existed the substitution
    # could only produce a parse error and burn the turn.
    agent_safe_action = "list_all_product_types()"
    name = "tau2_retail"
    version = "1"
    executor_name = "tau2_inproc"

    ENV_KEY = "tau2_retail_shop"

    def __init__(self, config: dict | None = None) -> None:
        cfg = (config or {}).get("tau2_retail", {}) if config else {}
        base = os.environ.get("TAU2_DATA", "") + "/domains/retail/"
        self.tasks_path = cfg.get("tasks_path", base + "tasks.json")
        self.db_path = cfg.get("db_path", base + "db.json")
        self._db = None

    # ------------------------------------------------------------------ data
    @property
    def db(self) -> dict:
        if self._db is None:
            with open(self.db_path, encoding="utf-8") as f:
                self._db = json.load(f)
        return self._db

    # ------------------------------------------------------------------ load
    def load(self) -> list[Seed]:
        with open(self.tasks_path, encoding="utf-8") as f:
            tasks = json.load(f)
        seeds: list[Seed] = []
        for t in tasks:
            crit = t.get("evaluation_criteria") or {}
            actions = crit.get("actions") or []
            writes = [a for a in actions if a.get("name") in WRITE_TOOLS]
            if not writes:
                continue                      # read-only/refusal task: no state to move
            conds = []
            for i, a in enumerate(writes):
                for k, v in sorted((a.get("arguments") or {}).items()):
                    conds.append((f"{ARG}{i}:{a['name']}:{k}", "=", canonical_dumps(v)))
            if not conds:
                continue
            instr = ((t.get("user_scenario") or {}).get("instructions") or {})
            seeds.append(Seed(
                record_id=str(t.get("id")),
                env_key=self.ENV_KEY,
                base={"kind": "retail_request",
                      "writes": [a["name"] for a in writes],
                      "reads": [a["name"] for a in actions if a.get("name") not in WRITE_TOOLS]},
                conditions=tuple(sorted(conds)),
                shipped_answer={"actions": actions},
                meta={"n_writes": len(writes),
                      "reason_for_call": instr.get("reason_for_call") if isinstance(instr, dict) else None,
                      "known_info": instr.get("known_info") if isinstance(instr, dict) else None,
                      "unknown_info": instr.get("unknown_info") if isinstance(instr, dict) else None},
            ))
        return seeds

    # ------------------------------------------------------------------ recipes
    def compile(self, base: dict, conditions: tuple[Condition, ...]) -> Any:
        """Rebuild the write-action sequence from its argument slots."""
        by_action: dict[int, dict] = {}
        for slot, _op, value in conditions:
            if not str(slot).startswith(ARG):
                continue
            body = str(slot)[len(ARG):]
            idx, tool, key = body.split(":", 2)
            entry = by_action.setdefault(int(idx), {"name": tool, "arguments": {}})
            try:
                entry["arguments"][key] = json.loads(value)
            except (TypeError, json.JSONDecodeError):
                entry["arguments"][key] = value
        actions = [by_action[i] for i in sorted(by_action)]
        # `also:` conditions: one condition = one whole ADDED request (a refinement).
        # The value is the complete argument dict, so adding the condition adds an
        # executable action and dropping it removes one -- which is exactly how the
        # engine's set-based operator classification expects refinement to look.
        for slot, _op, value in sorted(conditions):
            if not str(slot).startswith("also:"):
                continue
            _tag, _idx, tool = str(slot).split(":", 2)
            try:
                args = json.loads(value)
            except (TypeError, json.JSONDecodeError):
                continue
            actions.append({"name": tool, "arguments": args})
        return {"actions": actions, "reads": base.get("reads") or []}

    def execute(self, recipe: Any, session) -> GroundTruth:
        """Ground truth = the shop database after the writes, by tau2's own hash.

        Pair order is canonicalized here exactly as at the agent boundary, so the stored
        GT and a semantically-equal agent call always land on the same state despite
        tau2's pair-order-sensitive modify/exchange (see Tau2Session._canon_pairs)."""
        from ..executors.tau2_inproc import Tau2Session
        session.rematerialize()
        env = session.env
        errors = []
        for a in recipe.get("actions") or []:
            try:
                args = Tau2Session._canon_pairs(a["name"], a.get("arguments") or {})
                env.make_tool_call(tool_name=a["name"], requestor="assistant", **args)
                env.sync_tools()
            except Exception as e:
                errors.append(f"{a['name']}: {type(e).__name__}")
        if errors:
            # A rejected write means this intent is not achievable in the shop as-is. For
            # SEEDS that is a validation failure; for BRANCH CANDIDATES the engine drops
            # candidates whose execution raises -- embedding the error in the hash instead
            # would let an unachievable branch masquerade as a moved ground truth.
            raise ValueError("write_rejected: " + "; ".join(errors))
        digest = env.get_db_hash()
        return GroundTruth.statehash(canonical_dumps({"db": digest, "errors": []}))

    def validate(self, seed: Seed, session) -> bool:
        try:
            recipe = self.compile(seed.base, seed.conditions)
            gt = self.execute(recipe, session)
        except Exception:
            return False
        payload = json.loads(gt.value)
        session.rematerialize()
        return payload["db"] != session.env.get_db_hash()   # must actually change the shop

    def is_valid_intent(self, base: dict, conditions: tuple[Condition, ...]) -> bool:
        return bool([c for c in conditions if str(c[0]).startswith(ARG)])

    def gt_extensional(self, recipe: Any) -> bool:
        return False

    def gt_equal(self, a: GroundTruth, b: GroundTruth) -> bool:
        return a.kind == b.kind and a.value == b.value

    # ------------------------------------------------------------------ mining
    def _order_of(self, conditions) -> tuple[str | None, str | None]:
        """(order_id, user_id) of the seed's own request, from the shop database."""
        for slot, _op, value in conditions:
            if str(slot).endswith(":order_id"):
                try:
                    oid = json.loads(value)
                except (TypeError, json.JSONDecodeError):
                    oid = value
                order = (self.db.get("orders") or {}).get(str(oid))
                if order:
                    return str(oid), order.get("user_id")
        return None, None

    def witness(self, base: dict, conditions: tuple[Condition, ...], session) -> list[dict]:
        """Refinement candidates: further requests this SAME customer could really make.

        Each witness object carries one `also:` slot whose value is a complete, executable
        extra request, mined from the shop database: cancelling another of the customer's
        pending orders, or moving another pending order onto a different payment method
        they actually hold. The engine's some-but-not-all rule then offers each as a
        one-condition refinement, and execution (which now raises on any rejected write)
        filters out combinations the shop's own rules refuse.
        """
        _oid, uid = self._order_of(conditions)
        if not uid:
            return []
        user = (self.db.get("users") or {}).get(uid) or {}
        touched = set()
        for slot, _op, value in conditions:
            if str(slot).endswith(":order_id"):
                try:
                    touched.add(str(json.loads(value)))
                except (TypeError, json.JSONDecodeError):
                    touched.add(str(value))
        out, n = [], 0
        pay_ids = list(user.get("payment_methods") or {})
        for other in user.get("orders") or []:
            if str(other) in touched:
                continue
            order = (self.db.get("orders") or {}).get(str(other)) or {}
            if order.get("status") != "pending":
                continue
            out.append({f"also:{n}:cancel_pending_order": canonical_dumps(
                {"order_id": str(other), "reason": "no longer needed"})})
            n += 1
            current_pay = ((order.get("payment_history") or [{}])[0] or {}).get("payment_method_id")
            for pid in pay_ids:
                if pid != current_pay:
                    out.append({f"also:{n}:modify_pending_order_payment": canonical_dumps(
                        {"order_id": str(other), "payment_method_id": pid})})
                    n += 1
                    break
            if n >= 6:
                break
        return out

    def extra_candidates(self, seed):
        """Relaxations the generic enumerator cannot express: drop a WHOLE request.

        A retail intent with several write actions relaxes by the customer abandoning one
        of them ("actually leave the address as it is"), i.e. removing ALL of that
        action's conditions together. Removing a single condition instead produces a write
        with a missing required argument, which the shop rejects -- the strict-execution
        fix exposed that v1's single-condition relaxations were exactly that.
        """
        from ..engine import Candidate, Provenance
        from ..models import Operator, sort_conditions
        writes = seed.base.get("writes") or []
        if len(writes) < 2:
            return []
        out = []
        for i in range(len(writes)):
            prefix = f"{ARG}{i}:"
            rest = tuple(c for c in seed.conditions if not str(c[0]).startswith(prefix))
            if len(rest) == len(seed.conditions) or not rest:
                continue
            new_writes = [w for j, w in enumerate(writes) if j != i]
            # re-index the remaining arg: conditions so compile() sees a dense sequence
            remap = {}
            fixed = []
            for slot, op, value in rest:
                if str(slot).startswith(ARG):
                    body = str(slot)[len(ARG):]
                    idx, tool, key = body.split(":", 2)
                    new_idx = remap.setdefault(int(idx), len(remap))
                    slot = f"{ARG}{new_idx}:{tool}:{key}"
                fixed.append((slot, op, value))
            out.append(Candidate(
                conditions=sort_conditions(fixed),
                base={**seed.base, "writes": new_writes},
                operator=Operator.RELAXATION,
                delta={"removed_request": writes[i]},
                provenance=Provenance.SYNTHETIC,
            ))
        return out

    def domains(self, slot: str, base: dict, session) -> list[Any]:
        """Substitution candidates mined from the shop itself.

        Only variants of the SAME product are offered for an item slot: a replacement from
        a different product is rejected by the tool, so proposing one would manufacture
        branches that cannot execute.
        """
        if not str(slot).startswith(ARG):
            return []
        key = str(slot).rsplit(":", 1)[-1]
        out: list[Any] = []
        if key in ("new_item_ids", "item_ids"):
            for prod in (self.db.get("products") or {}).values():
                variants = list((prod.get("variants") or {}).keys())
                if len(variants) > 1:
                    out.extend(canonical_dumps([v]) for v in variants[:4])
        elif key == "payment_method_id":
            for user in (self.db.get("users") or {}).values():
                out.extend(canonical_dumps(p) for p in (user.get("payment_methods") or {}))
        return out[:40]

    # ------------------------------------------------------------------ env
    def env_spec(self, env_key: str) -> dict:
        return {"domain": "retail", "solo_mode": False, "env_id": env_key,
                "env_key": env_key, "initialization": {}}

    def is_mutating(self, recipe: Any) -> bool:
        return True

    # ------------------------------------------------------------------ episode
    def slot_phrase(self, slot: str) -> str:
        if str(slot).startswith("also:"):
            _t, _i, tool = str(slot).split(":", 2)
            return "an additional request (" + tool.replace("_", " ") + ")"
        body = str(slot)[len(ARG):] if str(slot).startswith(ARG) else str(slot)
        try:
            _idx, tool, key = body.split(":", 2)
        except ValueError:
            return body
        label = {"order_id": "which order", "item_ids": "which item to change",
                 "new_item_ids": "what to change it to",
                 "payment_method_id": "which payment method",
                 "reason": "the reason", "address1": "the street address",
                 "city": "the city", "state": "the state", "zip": "the postcode",
                 "country": "the country"}.get(key, key.replace("_", " "))
        verb = tool.replace("_", " ")
        return f"{label} (for the {verb})"

    def spoken_value(self, slot: str, value) -> str:
        """The words for the VALUE itself (the label is slot_phrase's job): a customer
        reads item and order ids off their order page, so ids are spoken verbatim."""
        v = value
        if isinstance(v, str):
            try:
                v = json.loads(v)
            except (TypeError, json.JSONDecodeError):
                pass
        if isinstance(v, list):
            return ", ".join(str(x) for x in v)
        if isinstance(v, dict):
            return ", ".join(f"{k} {x}" for k, x in v.items())
        return str(v)

    def describe_intent(self, conditions) -> str:
        return "\n".join(f"  - {self.slot_phrase(s)}: {v}" for s, _o, v in sorted(conditions))

    generation_voice = "a plain online-shopping customer's voice"
    generation_domain_word = "online shopping"

    # Same closed-vocabulary problem as telecom: a retail condition is an ARGUMENT of a
    # specific write action, so an "extra irrelevant detail" has no slot to compile into.
    # Admitted on string checks and labelled non-executable. FLAGGED FOR REVIEW.
    extras_may_be_pragmatic = True

    def slot_supports_subjective(self, slot: str) -> bool:
        """Which item to swap to is a judgement the shopper can make vaguely.

        Product variants differ in price and specification, so "swap it for a nicer one",
        "the cheaper version" are meaningful without naming the item id -- the ordering is
        over the catalogue rather than over the slot's own value, which is why the numeric
        rule saw nothing orderable here.
        """
        key = str(slot).rsplit(":", 1)[-1]
        return key in ("new_item_ids", "item_ids")

    gen_hints = {
        "vagueness_subjectivity": (
            "- say which item you want changed only through a subjective judgement --\n"
            "  'swap it for a nicer one', 'the cheaper version', 'something better' --\n"
            "  never naming the replacement. State every OTHER requirement exactly."),
    }

    fp_note = ("A shopping presupposition is usually that part of the request has ALREADY\n"
               "been handled ('since you already refunded the other item...') or that\n"
               "something is unavailable. Record the clause as `presupposition` and the true\n"
               "requirement it concerns in `concerns` as [slot, value].\n"
               "The requirement named in `concerns` is PART OF the presupposition, not a\n"
               "separate change: never also list it in `other_changes`. Leave\n"
               "`other_changes` empty unless some DIFFERENT requirement was altered.\n")

    extraction_note = (
        "\nHow to read this domain:\n"
        "  - Each slot is one requirement of the change the customer wants made to their\n"
        "    order. Values are identifiers as the shop stores them.\n"
        "  - A requirement is STATED if the customer conveys it in any wording, including\n"
        "    describing an item rather than naming its id.\n")

    def render_context(self, base: dict) -> str:
        return ("The user is an online-shopping customer contacting support about an order "
                "they have already placed.")

    _tool_manual_cache: str | None = None
    _policy_cache: str | None = None

    def tool_manual(self, session=None) -> str:
        if Tau2RetailAdapter._tool_manual_cache:
            return Tau2RetailAdapter._tool_manual_cache
        _ensure_tau2()
        from tau2.registry import registry
        env = getattr(session, "env", None) or registry.get_env_constructor("retail")()
        lines = []
        for tool in env.get_tools():
            schema = getattr(tool, "openai_schema", None) or {}
            fn = schema.get("function") or {}
            props = ((fn.get("parameters") or {}).get("properties") or {})
            required = set((fn.get("parameters") or {}).get("required") or [])
            args = ", ".join(f"{p}: {i.get('type','value')}"
                             + ("" if p in required else " (optional)")
                             for p, i in props.items()) or "no arguments"
            desc = " ".join(str(fn.get("description") or "").split())[:90]
            lines.append(f"  {getattr(tool,'name','?')}({args}) -- {desc}")
        Tau2RetailAdapter._tool_manual_cache = "\n".join(lines)
        return Tau2RetailAdapter._tool_manual_cache

    def agent_policy(self) -> str:
        # O8 FIX (2026-08-23): the file on disk is `policy.md`; this method read
        # `main_policy.md`, failed silently, and every retail campaign through R3 ran
        # with an EMPTY policy block (register O8). policy.md is now primary;
        # main_policy.md stays as a legacy fallback; a MISSING policy warns loudly
        # instead of failing silent -- the empty string was the bug class.
        if Tau2RetailAdapter._policy_cache is not None:
            return Tau2RetailAdapter._policy_cache
        base = os.environ.get("TAU2_DATA", "") + "/domains/retail/"
        for name in ("policy.md", "main_policy.md"):
            try:
                with open(base + name, encoding="utf-8") as f:
                    Tau2RetailAdapter._policy_cache = f.read()
                    return Tau2RetailAdapter._policy_cache
            except OSError:
                continue
        import logging
        logging.getLogger(__name__).warning(
            "retail policy file missing under %s -- agents run POLICY-OFF", base)
        Tau2RetailAdapter._policy_cache = ""
        return Tau2RetailAdapter._policy_cache

    roleplay_voice = "online-shop customer"

    @staticmethod
    def proposal_frame(item_summary: str) -> str:
        return ("The agent says it has now finished handling your request(s). Think about "
                "whether EVERYTHING you wanted has actually been taken care of.\n"
                "Is it all done, as far as you are concerned?")

    final_answer_format = (
        "Final answer required now, in exactly this form:\n"
        "Action: Answer\nFinal Answer: done\n"
        "The database as you leave it is what gets checked. If any requested write has "
        "not been made yet, make it FIRST; an apology or a question scores zero.")

    def agent_domain_prompt(self) -> str:
        """tau2 retail framing: tool manual WITH signatures + tau2's own operator policy.

        The id-format lesson is task literacy, not answer leakage: the AUTOPSY
        (2026-08-14) measured 90.4% of all write attempts dying on "Order not found",
        97.5% of them because the '#' prefix was dropped from an order id the query
        stated correctly. Every id is user-visible information; the note only teaches
        the FORM the tools expect, exactly like WebShop's search-literacy note.
        """
        return (
            "\nYou are a customer-service agent for an online shop, acting on the "
            "customer's behalf. Use the tools below by writing:\n"
            "Action: Operation\n```\ntool_name(arg=\"value\", ...)\n```\n"
            "One tool call per turn; the observation follows.\n\n"
            "TOOLS:\n" + self.tool_manual() + "\n\n"
            "ID FORMAT (basic tool literacy): order ids start with '#' and must be "
            "passed EXACTLY as the customer states them -- '#W1234567', never "
            "'W1234567'. Item and user ids are copied verbatim too. If a lookup says "
            "'not found', re-check the id's exact form before concluding anything.\n\n"
            "IMPORTANT -- anything you put in Action: Answer is treated as your FINAL "
            "submission, never as a message to the customer: a question or an apology "
            "there ends the episode as a failed answer. Identify the customer with the "
            "lookup tools (find_user_id_by_name_zip, find_user_id_by_email) and read "
            "their orders; do not ask for what the tools can tell you.\n\n"
            "When every requested change has been made, finish with:\n"
            "Action: Answer\nFinal Answer: done\n"
            "The database state is what gets checked, not your words.\n\n"
            + self._policy_block())

    # POLICY IS OPTIONAL ON RETAIL (ruling 2026-08-26). Acceptance here is
    # `accept_actions_only` -- the golden WRITE actions -- so the procedural steps the
    # store policy mandates (authenticate, verify, confirm) earn no credit, while the
    # patience economy charges a turn for each one. Measured causally, deepseek A0 over
    # 80 identical samples, policy-ON minus policy-OFF: Success -13.8 (z=-2.17), patience
    # -0.85 (z=-2.39), EXHAUSTED 46.2% vs 31.2%. Native tau2 credits compliance so the
    # cost is matched there; we grade writes only, so policy-ON is cost without credit.
    # Retail therefore runs POLICY-OFF, matching the 11-arm campaign. Airline keeps its
    # policy (that domain's rules carry information the golden actions actually need).
    include_policy = True

    def _policy_block(self) -> str:
        return ("STORE POLICY (binding):\n" + self.agent_policy() + "\n"
                if self.include_policy else "")

    def unmet_requests(self, session, node) -> list[str]:
        """The user's OWN unfinished requests, as a customer would name them after
        checking their orders. Grounded in the golden-action gap (their intent, never
        the answer sheet): the same keying as the golden-action grading."""
        from ..executors.tau2_inproc import Tau2Session
        recipe = self.compile(dict(node.base), tuple(node.conditions))
        done = {self._action_key(n, k)
                for n, k in getattr(session, "executed_calls", [])}
        out = []
        for a in recipe.get("actions") or []:
            args = Tau2Session._uncanon(a.get("arguments") or {})
            if self._action_key(a["name"], args) in done:
                continue
            verb = {"exchange_delivered_order_items": "the exchange",
                    "return_delivered_order_items": "the return",
                    "cancel_pending_order": "the cancellation",
                    "modify_pending_order_items": "the item change",
                    "modify_pending_order_address": "the delivery-address change",
                    "modify_pending_order_payment": "the payment-method change",
                    "modify_user_address": "my profile-address update"}.get(
                        a["name"], a["name"].replace("_", " "))
            oid = args.get("order_id")
            bits = [verb + (f" for order {oid}" if oid else "")]
            # the customer states THEIR OWN specifics -- without the values a no-ask
            # agent can rediscover the order but never the user's choices (measured
            # 2026-08-17: 74/86 shifted B0 failures were exactly this information gap)
            if args.get("item_ids"):
                bits.append("items " + ", ".join(str(x) for x in args["item_ids"]))
            if args.get("new_item_ids"):
                bits.append("swapped to " + ", ".join(str(x) for x in args["new_item_ids"]))
            if args.get("payment_method_id"):
                bits.append(f"via {args['payment_method_id']}")
            for k in ("address1", "city"):
                if args.get(k):
                    bits.append(f"new address {args.get('address1','')} {args.get('city','')}".strip())
                    break
            out.append(" -- ".join(bits))
        return out

    def parse_proposal(self, raw: str) -> Any:
        return {"declared_done": True, "raw": raw}

    # Golden-action grading (ruling 2026-08-17: "golden action check could be involved"):
    # success requires the expected tool CALLS to have been made, not just their end
    # state reached -- tau2's own ACTION reward component, adopted for retail. Telecom
    # stays assertion-based: alternate legitimate repair paths would be falsely punished.
    golden_action_check = True

    @staticmethod
    def _action_key(name: str, args: dict) -> tuple:
        """Canonical identity of one tool call. Exchange/modify item lists are PAIRED by
        position, so they compare as a set of (old, new) pairs -- a consistently reordered
        call matches, a re-paired one never does."""
        a = {k: v for k, v in (args or {}).items()}
        if name in ("exchange_delivered_order_items", "modify_pending_order_items"):
            old = [str(x) for x in (a.pop("item_ids", None) or [])]
            new = [str(x) for x in (a.pop("new_item_ids", None) or [])]
            a["_pairs"] = sorted(zip(old, new))
        elif isinstance(a.get("item_ids"), list):
            a["item_ids"] = sorted(str(x) for x in a["item_ids"])
        return (name, json.dumps(a, sort_keys=True, default=str))

    def accepts(self, proposal: Any, node, session, executor=None) -> tuple[bool, str]:
        """Executable acceptance: the live shop matches the intent's own end state, AND
        (golden-action check) every expected write was actually issued by the agent."""
        env = getattr(session, "env", None)
        if env is None:
            return False, "no_environment"
        want = getattr(node, "ground_truth", None)
        if want is None:
            return False, "node_has_no_ground_truth"
        try:
            target = json.loads(want.value)["db"]
        except Exception:
            return False, "unreadable_ground_truth"
        # AXIS-B GRADING (2026-08-17, measured: 72% of shifted failures were state-
        # unreachable after one stale irreversible write): when a shift has occurred,
        # success = the NEW intent's golden actions were all performed. Stale extra
        # writes are the recorded cost of staleness, not an automatic loss; a burned
        # write-once still fails because the golden action itself is impossible.
        # tau2's own ACTION reward component grades exactly this way.
        if getattr(self, "accept_actions_only", False):
            from ..executors.tau2_inproc import Tau2Session
            recipe = self.compile(dict(node.base), tuple(node.conditions))
            done = {self._action_key(n, k)
                    for n, k in getattr(session, "executed_calls", [])}
            missing = [a["name"] for a in (recipe.get("actions") or [])
                       if self._action_key(a["name"],
                                           Tau2Session._uncanon(a.get("arguments") or {}))
                       not in done]
            return ((True, "golden_actions_complete") if not missing else
                    (False, "golden_actions_missing:" + ",".join(missing[:3])))
        if env.get_db_hash() != target:
            return False, "db_state_mismatch"
        if self.golden_action_check:
            from ..executors.tau2_inproc import Tau2Session
            recipe = self.compile(dict(node.base), tuple(node.conditions))
            done = {self._action_key(n, k)
                    for n, k in getattr(session, "executed_calls", [])}
            missing = [a["name"] for a in (recipe.get("actions") or [])
                       if self._action_key(a["name"],
                                           Tau2Session._uncanon(a.get("arguments") or {}))
                       not in done]
            if missing:
                return False, "golden_actions_missing:" + ",".join(missing[:3])
        return True, "db_state_matches"

    def act_is_mutating(self, raw: str) -> bool:
        return any(w in (raw or "") for w in WRITE_TOOLS)


from .base import register  # noqa: E402
from ..executors.tau2_inproc import Tau2Executor  # noqa: E402

register("tau2_retail", Tau2RetailAdapter, Tau2Executor)
