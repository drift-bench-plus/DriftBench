"""tau2-bench airline adapter.

An airline task is a passenger asking an agent to change something about their travel, on a
shared flight/reservation database. Like retail (and unlike telecom), airline intents are
PARAMETER-shaped -- the golden actions carry the user's requirements as typed arguments
(which reservation, which flights, which cabin, how many bags, which payment method) -- so
conditions are argument slots and ground truth is tau2's own final-database-state hash.

Two airline-specific facts shape this adapter, both absent from retail:

* **Some write arguments are DERIVED, not free.** `book_reservation` raises unless the
  payment amounts sum to a total the environment computes (fares x passengers + $30 x
  passengers insurance + $50 x paid bag), and the free-bag allowance is a function of
  membership x cabin. Binding those numbers as conditions would make every sibling edit
  (a different cabin, one more passenger, no insurance) an automatically rejected write.
  So `payment_methods` conditions bind the METHOD IDS ONLY and `nonfree_baggages` is not
  a condition at all; `compile()` recomputes both from the database. The free choice is
  which methods in which order, never the arithmetic.
* **The environment enforces almost none of the policy.** Cancel eligibility, basic-economy
  immutability, payment composition, the five-passenger cap, baggage decrease -- all
  execute silently and hash as "distinct" ground truths. Strict raise-on-rejected-write
  therefore filters much less than it did on retail, and the airline policy gate
  (airline_policy_gate.py) is the load-bearing legitimacy check for these graphs.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

from ..ids import canonical_dumps
from ..models import Condition, GroundTruth, Seed

ARG = "arg:"

NOW = "2024-05-15T15:00:00"          # the environment's frozen clock (tools._get_datetime)


def _ensure_tau2() -> None:
    src = os.environ.get("TAU2_SRC")
    if src and src not in sys.path:
        sys.path.insert(0, src)


# Write tools: the ones whose arguments carry user requirements. send_certificate is a
# WRITE by tool type but appears in zero golden traces; it stays here so act_is_mutating
# and the gate see it, and simply never seeds anything.
WRITE_TOOLS = {
    "book_reservation", "cancel_reservation", "send_certificate",
    "update_reservation_baggages", "update_reservation_flights",
    "update_reservation_passengers",
}

# policy.md's free checked-bag matrix: membership -> cabin -> free bags per passenger.
FREE_BAGS = {
    "regular": {"basic_economy": 0, "economy": 1, "business": 2},
    "silver":  {"basic_economy": 1, "economy": 2, "business": 3},
    "gold":    {"basic_economy": 2, "economy": 3, "business": 4},
}

# Derived (environment-computed) argument slots, excluded from the condition set.
DERIVED_KEYS = {("book_reservation", "nonfree_baggages"),
                ("update_reservation_baggages", "nonfree_baggages")}


class Tau2AirlineAdapter:

    # A read-only, no-argument call the harness may substitute when an arm's own
    # rule forbids the action it chose (see LLMAgent._harness_action). WebShop's
    # `search[...]` is not a tau2 action, so before this existed the substitution
    # could only produce a parse error and burn the turn.
    agent_safe_action = "list_all_airports()"
    name = "tau2_airline"
    version = "1"
    executor_name = "tau2_inproc"

    ENV_KEY = "tau2_airline_desk"

    def __init__(self, config: dict | None = None) -> None:
        cfg = (config or {}).get("tau2_airline", {}) if config else {}
        base = os.environ.get("TAU2_DATA", "") + "/domains/airline/"
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

    def _reservation(self, rid) -> dict:
        return (self.db.get("reservations") or {}).get(str(rid)) or {}

    def _user(self, uid) -> dict:
        return (self.db.get("users") or {}).get(str(uid)) or {}

    def _allowance(self, membership: str, cabin: str) -> int:
        return FREE_BAGS.get(membership, FREE_BAGS["regular"]).get(cabin, 0)

    def _flight_price(self, flight_number: str, date: str, cabin: str) -> int | None:
        day = ((self.db.get("flights") or {}).get(str(flight_number)) or {}).get(
            "dates", {}).get(str(date)) or {}
        if day.get("status") != "available":
            return None
        return (day.get("prices") or {}).get(cabin)

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
                    if (a["name"], k) in DERIVED_KEYS:
                        continue              # environment arithmetic, not a requirement
                    if a["name"] == "book_reservation" and k == "payment_methods":
                        # the requirement is which methods in which order; amounts are
                        # recomputed by compile() from the database
                        v = [p.get("payment_id") for p in (v or [])]
                    conds.append((f"{ARG}{i}:{a['name']}:{k}", "=", canonical_dumps(v)))
            if not conds:
                continue
            instr = ((t.get("user_scenario") or {}).get("instructions") or {})
            seeds.append(Seed(
                record_id=str(t.get("id")),
                env_key=self.ENV_KEY,
                base={"kind": "airline_request",
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

    # ------------------------------------------------------------------ derived args
    def _default_payment(self, uid) -> str | None:
        """The natural unstated payment method: the user's first credit card, else
        their first non-certificate method (certificates raise on updates)."""
        pm = (self._user(uid).get("payment_methods") or {})
        cards = [p for p in pm if (pm[p] or {}).get("source") == "credit_card"]
        if cards:
            return sorted(cards)[0]
        rest = [p for p in pm if (pm[p] or {}).get("source") != "certificate"]
        return sorted(rest)[0] if rest else None

    def _book_finalize(self, args: dict) -> dict:
        """Recompute the derived arguments of one book_reservation call from the db.

        Never raises: on missing data (an unavailable flight, an unknown user) it emits
        arguments the environment will reject, so the candidate is dropped by strict
        execution instead of crashing compile().
        """
        user = self._user(args.get("user_id"))
        passengers = args.get("passengers") or []
        cabin = args.get("cabin") or "economy"      # unstated cabin: the default fare
        args = dict(args, cabin=cabin,
                    insurance=args.get("insurance") or "no")
        total_bags = int(args.get("total_baggages") or 0)
        allowance = self._allowance(user.get("membership", "regular"), cabin)
        nonfree = max(0, total_bags - allowance * len(passengers))

        total = 0
        priced = True
        for f in args.get("flights") or []:
            p = self._flight_price(f.get("flight_number"), f.get("date"), cabin)
            if p is None:
                priced = False
                break
            total += p * len(passengers)
        if args.get("insurance") == "yes":
            total += 30 * len(passengers)
        total += 50 * nonfree

        ids = args.get("payment_methods") or []
        if ids and isinstance(ids[0], dict):          # already full payment objects
            ids = [p.get("payment_id") for p in ids]
        methods = []
        if priced:
            remaining = total
            pm = user.get("payment_methods") or {}
            for j, pid in enumerate(ids):
                src = (pm.get(str(pid)) or {})
                if j == len(ids) - 1:
                    amt = remaining                    # last method takes the remainder
                elif src.get("source") in ("gift_card", "certificate"):
                    amt = min(int(src.get("amount") or 0), remaining)
                else:
                    amt = remaining                    # a mid-list card absorbs the rest
                methods.append({"payment_id": pid, "amount": amt})
                remaining -= amt
            methods = [m for m in methods if m["amount"] != 0] or methods[-1:]
        else:
            # unpriceable itinerary: emit a sum the tool must reject
            methods = [{"payment_id": pid, "amount": 0} for pid in ids]

        out = dict(args)
        out["payment_methods"] = methods
        out["nonfree_baggages"] = nonfree
        return out

    def _baggage_finalize(self, args: dict, cabin_override: str | None) -> dict:
        """nonfree = max(0, total - allowance x passengers), from the reservation."""
        res = self._reservation(args.get("reservation_id"))
        user = self._user(res.get("user_id"))
        cabin = cabin_override or res.get("cabin")
        n_pax = len(res.get("passengers") or []) or 1
        total = int(args.get("total_baggages") or 0)
        nonfree = max(0, total - self._allowance(user.get("membership", "regular"), cabin) * n_pax)
        out = dict(args)
        out["nonfree_baggages"] = nonfree
        return out

    # ------------------------------------------------------------------ recipes
    def compile(self, base: dict, conditions: tuple[Condition, ...]) -> Any:
        """Rebuild the write-action sequence from its argument slots, recomputing the
        environment-derived arguments (see module docstring)."""
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
        actions = []
        cabin_by_res: dict[str, str] = {}     # a flights-update changes the allowance
        for i in sorted(by_action):
            a = by_action[i]
            args = a["arguments"]
            # DEFAULT COMPLETIONS for unstated arguments (2026-08-18, measured): a
            # withheld/vague reading drops a condition, and unlike retail's tools the
            # airline tools raise on a missing required argument -- which killed the
            # entire mark-family admission (readings 0/24-8/24). The literal reading of
            # a request that does not state a slot is the slot's natural default: an
            # update keeps the reservation's current flights/cabin, a booking is
            # economy without insurance unless said otherwise. Admission still enforces
            # consequence (a default-completed reading identical to the truth is
            # rejected as consequence-free), so this changes executability, never the
            # contract. Stored recipes/hashes are untouched (they carry full arguments).
            if a["name"] == "update_reservation_flights":
                res = self._reservation(args.get("reservation_id"))
                if res:
                    if not args.get("flights"):
                        args["flights"] = [{"flight_number": f.get("flight_number"),
                                            "date": f.get("date")}
                                           for f in res.get("flights") or []]
                    if not args.get("cabin"):
                        args["cabin"] = res.get("cabin")
                    if not args.get("payment_id"):
                        args["payment_id"] = self._default_payment(res.get("user_id"))
                if args.get("reservation_id") is not None and args.get("cabin"):
                    cabin_by_res[str(args["reservation_id"])] = args["cabin"]
            elif a["name"] == "update_reservation_baggages":
                res = self._reservation(args.get("reservation_id"))
                if res:
                    if args.get("total_baggages") is None:
                        args["total_baggages"] = res.get("total_baggages")
                    if not args.get("payment_id"):
                        args["payment_id"] = self._default_payment(res.get("user_id"))
            elif a["name"] == "update_reservation_passengers":
                res = self._reservation(args.get("reservation_id"))
                if res and not args.get("passengers"):
                    args["passengers"] = [dict(p) for p in res.get("passengers") or []]
            if a["name"] == "book_reservation":
                args = self._book_finalize(args)
            elif a["name"] == "update_reservation_baggages":
                args = self._baggage_finalize(
                    args, cabin_by_res.get(str(args.get("reservation_id"))))
            actions.append({"name": a["name"], "arguments": args})
        # `also:` conditions: one condition = one whole ADDED request (a refinement),
        # arguments mined complete and executable at witness time -- appended verbatim.
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
        """Ground truth = the flight database after the writes, by tau2's own hash.

        Strict: any rejected write raises. For SEEDS that is a validation failure; for
        BRANCH CANDIDATES the engine drops the candidate -- embedding the error in the
        hash would let an unachievable branch masquerade as a moved ground truth (the
        retail v1 relaxation defect)."""
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
        return payload["db"] != session.env.get_db_hash()   # must actually move the db

    def is_valid_intent(self, base: dict, conditions: tuple[Condition, ...]) -> bool:
        """Structural validity AND policy validity.

        Unlike retail, the airline environment enforces almost none of its policy, so a
        policy-violating branch executes fine and hashes as a distinct ground truth --
        while a policy-compliant agent must refuse it. Excluding those candidates HERE
        (the engine filters on is_valid_intent before execution) lets the branching
        quota fill with legitimate siblings instead of leaving post-prune holes; the
        airline policy gate then re-audits the built graphs as verification."""
        if not [c for c in conditions if str(c[0]).startswith(ARG)]:
            return False
        tags, _reason_dep = self.policy_violations(self.compile(base, conditions))
        return not tags

    def gt_extensional(self, recipe: Any) -> bool:
        return False

    def gt_equal(self, a: GroundTruth, b: GroundTruth) -> bool:
        return a.kind == b.kind and a.value == b.value

    # ------------------------------------------------------------------ mining helpers
    def _cond_value(self, conditions, suffix: str):
        for slot, _op, value in conditions:
            if str(slot).endswith(suffix):
                try:
                    return json.loads(value)
                except (TypeError, json.JSONDecodeError):
                    return value
        return None

    def _seed_user(self, conditions) -> str | None:
        uid = self._cond_value(conditions, ":user_id")
        if uid:
            return str(uid)
        rid = self._cond_value(conditions, ":reservation_id")
        if rid:
            return self._reservation(rid).get("user_id")
        return None

    def _touched_reservations(self, conditions) -> set[str]:
        out = set()
        for slot, _op, value in conditions:
            if str(slot).endswith(":reservation_id"):
                try:
                    out.add(str(json.loads(value)))
                except (TypeError, json.JSONDecodeError):
                    out.add(str(value))
        return out

    def _flight_day_status(self, flight_number: str, date: str) -> str:
        day = ((self.db.get("flights") or {}).get(str(flight_number)) or {}).get(
            "dates", {}).get(str(date)) or {}
        return day.get("status", "")

    def _cancel_eligible_reason_free(self, res: dict) -> bool:
        """Eligible to cancel WITHOUT consulting the (hash-invisible) stated reason:
        booked within 24h of the frozen clock, business cabin, or an airline-cancelled
        segment. Insurance+covered-reason cancels are irreducibly reason-dependent and
        are never mined here. A flown segment makes the reservation transfer-only
        (gate predicate A4), so those are excluded regardless of eligibility."""
        if res.get("status") == "cancelled":
            return False
        for f in res.get("flights") or []:
            if self._segment_flown(f.get("flight_number"), f.get("date")):
                return False
        created = str(res.get("created_at") or "")
        if created >= "2024-05-14T15:00:00":            # within 24h of NOW
            return True
        if res.get("cabin") == "business":
            return True
        for f in res.get("flights") or []:
            if self._flight_day_status(f.get("flight_number"), f.get("date")) == "cancelled":
                return True
        return False

    def _reservation_open(self, res: dict) -> bool:
        """No segment flown/landed/cancelled and not itself cancelled -- the reservations
        a modify can still legitimately touch."""
        if res.get("status") == "cancelled":
            return False
        for f in res.get("flights") or []:
            if str(f.get("date", "")) < "2024-05-15":
                return False
            if self._flight_day_status(f.get("flight_number"), f.get("date")) in (
                    "flying", "landed", "cancelled"):
                return False
        return True

    # ------------------------------------------------------------------ policy
    def policy_violations(self, recipe: Any) -> tuple[list[str], bool]:
        """Mechanical predicates over one recipe's actions plus the database -- the
        prose-only rules the environment does not enforce (see airline_policy_gate.py
        for the predicate glossary A0-A11). Returns (tags, reason_dependent_cancel).

        State is tracked INTRA-RECIPE: a golden trace may upgrade a cabin and then
        cancel (task 7), or change cabin before the baggage arithmetic (task 17), so
        each action is judged against the reservation as the previous actions left it,
        never against the pristine database. Reservations a book creates mid-recipe are
        not tracked (no golden trace touches one afterwards)."""
        out: list[str] = []
        reason_dep = False
        touched_users: set[str] = set()
        state: dict[str, dict] = {}

        def facts(rid: str) -> dict | None:
            if rid in state:
                return state[rid]
            res = self._reservation(rid)
            if not res:
                return None
            flights = [(str(f.get("flight_number")), str(f.get("date")))
                       for f in res.get("flights") or []]
            state[rid] = {
                "user_id": res.get("user_id"),
                "cabin": res.get("cabin"),
                "origin": res.get("origin"),
                "destination": res.get("destination"),
                "flight_type": res.get("flight_type"),
                "created_at": str(res.get("created_at") or ""),
                "insurance": res.get("insurance"),
                "status": res.get("status"),
                "total_baggages": int(res.get("total_baggages") or 0),
                "nonfree_baggages": int(res.get("nonfree_baggages") or 0),
                "flights": flights,
                "any_flown": any(self._segment_flown(fn, d) for fn, d in flights),
                "any_cancelled_segment": any(
                    self._flight_day_status(fn, d) == "cancelled" for fn, d in flights),
                "n_passengers": len(res.get("passengers") or []),
                "passengers": sorted(
                    (str(p.get("first_name")), str(p.get("last_name")), str(p.get("dob")))
                    for p in res.get("passengers") or []),
            }
            return state[rid]

        for a in recipe.get("actions") or []:
            name, args = a["name"], a.get("arguments") or {}
            rid = str(args.get("reservation_id")) if args.get("reservation_id") else None
            f = facts(rid) if rid else None
            if f:
                touched_users.add(f["user_id"])

            if name == "send_certificate":
                out.append("A9:compensation")
                if args.get("user_id"):
                    touched_users.add(str(args["user_id"]))

            elif name == "cancel_reservation":
                if not f:
                    continue
                if f["status"] == "cancelled":
                    out.append(f"A11:recancel:{rid}")
                elif f["any_flown"]:
                    out.append(f"A4:cancel_flown_segment:{rid}")
                else:
                    reason_free = (f["created_at"] >= "2024-05-14T15:00:00"
                                   or f["cabin"] == "business"
                                   or f["any_cancelled_segment"])
                    if not reason_free:
                        if f["insurance"] == "yes":
                            reason_dep = True
                        else:
                            out.append(f"A4:cancel_ineligible:{rid}")
                f["status"] = "cancelled"

            elif name == "update_reservation_flights":
                if not f:
                    continue
                new_set = [(str((x or {}).get("flight_number")), str((x or {}).get("date")))
                           for x in args.get("flights") or []]
                cabin = args.get("cabin")
                same_flights = sorted(new_set) == sorted(f["flights"])
                if f["status"] == "cancelled":
                    out.append(f"A11:update_cancelled:{rid}")
                if f["cabin"] == "basic_economy" and not same_flights:
                    out.append(f"A1:basic_economy_flight_change:{rid}")
                if f["any_flown"] and cabin != f["cabin"]:
                    out.append(f"A3:cabin_after_flown:{rid}")
                if same_flights and cabin == f["cabin"]:
                    out.append(f"A11:noop_flights:{rid}")
                err = self._chain_violation(args.get("flights"), f["origin"],
                                            f["destination"], f["flight_type"])
                if err:
                    tag = "A2" if err.split(":")[0] in (
                        "origin", "destination", "round_trip_end",
                        "turnaround_missing") else "A10"
                    out.append(f"{tag}:{err}:{rid}")
                f["cabin"] = cabin
                f["flights"] = new_set
                f["any_flown"] = any(self._segment_flown(fn, d) for fn, d in new_set)
                f["any_cancelled_segment"] = any(
                    self._flight_day_status(fn, d) == "cancelled" for fn, d in new_set)

            elif name == "update_reservation_baggages":
                if not f:
                    continue
                total = int(args.get("total_baggages") or 0)
                nonfree = int(args.get("nonfree_baggages") or 0)
                if f["status"] == "cancelled":
                    out.append(f"A11:update_cancelled:{rid}")
                if total < f["total_baggages"]:
                    out.append(f"A7:baggage_decrease:{rid}")
                membership = self._user(f["user_id"]).get("membership", "regular")
                allowance = self._allowance(membership, f["cabin"])
                if nonfree != max(0, total - allowance * f["n_passengers"]):
                    out.append(f"A8:baggage_arithmetic:{rid}")
                if total == f["total_baggages"] and nonfree == f["nonfree_baggages"]:
                    out.append(f"A11:noop_baggages:{rid}")
                f["total_baggages"] = total
                f["nonfree_baggages"] = nonfree

            elif name == "update_reservation_passengers":
                if not f:
                    continue
                new_pax = sorted((str((p or {}).get("first_name")),
                                  str((p or {}).get("last_name")),
                                  str((p or {}).get("dob")))
                                 for p in args.get("passengers") or [])
                if new_pax == f["passengers"]:
                    out.append(f"A11:noop_passengers:{rid}")
                f["passengers"] = new_pax

            elif name == "book_reservation":
                uid = str(args.get("user_id"))
                touched_users.add(uid)
                pax = args.get("passengers") or []
                if len(pax) > 5:
                    out.append(f"A5:passenger_cap:{len(pax)}")
                pm_of = (self._user(uid).get("payment_methods") or {})
                srcs: dict[str, int] = {}
                for pm in args.get("payment_methods") or []:
                    pid = str((pm or {}).get("payment_id"))
                    src = (pm_of.get(pid) or {}).get("source", "unknown")
                    srcs[src] = srcs.get(src, 0) + 1
                if srcs.get("certificate", 0) > 1:
                    out.append("A6:certificates")
                if srcs.get("credit_card", 0) > 1:
                    out.append("A6:credit_cards")
                if srcs.get("gift_card", 0) > 3:
                    out.append("A6:gift_cards")
                err = self._chain_violation(args.get("flights"), args.get("origin"),
                                            args.get("destination"),
                                            args.get("flight_type"))
                if err:
                    out.append(f"A10:{err}")

        if len(touched_users) > 1:
            out.append("A0:mixed_users:" + ",".join(sorted(touched_users)))
        return out, reason_dep

    def _segment_flown(self, flight_number, date) -> bool:
        if str(date) < "2024-05-15":
            return True
        return self._flight_day_status(flight_number, date) in ("flying", "landed")

    def _chain_violation(self, flights, origin, destination, flight_type) -> str | None:
        segs = []
        for f in flights or []:
            fl = (self.db.get("flights") or {}).get(str((f or {}).get("flight_number")))
            if fl is None:
                segs.append(("?", "?", str((f or {}).get("date"))))
            else:
                segs.append((fl.get("origin"), fl.get("destination"),
                             str((f or {}).get("date"))))
        if not segs:
            return "empty_flight_list"
        for (o1, d1, t1), (o2, d2, t2) in zip(segs, segs[1:]):
            if d1 != o2:
                return f"broken_chain:{d1}->{o2}"
            if t2 < t1:
                return f"date_regression:{t1}->{t2}"
        if any(t < "2024-05-15" for _o, _d, t in segs):
            return "past_date"
        if segs[0][0] != origin:
            return f"origin:{segs[0][0]}!={origin}"
        if flight_type == "round_trip":
            if segs[-1][1] != origin:
                return f"round_trip_end:{segs[-1][1]}!={origin}"
            if not any(d == destination for _o, d, _t in segs):
                return f"turnaround_missing:{destination}"
        else:
            if segs[-1][1] != destination:
                return f"destination:{segs[-1][1]}!={destination}"
        return None

    # ------------------------------------------------------------------ refinement
    def witness(self, base: dict, conditions: tuple[Condition, ...], session) -> list[dict]:
        """Refinement candidates: further requests this SAME passenger could really make,
        mined from their other reservations. Each witness carries one `also:` slot whose
        value is a complete, executable extra request: adding a checked bag to another
        open reservation (paid with a real non-certificate method -- certificates raise
        on updates), or cancelling another reservation that is eligible WITHOUT a stated
        reason (see _cancel_eligible_reason_free). Execution then filters whatever the
        environment itself refuses."""
        uid = self._seed_user(conditions)
        if not uid:
            return []
        user = self._user(uid)
        touched = self._touched_reservations(conditions)
        pay_ids = [pid for pid, pm in (user.get("payment_methods") or {}).items()
                   if pm.get("source") == "credit_card"
                   or (pm.get("source") == "gift_card" and (pm.get("amount") or 0) >= 50)]
        out, n = [], 0
        for rid in user.get("reservations") or []:
            if str(rid) in touched:
                continue
            res = self._reservation(rid)
            if not res:
                continue
            if self._cancel_eligible_reason_free(res):
                out.append({f"also:{n}:cancel_reservation": canonical_dumps(
                    {"reservation_id": str(rid)})})
                n += 1
            if self._reservation_open(res) and pay_ids:
                total = int(res.get("total_baggages") or 0) + 1
                cabin = res.get("cabin")
                n_pax = len(res.get("passengers") or []) or 1
                allowance = self._allowance(user.get("membership", "regular"), cabin)
                out.append({f"also:{n}:update_reservation_baggages": canonical_dumps(
                    {"reservation_id": str(rid), "total_baggages": total,
                     "nonfree_baggages": max(0, total - allowance * n_pax),
                     "payment_id": pay_ids[0]})})
                n += 1
            if n >= 6:
                break
        return out

    # ------------------------------------------------------------------ extra candidates
    def extra_candidates(self, seed):
        """Candidates the condition-blind enumerators cannot express.

        (a) Whole-request relaxation on multi-write seeds (the retail rule: dropping a
            single argument produces a rejected write, dropping a WHOLE request is what a
            passenger abandoning one errand actually does).
        (b) Condition-aware flight substitutions: swap one segment for another real
            flight on the same route and date (or the same flight one day later), from
            the flight database. The generic domains() hook cannot see the current
            value, and a global flight pool would mostly manufacture policy-invalid
            route changes, so these are mined here where the conditions are visible.
        (c) A different single payment method for a book, from the same user's profile.
        """
        from ..engine import Candidate, Provenance
        from ..models import Operator, sort_conditions
        out = []
        out += self._relaxation_candidates(seed)

        for slot, op, value in seed.conditions:
            body = str(slot)[len(ARG):] if str(slot).startswith(ARG) else ""
            if not body:
                continue
            _idx, tool, key = body.split(":", 2)
            if key == "flights" and tool in ("book_reservation", "update_reservation_flights"):
                try:
                    flights = json.loads(value)
                except (TypeError, json.JSONDecodeError):
                    continue
                if tool == "update_reservation_flights":
                    # basic-economy flight sets are immutable (gate predicate A1):
                    # never mine a swap the policy forbids
                    rid = self._cond_value(seed.conditions,
                                           f"{_idx}:{tool}:reservation_id")
                    if self._reservation(rid).get("cabin") == "basic_economy":
                        continue
                cabin = self._cond_value(seed.conditions, f"{_idx}:{tool}:cabin") or "economy"
                n_swaps = 0
                for j, f in enumerate(flights or []):
                    for alt in self._alt_segments(f, cabin):
                        new = [dict(x) for x in flights]
                        new[j] = alt
                        nv = canonical_dumps(new)
                        if nv == value:
                            continue
                        conds = sort_conditions(
                            [(s, o, nv) if (s, o) == (slot, op) else (s, o, v)
                             for s, o, v in seed.conditions])
                        out.append(Candidate(
                            conditions=conds, base=seed.base,
                            operator=Operator.SUBSTITUTION,
                            delta={"changed": [[slot, op, value, nv]]},
                            provenance=Provenance.SYNTHETIC))
                        n_swaps += 1
                        if n_swaps >= 8:
                            break
                    if n_swaps >= 8:
                        break
            elif key == "payment_methods" and tool == "book_reservation":
                try:
                    ids = json.loads(value)
                except (TypeError, json.JSONDecodeError):
                    continue
                uid = self._cond_value(seed.conditions, f"{_idx}:{tool}:user_id")
                pm = (self._user(uid).get("payment_methods") or {})
                added = 0
                for pid, m in pm.items():
                    if pid in (ids or []) or m.get("source") == "certificate":
                        continue
                    nv = canonical_dumps([pid])
                    conds = sort_conditions(
                        [(s, o, nv) if (s, o) == (slot, op) else (s, o, v)
                         for s, o, v in seed.conditions])
                    out.append(Candidate(
                        conditions=conds, base=seed.base,
                        operator=Operator.SUBSTITUTION,
                        delta={"changed": [[slot, op, value, nv]]},
                        provenance=Provenance.SYNTHETIC))
                    added += 1
                    if added >= 2:
                        break
        return out

    def _alt_segments(self, f: dict, cabin: str) -> list[dict]:
        """Real alternatives for one segment: same route same date, or same flight the
        next day -- available with a published price for the cabin."""
        fn, date = str(f.get("flight_number")), str(f.get("date"))
        me = (self.db.get("flights") or {}).get(fn) or {}
        alts = []
        for other_fn, other in (self.db.get("flights") or {}).items():
            if other_fn == fn:
                continue
            if other.get("origin") != me.get("origin"):
                continue
            if other.get("destination") != me.get("destination"):
                continue
            if self._flight_price(other_fn, date, cabin) is not None:
                alts.append({"flight_number": other_fn, "date": date})
            if len(alts) >= 3:
                break
        if date.startswith("2024-05-"):
            try:
                nxt = f"2024-05-{int(date[-2:]) + 1:02d}"
                if self._flight_price(fn, nxt, cabin) is not None:
                    alts.append({"flight_number": fn, "date": nxt})
            except ValueError:
                pass
        return alts

    def _relaxation_candidates(self, seed):
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

    # ------------------------------------------------------------------ substitution pools
    def domains(self, slot: str, base: dict, session) -> list[Any]:
        """Condition-blind pools for scalar slots. Flights, payment methods and
        reservation ids need the current conditions and live in extra_candidates();
        offering them globally here would mostly manufacture branches that execute
        fine and violate policy (a route change, another user's reservation) -- the
        exact class the gate exists to catch, so we decline to mass-produce it."""
        if not str(slot).startswith(ARG):
            return []
        key = str(slot).rsplit(":", 1)[-1]
        if key == "cabin":
            return [canonical_dumps(c) for c in ("basic_economy", "economy", "business")]
        if key == "insurance":
            return [canonical_dumps(c) for c in ("yes", "no")]
        if key == "total_baggages":
            return [canonical_dumps(n) for n in range(5)]
        return []

    # ------------------------------------------------------------------ env
    def env_spec(self, env_key: str) -> dict:
        return {"domain": "airline", "solo_mode": False, "env_id": env_key,
                "env_key": env_key, "initialization": {}}

    def is_mutating(self, recipe: Any) -> bool:
        return True

    # ------------------------------------------------------------------ episode
    def context_facts(self, node) -> dict:
        """Facts this passenger plainly knows that are NOT requirements of the intent.

        tau2's airline policy demands the caller's user id (four separate times) and a
        reason for every cancellation, but neither is an argument of cancel/update, so
        neither is a condition -- and a slot-only user simulator therefore cannot answer,
        which deadlocked the agent in 10 of 14 dead-graph failures (measured 2026-08-19).
        tau2's own tasks supply exactly these in `known_info`; this restores them.
        Answering costs no reveal and earns no recovery credit: nothing was hidden.

        The cancellation reason is chosen to match the reservation's REAL eligibility,
        never to manufacture it: a booking that is only cancellable because it carries
        insurance gets a covered reason, everything else a change of plan.
        """
        conds = tuple(node.conditions)
        uid = self._seed_user(conds)
        facts: dict[str, str] = {}
        if uid:
            facts["your user id|user id|account|who you are|identify"] = str(uid)
        recipe = self.compile(dict(node.base), conds)
        cancels = [a for a in (recipe.get("actions") or [])
                   if a["name"] == "cancel_reservation"]
        if cancels:
            reasons = []
            for a in cancels:
                rid = str((a.get("arguments") or {}).get("reservation_id"))
                res = self._reservation(rid)
                if self._cancel_eligible_reason_free(res):
                    reasons.append(f"{rid}: a change of plan")
                elif res.get("insurance") == "yes":
                    reasons.append(f"{rid}: illness in the family, which the travel "
                                   f"insurance covers")
                else:
                    reasons.append(f"{rid}: a change of plan")
            facts["reason for cancel|why you|cancellation reason|reason"] = "; ".join(reasons)
        return facts

    def slot_phrase(self, slot: str) -> str:
        if str(slot).startswith("also:"):
            _t, _i, tool = str(slot).split(":", 2)
            return "an additional request (" + tool.replace("_", " ") + ")"
        body = str(slot)[len(ARG):] if str(slot).startswith(ARG) else str(slot)
        try:
            _idx, tool, key = body.split(":", 2)
        except ValueError:
            return body
        label = {"reservation_id": "which reservation",
                 "user_id": "the account user id",
                 "origin": "the trip origin airport",
                 "destination": "the trip destination airport",
                 "flight_type": "one-way or round trip",
                 "cabin": "the cabin class",
                 "flights": "which flights to take",
                 "passengers": "the passenger details",
                 "payment_methods": "which payment method(s)",
                 "payment_id": "which payment method",
                 "total_baggages": "how many checked bags",
                 "insurance": "whether to add travel insurance",
                 "amount": "the certificate amount"}.get(key, key.replace("_", " "))
        verb = {"book_reservation": "the new booking",
                "cancel_reservation": "the cancellation",
                "update_reservation_flights": "the flight change",
                "update_reservation_baggages": "the baggage update",
                "update_reservation_passengers": "the passenger update",
                "send_certificate": "the compensation"}.get(tool, tool.replace("_", " "))
        return f"{label} (for {verb})"

    def spoken_value(self, slot: str, value) -> str:
        """The words for the VALUE itself: a passenger reads ids and flight numbers off
        their confirmation email, so they are spoken verbatim; structured values are
        flattened to natural phrases."""
        v = value
        if isinstance(v, str):
            try:
                v = json.loads(v)
            except (TypeError, json.JSONDecodeError):
                pass
        if isinstance(v, list):
            parts = []
            for x in v:
                if isinstance(x, dict) and "flight_number" in x:
                    parts.append(f"flight {x.get('flight_number')} on {x.get('date')}")
                elif isinstance(x, dict) and "first_name" in x:
                    parts.append(f"{x.get('first_name')} {x.get('last_name')}"
                                 + (f" (born {x['dob']})" if x.get("dob") else ""))
                else:
                    parts.append(str(x))
            return ", ".join(parts)
        if isinstance(v, dict):
            return ", ".join(f"{k} {x}" for k, x in v.items())
        return str(v)

    def describe_intent(self, conditions) -> str:
        return "\n".join(f"  - {self.slot_phrase(s)}: {v}" for s, _o, v in sorted(conditions))

    generation_voice = "a plain airline passenger's voice"
    generation_domain_word = "air travel"

    # Same closed-vocabulary situation as retail: a condition is an ARGUMENT of a specific
    # write action, so an "extra irrelevant detail" has no slot to compile into. Admitted
    # on string checks and labelled non-executable, then collision-audited.
    extras_may_be_pragmatic = True

    def slot_supports_subjective(self, slot: str) -> bool:
        """Which flights to take -- and which cabin -- are judgements a passenger can
        state vaguely ('the earliest one', 'whatever is cheapest', 'something more
        comfortable than economy'): the ordering is over the flight catalogue and the
        cabin ladder, not over the slot's own value."""
        key = str(slot).rsplit(":", 1)[-1]
        return key in ("flights", "cabin")

    # The stock strategy cards for these three types use flight-booking examples; on
    # airline the example IS the domain, and the generation prompt's "NEVER copy entities
    # from the example" rule would forbid the domain's own entities. Same flaw semantics,
    # out-of-domain surface (v2 definitions preserved).
    card_overrides = {
        "factual_error": (
            "One requirement is stated with a WRONG value in place of the true one; "
            "everything else stays accurate.",
            "Original: 'Return the blue kettle from my order' -> Rewritten: 'Return "
            "the red kettle from my order'",
        ),
        "insufficient_information": (
            "One or more required details are simply missing; what remains is stated "
            "accurately.",
            "Original: 'Book a table for four at Nonna's at 7pm' -> Rewritten: 'Book "
            "a table at Nonna's'",
        ),
        "irrelevant_information": (
            "Extra details that change nothing about what must be done are woven in; "
            "every requirement stays accurately stated.",
            "Original: 'Cancel my gym membership' -> Rewritten: 'My cousin talked me "
            "into that gym years ago and I hardly ever went -- cancel my membership.'",
        ),
        # mark-family cards, replaced for the same reason as above (the stock examples
        # are air-travel ones) and re-worded to name the operative structure, since the
        # stock wording taught "insert a punny noun" rather than "replace one value".
        "lexical_ambiguity": (
            "One requirement is named with a word that has TWO established senses in "
            "this setting, so the same sentence describes two different jobs; every "
            "other requirement is stated exactly, and the true value of the named "
            "requirement never appears.",
            "Original: 'Put the deposit on my Visa ending 4471' -> Rewritten: 'Put the "
            "deposit on my card' -- the guest has a credit card and a stored gift card "
            "on file, and 'card' means either one.",
        ),
        "syntactic_ambiguity": (
            "One modifier is placed where it could govern either of two things the "
            "speaker has already asked for, so it is unclear which of them it belongs "
            "to. Every value is still stated exactly; what is uncertain is which "
            "request the modifier applies to.",
            "Original: 'Cancel Tuesday's cleaning and reschedule Friday's; bill both to "
            "the office account' -> Rewritten: 'Cancel Tuesday's cleaning and reschedule "
            "Friday's on the office account' (is the office account paying for the "
            "reschedule only, or for both?)",
        ),
        "referential_ambiguity": (
            "Exactly one requirement is left as a bare referring term -- 'that one', "
            "'the usual one' -- pointing at something the speaker assumes is already "
            "shared, where more than one thing could be meant. Nothing is deleted from "
            "the request; only that one value is never given, and it cannot be worked "
            "out from the rest of the text.",
            "Original: 'Renew my library card and put the $12 late fee on my Visa "
            "ending 4402' -> Rewritten: 'Renew my library card and put the $12 late fee "
            "on the usual one.'",
        ),
        "insufficient_information": (
            "One or more required details are simply missing -- not hinted at, not "
            "half-given, not pointed at -- while everything that remains is stated in "
            "full and accurately.",
            "Original: 'Move booking 4821 to 7pm for six people and booking 4902 to 8pm "
            "for four' -> Rewritten: 'Move booking 4821 to 7pm and booking 4902 to 8pm.'",
        ),
    }

    gen_hints = {
        "vagueness_subjectivity": (
            "- state which flight (or cabin) you want only through a subjective\n"
            "  judgement -- 'the earliest flight that day', 'whatever is cheapest',\n"
            "  'something nicer than economy' -- never naming the flight number or\n"
            "  class. State every OTHER requirement exactly."),
        "syntactic_ambiguity": (
            "- build the ambiguity from ATTACHMENT: one modifier (a date, a city, a\n"
            "  person) placed so it can belong to either of two requirements -- e.g. a\n"
            "  date that could be the flight's date or the day the change should\n"
            "  happen, or a city that could name either leg of the trip. Both readings\n"
            "  must be actions the airline could really take."),
        "lexical_ambiguity": (
            "- hinge on ONE word with two meanings in air travel: 'the first flight'\n"
            "  (earliest that day, or the first leg), 'change my seat class upward'\n"
            "  vs a named cabin, 'bags' (the total carried, or the extra ones). The\n"
            "  rest of the request stays exact."),
        "referential_ambiguity": (
            "- point at the reservation or flight only with 'it', 'that trip', 'my\n"
            "  earlier booking', 'the return one' -- where the account has several\n"
            "  that could match -- and never say the id. Everything else stays exact."),
        "irrelevant_information": (
            "- the extra detail must be something the airline is NOT being asked to do\n"
            "  anything about: a memory, a feeling, a companion's unrelated plans, why\n"
            "  the trip matters. NEVER phrase it as 'make sure', 'I want', 'I'd like',\n"
            "  'can you also', or any wish about the flight, seat, food, or service --\n"
            "  those read as requests and fail the collision audit."),

        # The four mark-family hints below were derived 2026-08-18 by reading the ACTUAL
        # rejected generations of the first two runs (3-5 admissions each of 24), then
        # adversarially verified against admit(), the v2 definitions and db.json. Each
        # points generation at the airline slots that have the shape retail's winners had,
        # and forbids the specific failure mode that dominated the rejections.
        "lexical_ambiguity": (
            "- your flaw REPLACES one requirement: refer to exactly ONE requirement only\n"
            "  by an everyday word with TWO established senses in air travel, so two\n"
            "  different bookings both answer the sentence. Never write that\n"
            "  requirement's true value anywhere; the ambiguous word must be its ONLY\n"
            "  mention. Every OTHER requirement stays stated exactly.\n"
            "- FIRST CHOICE, the luggage, whenever the request names a number of checked\n"
            "  bags: 'bag' covers what you check in and what you carry on, so 'I'll have\n"
            "  one bag with me', 'just a small bag for the trip' leave the CHECKED count\n"
            "  open. Never write the true number and never say 'checked' or 'carry-on'.\n"
            "- SECOND CHOICE, the payment, when a gift card pays: 'my card', 'my usual\n"
            "  card' name more than one thing the passenger holds. Never the payment id.\n"
            "- LAST RESORT, the flights: 'the first flight that day' means the earliest\n"
            "  departure or the first leg. Then name NO flight number for that\n"
            "  reservation, and both readings must still connect end to end.\n"
            "- do NOT hinge on the fare class: this airline treats basic economy as a\n"
            "  class completely distinct from economy, so 'coach'/'the main cabin'\n"
            "  simply name economy and 'upper class' names a fare it does not sell.\n"
            "- never bolt a vague clause onto an otherwise complete request ('...make\n"
            "  sure I get the right fare'). That hides nothing and is the single most\n"
            "  common way this flaw fails.\n"
            "- vary the ambiguous word between attempts."),
        "syntactic_ambiguity": (
            "- your flaw: place ONE modifier where it could govern EITHER OF TWO THINGS\n"
            "  YOU HAVE ALREADY ASKED FOR, so which one it belongs to is unclear. Both\n"
            "  hosts must be requirements from the list above; never invent a third\n"
            "  thing for the modifier to describe.\n"
            "- the shape that works is TWO REQUESTS sharing one detail: '<move a\n"
            "  reservation to the listed flights> and <add N checked bags>, with <one\n"
            "  card>' (does the card pay for both, or only the bags?); 'cancel <code A>\n"
            "  and <code B> <when-phrase>'; 'move <code A> and <code B> onto the listed\n"
            "  flights, in <a cabin>'. Exactly TWO hosts, never three.\n"
            "- the doubt may fall ONLY on which card pays, which cabin, or whether one\n"
            "  of two cancellations was meant -- NEVER on which flights, who travels,\n"
            "  a reservation code, the account id, or the cities: the other reading of\n"
            "  those is one nobody would take.\n"
            "- NEVER end with an invented purpose phrase ('for the flight to that\n"
            "  city', 'processed on that date'): it describes nothing on the list, so\n"
            "  no stated requirement is left in doubt and the sentence is merely padded.\n"
            "- never state anything about the booking as it stands in order to create\n"
            "  the doubt; it must come from word order alone.\n"
            "- if there is only ONE request, the single workable doubt is the fare word\n"
            "  ('move me onto the basic economy <FLIGHT> and <FLIGHT> services on\n"
            "  <date>'). Never with flight numbers, bag counts or passenger names.\n"
            "- keep the two hosts apart, state every value exactly once, add nothing."),
        "referential_ambiguity": (
            "- NEVER leave the reservation's confirmation code vague on a CHANGE\n"
            "  request: that makes the request impossible to act on rather than\n"
            "  ambiguous. Say the code out loud.\n"
            "- use exactly ONE vague pointer in the whole message; every other\n"
            "  requirement, including every other code, stays spelled out.\n"
            "- prefer, in order: (1) WHICH PAYMENT METHOD pays a fare difference --\n"
            "  'put it on the usual one' -- on an account holding several cards, and\n"
            "  only when that method pays for nothing else in the request; (2) WHICH\n"
            "  RESERVATION to cancel when several are cancelled: name all but one and\n"
            "  point at the last; (3) WHICH CABIN or WHICH FLIGHTS, only when the code\n"
            "  is already named and the flights are what change.\n"
            "- the pointer must be a bare referring term the airline cannot look up:\n"
            "  'that one', 'the usual one'. NOT 'the class I had before' or 'the card\n"
            "  on my profile' -- a single lookup settles those and the flaw disappears.\n"
            "- do NOT invent a prior conversation ('the one you quoted me'): that is a\n"
            "  second, different flaw.\n"
            "- the pointed-at value must appear NOWHERE in the text."),
        "insufficient_information": (
            "- your flaw: pick ONE KIND of requirement and leave it out of EVERY change\n"
            "  in the request that needs it. Say nothing at all about that kind: no\n"
            "  partial form, no pointer, no hedge, no question about it.\n"
            "- state every OTHER requirement with its exact value in a passenger's\n"
            "  plain words -- read codes out as they appear on a confirmation email\n"
            "  ('gift_card_6941833', 'FQ8APE', 'HAT266 on 2024-05-19'), never 'my gift\n"
            "  card ending in 6941833' or 'that same reservation'. If two changes share\n"
            "  a code or a payment method, say it in full for each.\n"
            "- a flight list and a passenger list are each ONE requirement: name EVERY\n"
            "  segment with flight number and date, and EVERY passenger with first\n"
            "  name, last name and date of birth -- or leave that whole list out.\n"
            "  Never a partial list.\n"
            "- for a change to an existing reservation you may leave out: the new cabin\n"
            "  class, the new flight list, the payment method, the new bag count, or the\n"
            "  new passenger list -- prefer the cabin or the flight list when the\n"
            "  request changes them. For a NEW booking: the cabin class only. NEVER\n"
            "  leave out a reservation code, an account user id, an airport, the\n"
            "  one-way/round-trip choice, or the insurance choice."),
    }

    fp_note = ("An air-travel presupposition is usually that part of the request has\n"
               "ALREADY been handled ('since you already moved me to the later flight...',\n"
               "'now that the refund went through...') or that something is impossible or\n"
               "unavailable. Record the clause as `presupposition` and the true requirement\n"
               "it concerns in `concerns` as [slot, value].\n"
               "The requirement named in `concerns` is PART OF the presupposition, not a\n"
               "separate change: never also list it in `other_changes`. Leave\n"
               "`other_changes` empty unless some DIFFERENT requirement was altered.\n")

    extraction_note = (
        "\nHow to read this domain:\n"
        "  - Each slot is one requirement of the change the passenger wants made to\n"
        "    their travel. Values are identifiers as the airline stores them\n"
        "    (reservation ids like 'ZFA04Y', flight numbers like 'HAT001',\n"
        "    dates as YYYY-MM-DD).\n"
        "  - A requirement is STATED if the passenger conveys it in any wording,\n"
        "    including describing a flight rather than naming its number.\n")

    def render_context(self, base: dict) -> str:
        return ("The user is an airline passenger contacting the airline about their "
                "travel plans or an existing reservation.")

    _tool_manual_cache: str | None = None
    _policy_cache: str | None = None

    def tool_manual(self, session=None) -> str:
        if Tau2AirlineAdapter._tool_manual_cache:
            return Tau2AirlineAdapter._tool_manual_cache
        _ensure_tau2()
        from tau2.registry import registry
        env = getattr(session, "env", None) or registry.get_env_constructor("airline")()
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
        Tau2AirlineAdapter._tool_manual_cache = "\n".join(lines)
        return Tau2AirlineAdapter._tool_manual_cache

    def agent_policy(self) -> str:
        if Tau2AirlineAdapter._policy_cache is not None:
            return Tau2AirlineAdapter._policy_cache
        # NOTE: the file is policy.md -- the retail adapter's main_policy.md path was the
        # empty-STORE-POLICY bug (tau2_retail.py:414), do not copy it.
        path = os.environ.get("TAU2_DATA", "") + "/domains/airline/policy.md"
        try:
            with open(path, encoding="utf-8") as f:
                text = f.read()
        except OSError:
            Tau2AirlineAdapter._policy_cache = ""
            return ""
        Tau2AirlineAdapter._policy_cache = self._strip_protocol_rules(text)
        return Tau2AirlineAdapter._policy_cache

    # Clauses that legislate the CONVERSATION rather than what may be done. tau2's
    # airline policy assumes a live two-way phone call: the agent must obtain the user
    # id from the caller and must collect an explicit "yes" before every write. This
    # harness gives the baseline arm no channel to receive either -- its only way to
    # address the user is a final submission -- so those clauses deadlock every episode:
    # measured 2026-08-19, the agent read the reservation, computed the correct change,
    # and then spent its whole patience budget writing confirmation requests without
    # ever issuing the write (13 of 16 clean-control failures made ZERO writes).
    #
    # Every SUBSTANTIVE rule is kept verbatim -- cancellation eligibility, basic-economy
    # immutability, baggage allowances, payment composition, compensation. Only the
    # protocol clauses are replaced, and the replacement states the harness's own
    # protocol so the agent is not left guessing. Recorded as a documented deviation,
    # exactly like solo mode.
    _PROTOCOL_REWRITES = (
        ("Before taking any actions that update the booking database (booking, modifying "
         "flights, editing baggage, changing cabin class, or updating passenger "
         "information), you must list the action details and obtain explicit user "
         "confirmation (yes) to proceed.",
         "The caller's request in this conversation IS their authorisation to act: make "
         "the writes they have asked for. Do not ask for a separate confirmation."),
        ("The agent must first obtain the user id from the user.",
         "The caller is already authenticated; their user id is given to you as call "
         "context."),
        ("The user must provide their user id.",
         "The caller's user id is supplied to you as call context."),
        ("You should only make one tool call at a time, and if you make a tool call, you "
         "should not respond to the user simultaneously. If you respond to the user, you "
         "should not make a tool call at the same time.",
         "Make one tool call per turn."),
    )

    def _strip_protocol_rules(self, text: str) -> str:
        for old, new in self._PROTOCOL_REWRITES:
            text = text.replace(old, new)
        return text

    roleplay_voice = "airline passenger"

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
        """tau2 airline framing: tool manual WITH signatures + the airline's own policy.

        The id-format note is task literacy, not answer leakage (the retail autopsy
        lesson: 90.4% of write attempts died on a normalized id). Airline ids are
        6-character uppercase reservation codes, 'HAT'-prefixed flight numbers, and
        lowercase user ids; every one is user-visible information."""
        return (
            "\nYou are a customer-service agent for an airline, acting on the "
            "passenger's behalf. Always write in ENGLISH. Use the tools below by "
            "writing:\n"
            "Action: Operation\n```\ntool_name(arg=\"value\", ...)\n```\n"
            "One tool call per turn; the observation follows.\n\n"
            "TOOLS:\n" + self.tool_manual() + "\n\n"
            "ID AND FORMAT LITERACY (basic tool use): reservation ids are 6-character "
            "uppercase codes ('ZFA04Y') and flight numbers look like 'HAT001' -- pass "
            "both EXACTLY as given, preserving case. Dates are 'YYYY-MM-DD'. "
            "update_reservation_flights takes the ENTIRE new flight list in travel "
            "order, including unchanged segments -- when only the cabin changes, read "
            "that list off get_reservation_details and resubmit it unchanged; searching "
            "is only for finding NEW flights. get_reservation_details needs the "
            "reservation code ALONE: you do not need a user id to read or change an "
            "existing reservation. Payment ids are copied verbatim from the user's "
            "profile.\n\n"
            # ARGUMENT SHAPES (measured 2026-08-20): 22.3% of write calls were
            # failing, and 74% of those were shape errors, not wrong answers --
            # payment_methods built as bare strings, flights as bare strings, and
            # the cabin written with a space. This is the airline analogue of
            # retail's '#W' prefix lesson: knowing the SHAPE of an argument is
            # tool literacy, not knowledge of what the customer wants.
            "ARGUMENT SHAPES (get these exactly right -- they are the most common "
            "cause of failed writes):\n"
            "  flights=[{\"flight_number\": \"HAT001\", \"date\": \"2024-05-20\"}, ...] "
            "-- a LIST OF OBJECTS, never bare strings like [\"HAT001\"], and each "
            "segment needs BOTH keys.\n"
            "  passengers=[{\"first_name\": \"Sofia\", \"last_name\": \"Kim\", "
            "\"dob\": \"1990-04-05\"}, ...] -- objects with all three keys.\n"
            "  payment_methods=[{\"payment_id\": \"credit_card_1234567\", "
            "\"amount\": 250}, ...] -- objects with BOTH a payment_id and an "
            "integer amount, and the amounts MUST sum to the exact total the "
            "booking costs (fares x passengers, plus $30 per passenger if "
            "insurance is yes, plus $50 per non-free checked bag). Use calculate() "
            "to get that total right before you book.\n"
            "  cabin is one of exactly: basic_economy, economy, business -- with "
            "an UNDERSCORE, never 'basic economy'.\n"
            "  update_reservation_baggages takes total_baggages AND "
            "nonfree_baggages as integers, plus a payment_id string.\n\n"
            "IDENTITY (harness convention, stated because the policy below assumes a "
            "phone call): the passenger was ALREADY AUTHENTICATED when this call was "
            "connected, so their identity is established. Do NOT ask for a user id in "
            "order to read or change a reservation they name by code -- act on the "
            "code. A NEW booking still needs their user id as a tool argument, and they "
            "will give it if you ask. Likewise the REASON for a cancellation is recorded "
            "out of band when the call is logged -- do not hold a cancellation the "
            "passenger has asked for while you interrogate them for a reason; apply the "
            "eligibility rules to the reservation's own facts (booking time, cabin, "
            "insurance, whether a segment was cancelled by the airline).\n\n"
            "IMPORTANT -- anything you put in Action: Answer is treated as your FINAL "
            "submission, never as a message to the passenger: a question or an apology "
            "there ends the episode as a failed answer. Read the passenger's profile "
            "and reservations with the lookup tools (get_user_details, "
            "get_reservation_details); do not ask for what the tools can tell you. "
            "Where policy requires the passenger's confirmation, their request and "
            "their answers in this conversation ARE that confirmation -- never use "
            "Action: Answer to ask for it.\n\n"
            "When every requested change has been made, finish with:\n"
            "Action: Answer\nFinal Answer: done\n"
            "The database state is what gets checked, not your words.\n\n"
            "AIRLINE POLICY (binding):\n" + self.agent_policy() + "\n")

    def unmet_requests(self, session, node) -> list[str]:
        """The passenger's OWN unfinished requests, as they would name them after
        checking their reservations -- their intent, never the answer sheet."""
        from ..executors.tau2_inproc import Tau2Session
        recipe = self.compile(dict(node.base), tuple(node.conditions))
        done = {self._action_key(n, k)
                for n, k in getattr(session, "executed_calls", [])}
        out = []
        for a in recipe.get("actions") or []:
            args = Tau2Session._uncanon(a.get("arguments") or {})
            if self._action_key(a["name"], args) in done:
                continue
            verb = {"book_reservation": "the new booking",
                    "cancel_reservation": "the cancellation",
                    "update_reservation_flights": "the flight change",
                    "update_reservation_baggages": "the baggage update",
                    "update_reservation_passengers": "the passenger-details update",
                    "send_certificate": "the compensation certificate"}.get(
                        a["name"], a["name"].replace("_", " "))
            rid = args.get("reservation_id")
            bits = [verb + (f" for reservation {rid}" if rid else "")]
            # F4 (audit 2026-08-20): this text becomes the user's reply after a rejected
            # proposal ("these things are STILL not done: ..."). Reciting the value of a
            # slot the perturbation HID hands the answer to an arm that never asked --
            # measured: 7 B0 episodes had their hidden value read back, 3 of which then
            # succeeded. Name the action and the reservation; never the hidden value.
            hidden = getattr(self, "hidden_slots", None) or set()
            idx = recipe.get("actions", []).index(a) if a in recipe.get("actions", []) else 0

            def _hidden(key: str) -> bool:
                return any(str(h).endswith(f":{a['name']}:{key}") for h in hidden)

            if args.get("flights") and not _hidden("flights"):
                bits.append("flights " + ", ".join(
                    f"{f.get('flight_number')} on {f.get('date')}"
                    for f in args["flights"] if isinstance(f, dict)))
            if args.get("cabin") and not _hidden("cabin"):
                bits.append(f"in {str(args['cabin']).replace('_', ' ')}")
            if args.get("total_baggages") is not None and not _hidden("total_baggages"):
                bits.append(f"{args['total_baggages']} checked bags")
            if args.get("payment_id") and not _hidden("payment_id"):
                bits.append(f"via {args['payment_id']}")
            out.append(" -- ".join(bits))
        return out

    def parse_proposal(self, raw: str) -> Any:
        return {"declared_done": True, "raw": raw}

    # Golden-action grading, as ruled for retail (2026-08-17): success requires the
    # expected tool CALLS to have been made, not just their end state reached. Airline's
    # alternate-path structure lives only in list orderings, which _action_key
    # canonicalizes deliberately: flights stay a SEQUENCE (segment order is itinerary
    # semantics), passengers and payments compare as SETS. Flagged for the author's ruling.
    golden_action_check = True

    @staticmethod
    def _action_key(name: str, args: dict) -> tuple:
        a = {k: v for k, v in (args or {}).items()}
        if isinstance(a.get("flights"), list):
            a["flights"] = [(str((f or {}).get("flight_number")), str((f or {}).get("date")))
                            for f in a["flights"] if isinstance(f, dict)]
        if isinstance(a.get("passengers"), list):
            a["passengers"] = sorted(
                (str((p or {}).get("first_name")), str((p or {}).get("last_name")),
                 str((p or {}).get("dob")))
                for p in a["passengers"] if isinstance(p, dict))
        if isinstance(a.get("payment_methods"), list):
            a["payment_methods"] = sorted(
                (str((p or {}).get("payment_id")), int((p or {}).get("amount") or 0))
                for p in a["payment_methods"] if isinstance(p, dict))
        return (name, json.dumps(a, sort_keys=True, default=str))

    def accepts(self, proposal: Any, node, session, executor=None) -> tuple[bool, str]:
        """Executable acceptance: the live flight db matches the intent's own end state,
        AND (golden-action check) every expected write was actually issued."""
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

    def shift_edge_ok(self, graph, src, dst, operator=None) -> bool:
        """A shift may not move the goal to a DIFFERENT passenger.

        AUDIT 2026-08-20: 97 of 276 child nodes (35%) belong to another tau2 task and
        therefore another user, and nothing filtered them -- 18 campaign episodes shifted
        across users and ALL 18 ended EXHAUSTED. They are unwinnable by construction: the
        agent is told it is authenticated as the original caller, this adapter's own
        policy gate calls cross-user action a violation (A0:mixed_users), and the
        injected call context still names the old user. traversal.py probes this hook
        with getattr, so defining it is enough to gate sampling fail-closed.
        """
        try:
            return (self._seed_user(tuple(src.conditions))
                    == self._seed_user(tuple(dst.conditions)))
        except Exception:
            return False

    def act_is_mutating(self, raw: str) -> bool:
        return any(w in (raw or "") for w in WRITE_TOOLS)


from .base import register  # noqa: E402
from ..executors.tau2_inproc import Tau2Executor  # noqa: E402

register("tau2_airline", Tau2AirlineAdapter, Tau2Executor)
