"""The retail policy gate mandated by the spike (docs/tau2-spike-report.md section 3).

tau2 retail enforces 13 of its 20 binding policy rules in code, and v3's
raise-on-rejected-write already removes any branch that trips those. What remains are the
prose-only rules a mutated branch can violate SILENTLY (the write executes, the hash is
valid, but a policy-compliant agent must refuse to produce it). Those branches are not
legitimate goals: a perfect agent fails on them by doing the right thing. The gate finds
them mechanically -- predicates over the branch's golden actions plus the database; no
LLM anywhere (tree construction is code-only by project rule).

Predicates (ids follow the spike's shortlist):
  P2 repeat-address-modify   modify_pending_order_address called twice on one order
                             (prose: modify tools "can only be called once per order";
                             the tool only checks status='pending', which address-modify
                             does not change, so a second call executes silently)
  P3 modify-after-item-modify any modify_*/cancel on an order AFTER
                             modify_pending_order_items on that order (prose: "will not
                             be able to modify or cancel the order anymore")
  P4 no-op edit              exchange/modify-items with old==new item id; modify-payment
                             to the order's current payment method; modify-address to the
                             order's current address (a request for no change is not a
                             goal)
  P5 refund routing          return refund must go to the original payment method or an
                             existing gift card of the order's owner (runtime-rejected in
                             the spike's probes, so v3 should already be clean -- kept as
                             defense in depth)
  P6 pristine degeneracy     branch ground truth equals the untouched database (success
                             indistinguishable from doing nothing)
"""
import json
import os
import sys
from collections import Counter
from pathlib import Path

WS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, WS + "/../../pipeline/src")
os.environ["TAU2_SRC"] = WS + "/tau2-bench/src"
os.environ["TAU2_DATA"] = WS + "/tau2-bench/data/tau2"
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
from loguru import logger as _l          # noqa: E402
_l.remove()

from intent_tree import storage                                 # noqa: E402
from intent_tree.adapters.tau2_retail import Tau2RetailAdapter  # noqa: E402
from intent_tree.executors.tau2_inproc import Tau2Executor      # noqa: E402

MODIFY_TOOLS = {"modify_pending_order_address", "modify_pending_order_payment",
                "modify_pending_order_items", "cancel_pending_order"}


def order_facts(db) -> dict:
    facts = {}
    for oid, o in db.orders.items():
        paid = [p.payment_method_id for p in (o.payment_history or [])
                if getattr(p, "transaction_type", "payment") == "payment"]
        facts[oid] = {
            "original_payment": paid[0] if paid else None,
            "address": (o.address.model_dump() if hasattr(o.address, "model_dump")
                        else dict(o.address)) if o.address else None,
            "user_id": o.user_id,
        }
    return facts


def user_gift_cards(db) -> dict:
    out = {}
    for uid, u in db.users.items():
        pm = u.payment_methods or {}
        out[uid] = {k for k in pm if str(k).startswith("gift_card")}
    return out


def violations(recipe, facts, giftcards, user_addresses=None) -> list[str]:
    out = []
    addr_seen: Counter = Counter()
    items_modified: set = set()
    for a in recipe.get("actions") or []:
        name, args = a["name"], a.get("arguments") or {}
        oid = args.get("order_id")
        f = facts.get(oid, {})
        if name == "modify_user_address":
            uid = args.get("user_id")
            new_addr = {k: v for k, v in args.items() if k != "user_id"}
            if (user_addresses or {}).get(uid) == new_addr:
                out.append(f"P4:noop_user_address:{uid}")
        if name in MODIFY_TOOLS and oid in items_modified:
            out.append(f"P3:{name}:{oid}")
        if name == "modify_pending_order_address":
            addr_seen[oid] += 1
            if addr_seen[oid] > 1:
                out.append(f"P2:repeat_address_modify:{oid}")
            new_addr = {k: v for k, v in args.items() if k != "order_id"}
            if f.get("address") and new_addr == f["address"]:
                out.append(f"P4:noop_address:{oid}")
        if name == "modify_pending_order_items":
            items_modified.add(oid)
        if name in ("modify_pending_order_items", "exchange_delivered_order_items"):
            old = list(args.get("item_ids") or [])
            new = list(args.get("new_item_ids") or [])
            for o_id, n_id in zip(old, new):
                if o_id == n_id:
                    out.append(f"P4:noop_item_swap:{oid}:{o_id}")
        if name == "modify_pending_order_payment":
            if args.get("payment_method_id") == f.get("original_payment"):
                out.append(f"P4:noop_payment:{oid}")
        if name == "return_delivered_order_items":
            pm = args.get("payment_method_id")
            ok = (pm == f.get("original_payment")
                  or pm in giftcards.get(f.get("user_id"), set()))
            if not ok:
                out.append(f"P5:refund_routing:{oid}:{pm}")
    return out


def main() -> None:
    adapter, executor = Tau2RetailAdapter(), Tau2Executor()
    trees = sorted(storage.iter_trees(Path(WS + "/artifacts/trees/tau2_retail")),
                   key=lambda t: t.tree_id)
    audit: dict = {}
    per_op: dict = {}
    stats = Counter()
    with executor.open(adapter.env_spec(adapter.ENV_KEY)) as session:
        db = session.env.tools.db
        facts, gifts = order_facts(db), user_gift_cards(db)
        uaddr = {uid: (u.address.model_dump() if hasattr(u.address, "model_dump")
                       else dict(u.address)) if u.address else None
                 for uid, u in db.users.items()}
        pristine = adapter.execute({"actions": [], "reads": []}, session)
        for tree in trees:
            op_of = {e.dst: str(e.operator) for e in (tree.edges or [])}
            rows = {}
            for node in tree.children:
                recipe = adapter.compile(dict(node.base), tuple(node.conditions))
                v = violations(recipe, facts, gifts, uaddr)
                if node.ground_truth == pristine:
                    v.append("P6:pristine_degenerate")
                op = op_of.get(node.intent_id, "?")
                d = per_op.setdefault(op, Counter())
                d["nodes"] += 1
                stats["nodes"] += 1
                if v:
                    d["violating"] += 1
                    stats["violating"] += 1
                    for tag in v:
                        d[tag.split(":", 1)[0]] += 1
                    rows[node.intent_id] = v
            if rows:
                audit[tree.tree_id] = rows
    summary = {"nodes": stats["nodes"], "violating": stats["violating"],
               "rate": round(stats["violating"] / max(stats["nodes"], 1), 4),
               "per_operator": {k: dict(v) for k, v in sorted(per_op.items())}}
    Path(WS + "/artifacts/retail_policy_audit.json").write_text(
        json.dumps({"summary": summary, "violations": audit}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
