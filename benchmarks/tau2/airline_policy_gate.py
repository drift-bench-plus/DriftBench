"""The airline policy gate -- the load-bearing legitimacy audit for airline trees.

Where retail's environment enforces most of its policy in code (the retail gate found 0
violations because raise-on-rejected-write had already filtered them), airline's
environment enforces almost NONE of its prose rules: cancels have no eligibility check,
basic-economy reservations modify silently, payment composition and the five-passenger
cap are unchecked, baggage counts may silently decrease. A branch violating any of these
executes fine and hashes as a "distinct ground truth" -- but a policy-compliant agent
must REFUSE it, so success would require disobeying the policy. Such branches are not
legitimate goals.

The predicates live on the adapter (Tau2AirlineAdapter.policy_violations, with
intra-recipe state tracking -- a golden trace may upgrade a cabin and then cancel), and
the ENGINE already applies them at build time through is_valid_intent, so this audit is
the verification pass and the shipped artifact, expected to report zero violating
branches. Predicate glossary:

  A0  user coherence       one branch touches reservations of more than one user
  A1  basic-economy        flight-set change on a basic_economy reservation
                           (cabin-only changes are allowed)
  A2  route immutable      new flights change origin/destination/trip shape
  A3  cabin-after-flown    cabin change while a segment has been flown
  A4  cancel eligibility   cancel with a flown segment (transfer-only), or with no
                           reason-independent eligibility (<24h / business /
                           airline-cancelled segment) and no insurance. Insurance-
                           carried cancels are eligible but listed under
                           `reason_dependent_cancels`: the perturbation stage must not
                           perturb the reason on those branches.
  A5  passenger cap        book with more than 5 passengers
  A6  payment composition  >1 certificate, >1 credit card, or >3 gift cards on a book
  A7  baggage decrease     baggage update below the current total (add-only)
  A8  baggage arithmetic   nonfree != max(0, total - allowance x passengers)
  A9  compensation         any send_certificate (outside the task distribution)
  A10 itinerary coherence  flight list fails to chain / regressive or past dates
  A11 no-op forms          re-cancel, update of a cancelled reservation, identical-value
                           updates; plus pristine ground truth (execution-level, checked
                           here because is_valid_intent runs before execution)

Roots are audited too: a root violation is a predicate bug or an upstream task fact to
reconcile, never something to prune -- roots are tau2's own tasks.
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
import airline_compat  # noqa: F401  (tree->graph rename shim)
from loguru import logger as _l          # noqa: E402
_l.remove()

from intent_tree import storage                                   # noqa: E402
from intent_tree.adapters.tau2_airline import Tau2AirlineAdapter  # noqa: E402
from intent_tree.executors.tau2_inproc import Tau2Executor        # noqa: E402


def main() -> None:
    adapter, executor = Tau2AirlineAdapter(), Tau2Executor()
    trees = sorted(storage.iter_trees(Path(WS + "/artifacts/trees/tau2_airline")),
                   key=lambda t: t.tree_id)
    audit: dict = {}
    reason_dep_rows: dict = {}
    per_op: dict = {}
    stats = Counter()
    with executor.open(adapter.env_spec(adapter.ENV_KEY)) as session:
        pristine = adapter.execute({"actions": [], "reads": []}, session)
        for tree in trees:
            op_of = {e.dst: str(e.operator) for e in (tree.edges or [])}
            rows = {}
            for node in tree.children:
                recipe = adapter.compile(dict(node.base), tuple(node.conditions))
                v, reason_dep = adapter.policy_violations(recipe)
                if node.ground_truth == pristine:
                    v.append("A11:pristine_degenerate")
                if reason_dep:
                    reason_dep_rows.setdefault(tree.tree_id, []).append(node.intent_id)
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
            r_v, r_dep = adapter.policy_violations(
                adapter.compile(dict(tree.root.base), tuple(tree.root.conditions)))
            if r_v:
                rows["ROOT:" + tree.root.intent_id] = r_v
                stats["root_violations"] += 1
            if r_dep:
                reason_dep_rows.setdefault(tree.tree_id, []).append(
                    "ROOT:" + tree.root.intent_id)
            if rows:
                audit[tree.tree_id] = rows
    summary = {"trees": len(trees), "nodes": stats["nodes"],
               "violating": stats["violating"],
               "rate": round(stats["violating"] / max(stats["nodes"], 1), 4),
               "root_violations": stats["root_violations"],
               "reason_dependent_cancel_nodes":
                   sum(len(v) for v in reason_dep_rows.values()),
               "per_operator": {k: dict(v) for k, v in sorted(per_op.items())}}
    Path(WS + "/artifacts/airline_policy_audit.json").write_text(
        json.dumps({"summary": summary, "violations": audit,
                    "reason_dependent_cancels": reason_dep_rows}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
