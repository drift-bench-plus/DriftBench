"""Certify every telecom sibling-shift edge by execution.

For each ordered pair of nodes the traversal could shift between, run the three-execution
certificate (non-vacuous / moved / achievable) and store the verdicts. Fail-closed: the
episode layer only samples edges whose certificate says eligible.
"""
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

WS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, WS + "/../../pipeline/src")
os.environ["TAU2_SRC"] = WS + "/tau2-bench/src"
os.environ["TAU2_DATA"] = WS + "/tau2-bench/data/tau2"
os.environ["TAU2_SHIFT_CERTS"] = WS + "/artifacts/telecom_shift_certificates.json"
from loguru import logger as _l          # noqa: E402
_l.remove()

from intent_tree import storage                                   # noqa: E402
from intent_tree.adapters.tau2_telecom import Tau2TelecomAdapter  # noqa: E402
from intent_tree.executors.tau2_inproc import Tau2Executor        # noqa: E402
from intent_tree.runtime.traversal import classify                # noqa: E402


def main() -> None:
    adapter, executor = Tau2TelecomAdapter(), Tau2Executor()
    trees = sorted(storage.iter_trees(Path(WS + "/artifacts/trees/tau2_telecom")),
                   key=lambda t: t.tree_id)
    certs: dict = {}
    stats = Counter()
    per_op = {}
    t0 = time.time()
    with executor.open(adapter.env_spec(adapter.ENV_KEY)) as session:
        for ti, tree in enumerate(trees, 1):
            nodes = {n.intent_id: n for n in (tree.root, *tree.children)}
            tree_certs = {}
            for sid, src in nodes.items():
                for did, dst in nodes.items():
                    if sid == did:
                        continue
                    op = classify(src.conditions, dst.conditions, src.base, dst.base)
                    if op is None:
                        op = adapter.classify_shift(src, dst)
                    if op is None:
                        continue
                    c = adapter.certify_shift(session, src, dst)
                    c["operator"] = op.value
                    tree_certs[f"{sid}->{did}"] = c
                    stats["edges"] += 1
                    c["eligible"] = adapter.cert_eligible(c, op.value)
                    stats["eligible"] += c["eligible"]
                    d = per_op.setdefault(op.value, Counter())
                    d["edges"] += 1
                    d["eligible"] += c["eligible"]
                    d["fail_non_vacuous"] += not c["non_vacuous"]
                    d["fail_moved"] += (c["non_vacuous"] and not c["moved"])
                    d["fail_achievable"] += (c["non_vacuous"] and c["moved"]
                                             and not c["achievable"])
                    if "error" in c:
                        d["errors"] += 1
            certs[tree.tree_id] = tree_certs
            if ti % 25 == 0:
                print(f"  {ti}/{len(trees)} trees | {stats['edges']} edges | "
                      f"eligible {stats['eligible']} | {time.time()-t0:.0f}s", flush=True)

    out = {"summary": {"trees": len(trees), "edges": stats["edges"],
                       "eligible": stats["eligible"],
                       "eligible_rate": round(stats["eligible"] / max(stats["edges"], 1), 3),
                       "wall_s": round(time.time() - t0, 1),
                       "per_operator": {k: dict(v) for k, v in sorted(per_op.items())}},
           "certificates": certs}
    Path(os.environ["TAU2_SHIFT_CERTS"]).write_text(json.dumps(out["certificates"], indent=1))
    Path(WS + "/artifacts/telecom_shift_cert_summary.json").write_text(
        json.dumps(out["summary"], indent=1))
    print(json.dumps(out["summary"], indent=1))


if __name__ == "__main__":
    main()
