"""Post-hoc golden-action audit of finished validation runs.

For every SUCCESS episode: parse the transcript's successfully-executed tool calls,
apply the same canonical action keying as the new acceptance check (pairing-aware),
and test whether every golden action of the episode's intent was actually issued.
Reports how many of today's db-hash successes survive the stricter grading.
"""
import ast
import json
import os
import re
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
from intent_tree import storage                                  # noqa: E402
from intent_tree.adapters.tau2_retail import Tau2RetailAdapter   # noqa: E402
from intent_tree.executors.tau2_inproc import Tau2Session        # noqa: E402


def parse_cmd(cmd: str):
    m = re.match(r"\s*(\w+)\s*\((.*)\)\s*$", cmd or "", re.S)
    if not m:
        return None, {}
    try:
        node = ast.parse(f"_f({m.group(2)})", mode="eval").body
        kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in node.keywords}
    except Exception:
        return m.group(1), {}
    # same '#' normalization the executor applies
    oid = kwargs.get("order_id")
    if isinstance(oid, str) and re.fullmatch(r"W\d+", oid):
        kwargs["order_id"] = "#" + oid
    return m.group(1), kwargs


def audit(d: str, adapter, trees) -> dict:
    out = Counter()
    weak = []
    for p in sorted(Path(d).glob("*.json")):
        e = json.loads(p.read_text())
        if e.get("outcome") != "SUCCESS":
            continue
        out["successes"] += 1
        tree = trees.get(e["header"].get("tree_id"))
        recipe = adapter.compile(dict(tree.root.base), tuple(tree.root.conditions))
        done = set()
        for t in e.get("turns") or []:
            a = t.get("action") or {}
            if a.get("kind") != "ACT":
                continue
            if str(t.get("observation", "")).startswith("error"):
                continue
            name, kwargs = parse_cmd(a.get("command") or "")
            if name:
                done.add(adapter._action_key(name, kwargs))
        missing = [x["name"] for x in (recipe.get("actions") or [])
                   if adapter._action_key(x["name"],
                                          Tau2Session._uncanon(x.get("arguments") or {}))
                   not in done]
        if missing:
            out["would_fail_golden"] += 1
            weak.append((p.name, missing[:3]))
    return {"summary": dict(out), "downgraded": weak[:8]}


def main():
    adapter = Tau2RetailAdapter()
    trees = {t.tree_id: t for t in storage.iter_trees(Path(WS + "/artifacts/trees/tau2_retail"))}
    for d in ("artifacts/retail_validate_clean", "artifacts/retail_validate_pert",
              "artifacts/retail_validate_A3"):
        if Path(WS + "/" + d).exists():
            print(d, "->", json.dumps(audit(WS + "/" + d, adapter, trees)))


if __name__ == "__main__":
    main()
