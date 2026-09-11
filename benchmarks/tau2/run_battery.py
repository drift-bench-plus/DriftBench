import os, sys, json
WS=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,WS+"/../../pipeline/src")
os.environ["TAU2_SRC"]=WS+"/tau2-bench/src"; os.environ["TAU2_DATA"]=WS+"/tau2-bench/data/tau2"
from loguru import logger as L; L.remove()
from pathlib import Path
from collections import Counter
from intent_tree import storage, battery
from intent_tree.adapters.tau2_telecom import Tau2TelecomAdapter
from intent_tree.adapters.tau2_retail import Tau2RetailAdapter
from intent_tree.adapters.tau2_airline import Tau2AirlineAdapter
from intent_tree.executors.tau2_inproc import Tau2Executor

out = {}
for name, Adapter in (("tau2_telecom", Tau2TelecomAdapter), ("tau2_retail", Tau2RetailAdapter),
                      ("tau2_airline", Tau2AirlineAdapter)):
    trees = list(storage.iter_trees(Path(WS + f"/artifacts/trees/{name}")))
    findings = battery.run(trees, Adapter(), Tau2Executor(), reproduce=True)
    cats = Counter(f.check for f in findings)
    out[name] = {"trees": len(trees), "findings": len(findings), "by_check": dict(cats),
                 "examples": [f"{f.tree_id[:12]} {f.check}: {f.detail[:90]}" for f in findings[:6]]}
    print(name, json.dumps(out[name], indent=1))
json.dump(out, open(WS+"/artifacts/battery_report.json","w"), indent=1)
