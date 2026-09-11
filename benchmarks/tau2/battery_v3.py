import os, sys, json
WS=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,WS+"/../../pipeline/src")
os.environ["TAU2_SRC"]=WS+"/tau2-bench/src"; os.environ["TAU2_DATA"]=WS+"/tau2-bench/data/tau2"
from loguru import logger as L; L.remove()
from pathlib import Path
from collections import Counter
from intent_tree import storage, battery
from intent_tree.adapters.tau2_retail import Tau2RetailAdapter
from intent_tree.executors.tau2_inproc import Tau2Executor
trees=list(storage.iter_trees(Path(WS+"/artifacts/trees/tau2_retail")))
f=battery.run(trees, Tau2RetailAdapter(), Tau2Executor(), reproduce=True)
print(json.dumps({"trees":len(trees),"findings":len(f),
                  "by_check":dict(Counter(x.check for x in f)),
                  "examples":[f"{x.tree_id[:10]} {x.check}: {x.detail[:80]}" for x in f[:5]]}, indent=1))
