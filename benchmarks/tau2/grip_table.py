"""Full GRIP table, all four dimensions, per persona per arm."""
import json, os, sys, re, ast, statistics as st
from pathlib import Path
sys.path.insert(0,"pipeline/src")
os.environ.setdefault("TAU2_SRC","tau2-bench/src"); os.environ.setdefault("TAU2_DATA","tau2-bench/data/tau2")
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP","True")
from loguru import logger as L; L.remove()
from intent_graph.runtime import grip
from intent_graph import storage
from intent_graph.adapters.tau2_retail import Tau2RetailAdapter
from intent_graph.executors.tau2_inproc import Tau2Session

AD=Tau2RetailAdapter()
G={g.graph_id:g for g in storage.iter_graphs(Path("artifacts/trees/tau2_retail"))}
WR={"exchange_delivered_order_items","return_delivered_order_items","cancel_pending_order",
    "modify_pending_order_items","modify_pending_order_address","modify_pending_order_payment",
    "modify_user_address"}
def parse(c):
    m=re.match(r"\s*(\w+)\s*\((.*)\)\s*$",c or "",re.S)
    if not m: return None,{}
    try:
        n=ast.parse(f"_f({m.group(2)})",mode="eval").body
        kw={k.arg:ast.literal_eval(k.value) for k in n.keywords}
    except Exception: return m.group(1),{}
    o=kw.get("order_id")
    if isinstance(o,str) and re.fullmatch(r"W\d+",o): kw["order_id"]="#"+o
    return m.group(1),kw
GC={}
def gold(g,nid):
    k=(g.graph_id,nid)
    if k not in GC:
        nd=next((x for x in (g.root,*g.children) if x.intent_id==nid),None)
        GC[k]=set() if nd is None else {AD._action_key(a["name"],Tau2Session._uncanon(a.get("arguments") or {}))
              for a in AD.compile(dict(nd.base),tuple(nd.conditions)).get("actions") or []}
    return GC[k]

def cell(a,p,s):
    d=Path(f"artifacts/persona2/{a}_{p}_s{s}")
    if not d.exists(): return None
    rows=[]
    for f in d.glob("*.json"):
        if f.name=="run_config.json": continue
        try:
            e=json.loads(f.read_text()); rows.append((e,grip.score(e)))
        except Exception: pass
    n=len(rows) or 1
    rec=[g.recovery for _,g in rows if g.recovery is not None]
    aim=[g.aim for _,g in rows if g.aim is not None]
    shd=[(e,g) for e,g in rows if g.n_shifts>0]
    # STALENESS: shifted episodes that never executed any of the NEW intent's requests and failed
    # REACTION : turns from the shift to the agent's first commit (write) after it
    pure=tot=0; react=[]
    for e,g in shd:
        gr=G.get(e["header"].get("graph_id") or e["header"].get("tree_id"))
        if gr is None: continue
        sh=[t for t in (e.get("turns") or []) if t.get("shift")]
        if not sh: continue
        old=gold(gr,gr.root.intent_id); new=gold(gr,e.get("final_node"))
        stale_set,new_only=old-new,new-old
        st_turn=sh[0]["turn"]; done=set(); first=None
        for t in e.get("turns") or []:
            act=t.get("action") or {}
            if act.get("kind")!="ACT" or str(t.get("observation","")).startswith("error"): continue
            nm,kw=parse(act.get("command") or "")
            if nm in WR:
                done.add(AD._action_key(nm,kw))
                if first is None and t["turn"]>=st_turn: first=t["turn"]
        if stale_set:
            tot+=1
            if not (done & new_only) and not g.success: pure+=1
        if first is not None: react.append(first-st_turn)
    return {"succ":100*sum(g.success for _,g in rows)/n,
        "earned":100*sum(g.bucket=="EARNED" for _,g in rows)/n,
        "inferred":100*sum(g.bucket=="INFERRED" for _,g in rows)/n,
        "aim":100*st.mean(aim) if aim else None,
        "recovery":100*st.mean(rec) if rec else None,
        "patience":st.mean(g.patience_spent for _,g in rows),
        "stale":100*pure/tot if tot else None,
        "react":st.mean(react) if react else None,
        "postshift":100*sum(g.success for _,g in shd)/max(len(shd),1) if shd else None}

def ms(v,f="{:.2f}"):
    v=[x for x in v if x is not None]
    if not v: return "--"
    return (f+"±"+f).format(st.mean(v),st.stdev(v)) if len(v)>1 else f.format(v[0])

LB={"A0":"A0 · Just Ask (ICLR 25)","A1":"A1 · + memory","A2":"A2 · + planner/verifier","A3":"A3 · Self-evolving"}
C=["Success","Earned","Inferred","Aim","Recovery","Patience","Staleness","Reaction","PostShift"]
K=["succ","earned","inferred","aim","recovery","patience","stale","react","postshift"]
print("Persona\tMethod\t"+"\t".join(C))
for p in ("dependent","avoidant","intuitive","spontaneous"):
    for a in ("A0","A1","A2","A3"):
        v=[cell(a,p,s) for s in (1,2,3)]; v=[x for x in v if x]
        if not v: continue
        print(f"{p}\t{LB[a]}\t"+"\t".join(ms([x[k] for x in v]) for k in K))
