"""GRIP across all campaign cells, aggregated mean±sd over the three seeds.
Scored with TODAY's runtime/grip.py so every arm is on one scorer."""
import json, glob, statistics as st, sys, os
sys.path.insert(0,'pipeline/src'); sys.path.insert(0,'.')
os.environ.setdefault('TAU2_SRC', os.getcwd()+'/tau2-bench/src')
os.environ.setdefault('TAU2_DATA', os.getcwd()+'/tau2-bench/data/tau2')
import airline_compat
from loguru import logger as L; L.remove()
from intent_graph.runtime import grip

def cell(d):
    fs = glob.glob(f'artifacts/{d}/*.json')
    if len(fs) < 208: return None
    rows=[]
    for f in fs:
        try: rows.append(grip.score(json.load(open(f))))
        except Exception: pass
    g = grip.report(rows) if rows else {}
    gw = g.get('grounding_where_hidden') or {}
    return {"succ": (g.get('success_rate') or 0)*100,
            "EARNED": gw.get('EARNED'), "INFERRED": gw.get('INFERRED'),
            "RNS": gw.get('RECOVERED_NOT_SOLVED'), "PREM": gw.get('PREMATURE'),
            "rec": (g.get('recovery') or {}).get('mean'),
            "aim": (g.get('inquiry') or {}).get('aim'),
            "asks": (g.get('inquiry') or {}).get('mean_asks')}

def ms(vals):
    v=[x for x in vals if x is not None]
    if not v: return "   --      "
    return f"{st.mean(v):.3f}±{(st.stdev(v) if len(v)>1 else 0):.3f}"

arms = "A0 A1 A2 B0 B1 B2 B3 B4 B5 B6".split()
res={}
for a in arms:
    cells=[cell(f'camp_{a}_s{s}') for s in (71,72,73)]
    cells=[c for c in cells if c]
    res[a]=cells
    print(f"scored {a} ({len(cells)} seeds)", flush=True)

print("\nGRIP, mean±sd over 3 seeds (208 episodes per seed), scored with today's grip.py")
print(f"\n  {'arm':4s} {'success':>13s} {'EARNED':>13s} {'INFERRED':>13s} {'REC_NOT_SOLVED':>14s} {'recovery':>13s} {'aim':>13s} {'asks/ep':>13s}")
print("  " + "-"*100)
for a in sorted(arms, key=lambda x: -st.mean([c['succ'] for c in res[x]])):
    cs=res[a]
    print(f"  {a:4s} {ms([c['succ'] for c in cs]):>13s} {ms([c['EARNED'] for c in cs]):>13s} "
          f"{ms([c['INFERRED'] for c in cs]):>13s} {ms([c['RNS'] for c in cs]):>14s} "
          f"{ms([c['rec'] for c in cs]):>13s} {ms([c['aim'] for c in cs]):>13s} {ms([c['asks'] for c in cs]):>13s}")
json.dump({a:res[a] for a in arms}, open('artifacts/airline_grip_agg.json','w'), indent=1)
