"""Run interaction episodes (B0 and reference arms) over the telecom sample set.

Axis A only tonight: the perturbed opening query misrepresents the user's problems and the
agent must recover them through the conversation. Axis B (mid-episode intent shift) is NOT
run for telecom -- see the note in the overnight log: on this domain the conditions ARE the
faults injected into the world, so "the intent moved" would mean the phone retroactively
had different problems. Refinement-shaped shifts (the user notices a further problem) are
coherent and left as future work.
"""
import argparse
import hashlib
import json
import os
import random
import sys
import threading
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

WS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, WS + "/../../pipeline/src")
os.environ["TAU2_SRC"] = WS + "/tau2-bench/src"
os.environ["TAU2_DATA"] = WS + "/tau2-bench/data/tau2"
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
if "ARK_API_KEY" not in os.environ:
    _envf = os.path.join(WS, "..", ".env")
    if os.path.exists(_envf):
        os.environ["ARK_API_KEY"] = [l.split("=", 1)[1].strip() for l in
                                     open(_envf)
                                     if l.startswith("ARK_API_KEY=")][0]
from loguru import logger as _loguru          # noqa: E402
_loguru.remove()

from intent_tree import storage                                   # noqa: E402
from intent_tree.adapters.tau2_telecom import Tau2TelecomAdapter  # noqa: E402
from intent_tree.executors.tau2_inproc import Tau2Executor        # noqa: E402
from intent_tree.runtime import grip                              # noqa: E402
from intent_tree.runtime.agents import build_agent                # noqa: E402
from intent_tree.runtime.episode import Episode                   # noqa: E402
from intent_tree.runtime.llm import LLMClient                     # noqa: E402
from intent_tree.runtime.persona import PERSONA_IDS               # noqa: E402
from perturb_run import LLM_CFG                                   # noqa: E402

RUNTIME = {
    # A tau2 task needs a check and a fix per request/fault, so short turn budgets cut
    # B0 off mid-repair (measured on the first smoke runs).
    "max_turns": 30,
    # SETTLED ECONOMY (WebShop 2026-08-14, ported 2026-08-16): base 10 x persona
    # multiplier applied ONCE as an exact float; ask 2, rejected proposal 4; the
    # affordability rule in episode.py does the rest. There is no separate question cap.
    "patience_init": 10,
    "cost_ask": 2, "cost_reject": 4, "cost_act": 0, "cost_malformed": 1,
    # USER V2 (7-step redesign): code decides answering; [already mentioned]/[PRIVATE]
    # tagging; rejections carry a hint; volunteering is a separate code-triggered prompt.
    "user_v2": True,
    "p_shift": 0.0,                 # axis A only; --shift overrides
    "axis_b_on_write_trees": False,
    "forced_final": True,
}


def load_samples(path: Path, limit: int, seed: int) -> list[dict]:
    data = json.loads(path.read_text())
    adm = [r for r in data["results"] if r["status"] == "admitted"]
    rng = random.Random(seed)
    rng.shuffle(adm)
    return adm[:limit] if limit else adm


def clean_query(adapter, conditions, llm=None) -> str:
    """The control arm's opening line: the SAME intent, stated plainly and naturally.

    The first version of this control concatenated tau2's internal fault descriptions
    ("Bad vpn. Data saver mode is on.") and scored BELOW the perturbed arm -- a stilted,
    machine-sounding request is its own handicap, so the comparison measured phrasing
    quality rather than misalignment. The control is now written by the same generator
    that writes the flawed queries, with the flaw instruction removed, so the only
    difference between arms is whether the intent is misrepresented.
    """
    problems = [adapter.fault_description(str(v)) for _s, _o, v in conditions]
    listing = "; ".join(problems)
    if llm is None:
        return ("Hi, I'm having trouble with my phone. " + listing +
                ". Could you fix all of that for me?")
    prompt = ("Write what a mobile-phone customer would say to support, in one or two "
              "sentences, first person, plain everyday voice. State ALL of these problems "
              "accurately and completely, leaving nothing out and adding nothing:\n  - "
              + "\n  - ".join(problems) +
              "\nOutput only the message.")
    try:
        return llm.complete(prompt, role="render").strip().strip('"')
    except Exception:
        return ("Hi, I'm having trouble with my phone. " + listing +
                ". Could you fix all of that for me?")


def to_sample_dict(r: dict, persona: str) -> dict:
    """The episode's preset: the agent sees `query` and nothing else."""
    mask = r.get("mask") or {}
    return {
        "sample_id": f"{r['tree']}__{r['strategy']}",
        "query": r["query"],
        "persona": persona,
        "strategy_ids": [r["strategy"]],
        "mask": mask,
        "signature": r.get("signature"),
        "hidden_intent": {"conditions": r.get("root_conditions") or [],
                          # ambiguous placements ARE hidden slots (see episodes_run_retail)
                          "hidden_slots": sorted(set(mask.get("withheld") or [])
                                                 | {m[0] for m in (mask.get("marked") or [])}
                                                 | {s[0] for s in (mask.get("substituted") or [])}
                                                 | {pl[0] for a in (mask.get("ambiguous") or [])
                                                    for pl in (a.get("placements") or []) if pl}
                                                 | {p["about"][0] for p in (mask.get("presupposed") or [])
                                                    if p.get("about")})},
    }


_local = threading.local()


def run_one(job):
    r, persona, arm, trees_by_id, cfg = job
    adapter, executor = Tau2TelecomAdapter(), Tau2Executor()
    llm = getattr(_local, "llm", None)
    if llm is None:
        llm = LLMClient({"llm": LLM_CFG})
        _local.llm = llm
    tree = trees_by_id[r["tree"]]
    # the episode's world must carry THIS intent's faults
    recipe = adapter.compile(tree.root.base, tree.root.conditions)
    spec = dict(tree.env_spec)
    spec["initialization"] = recipe["initialization"]
    tree = replace(tree, env_spec=spec)
    sample = to_sample_dict(r, persona)
    if cfg.get("_clean"):
        sample["query"] = clean_query(adapter, tree.root.conditions, llm)
        sample["strategy_ids"] = ["CLEAN_CONTROL"]
        sample["mask"] = {}
        sample["hidden_intent"]["hidden_slots"] = []
    t0 = time.time()
    try:
        # `tree=` -> `graph=` : renamed 2026-08-19. The deprecated module alias keeps the
        # IMPORTS working but not the keyword names, and both of these are keyword-only
        # (Episode.__init__(self, *, graph: Graph, ...)), so this runner raised TypeError on
        # its first episode -- dead on arrival rather than subtly wrong.
        agent = build_agent(arm, llm=llm, config=cfg, graph=tree, sample=sample,
                            adapter=adapter, executor=executor)
        ep = Episode(graph=tree, adapter=adapter, executor=executor, config=cfg, llm=llm,
                     agent=agent, persona_name=persona,
                     # STABLE + SEED-DEPENDENT (2026-08-19), matching episodes_run_retail.
                     # Was abs(hash(sample_id)), which (a) Python randomises per process, so
                     # runs were irreproducible, and (b) ignored --seed entirely, so seeds
                     # 1/2/3 were inert and three "repeats" differed only by LLM noise.
                     seed=int(hashlib.sha1(
                         f"{sample['sample_id']}:{cfg['_episode_seed']}".encode()
                     ).hexdigest()[:8], 16) % 10**6,
                     preset=sample)
        traj = ep.run()
        d = traj.to_dict() if hasattr(traj, "to_dict") else traj
        d["header"]["arm"] = arm
        d["header"]["sample_id"] = sample["sample_id"]
        d["header"]["strategy_id"] = r["strategy"]
        d["header"]["query"] = sample["query"]
        d["header"]["perturbation"] = {"hidden_slots": sample["hidden_intent"]["hidden_slots"],
                                       "governing_kind": (r.get("mask") or {}).get("mask_kind")}
        d["wall_s"] = round(time.time() - t0, 1)
        return d
    except Exception as e:
        return {"header": {"arm": arm, "sample_id": sample["sample_id"],
                           "strategy_id": r["strategy"], "persona": persona},
                "outcome": "ERROR", "error": f"{type(e).__name__}: {str(e)[:200]}",
                "turns": [], "wall_s": round(time.time() - t0, 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default="artifacts/telecom_samples_release.json")
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--arms", default="B0")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default="artifacts/telecom_episodes")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--max-faults", type=int, default=0, help="0 = no limit")
    ap.add_argument("--turns", type=int, default=0, help="override max_turns")
    ap.add_argument("--shift", action="store_true",
                    help="axis B: p_shift=1.0, one silent mid-episode intent shift")
    ap.add_argument("--clean", action="store_true",
                    help="CONTROL ARM: state the true intent plainly instead of the "
                         "perturbed query. Separates harness difficulty from perturbation "
                         "difficulty -- the reference row every reported number needs.")
    ap.add_argument("--resume", action="store_true",
                    help="skip episodes whose output file already exists in --out")
    a = ap.parse_args()

    rt = dict(RUNTIME)
    if a.turns:
        rt["max_turns"] = a.turns
    if a.shift:
        rt.update({"shift_scheduled": True,   # MASTER SWITCH -- episode.py returns early on
                   # `if not shift_scheduled` before any shift can fire, and it defaults
                   # FALSE, so p_shift alone fires NOTHING. Without this line `--shift` was
                   # accepted and completely inert: axis B could not happen in telecom at
                   # all. Matches the same fix already applied in episodes_run_airline.py.
                   "shift_schedule": "persona",  # the 2026-08-19 semantics: shifts fire at
                   # the persona's own patience thresholds rather than one episode-start coin
                   "p_shift": 1.0, "max_shifts": 1, "shift_roll_window": 6,
                   # every tau2 task "mutates", so the WebShop write-tree scope rule
                   # would disable axis B wholesale; here mutation is accounted for
                   # (telecom: on_shift injects into the live world; retail: shifts
                   # are forbidden after the agent commits a write)
                   "axis_b_on_write_trees": True,
                   "category_probs": {"REFINEMENT": 0.4, "PIVOT": 0.4, "SUBSTITUTION": 0.2},
                   "shift_category_mode": "skip",
                   "shift_roll_on_acts": True})
    cfg = {"runtime": rt, "llm": LLM_CFG, "_clean": bool(a.clean),
           # --seed reaches the episode seed through here; without it the flag was accepted
           # and discarded.
           "_episode_seed": a.seed}
    trees = {t.tree_id: t for t in storage.iter_trees(Path(WS + "/artifacts/trees/tau2_telecom"))}
    samples = load_samples(Path(WS + "/" + a.samples), 0, a.seed)
    if a.max_faults:
        samples = [r for r in samples
                   if len(r.get("root_conditions") or []) <= a.max_faults]
    samples = samples[:a.limit] if a.limit else samples
    personas = list(PERSONA_IDS)
    rng = random.Random(a.seed)
    jobs = []
    for arm in a.arms.split(","):
        for i, r in enumerate(samples):
            jobs.append((r, personas[i % len(personas)], arm, trees, cfg))
    print(f"{len(jobs)} episodes ({len(samples)} samples x {len(a.arms.split(','))} arms),"
          f" {a.workers} workers", flush=True)

    outdir = Path(WS + "/" + a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    if a.resume:
        def _done(job):
            r, p, arm, *_ = job
            return (outdir / f"{arm}__{r['tree']}__{r['strategy']}__{p}.json").exists()
        before = len(jobs)
        jobs = [j for j in jobs if not _done(j)]
        print(f"resume: {before - len(jobs)} already on disk, {len(jobs)} to run", flush=True)
    import concurrent.futures as cf
    results, t0 = [], time.time()
    with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
        for i, d in enumerate(ex.map(run_one, jobs), 1):
            results.append(d)
            (outdir / f"{d['header'].get('arm','?')}__{d['header'].get('sample_id','?')}"
                      f"__{d['header'].get('persona','?')}.json").write_text(
                json.dumps(d, indent=1)[:2_000_000])
            if i % 10 == 0:
                ok = sum(1 for x in results if x.get("outcome") == "SUCCESS")
                err = sum(1 for x in results if x.get("outcome") == "ERROR")
                print(f"  {i}/{len(jobs)} | success {ok} ({ok/i:.0%}) | errors {err} |"
                      f" {time.time()-t0:.0f}s", flush=True)

    by_arm = {}
    for arm in a.arms.split(","):
        sub = [r for r in results if r["header"].get("arm") == arm]
        outc = Counter(r.get("outcome") for r in sub)
        rows = []
        for r in sub:
            try:
                rows.append(grip.score(r))
            except Exception:
                pass
        by_arm[arm] = {
            "episodes": len(sub),
            "outcomes": dict(outc),
            "success_rate": round(outc.get("SUCCESS", 0) / max(len(sub), 1), 3),
            "mean_turns": round(sum(len(r.get("turns") or []) for r in sub) / max(len(sub), 1), 1),
            "grip": grip.report(rows) if rows else {},
        }
    summary = {"samples": len(samples), "wall_s": round(time.time() - t0, 1), "by_arm": by_arm}
    (Path(WS) / "artifacts" / "telecom_episode_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1)[:4000], flush=True)


if __name__ == "__main__":
    main()
