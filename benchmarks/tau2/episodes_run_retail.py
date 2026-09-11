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
from intent_tree.adapters.tau2_retail import Tau2RetailAdapter  # noqa: E402
from intent_tree.executors.tau2_inproc import Tau2Executor        # noqa: E402
from intent_tree.runtime import grip                              # noqa: E402
from intent_tree.runtime.agents import build_agent                # noqa: E402
from intent_tree.runtime.episode import Episode                   # noqa: E402
from intent_tree.runtime.llm import LLMClient                     # noqa: E402
from intent_tree.runtime.persona import PERSONA_IDS               # noqa: E402
from perturb_run import LLM_CFG                                   # noqa: E402

# RETRY BUDGET (2026-08-25, parity with episodes_run_airline.py). The inherited default
# is 8 attempts with a 60s cap -- it gives up after ~4 minutes and leaves an ERROR husk,
# which is exactly how a rate-limit storm corrupts a campaign. 40 attempts at a 45s cap
# rides out a ~30-minute burst. Models are untouched; only the retry knobs change.
LLM_CFG = dict(LLM_CFG)
LLM_CFG.update({"retry_attempts": 40, "retry_backoff_cap_s": 45})

def _load_runtime() -> dict:
    """The runtime block of pipeline/config/runtime.yaml -- the fork's config of record.

    REPLACED a hardcoded dict here (2026-08-19). That dict predated the pipeline change:
    it carried max_turns 30 / a stale hard question cap and declared NEITHER `shift_scheduled` NOR
    `shift_schedule`, so every run it produced silently used coin-mode axis B (or none)
    and could not exercise the persona shift schedule at all. Reading the yaml makes the
    runner reproduce the documented operating point (10 / 2 / 4 / 0, persona schedule,
    user_v2) instead of a stale copy of it.

    The yaml is RECONSTRUCTED, not recovered -- it reproduces none of the seven
    config_hash values the frozen retail runs used, so runs made with it are internally
    consistent and NOT comparable to those frozen numbers.
    """
    import yaml
    with open(os.path.join(WS, "pipeline", "config", "runtime.yaml")) as f:
        rt = dict((yaml.safe_load(f) or {}).get("runtime") or {})
    # ---- REQUIRED overrides, both recorded in every run_config.json ----
    # 1. Retail ground truths are all `statehash`, so _is_write_graph() is True for every
    #    graph. The yaml ships axis_b_on_write_graphs: false (a DBBench-era scope rule),
    #    which switches axis B off for the ENTIRE domain -- no shift would ever fire and
    #    the persona schedule would be unobservable. This experiment is about the shifts,
    #    so the scope rule has to be lifted for retail.
    rt["axis_b_on_write_graphs"] = True
    # 2. Not declared in the yaml; the runner has always ended a patience-exhausted
    #    episode by demanding a final answer rather than scoring it unattempted.
    rt.setdefault("forced_final", True)
    # 3+4. THE WRITE-DOMAIN HAZARD, measured 2026-08-19 on a 12-episode sanity cell
    #    (artifacts/persona2/sanity2_A0_dependent) and flagged in both runtime.yaml and
    #    docs/runtime-implementation-plan.md section 6.3:
    #      - every retail goal IS a write, so the agent mutates early: median turn 3,
    #        12/12 episodes;
    #      - the persona marks sit on the patience axis and are reached later: median
    #        turn 6, and in 0/12 episodes before the first write;
    #      - `shift_after_mutation: forbid` therefore bars EVERY scheduled shift.
    #    Measured consequence: 0 shifts in 12 episodes with the schedule provably live
    #    (header mode=persona, thresholds [7.92, 3.96]). Axis B is structurally dead on
    #    retail under `forbid`, so a persona sweep would compare four identical
    #    no-shift conditions.
    #
    #    So shifts are ALLOWED after mutation. That breaks exact-state grading -- the new
    #    node's ground truth was computed against the pristine shop, and the agent's
    #    pre-shift writes persist -- so acceptance moves to the NEW intent's golden
    #    actions (tau2's own ACTION reward component, already implemented by the retail
    #    adapter). Stale writes then cost patience and show up in the staleness metric
    #    instead of making the episode unwinnable by construction.
    rt["shift_after_mutation"] = "allow"
    rt["accept_actions_only"] = True
    # 5. MAX TURNS: the yaml ships 16, which TRUNCATES retail. Measured on 300 historical
    #    retail episodes that ran with room (cap 40): median 10, p90 20, p95 25, max 41 --
    #    18.3% used MORE than 16 turns. A cap of 16 therefore cuts off roughly a fifth of
    #    episodes mid-task, and the cut is invisible in cap-16 data itself (every episode
    #    is censored at the cap, so the tail cannot be observed from inside it). 30 covers
    #    ~p98 of the observed distribution; episodes that finish early are unaffected,
    #    since the median is 10.
    #    RAISED 30 -> 100 (2026-08-25). Those 300 episodes ran thinking-OFF, policy-OFF
    #    and without the caller-context line. Under the policy-on + thinking-on regime
    #    episodes are far longer, and the old cap truncated real work UNEQUALLY --
    #    measured B0 turns-exceeded: gpt-5.5 2.9%, qwen 7.6%, glm-4.7 18.4%. That
    #    penalises a model for verbosity rather than competence, the same class of
    #    unfairness as the parser defects. Airline uses 100 because the author required ZERO
    #    turns-exceeded; retail now matches that standard.
    #    REVERTED to 30 (2026-08-25, same night): raising it to 100 was based on the
    #    assumption that truncation loses legitimate work. Inspection of the truncated
    #    episodes disproves that -- glm-4.7's capped episodes are LOOPS (8 distinct
    #    commands across 100 calls; the same item fetched 18 times against a valid
    #    result). Cutting an unproductive loop at turn 30 discards nothing, while a cap
    #    of 100 lets it burn 3x the wall-clock (measured 0.75 eps/min vs ~11 for
    #    gpt-5.5). 30 also keeps this campaign comparable to every prior retail run.
    rt["max_turns"] = 30
    # 6. NO SEPARATE QUESTION CAP. Asking is governed by the PATIENCE economy alone: a
    #    question costs 2 against a 7-12 budget, and the affordability rule refuses a
    #    question the shopper can no longer pay for, forcing the final answer.
    return rt


RUNTIME = _load_runtime()


def load_samples(path: Path, limit: int, seed: int) -> list[dict]:
    data = json.loads(path.read_text())
    adm = [r for r in data["results"] if r["status"] == "admitted"]
    rng = random.Random(seed)
    rng.shuffle(adm)
    return adm[:limit] if limit else adm


def clean_query(adapter, conditions) -> str:
    parts = [f"{adapter.slot_phrase(s)}: {v}" for s, _o, v in sorted(conditions)]
    return ("Hi, I need a change made to my order. " + "; ".join(parts) +
            ". Could you take care of that?")


def caller_context(adapter, r: dict, tree) -> str:
    """A CRM-style screen-pop the agent sees before the customer speaks.

    PORTED FROM AIRLINE 2026-08-24 after the policy-on autopsy. The retail policy
    (now actually delivered -- register O8) tells the agent to authenticate the
    customer with find_user_id_by_email / find_user_id_by_name_zip. But the
    customer's identity is ENVIRONMENT data, not a condition of the hidden intent,
    so the simulated user has nothing true to say and INVENTS an address; the
    lookup fails, the agent asks again, and patience drains. Measured on the first
    policy-on backbone cell: 76% of tool observations were "user not found", and
    Success collapsed to 2%. tau2 itself hands the caller's id to the user as
    `known_info`; this restores it as call metadata rather than by editing the
    query or weakening the policy.

    Withheld ONLY when user_id is itself a hidden slot, so the context can never
    reveal something the perturbation deliberately concealed.
    """
    mask = r.get("mask") or {}
    hidden = (set(mask.get("withheld") or [])
              | {m[0] for m in (mask.get("marked") or [])}
              | {x[0] for x in (mask.get("substituted") or [])}
              | {pl[0] for amb in (mask.get("ambiguous") or [])
                 for pl in (amb.get("placements") or []) if pl}
              | {p["about"][0] for p in (mask.get("presupposed") or []) if p.get("about")})
    if any(str(s).endswith(":user_id") for s in hidden):
        return ""
    _oid, uid = adapter._order_of(tuple(tree.root.conditions))
    return f"[Call context: caller is authenticated as user id {uid}.] " if uid else ""


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
                          # ambiguous placements ARE hidden slots: the agent must pin down
                          # which slot the ambiguous surface belongs to (lexical/syntactic
                          # masks store placements, not marked -- measured 2026-08-14:
                          # without this, those episodes scored NO_HIDDEN_INTENT)
                          "hidden_slots": sorted(set(mask.get("withheld") or [])
                                                 | {m[0] for m in (mask.get("marked") or [])}
                                                 | {s[0] for s in (mask.get("substituted") or [])}
                                                 | {pl[0] for a in (mask.get("ambiguous") or [])
                                                    for pl in (a.get("placements") or []) if pl}
                                                 # an FP's false claim contests the slot it is
                                                 # ABOUT: confirming the true choice is the
                                                 # recovery target (all tau2 FP conditions are
                                                 # non-compilable, so `condition` is never set)
                                                 | {p["about"][0] for p in (mask.get("presupposed") or [])
                                                    if p.get("about")})},
    }


_local = threading.local()


def run_one(job):
    r, persona, arm, trees_by_id, cfg = job
    adapter, executor = Tau2RetailAdapter(), Tau2Executor()
    adapter.accept_actions_only = bool((cfg.get("runtime") or {}).get("accept_actions_only"))
    llm = getattr(_local, "llm", None)
    if llm is None:
        llm = LLMClient({"llm": LLM_CFG})
        _local.llm = llm
    tree = trees_by_id.get(r["tree"]) or trees_by_id[cfg["_remap"][r["tree"]]]
    # the episode's world must carry THIS intent's faults
    # retail: one shared pristine shop; no per-intent initialization
    sample = to_sample_dict(r, persona)
    if cfg.get("_clean"):
        sample["query"] = clean_query(adapter, tree.root.conditions)
        sample["strategy_ids"] = ["CLEAN_CONTROL"]
        sample["mask"] = {}
        sample["hidden_intent"]["hidden_slots"] = []
    if bool((cfg.get("runtime") or {}).get("caller_context", True)):
        sample["query"] = caller_context(adapter, r, tree) + sample["query"]
    t0 = time.time()
    try:
        # `tree=` -> `graph=` : renamed 2026-08-19; the deprecated module alias keeps the
        # IMPORTS working but not the keyword names, so this call site had to move too.
        agent = build_agent(arm, llm=llm, config=cfg, graph=tree, sample=sample,
                            adapter=adapter, executor=executor)
        ep = Episode(graph=tree, adapter=adapter, executor=executor, config=cfg, llm=llm,
                     agent=agent, persona_name=persona,
                     # STABLE + SEED-DEPENDENT (2026-08-19). Was abs(hash(sample_id)),
                     # which (a) Python randomises per process, so runs were not
                     # reproducible, and (b) ignored --seed entirely, so the three
                     # "repeats" differed only by LLM nondeterminism and seeds 1/2/3 were
                     # inert. sha1 over (sample_id, seed) is deterministic given both.
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
        # true values for the any-channel discovery ruling (2026-08-21): lets the scorer
        # credit world observations without a sample join. Agent never reads trajectories.
        d["header"]["hidden_intent"] = {"conditions": sample["hidden_intent"]["conditions"]}
        if "_lessonbook_gen" in cfg:
            # deterministic visibility (design invariant 7): every trajectory records
            # exactly which lesson snapshot its agent saw
            d["header"]["lessonbook_gen"] = cfg["_lessonbook_gen"]
            d["header"]["lessonbook_sha"] = cfg.get("_lessonbook_sha") or ""
        d["wall_s"] = round(time.time() - t0, 1)
        return d
    except Exception as e:
        return {"header": {"arm": arm, "sample_id": sample["sample_id"],
                           "strategy_id": r["strategy"], "persona": persona},
                "outcome": "ERROR", "error": f"{type(e).__name__}: {str(e)[:200]}",
                "turns": [], "wall_s": round(time.time() - t0, 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default="artifacts/retail_samples_release.json")
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--arms", default="B0")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default="artifacts/retail_episodes")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--max-faults", type=int, default=0, help="0 = no limit")
    ap.add_argument("--turns", type=int, default=0, help="override max_turns")
    ap.add_argument("--shift", action="store_true",
                    help="axis B: p_shift=1.0, one silent mid-episode intent shift")
    ap.add_argument("--clean", action="store_true",
                    help="CONTROL ARM: state the true intent plainly instead of the "
                         "perturbed query. Separates harness difficulty from perturbation "
                         "difficulty -- the reference row every reported number needs.")
    ap.add_argument("--all-personas", action="store_true",
                    help="run every sample under EVERY persona (full cross product) "
                         "instead of the default round-robin assignment")
    ap.add_argument("--backbone", default="deepseek",
                    help="D27 agent backbone (backbones.py); user side never changes")
    ap.add_argument("--thinking", action="store_true",
                    help="D28: let the AGENT model reason (symmetric across backbones). "
                         "Stamped in run_config; thinking-on and thinking-off are "
                         "SEPARATE regimes and must not share a table.")
    ap.add_argument("--policy", action="store_true",
                    help="give the agent the binding store policy. DEFAULT OFF from "
                         "2026-08-26: acceptance is golden WRITE actions only, so the "
                         "policy's procedural steps cost patience and earn no credit "
                         "(measured -13.8 Success paired). policy_chars is stamped 0 "
                         "when off, and policy-on/off are SEPARATE regimes.")
    ap.add_argument("--no-shift", action="store_true",
                    help="ABLATION: disable axis B completely (no scheduled "
                         "shift, no trigger, no coin). Stamps shift_schedule "
                         "mode=off. Pairs against the with-shift reference run.")
    ap.add_argument("--shift-fire-prob", type=float, default=None,
                    help="fire coin (register D2): crossing consumes the threshold, the "
                         "intent moves only with this probability; fates pre-drawn per "
                         "sample, scheduled shifts only. Unset = 1.0 = regime R2. Set "
                         "per CAMPAIGN, never mid-chain.")
    ap.add_argument("--show-patience", action="store_true",
                    help="publish the patience meter to the agent (A2/A3's mechanism)")
    ap.add_argument("--askbook", default="",
                    help="RETIRED -- raw-example askbooks are quarantined; passing this "
                         "is an error (docs/a3-self-evolving-method-review.md section 9)")
    ap.add_argument("--exact-state", action="store_true",
                    help="BRIDGING DIAGNOSTIC ONLY: grade by exact state-hash (the "
                         "pre-2026-08-19 regime) instead of golden actions. Unsound with "
                         "post-mutation shifts -- for attributing the era gap, never for "
                         "reported comparisons.")
    ap.add_argument("--reflector", default="agent", choices=["agent", "user"],
                    help="A3: which model runs reflection (agent = self-reflection; "
                         "user = the stronger user-side model, v7)")
    ap.add_argument("--lessonbook-gate", default="reject", choices=["reject", "write_or_reject"],
                    help="A3: when the rendered book becomes visible (v5: after first "
                         "rejected submission; v6: also after the first write)")
    ap.add_argument("--gen-warmup", default="",
                    help="A3: comma list of SMALLER first-generation sizes (e.g. '10,15') "
                         "so the first reflection happens early and the empty-book window "
                         "shrinks; later generations use --gen-size. Part of the snapshot "
                         "binding: changing it is a different run, not a resume.")
    ap.add_argument("--gen-size", type=int, default=25,
                    help="A3 online evolution: episodes per generation between "
                         "reflections (0 = static scaffold-only A3, no lessonbook). "
                         "Default 25 so a default-limit run still reflects at least once.")
    ap.add_argument("--persona", default="",
                    help="fix ONE persona for every episode (persona experiments); "
                         "default rotates all five round-robin")
    ap.add_argument("--max-rps", type=float, default=0.0,
                    help="override the per-process request pacing (0 = use config)")
    ap.add_argument("--endpoint", default="pool", choices=["a", "b", "pool"],
                    help="which API identity this run uses. 'a' = public model names, "
                         "'b' = the dedicated endpoints, 'pool' = round-robin both. "
                         "Two runs pinned to a and b respectively get one independent "
                         "TPM allowance each and do not contend.")
    ap.add_argument("--cache-tag", default="",
                    help="isolate this run's response cache under its own namespace; "
                         "required for honest throughput probes, which otherwise replay "
                         "the previous width's cached answers")
    ap.add_argument("--resume", action="store_true",
                    help="skip episodes whose output file already exists in --out "
                         "(restart a killed run without redoing finished episodes)")
    a = ap.parse_args()

    rt = dict(RUNTIME)
    # A3's playbook is domain knowledge, so it is built and rendered under THIS dataset's
    # ask-vocabulary. Without this the agent inherits WebShop's taxonomy, which matches
    # nothing in retail and prints advice about "purpose" and "feature" (ruling 2026-08-20).
    rt["askbook_profile"] = "tau2_retail"
    rt["lessonbook_gate"] = a.lessonbook_gate
    if a.exact_state:
        rt["accept_actions_only"] = False
    if a.show_patience:
        rt["show_patience"] = True
    if a.askbook:
        raise SystemExit(
            "--askbook is retired: raw-example books are quarantined and cannot back A3 "
            "(docs/a3-self-evolving-method-review.md section 9). A3 now evolves an online "
            "lessonbook automatically -- see --gen-size and docs/a3v2-lessonbook-design.md.")
    if a.turns:
        rt["max_turns"] = a.turns
    if a.shift_fire_prob is not None:
        rt["shift_fire_prob"] = a.shift_fire_prob
    if a.shift:
        rt.update({"p_shift": 1.0, "max_shifts": 1, "shift_roll_window": 6,
                   # every tau2 task "mutates", so the WebShop write-tree scope rule
                   # would disable axis B wholesale; here mutation is accounted for
                   # (telecom: on_shift injects into the live world; retail: shifts
                   # are forbidden after the agent commits a write)
                   "axis_b_on_write_trees": True,
                   "category_probs": {"REFINEMENT": 0.25, "RELAXATION": 0.25, "SUBSTITUTION": 0.25, "PIVOT": 0.25},
                   "shift_category_mode": "skip",
                   "shift_roll_on_acts": True,
                   "accept_actions_only": True})
    # PER-SEED CACHE NAMESPACE. The response cache is keyed on the prompt, so with one
    # shared cache the three repeats would return byte-identical model output wherever
    # their prompts coincided -- understating exactly the run-to-run spread this
    # experiment is asked to report. Each seed gets its own namespace: repeats are
    # independent, while re-running or --resume-ing a cell still reuses its own cache.
    LLM_CFG["cache_dir"] = os.path.join(WS, "artifacts", "llm_cache",
                                        a.cache_tag or f"seed{a.seed}")
    # Pin this run to one API identity, or leave it rotating over both. Cache keys use the
    # CANONICAL model name either way, so a cell run on endpoint b still shares cache
    # entries with one run on endpoint a -- the identity is transport, never semantics.
    if a.max_rps:
        LLM_CFG["max_rps"] = a.max_rps
    if a.endpoint != "pool":
        i = 0 if a.endpoint == "a" else 1
        LLM_CFG["model_pool"] = [LLM_CFG["model_pool"][i]]
        LLM_CFG["agent_model_pool"] = [LLM_CFG["agent_model_pool"][i]]
    # D27 BACKBONE OVERLAY (agent side only; run-record stamp per protocol rule 3)
    if a.backbone != "deepseek":
        import backbones as _bb
        globals()["LLM_CFG"] = _bb.apply(LLM_CFG, a.backbone)
    rt["backbone"] = a.backbone
    if a.thinking:
        LLM_CFG["disable_thinking"] = False
    rt["thinking"] = bool(a.thinking)
    # NO-SHIFT ABLATION (ruling 2026-08-27). Axis B off entirely: no scheduled shift, no
    # trigger, no coin. This measures how much the moving intent costs, against the
    # with-shift reference campaigns (retail r3_rational / airline r3a_rational) which are
    # identical in every other respect. `shift_scheduled` gates the persona schedule AND
    # the trigger path; `p_shift` gates the legacy coin mode -- both must be off, or the
    # ablation silently keeps one channel and the "no-shift" number is not one.
    # Episodes stamp shift_schedule.mode == "off", which the review asserts.
    if a.no_shift:
        rt["shift_scheduled"] = False
        rt["p_shift"] = 0.0
        rt["axis_b_on_write_graphs"] = False
    # O8 AUDIT STAMP (2026-08-23): the empty-policy bug was invisible for four days
    # because nothing recorded what the agents saw. Every run now stamps the policy
    # block's length; a 0 here means POLICY-OFF and must be a choice, not an accident.
    from intent_graph.adapters.tau2_retail import Tau2RetailAdapter as _PolA
    # POLICY-OFF is the retail default from 2026-08-26 (the author). See the note on
    # Tau2RetailAdapter.include_policy: we grade golden WRITE actions only, so the
    # policy's procedural steps cost patience and earn nothing. --policy restores it.
    _PolA.include_policy = bool(a.policy)
    rt["policy_chars"] = len(_PolA().agent_policy()) if a.policy else 0
    # caller_context exists ONLY to break the policy-induced auth deadlock (policy-on +
    # no identity measured 76% "user not found"). With the policy off the agent never
    # authenticates, and the line is a prompt difference the 11-arm campaign did not
    # have, so it is switched off with the policy unless asked for explicitly.
    if not a.policy:
        rt["caller_context"] = False
    cfg = {"runtime": rt, "llm": LLM_CFG, "_clean": bool(a.clean), "_remap": {},
           "_episode_seed": a.seed}
    trees = {t.tree_id: t for t in storage.iter_trees(Path(WS + "/artifacts/trees/tau2_retail"))}
    # Samples were generated against an earlier tree corpus; tree ids include the child
    # set, so rebuilding trees (e.g. adding refinement edges) changes every id even though
    # the ROOTS are identical. Remap by root source record: same seed -> same tree.
    by_record = {t.root.source_record: t.tree_id for t in trees.values() if t.root.source_record}
    old_dirs = [Path(WS + "/artifacts/trees/tau2_retail_v1_norefine"),
                Path(WS + "/artifacts/trees/tau2_retail_v2_norelax")]
    remap = {}
    # the v1/v2 tree dirs were lost to the 2026-08-16 tmp purge; the derived map was
    # released with the dataset and is authoritative for historical sample ids
    saved = Path(WS + "/artifacts/retail_tree_id_remap.json")
    if saved.exists():
        remap.update(json.loads(saved.read_text()))
    for od in old_dirs:
        if not od.exists():
            continue
        for t in storage.iter_trees(od):
            new_id = by_record.get(t.root.source_record)
            if new_id:
                remap[t.tree_id] = new_id
    if remap:
        print(f"remapped {len(remap)} historical tree ids onto the current corpus", flush=True)
    cfg["_remap"] = remap
    samples = load_samples(Path(WS + "/" + a.samples), 0, a.seed)
    if a.max_faults:
        samples = [r for r in samples
                   if len(r.get("root_conditions") or []) <= a.max_faults]
    samples = samples[:a.limit] if a.limit else samples
    personas = [a.persona] if a.persona else list(PERSONA_IDS)
    rng = random.Random(a.seed)
    jobs = []
    for arm in a.arms.split(","):
        for i, r in enumerate(samples):
            if a.all_personas:
                for p in personas:
                    jobs.append((r, p, arm, trees, cfg))
            else:
                jobs.append((r, personas[i % len(personas)], arm, trees, cfg))
    per = f" x {len(personas)} personas" if a.all_personas else ""
    print(f"{len(jobs)} episodes ({len(samples)} samples x {len(a.arms.split(','))} arms"
          f"{per}), {a.workers} workers", flush=True)

    outdir = Path(WS + "/" + a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    # Run provenance WITHOUT any model identity, by instruction.
    _rc = {
        "argv": sys.argv[1:],
        "arm": a.arms, "persona": a.persona, "seed": a.seed,
        "workers": a.workers, "limit": a.limit, "samples": a.samples,
        # MODEL IDENTITIES ARE DELIBERATELY NOT RECORDED (instruction 2026-08-19).
        "max_output_tokens": LLM_CFG.get("max_output_tokens"),
        "lessonbook": {"gen_size": a.gen_size, "gen_warmup": a.gen_warmup,
                       "format": "lessonbook_v2",   # v4 harness: outcome brake + dedupe
                       "design": "docs/a3v2-lessonbook-design.md"},
        "runtime": rt,
        "config_source": "pipeline/config/runtime.yaml (RECONSTRUCTED; not comparable "
                         "to the seven frozen retail config_hash values)",
        "required_overrides": {
            "axis_b_on_write_graphs": "true -- retail is 100% write-type; the yaml's "
                                      "false disables axis B for the whole domain",
            "forced_final": "true -- not declared in the yaml"},
    }
    _rc_path = outdir / "run_config.json"
    if a.resume and _rc_path.exists():
        # A RESUME MAY NOT CHANGE THE METHOD (2026-08-20 review): verify every flag that
        # defines the partition or the science, and never clobber the original record.
        old = json.loads(_rc_path.read_text())
        for k in ("arm", "persona", "seed", "limit", "samples"):
            if old.get(k) != _rc[k]:
                raise SystemExit(f"--resume with changed {k!r}: run_config has "
                                 f"{old.get(k)!r}, this invocation {_rc[k]!r}")
        _lb_old = old.get("lessonbook") or {}
        if _lb_old.get("gen_size") != a.gen_size or _lb_old.get("gen_warmup", "") != a.gen_warmup:
            raise SystemExit("--resume with a different --gen-size/--gen-warmup changes "
                             "the generation partition; that is a different method, not a "
                             "resume")
        (outdir / "run_config.resume.json").write_text(json.dumps(_rc, indent=1))
    else:
        _rc_path.write_text(json.dumps(_rc, indent=1))

    def _ep_path(job):
        r, p, arm, *_ = job
        return outdir / f"{arm}__{r['tree']}__{r['strategy']}__{p}.json"

    def _not_done(js, label=""):
        if not a.resume:
            return js
        left = [j for j in js if not _ep_path(j).exists()]
        print(f"resume{label}: {len(js) - len(left)} already on disk, {len(left)} to run",
              flush=True)
        return left

    import concurrent.futures as cf
    results, t0 = [], time.time()

    def run_pool(pool_jobs, label=""):
        with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
            for i, d in enumerate(ex.map(run_one, pool_jobs), 1):
                results.append(d)
                fp = (outdir / f"{d['header'].get('arm','?')}__{d['header'].get('sample_id','?')}"
                               f"__{d['header'].get('persona','?')}.json")
                # atomic: --resume treats existence as done, so a torn/interrupted write
                # must never leave a plausible-looking file behind (2026-08-20 review)
                tmp = fp.with_suffix(f".tmp{os.getpid()}")
                tmp.write_text(json.dumps(d, indent=1)[:2_000_000])
                tmp.replace(fp)
                if i % 10 == 0:
                    ok = sum(1 for x in results if x.get("outcome") == "SUCCESS")
                    err = sum(1 for x in results if x.get("outcome") == "ERROR")
                    print(f"  {label}{i}/{len(pool_jobs)} | success {ok} ({ok/i:.0%}) | "
                          f"errors {err} | {time.time()-t0:.0f}s", flush=True)

    # ---- A3 runs in deterministic GENERATIONS; every other arm runs flat ----------------
    # Online self-evolution (docs/a3v2-lessonbook-design.md): the A3 job list is
    # partitioned into consecutive chunks of --gen-size; all episodes of a chunk see the
    # SAME immutable lesson snapshot; between chunks the agent model reflects on the
    # chunk's episodes (observable projection, redacted) on the MAIN thread and the next
    # snapshot is written atomically. Parallel cells cannot collide: the snapshot chain
    # lives inside this run's own --out and is bound to (profile, run_id, arm, persona,
    # seed, generation), checked fail-closed on both ends.
    a3_jobs = [j for j in jobs if j[2].startswith("A3")] if a.gen_size > 0 else []
    flat_jobs = [j for j in jobs if not (a.gen_size > 0 and j[2].startswith("A3"))]
    if not a3_jobs and a.gen_size == 0 and any(j[2] == "A3" for j in jobs):
        # scaffold-only A3 is a DIFFERENT declared variant; stamp it so no table can
        # silently mix it with evolving A3 (2026-08-20 review)
        cfg["_lessonbook_gen"] = -1
        cfg["_lessonbook_sha"] = ""
    if flat_jobs:
        run_pool(_not_done(flat_jobs), label="")

    if a3_jobs:
        from intent_graph.runtime import lessonbook
        if not a.persona:
            raise SystemExit("A3 online evolution requires --persona: the experimental "
                             "unit is benchmark x arm x persona x seed, and a round-robin "
                             "cell would share one book across five personas "
                             "(docs/a3v2-lessonbook-design.md checklist 2)")
        warm = [int(x) for x in a.gen_warmup.split(",") if x.strip()]
        sizes = warm + [a.gen_size] * (len(a3_jobs) // max(a.gen_size, 1) + 2)
        chunks, _i, _k = [], 0, 0
        while _i < len(a3_jobs):
            chunks.append(a3_jobs[_i:_i + max(sizes[_k], 1)])
            _i += max(sizes[_k], 1)
            _k += 1
        lessons_dir = outdir / "lessonbook"
        # run_id carries the round directory too: basename alone collides across rounds
        # (a3v2_r1/A3_dependent vs a3v2_r2/A3_dependent) and would defeat the binding
        samples_sha = hashlib.sha1(
            Path(WS + "/" + a.samples).read_bytes()
            + f":{a.limit}:{a.seed}:{a.gen_warmup}".encode()).hexdigest()[:12]
        binding = {"profile": rt["askbook_profile"],
                   "run_id": f"{outdir.parent.name}/{outdir.name}", "arm": a3_jobs[0][2],
                   "persona": a.persona, "seed": a.seed,
                   "gen_size": a.gen_size, "samples_sha": samples_sha}
        # tool names feed the no-executable-answers validator; the entity lexicon bans
        # names/cities/streets/products from the benchmark's own DB (review: regex classes
        # alone cannot reject a person or product name)
        import re as _re
        tool_names = tuple(sorted(set(
            _re.findall(r"\b([a-z_]{6,})\(", Tau2RetailAdapter().tool_manual()))))
        lexicon = lessonbook.build_entity_lexicon(
            os.environ["TAU2_DATA"] + "/domains/retail/db.json")
        llm_main = LLMClient({"llm": LLM_CFG})
        snap, all_grams = None, set()
        print(f"[lessonbook] {len(a3_jobs)} A3 episodes in {len(chunks)} generations of "
              f"<= {a.gen_size}", flush=True)
        for g, chunk in enumerate(chunks):
            rt_gen = dict(rt)
            if snap is not None:
                rt_gen["lessonbook_path"] = str(lessons_dir / f"gen_{g:03d}.json")
                rt_gen["lessonbook_binding"] = {**binding, "generation": g}
            cfg_gen = dict(cfg)
            cfg_gen["runtime"] = rt_gen
            cfg_gen["_lessonbook_gen"] = g
            cfg_gen["_lessonbook_sha"] = (snap or {}).get("sha") or ""
            chunk = [(r, p, arm, trees, cfg_gen) for (r, p, arm, _t, _c) in chunk]
            run_pool(_not_done(chunk, label=f" gen{g}"), label=f"gen{g} ")
            # digests come FROM DISK so resumed and fresh episodes are treated identically
            digs = []
            for job in chunk:
                f = _ep_path(job)
                if f.exists():
                    try:
                        traj = json.loads(f.read_text())
                    except (OSError, json.JSONDecodeError):
                        continue
                    if traj.get("outcome") not in (None, "ERROR"):
                        digs.append(lessonbook.episode_digest(traj))
            if g == len(chunks) - 1:
                break                            # nothing left to learn for
            next_path = lessons_dir / f"gen_{g + 1:03d}.json"
            if a.resume and next_path.exists():
                snap = lessonbook.check_binding(
                    json.loads(next_path.read_text()), **{**binding, "generation": g + 1})
            else:
                prev_seen = (snap or {}).get("n_episodes_seen", 0)
                if digs:
                    res = lessonbook.summarize(
                        llm_main, prior=(snap or {}).get("lessons") or [], digests=digs,
                        profile=binding["profile"], tool_names=tool_names,
                        extra_grams=all_grams, lexicon=lexicon,
                        # v7: which model reflects (agent = self-reflection; user = the
                        # stronger user-side model, same allocation, cache-safe by role)
                        reflector_role=("select" if a.reflector == "user" else "agent_audit"),
                        # v4 outcome brake: the chain of per-generation success rates,
                        # so a losing generation freezes and shrinks lesson weights
                        prior_gen_success=(snap or {}).get("gen_success") or [])
                else:                            # a fully-husked chunk teaches nothing
                    res = {"lessons": (snap or {}).get("lessons") or [],
                           "dropped_by_validator": {}, "parse_error": None,
                           "gen_success": (snap or {}).get("gen_success") or [],
                           "outcome_brake": False}
                snap = lessonbook.make_snapshot(
                    lessons_result=res, profile=binding["profile"],
                    run_id=binding["run_id"], arm=binding["arm"], persona=binding["persona"],
                    seed=a.seed, generation=g + 1, parent=snap,
                    source_episode_ids=[d["sample_id"] for d in digs],
                    n_seen=prev_seen + len(digs),
                    gen_size=a.gen_size, samples_sha=binding["samples_sha"])
                lessonbook.save_snapshot(snap, next_path)
            all_grams |= lessonbook.source_ngrams(digs)
            print(f"  [lessonbook] gen {g} distilled -> {len(snap['lessons'])} lessons "
                  f"(dropped {sum((snap.get('dropped_by_validator') or {}).values())}, "
                  f"seen {snap['n_episodes_seen']} eps, sha {snap['sha']})", flush=True)

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
    # into the run's own directory: the old global artifacts/retail_episode_summary.json
    # was clobbered by whichever concurrent cell finished last (2026-08-20 review)
    (outdir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1)[:4000], flush=True)


if __name__ == "__main__":
    main()
