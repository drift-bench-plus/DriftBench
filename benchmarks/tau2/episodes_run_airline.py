"""Run interaction episodes (B0 and reference arms) over the AIRLINE sample set.

Airline follows the retail shift semantics: one shared pristine database, shifts (when
--shift is on) fire silently and only before the agent's first committed write, and
acceptance after a shift is golden-actions-only. GRIP wiring is the retail-adapted form:
hidden_slots is the union of withheld/marked/substituted slots, ambiguity PLACEMENTS and
the FP-contested slot, and the user simulator speaks through the adapter's
slot_phrase/spoken_value hooks (the 2026-08-14 instrument-audit lessons).
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
import airline_compat  # noqa: F401  (tree->graph rename shim)
from loguru import logger as _loguru          # noqa: E402
_loguru.remove()

from intent_tree import storage                                   # noqa: E402
from intent_tree.adapters.tau2_airline import Tau2AirlineAdapter  # noqa: E402
from intent_tree.executors.tau2_inproc import Tau2Executor        # noqa: E402
from intent_tree.runtime import grip                              # noqa: E402
from intent_tree.runtime.agents import build_agent                # noqa: E402
from intent_tree.runtime.episode import Episode                   # noqa: E402
from intent_tree.runtime.llm import LLMClient                     # noqa: E402
from intent_tree.runtime.persona import PERSONA_IDS               # noqa: E402
# MODEL PARITY WITH RETAIL (ruling 2026-08-20): airline episodes must run the SAME
# models as retail, not the WebShop allocation. episodes_run_retail.py imports
# LLM_CFG from perturb_run.py, so importing from the same place keeps the two
# domains locked together permanently instead of drifting again -- currently
# user sim doubao-seed-2-0-pro-260215, agent under test deepseek-v4-pro-260425.
# NOTE: the airline DATASET was generated on 2-1-pro/2-1-turbo (see
# perturb_run_airline.py); that difference is a datasheet item, not an episode one.
from perturb_run import LLM_CFG as _RETAIL_LLM_CFG                    # noqa: E402

# Retry policy (ruling 2026-08-20): the retail campaign runs on the same endpoints, so
# 429s are contention, not a real failure -- WAIT THEM OUT rather than killing the
# episode. The default 8 attempts with a 60s cap gives up after ~4 minutes and leaves
# an ERROR husk (5 of the first 122 episodes at width 32). 40 attempts rides out a
# ~30-minute burst from the other session. Models are NOT touched: this is a copy of
# retail's config with only the retry knobs changed, so model parity is preserved.
LLM_CFG = dict(_RETAIL_LLM_CFG)
LLM_CFG.update({"retry_attempts": 40, "retry_backoff_cap_s": 45})

# ENDPOINT ALLOCATION (updated 2026-08-22, register D23/D24): airline inherits
# perturb_run's model pools VERBATIM (canonical name + provisioned endpoint per
# model, wire-rotated; cache keys canonical only), keeping the two domains locked
# together. Only the retry knobs above differ, documented there.
assert LLM_CFG["model"] == "doubao-seed-2-0-pro-260215", "airline user model off-allocation"
assert LLM_CFG["agent_model"] == "deepseek-v4-pro-260425", "airline agent model off-allocation"
assert LLM_CFG["model_pool"][0] == "doubao-seed-2-0-pro-260215"
assert LLM_CFG["agent_model_pool"][0] == "deepseek-v4-pro-260425"

RUNTIME = {
    # AIRLINE budget, measured 2026-08-20 on the retail model pair (deepseek agent):
    # 36 probe episodes at a cap of 150 gave median 7 turns, p95 21, single max 90.
    # A cap of 80 still clipped 1/36; 100 clipped none. the author requires ZERO
    # TURNS_EXCEEDED, so 100 -- and it is nearly free, since the median episode uses
    # 7 turns and the cap binds only on outliers.
    "max_turns": 100,
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


def clean_query(adapter, conditions) -> str:
    """State the true intent the way a passenger actually would.

    The generic slot-phrase enumeration invited a misreading measured in the first clean
    smoke: a cabin-only upgrade lists its (unchanged) flights, the phrase "which flights
    to take (for the flight change)" reads as a flight change, and the agent -- correctly
    reading the basic-economy policy -- refuses a legal request. A cabin-only update is
    therefore phrased as "keep the current flights"; everything else stays a plain
    per-request enumeration of the same condition values."""
    recipe = adapter.compile({"reads": []}, tuple(tuple(c) for c in conditions))
    parts = []
    # a recipe is an ORDERED sequence; track each reservation's flights as the earlier
    # steps leave them, so "keeping its current flights" means current AT THAT STEP
    live: dict = {}
    for a in recipe.get("actions") or []:
        name, args = a["name"], a.get("arguments") or {}
        rid = args.get("reservation_id")
        if name == "update_reservation_flights" and rid:
            res = adapter._reservation(rid)
            cur = live.get(str(rid), sorted((f.get("flight_number"), f.get("date"))
                                            for f in res.get("flights") or []))
            live[str(rid)] = sorted((f.get("flight_number"), f.get("date"))
                                    for f in args.get("flights") or [])
            new = sorted((f.get("flight_number"), f.get("date"))
                         for f in args.get("flights") or [])
            cab = str(args.get("cabin") or "").replace("_", " ")
            if cur == new:
                parts.append(f"change reservation {rid} to {cab} cabin, keeping its "
                             f"current flights exactly as they are, paid with "
                             f"{args.get('payment_id')}")
            else:
                fl = ", ".join(f"flight {f.get('flight_number')} on {f.get('date')}"
                               for f in args.get("flights") or [])
                parts.append(f"change reservation {rid} to {cab} cabin on {fl}, "
                             f"paid with {args.get('payment_id')}")
        elif name == "cancel_reservation":
            parts.append(f"cancel reservation {rid}")
        elif name == "update_reservation_baggages":
            parts.append(f"set reservation {rid} to {args.get('total_baggages')} checked "
                         f"bags total, paid with {args.get('payment_id')}")
        elif name == "update_reservation_passengers":
            pax = ", ".join(f"{p.get('first_name')} {p.get('last_name')} "
                            f"(born {p.get('dob')})" for p in args.get("passengers") or [])
            parts.append(f"set the passengers on reservation {rid} to exactly: {pax}")
        elif name == "book_reservation":
            fl = ", ".join(f"flight {f.get('flight_number')} on {f.get('date')}"
                           for f in args.get("flights") or [])
            pax = ", ".join(f"{p.get('first_name')} {p.get('last_name')} "
                            f"(born {p.get('dob')})" for p in args.get("passengers") or [])
            pay = " and ".join(str(p.get("payment_id")) for p in args.get("payment_methods") or [])
            ins = "with" if args.get("insurance") == "yes" else "without"
            parts.append(
                f"book a {str(args.get('flight_type') or '').replace('_', ' ')} "
                f"{str(args.get('cabin') or '').replace('_', ' ')} trip from "
                f"{args.get('origin')} to {args.get('destination')} on {fl} for {pax}, "
                f"{args.get('total_baggages') or 0} checked bags total, {ins} travel "
                f"insurance, under user id {args.get('user_id')}, paid with {pay}")
        else:
            parts.append(name.replace("_", " ") + " " + json.dumps(args))
    if len(parts) > 1:
        body = "; then ".join(parts)
        return ("Hi, here is exactly what I need for my travel plans, in this order: "
                + body + ". Please take care of all of it, in that order.")
    return ("Hi, here is exactly what I need for my travel plans: " + parts[0]
            + ". Please take care of it.")


def caller_context(adapter, r: dict, tree) -> str:
    """A CRM-style screen-pop the agent sees before the caller speaks.

    tau2's airline policy requires the caller's user id four separate times, but user_id
    is not an argument of cancel/update, so it is not a condition -- and B0 has no ask
    channel, so it can neither look it up nor request it, and every such episode
    deadlocks (measured 2026-08-19: 10 of 14 dead-tree failures). tau2 itself hands the
    caller's id to the user as `known_info`; this restores it as call metadata rather
    than by editing the query or the policy.

    Withheld ONLY when user_id is itself a hidden slot (book tasks can hide it), so the
    context can never reveal something the perturbation deliberately concealed.
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
    uid = adapter._seed_user(tuple(tree.root.conditions))
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
    adapter, executor = Tau2AirlineAdapter(), Tau2Executor()
    adapter.accept_actions_only = bool((cfg.get("runtime") or {}).get("accept_actions_only"))
    llm = getattr(_local, "llm", None)
    if llm is None:
        llm = LLMClient({"llm": LLM_CFG})
        _local.llm = llm
    tree = trees_by_id[r["tree"]]      # airline sample ids ARE current tree ids
    # one shared pristine flight database; no per-intent initialization
    sample = to_sample_dict(r, persona)
    # F7 (audit 2026-08-20): episode.py reads preset["mask"]["hidden_slots"] to decide
    # what the simulated user keeps PRIVATE, while GRIP scores the runner's recomputed
    # set. They disagreed on all 16 false_presupposition samples (dataset stores [],
    # to_sample_dict adds the FP-contested slot), so the user treated that slot as
    # already-mentioned while GRIP credited the agent for recovering it. One set now.
    hidden = list(sample["hidden_intent"]["hidden_slots"])
    sample.setdefault("mask", {})
    if isinstance(sample["mask"], dict):
        sample["mask"] = dict(sample["mask"])
        sample["mask"]["hidden_slots"] = hidden
    # F4: the adapter's unmet_requests() feeds the rejection reply; it must not recite
    # the VALUES of slots the perturbation hid (that hands a no-ask arm the answer).
    adapter.hidden_slots = set(hidden)
    ctx = caller_context(adapter, r, tree)
    if cfg.get("_clean"):
        sample["query"] = clean_query(adapter, tree.root.conditions)
        sample["strategy_ids"] = ["CLEAN_CONTROL"]
        sample["mask"] = {}
        sample["hidden_intent"]["hidden_slots"] = []
        sample["mask"] = {"hidden_slots": []}
        adapter.hidden_slots = set()
    sample["query"] = ctx + sample["query"]
    t0 = time.time()
    try:
        agent = build_agent(arm, llm=llm, config=cfg, graph=tree, sample=sample,
                            adapter=adapter, executor=executor)
        ep = Episode(graph=tree, adapter=adapter, executor=executor, config=cfg, llm=llm,
                     agent=agent, persona_name=persona,
                     # STABLE + SEED-DEPENDENT (2026-08-19), matching episodes_run_retail.
                     # Was abs(hash(sample_id)): randomised per process by PYTHONHASHSEED, so
                     # no run could be reproduced, and --seed was accepted but discarded.
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
        # ANY-CHANNEL DISCOVERY (register D9, retail parity): the scorer needs the hidden
        # slots' TRUE VALUES in the header, or world-channel reveals earn nothing.
        d["header"]["hidden_intent"] = {"conditions": sample["hidden_intent"]["conditions"]}
        if "_lessonbook_gen" in cfg:
            # A3 provenance (retail parity): which book generation scored this episode
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
    ap.add_argument("--samples", default="artifacts/airline_samples_eval.json")
    ap.add_argument("--trees-dir", default="artifacts/trees/tau2_airline_sel")
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--arms", default="B0")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default="artifacts/airline_episodes")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--max-faults", type=int, default=0, help="0 = no limit")
    ap.add_argument("--turns", type=int, default=0, help="override max_turns")
    ap.add_argument("--shift", action="store_true",
                    help="axis B: p_shift=1.0, one silent mid-episode intent shift")
    ap.add_argument("--clean", action="store_true",
                    help="CONTROL ARM: state the true intent plainly instead of the "
                         "perturbed query. Separates harness difficulty from perturbation "
                         "difficulty -- the reference row every reported number needs.")
    ap.add_argument("--persona", default="",
                    help="pin every episode to ONE persona (e.g. rational) instead of "
                         "the round-robin assignment")
    ap.add_argument("--all-personas", action="store_true",
                    help="run every sample under EVERY persona (full cross product) "
                         "instead of the default round-robin assignment")
    ap.add_argument("--show-patience", action="store_true",
                    help="publish the patience meter to the agent (A2/A3's mechanism)")
    ap.add_argument("--askbook", default="",
                    help="RETIRED -- hard-errors, kept so old commands fail loudly")
    ap.add_argument("--no-shift", action="store_true",
                    help="ABLATION: disable axis B completely (no scheduled "
                         "shift, no trigger, no coin). Stamps shift_schedule "
                         "mode=off. Pairs against the with-shift reference run.")
    ap.add_argument("--shift-fire-prob", type=float, default=None,
                    help="fire coin (register D2): crossing consumes the threshold, the "
                         "intent moves only with this probability; fates pre-drawn per "
                         "sample, scheduled shifts only. Unset = 1.0 = regime R2. Set "
                         "per CAMPAIGN, never mid-chain.")
    ap.add_argument("--max-rps", type=float, default=0.0,
                    help="token-bucket pacing shared across worker threads (0 = config)")
    ap.add_argument("--endpoint", default="pool", choices=["a", "b", "pool"],
                    help="pin one API identity or rotate both (cache keys canonical)")
    ap.add_argument("--cache-tag", default="",
                    help="LLM cache namespace; default seed<N> (per-seed independence)")
    ap.add_argument("--reflector", default="agent", choices=["agent", "user"],
                    help="A3: which model runs reflection (agent = self-reflection)")
    ap.add_argument("--lessonbook-gate", default="reject",
                    choices=["reject", "write_or_reject"],
                    help="A3: when the rendered book becomes visible (v5: after first "
                         "rejected submission)")
    ap.add_argument("--gen-warmup", default="",
                    help="A3: comma list of SMALLER first-generation sizes (e.g. "
                         "'10,15'); later generations use --gen-size. Part of the "
                         "snapshot binding.")
    ap.add_argument("--gen-size", type=int, default=25,
                    help="A3 online evolution: episodes per generation between "
                         "reflections (0 = static scaffold-only A3, no lessonbook)")
    ap.add_argument("--thinking", action="store_true",
                    help="D27 fairness probe: let the AGENT model reason (default off "
                         "per the experiment protocol). Diagnostic for reasoning-first "
                         "backbones; stamped in run_config.")
    ap.add_argument("--backbone", default="deepseek",
                    help="D27 agent backbone (backbones.py); user side never changes. "
                         "'deepseek' = campaign default")
    ap.add_argument("--resume", action="store_true",
                    help="skip episodes whose output file already exists in --out "
                         "(restart a killed run without redoing finished episodes)")
    a = ap.parse_args()

    rt = dict(RUNTIME)
    if a.show_patience:
        rt["show_patience"] = True
    rt["askbook_profile"] = "tau2_airline"
    rt["lessonbook_gate"] = a.lessonbook_gate
    if a.askbook:
        raise SystemExit(
            "--askbook is retired: raw-example books are quarantined and cannot back A3 "
            "(docs/a3-self-evolving-method-review.md section 9). A3 evolves an online "
            "lessonbook automatically -- see --gen-size and docs/a3v2-lessonbook-design.md.")
    if a.turns:
        rt["max_turns"] = a.turns
    if a.shift_fire_prob is not None:
        rt["shift_fire_prob"] = a.shift_fire_prob
    if a.shift:
        rt.update({"shift_scheduled": True,   # MASTER SWITCH; without it nothing shifts.
                   # PERSONA SCHEDULE under the CURRENT physics (register D1, 2026-08-21):
                   # crossings are at-or-below, so rational's single mark (1.0 x initial
                   # patience) is due AT EPISODE START -- the 2026-08-20 audit note about
                   # "crossed on the first charge / only 7% ever shifted" described the
                   # OLD strictly-below rule and no longer applies. With --shift-fire-prob
                   # (register D2) the due crossing actually moves the intent only on a
                   # fire fate; fates are pre-drawn per sample.
                   "shift_schedule": "persona",
                   "p_shift": 1.0, "max_shifts": 1, "shift_roll_window": 6,
                   # every tau2 task "mutates", so the WebShop write-tree scope rule
                   # would disable axis B wholesale; here mutation is accounted for
                   # (telecom: on_shift injects into the live world; retail: shifts
                   # are forbidden after the agent commits a write)
                   "axis_b_on_write_trees": True,
                   "category_probs": {"REFINEMENT": 0.25, "RELAXATION": 0.25, "SUBSTITUTION": 0.25, "PIVOT": 0.25},
                   # F9: "shift_category_mode" is read nowhere in the package; the live
                   # key is on_empty_category (traversal.py:104), which defaults to
                   # "resample" and silently rewrites the category mix. Set it explicitly.
                   "on_empty_category": "skip",
                   "shift_roll_on_acts": True,
                   # SHIFT-AFTER-WRITE, matching retail exactly (episodes_run_retail.py:89).
                   # The two shift mechanisms are mutually exclusive: episode.py:445-452
                   # only takes the roll-window path when shift_scheduled is FALSE, so with
                   # the persona schedule live the only trigger is a patience threshold --
                   # and cost_act is 0, so the agent writes for free before any charge and
                   # `forbid_after_mutation` then bars every later shift. Retail measured
                   # the same thing (0 shifts in 12 episodes with the schedule provably
                   # live) and resolved it by ALLOWING post-write shifts, accepting that
                   # exact-state grading no longer applies because the new node's ground
                   # truth was computed against a pristine database while the agent's
                   # pre-shift writes persist. Acceptance therefore moves to the NEW
                   # intent's golden actions -- tau2's own ACTION reward component. Stale
                   # writes cost patience and surface in the staleness metric instead of
                   # making the episode unwinnable. Airline follows retail so the two
                   # domains are comparable.
                   "shift_after_mutation": "allow",
                   "accept_actions_only": True})
        # D4 PARITY (comment corrected 2026-08-22; the old F1 paragraph here argued the
        # forbid+database-hash regime while the code above says "allow" -- the code is
        # right, the paragraph was stale). Airline follows retail exactly: shifts may
        # land after a write, which breaks exact-state grading (the new node's ground
        # truth was computed against a pristine database), so acceptance is the NEW
        # intent's golden actions -- tau2's own ACTION reward. Stale writes cost
        # patience and surface in Staleness. This keeps the two domains comparable
        # (register D4; airline plan decision 3).
    # D27 BACKBONE OVERLAY: agent side only.
    # Applied BEFORE cache/pool wiring so pacing and namespaces see the final config.
    if a.backbone != "deepseek":
        import backbones as _bb
        globals()["LLM_CFG"] = _bb.apply(LLM_CFG, a.backbone)
    rt["backbone"] = a.backbone   # run-record verification, protocol rule 3
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
    # PER-SEED CACHE NAMESPACE (retail parity, D13): repeats must be independent, while
    # re-running or --resume-ing a cell still reuses its own cache.
    LLM_CFG["cache_dir"] = os.path.join(WS, "artifacts", "llm_cache",
                                        a.cache_tag or f"airline_seed{a.seed}")
    if a.max_rps:
        LLM_CFG["max_rps"] = a.max_rps
    if a.endpoint != "pool":
        i = 0 if a.endpoint == "a" else 1
        # pin per pool ONLY where two identities exist: gateway backbones carry a
        # single-entry agent pool, and the pin is about the USER side anyway
        # (split-identity parallel lanes, the author 2026-08-24)
        if len(LLM_CFG["model_pool"]) > i:
            LLM_CFG["model_pool"] = [LLM_CFG["model_pool"][i]]
        if len(LLM_CFG["agent_model_pool"]) > i:
            LLM_CFG["agent_model_pool"] = [LLM_CFG["agent_model_pool"][i]]
    # O8 AUDIT STAMP (2026-08-23): a 0 means POLICY-OFF -- choice, never accident.
    rt["policy_chars"] = len(Tau2AirlineAdapter().agent_policy())
    cfg = {"runtime": rt, "llm": LLM_CFG, "_clean": bool(a.clean), "_remap": {},
           # --seed reaches the episode seed through here; it was discarded before.
           "_episode_seed": a.seed}
    trees = {t.tree_id: t for t in storage.iter_trees(Path(WS + "/" + a.trees_dir))}
    samples = load_samples(Path(WS + "/" + a.samples), 0, a.seed)
    if a.max_faults:
        samples = [r for r in samples
                   if len(r.get("root_conditions") or []) <= a.max_faults]
    samples = samples[:a.limit] if a.limit else samples
    personas = [a.persona] if a.persona else list(PERSONA_IDS)
    if a.persona and a.persona not in PERSONA_IDS:
        raise SystemExit(f"unknown persona {a.persona!r}; known: {PERSONA_IDS}")
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

    # RUN CONFIG OF RECORD (register D17): the authoritative copy of every knob, plus
    # the lessonbook partition so --resume can refuse a changed method.
    run_config = {"runtime": rt, "args": {k: v for k, v in vars(a).items()},
                  "lessonbook": {"gen_size": a.gen_size, "gen_warmup": a.gen_warmup,
                                 "format": "lessonbook_v2"}}
    rc_path = outdir / "run_config.json"
    if a.resume and rc_path.exists():
        old = json.loads(rc_path.read_text())
        _lb_old = old.get("lessonbook") or {}
        if (_lb_old.get("gen_size") != a.gen_size
                or _lb_old.get("gen_warmup", "") != a.gen_warmup):
            raise SystemExit("--resume with a different --gen-size/--gen-warmup changes "
                             "the generation partition; that is a different method, not "
                             "a resume")
        (outdir / "run_config.resume.json").write_text(json.dumps(run_config, indent=1))
    else:
        rc_path.write_text(json.dumps(run_config, indent=1))

    import concurrent.futures as cf
    results, t0 = [], time.time()

    def _ep_path(job):
        r, p, arm, *_ = job
        return outdir / f"{arm}__{r['tree']}__{r['strategy']}__{p}.json"

    def _not_done(pool_jobs, label=""):
        if not a.resume:
            return pool_jobs
        keep = [j for j in pool_jobs if not _ep_path(j).exists()]
        print(f"resume{label}: {len(pool_jobs) - len(keep)} already on disk, "
              f"{len(keep)} to run", flush=True)
        return keep

    def run_pool(pool_jobs, label=""):
        if not pool_jobs:
            return
        with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
            for i, d in enumerate(ex.map(run_one, pool_jobs), 1):
                results.append(d)
                (outdir / f"{d['header'].get('arm','?')}"
                          f"__{d['header'].get('sample_id','?')}"
                          f"__{d['header'].get('persona','?')}.json").write_text(
                    json.dumps(d, indent=1)[:2_000_000])
                if i % 10 == 0:
                    ok = sum(1 for x in results if x.get("outcome") == "SUCCESS")
                    err = sum(1 for x in results if x.get("outcome") == "ERROR")
                    print(f"  {label}{i}/{len(pool_jobs)} | success {ok} ({ok/i:.0%}) | "
                          f"errors {err} | {time.time()-t0:.0f}s", flush=True)

    # ---- A3 runs in deterministic GENERATIONS; every other arm runs flat -------------
    # Port of episodes_run_retail.py's online self-evolution block (register D22):
    # consecutive chunks of --gen-size; one immutable snapshot per chunk; reflection
    # between chunks on the main thread; snapshots sha-chained and bound fail-closed to
    # (profile, run_id, arm, persona, seed, generation, gen_size, samples_sha).
    a3_jobs = [j for j in jobs if j[2].startswith("A3")] if a.gen_size > 0 else []
    flat_jobs = [j for j in jobs if not (a.gen_size > 0 and j[2].startswith("A3"))]
    if not a3_jobs and a.gen_size == 0 and any(j[2] == "A3" for j in jobs):
        cfg["_lessonbook_gen"] = -1
        cfg["_lessonbook_sha"] = ""
    run_pool(_not_done(flat_jobs), label="")

    if a3_jobs:
        from intent_tree.runtime import lessonbook
        if not a.persona:
            raise SystemExit("A3 online evolution requires --persona: the experimental "
                             "unit is benchmark x arm x persona x seed "
                             "(docs/a3v2-lessonbook-design.md checklist 2)")
        warm = [int(x) for x in a.gen_warmup.split(",") if x.strip()]
        sizes = warm + [a.gen_size] * (len(a3_jobs) // max(a.gen_size, 1) + 2)
        chunks, _i, _k = [], 0, 0
        while _i < len(a3_jobs):
            chunks.append(a3_jobs[_i:_i + max(sizes[_k], 1)])
            _i += max(sizes[_k], 1)
            _k += 1
        lessons_dir = outdir / "lessonbook"
        samples_sha = hashlib.sha1(
            Path(WS + "/" + a.samples).read_bytes()
            + f":{a.limit}:{a.seed}:{a.gen_warmup}".encode()).hexdigest()[:12]
        binding = {"profile": rt["askbook_profile"],
                   "run_id": f"{outdir.parent.name}/{outdir.name}",
                   "arm": a3_jobs[0][2], "persona": a.persona, "seed": a.seed,
                   "gen_size": a.gen_size, "samples_sha": samples_sha}
        import re as _re
        tool_names = tuple(sorted(set(
            _re.findall(r"\b([a-z_]{6,})\(", Tau2AirlineAdapter().tool_manual()))))
        lexicon = lessonbook.build_entity_lexicon(
            os.environ["TAU2_DATA"] + "/domains/airline/db.json")
        llm_main = LLMClient({"llm": LLM_CFG})
        snap, all_grams = None, set()
        print(f"[lessonbook] {len(a3_jobs)} A3 episodes in {len(chunks)} generations "
              f"of <= {a.gen_size}", flush=True)
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
                break
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
                        reflector_role=("select" if a.reflector == "user"
                                        else "agent_audit"),
                        prior_gen_success=(snap or {}).get("gen_success") or [])
                else:
                    res = {"lessons": (snap or {}).get("lessons") or [],
                           "dropped_by_validator": {}, "parse_error": None,
                           "gen_success": (snap or {}).get("gen_success") or [],
                           "outcome_brake": False}
                snap = lessonbook.make_snapshot(
                    lessons_result=res, profile=binding["profile"],
                    run_id=binding["run_id"], arm=binding["arm"],
                    persona=binding["persona"], seed=a.seed, generation=g + 1,
                    parent=snap, source_episode_ids=[d["sample_id"] for d in digs],
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
    (Path(WS) / "artifacts" / "airline_episode_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1)[:4000], flush=True)


if __name__ == "__main__":
    main()
