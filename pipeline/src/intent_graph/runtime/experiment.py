"""Experiment runner: manifest, sharded/threaded execution, quota guard, GRIP reporting.

Three properties the campaign depends on:

**Idempotent.** One trajectory file per (arm, sample). A relaunch skips what exists, so resume
is just "run it again" -- which matters because a long run will be interrupted.

**Quota-guarded.** A drained account is a TERMINAL condition, not something to retry: eight
shards each grinding through backoff wastes a night and tells nobody. The first shard to see
one writes a HALT marker, every shard stops cleanly at its next episode boundary, and the
marker is what a human or a watcher reads to know why.

**Paired.** Every arm replays the SAME manifest rows with the same episode seed and the same
already-verified perturbation, so an arm-to-arm delta is the mechanism and nothing else.
"""

from __future__ import annotations

import collections
import json
import logging
import random
import threading
import time
from pathlib import Path

from ..ids import canonical_dumps
from . import agents as agents_mod
from . import grip
from .episode import Episode
from .llm import ConfigError, QuotaExhausted

log = logging.getLogger(__name__)

HALT_FILE = "HALT"


# --------------------------------------------------------------------- manifest
def build_manifest(samples: list[dict], *, per_cell: int, seed: int = 20260809,
                   personas=None) -> dict:
    """A stratified draw over fault types, CROSSED with personas at evaluation time.

    Records carry no persona (the dataset is graphs + perturbed queries; the author
    2026-08-10). Personas belong to the simulated user, so the manifest draws
    `per_cell` records per fault type and replays EACH drawn record under EVERY
    persona -- the same record appears once per persona, which makes persona
    comparisons paired by construction, on top of the arm pairing. Types that
    cannot supply `per_cell` records give what they have and the shortfall is
    recorded rather than hidden.
    """
    from .persona import PERSONA_IDS
    personas = list(personas or PERSONA_IDS)
    cells: dict[str, list[dict]] = collections.defaultdict(list)
    for s in samples:
        cells["+".join(s["strategy_ids"])].append(s)

    rng = random.Random(seed)
    rows, short = [], {}
    for key in sorted(cells):
        pool = sorted(cells[key], key=lambda s: s["sample_id"])
        rng.shuffle(pool)
        take = pool[:per_cell]
        if len(take) < per_cell:
            short[key] = {"wanted": per_cell, "got": len(take)}
        for s in take:
            for persona in personas:
                rows.append({
                    "sample_id": s["sample_id"],
                    # legacy samples key this "tree_id"
                    "graph_id": s.get("graph_id") or s["tree_id"],
                    "strategy_id": "+".join(s["strategy_ids"]), "persona": persona,
                    "governing_kind": (s.get("mask") or {}).get("governing_kind"),
                    "cluster": s.get("cluster"),
                    # frozen per (record, persona) so every arm sees identical
                    # user behaviour for that pairing
                    "episode_seed": rng.randrange(10**9),
                })
    rows.sort(key=lambda r: (r["cluster"] or "", r["sample_id"], r["persona"]))
    return {
        "seed": seed, "per_cell": per_cell, "personas": personas, "n": len(rows),
        "cells": len(cells), "shortfall": short,
        "by_strategy": dict(collections.Counter(r["strategy_id"] for r in rows)),
        "by_persona": dict(collections.Counter(r["persona"] for r in rows)),
        "rows": rows,
    }


# ----------------------------------------------------------------------- runner
class Halted(RuntimeError):
    """Raised to unwind a shard once the run has been halted."""


class Runner:
    def __init__(self, *, adapter, executor, config: dict, llm_factory, graphs: dict,
                 samples: dict, out_dir: Path, arm: str, seed_offset: int = 0) -> None:
        self.adapter = adapter
        self.executor = executor
        self.config = config
        self.llm_factory = llm_factory
        self.graphs = graphs
        self.samples = samples
        self.arm = arm
        # REPEATS (2026-08-19). The episode seed is drawn once, when the manifest is built,
        # and stored per row -- that is what makes arms comparable, because every arm replays
        # the identical seed. It also means re-running the same manifest reproduces the same
        # episodes exactly, so three "repeats" of one manifest are one run performed three
        # times. `--set episode_seed=N` does NOT change this: the row's seed wins.
        # `seed_offset` is the supported way to get genuine repeats: it shifts every row's
        # seed by a constant, so the sample set and the pairing across arms are untouched
        # while the randomness differs. Pair it with `--cache-tag`, or the cached completions
        # replay anyway and the repeats stay identical where the prompts match.
        self.seed_offset = int(seed_offset)
        self.out = Path(out_dir) / arm
        self.out.mkdir(parents=True, exist_ok=True)
        self.halt_path = Path(out_dir) / HALT_FILE
        self._lock = threading.Lock()
        self.done = 0
        self.skipped = 0
        self.errors = 0
        self.halt_reason: str | None = None

    # ------------------------------------------------------------------ control
    def halted(self) -> bool:
        return self.halt_reason is not None or self.halt_path.exists()

    def halt(self, reason: str) -> None:
        with self._lock:
            if self.halt_reason is None:
                self.halt_reason = reason
                payload = {"arm": self.arm, "reason": reason, "at": int(time.time()),
                           "done": self.done}
                self.halt_path.write_text(canonical_dumps(payload, indent=1), encoding="utf-8")
                log.error("HALTED: %s", reason)

    # ------------------------------------------------------------------ one row
    def run_row(self, row: dict) -> None:
        if self.halted():
            raise Halted(self.halt_reason or "halt file present")
        # one record now runs under several personas per arm -- the persona is part
        # of the trajectory's identity, or the five runs would overwrite each other
        path = self.out / f"{row['sample_id']}__{row['persona']}.json"
        if path.exists():
            # Idempotent resume -- but an ERROR trajectory is NOT a result. A transient
            # environment failure (or, once, a missing API key that produced 500 useless
            # files) must be retried on relaunch, or the garbage becomes permanent precisely
            # because the runner is idempotent.
            try:
                prior = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                prior = None
            if prior and prior.get("outcome") != "ERROR":
                with self._lock:
                    self.skipped += 1
                return

        graph = self.graphs.get(row.get("graph_id") or row.get("tree_id"))
        sample = self.samples.get(row["sample_id"])
        if graph is None or sample is None:
            with self._lock:
                self.errors += 1
            log.warning("missing graph/sample for %s", row["sample_id"])
            return

        llm = self.llm_factory()
        # the arm name enters the cache key, so two arms never share an agent-side entry
        llm.prompt_version = f"{llm.prompt_version}-{self.arm}"
        try:
            agent = agents_mod.build_agent(self.arm, llm=llm, config=self.config,
                                           graph=graph, sample=sample,
                                           adapter=self.adapter, executor=self.executor)
            traj = Episode(graph=graph, adapter=self.adapter, executor=self.executor,
                           config=self.config, llm=llm, agent=agent,
                           persona_name=row["persona"],
                           seed=row["episode_seed"] + self.seed_offset,
                           preset=sample).run()
        except QuotaExhausted as exc:
            self.halt(f"quota/credit exhausted: {exc}")
            raise Halted(str(exc)) from exc
        except ConfigError as exc:
            # Misconfiguration is not a per-episode failure: every remaining episode would
            # fail the same way, so stop the whole run and say why.
            self.halt(f"misconfigured, not a model problem: {exc}")
            raise Halted(str(exc)) from exc
        except Exception as exc:                # a broken episode must not end the shard
            with self._lock:
                self.errors += 1
            log.warning("episode %s failed: %s: %s", row["sample_id"], type(exc).__name__, exc)
            return

        if traj.outcome == "ERROR" and traj.error and "not set" in str(traj.error):
            self.halt(f"misconfigured: {traj.error}")
            raise Halted(str(traj.error))
        payload = traj.to_dict()
        payload["experiment"] = {
            "arm": self.arm, "sample_id": row["sample_id"], "strategy_id": row["strategy_id"],
            "persona": row["persona"], "governing_kind": row["governing_kind"],
            "agent_notes": list(getattr(agent, "notes", ())),
            "usage": llm.usage(),
            # WHICH MODEL PRODUCED THIS EPISODE (2026-08-26, backbone experiment D27).
            # Nothing here recorded the agent side, so a backbone campaign would be
            # unverifiable after the fact -- and un-stamped configuration is exactly how
            # the tau2 fork ran four days on a silently EMPTY store policy. The user side
            # is stamped too because protocol rule 1 (never change the user simulator) is
            # only checkable if the user model is on the record next to the agent's.
            # thinking is part of the REGIME: thinking-on and thinking-off cells must
            # never share a table.
            "agent_model": getattr(llm, "agent_model", None),
            "user_model": getattr(llm, "model", None),
            "agent_base_url": getattr(llm, "agent_base_url", None),
            "thinking": not bool(getattr(llm, "disable_thinking", False)),
        }
        tmp = path.with_suffix(".tmp")
        # Plain JSON, not canonical_dumps: canonicalization renders non-integral floats as
        # STRINGS (repr) for hash stability, which is right for hashed inputs but wrong for
        # a data file -- the float-budget economy writes patience_after=1.6 and every
        # scorer downstream must see a number. Trajectory files are never content-hashed.
        tmp.write_text(json.dumps(payload, sort_keys=True, ensure_ascii=False,
                                  indent=1, default=str), encoding="utf-8")
        tmp.replace(path)                        # atomic: a killed shard leaves no half file
        with self._lock:
            self.done += 1

    # ------------------------------------------------------------------ threads
    def run(self, rows: list[dict], *, threads: int = 1) -> dict:
        started = time.time()
        if threads <= 1:
            for row in rows:
                try:
                    self.run_row(row)
                except Halted:
                    break
        else:
            queue = list(rows)
            index = [0]
            qlock = threading.Lock()

            def worker():
                while True:
                    with qlock:
                        if index[0] >= len(queue):
                            return
                        row = queue[index[0]]
                        index[0] += 1
                    try:
                        self.run_row(row)
                    except Halted:
                        return
            pool = [threading.Thread(target=worker, daemon=True) for _ in range(threads)]
            for t in pool:
                t.start()
            for t in pool:
                t.join()
        return {"arm": self.arm, "done": self.done, "skipped": self.skipped,
                "errors": self.errors, "halted": self.halt_reason,
                "seconds": round(time.time() - started, 1)}


# ---------------------------------------------------------------------- reporting
def load_trajectories(out_dir: Path, arm: str) -> list[dict]:
    d = Path(out_dir) / arm
    if not d.exists():
        return []
    out = []
    for p in sorted(d.glob("*.json")):
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            log.warning("unreadable trajectory %s", p)
    return out


def bootstrap_ci(values: list[float], *, draws: int = 2000, seed: int = 7) -> tuple:
    """Percentile bootstrap 95% CI. Returns (lo, hi), or (None, None) if undefined."""
    vals = [v for v in values if v is not None]
    if len(vals) < 2:
        return (None, None)
    rng = random.Random(seed)
    n = len(vals)
    means = []
    for _ in range(draws):
        means.append(sum(vals[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return (round(means[int(0.025 * draws)], 4), round(means[int(0.975 * draws)], 4))


def report_arm(trajectories: list[dict]) -> dict:
    """GRIP for one arm, recomputed offline from stored trajectories."""
    eps = [grip.score(t) for t in trajectories]
    out = grip.report(eps)
    out["ci"] = {
        "success_rate": bootstrap_ci([1.0 if e.success else 0.0 for e in eps]),
        "earned": bootstrap_ci([1.0 if e.bucket == grip.EARNED else 0.0 for e in eps]),
        "recovery": bootstrap_ci([e.recovery for e in eps if e.recovery is not None]),
        "aim": bootstrap_ci([e.aim for e in eps if e.aim is not None]),
    }
    calls = [t.get("experiment", {}).get("usage", {}) for t in trajectories]
    reqs = [c.get("requests", 0) + c.get("cached", 0) for c in calls if c]
    out["cost"] = {
        "mean_llm_calls": round(sum(reqs) / len(reqs), 2) if reqs else None,
        "fresh_requests": sum(c.get("requests", 0) for c in calls),
        "cache_hits": sum(c.get("cached", 0) for c in calls),
        "agent_fallbacks": sum(c.get("agent_fallbacks", 0) for c in calls),
    }
    if out["cost"]["mean_llm_calls"]:
        out["cost"]["success_per_100_calls"] = round(
            100.0 * out["success_rate"] / out["cost"]["mean_llm_calls"], 3)
    notes = collections.Counter()
    for t in trajectories:
        for n in t.get("experiment", {}).get("agent_notes", ()):
            notes[n.split("=")[0]] += 1
    out["agent_notes"] = dict(notes.most_common(12))
    out["outcomes"] = dict(collections.Counter(t["outcome"] for t in trajectories))
    return out


def paired_delta(a: list[dict], b: list[dict], key="success") -> dict:
    """Delta between two arms on the samples BOTH completed.

    Paired on sample_id: marginal CIs on two arms would ignore that they faced identical
    perturbations and identical user seeds, which is the whole reason the manifest is shared.
    """
    def index(trajs):
        out = {}
        for t in trajs:
            sid = t.get("experiment", {}).get("sample_id")
            if sid:
                out[sid] = grip.score(t)
        return out
    ia, ib = index(a), index(b)
    shared = sorted(set(ia) & set(ib))
    if not shared:
        return {"n": 0}
    pick = {"success": lambda e: 1.0 if e.success else 0.0,
            "earned": lambda e: 1.0 if e.bucket == grip.EARNED else 0.0,
            "recovery": lambda e: e.recovery,
            "aim": lambda e: e.aim}[key]
    diffs = []
    for sid in shared:
        x, y = pick(ia[sid]), pick(ib[sid])
        if x is not None and y is not None:
            diffs.append(y - x)
    if not diffs:
        return {"n": len(shared), "metric": key, "delta": None}
    return {"n": len(diffs), "metric": key,
            "delta": round(sum(diffs) / len(diffs), 4),
            "ci95": bootstrap_ci(diffs)}


__all__ = ["build_manifest", "Runner", "Halted", "load_trajectories", "report_arm",
           "paired_delta", "bootstrap_ci", "HALT_FILE"]
