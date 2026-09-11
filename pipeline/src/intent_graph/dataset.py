"""Exporting perturbation samples: the downstream test set.

A *sample* is one task instance, not a solved episode. It carries the misaligned opening
query the agent sees, the mask that produced it, and everything a harness needs to score the
attempt -- the hidden intent, the legal intent shifts, and the executable verifier's
parameters. The interaction itself happens at test time, against whatever agent is under
evaluation.

The dataset is intent graphs + perturbed queries, nothing else (ruling 2026-08-10). A v2
sample is (graph x strategy): the strategy decides what is misstated, and no persona touches
generation. Personas are an EVALUATION concept -- the simulated user is activated when the
agent runs, holds the graph's intent as its own, and every persona starts from the same
perturbed query. (The legacy v1 path below still loops personas into the query voice; it is
kept only for ablation against v2 and must not be used for new data.)

The agent-visible part is exactly `query`. Everything else is scoring material and must not
be shown: `hidden_intent` holds the conditions and `ground_truth` the accepted purchases.
"""

from __future__ import annotations

import collections
import logging
import re

from . import storage
from .ids import content_hash
from .runtime import perturb as perturb_mod
from .runtime import signature as signature_mod
from .runtime import strategies as st
from .runtime.persona import PERSONA_IDS
from .runtime.persona import get as get_persona
from .runtime.traversal import announce, transition_graph

log = logging.getLogger(__name__)


def sample_id(graph_id: str, strategy_ids, persona: str, spec_id: str) -> str:
    return content_hash(graph_id, tuple(strategy_ids), persona, spec_id)


def _shift_options(graph, adapter) -> list[dict]:
    """Where the hidden intent may legally move, and what the user would say.

    ``graph`` is the stored intent graph; ``transitions`` is the runtime move table derived
    from it. The two must not share a name -- they did briefly during the 2026-08-19
    rename, and a dict silently stood in for the object.
    """
    transitions = transition_graph(graph, adapter)
    out = []
    for op, edges in sorted(transitions.get(graph.root.intent_id, {}).items(),
                            key=lambda kv: kv[0].value):
        for edge in sorted(edges, key=lambda e: e.dst):
            out.append({
                "operator": op.value,
                "to_intent": edge.dst,
                "gt_moved": edge.gt_moved,
                "provenance": edge.provenance.value,
                "announcement": announce(edge, adapter),
            })
    return out


def build_samples(graph, adapter, executor, config: dict, llm, *,
                  personas=None, strategies=None, seeds_meta=None,
                  pipeline: str = "v1") -> list[dict]:
    """Every (strategy, persona) sample for one graph.

    A strategy that cannot be masked onto this intent is skipped with a recorded reason
    rather than dropped silently: a run where half the strategies quietly failed would
    otherwise look like a smaller run.
    """
    rt = config["runtime"]
    min_retained = int(rt.get("min_retained_conditions", 1))
    k = max(1, int(rt.get("misalignments_k", 1)))
    root = graph.root
    persona_ids = list(personas or PERSONA_IDS)
    pool = list(strategies or st.applicable_strategies(root.conditions,
                                                       min_retained=min_retained))
    meta = (seeds_meta or {}).get(root.source_record) or {}

    if pipeline == "v2":
        return _build_samples_v2(graph, adapter, executor, config, llm,
                                 persona_ids=persona_ids, pool=pool, meta=meta)

    samples: list[dict] = []
    skipped: list[dict] = []
    with executor.open(graph.env_spec) as session:
        pristine = None
        verify = signature_mod.make_verifier(adapter, session, root, pristine_hash=pristine)
        domains, foreign = {}, {}
        get_foreign = getattr(adapter, "foreign_domains", None)
        for slot, _, _ in root.conditions:
            try:
                domains[slot] = list(adapter.domains(slot, root.base, session))
            except Exception:
                domains[slot] = []
            if get_foreign is None:
                continue
            try:
                foreign[slot] = list(get_foreign(slot, root.base, session))
            except Exception:
                foreign[slot] = []
        shifts = _shift_options(graph, adapter)

        for strategy in pool:
            for pid in persona_ids:
                rng_key = (graph.graph_id, strategy.id, pid)
                persona = get_persona(pid, _FixedRng(*rng_key, "persona"))
                combo = [strategy]
                if k > 1:
                    combo = st.sample_strategies(root.conditions,
                                                 _FixedRng(*rng_key, "combo"),
                                                 k=k, min_retained=min_retained) or [strategy]
                try:
                    pert = perturb_mod.perturb(
                        node=root, conditions=root.conditions, strategies=combo,
                        rng=_FixedRng(*rng_key, "mask"), llm=llm,
                        surface=perturb_mod.surface_conventions(graph.adapter),
                        config=config, domains=domains, verify=verify,
                        persona_tone=persona.bio.split("\n")[0],
                        describe_slot=getattr(adapter, "slot_phrase", None),
                        foreign_domains=foreign,
                        render_context=(getattr(adapter, "render_context", None)
                                        or (lambda _b: ""))(root.base),
                    )
                except perturb_mod.Unperturbable as exc:
                    skipped.append({"strategy_id": strategy.id, "persona": pid,
                                    "reason": str(exc)})
                    continue

                spec = pert.spec
                samples.append({
                    "sample_id": sample_id(graph.graph_id, spec.components, pid, spec.spec_id()),
                    "graph_id": graph.graph_id,
                    "adapter": graph.adapter,
                    "cluster": graph.env_spec.get("cluster"),
                    "goal_idx": meta.get("goal_idx"),
                    "split": meta.get("split"),
                    "source_instruction": meta.get("instruction"),
                    # ---- what the agent sees, and nothing else ----
                    "query": pert.query,
                    "persona": pid,
                    # ---- how it was corrupted ----
                    "strategy_ids": list(spec.components),
                    "misalignments": len(spec.components),
                    "mask": spec.to_dict(),
                    "signature": pert.signature,
                    "fidelity_notes": list(pert.fidelity_notes),
                    "render_attempts": pert.attempts,
                    # ---- scoring material: never show this to the agent ----
                    "hidden_intent": {
                        "intent_id": root.intent_id,
                        "conditions": [list(c) for c in root.conditions],
                        "hidden_slots": list(spec.hidden_slots),
                    },
                    "ground_truth": {
                        "kind": root.ground_truth.kind,
                        "cardinality": root.ground_truth.cardinality(),
                        "value": list(root.ground_truth.value),
                        "hash": root.ground_truth.hash,
                    },
                    "shift_options": shifts,
                    "verifier": {
                        "adapter": graph.adapter,
                        "kind": "adapter_accepts",
                        "note": "acceptance is executable: adapter.accepts(proposal, node, "
                                "session) using the benchmark's own comparison",
                    },
                })
    return samples, skipped


def _build_samples_v2(graph, adapter, executor, config, llm, *, persona_ids, pool, meta):
    """Generate-then-verify (docs/pipeline-v2.md).

    The dataset is intent graphs + perturbed queries, nothing else (ruling 2026-08-10).
    Personas belong to EVALUATION: the simulated user is activated when the agent runs,
    every persona starts from the same perturbed query, and no persona touches
    generation. ``persona_ids`` is accepted for interface parity with v1 and ignored.
    """
    from .runtime.genverify import Rejected, build_v2_mask
    from .runtime.traversal import announce  # noqa: F401  (kept parallel with v1 imports)

    del persona_ids  # evaluation-time concept; must not steer generation
    rt = config["runtime"]
    dual = bool(rt.get("v2_dual_extract", True))
    attempts = int(rt.get("v2_attempts", 4))
    root = graph.root
    samples, skipped = [], []
    context = ""
    fn = getattr(adapter, "render_context", None)
    if fn is not None:
        try:
            context = fn(root.base) or ""
        except Exception:
            context = ""

    with executor.open(graph.env_spec) as session:
        shifts = _shift_options(graph, adapter)
        for strategy in pool:
                try:
                    out, _ = build_v2_mask(graph, strategy, llm=llm, adapter=adapter,
                                           executor=executor, session=session,
                                           context=context,
                                           dual_extract=dual, attempts=attempts)
                except Rejected as exc:
                    skipped.append({"strategy_id": strategy.id,
                                    "reason": f"{strategy.id}: v2:{exc.verdict}"})
                    continue
                except Exception as exc:
                    # One strategy's crash must cost ONE cell, not the graph: an unisolated
                    # ZeroDivisionError here killed 11 of the pilot's 20 graphs, taking nine
                    # healthy strategies down with each bad one.
                    skipped.append({"strategy_id": strategy.id,
                                    "reason": f"{strategy.id}: "
                                              f"v2:error:{type(exc).__name__}"})
                    continue
                mask, query, sig = out["mask"], out["query"], out["signature"]
                samples.append({
                    "sample_id": sample_id(graph.graph_id, (strategy.id,), "",
                                           mask["spec_id"]),
                    "graph_id": graph.graph_id, "adapter": graph.adapter,
                    "cluster": graph.env_spec.get("cluster"),
                    "goal_idx": meta.get("goal_idx"), "split": meta.get("split"),
                    "source_instruction": meta.get("instruction"),
                    "query": query,
                    "strategy_ids": [strategy.id], "misalignments": 1,
                    "mask": mask,
                    "signature": {"ok": sig.get("verdict") == "ok",
                                  "verdict": sig.get("verdict"),
                                  "true_card": sig.get("true_card"),
                                  "lit_card": sig.get("lit_card"),
                                  "governing_kind": mask["governing_kind"],
                                  **{k: v for k, v in sig.items()
                                     if k in ("reading_cards", "rule")}},
                    "fidelity_notes": [], "render_attempts": mask["attempt"],
                    "hidden_intent": {
                        "intent_id": root.intent_id,
                        "conditions": [list(c) for c in root.conditions],
                        "hidden_slots": list(mask["hidden_slots"]),
                    },
                    "ground_truth": {
                        "kind": root.ground_truth.kind,
                        "cardinality": root.ground_truth.cardinality(),
                        "value": list(root.ground_truth.value),
                        "hash": root.ground_truth.hash,
                    },
                    "shift_options": shifts,
                    "verifier": {"adapter": graph.adapter, "kind": "adapter_accepts",
                                 "note": "acceptance is executable via the benchmark's own "
                                         "comparison"},
                })
    return samples, skipped


class _FixedRng:
    """Reproducible per-sample randomness, seeded from the sample's own identity.

    Two requirements pull against each other. Reproducibility across shards and reruns rules
    out a shared Random, which would make a sample depend on how many were built before it.
    But seeding every sample from the same constant makes every sample draw the SAME
    sequence -- which showed up as "argan oil" appearing as the false premise over and over,
    because the foreign-value shuffle was identical each time.

    Seeding from (graph_id, strategy, persona) gives both: the same sample always draws the
    same values, and different samples draw different ones.
    """

    def __init__(self, *parts) -> None:
        import hashlib
        import random
        key = "|".join(str(p) for p in parts)
        seed = int(hashlib.sha256(key.encode()).hexdigest()[:16], 16)
        self._r = random.Random(seed)

    def __getattr__(self, name):
        return getattr(self._r, name)


def write_samples(out_dir, adapter_name: str, samples: list[dict]) -> int:
    """One JSONL shard per adapter, appended atomically enough for parallel shards."""
    from pathlib import Path

    from .ids import canonical_dumps
    d = Path(out_dir) / "samples" / adapter_name
    d.mkdir(parents=True, exist_ok=True)
    written = 0
    for s in samples:
        p = d / f"{s['sample_id']}.json"
        p.write_text(canonical_dumps(s, indent=1), encoding="utf-8")
        written += 1
    return written


def iter_samples(root, adapter_name: str):
    """Read samples back, for measurement and for a harness to consume."""
    import json
    from pathlib import Path
    d = Path(root) / "samples" / adapter_name
    if not d.exists():
        return
    for p in sorted(d.glob("*.json")):
        yield json.loads(p.read_text(encoding="utf-8"))


__all__ = ["build_samples", "write_samples", "iter_samples", "sample_id",
           "ExportStats", "storage"]


_SKIP_VERDICT = re.compile(r"last verdict:\s*([a-z_]+)")


def classify_skip(reason: str) -> str:
    """The diagnostic class of a skip, not the strategy that hit it.

    Keying on the leading strategy id -- the obvious thing -- reports "vagueness_subjectivity"
    as the *reason* vagueness_subjectivity was skipped, which says nothing. The informative
    part is the signature verdict: `withhold_did_not_widen` means the mask hid a redundant
    condition and hid nothing, which is the check working, whereas
    `false_premise_still_satisfiable` would mean the bogus value happened to be real.
    """
    text = str(reason or "")
    m = _SKIP_VERDICT.search(text)
    if m:
        return f"signature:{m.group(1)}"
    tail = text.split(":", 1)[1].strip() if ":" in text else text
    for probe, label in (
        ("nothing can be withheld", "too_few_conditions_to_withhold"),
        ("no eligible slot to mark", "no_markable_slot"),
        ("no candidate mask", "no_candidate_mask"),
        ("no coherent composite", "composite_incoherent"),
        ("template fallback also failed", "template_fallback_failed"),
        ("empty_literal_reading", "empty_literal_reading"),
    ):
        if probe in tail:
            return label
    return tail[:60] or "unknown"


class ExportStats:
    """Dataset-quality accounting for an export run.

    The point is that a degraded run must not look like a smaller one. Three ways this dataset
    could quietly come out weaker than requested, each recorded here:

      * a strategy that cannot be masked onto an intent (`skips_by_strategy`)
      * a k>1 composite that had to degrade to fewer flaws (`misalignments_achieved`)
      * a render the model would not get right, falling back to the template
        (`template_fallbacks`) or needing retries (`render_attempts`)

    Without these, an export that produced 11,000 single-flaw template renders would report
    the same headline number as 11,000 good ones.
    """

    def __init__(self) -> None:
        self.graphs = 0
        self.graphs_failed = 0
        self.samples = 0
        self.by_strategy: collections.Counter = collections.Counter()
        self.by_persona: collections.Counter = collections.Counter()
        self.signature_verdicts: collections.Counter = collections.Counter()
        self.governing_kind: collections.Counter = collections.Counter()
        self.misalignments_achieved: collections.Counter = collections.Counter()
        self.render_attempts: collections.Counter = collections.Counter()
        self.template_fallbacks = 0
        self.fidelity_notes: collections.Counter = collections.Counter()
        self.hides_information = 0
        self.skips_by_strategy: collections.Counter = collections.Counter()
        self.skip_reasons: collections.Counter = collections.Counter()

    def record(self, samples: list[dict], skipped: list[dict]) -> None:
        self.graphs += 1
        for s in samples:
            self.samples += 1
            for sid in s["strategy_ids"]:
                self.by_strategy[sid] += 1
            if "persona" in s:      # v1 samples only; v2 records carry no persona
                self.by_persona[s["persona"]] += 1
            self.misalignments_achieved[s["misalignments"]] += 1
            self.render_attempts[s.get("render_attempts", 0)] += 1
            sig = s.get("signature") or {}
            self.signature_verdicts[str(sig.get("verdict"))] += 1
            self.governing_kind[str((s.get("mask") or {}).get("governing_kind"))] += 1
            notes = s.get("fidelity_notes") or []
            if "template_fallback" in notes:
                self.template_fallbacks += 1
            for n in notes:
                self.fidelity_notes[n.split(":")[0]] += 1
            if s["hidden_intent"]["hidden_slots"]:
                self.hides_information += 1
        for entry in skipped:
            self.skips_by_strategy[entry["strategy_id"]] += 1
            self.skip_reasons[classify_skip(entry["reason"])] += 1

    def to_dict(self) -> dict:
        return {
            "graphs": self.graphs,
            "graphs_failed": self.graphs_failed,
            "samples": self.samples,
            "samples_hiding_information": self.hides_information,
            "by_strategy": dict(sorted(self.by_strategy.items())),
            "by_persona": dict(sorted(self.by_persona.items())),
            "signature_verdicts": dict(sorted(self.signature_verdicts.items())),
            "governing_kind": dict(sorted(self.governing_kind.items())),
            "misalignments_achieved": {str(k): v for k, v in
                                       sorted(self.misalignments_achieved.items())},
            "render_attempts": {str(k): v for k, v in sorted(self.render_attempts.items())},
            "template_fallbacks": self.template_fallbacks,
            "fidelity_notes": dict(sorted(self.fidelity_notes.items())),
            "strategy_skips": sum(self.skips_by_strategy.values()),
            "skips_by_strategy": dict(sorted(self.skips_by_strategy.items())),
            "skip_reasons": dict(sorted(self.skip_reasons.items())),
        }
