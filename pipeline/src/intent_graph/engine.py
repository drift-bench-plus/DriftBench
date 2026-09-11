"""The dataset-agnostic graph-generation algorithm.

    validate -> group by environment -> retrieve real branches -> enumerate mutations
    -> execute + gate -> rank -> build root -> emit

Nothing here knows what a dataset is.  Everything dataset-specific arrives through the
Adapter protocol, and everything that touches an environment goes through an
EnvSession.  Retrieval always runs before mutation (D10): a branch that already exists in
the corpus is human-authored, already validated, and evidence that the branch type occurs
naturally, so it outranks anything we synthesize.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, replace, field

from . import classify as cl
from .gate import GateResult, admit, default_gt_equal, rank_and_cap, rejection_summary
from .ids import config_hash
from .models import (
    Candidate,
    Edge,
    Node,
    Operator,
    Provenance,
    Seed,
    Graph,
    sort_conditions,
)

log = logging.getLogger(__name__)

DEFAULT_CAPS = {
    Operator.REFINEMENT: 24,
    Operator.RELAXATION: 12,
    Operator.SUBSTITUTION: 24,
    Operator.PIVOT: 12,
}


@dataclass
class RunStats:
    seeds_loaded: int = 0
    seeds_parseable: int = 0
    seeds_seen: int = 0          # seeds inside the environments actually processed
    seeds_validated: int = 0
    environments: int = 0
    graphs_emitted: int = 0
    graphs_dropped_no_moved_edge: int = 0
    graphs_dropped_no_branches: int = 0
    candidates_generated: Counter = field(default_factory=Counter)
    candidates_gated_ok: Counter = field(default_factory=Counter)
    candidates_picked: Counter = field(default_factory=Counter)
    quota_shortfall: dict = field(default_factory=dict)
    quota_filled: Counter = field(default_factory=Counter)
    graphs_dropped_partial_quota: int = 0
    provenance: Counter = field(default_factory=Counter)
    gt_moved: Counter = field(default_factory=Counter)
    rejections: Counter = field(default_factory=Counter)
    validation_failures: Counter = field(default_factory=Counter)

    def to_dict(self) -> dict:
        return {
            "seeds_loaded": self.seeds_loaded,
            "seeds_parseable": self.seeds_parseable,
            "seeds_seen": self.seeds_seen,
            "seeds_validated": self.seeds_validated,
            "environments": self.environments,
            "graphs_emitted": self.graphs_emitted,
            "graphs_dropped_no_moved_edge": self.graphs_dropped_no_moved_edge,
            "graphs_dropped_no_branches": self.graphs_dropped_no_branches,
            "candidates_generated": dict(self.candidates_generated),
            "candidates_gated_ok": dict(self.candidates_gated_ok),
            "candidates_picked": dict(self.candidates_picked),
            "quota_shortfall": dict(self.quota_shortfall),
            "quota_filled": dict(self.quota_filled),
            "graphs_dropped_partial_quota": self.graphs_dropped_partial_quota,
            "provenance": dict(self.provenance),
            "gt_moved": dict(self.gt_moved),
            "rejections": dict(self.rejections),
            "validation_failures": dict(self.validation_failures),
        }


# --------------------------------------------------------------------------- retrieval
def real_candidates(seed: Seed, others: list[Seed]) -> list[Candidate]:
    """Existing dataset records that already stand in a branch relation to ``seed``."""
    out: list[Candidate] = []
    for other in others:
        if other.record_id == seed.record_id or not other.parseable:
            continue
        op = cl.classify(seed.conditions, other.conditions, seed.base, other.base)
        if op is None:
            continue
        out.append(
            Candidate(
                conditions=other.conditions,
                base=other.base,
                operator=op,
                delta=cl.delta(seed.conditions, other.conditions, op, other.base),
                provenance=Provenance.REAL,
                source_record=other.record_id,
            )
        )
    return out


# ------------------------------------------------------------------------- enumeration
def enumerate_refinements(seed: Seed, witness: list[dict], cap: int) -> list[Candidate]:
    """Add a property held by SOME BUT NOT ALL witness objects.

    That "some but not all" is the whole trick: survivors are guaranteed (some have it,
    so the answer stays non-empty) and the answer necessarily shrinks (not all do).
    Fulfillability is therefore structural, not something we check afterwards and hope.
    """
    n = len(witness)
    if n < 2:
        return []
    taken = {s for s, _, _ in seed.conditions}
    counts: Counter = Counter()
    for w in witness:
        for slot, value in w.items():
            if slot in taken or value is None:
                continue
            counts[(slot, str(value))] += 1

    out: list[Candidate] = []
    for (slot, value), count in sorted(counts.items(), key=lambda kv: (-kv[1], str(kv[0]))):
        if not (0 < count < n):
            continue
        conds = sort_conditions([*seed.conditions, (slot, "=", value)])
        out.append(
            Candidate(
                conditions=conds,
                base=seed.base,
                operator=Operator.REFINEMENT,
                delta={"added": [[slot, "=", value]]},
                provenance=Provenance.SYNTHETIC,
            )
        )
        if len(out) >= cap:
            break
    return out


def enumerate_relaxations(seed: Seed, cap: int) -> list[Candidate]:
    """Drop each condition in turn (monotone: the answer can only widen)."""
    out: list[Candidate] = []
    for c in seed.conditions:
        rest = tuple(x for x in seed.conditions if x != c)
        if not rest:
            continue  # dropping the last condition leaves no intent at all
        out.append(
            Candidate(
                conditions=sort_conditions(rest),
                base=seed.base,
                operator=Operator.RELAXATION,
                delta={"removed": [list(c)]},
                provenance=Provenance.SYNTHETIC,
            )
        )
        if len(out) >= cap:
            break
    return out


def enumerate_substitutions(seed: Seed, domains: dict[str, list], cap: int) -> list[Candidate]:
    """Swap one condition's value for another value that occurs in this environment."""
    out: list[Candidate] = []
    for slot, op, value in seed.conditions:
        for alt in domains.get(slot, []):
            if str(alt) == str(value):
                continue
            conds = sort_conditions(
                [(s, o, alt) if (s, o) == (slot, op) else (s, o, v) for s, o, v in seed.conditions]
            )
            out.append(
                Candidate(
                    conditions=conds,
                    base=seed.base,
                    operator=Operator.SUBSTITUTION,
                    delta={"changed": [[slot, op, value, alt]]},
                    provenance=Provenance.SYNTHETIC,
                )
            )
            if len(out) >= cap:
                return out
    return out


# ------------------------------------------------------------------------------- root
def soft_slots(seed: Seed, children: list[Candidate]) -> tuple[str, ...]:
    """Slots the seed's same-frame siblings disagree about.

    These are the conditions a later rendering stage can safely withhold or leave vague:
    the graph already contains branches where they are dropped or changed, so vagueness
    about them is grounded in real alternatives rather than guessed.

    Note this is metadata on the seed, not a separate node.  An earlier design made the
    siblings' *common ground* a scored root, which was wrong: the user's true intent is
    the seed, so scoring against a vaguer intersection would let an agent that never
    recovered the withheld details still earn full marks -- exactly the guess-versus-
    confirm distinction the evaluation protocol exists to detect.  Relaxation branches
    already supply the vaguer intents, which makes the intersection node redundant too.
    """
    seed_slots = {s for s, _, _ in seed.conditions}
    varying: set[str] = set()
    for c in children:
        if c.base != seed.base:
            continue  # a pivot is a different goal, not disagreement about this one
        child_slots = {s for s, _, _ in c.conditions}
        varying |= seed_slots ^ child_slots                       # dropped or added
        varying |= {s for s, o, v in c.conditions
                    if (s, o, v) not in set(seed.conditions) and s in seed_slots}
    return tuple(sorted(varying))


# ------------------------------------------------------------------------------ engine
def generate(adapter, executor, config: dict, *, limit_envs: int | None = None,
             only_envs=None, shard: tuple[int, int] | None = None,
             stats: RunStats | None = None) -> Iterator[Graph]:
    """Yield one graph per validated seed that supports at least one usable branch.

    ``only_envs`` restricts generation to named environments; ``shard=(i, n)`` keeps every
    n-th environment. Sharding is how a long sweep is parallelised: environments are fully
    independent -- one cluster per process, no shared state -- so N processes divide the
    wall clock almost exactly. It is also what lets the golden fixture regenerate exactly
    one small cluster in seconds.
    """
    stats = stats or RunStats()
    cfg_hash = config_hash(config)
    branching = int(config.get("branching", 8))
    min_moved = int(config.get("min_gt_moved_edges", 1))
    caps = {**DEFAULT_CAPS, **{Operator(k): v for k, v in (config.get("caps") or {}).items()}}
    gt_equal = getattr(adapter, "gt_equal", default_gt_equal)

    if int(config.get("depth", 1)) != 1:
        raise NotImplementedError(
            "depth>1 requires computing a child's ground truth against the parent's "
            "post-action state (decision D1); only depth=1 is implemented."
        )

    seeds = adapter.load()
    stats.seeds_loaded = len(seeds)
    stats.seeds_parseable = sum(1 for s in seeds if s.parseable)

    by_env: dict[str, list[Seed]] = defaultdict(list)
    for s in seeds:
        by_env[s.env_key].append(s)
    env_keys = sorted(by_env)
    # An environment with no root-eligible seed can only ever be a retrieval pool, and
    # materialising it to validate seeds that will then be skipped is pure cost. On the
    # WebShop test split this is 61 of 255 clusters -- a quarter of the sweep.
    env_keys = [k for k in env_keys if any(s.root_eligible for s in by_env[k])]
    if only_envs is not None:
        wanted = set(only_envs)
        env_keys = [k for k in env_keys if k in wanted]
    if shard is not None:
        i, n = shard
        if not (0 <= i < n):
            raise ValueError(f"shard index {i} out of range for {n} shards")
        env_keys = [k for j, k in enumerate(env_keys) if j % n == i]
    if limit_envs is not None:
        env_keys = env_keys[:limit_envs]
    stats.environments = len(env_keys)

    for env_key in env_keys:
        env_spec = adapter.env_spec(env_key)
        env_id = env_spec["env_id"]
        try:
            with executor.open(env_spec) as session:
                validated = []
                stats.seeds_seen += len(by_env[env_key])
                for s in by_env[env_key]:
                    try:
                        if adapter.validate(s, session):
                            validated.append(s)
                        else:
                            stats.validation_failures["mismatch"] += 1
                    except Exception as exc:  # a seed that cannot even run is not a seed
                        stats.validation_failures[type(exc).__name__] += 1
                        log.debug("validate failed for %s: %s", s.record_id, exc)
                stats.seeds_validated += len(validated)

                for seed in validated:
                    if not seed.parseable:
                        continue  # pivot target only
                    if not seed.root_eligible:
                        continue  # retrieval pool only (e.g. an eval-split goal)
                    graph = _build_graph(
                        adapter, session, seed, validated, env_id, env_spec,
                        branching, min_moved, caps, gt_equal, config, cfg_hash, stats,
                    )
                    if graph is not None:
                        stats.graphs_emitted += 1
                        yield graph
        except Exception as exc:
            log.warning("environment %s failed: %s", env_key, exc)
            stats.validation_failures[f"env:{type(exc).__name__}"] += 1


def _canonicalize(adapter, candidates):
    """Let an adapter keep condition identifiers consistent after a generic edit.

    Optional hook (probed with getattr, like gt_equal/is_valid_intent): adapters whose slot
    identifiers encode part of the value need this after substitution.
    """
    fn = getattr(adapter, "canonicalize_conditions", None)
    if fn is None:
        return candidates
    out = []
    for c in candidates:
        conds = fn(c.conditions)
        out.append(c if conds == c.conditions else replace(c, conditions=conds))
    return out


def _build_graph(adapter, session, seed, validated, env_id, env_spec, branching,
                min_moved, caps, gt_equal, config, cfg_hash, stats) -> Graph | None:
    seed_recipe = adapter.compile(seed.base, seed.conditions)
    try:
        seed_gt = adapter.execute(seed_recipe, session)
    except Exception as exc:
        log.debug("seed %s failed to execute: %s", seed.record_id, exc)
        return None
    if seed_gt.is_empty:
        return None

    # --- retrieval first (D10), then enumerate only to fill gaps or spare capacity -----
    candidates = real_candidates(seed, validated)
    have_ops = {c.operator for c in candidates}
    need_capacity = len(candidates) < branching

    def wanted(op: Operator) -> bool:
        # Enumerate when retrieval produced nothing for this operator (a gap), or when
        # retrieval did not fill the layer.  Pivots are retrieval-only: a pivot target
        # must be an already-validated intent, never an invented one.
        return op not in have_ops or need_capacity

    if wanted(Operator.RELAXATION):
        candidates += enumerate_relaxations(seed, caps[Operator.RELAXATION])
    if wanted(Operator.REFINEMENT):
        try:
            witness = adapter.witness(seed.base, seed.conditions, session)
            candidates += enumerate_refinements(seed, witness, caps[Operator.REFINEMENT])
        except Exception as exc:
            log.debug("witness failed for %s: %s", seed.record_id, exc)
    if wanted(Operator.SUBSTITUTION):
        try:
            domains = {slot: adapter.domains(slot, seed.base, session)
                       for slot in {s for s, _, _ in seed.conditions}}
            candidates += enumerate_substitutions(seed, domains, caps[Operator.SUBSTITUTION])
        except Exception as exc:
            log.debug("domains failed for %s: %s", seed.record_id, exc)

    # adapter-specific candidates the generic enumerators cannot express (optional hook):
    # e.g. retail relaxation = dropping a WHOLE request (all of one write action's
    # conditions together) -- the one-condition-at-a-time enumerator would instead
    # produce a write with a missing required argument, which the shop rejects.
    extra = getattr(adapter, "extra_candidates", None)
    if extra is not None:
        try:
            candidates += list(extra(seed))
        except Exception as exc:
            log.debug("extra_candidates failed for %s: %s", seed.record_id, exc)

    # keep slot identifiers in step with edited values (adapter-optional, see _canonicalize)
    candidates = _canonicalize(adapter, candidates)

    # drop candidates outside the benchmark's own intent space before we ever run them.
    # WebShop, for instance, divides by the attribute count, so a goal with zero
    # attributes is not a task its reward function can score -- we decline to invent one
    # rather than patch the verifier (north-star #2).
    is_valid = getattr(adapter, "is_valid_intent", None)
    if is_valid is not None:
        candidates = [c for c in candidates if is_valid(c.base, c.conditions)]

    # de-duplicate by (base, conditions), keeping REAL over SYNTHETIC
    dedup: dict[tuple, Candidate] = {}
    for c in candidates:
        key = (str(sorted(c.base.items())), c.conditions)
        if key == (str(sorted(seed.base.items())), seed.conditions):
            continue  # the seed is not its own branch
        prev = dedup.get(key)
        if prev is None or (prev.provenance is Provenance.SYNTHETIC and c.provenance is Provenance.REAL):
            dedup[key] = c
    candidates = list(dedup.values())
    for c in candidates:
        stats.candidates_generated[c.operator.value] += 1

    # --- execute + gate ---------------------------------------------------------------
    results: list[GateResult] = []
    for c in candidates:
        recipe = adapter.compile(c.base, c.conditions)
        try:
            gt = adapter.execute(recipe, session)
            res = admit(c, gt, seed_gt, gt_equal=gt_equal)
        except Exception as exc:
            res = admit(c, None, seed_gt, gt_equal=gt_equal, error=type(exc).__name__)
        results.append(res)
    for k, v in rejection_summary(results).items():
        stats.rejections[k] += v
    for r in results:
        if r.admitted:
            stats.candidates_gated_ok[r.candidate.operator.value] += 1

    balanced = bool(config.get("balanced_branching", True))
    require_full = bool(config.get("require_full_quota", True))
    picked = rank_and_cap(results, seed_gt, branching, balanced=balanced,
                          shortfall=stats.quota_shortfall)
    if balanced:
        full = len(picked) == branching
        stats.quota_filled["full" if full else "partial"] += 1
        if require_full and not full:
            # A graph missing a whole operator category cannot honour a category-probability
            # vector that asks for it, so the shift sampler would silently renormalise and
            # the realised mix would stop matching the hyperparameter. Dropping is the
            # honest option; `require_full_quota: false` keeps partial graphs.
            stats.graphs_dropped_partial_quota += 1
            return None
    if not picked:
        stats.graphs_dropped_no_branches += 1
        return None
    if sum(1 for r in picked if r.gt_moved) < min_moved:
        stats.graphs_dropped_no_moved_edge += 1
        return None

    # --- assemble ---------------------------------------------------------------------
    extensional = getattr(adapter, "gt_extensional", None)

    def mk(**kw) -> Node:
        if extensional is not None and "recipe" in kw:
            kw.setdefault("gt_extensional", extensional(kw["recipe"]))
        return Node.build(adapter=adapter.name, adapter_version=adapter.version,
                          env_id=env_id, **kw)

    # The seed IS the initial intent: it is what the user actually wants, it is real and
    # already validated, and the relaxation branches supply the vaguer readings.  Its
    # soft_slots record which conditions a rendering stage may withhold.
    root_node = mk(base=seed.base, conditions=seed.conditions, recipe=seed_recipe,
                   ground_truth=seed_gt, is_seed=True, source_record=seed.record_id,
                   unbound_slots=soft_slots(seed, [r.candidate for r in picked]))

    child_nodes: list[Node] = []
    edges: list[Edge] = []
    for r in picked:
        c = r.candidate
        node = mk(base=c.base, conditions=c.conditions,
                  recipe=adapter.compile(c.base, c.conditions),
                  ground_truth=r.ground_truth, source_record=c.source_record)
        child_nodes.append(node)
        edges.append(Edge(src=root_node.intent_id, dst=node.intent_id, operator=c.operator,
                          delta=c.delta, provenance=c.provenance, gt_moved=r.gt_moved,
                          source_record=c.source_record))
        stats.candidates_picked[c.operator.value] += 1
        stats.provenance[c.provenance.value] += 1
        stats.gt_moved["moved" if r.gt_moved else "unmoved"] += 1

    graph = Graph.build(adapter=adapter.name, adapter_version=adapter.version, env_id=env_id,
                      env_spec=env_spec, root=root_node, children=child_nodes, edges=edges,
                      config={k: v for k, v in config.items() if k not in ("paths", "workers")},
                      cfg_hash=cfg_hash)
    graph.assert_single_environment()
    return graph
