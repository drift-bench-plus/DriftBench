"""Core data model for intent graphs.

An *intent* is a goal expressed as a set of conditions over a base frame.  Its *ground
truth* is obtained by compiling those conditions into a recipe and executing it in an
environment -- never authored, never judged by a model.

Deliberately absent: any natural-language text.  What the simulated user says (and any
perturbation of it) is produced by a later, separate stage that consumes these graphs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal

from .ids import canonical_dumps, content_hash, intent_id, graph_id

# (slot, op, value) -- e.g. ("size", "=", "small") / ("prc", ">", 50) / ("mtime", "<", 7)
Condition = tuple[str, str, Any]

GTKind = Literal["rowset", "purchaseset", "scalar", "statehash"]


class Operator(StrEnum):
    """The four intent-shift moves (see plan D5/D6, taxonomy M1/M2/M3/M5)."""

    REFINEMENT = "REFINEMENT"      # add one condition -> answer set narrows
    RELAXATION = "RELAXATION"      # drop one condition -> answer set widens
    SUBSTITUTION = "SUBSTITUTION"  # change one condition's value
    PIVOT = "PIVOT"                # different goal, same environment


class Provenance(StrEnum):
    REAL = "REAL"            # a record that already exists in the dataset
    SYNTHETIC = "SYNTHETIC"  # enumerated by us and admitted by the gate


def sort_conditions(conds) -> tuple[Condition, ...]:
    """Canonical condition order: identity must not depend on discovery order."""
    return tuple(sorted(((str(s), str(o), v) for s, o, v in conds), key=lambda c: canonical_dumps(c)))


@dataclass(frozen=True, slots=True)
class GroundTruth:
    """The executed answer, in a form that can be compared and hashed.

    ``value`` must already be canonical (sorted collections, normalized scalars); use
    the ``from_*`` constructors rather than building one by hand.
    """

    kind: GTKind
    value: Any
    hash: str

    @staticmethod
    def _make(kind: GTKind, value: Any) -> GroundTruth:
        return GroundTruth(kind=kind, value=value, hash=content_hash(kind, value, length=32))

    @classmethod
    def rowset(cls, rows) -> GroundTruth:
        """SQL SELECT results: order-independent multiset of stringified rows."""
        return cls._make("rowset", sorted(canonical_dumps(list(r)) for r in rows))

    @classmethod
    def purchaseset(cls, purchases) -> GroundTruth:
        """WebShop: {(asin, {option: value})} scoring maximal under the shipped reward."""
        norm = [[a, sorted((str(k), str(v)) for k, v in dict(o).items())] for a, o in purchases]
        return cls._make("purchaseset", sorted(canonical_dumps(p) for p in norm))

    @classmethod
    def scalar(cls, value: Any) -> GroundTruth:
        """OS tasks: the stdout of the reference command (compared by the task's matcher)."""
        return cls._make("scalar", "" if value is None else str(value).strip())

    @classmethod
    def statehash(cls, digest: str) -> GroundTruth:
        """DB write tasks: the post-execution table hash (DBBench's own pipeline)."""
        return cls._make("statehash", str(digest))

    @property
    def is_empty(self) -> bool:
        """Hard admission gate: an intent nobody can satisfy is not a usable branch."""
        if self.kind in ("rowset", "purchaseset"):
            return len(self.value) == 0
        if self.kind == "scalar":
            return self.value == ""
        return False  # a state hash always exists

    def contains(self, item) -> bool:
        """Is this a member of the answer set?

        Acceptance is set membership, never equality with one canonical answer: WebShop
        ground-truth sets have median 2 and reach 140 members.  ``item`` is canonicalized
        exactly as the ``from_*`` constructors do, because ``value`` holds canonical-JSON
        strings rather than raw objects.
        """
        if self.kind == "rowset":
            return canonical_dumps(list(item)) in set(self.value)
        if self.kind == "purchaseset":
            asin, options = item
            norm = [asin, sorted((str(k), str(v)) for k, v in dict(options).items())]
            return canonical_dumps(norm) in set(self.value)
        if self.kind == "scalar":
            return str(item).strip() == self.value
        if self.kind == "statehash":
            return str(item) == self.value
        return False

    def rows(self) -> list:
        """Decode a rowset back into real rows (``value`` stores canonical JSON strings)."""
        if self.kind != "rowset":
            raise ValueError(f"rows() is only defined for rowset, not {self.kind}")
        return [json.loads(v) for v in self.value]

    def monotonic_view(self) -> set | None:
        """The part of the answer that subset/superset reasoning applies to.

        A WebShop answer is a (product, chosen options) pair, and the option selection is
        a function of the goal: drop an option requirement and the correct selection
        changes even though the same products still qualify.  So monotonicity is a claim
        about *products*, not about the full tuple.  Row sets have no such split.
        """
        if self.kind == "purchaseset":
            return {canonical_dumps(json.loads(v)[0]) for v in self.value}
        if self.kind == "rowset":
            return set(self.value)
        return None

    def cardinality(self) -> int:
        if self.kind in ("rowset", "purchaseset"):
            return len(self.value)
        return 1

    def to_dict(self) -> dict:
        return {"kind": self.kind, "value": self.value, "hash": self.hash}

    @classmethod
    def from_dict(cls, d: dict) -> GroundTruth:
        return cls(kind=d["kind"], value=d["value"], hash=d["hash"])


@dataclass(frozen=True, slots=True)
class Seed:
    """A validated dataset record: the raw material a graph is grown from."""

    record_id: str
    env_key: str
    base: dict                       # frame: what kind of thing is asked, over what target
    conditions: tuple[Condition, ...] | None  # None => unparseable (pivot target only)
    shipped_answer: Any = None       # what the dataset claims; used only by validate()
    meta: dict = field(default_factory=dict)
    # False => may still be retrieved as somebody else's branch, but never becomes a root.
    # This is how a benchmark split is honoured without starving retrieval: WebShop roots
    # come from the 500-goal test split, while eval-split goals stay available as real
    # sibling intents. Nothing about a branch leaks the split it came from.
    root_eligible: bool = True

    @property
    def parseable(self) -> bool:
        return self.conditions is not None


@dataclass(frozen=True, slots=True)
class Node:
    intent_id: str
    conditions: tuple[Condition, ...]
    bound_slots: tuple[str, ...]
    unbound_slots: tuple[str, ...]   # root only: slots its children bind but it does not
    recipe: Any
    ground_truth: GroundTruth
    base: dict
    is_seed: bool = False
    source_record: str | None = None
    # True when ground truth IS the set of matching objects, so refinement must narrow it
    # and relaxation must widen it.  False for computed answers (COUNT/SUM/a shell count),
    # where the value changes without any subset relation holding.
    gt_extensional: bool = True

    @classmethod
    def build(cls, *, adapter, adapter_version, env_id, base, conditions, recipe,
              ground_truth, bound_slots=None, unbound_slots=(), is_seed=False,
              source_record=None, gt_extensional=True) -> Node:
        conds = sort_conditions(conditions)
        bound = tuple(bound_slots) if bound_slots is not None else tuple(sorted({c[0] for c in conds}))
        return cls(
            intent_id=intent_id(adapter, adapter_version, env_id, base, conds),
            conditions=conds,
            bound_slots=bound,
            unbound_slots=tuple(sorted(unbound_slots)),
            recipe=recipe,
            ground_truth=ground_truth,
            base=base,
            is_seed=is_seed,
            source_record=source_record,
            gt_extensional=gt_extensional,
        )

    def to_dict(self) -> dict:
        return {
            "intent_id": self.intent_id,
            "conditions": [list(c) for c in self.conditions],
            "bound_slots": list(self.bound_slots),
            "unbound_slots": list(self.unbound_slots),
            "recipe": self.recipe,
            "ground_truth": self.ground_truth.to_dict(),
            "base": self.base,
            "is_seed": self.is_seed,
            "source_record": self.source_record,
            "gt_extensional": self.gt_extensional,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Node:
        return cls(
            intent_id=d["intent_id"],
            conditions=tuple(tuple(c) for c in d["conditions"]),
            bound_slots=tuple(d["bound_slots"]),
            unbound_slots=tuple(d["unbound_slots"]),
            recipe=d["recipe"],
            ground_truth=GroundTruth.from_dict(d["ground_truth"]),
            base=d["base"],
            is_seed=d.get("is_seed", False),
            source_record=d.get("source_record"),
            gt_extensional=d.get("gt_extensional", True),
        )


@dataclass(frozen=True, slots=True)
class Edge:
    src: str
    dst: str
    operator: Operator
    delta: dict
    provenance: Provenance
    gt_moved: bool
    source_record: str | None = None

    def to_dict(self) -> dict:
        return {
            "src": self.src, "dst": self.dst,
            "operator": self.operator.value,
            "delta": self.delta,
            "provenance": self.provenance.value,
            "gt_moved": self.gt_moved,
            "source_record": self.source_record,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Edge:
        return cls(
            src=d["src"], dst=d["dst"],
            operator=Operator(d["operator"]),
            delta=d["delta"],
            provenance=Provenance(d["provenance"]),
            gt_moved=d["gt_moved"],
            source_record=d.get("source_record"),
        )


@dataclass(frozen=True, slots=True)
class Graph:
    graph_id: str
    adapter: str
    adapter_version: str
    env_id: str
    env_spec: dict
    root: Node
    children: tuple[Node, ...]
    edges: tuple[Edge, ...]
    config: dict

    @classmethod
    def build(cls, *, adapter, adapter_version, env_id, env_spec, root, children, edges,
              config, cfg_hash) -> Graph:
        return cls(
            graph_id=graph_id(env_id, root.intent_id, [c.intent_id for c in children], cfg_hash),
            adapter=adapter, adapter_version=adapter_version,
            env_id=env_id, env_spec=env_spec,
            root=root, children=tuple(children), edges=tuple(edges), config=config,
        )

    def to_dict(self) -> dict:
        return {
            "graph_id": self.graph_id,
            "adapter": self.adapter,
            "adapter_version": self.adapter_version,
            "env_id": self.env_id,
            "env_spec": self.env_spec,
            "root": self.root.to_dict(),
            "children": [c.to_dict() for c in self.children],
            "edges": [e.to_dict() for e in self.edges],
            "config": self.config,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Graph:
        return cls(
            # BACKWARD READ (2026-08-19 rename): 1,560 stored artifacts and every
            # trajectory header written before today carry "tree_id". They are the same
            # id under a name we no longer use, so they are read, not migrated. New
            # writes always emit "graph_id" (see to_dict).
            graph_id=d.get("graph_id") or d["tree_id"],
            adapter=d["adapter"], adapter_version=d["adapter_version"],
            env_id=d["env_id"], env_spec=d["env_spec"],
            root=Node.from_dict(d["root"]),
            children=tuple(Node.from_dict(c) for c in d["children"]),
            edges=tuple(Edge.from_dict(e) for e in d["edges"]),
            config=d["config"],
        )

    def assert_single_environment(self) -> None:
        """Invariant: one graph, one environment (plan D6 / section 3)."""
        node_envs = {n.base.get("env_id", self.env_id) for n in (self.root, *self.children)}
        if node_envs != {self.env_id}:
            raise ValueError(f"graph {self.graph_id} spans environments {node_envs}")


@dataclass(frozen=True, slots=True)
class Candidate:
    """A proposed branch before the admission gate has run."""

    conditions: tuple[Condition, ...]
    base: dict
    operator: Operator
    delta: dict
    provenance: Provenance
    source_record: str | None = None
    is_seed: bool = False
