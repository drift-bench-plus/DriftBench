"""Graph persistence: canonical JSON on disk plus a queryable parquet index.

Graphs are stored as ``artifacts/graphs/{adapter}/{env_id}/{graph_id}.json`` written through
the same canonical dumper that produces the ids, so regenerating identical graphs produces
byte-identical files (and golden fixtures stay stable).

Conditions are the canonical form: a recipe is always ``compile(base, conditions)`` and is
stored only as a convenience/cache.  Ground truth is likewise cached but re-verifiable --
``verify_gt`` re-executes and compares, which is how we detect environment drift or a
corrupted cache rather than trusting a stored answer forever.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path

from .ids import canonical_dumps
from .models import Graph


def graph_path(root: Path, graph: Graph) -> Path:
    return Path(root) / "graphs" / graph.adapter / graph.env_id / f"{graph.graph_id}.json"


def write_graph(root: Path, graph: Graph) -> Path:
    path = graph_path(root, graph)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_dumps(graph.to_dict(), indent=1), encoding="utf-8")
    return path


def read_graph(path: Path) -> Graph:
    return Graph.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


LEGACY_DIR = "trees"          # pre-2026-08-19 layout; read, never written


def iter_graphs(root: Path, adapter: str | None = None) -> Iterator[Graph]:
    """Every stored graph, from the current layout and the legacy one.

    Graphs written before the 2026-08-19 rename live under ``artifacts/trees/`` and key
    their id as ``tree_id``. Both are read here so an existing corpus keeps working
    without a migration pass; ``write_graph`` only ever writes the new layout, so the
    legacy directory drains naturally as data is regenerated.
    """
    seen: set[str] = set()
    for top in ("graphs", LEGACY_DIR):
        base = Path(root) / top
        if adapter:
            base = base / adapter
        if not base.exists():
            continue
        for path in sorted(base.rglob("*.json")):
            g = read_graph(path)
            if g.graph_id in seen:
                continue          # regenerated into the new layout; prefer the new one
            seen.add(g.graph_id)
            yield g


def index_rows(graphs: Iterable[Graph]) -> list[dict]:
    rows = []
    for t in graphs:
        ops = [e.operator.value for e in t.edges]
        rows.append({
            "graph_id": t.graph_id,
            "adapter": t.adapter,
            "adapter_version": t.adapter_version,
            "env_id": t.env_id,
            "n_children": len(t.children),
            "n_edges": len(t.edges),
            "n_real": sum(1 for e in t.edges if e.provenance.value == "REAL"),
            "n_synthetic": sum(1 for e in t.edges if e.provenance.value == "SYNTHETIC"),
            "n_gt_moved": sum(1 for e in t.edges if e.gt_moved),
            "n_refinement": ops.count("REFINEMENT"),
            "n_relaxation": ops.count("RELAXATION"),
            "n_substitution": ops.count("SUBSTITUTION"),
            "n_pivot": ops.count("PIVOT"),
            "root_gt_kind": t.root.ground_truth.kind,
            "root_gt_card": t.root.ground_truth.cardinality(),
            "n_soft_slots": len(t.root.unbound_slots),
            "seed_record": t.root.source_record,
        })
    return rows


def write_index(root: Path, graphs: Iterable[Graph]) -> Path:
    import pandas as pd

    rows = index_rows(graphs)
    path = Path(root) / "index.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path)
    return path


def verify_gt(graph: Graph, adapter, executor, *, sample: int | None = None) -> list[dict]:
    """Re-execute stored recipes and report any node whose ground truth has drifted."""
    nodes = [graph.root, *graph.children]
    if sample is not None:
        nodes = nodes[:sample]
    problems = []
    with executor.open(graph.env_spec) as session:
        for node in nodes:
            mutating = adapter.is_mutating(node.recipe)
            try:
                gt = adapter.execute(node.recipe, session) if not mutating else _fresh_execute(
                    adapter, session, node.recipe
                )
            except Exception as exc:
                problems.append({"intent_id": node.intent_id, "issue": f"error:{type(exc).__name__}"})
                continue
            if gt.hash != node.ground_truth.hash:
                problems.append({
                    "intent_id": node.intent_id,
                    "issue": "gt_drift",
                    "stored": node.ground_truth.hash,
                    "recomputed": gt.hash,
                })
    return problems


def _fresh_execute(adapter, session, recipe):
    session.rematerialize()
    return adapter.execute(recipe, session)
