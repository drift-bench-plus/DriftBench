"""Consistency battery: properties every emitted graph must satisfy.

These are the checks that would catch a silently-wrong pipeline -- a verifier that
accepts anything, ground truth that drifts, an operator whose label does not match what
it did to the answer.  Run over stored graphs, re-executing where needed.

  reproducible   re-executing a node's recipe reproduces its stored ground truth
                 (catches environment drift and cache corruption)
  non_empty      no node is an intent nobody can satisfy
  gt_moved_label the edge's gt_moved flag agrees with the two ground truths
  monotonic      REFINEMENT narrows the answer, RELAXATION widens it -- the operator
                 semantics themselves, checked rather than assumed
  discriminative at least one edge per graph actually moves the answer (D7)
  single_env     one graph, one environment (D6)
  one_seed       the root is the originating record and no child duplicates it
"""

from __future__ import annotations

from dataclasses import dataclass

from .gate import default_gt_equal
from .models import GroundTruth, Operator, Graph


@dataclass(frozen=True, slots=True)
class Finding:
    graph_id: str
    check: str
    detail: str


def _as_set(gt: GroundTruth) -> set | None:
    return gt.monotonic_view()


def check_structure(graph: Graph) -> list[Finding]:
    """Checks that need no execution."""
    out: list[Finding] = []
    try:
        graph.assert_single_environment()
    except ValueError as exc:
        out.append(Finding(graph.graph_id, "single_env", str(exc)))

    if not graph.root.is_seed:
        out.append(Finding(graph.graph_id, "one_seed", "root is not the seed intent"))
    stray = [c for c in graph.children if c.is_seed]
    if stray:
        out.append(Finding(graph.graph_id, "one_seed", f"{len(stray)} seed nodes among children"))

    for node in (graph.root, *graph.children):
        if node.ground_truth.is_empty:
            out.append(Finding(graph.graph_id, "non_empty", f"{node.intent_id} has empty GT"))

    if graph.edges and not any(e.gt_moved for e in graph.edges):
        out.append(Finding(graph.graph_id, "discriminative", "no edge moves the answer"))

    nodes = {n.intent_id: n for n in (graph.root, *graph.children)}
    for edge in graph.edges:
        src, dst = nodes.get(edge.src), nodes.get(edge.dst)
        if src is None or dst is None:
            out.append(Finding(graph.graph_id, "dangling_edge", f"{edge.src}->{edge.dst}"))
            continue

        moved = not default_gt_equal(src.ground_truth, dst.ground_truth)
        if moved != edge.gt_moved:
            # adapters may override equality (e.g. DBBench float tolerance), so a
            # disagreement here is reported rather than asserted
            out.append(Finding(graph.graph_id, "gt_moved_label",
                               f"edge says {edge.gt_moved}, hashes say {moved}"))

        a, b = _as_set(src.ground_truth), _as_set(dst.ground_truth)
        # monotonicity is only meaningful when GT is the matching set itself; an
        # aggregate answer (COUNT/SUM) changes value without any subset relation
        if a is not None and b is not None and src.gt_extensional and dst.gt_extensional:
            if edge.operator is Operator.REFINEMENT and not b <= a:
                out.append(Finding(graph.graph_id, "monotonic",
                                   f"refinement result is not a subset "
                                   f"({len(a)}->{len(b)}, {len(b - a)} new)"))
            if edge.operator is Operator.RELAXATION and not b >= a:
                out.append(Finding(graph.graph_id, "monotonic",
                                   f"relaxation result is not a superset "
                                   f"({len(a)}->{len(b)}, {len(a - b)} lost)"))
    return out


def check_reproducible(graph: Graph, adapter, executor) -> list[Finding]:
    """Re-execute every node and compare against what was stored."""
    out: list[Finding] = []
    with executor.open(graph.env_spec) as session:
        for node in (graph.root, *graph.children):
            try:
                mutating = adapter.is_mutating(node.recipe)
                if mutating:
                    session.rematerialize()
                gt = adapter.execute(node.recipe, session)
            except Exception as exc:
                out.append(Finding(graph.graph_id, "reproducible",
                                   f"{node.intent_id}: {type(exc).__name__}"))
                continue
            if gt.hash != node.ground_truth.hash:
                out.append(Finding(graph.graph_id, "reproducible",
                                   f"{node.intent_id}: stored {node.ground_truth.hash[:8]} "
                                   f"recomputed {gt.hash[:8]}"))
    return out


def run(graphs, adapter=None, executor=None, *, reproduce: bool = False) -> list[Finding]:
    findings: list[Finding] = []
    for graph in graphs:
        findings.extend(check_structure(graph))
        if reproduce and adapter is not None and executor is not None:
            findings.extend(check_reproducible(graph, adapter, executor))
    return findings


def summarize(findings: list[Finding]) -> dict[str, int]:
    out: dict[str, int] = {}
    for f in findings:
        out[f.check] = out.get(f.check, 0) + 1
    return out
