"""Regenerating the golden fixture, shared by the maker script and the test.

Why a byte-exact fixture exists at all: determinism was only checked by generating twice
inside one process and comparing ids, which catches randomness but not a change in what a
graph *contains*. Reorder conditions, or change what goes into a ground-truth record, and the
ids move consistently while every test still passes.

That is not hypothetical. Graph content changed twice on 2026-08-07 -- the cluster-pool fix
and the balanced-quota fix -- and both were found by reading output rather than by a test.
Two graph sets had to be quarantined afterwards because they still looked current.

The cluster is small and fixed so the check costs seconds.
"""

from __future__ import annotations

from pathlib import Path

from .cli import build, load_config
from .engine import RunStats, generate

# The smallest cluster that actually yields full-quota graphs (1,127 products, 2 graphs).
# A cluster whose only root cannot fill the 2/2/2/2 quota emits nothing and would pin an
# empty fixture, which proves nothing.
GOLDEN_CLUSTER = "area rugs"


def golden_path() -> Path:
    return Path(__file__).resolve().parents[2] / "tests" / "golden" / "webshop_cluster.json"


def regenerate() -> dict:
    """Generate the golden cluster's graphs and return a canonical payload."""
    cfg = load_config()
    adapter, executor = build("webshop", cfg)
    try:
        graphs = list(generate(adapter, executor, cfg, only_envs={GOLDEN_CLUSTER},
                              stats=RunStats()))
    finally:
        executor.shutdown()
    if not graphs:
        raise SystemExit(f"golden cluster {GOLDEN_CLUSTER!r} produced no graphs")
    return {
        "cluster": GOLDEN_CLUSTER,
        "config_fingerprint": {k: cfg.get(k) for k in
                               ("seed", "depth", "branching", "min_gt_moved_edges",
                                "balanced_branching", "require_full_quota")},
        "graphs": [t.to_dict() for t in sorted(graphs, key=lambda t: t.graph_id)],
    }
