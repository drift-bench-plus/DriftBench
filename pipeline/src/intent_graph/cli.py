"""Command-line entry points."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import typer
import yaml

from . import storage
from .adapters import base as registry
from .engine import RunStats
from .engine import generate as engine_generate
from .ids import canonical_dumps

app = typer.Typer(add_completion=False, help="Intent graph generation")

REPO_ROOT = Path(__file__).resolve().parents[2]


def _home() -> Path:
    """The benchmark home: the directory holding this run's config/ and artifacts/.

    Resolution order: the IG_HOME environment variable, then the current working
    directory when it carries a config/default.yaml (the documented way to run the
    per-benchmark drivers is from their benchmarks/<name>/ directory), then the
    package root as a last resort.  The pipeline package itself ships no config:
    graph identity hashes over config content, so each benchmark owns its own
    config/ byte-for-byte under benchmarks/<name>/config/.
    """
    env = os.environ.get("IG_HOME")
    if env:
        return Path(env).resolve()
    if (Path.cwd() / "config" / "default.yaml").exists():
        return Path.cwd()
    return REPO_ROOT


HOME = _home()
DEFAULT_CONFIG = HOME / "config" / "default.yaml"


def load_config(path: Path | None = None) -> dict:
    cfg = yaml.safe_load(Path(path or DEFAULT_CONFIG).read_text(encoding="utf-8"))
    # paths are resolved against the repo root, never the caller's cwd
    for key, value in (cfg.get("paths") or {}).items():
        p = Path(value)
        cfg["paths"][key] = str(p if p.is_absolute() else (HOME / p))
    return cfg


def build(adapter_name: str, cfg: dict):
    registry.load_builtin()
    adapter_cls, executor_cls = registry.get(adapter_name)
    return adapter_cls(cfg) if _takes_config(adapter_cls) else adapter_cls(), \
        executor_cls(cfg) if _takes_config(executor_cls) else executor_cls()


def _takes_config(cls) -> bool:
    """True if the class defines its own __init__ that accepts a config argument."""
    import inspect

    if cls.__init__ is object.__init__:
        return False
    try:
        params = inspect.signature(cls.__init__).parameters
    except (TypeError, ValueError):
        return False
    return any(
        name != "self" and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        for name, p in params.items()
    )


@app.command()
def adapters() -> None:
    """List registered adapters."""
    registry.load_builtin()
    for name in registry.available():
        typer.echo(name)


@app.command()
def generate(
    adapter: str = typer.Option(..., "--adapter", "-a"),
    config: Path = typer.Option(None, "--config", "-c"),
    limit_envs: int = typer.Option(None, "--limit-envs"),
    shard: str = typer.Option(None, "--shard",
                              help="i/n -- generate every n-th environment (parallel sweeps)"),
    out: Path = typer.Option(None, "--out"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Generate intent graphs for an adapter."""
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config(config)
    art = Path(out or cfg["paths"]["artifacts"])
    adp, ex = build(adapter, cfg)
    stats = RunStats()
    graphs = []
    try:
        parts = None
        if shard:
            i, n = (int(x) for x in shard.split("/"))
            parts = (i, n)
        for graph in engine_generate(adp, ex, cfg, limit_envs=limit_envs, shard=parts,
                                    stats=stats):
            storage.write_graph(art, graph)
            graphs.append(graph)
    finally:
        ex.shutdown()
    if graphs:
        storage.write_index(art, graphs)
    stats_path = Path(cfg["paths"]["reports"]) / f"{adapter}_stats.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(canonical_dumps(stats.to_dict(), indent=1), encoding="utf-8")
    typer.echo(canonical_dumps(stats.to_dict(), indent=1))


@app.command()
def measure(config: Path = typer.Option(None, "--config", "-c")) -> None:
    """Summarize everything generated so far."""
    cfg = load_config(config)
    graphs = list(storage.iter_graphs(Path(cfg["paths"]["artifacts"])))
    rows = storage.index_rows(graphs)
    if not rows:
        typer.echo("no graphs found")
        raise typer.Exit(1)
    import pandas as pd

    df = pd.DataFrame(rows)
    summary = df.groupby("adapter").agg(
        graphs=("graph_id", "count"),
        environments=("env_id", "nunique"),
        mean_children=("n_children", "mean"),
        real_edges=("n_real", "sum"),
        synthetic_edges=("n_synthetic", "sum"),
        moved_edges=("n_gt_moved", "sum"),
        refinement=("n_refinement", "sum"),
        relaxation=("n_relaxation", "sum"),
        substitution=("n_substitution", "sum"),
        pivot=("n_pivot", "sum"),
    )
    typer.echo(summary.to_string())


@app.command()
def verify_gt(
    adapter: str = typer.Option(..., "--adapter", "-a"),
    config: Path = typer.Option(None, "--config", "-c"),
    sample: int = typer.Option(10, "--sample"),
) -> None:
    """Re-execute stored recipes and report ground-truth drift."""
    cfg = load_config(config)
    adp, ex = build(adapter, cfg)
    problems = []
    try:
        for i, graph in enumerate(storage.iter_graphs(Path(cfg["paths"]["artifacts"]), adapter)):
            if i >= sample:
                break
            problems.extend(storage.verify_gt(graph, adp, ex))
    finally:
        ex.shutdown()
    typer.echo(json.dumps(problems, indent=1) if problems else "all sampled ground truths reproduce")
    raise typer.Exit(1 if problems else 0)


@app.command()
def manifest(
    action: str = typer.Argument("verify", help="write | verify"),
    config: Path = typer.Option(None, "--config", "-c"),
    full: bool = typer.Option(False, "--full", help="recompute sha256 instead of size+mtime"),
) -> None:
    """Record or check checksums of the input datasets."""
    from .manifest import verify_manifest, write_manifest

    cfg = load_config(config)
    if action == "write":
        path = write_manifest(cfg)
        typer.echo(f"wrote {path}")
    else:
        bad = verify_manifest(cfg, full=full)
        typer.echo("manifest ok" if not bad else json.dumps(bad, indent=1))
        raise typer.Exit(1 if bad else 0)


# `it episode ...` lives in its own Typer app (add_typer, not @app.command: `it battery`
# already exists for graph structure)
def _mount_runtime() -> None:
    try:
        from .runtime.cli import app as episode_app
    except ImportError as exc:  # optional deps absent
        import logging
        logging.getLogger(__name__).debug("runtime CLI unavailable: %s", exc)
        return
    app.add_typer(episode_app, name="episode")


_mount_runtime()


if __name__ == "__main__":
    app()


@app.command()
def battery(
    adapter: str = typer.Option(None, "--adapter", "-a"),
    config: Path = typer.Option(None, "--config", "-c"),
    reproduce: bool = typer.Option(False, "--reproduce", help="also re-execute every node"),
    limit: int = typer.Option(None, "--limit"),
) -> None:
    """Run the consistency battery over emitted graphs."""
    from . import battery as bat

    cfg = load_config(config)
    graphs = list(storage.iter_graphs(Path(cfg["paths"]["artifacts"]), adapter))
    if limit:
        graphs = graphs[:limit]
    adp = ex = None
    if reproduce and adapter:
        adp, ex = build(adapter, cfg)
    try:
        findings = bat.run(graphs, adp, ex, reproduce=reproduce)
    finally:
        if ex is not None:
            ex.shutdown()
    typer.echo(f"{len(graphs)} graphs checked")
    if findings:
        typer.echo(canonical_dumps(bat.summarize(findings), indent=1))
        for f in findings[:15]:
            typer.echo(f"  {f.check}: {f.graph_id} {f.detail}")
        raise typer.Exit(1)
    typer.echo("battery clean")
