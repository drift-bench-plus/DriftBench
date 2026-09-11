"""`it episode ...` — run, validate and report on interaction episodes."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import typer
import yaml

from .. import storage
from ..cli import HOME as REPO_ROOT, build, load_config
from ..ids import canonical_dumps
from . import metrics as M
from . import scripted as S
from . import strategies as st
from .episode import Episode
from .llm import LLMClient
from .persona import PERSONA_IDS

app = typer.Typer(add_completion=False, help="Interaction episodes over intent graphs")
log = logging.getLogger(__name__)

RUNTIME_CONFIG = REPO_ROOT / "config" / "runtime.yaml"


def _coerce(val: str):
    """'11' -> 11, 'true' -> True, '0.2' -> 0.2, anything else stays a string."""
    low = val.lower()
    if low in ("true", "yes", "on"):
        return True
    if low in ("false", "no", "off"):
        return False
    for cast in (int, float):
        try:
            return cast(val)
        except ValueError:
            pass          # not this type; fall through to the next, then to str
    return val


def load_runtime(config: Path | None = None, runtime: Path | None = None) -> dict:
    """Graph config and runtime config are separate files on purpose.

    `config_hash()` excludes only `paths` and `workers`, and `Graph.build` folds that hash
    into every `graph_id` -- so putting runtime settings in default.yaml would change every
    graph id and bake the LLM settings into every graph file.
    """
    cfg = load_config(config)
    rt = yaml.safe_load(Path(runtime or RUNTIME_CONFIG).read_text(encoding="utf-8"))
    cfg["runtime"] = rt["runtime"]
    cfg["llm"] = rt["llm"]
    return cfg


def make_llm(cfg: dict, *, fake: bool):
    if fake:
        return S.stub_llm()
    client = LLMClient(cfg)
    if not client.offline:
        log.info("llm: live calls enabled (model=%s)", client.model)
    return client


def _graphs(cfg: dict, adapter: str, limit: int | None):
    root = Path(cfg["runtime"].get("graphs_dir", cfg["paths"]["artifacts"]))
    if not root.is_absolute():
        root = REPO_ROOT / root
    out = []
    for graph in storage.iter_graphs(root, adapter):
        if not graph.root.is_seed:
            raise SystemExit(
                f"stale graph {graph.graph_id}: root is not the seed. Regenerate with "
                f"`it generate --adapter {adapter}` (see plan section 0.1)."
            )
        out.append(graph)
        if limit and len(out) >= limit:
            break
    if not out:
        raise SystemExit(f"no graphs for adapter {adapter!r} under {root}")
    return out


def _oracle_for(graph, header) -> S.Oracle:
    spec = (header.get("perturbation") or {}).get("hidden_slots") or []
    ordered = [s for s, _, _ in sorted(graph.root.conditions)]
    ids = [f"slot_{ordered.index(s)}" for s in spec if s in ordered]
    return S.Oracle(graph=graph, hidden_slot_ids=ids)


def _run_one(graph, adapter, executor, cfg, llm, agent_factory, persona=None, seed=None):
    """Probe for the mask, then play with a script that knows which slots to ask about.

    The episode is constructed *after* the probe rather than wrapping a placeholder agent:
    a forwarding wrapper would only proxy ``act`` and would silently swallow the
    ``observe_node`` hook, turning the oracle into the intent-ignorer.
    """
    probe = Episode(graph=graph, adapter=adapter, executor=executor, config=cfg,
                    llm=llm, agent=S.NoOp(graph=graph), persona_name=persona, seed=seed).run()
    agent = agent_factory(graph, probe.header)
    ep = Episode(graph=graph, adapter=adapter, executor=executor, config=cfg, llm=llm,
                 agent=agent, persona_name=persona, seed=seed)
    return ep.run(), probe


@app.command()
def run(
    adapter: str = typer.Option(..., "--adapter", "-a"),
    limit: int = typer.Option(5, "--limit"),
    persona: str = typer.Option(None, "--persona"),
    seed: int = typer.Option(None, "--seed"),
    fake_llm: bool = typer.Option(True, "--fake-llm/--live-llm",
                                 help="fake = deterministic stub, no network"),
    out: Path = typer.Option(None, "--out"),
    config: Path = typer.Option(None, "--config"),
    verbose: bool = typer.Option(False, "-v"),
) -> None:
    """Run oracle episodes over real graphs — the smoke test that an environment is winnable."""
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    cfg = load_runtime(config)
    adp, ex = build(adapter, cfg)
    llm = make_llm(cfg, fake=fake_llm)
    graphs = _graphs(cfg, adapter, limit)
    results = []
    try:
        for graph in graphs:
            traj, _ = _run_one(graph, adp, ex, cfg, llm, _oracle_for, persona, seed)
            results.append(M.score(traj))
            if out:
                d = Path(out) / adapter
                d.mkdir(parents=True, exist_ok=True)
                (d / f"{graph.graph_id}.json").write_text(
                    canonical_dumps(traj.to_dict(), indent=1), encoding="utf-8")
    finally:
        ex.shutdown()
    typer.echo(canonical_dumps(M.aggregate(results), indent=1))


@app.command()
def battery(
    adapter: str = typer.Option(..., "--adapter", "-a"),
    limit: int = typer.Option(3, "--limit"),
    config: Path = typer.Option(None, "--config"),
) -> None:
    """The consistency battery: each scripted agent must behave exactly as specified."""
    from .battery import run_battery

    cfg = load_runtime(config)
    adp, ex = build(adapter, cfg)
    llm = S.stub_llm()
    try:
        findings, checked = run_battery(_graphs(cfg, adapter, limit), adp, ex, cfg, llm)
    finally:
        ex.shutdown()
    typer.echo(f"{checked} checks on {adapter}")
    if findings:
        typer.echo(canonical_dumps(findings, indent=1))
        raise typer.Exit(1)
    typer.echo("episode battery clean")


@app.command()
def personas(
    adapter: str = typer.Option(..., "--adapter", "-a"),
    limit: int = typer.Option(3, "--limit"),
    config: Path = typer.Option(None, "--config"),
) -> None:
    """Oracle success under every persona: a persona that cannot be satisfied is a bug."""
    cfg = load_runtime(config)
    adp, ex = build(adapter, cfg)
    llm = S.stub_llm()
    rows = {}
    try:
        for pid in PERSONA_IDS:
            ms = []
            for graph in _graphs(cfg, adapter, limit):
                traj, _ = _run_one(graph, adp, ex, cfg, llm, _oracle_for, pid)
                ms.append(M.score(traj))
            rows[pid] = M.aggregate(ms)["all"]
    finally:
        ex.shutdown()
    typer.echo(canonical_dumps(rows, indent=1))
    bad = [p for p, r in rows.items() if (r.get("success_rate") or 0) < 1.0]
    if bad:
        typer.echo(f"FAIL: oracle cannot always win under {bad}")
        raise typer.Exit(1)


@app.command()
def smoke(config: Path = typer.Option(None, "--config")) -> None:
    """One live API call, to confirm the Ark contract before anything is built on it."""
    cfg = load_runtime(config)
    client = LLMClient(cfg)
    client.offline = False
    typer.echo(f"model={client.model} base_url={client.base_url} key_env={client.api_key_env}")
    text = client.complete(
        "Reply with exactly the word: ready", role="select", system="You are terse.")
    typer.echo(f"output_text -> {text!r}")
    typer.echo(canonical_dumps(client.usage(), indent=1))
    if not text:
        raise typer.Exit(1)


@app.command("export")
def export_samples(
    adapter: str = typer.Option("webshop", "--adapter", "-a"),
    graphs: Path = typer.Option(None, "--graphs", help="graph root (default: config artifacts)"),
    out: Path = typer.Option(None, "--out", help="where samples land (default: graph root)"),
    limit: int = typer.Option(None, "--limit", help="max graphs"),
    shard: str = typer.Option(None, "--shard", help="i/n -- every n-th graph (parallel export)"),
    personas: str = typer.Option(None, "--personas", help="comma-separated (default: all 5)"),
    strategies: str = typer.Option(None, "--strategies", help="comma-separated strategy ids"),
    live_llm: bool = typer.Option(True, "--live-llm/--fake-llm"),
    model: str = typer.Option(None, "--model",
                              help="override the generation/user-side model for this export "
                                   "(e.g. doubao-seed-2-1-turbo-260628)"),
    pipeline: str = typer.Option("v1", "--pipeline",
                                 help="v1 = mask-first render; v2 = generate-then-verify "
                                      "(docs/pipeline-v2.md)"),
    config: Path = typer.Option(None, "--config"),
    verbose: bool = typer.Option(False, "-v"),
) -> None:
    """Export perturbation samples: one task instance per (graph, strategy, persona).

    A sample is not a solved episode. It holds the misaligned query an agent would see plus
    the scoring material behind it, so a harness can run any agent against it later.
    """
    from .. import dataset as ds

    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    cfg = load_runtime(config)
    if model:
        cfg["llm"]["model"] = model
    adp, ex = build(adapter, cfg)
    llm = make_llm(cfg, fake=not live_llm)

    root = Path(graphs) if graphs else Path(cfg["runtime"].get("graphs_dir",
                                                             cfg["paths"]["artifacts"]))
    if not root.is_absolute():
        root = REPO_ROOT / root
    out_root = Path(out) if out else root
    if not out_root.is_absolute():
        out_root = REPO_ROOT / out_root

    all_graphs = [t for t in storage.iter_graphs(root, adapter)]
    if shard:
        i, n = (int(x) for x in shard.split("/"))
        all_graphs = [t for j, t in enumerate(all_graphs) if j % n == i]
    if limit:
        all_graphs = all_graphs[:limit]
    if not all_graphs:
        raise SystemExit(f"no {adapter} graphs under {root}")

    seeds_meta = {s.record_id: s.meta for s in adp.load()}
    pool = None
    if strategies:
        pool = [st.BY_ID[s.strip()] for s in strategies.split(",") if s.strip()]
    plist = [p.strip() for p in personas.split(",")] if personas else None

    stats = ds.ExportStats()
    try:
        for graph in all_graphs:
            try:
                samples, skipped = ds.build_samples(graph, adp, ex, cfg, llm,
                                                    personas=plist, strategies=pool,
                                                    seeds_meta=seeds_meta, pipeline=pipeline)
            except Exception as exc:
                log.warning("graph %s failed to export: %s", graph.graph_id, exc)
                stats.graphs_failed += 1
                continue
            ds.write_samples(out_root, adapter, samples)
            stats.record(samples, skipped)
    finally:
        ex.shutdown()
    payload = stats.to_dict()
    payload["usage"] = llm.usage() if hasattr(llm, "usage") else None
    name = (f"{adapter}_samples_stats" + ("_v2" if pipeline == "v2" else "")
            + (f"_{shard.replace('/', 'of')}" if shard else ""))
    (REPO_ROOT / "reports" / f"{name}.json").write_text(
        canonical_dumps(payload, indent=1), encoding="utf-8")
    typer.echo(canonical_dumps(payload, indent=1))


exp = typer.Typer(add_completion=False, help="Experiments: manifest, run, report")
app.add_typer(exp, name="exp")


def _load_graphs_and_samples(cfg, adapter: str, samples_root: Path | None = None):
    from .. import dataset as ds
    root = Path(cfg["runtime"].get("graphs_dir", cfg["paths"]["artifacts"]))
    if not root.is_absolute():
        root = REPO_ROOT / root
    srcroot = Path(samples_root) if samples_root else root
    if not srcroot.is_absolute():
        srcroot = REPO_ROOT / srcroot
    graphs = {t.graph_id: t for t in storage.iter_graphs(root, adapter)}
    samples = {s["sample_id"]: s for s in ds.iter_samples(srcroot, adapter)}
    if not graphs or not samples:
        raise SystemExit(f"need graphs under {root} and samples under {srcroot} for {adapter}")
    return graphs, samples


@exp.command("manifest")
def exp_manifest(
    adapter: str = typer.Option("webshop", "--adapter", "-a"),
    per_cell: int = typer.Option(2, "--per-cell"),
    out: Path = typer.Option(..., "--out"),
    samples_root: Path = typer.Option(None, "--samples",
                                      help="sample root (default: graphs dir)"),
    config: Path = typer.Option(None, "--config"),
) -> None:
    """Freeze a stratified per-fault-type draw, crossed with every persona at
    evaluation time; every arm replays the same rows."""
    from .experiment import build_manifest

    cfg = load_runtime(config)
    _, samples = _load_graphs_and_samples(cfg, adapter, samples_root)
    man = build_manifest(list(samples.values()), per_cell=per_cell)
    man["adapter"] = adapter
    out = out if out.is_absolute() else REPO_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(canonical_dumps(man, indent=1), encoding="utf-8")
    typer.echo(canonical_dumps({k: v for k, v in man.items() if k != "rows"}, indent=1))


@exp.command("run")
def exp_run(
    arm: str = typer.Option(..., "--arm"),
    manifest: Path = typer.Option(..., "--manifest"),
    adapter: str = typer.Option("webshop", "--adapter", "-a"),
    out: Path = typer.Option(Path("experiments/runs"), "--out"),
    shard: str = typer.Option(None, "--shard", help="i/n"),
    threads: int = typer.Option(1, "--threads"),
    p_shift: float = typer.Option(0.0, "--p-shift",
                                  help="default 0: experiments are axis-A unless asked. The "
                                       "first campaign inherited runtime.yaml's 0.25 and the "
                                       "more an arm clarified, the more the intent shifted "
                                       "under it -- a confound, not a finding."),
    after_proposal: bool = typer.Option(False, "--shift-after-proposal"),
    limit: int = typer.Option(None, "--limit"),
    samples_root: Path = typer.Option(None, "--samples",
                                      help="sample root (default: graphs dir)"),
    seed_offset: int = typer.Option(0, "--seed-offset",
                                    help="shift every manifest row's episode seed by this "
                                         "constant. THE way to run genuine repeats: the "
                                         "sample set and cross-arm pairing are untouched, "
                                         "only the randomness moves. Note --set "
                                         "episode_seed has no effect here, because the "
                                         "manifest's per-row seed is what makes arms "
                                         "comparable. Pair with --cache-tag."),
    cache_tag: str = typer.Option(None, "--cache-tag",
                                  help="appended to the LLM cache namespace: two runs of "
                                       "the same arm with different tags sample fresh "
                                       "completions instead of replaying each other"),
    max_turns: int = typer.Option(None, "--max-turns",
                                  help="override runtime.max_turns for this run (the "
                                       "tool-loop + user turns need more than WebShop's "
                                       "bare 10-16 when conversation is involved)"),
    procs: int = typer.Option(1, "--procs",
                              help="fork-after-load worker processes. The parent loads the "
                                   "catalog ONCE and forks thin workers that share it "
                                   "copy-on-write -- N workers cost ~one catalog, not N "
                                   "(the separate-process shards paid 1.8G each)"),
    config: Path = typer.Option(None, "--config"),
    set_rt: list[str] = typer.Option(None, "--set",
                                     help="runtime override as key=value, repeatable: "
                                          "--set patience_init=11 --set show_patience=true. "
                                          "For sweeping environment settings without "
                                          "editing the checked-in runtime.yaml. "
                                          "ints/floats/bools are parsed."),
    verbose: bool = typer.Option(False, "-v"),
) -> None:
    """Run one arm over the manifest. Idempotent: relaunch to resume."""
    from .experiment import HALT_FILE, Runner

    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    cfg = load_runtime(config)
    cfg["runtime"]["p_shift"] = float(p_shift)
    if max_turns:
        cfg["runtime"]["max_turns"] = int(max_turns)
    for item in (set_rt or []):
        key, _, val = str(item).partition("=")
        key = key.strip()
        # "llm.model=..." routes to the llm block (e.g. pointing the USER SIMULATOR at a
        # different doubao model for capacity tests); bare keys stay runtime settings.
        # REFUSED, not ignored (2026-08-19). Both of these are read from the MANIFEST ROW
        # (Runner.run_row: persona_name=row["persona"], seed=row["episode_seed"]+offset), so
        # setting them here changes nothing while LOOKING like it configured the run. Measured
        # end-to-end: `--set episode_seed=999` reproduced its reference run 10/10 episodes,
        # seed for seed. Left silent, a persona sweep driven by `--set persona=` silently
        # evaluates EVERY persona the manifest carries, and three "repeats" are one run
        # performed three times. Fail loudly and name the mechanism that does work.
        if key in ("episode_seed", "persona"):
            raise typer.BadParameter(
                f"--set {key}= has no effect on `exp run`: the manifest row supplies "
                f"{key}. " + ("Use --seed-offset N (with a distinct --cache-tag) for genuine "
                              "repeats." if key == "episode_seed" else
                              "Pass a manifest whose rows carry only the persona you want."))
        if key.startswith("llm."):
            cfg["llm"][key[4:]] = _coerce(val.strip())
        else:
            cfg["runtime"][key] = _coerce(val.strip())
    # The cache is keyed on request CONTENT alone, so two runs whose prompts coincide replay
    # each other's completions and a repeat stops being a repeat. `--seed-offset` moves the
    # user's behaviour, but every prompt that happens to match still replays -- notably the
    # first agent turn, which sees the same query in every repeat. Namespace the cache by the
    # offset unless the caller named a tag, so genuine repeats are the DEFAULT rather than
    # something you have to remember.
    if not cache_tag and seed_offset:
        cache_tag = f"off{seed_offset}"
    if cache_tag:
        cfg["llm"]["prompt_version"] = f"{cfg['llm'].get('prompt_version', 'v1')}-{cache_tag}"
    # L0 (ruling 2026-08-11): same agent as B0, but the simulated user never speaks
    cfg["runtime"]["mute_user"] = (arm == "L0")
    if after_proposal:
        cfg["runtime"]["shift_after_first_proposal"] = True
    # threads share one executor, so the resident-cluster window must cover them
    cfg.setdefault("webshop", {})["max_resident_clusters"] = max(
        int((cfg.get("webshop") or {}).get("max_resident_clusters", 2)), threads * 2)

    man = json.loads((manifest if manifest.is_absolute()
                      else REPO_ROOT / manifest).read_text(encoding="utf-8"))
    rows = man["rows"]
    if shard:
        i, n = (int(x) for x in shard.split("/"))
        rows = [r for j, r in enumerate(rows) if j % n == i]
    if limit:
        rows = rows[:limit]

    out_dir = out if out.is_absolute() else REPO_ROOT / out
    if (out_dir / HALT_FILE).exists():
        typer.echo(f"HALT marker present at {out_dir / HALT_FILE}; refusing to start. "
                   f"Delete it once credit is restored.")
        raise typer.Exit(2)

    adp, ex = build(adapter, cfg)
    graphs, samples = _load_graphs_and_samples(cfg, adapter, samples_root)

    if procs > 1:
        # Fork-after-load: the catalog above is loaded exactly once; children share it
        # copy-on-write. gc.freeze() moves the loaded objects to the permanent
        # generation so refcount/GC churn does not gradually un-share the pages.
        # Each child builds its OWN llm client (sockets must never cross a fork) and
        # runs its row slice single-threaded (threads fought over the GIL all night).
        import gc
        import os as _os
        preload = getattr(ex, "preload", None)
        if preload is not None:
            log.info("preloading executor in the parent before forking %d workers", procs)
            preload()
        gc.collect()
        gc.freeze()
        pids = []
        for w in range(procs):
            pid = _os.fork()
            if pid == 0:
                code = 1
                try:
                    runner = Runner(seed_offset=seed_offset, adapter=adp, executor=ex, config=cfg,
                                    llm_factory=lambda: make_llm(cfg, fake=False),
                                    graphs=graphs, samples=samples,
                                    out_dir=out_dir, arm=arm)
                    result = runner.run(rows[w::procs], threads=1)
                    code = 3 if result.get("halted") else 0
                except BaseException as exc:   # noqa: BLE001 -- a child must never re-run typer
                    log.error("worker %d died: %s", w, exc)
                finally:
                    _os._exit(code)
            pids.append(pid)
        halted = failed = 0
        for pid in pids:
            _, status = _os.waitpid(pid, 0)
            code = _os.waitstatus_to_exitcode(status)
            halted += code == 3
            failed += code not in (0, 3)
        ex.shutdown()
        typer.echo(canonical_dumps({"workers": procs, "halted_workers": halted,
                                    "failed_workers": failed}, indent=1))
        if halted:
            raise typer.Exit(3)
        if failed:
            raise typer.Exit(1)
        return

    try:
        runner = Runner(seed_offset=seed_offset, adapter=adp, executor=ex, config=cfg,
                        llm_factory=lambda: make_llm(cfg, fake=False),
                        graphs=graphs, samples=samples, out_dir=out_dir, arm=arm)
        result = runner.run(rows, threads=threads)
    finally:
        ex.shutdown()
    typer.echo(canonical_dumps(result, indent=1))
    if result.get("halted"):
        raise typer.Exit(3)


@exp.command("report")
def exp_report(
    arms: str = typer.Option(..., "--arms", help="comma separated"),
    out: Path = typer.Option(Path("experiments/runs"), "--out"),
    report_to: Path = typer.Option(None, "--report-to"),
    baseline: str = typer.Option("A0", "--baseline", help="arm for paired deltas"),
) -> None:
    """Recompute GRIP offline from stored trajectories, with bootstrap CIs."""
    from .experiment import load_trajectories, paired_delta, report_arm

    out_dir = out if out.is_absolute() else REPO_ROOT / out
    names = [a.strip() for a in arms.split(",") if a.strip()]
    trajs = {a: load_trajectories(out_dir, a) for a in names}
    payload = {"n_by_arm": {a: len(t) for a, t in trajs.items()},
               "arms": {a: report_arm(t) for a, t in trajs.items() if t}}
    if baseline in trajs and trajs.get(baseline):
        payload["paired_vs_" + baseline] = {
            a: {k: paired_delta(trajs[baseline], trajs[a], key=k)
                for k in ("success", "earned", "recovery", "aim")}
            for a in names if a != baseline and trajs.get(a)}
    body = canonical_dumps(payload, indent=1)
    if report_to:
        dest = report_to if report_to.is_absolute() else REPO_ROOT / report_to
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(body, encoding="utf-8")
    typer.echo(body)


if __name__ == "__main__":
    app()
