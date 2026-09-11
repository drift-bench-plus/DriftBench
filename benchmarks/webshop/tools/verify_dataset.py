"""Accept-or-reject the finished WebShop dataset.

Not a summary: every check either passes or names what failed. A dataset that ships with a
silent defect is worse than one that fails loudly here, because every downstream number
computed from it inherits the defect and nothing later reveals it.

    python tools/verify_dataset.py [--graphs artifacts] [--samples artifacts]
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "pipeline" / "src"))

from intent_graph import storage  # noqa: E402
from intent_graph.cli import build, load_config  # noqa: E402
from intent_graph.dataset import iter_samples  # noqa: E402
from intent_graph.runtime.perturb import internal_syntax_leaks  # noqa: E402

OPS = ("REFINEMENT", "RELAXATION", "SUBSTITUTION", "PIVOT")


class Checks:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.notes: list[str] = []

    def check(self, ok: bool, label: str, detail: str = "") -> bool:
        if ok:
            print(f"  PASS  {label}")
        else:
            print(f"  FAIL  {label}" + (f"\n          {detail}" if detail else ""))
            self.failures.append(label)
        return ok

    def note(self, text: str) -> None:
        print(f"        {text}")
        self.notes.append(text)


def verify_trees(root: Path, adapter_name: str, c: Checks) -> list:
    print("\n=== TREES ===")
    graphs = list(storage.iter_graphs(root, adapter_name))
    c.check(bool(graphs), f"graphs exist ({len(graphs)} found)")
    if not graphs:
        return graphs

    bad_mix = [t.graph_id for t in graphs
               if collections.Counter(e.operator.value for e in t.edges)
               != {op: 2 for op in OPS}]
    c.check(not bad_mix, "every graph is exactly 2/2/2/2",
            f"{len(bad_mix)} graphs with a different mix, e.g. {bad_mix[:3]}")

    empty = [t.graph_id for t in graphs
             if t.root.ground_truth.is_empty
             or any(ch.ground_truth.is_empty for ch in t.children)]
    c.check(not empty, "no node has empty ground truth", f"{len(empty)} offenders")

    unmoved = [t.graph_id for t in graphs if not any(e.gt_moved for e in t.edges)]
    c.check(not unmoved, "every graph has at least one answer-moving edge",
            f"{len(unmoved)} graphs where no edge moves the answer")

    seeded = [t.graph_id for t in graphs if not t.root.is_seed]
    c.check(not seeded, "every root is the seed intent", f"{len(seeded)} non-seed roots")

    mixed = []
    for t in graphs:
        try:
            t.assert_single_environment()
        except ValueError as exc:
            mixed.append(str(exc))
    c.check(not mixed, "every graph stays inside one environment",
            f"{len(mixed)} mixed, e.g. {mixed[:2]}")

    one_cluster = [t.graph_id for t in graphs
                   if len({(c_.base or {}).get("cluster") for c_ in t.children}
                          | {(t.root.base or {}).get("cluster")}) != 1]
    c.check(not one_cluster, "every branch stays in the root's cluster",
            f"{len(one_cluster)} graphs whose pivot left the cluster")

    dupes = [tid for tid, n in collections.Counter(t.graph_id for t in graphs).items() if n > 1]
    c.check(not dupes, "graph ids are unique", f"duplicates: {dupes[:5]}")

    card = collections.Counter(t.root.ground_truth.cardinality() for t in graphs)
    c.note(f"root |GT|: min={min(card)} median~{sorted(card.elements())[len(graphs)//2]} "
           f"max={max(card)}")
    c.note(f"clusters covered: {len({t.env_spec.get('cluster') for t in graphs})}")
    prov = collections.Counter(e.provenance.value for t in graphs for e in t.edges)
    total = sum(prov.values())
    c.note(f"branch provenance: REAL {prov['REAL']} ({100*prov['REAL']/total:.0f}%) / "
           f"SYNTHETIC {prov['SYNTHETIC']}")
    return graphs


def verify_split(graphs: list, c: Checks) -> None:
    print("\n=== SPLIT PURITY ===")
    cfg = load_config()
    adapter, _ = build("webshop", cfg)
    meta = {s.record_id: s.meta for s in adapter.load()}
    splits = collections.Counter()
    idxs = []
    unknown = 0
    for t in graphs:
        m = meta.get(t.root.source_record)
        if not m:
            unknown += 1
            continue
        splits[m.get("split")] += 1
        idxs.append(m.get("goal_idx"))
    c.check(unknown == 0, "every root maps back to a loaded seed", f"{unknown} unmatched")
    c.check(set(splits) <= {"test"},
            "every root comes from the TEST split", f"splits present: {dict(splits)}")
    c.check(all(i is not None and i < 500 for i in idxs),
            "every root goal_idx is inside 0..499",
            f"out of range: {[i for i in idxs if i is None or i >= 500][:5]}")
    dupe = [i for i, n in collections.Counter(idxs).items() if n > 1]
    c.check(not dupe, "no goal produced two graphs", f"repeated goal_idx: {dupe[:5]}")
    c.note(f"distinct test goals used: {len(set(idxs))} of 500")


def verify_samples(root: Path, adapter_name: str, graphs: list, c: Checks) -> None:
    print("\n=== SAMPLES ===")
    samples = list(iter_samples(root, adapter_name))
    c.check(bool(samples), f"samples exist ({len(samples)} found)")
    if not samples:
        return

    tree_ids = {t.graph_id for t in graphs}
    orphan = [s["sample_id"] for s in samples if s["graph_id"] not in tree_ids]
    c.check(not orphan, "every sample points at an existing graph", f"{len(orphan)} orphans")

    dupes = [i for i, n in collections.Counter(s["sample_id"] for s in samples).items() if n > 1]
    c.check(not dupes, "sample ids are unique", f"duplicates: {dupes[:5]}")

    # the load-bearing one
    leaked = []
    for s in samples:
        q = s["query"].lower()
        for raw in s["ground_truth"]["value"]:
            try:
                asin, _ = json.loads(raw)
            except Exception:
                continue
            if asin.lower() in q:
                leaked.append((s["sample_id"], asin))
    c.check(not leaked, "no ground-truth product id appears in any query",
            f"{len(leaked)} leaks, e.g. {leaked[:3]}")

    # Word-boundary matching, matching production `check_fidelity` exactly. A substring test
    # reports a leak whenever a withheld value happens to sit inside a retained one: the
    # colour "blue" inside the faithfully-stated attribute "wireless bluetooth" is not a leak,
    # and an agent cannot recover the colour from that text.
    hidden_leaks = []
    for s in samples:
        q = s["query"].lower()
        withheld = set(s["mask"]["withheld"])
        for slot, _op, value in s["hidden_intent"]["conditions"]:
            if slot not in withheld:
                continue
            v = str(value).lower().strip()
            if len(v) < 3:
                continue          # production skips values under 3 chars as unsafe to match
            if re.search(rf"(?<!\w){re.escape(v)}(?!\w)", q):
                hidden_leaks.append((s["sample_id"], slot, value))
    c.check(not hidden_leaks, "no withheld value appears in its own query (word-boundary)",
            f"{len(hidden_leaks)} leaks, e.g. {hidden_leaks[:3]}")

    syntax = [s["sample_id"] for s in samples if internal_syntax_leaks(s["query"])]
    c.check(not syntax, "no query contains internal field syntax", f"{len(syntax)} offenders")

    bad_sig = [s["sample_id"] for s in samples
               if not (s.get("signature") or {}).get("ok", True)]
    c.check(not bad_sig, "every sample's perturbation passed its signature check",
            f"{len(bad_sig)} failures")

    mismatch = [s["sample_id"] for s in samples
                if set(s["hidden_intent"]["hidden_slots"]) != set(s["mask"]["hidden_slots"])]
    c.check(not mismatch, "hidden slots agree between mask and scoring block",
            f"{len(mismatch)} mismatches")

    empty_q = [s["sample_id"] for s in samples if not s["query"].strip()]
    c.check(not empty_q, "no query is empty", f"{len(empty_q)} empty")

    hiding = sum(1 for s in samples if s["hidden_intent"]["hidden_slots"])
    c.note(f"samples that hide something: {hiding} ({100*hiding/len(samples):.0f}%)")
    c.note(f"per strategy: {dict(sorted(collections.Counter(sid for s in samples for sid in s['strategy_ids']).items()))}")
    c.note(f"per persona: {dict(sorted(collections.Counter(s['persona'] for s in samples).items()))}")
    c.note(f"misalignments: {dict(sorted(collections.Counter(s['misalignments'] for s in samples).items()))}")
    fallback = sum(1 for s in samples if "template_fallback" in (s.get("fidelity_notes") or []))
    c.note(f"template fallbacks: {fallback} ({100*fallback/len(samples):.1f}%)")
    c.note(f"samples per graph: {len(samples)/len(tree_ids):.1f} mean")


def report_export_stats(base: Path, adapter_name: str, c: Checks) -> None:
    """Merge the per-shard export stats.

    Skips cannot be recomputed from the samples: a skip is an ABSENCE, so the only record it
    leaves is the shard's own accounting. Without merging these, a run where a whole strategy
    failed to mask would look like a smaller run rather than a degraded one.
    """
    print("\n=== EXPORT ACCOUNTING (merged over shards) ===")
    all_files = sorted((base / "reports").glob(f"{adapter_name}_samples_stats*.json"))
    sharded = [f for f in all_files if "of" in f.stem.rsplit("_", 1)[-1]]
    # a leftover un-sharded stats file from a probe run would otherwise be merged in and
    # double-count whatever graphs that probe touched
    files = sharded or all_files
    if not c.check(bool(files), f"per-shard export stats found ({len(files)} shards)"):
        return
    total = collections.Counter()
    sub: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        for k, v in d.items():
            if isinstance(v, int):
                total[k] += v
            elif isinstance(v, dict):
                for kk, vv in v.items():
                    if isinstance(vv, int):
                        sub[k][kk] += vv
    c.check(total.get("trees_failed", 0) == 0, "no graph failed to export",
            f"{total.get('trees_failed')} failed")
    attempted = total["samples"] + total["strategy_skips"]
    rate = total["samples"] / max(attempted, 1)
    c.check(rate >= 0.85, f"at least 85% of (strategy, persona) attempts produced a sample "
                          f"({100*rate:.1f}%)",
            f"{total['strategy_skips']} skips of {attempted} attempts")
    fb = total["template_fallbacks"] / max(total["samples"], 1)
    c.check(fb <= 0.15, f"template fallbacks under 15% ({100*fb:.1f}%)",
            "template prose is stiffer than model prose; a high rate degrades realism")
    c.note(f"graphs exported: {total['graphs']}   samples: {total['samples']}")
    c.note(f"skips by strategy: {dict(sorted(sub['skips_by_strategy'].items()))}")
    c.note(f"skip reasons: {dict(sorted(sub['skip_reasons'].items()))}")
    c.note(f"misalignments achieved: {dict(sorted(sub['misalignments_achieved'].items()))}")
    c.note(f"render attempts: {dict(sorted(sub['render_attempts'].items()))}")
    c.note(f"signature verdicts: {dict(sorted(sub['signature_verdicts'].items()))}")
    c.note(f"governing kind: {dict(sorted(sub['governing_kind'].items()))}")
    api = sub.get("usage") or {}
    if api:
        c.note(f"API: {api.get('requests', 0)} requests, {api.get('cached', 0)} cache hits, "
               f"{api.get('output_tokens', 0)} output tokens")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", default="artifacts")
    ap.add_argument("--samples", default=None)
    ap.add_argument("--adapter", default="webshop")
    args = ap.parse_args()

    base = Path(__file__).resolve().parents[1]
    troot = Path(args.graphs) if Path(args.graphs).is_absolute() else base / args.graphs
    sroot = Path(args.samples) if args.samples else troot
    if not sroot.is_absolute():
        sroot = base / sroot

    c = Checks()
    graphs = verify_trees(troot, args.adapter, c)
    if graphs and args.adapter == "webshop":
        verify_split(graphs, c)
    verify_samples(sroot, args.adapter, graphs, c)
    report_export_stats(base, args.adapter, c)

    print("\n" + "=" * 70)
    if c.failures:
        print(f"REJECTED: {len(c.failures)} check(s) failed")
        for f in c.failures:
            print(f"   - {f}")
        return 1
    print("ACCEPTED: every check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
