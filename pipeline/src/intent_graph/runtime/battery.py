"""The episode battery: properties a working runtime must have, per adapter.

Named ``episode_battery`` to avoid colliding with the graph-structure battery that already
owns ``battery.py`` at the package root and the ``battery`` marker.

Applicability matters and is reported rather than silently skipped: ``second_valid`` needs a
ground-truth set with two members, so it cannot run where ground truth is a scalar (all OS)
or a state hash (77% of DBBench); ``destructive_rejected`` only means something where the
agent can write.
"""

from __future__ import annotations

from . import metrics as M
from . import scripted as S
from .accept import accepts as accepts_fn
from .accept import default_parse_proposal
from .episode import Episode, Outcome


def _oracle_ids(graph, header) -> list[str]:
    spec = (header.get("perturbation") or {}).get("hidden_slots") or []
    ordered = [s for s, _, _ in sorted(graph.root.conditions)]
    return [f"slot_{ordered.index(s)}" for s in spec if s in ordered]


def _play(graph, adapter, executor, cfg, llm, factory, persona=None, **over):
    conf = {**cfg, "runtime": {**cfg["runtime"], **over}}
    probe = Episode(graph=graph, adapter=adapter, executor=executor, config=conf, llm=llm,
                    agent=S.NoOp(graph=graph), persona_name=persona).run()
    agent = factory(graph, probe.header)      # constructed AFTER the probe, never wrapped
    ep = Episode(graph=graph, adapter=adapter, executor=executor, config=conf, llm=llm,
                 agent=agent, persona_name=persona)
    return ep.run(), probe


def run_battery(graphs, adapter, executor, cfg, llm) -> tuple[list[dict], int]:
    """Returns ``(findings, checks_run)``. A finding is a violated property."""
    findings: list[dict] = []
    checks = 0

    def fail(graph, check, detail):
        findings.append({"graph_id": graph.graph_id, "check": check, "detail": detail})

    for graph in graphs:
        kind = graph.root.ground_truth.kind
        card = graph.root.ground_truth.cardinality()

        # ---- oracle must win, and recover everything, with no premature action ----
        checks += 1
        traj, probe = _play(graph, adapter, executor, cfg, llm,
                            lambda t, h: S.Oracle(graph=t, hidden_slot_ids=_oracle_ids(t, h)))
        if traj.outcome != Outcome.SUCCESS.value:
            fail(graph, "oracle_must_win", f"{traj.outcome}: {traj.error}")
        else:
            m = M.score(traj)
            if m.hidden_slots and m.intent_recovery != 1.0:
                fail(graph, "oracle_recovery", f"{m.recovered_slots} of {m.hidden_slots}")
            if m.premature_action:
                fail(graph, "oracle_premature", "oracle acted before recovering")

        # ---- no-op must never win ----
        checks += 1
        traj, _ = _play(graph, adapter, executor, cfg, llm, lambda t, h: S.NoOp(graph=t))
        if traj.outcome == Outcome.SUCCESS.value:
            fail(graph, "noop_must_lose", "an agent that proposes nothing succeeded")

        # ---- a stale answer is rejected exactly when it stops being valid ----
        checks += 1
        traj, _ = _play(graph, adapter, executor, cfg, llm,
                        lambda t, h: S.IntentIgnorer(graph=t), p_shift=1.0, max_shifts=1)
        m = M.score(traj)
        if m.n_shifts and traj.final_node:
            nodes = {n.intent_id: n for n in (graph.root, *graph.children)}
            final = nodes[traj.final_node]
            stale = S._gt_member(graph.root)
            if stale is not None:
                parsed = (adapter.parse_proposal(stale)
                          if hasattr(adapter, "parse_proposal")
                          else default_parse_proposal(stale))
                with executor.open(graph.env_spec) as sess:
                    still_ok, _ = accepts_fn(adapter, parsed, final, sess, executor=executor)
                won = traj.outcome == Outcome.SUCCESS.value
                if still_ok and not won:
                    fail(graph, "stale_still_valid_must_pass",
                         "the old answer still satisfies the new intent but was rejected")
                if not still_ok and won:
                    fail(graph, "stale_invalid_must_fail",
                         "the old answer no longer satisfies the new intent but was accepted")

        # ---- a second, equally valid answer must also be accepted ----
        if S.has_multiple_valid_answers(graph.root):
            checks += 1
            traj, _ = _play(graph, adapter, executor, cfg, llm,
                            lambda t, h: S.SecondValid(graph=t))
            if traj.outcome != Outcome.SUCCESS.value:
                fail(graph, "second_valid_must_pass",
                     f"a different member of a {card}-member answer set was rejected")
        else:
            findings.append({"graph_id": graph.graph_id, "check": "second_valid",
                             "detail": f"not applicable: a {kind} node has exactly one "
                                       f"correct answer (cardinality {card})",
                             "skipped": True})

    # skipped entries are informational, not failures
    real = [f for f in findings if not f.get("skipped")]
    return real, checks
