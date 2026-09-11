# Drift-Bench++

**Beyond Oracle Communication: Benchmarking Interactive Intent Alignment
Under Miscommunication and Evolving User Intent**

Anonymous code and data release for double-blind review ·
[Project page](https://drift-bench-plus.github.io/)

---

Most agent benchmarks assume *oracle communication*: the user states the
task perfectly, once, and never changes their mind. Drift-Bench++ drops that
assumption. Every task carries a **verified miscommunicated opening request**,
an executable **intent graph** of alternative goals the user may **silently
drift to** mid-conversation, and a **finite patience budget** spent by every
question and rejected proposal. Episodes are played against
**persona-conditioned simulated users** and scored with **GRIP** — an
eleven-metric protocol for Grounding, Role-realism, Inquiry, and
Persistence — instead of a single success rate.

Drift-Bench++ is built as a **benchmark-agnostic pipeline**: one shared
package implements construction, verification, the interaction runtime, all
evaluated methods, and the evaluation protocol. Each host benchmark
contributes only a thin adapter (how to read its tasks, execute its ground
truth, and speak its domain vocabulary) plus its own configuration and data.

## Contents

- [Released data](#released-data)
- [How it works](#how-it-works)
- [Repository layout](#repository-layout)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Reproducing the benchmark](#reproducing-the-benchmark)
- [Tests](#tests)
- [Notes](#notes)
- [Citation](#citation)

## Released data

2,543 verified instances over 395 intent graphs on three executable
environments — the three-step verifier rejects 31–48% of generated
candidates:

| Benchmark | Tasks | Intent graphs | Verifier rejection | Samples | Where |
|---|---|---|---|---|---|
| WebShop | 500 | 257 | 31% | **1,780** | `benchmarks/webshop/artifacts/samples_v2` |
| τ²-Retail | 114 | 98 | 47% | **555** | `benchmarks/tau2/artifacts` |
| τ²-Airline | 50 | 40 | 48% | **208** | `benchmarks/tau2/artifacts` |

On top of the verified WebShop pool, composed-fault subsets ship at
k = 2 (515) and k = 4 (501) under
`benchmarks/webshop/artifacts/samples_composite` for the complexity study.
`data/tau2/` additionally packages the retail/telecom bundle
self-contained — including the retail learn split held out for method
development — with a datasheet and the axis-B (intent-shift) metadata.
Every instance carries: the latent executable intent, the verified faulty
opening query with its fault type (eleven types across four families), the
typed intent graph (refine / relax / substitute / pivot), and recomputed
ground truth for every alternative goal.

## How it works

**Construction — generate-then-verify.** Native tasks become executable
condition sets; validated retrieval and witness-guided synthesis add typed
alternative intents with recomputed ground truth. A writer model renders
the true intent into a request carrying exactly one taxonomy fault; two
extractor models from different families must independently agree on what
the text miscommunicates; and the literal reading is executed against the
environment — a fault that does not change the executable answer is
rejected. No instance depends on a judge model for its ground truth.

**Interaction — patience, personas, silent shifts.** A state machine owns
everything true (intents, disclosures, budgets, shifts, adjudication); the
language model only voices replies. Questions cost 2, rejected proposals
cost 4, from a persona-scaled budget of 10. Five General Decision-Making
Style personas control disclosure, patience, response style, and shift
tendency. Scheduled and interaction-triggered intent shifts move the latent
goal over the intent graph — silently, with fire probability 0.6.

**Evaluation — GRIP.** Eleven metrics across four dimensions: Grounding
(Success, Earned, Inferred), Inquiry (Aim, Recovery, Patience), Persistence
(Staleness, Reaction, PostShift), and Role-realism (Identification,
Consistency). Each dimension keeps its own denominator; nothing is
collapsed into a composite. All metrics except the two role-realism judges
are recomputed deterministically from stored episode records.

## Repository layout

```
pipeline/                    the benchmark-agnostic package (installed once)
  pyproject.toml
  src/intent_graph/
    engine.py, dataset.py, …   intent-graph construction: validated retrieval,
                               witness-guided synthesis, executable ground truth
    adapters/                  the ONLY benchmark-specific code: webshop,
                               tau2_retail, tau2_airline, tau2_telecom
                               (+ auxiliary dbbench/os adapters)
    executors/                 environment execution backends per host
    runtime/
      genverify*.py            generate-then-verify perturbation (k = 1, 2, 4)
      episode.py               the interaction loop: patience economy,
                               scheduled + triggered silent shifts, adjudication
      user.py, persona.py      persona-conditioned simulated users
      agents.py                every evaluated method and baseline
      grip.py, metrics.py,     the GRIP protocol and its judges
      judges.py
      cli.py                   construction / episode / export entry points
  src/intent_tree              symlink alias kept for script compatibility

benchmarks/                  one directory per host benchmark: config + drivers + data
  webshop/
    config/                    identity-hashed graph config + runtime config
    tools/                     campaign drivers, GRIP reports, role-realism judges
    artifacts/                 released graphs and samples (see table above)
    experiments/               evaluation manifests (paired sample views)
    tests/                     unit suite run against the shared package
  tau2/
    config/
    *.py, run_air_smoke.sh     tree builders, perturbation runners, episode
                               drivers, dataset export, policy gates, audits
    artifacts/
    tests/

data/tau2/                   self-contained packaged retail/telecom dataset
```

Nothing under `pipeline/` knows which benchmark is running until an adapter
is chosen. The per-benchmark `config/` directories are intentionally
separate files: graph identity hashes over config content, so each
benchmark's `default.yaml` is part of its dataset's provenance.

## Installation

Python **3.11+** is required.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ./pipeline         # installs the `it` CLI and all dependencies
```

External hosts, only needed to re-run construction or live episodes (clone
under `external/`, or adjust `benchmarks/*/config/default.yaml`):

| Dependency | Pin | Used for |
|---|---|---|
| [tau2-bench](https://github.com/sierra-research/tau2-bench) | `79975ac` | Retail / Airline environments (set `TAU2_SRC` to the clone) |
| [WebShop](https://github.com/princeton-nlp/WebShop) | `64fa2a5c` | WebShop environment + product data (follow its own data setup) |

LLM access: put `ARK_API_KEY=...` in a `.env` at the repository root (or
export it), and point `llm.base_url` / model names in
`benchmarks/*/config/runtime.yaml` at your provider. Cross-provider
backbone runs additionally use a model gateway: export `LLM_GATEWAY_URL`
and `GPT_GATEWAY_KEY`, and replace the `ep-YOUR-ENDPOINT-*` placeholders
with your own provisioned endpoints if your provider uses them. Agent and
simulator run on different model families by design.

**Run every command from its benchmark directory**
(`benchmarks/webshop/` or `benchmarks/tau2/`): the pipeline resolves
`config/` and `artifacts/` from the working directory (or the `IG_HOME`
environment variable).

## Quickstart

The released datasets ship in the repository, so the fastest loop needs no
external host at all:

```bash
cd benchmarks/webshop
it episode run -a webshop --limit 3                           # oracle smoke test
                                                              # (fake LLM, no network)
pytest tests -m unit                                          # offline unit suite
```

## Reproducing the benchmark

**1. Construction** (optional — the released instances ship in
`benchmarks/*/artifacts` and `data/tau2`).

```bash
# WebShop (from benchmarks/webshop/)
it generate -a webshop                                   # intent graphs
it battery  -a webshop                                   # consistency battery
it episode export -a webshop                             # generate-then-verify
python tools/build_composite_subsets.py --k 2            # composed faults

# tau2 (from benchmarks/tau2/)
python build_trees_airline.py         # + airline_policy_gate.py
python perturb_run_retail.py          # writes artifacts/retail_samples_v2.json
python perturb_composite.py --domain retail --k 2
python export_dataset.py              # repackages the perturbation outputs present
                                      # in artifacts/ into data/tau2; the released
                                      # bundle at data/tau2 ships prebuilt
python run_battery.py && python collision_audit.py
```

**2. Episodes.** WebShop needs its environment server running (see the
header of `benchmarks/webshop/tools/run_ws_backbones.sh` for the measured
server/worker geometry).

```bash
# WebShop campaign driver (from benchmarks/webshop/)
python -m intent_graph.cli episode exp run --arm A0 \
    --manifest experiments/persona2/views/p500_rational.json \
    --samples artifacts/samples_v2 --out runs/demo --procs 8 \
    --set shift_fire_prob=0.6

# tau2 (from benchmarks/tau2/)
python episodes_run_retail.py  --arm A0 --persona dependent --shift
python episodes_run_airline.py --arm A3 --persona rational  --shift \
    --shift-fire-prob 0.6
```

The reported operating point: patience base 10 (ask = 2, rejected
proposal = 4, environment actions free), persona-scheduled silent shifts
with fire coin 0.6, turn caps WebShop 40 / retail 30 / airline 100.

**3. Evaluation (GRIP).**

```bash
# from benchmarks/webshop/
python tools/grip_v3_report.py --run <run_dir>     # G/I/P metrics
python tools/rolereal_judge.py --run <run_dir>     # R judges
# from benchmarks/tau2/
python analyze_grip_v2.py <set>   # <set> names a directory under artifacts/
                                  # holding the per-cell run dirs written by
                                  # the episode drivers
```

Every metric is recomputable from the stored per-episode JSON; only the
Role-realism judges call a model, once, offline, and their raw judgments
are written next to the run.

## Tests

```bash
cd benchmarks/webshop && pytest tests -m unit     # pure logic, offline
cd benchmarks/tau2    && pytest tests -m unit
```

Both suites exercise the same `pipeline/` package — one under each
benchmark's configuration and adapters.

## Notes

* `pipeline/src/intent_tree` is a symlink to `intent_graph` (the package
  was renamed during development; driver scripts import both names). On
  non-POSIX checkouts, recreate the link or copy the directory.
* Code comments preserve dated design rulings ("ruling 2026-08-19") from
  the development log; the dates are load-bearing for provenance, the
  attributions are anonymized. Placeholders such as
  `https://YOUR-GATEWAY.example/v1` and `ep-YOUR-ENDPOINT-A` mark
  account-specific values you must supply.
* The auxiliary dbbench/os adapters in `pipeline/src` are not part of the
  released benchmarks; their `agentbench_repo` config key is optional.

## Citation

```bibtex
@inproceedings{driftbenchpp2027,
  title     = {Beyond Oracle Communication: Benchmarking Interactive
               Intent Alignment Under Miscommunication and Evolving
               User Intent},
  author    = {Anonymous},
  booktitle = {Under review},
  year      = {2027},
  url       = {https://drift-bench-plus.github.io/}
}
```

## License

MIT (see `LICENSE`).
