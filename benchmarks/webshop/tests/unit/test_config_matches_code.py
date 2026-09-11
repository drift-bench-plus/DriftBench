"""The shipped config must describe the experiments we actually run.

Two failures on 2026-08-19 motivated this file, and both were invisible until someone read
the trajectories instead of the config:

  * `cost_ask`/`cost_reject`/`patience_init` said 1/2/14 while every reported run charged
    2/4 from 10. The file had been left behind by an older generation of experiments.
  * `shift_scheduled` and `user_v2` -- the master switches for axis B and for the v2
    simulated user -- were absent entirely, so the shipped config silently ran with the
    intent shift OFF and the old user, while every reported run passed them on the command
    line.

A config that does not reproduce the runs is worse than no config, because it looks
authoritative. These tests are cheap and they pin the operating point.
"""
import re
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
SRC = REPO.parents[1] / "pipeline" / "src" / "intent_graph"
_CFG = yaml.safe_load((REPO / "config" / "runtime.yaml").read_text())
RUNTIME = _CFG["runtime"]
LLM = _CFG["llm"]


def _keys_read_in_code() -> set[str]:
    out: set[str] = set()
    for f in SRC.rglob("*.py"):
        t = f.read_text()
        out |= set(re.findall(r'rt\.get\("([a-z_0-9]+)"', t))
        out |= set(re.findall(r'runtime"\]\.get\("([a-z_0-9]+)"', t))
    return out


@pytest.mark.parametrize("key,expected", [
    ("patience_init", 10),
    ("cost_ask", 2),
    ("cost_reject", 4),
    ("cost_act", 0),
])
def test_the_patience_economy_is_the_one_the_runs_used(key, expected):
    """Read back out of the trajectories: runs w32/w40/w48/nothink and every tau2 run
    charge a question 2 and a refused proposal 4, from a base of 10."""
    assert RUNTIME[key] == expected, (
        f"{key} is {RUNTIME[key]}, but the reported runs used {expected}")


@pytest.mark.parametrize("switch", ["shift_scheduled", "user_v2"])
def test_master_switches_are_declared_not_left_to_a_default(switch):
    """These gate whole subsystems. Leaving them out means the shipped config runs a
    different experiment from the one we report, silently."""
    assert switch in RUNTIME, f"{switch} is not declared; it would default to OFF"
    assert RUNTIME[switch] is True, f"{switch} is declared but OFF"


def test_the_shift_schedule_mode_is_actually_reachable():
    """`shift_schedule: persona` does nothing unless `shift_scheduled` is true. A config
    that names a mode its own master switch disables is a lie."""
    if RUNTIME.get("shift_schedule") == "persona":
        assert RUNTIME.get("shift_scheduled") is True, (
            "shift_schedule says 'persona' but shift_scheduled is off, so no shift fires")


def test_no_declared_knob_is_dead():
    """A key in the config that nothing reads is a knob that silently does nothing."""
    dead = sorted(set(RUNTIME) - _keys_read_in_code())
    assert not dead, f"declared but never read anywhere in the source: {dead}"


def test_questions_are_limited_only_by_patience():
    assert "ask_budget" not in RUNTIME
    agents = (SRC / "runtime" / "agents.py").read_text()
    assert 'rt.get("ask_budget"' not in agents


def test_no_thinking_mode_anywhere():
    """the author 2026-08-19: "none of the thinking mode everywhere". The post-2026-08-16
    campaigns ran with it on via the command line while the config said nothing, so a
    config-only run silently had hidden reasoning back ON (reasoning_effort: minimal is the
    elif branch). Declaring and pinning it closes that hole."""
    assert LLM.get("disable_thinking") is True
