"""Executor behaviour that only a real container can demonstrate.

Run with: pytest -m docker   (needs colima/docker running)
"""

import subprocess

import pytest

from intent_graph.adapters.osbench import integer_match, size_match, string_match
from intent_graph.cli import load_config
from intent_graph.executors.mysql_docker import janitor
from intent_graph.executors.os_docker import OSExecutor


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def os_executor(cfg):
    ex = OSExecutor(cfg)
    yield ex
    ex.shutdown()
    janitor(cfg["docker"]["label"])


ENV = {"env_id": "test", "kind": "os",
       "init": 'mkdir -p ~/d && echo -e "Linux one\\nLinux two\\nother" > ~/d/a.txt'}


def test_container_provides_gnu_userland(os_executor):
    """The reason everything runs in a container: macOS has BSD tools, tasks need GNU."""
    with os_executor.open(ENV) as s:
        assert s.run({"command": 'date --date="3 days ago" +%Y'}).strip().isdigit()
        assert s.run({"command": 'touch -d "8 days ago" ~/d/a.txt && echo ok'}) == "ok"


def test_gnu_flags_are_absent_on_the_host():
    """Documents why we never execute recipes on the host."""
    r = subprocess.run(["date", "--date=3 days ago"], capture_output=True, text=True)
    assert r.returncode != 0


def test_init_shapes_the_environment(os_executor):
    with os_executor.open(ENV) as s:
        assert s.run({"command": "grep -c Linux ~/d/a.txt"}) == "2"


def test_each_session_is_isolated(os_executor):
    """A file written in one session must not exist in the next."""
    with os_executor.open(ENV) as s:
        s.run({"command": "touch ~/d/leaked.txt && echo done"})
        assert s.run({"command": "ls ~/d/leaked.txt"}).endswith("leaked.txt")
    with os_executor.open(ENV) as s:
        assert s.run({"command": "ls ~/d/leaked.txt 2>/dev/null || echo absent"}) == "absent"


def test_rematerialize_resets_state(os_executor):
    with os_executor.open(ENV) as s:
        s.run({"command": "touch ~/d/tmp.txt && echo ok"})
        s.rematerialize()
        assert s.run({"command": "ls ~/d/tmp.txt 2>/dev/null || echo absent"}) == "absent"


def test_network_is_disabled(os_executor):
    """Uses bash's /dev/tcp rather than curl: our image ships no curl, so a
    command-not-found would otherwise make this pass for the wrong reason."""
    with os_executor.open(ENV) as s:
        out = s.run({"command": "timeout 3 bash -c 'echo > /dev/tcp/1.1.1.1/443' "
                                "2>/dev/null && echo reachable || echo blocked"})
        assert out == "blocked"


def test_the_network_probe_can_detect_reachability(cfg):
    """Guards the test above: the same probe must say 'reachable' with networking on."""
    r = subprocess.run(
        ["docker", "run", "--rm", "--label", f"{cfg['docker']['label']}=1",
         cfg["docker"]["os_image"], "bash", "-c",
         "timeout 5 bash -c 'echo > /dev/tcp/1.1.1.1/443' 2>/dev/null "
         "&& echo reachable || echo blocked"],
        capture_output=True, text=True, timeout=120)
    assert r.stdout.strip() == "reachable", "probe cannot distinguish network states"


def test_failing_init_is_reported_not_swallowed(os_executor):
    bad = {"env_id": "bad", "kind": "os", "init": "exit 3"}
    with pytest.raises(RuntimeError, match="init script exited"):
        with os_executor.open(bad):
            pass


def test_janitor_leaves_nothing_behind(cfg, os_executor):
    with os_executor.open(ENV) as s:
        s.run({"command": "echo hi"})
    janitor(cfg["docker"]["label"])
    remaining = subprocess.run(
        ["docker", "ps", "-aq", "--filter", f"label={cfg['docker']['label']}=1"],
        capture_output=True, text=True).stdout.split()
    assert remaining == []


# ------------------------------------------------- ported comparators (no docker needed)
@pytest.mark.parametrize("a,b,expected", [
    ("5", "5", True), (" 5\n", "5", True), ("5", "6", False), ("abc", "5", False),
])
def test_integer_match(a, b, expected):
    assert integer_match(a, b) is expected


def test_string_match_normalizes_line_endings():
    assert string_match("a\r\nb", "a\nb")
    assert not string_match("a", "b")


@pytest.mark.parametrize("a,b,expected", [
    ("1K", "1024", True), ("1MB", "1048576", True), ("2K", "1024", False),
    ("garbage", "1", False),
])
def test_size_match(a, b, expected):
    assert size_match(a, b) is expected
