"""Adapter loading and parsing against the real datasets (no docker needed).

Run with: pytest -m data
"""

import pytest

from intent_graph.adapters.dbbench import DBBenchAdapter, compile_sql, parse_sql
from intent_graph.adapters.dbbench_compare import compare_results
from intent_graph.adapters.osbench import OSBenchAdapter, compile_command, parse_command
from intent_graph.adapters.webshop import WebShopAdapter, deterministic_price_upper
from intent_graph.cli import load_config


@pytest.fixture(scope="module")
def cfg():
    return load_config()


# ------------------------------------------------------------------------- DBBench
def test_dbbench_loads_and_parses(cfg):
    seeds = DBBenchAdapter(cfg).load()
    assert len(seeds) > 4000
    parseable = [s for s in seeds if s.parseable]
    # parse coverage is a reported metric, not a silent failure: unparsed records stay
    # usable as pivot targets
    assert len(parseable) / len(seeds) > 0.5
    assert len({s.env_key for s in seeds}) > 1000


def test_dbbench_write_and_read_tasks_both_present(cfg):
    seeds = DBBenchAdapter(cfg).load()
    kinds = {s.base["kind"] for s in seeds}
    assert {"SELECT", "UPDATE", "INSERT"} <= kinds


@pytest.mark.parametrize("sql,kind,n_conds", [
    ("SELECT Opponent FROM `t` WHERE Date = 'Nov 1';", "SELECT", 1),
    ("SELECT COUNT(*) FROM t WHERE a = '1' AND b = '2';", "SELECT", 2),
    ("UPDATE `t` SET Class='F1', Circuit='Atlanta' WHERE Race='GP';", "UPDATE", 3),
    ("DELETE FROM t WHERE id = '5';", "DELETE", 1),
    ("INSERT INTO `t` (Year, Title) VALUES ('2019', 'Aoharu');", "INSERT", 2),
])
def test_dbbench_sql_round_trips(sql, kind, n_conds):
    parsed = parse_sql(sql)
    assert parsed is not None
    base, conds = parsed
    assert base["kind"] == kind and len(conds) == n_conds
    # recompiling must produce runnable SQL of the same shape
    assert compile_sql(base, conds).upper().startswith(kind)


def test_dbbench_refuses_sql_it_cannot_faithfully_rebuild():
    """Better to leave a record as a pivot target than to mangle it."""
    assert parse_sql("SELECT a FROM t WHERE x = '1' OR y = '2';") is None
    assert parse_sql("SELECT a FROM t WHERE x IN (1,2);") is None
    assert parse_sql("DELETE FROM t;") is None  # unconditional: no intent to move


def test_dbbench_comparison_uses_benchmark_tolerance():
    assert compare_results([["4.109"]], ["4.11"], "SELECT")     # within 1e-2
    assert not compare_results([["4.5"]], ["4.11"], "SELECT")
    assert compare_results([[None]], ["0"], "SELECT")           # null normalizes to "0"


# --------------------------------------------------------------------------- OSBench
def test_osbench_loads_with_reference_commands(cfg):
    seeds = OSBenchAdapter(cfg).load()
    assert len(seeds) > 1000
    assert all(s.meta["raw_example"] for s in seeds)
    assert {s.meta["comparator"] for s in seeds} <= {
        "integer-match.py", "string-match.py", "size-match.py"}


def test_osbench_excludes_dev_split(cfg):
    """dev.json holds the official dev tasks and the bare-literal (match-type) ones."""
    a = OSBenchAdapter(cfg)
    descriptions = {s.meta["description"] for s in a.load()}
    assert "There's a program that writes 'succeed' into /root/log periodically. " \
           "I only know the interval is 1 second, 2 seconds, or 3 seconds. " \
           "Please find out the interval. Answer me only the integer number of " \
           "seconds. (e.g. 1, 2, or 3)" not in descriptions


@pytest.mark.parametrize("cmd", [
    "find ~/ -type f -mtime -7 | wc -l",
    "find $HOME -type f -mtime -7 -size +100k | wc -l",
    "grep -r 'Linux' /home/student/documents/*.txt | wc -l",
])
def test_osbench_command_round_trips_semantically(cmd):
    parsed = parse_command(cmd)
    assert parsed is not None
    base, conds = parsed
    rebuilt = compile_command(base, conds)
    # flag order and quoting may differ; the token multiset must not
    assert set(rebuilt.replace("'", "").split()) == set(cmd.replace("'", "").split())


def test_osbench_refuses_multi_statement_scripts():
    assert parse_command("cd /tmp; ls | wc -l") is None
    assert parse_command("for f in *; do echo $f; done") is None


# -------------------------------------------------------------------------- WebShop
def test_webshop_loads_human_goals(cfg):
    """Only the configured splits load. Default config is roots=test, retrieval=test+eval,
    so this is ~1.4k seeds rather than all 12,087 -- loading `train` too would make the
    test set unusable as a test set."""
    seeds = WebShopAdapter(cfg).load()
    assert 1000 < len(seeds) < 2000, len(seeds)
    assert all(s.parseable for s in seeds)
    # every goal carries at least one attribute -- goal.py skips the others, so do we
    assert all(any(c[0].startswith("attr:") for c in s.conditions) for s in seeds)
    assert {s.meta["split"] for s in seeds} == {"test", "eval"}


def test_webshop_loads_every_goal_when_all_splits_are_requested(cfg):
    """The split filter must be the only thing limiting yield -- not a lost record."""
    wide = {**cfg, "webshop": {"root_splits": ["test", "eval", "train"],
                               "retrieval_splits": ["test", "eval", "train"]}}
    assert len(WebShopAdapter(wide).load()) > 10_000


def test_webshop_price_ceiling_is_deterministic():
    """WebShop samples this at random; we derive it, so ground truth cannot drift."""
    assert deterministic_price_upper(39.95) == deterministic_price_upper(39.95)
    assert deterministic_price_upper(39.95) > 39.95


def test_webshop_clusters_are_topical_not_environments(cfg):
    a = WebShopAdapter(cfg)
    seeds = a.load()
    clusters = {s.env_key for s in seeds}
    assert len(clusters) > 100
    spec = a.env_spec(next(iter(clusters)))
    assert spec["kind"] == "webshop" and "cluster" in spec
