"""Acceptance semantics and agent-action classification."""

import pytest

from intent_graph.adapters.dbbench import DBBenchAdapter
from intent_graph.adapters.osbench import OSBenchAdapter
from intent_graph.adapters.webshop import WebShopAdapter
from intent_graph.models import GroundTruth, Node
from intent_graph.runtime.accept import accepts, generic_accepts


def node(gt, base=None, recipe=None, conds=(("a", "=", 1),)):
    return Node.build(adapter="t", adapter_version="1", env_id="e", base=base or {"f": 1},
                      conditions=conds, recipe=recipe or {}, ground_truth=gt)


# ------------------------------------------------------- GroundTruth.contains
def test_rowset_membership_matches_canonicalization():
    """`value` holds canonical-JSON strings, so raw objects must be encoded to compare."""
    gt = GroundTruth.rowset([["Alice", 3], ["Bob", 4]])
    assert gt.contains(["Alice", 3])
    assert not gt.contains(["Alice", 4])


def test_rowset_rows_decodes_back_to_objects():
    gt = GroundTruth.rowset([["Alice", 3]])
    assert gt.rows() == [["Alice", 3]]
    with pytest.raises(ValueError):
        GroundTruth.scalar("5").rows()


def test_purchaseset_membership_ignores_option_order():
    gt = GroundTruth.purchaseset([("A1", {"size": "s", "color": "red"})])
    assert gt.contains(("A1", {"color": "red", "size": "s"}))
    assert not gt.contains(("A2", {"color": "red", "size": "s"}))


def test_scalar_membership_strips():
    assert GroundTruth.scalar("5").contains(" 5 ")


def test_accept_any_member_not_just_the_first():
    """WebShop GT sets have median 2 and reach 140; a single-canonical-answer verifier
    would reject correct behaviour."""
    gt = GroundTruth.purchaseset([("A1", {}), ("A2", {}), ("A3", {})])
    assert all(gt.contains((a, {})) for a in ("A1", "A2", "A3"))


def test_generic_accepts_rowset_requires_the_whole_answer():
    """A read answer is the COMPLETE result set, not one row of it — that is how the SQL
    benchmarks compare answers."""
    gt = GroundTruth.rowset([["a"], ["b"]])
    ok, why = generic_accepts(None, ["a", "b"], node(gt), None)
    assert ok and why == "row_set_match"
    partial, why = generic_accepts(None, ["a"], node(gt), None)
    assert not partial and why == "row_set_mismatch"


def test_generic_accepts_purchaseset_is_membership():
    """WebShop is the opposite: many distinct purchases are each fully correct."""
    gt = GroundTruth.purchaseset([("A1", {}), ("A2", {})])
    for asin in ("A1", "A2"):
        ok, why = generic_accepts(None, (asin, {}), node(gt), None)
        assert ok and why == "in_ground_truth"


def test_dispatcher_prefers_the_adapter_method():
    class Ad:
        name = "x"
        def accepts(self, p, n, s, *, executor=None):
            return True, "adapter_said_so"
    assert accepts(Ad(), "anything", node(GroundTruth.rowset([["a"]])), None) == \
        (True, "adapter_said_so")


# ------------------------------------------------------------------ dbbench
def cfg():
    return {"paths": {"agentbench_repo": "/nonexistent"}}


@pytest.mark.parametrize("sql,mutating", [
    ("SELECT a FROM t", False),
    ("  select a from t;", False),
    ("SHOW TABLES", False),
    ("EXPLAIN SELECT 1", False),
    ("SELECT a INTO OUTFILE '/tmp/x' FROM t", True),   # a SELECT that writes
    ("UPDATE t SET a=1", True),
    ("insert into t values (1)", True),
    ("DELETE FROM t WHERE a=1", True),
    ("DROP TABLE t", True),
    ("-- comment\nUPDATE t SET a=1", True),
    ("", True),                                        # unknown => fail closed
    ("gibberish !!", True),
])
def test_dbbench_act_classification(sql, mutating):
    assert DBBenchAdapter(cfg()).act_is_mutating(sql) is mutating


def test_dbbench_read_acceptance_decodes_ground_truth():
    """The bug this guards: passing gt.value (JSON strings) rejects every correct answer."""
    a = DBBenchAdapter(cfg())
    gt = GroundTruth.rowset([["FC Kharkiv"]])
    ok, why = a.accepts(["FC Kharkiv"], node(gt), None)
    assert ok, why


def test_dbbench_read_acceptance_uses_float_tolerance():
    """DBBench's own comparison allows 1e-2; we must not tighten it."""
    a = DBBenchAdapter(cfg())
    gt = GroundTruth.rowset([["4.11"]])
    assert a.accepts(["4.109"], node(gt), None)[0]
    assert not a.accepts(["4.5"], node(gt), None)[0]


def test_dbbench_write_acceptance_requires_an_executor():
    a = DBBenchAdapter(cfg())
    ok, why = a.accepts({"sql": "UPDATE t SET a=1"},
                        node(GroundTruth.statehash("abc"), base={"table": "t"}), None)
    assert not ok and "executor" in why


# ------------------------------------------------------------------ osbench
@pytest.mark.parametrize("cmd,mutating", [
    ("find ~ -type f | wc -l", False),
    ("grep -r x .", False),
    ("ls -la", False),
    ("cat f", False),
    ("echo hi > /root/f", True),          # redirection
    ("sed -i s/a/b/ f", True),
    ("find . -delete", True),
    ("find . -exec rm {} ;", True),
    ("rm -rf /root/data", True),
    ("touch f", True),
    ("cp a b", True),
    ("python3 script.py", True),          # unknown tool => fail closed
    ("", True),
    ("wc -l f | tee out", True),
])
def test_osbench_act_classification(cmd, mutating):
    assert OSBenchAdapter(cfg()).act_is_mutating(cmd) is mutating


def test_osbench_acceptance_uses_the_nodes_own_comparator():
    a = OSBenchAdapter(cfg())
    n = node(GroundTruth.scalar("5"), base={"tool": "find", "path": "~",
                                            "comparator": "integer-match.py"})
    assert a.accepts("5", n, None)[0]
    assert a.accepts(" 5 ", n, None)[0]            # integer-match tolerates whitespace
    assert not a.accepts("6", n, None)[0]


def test_osbench_acceptance_defaults_when_comparator_missing():
    a = OSBenchAdapter(cfg())
    n = node(GroundTruth.scalar("abc"), base={"tool": "find", "path": "~"})
    assert a.accepts("abc", n, None)[0]


def test_osbench_integer_comparator_rejects_non_numeric_rather_than_erroring():
    a = OSBenchAdapter(cfg())
    n = node(GroundTruth.scalar("5"), base={"comparator": "integer-match.py"})
    ok, _ = a.accepts("I think it is five", n, None)
    assert ok is False


# ------------------------------------------------------------------ webshop
@pytest.mark.parametrize("act,mutating", [
    ("buy B01", True), ("purchase B01", True), ("search[red shoes]", False),
    ("click[B01]", False), ("", True),
])
def test_webshop_act_classification(act, mutating):
    a = WebShopAdapter({"paths": {"webshop_repo": "/x", "webshop_derived": "/x"}})
    assert a.act_is_mutating(act) is mutating


@pytest.mark.parametrize("raw,expected", [
    ("buy B09HX5CD2D", ("B09HX5CD2D", {})),
    ('buy B09HX5CD2D {"size": "Small"}', ("B09HX5CD2D", {"size": "small"})),
    ("B09HX5CD2D", ("B09HX5CD2D", {})),
    ("nonsense here!!", None),
])
def test_webshop_proposal_parsing(raw, expected):
    a = WebShopAdapter({"paths": {"webshop_repo": "/x", "webshop_derived": "/x"}})
    assert a.parse_proposal(raw) == expected
