"""Identity and canonical serialization: the basis of determinism."""

import pytest

from intent_graph.ids import canonical_dumps, config_hash, content_hash, intent_id, graph_id


def test_canonical_dumps_key_order_irrelevant():
    assert canonical_dumps({"b": 1, "a": 2}) == canonical_dumps({"a": 2, "b": 1})


def test_canonical_dumps_set_order_irrelevant():
    assert canonical_dumps({1, 2, 3}) == canonical_dumps({3, 1, 2})


def test_canonical_dumps_integral_floats_collapse():
    # 4.0 and 4 must not hash differently, or a re-executed answer looks like drift
    assert canonical_dumps(4.0) == canonical_dumps(4)


def test_canonical_dumps_rejects_nan():
    with pytest.raises(ValueError):
        canonical_dumps(float("nan"))


def test_canonical_dumps_rejects_unknown_type():
    with pytest.raises(TypeError):
        canonical_dumps(object())


def test_intent_id_condition_order_irrelevant():
    a = intent_id("x", "1", "env", {"f": 1}, [("a", "=", 1), ("b", "=", 2)])
    b = intent_id("x", "1", "env", {"f": 1}, [("b", "=", 2), ("a", "=", 1)])
    assert a == b


@pytest.mark.parametrize(
    "kwargs",
    [
        {"adapter": "y"},
        {"adapter_version": "2"},
        {"env_id": "other"},
        {"base": {"f": 2}},
        {"conditions": [("a", "=", 9)]},
    ],
)
def test_intent_id_changes_with_every_field(kwargs):
    baseline = dict(adapter="x", adapter_version="1", env_id="env",
                    base={"f": 1}, conditions=[("a", "=", 1)])
    assert intent_id(**baseline) != intent_id(**{**baseline, **kwargs})


def test_tree_id_child_order_irrelevant():
    assert graph_id("e", "r", ["c1", "c2"], "h") == graph_id("e", "r", ["c2", "c1"], "h")


def test_config_hash_ignores_paths_and_workers():
    a = {"seed": 42, "paths": {"x": "/a"}, "workers": 4}
    b = {"seed": 42, "paths": {"x": "/b"}, "workers": 99}
    assert config_hash(a) == config_hash(b)


def test_config_hash_tracks_generation_knobs():
    assert config_hash({"seed": 1}) != config_hash({"seed": 2})
    assert config_hash({"branching": 8}) != config_hash({"branching": 4})


def test_content_hash_is_stable_across_calls():
    assert content_hash("a", [1, 2]) == content_hash("a", [1, 2])
