"""The official WebShop split must be honoured exactly.

Why this is load-bearing: goal_idx is what defines train/eval/test, and it is implied by
iteration order rather than stored in the dataset. Before this existed the adapter walked
`items_human_ins.json` as a dict, so seeds had no goal_idx and any benchmark built from them
silently mixed training goals into the test set.
"""

import json

import pandas as pd
import pytest

from intent_graph.cli import build, load_config

pytestmark = pytest.mark.data

TEST_MAX = 500      # test = goal_idx 0..499  (baseline_models/env.py: range(500))
EVAL_MAX = 1500


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def index(cfg):
    from pathlib import Path
    return pd.read_parquet(Path(cfg["paths"]["webshop_derived"]) / "goal_index.parquet")


def test_goal_index_reproduces_webshops_own_totals(index):
    """12,087 goals in 500/1000/10587 is WebShop's published split. Any drift here means
    the ordering reproduction broke, and every split label is then suspect."""
    assert len(index) == 12_087
    counts = index.split.value_counts().to_dict()
    assert counts == {"train": 10_587, "eval": 1_000, "test": 500}


def test_goal_index_boundaries_match_the_official_ranges(index):
    by = index.set_index("goal_idx").split
    assert by[0] == "test" and by[499] == "test"
    assert by[500] == "eval" and by[1499] == "eval"
    assert by[1500] == "train"


def test_goal_index_skips_attributeless_instructions(cfg, index):
    """WebShop's goal.py skips instructions with no attributes, and they consume no index.
    If we counted them, every goal_idx past the first would be shifted."""
    from pathlib import Path
    human = json.loads(
        (Path(cfg["paths"]["webshop_repo"]) / "data" / "items_human_ins.json")
        .read_text(encoding="utf-8"))
    total = sum(len(v) for v in human.values())
    attributed = sum(1 for v in human.values() for e in v if e.get("instruction_attributes"))
    assert total - attributed == 164
    assert len(index) == attributed


def test_only_test_split_goals_become_roots(cfg):
    adapter, _ = build("webshop", cfg)
    seeds = adapter.load()
    roots = [s for s in seeds if s.root_eligible]
    assert roots, "no root-eligible seeds"
    assert all(s.meta["split"] == "test" for s in roots)
    assert all(s.meta["goal_idx"] < TEST_MAX for s in roots)


def test_eval_goals_are_loaded_but_only_as_retrieval(cfg):
    """Eval goals must be present -- they are the real sibling intents that make branches
    retrieved rather than synthesized -- but must never become a root."""
    adapter, _ = build("webshop", cfg)
    seeds = adapter.load()
    evals = [s for s in seeds if s.meta["split"] == "eval"]
    assert evals, "eval split should be loaded for retrieval"
    assert all(not s.root_eligible for s in evals)
    assert all(TEST_MAX <= s.meta["goal_idx"] < EVAL_MAX for s in evals)


def test_train_split_is_not_loaded_at_all(cfg):
    adapter, _ = build("webshop", cfg)
    assert not [s for s in adapter.load() if s.meta["split"] == "train"]


def test_missing_goal_index_fails_loudly(cfg):
    """Falling back to dict order would silently reintroduce split mixing."""
    import copy
    broken = copy.deepcopy(cfg)
    broken["paths"]["webshop_derived"] = "/nonexistent-path-for-test"
    adapter, _ = build("webshop", broken)
    with pytest.raises(FileNotFoundError, match="goal_index"):
        adapter._goal_index()
