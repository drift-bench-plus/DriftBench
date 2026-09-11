"""Candidate arithmetic: compute what the pool disagrees on instead of asking the model.

B2's importance ("would the answer change which product I buy") and B3's split ("which
attribute divides the pool") are arithmetic over the candidate set in their papers. Left to
the model they never produce a "no": across seven arms the silent-episode rate was 0-2.7%.
The store's reply is structured, so the quantity is computable.
"""

from __future__ import annotations

from intent_graph.runtime.agents import (attribute_disagreement, candidate_report,
                                        parse_search_results)

STORE_REPLY = """3100 products match most of those words; best 3:
  B0001GUOIQ | $9.95 | match 100% | Vivitar ViviCam 4MP Digital Camera
      features: optical zoom, batteries included, digital camera
      options: {'color': ['black', 'silver'], 'size': ['one']}
  B0007CZ3EE | $17.59 | match 100% | Olympus D545 4MP Digital Camera
      features: optical zoom, batteries included, digital camera
      options: {'color': ['black'], 'size': ['one']}
  B000A7B9XU | $19.54 | match 90% | Olympus FE-120 6MP Digital Camera
      features: optical zoom, digital camera, waterproof
      options: {'color': ['red'], 'size': ['one']}
"""


def test_parses_every_candidate_with_its_details():
    cands = parse_search_results(STORE_REPLY)
    assert [c["asin"] for c in cands] == ["B0001GUOIQ", "B0007CZ3EE", "B000A7B9XU"]
    assert cands[0]["price"] == 9.95
    assert cands[0]["match"] == 100
    assert "optical zoom" in cands[1]["features"]
    assert cands[0]["options"]["color"] == ["black", "silver"]


def test_junk_input_is_safe():
    assert parse_search_results("") == []
    assert parse_search_results("No product matches most of ['xyz'].") == []
    assert parse_search_results(None) == []


def test_universal_feature_scores_zero_and_split_feature_scores_high():
    scores = attribute_disagreement(parse_search_results(STORE_REPLY))
    # every candidate is a digital camera with optical zoom -> asking cannot change the choice
    assert scores["digital camera"] == 0.0
    assert scores["optical zoom"] == 0.0
    # two of three have it -> genuinely divides the pool
    assert scores["batteries included"] > 0.5
    assert scores["waterproof"] > 0.5


def test_single_candidate_has_nothing_to_split():
    one = parse_search_results("""1 product matches; best 1:
  B1 | $5.00 | match 100% | Thing
      features: a, b
""")
    assert len(one) == 1
    assert attribute_disagreement(one) == {}


def test_report_names_the_agreements_and_the_splits():
    rep = candidate_report(STORE_REPLY)
    assert "DISAGREE on" in rep
    assert "waterproof" in rep or "batteries included" in rep
    assert "ALL already satisfy" in rep
    assert "optical zoom" in rep


def test_report_is_explicit_when_nothing_splits():
    same = """2 products match; best 2:
  B1 | $5.00 | match 100% | Thing one
      features: a, b
  B2 | $6.00 | match 100% | Thing two
      features: a, b
"""
    rep = candidate_report(same)
    assert "disagree on NOTHING" in rep


def test_report_empty_when_there_is_no_pool():
    assert candidate_report("") == ""
    assert candidate_report("No product matches most of ['zzz'].") == ""
