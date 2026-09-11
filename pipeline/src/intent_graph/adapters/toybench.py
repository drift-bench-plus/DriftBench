"""Toybench: a synthetic dataset that exists only so the engine can be tested.

A tiny in-memory "shape store" with four slots (color, size, material, price_band).  It
is deliberately small enough that every expected output can be worked out by hand, and it
covers the cases real adapters must handle: records that fail validation, sibling records
that form known REAL branch pairs, a second frame for pivots, and a redundant condition
whose removal leaves the answer unchanged (so ``gt_moved=false`` gets exercised).

No datasets, no docker, no network -- this is what ``pytest -m unit`` runs against.
"""

from __future__ import annotations

from ..executors.base import BaseSession
from ..models import Condition, GroundTruth, Seed
from . import base as registry

ITEMS = [
    # id      color    size     material  price_band  in_stock
    ("i01", "red",   "small",  "cotton", "low",  "yes"),
    ("i02", "red",   "small",  "wool",   "low",  "yes"),
    ("i03", "red",   "large",  "cotton", "high", "yes"),
    ("i04", "blue",  "small",  "cotton", "low",  "yes"),
    ("i05", "blue",  "large",  "wool",   "high", "yes"),
    ("i06", "green", "small",  "cotton", "mid",  "yes"),
    ("i07", "green", "large",  "linen",  "mid",  "yes"),
    ("i08", "red",   "small",  "cotton", "mid",  "yes"),
]
SLOTS = ("color", "size", "material", "price_band", "in_stock")

# `in_stock` is "yes" for every item, so a condition on it is redundant: dropping it
# leaves the answer unchanged (exercises gt_moved=false, decision D7) and it can never be
# a legal refinement candidate (refinement requires a property held by SOME but not ALL).
_ROWS = [dict(zip(("id", *SLOTS), row, strict=True)) for row in ITEMS]


class ToySession(BaseSession):
    """Read-only in-process environment; nothing can mutate, so nothing can leak."""

    def _materialize(self) -> None:
        self.rows = list(_ROWS)

    def _execute(self, recipe):
        conds = recipe["conditions"]
        out = []
        for r in self.rows:
            if all(str(r.get(slot)) == str(val) for slot, op, val in conds if op == "="):
                out.append(r["id"])
        return sorted(out)


class ToyExecutor:
    name = "toy"

    def open(self, env_spec: dict) -> ToySession:
        return ToySession(self, env_spec)

    def shutdown(self) -> None:
        pass


class ToyAdapter:
    name = "toybench"
    version = "1"
    executor_name = "toy"

    def load(self) -> list[Seed]:
        frame = {"frame": "shop"}
        other = {"frame": "outlet"}
        return [
            # seed: two conditions
            Seed("t_seed", "toyenv", frame, (("color", "=", "red"), ("size", "=", "small")),
                 shipped_answer=["i01", "i02", "i08"]),
            # REAL refinement of the seed
            Seed("t_refine", "toyenv", frame,
                 (("color", "=", "red"), ("size", "=", "small"), ("material", "=", "cotton")),
                 shipped_answer=["i01", "i08"]),
            # REAL relaxation of the seed
            Seed("t_relax", "toyenv", frame, (("color", "=", "red"),),
                 shipped_answer=["i01", "i02", "i03", "i08"]),
            # REAL substitution (same slots, different value)
            Seed("t_subst", "toyenv", frame, (("color", "=", "blue"), ("size", "=", "small")),
                 shipped_answer=["i04"]),
            # REAL pivot: different frame, same environment
            Seed("t_pivot", "toyenv", other, (("material", "=", "linen"),),
                 shipped_answer=["i07"]),
            # carries a redundant condition: relaxing `in_stock` leaves the answer put,
            # so this seed produces gt_moved=false edges in both directions
            Seed("t_redundant", "toyenv", frame,
                 (("color", "=", "red"), ("size", "=", "small"), ("in_stock", "=", "yes")),
                 shipped_answer=["i01", "i02", "i08"]),
            # invalid: shipped answer does not match what the recipe returns
            Seed("t_bad_answer", "toyenv", frame, (("color", "=", "green"),),
                 shipped_answer=["i99"]),
            # invalid: unsatisfiable
            Seed("t_bad_empty", "toyenv", frame, (("color", "=", "purple"),),
                 shipped_answer=[]),
            # unparseable -> pivot target only, excluded from constraint operators
            Seed("t_unparsed", "toyenv", frame, None, shipped_answer=None),
        ]

    # -- the seven contract methods -------------------------------------------------
    def env_spec(self, env_key: str) -> dict:
        return {"env_id": "toyenv", "kind": "toy"}

    def compile(self, base: dict, conditions: tuple[Condition, ...]):
        return {"frame": base["frame"], "conditions": [list(c) for c in conditions]}

    def execute(self, recipe, session) -> GroundTruth:
        return GroundTruth.rowset([[i] for i in session.run(recipe)])

    def validate(self, seed: Seed, session) -> bool:
        if not seed.parseable or seed.shipped_answer is None:
            return False
        got = session.run(self.compile(seed.base, seed.conditions))
        return bool(got) and sorted(got) == sorted(seed.shipped_answer)

    def witness(self, base, conditions, session) -> list[dict]:
        ids = set(session.run(self.compile(base, conditions)))
        return [{k: v for k, v in r.items() if k != "id"} for r in _ROWS if r["id"] in ids]

    def domains(self, slot: str, base: dict, session) -> list:
        return sorted({str(r[slot]) for r in _ROWS if slot in r})

    def is_mutating(self, recipe) -> bool:
        return False


registry.register("toybench", ToyAdapter, ToyExecutor)
