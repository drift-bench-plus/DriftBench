"""Adapter protocol + registry.

An adapter is the only dataset-specific code in the pipeline.  It answers seven
questions about one benchmark; the engine composes those answers into graphs without
knowing whether it is looking at SQL, shell commands or a shopping list.

Everything the engine needs from a dataset:

  load()      what records exist, grouped by which environment
  validate()  does this record's recipe reproduce the answer the dataset ships?
  compile()   conditions -> a runnable recipe
  execute()   recipe -> canonical ground truth
  witness()   the objects behind the answer (rows/files/products) with properties
  domains()   the values a slot actually takes in this environment
  env_spec()  how to materialize the environment
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from ..models import Condition, GroundTruth, Seed


@runtime_checkable
class Adapter(Protocol):
    name: str
    version: str
    executor_name: str

    def load(self) -> list[Seed]: ...

    def validate(self, seed: Seed, session) -> bool:
        """True iff executing the seed's recipe reproduces its shipped answer.

        Comparison must use the BENCHMARK'S OWN semantics (north-star #2): we never
        substitute our own notion of equality for the one the dataset ships with.
        """
        ...

    def compile(self, base: dict, conditions: tuple[Condition, ...]) -> Any: ...

    def execute(self, recipe: Any, session) -> GroundTruth: ...

    def witness(self, base: dict, conditions: tuple[Condition, ...], session) -> list[dict]:
        """Objects behind the answer, each a flat dict of inspectable properties.

        Refinement candidates are mined from here: a property held by SOME but not ALL
        witness objects, when added as a condition, structurally guarantees a non-empty
        and strictly smaller answer.
        """
        ...

    def domains(self, slot: str, base: dict, session) -> list[Any]:
        """Values ``slot`` actually takes in this environment (substitution candidates)."""
        ...

    def env_spec(self, env_key: str) -> dict: ...

    def is_mutating(self, recipe: Any) -> bool:
        """Does executing this recipe change environment state?"""
        ...


_REGISTRY: dict[str, tuple[type, type]] = {}


def register(name: str, adapter_cls: type, executor_cls: type) -> None:
    _REGISTRY[name] = (adapter_cls, executor_cls)


def get(name: str) -> tuple[type, type]:
    if name not in _REGISTRY:
        raise KeyError(f"unknown adapter {name!r}; registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def available() -> list[str]:
    return sorted(_REGISTRY)


def load_builtin() -> None:
    """Import adapter modules for their registration side effects."""
    from . import toybench  # noqa: F401

    for mod in ("dbbench", "osbench", "webshop"):
        try:
            __import__(f"intent_graph.adapters.{mod}")
        except ImportError:  # optional deps (mysql connector, spacy) may be absent
            pass
