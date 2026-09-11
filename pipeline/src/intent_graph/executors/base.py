"""Executor protocol: how a recipe gets run against an environment.

The engine never talks to MySQL, Docker or WebShop directly.  It opens a session on an
environment and runs recipes in it.  The session owns the isolation guarantee:

    A mutating recipe always sees a freshly materialized environment.

That rule exists because DB write tasks derive their ground truth from post-execution
state; running two write candidates against one database would make every hash after the
first silently wrong.  Read-only recipes may share a materialization, which is what makes
witness probing affordable.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class EnvSession(Protocol):
    """A live environment. Always used as a context manager."""

    def run(self, recipe: Any, *, mutating: bool = False) -> Any:
        """Execute a recipe. If ``mutating``, re-materialize the environment first."""
        ...

    def rematerialize(self) -> None:
        """Reset to the pristine state described by the env spec."""
        ...

    def close(self) -> None: ...

    def __enter__(self) -> EnvSession: ...
    def __exit__(self, *exc) -> None: ...


@runtime_checkable
class Executor(Protocol):
    name: str

    def open(self, env_spec: dict) -> EnvSession:
        """Materialize ``env_spec`` and return a session over it."""
        ...

    def shutdown(self) -> None:
        """Release process-wide resources (containers, connections, caches)."""
        ...


class BaseSession:
    """Convenience base implementing the context-manager and mutation discipline."""

    def __init__(self, executor: Executor, env_spec: dict) -> None:
        self.executor = executor
        self.env_spec = env_spec
        self._closed = False

    # -- subclasses implement these two --------------------------------------
    def _materialize(self) -> None:
        raise NotImplementedError

    def _execute(self, recipe: Any) -> Any:
        raise NotImplementedError

    # -- shared discipline ----------------------------------------------------
    def rematerialize(self) -> None:
        self._materialize()

    def run(self, recipe: Any, *, mutating: bool = False) -> Any:
        if self._closed:
            raise RuntimeError("session already closed")
        if mutating:
            self.rematerialize()
        return self._execute(recipe)

    def close(self) -> None:
        self._closed = True

    def __enter__(self) -> BaseSession:
        self._materialize()
        return self

    def __exit__(self, *exc) -> None:
        self.close()
