"""Cross-layer error types.

Lives outside runtime/ and executors/ so both can import it without cycles.
"""


class EnvTransportError(RuntimeError):
    """The environment server gave no answer after the full wait-and-retry budget.

    Raised INSTEAD of fabricating an "error: ..." observation (ruling 2026-08-21: a failed
    query just waits and retries; if the server is truly gone the episode must die loudly
    as an ERROR husk and be refilled, never be scored on fake pages). The episode runner
    deliberately lets this propagate past its per-turn catch-all."""
