"""Canonical serialization and content-hash identifiers.

Every id in the pipeline is a content hash so that regenerating a graph from the same
inputs yields byte-identical output (plan north-star #4, determinism).  All JSON we
write goes through :func:`canonical_dumps` so golden fixtures are stable by
construction rather than by luck.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

ID_LEN = 16


def _canonical(obj: Any) -> Any:
    """Recursively convert to a form with a single unambiguous JSON rendering."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            raise ValueError(f"non-finite float cannot be canonicalized: {obj!r}")
        # repr() round-trips exactly for IEEE doubles and is stable across platforms;
        # integral floats collapse so 4.0 and 4 do not hash differently.
        if obj == int(obj):
            return int(obj)
        return repr(obj)
    if isinstance(obj, bool | int | str) or obj is None:
        return obj
    if isinstance(obj, (list, tuple)):
        return [_canonical(x) for x in obj]
    if isinstance(obj, (set, frozenset)):
        return sorted((_canonical(x) for x in obj), key=lambda v: json.dumps(v, sort_keys=True))
    if isinstance(obj, dict):
        return {str(k): _canonical(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if hasattr(obj, "to_dict"):
        return _canonical(obj.to_dict())
    raise TypeError(f"cannot canonicalize {type(obj).__name__}: {obj!r}")


def canonical_dumps(obj: Any, *, indent: int | None = None) -> str:
    """Deterministic JSON text: sorted keys, normalized floats, stable set order."""
    return json.dumps(
        _canonical(obj),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":") if indent is None else (",", ": "),
        indent=indent,
    )


def content_hash(*parts: Any, length: int = ID_LEN) -> str:
    """sha256 over the canonical rendering of ``parts``, truncated to ``length``."""
    payload = canonical_dumps(list(parts))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def intent_id(adapter: str, adapter_version: str, env_id: str, base: Any, conditions: Any) -> str:
    """Identity of an intent = who made it, in which world, with which conditions.

    ``conditions`` is sorted before hashing, so two intents differing only in the
    order their conditions were discovered share an id.
    """
    return content_hash(adapter, adapter_version, env_id, base, sorted(conditions))


def graph_id(env_id: str, root_intent_id: str, child_ids: Any, config_hash: str) -> str:
    return content_hash(env_id, root_intent_id, sorted(child_ids), config_hash)


def config_hash(config: dict) -> str:
    """Hash of the knobs that can change generated output.

    Deliberately excludes ``paths`` and ``workers``: where the data lives and how many
    processes read it must not change a graph's identity.
    """
    relevant = {k: v for k, v in config.items() if k not in {"paths", "workers"}}
    return content_hash(relevant)
