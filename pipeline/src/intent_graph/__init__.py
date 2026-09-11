"""Deprecated alias for `intent_graph`.

The package was renamed `intent_tree` -> `intent_graph` on 2026-08-19 (`Tree` -> `Graph`,
`tree_id` -> `graph_id`, `iter_trees` -> `iter_graphs`). Thirty-three one-off driver
scripts under `tau2/` still import the old name, and those scripts are what BUILD the
released datasets -- so they cannot simply be left broken, and rewriting twenty-four
untested scripts to chase a rename is the larger risk. The data is unaffected either way:
`storage.LEGACY_DIR` still reads the "trees" directory and every id is byte-identical.

WHY NOT A META-PATH FINDER. The obvious trick -- a finder mapping `intent_tree.*` to
`intent_graph.*` -- was tried and rejected: it produced a SECOND module object for each
name. Two copies of `models` means two `Graph` classes, so `isinstance(g, Tree)` silently
returns False for an object built by the other name. That is a far worse bug than the one
being fixed. This file instead binds the SAME module objects under both names, and a test
asserts they are identical.

New code should import `intent_graph`.
"""
import importlib
import sys

_REAL = "intent_graph"

# Every `intent_tree.*` module the tau2 drivers import, found by parsing all 33 of them.
# Parents come first: `intent_tree.runtime.llm` needs `intent_tree.runtime` bound too.
_ALIASED = (
    "models", "engine", "storage", "classify", "gate", "ids",
    "adapters", "adapters.tau2_airline", "adapters.tau2_retail", "adapters.tau2_telecom",
    "executors", "executors.tau2_inproc",
    "runtime", "runtime.agents", "runtime.episode", "runtime.genverify",
    "runtime.llm", "runtime.persona", "runtime.traversal", "runtime.strategies",
)


def _bind(sub: str):
    """Bind one real module under the alias name, as the SAME object."""
    try:
        mod = importlib.import_module(f"{_REAL}.{sub}")
    except Exception:
        return None                      # optional dep (a tau2 adapter needs tau2-bench)
    sys.modules[f"{__name__}.{sub}"] = mod
    head, _, _tail = sub.partition(".")
    if "." not in sub:
        globals()[head] = mod
    return mod


for _sub in _ALIASED:
    _bind(_sub)


def __getattr__(name):
    """Anything not pre-bound resolves on demand, still as the same object."""
    mod = _bind(name)
    if mod is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return mod


# --- old call names, so the drivers' `storage.iter_trees` / `models.Tree` keep working ---
_storage = sys.modules.get(f"{__name__}.storage")
_models = sys.modules.get(f"{__name__}.models")
if _storage is not None:
    for _old, _new in (("iter_trees", "iter_graphs"), ("write_tree", "write_graph"),
                       ("read_tree", "read_graph"), ("tree_path", "graph_path")):
        if not hasattr(_storage, _old) and hasattr(_storage, _new):
            setattr(_storage, _old, getattr(_storage, _new))
if _models is not None:
    if not hasattr(_models, "Tree") and hasattr(_models, "Graph"):
        _models.Tree = _models.Graph
    if hasattr(_models, "Graph") and not hasattr(_models.Graph, "tree_id"):
        try:
            _models.Graph.tree_id = property(lambda self: self.graph_id)
        except (AttributeError, TypeError):
            pass
