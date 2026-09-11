"""Compatibility shim so the tau2 driver scripts keep working after the package rename.

On 2026-08-19 the shared pipeline package was renamed `intent_tree` -> `intent_graph`
(with `Tree` -> `Graph`, `tree_id` -> `graph_id`, `iter_trees` -> `iter_graphs`, ...).
The dozens of one-off driver scripts in this directory still say `intent_tree`, and the
DATA is unaffected -- `storage.LEGACY_DIR` keeps reading the "trees" directory and every
id is byte-identical -- so aliasing is the honest fix, not a mass rewrite of scripts that
are not under test.

An earlier version of this file tried `from intent_tree import ...`, which cannot work:
the package it names is gone, so every script importing this shim failed at import time.
This version registers `intent_tree` as an alias of `intent_graph` on the import system,
so `from intent_tree.adapters.tau2_airline import ...` resolves too.

Import it once, after the sys.path setup and before any intent_tree use.
"""
import importlib
import importlib.abc
import importlib.machinery
import sys

_ALIAS, _REAL = "intent_tree", "intent_graph"


class _AliasLoader(importlib.abc.Loader):
    def __init__(self, real: str) -> None:
        self.real = real

    def create_module(self, spec):
        return importlib.import_module(self.real)

    def exec_module(self, module) -> None:      # already executed under its real name
        pass


class _AliasFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == _ALIAS or fullname.startswith(_ALIAS + "."):
            return importlib.machinery.ModuleSpec(
                fullname, _AliasLoader(_REAL + fullname[len(_ALIAS):]))
        return None


if not any(isinstance(f, _AliasFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _AliasFinder())

from intent_tree import models, storage  # noqa: E402  (resolves via the alias above)

# old call names -> new ones
for _old, _new in (("iter_trees", "iter_graphs"), ("write_tree", "write_graph"),
                   ("read_tree", "read_graph"), ("tree_path", "graph_path")):
    if not hasattr(storage, _old) and hasattr(storage, _new):
        setattr(storage, _old, getattr(storage, _new))

if not hasattr(models, "Tree") and hasattr(models, "Graph"):
    models.Tree = models.Graph
if hasattr(models, "Graph") and not hasattr(models.Graph, "tree_id"):
    try:
        models.Graph.tree_id = property(lambda self: self.graph_id)
    except (AttributeError, TypeError):
        pass
