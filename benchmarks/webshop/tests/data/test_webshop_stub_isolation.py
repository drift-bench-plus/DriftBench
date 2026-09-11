"""Importing our WebShop code must never pull in the real pyserini.

The executor stubs pyserini because WebShop's engine imports Lucene (hence a JVM) at module
level purely to build the SEARCH index, which ground-truth computation never touches. If a
real pyserini ever became importable, the stub would silently stop being exercised and the
executor's behaviour would depend on whether a JVM happened to be installed.
"""

import subprocess
import sys

import pytest

pytestmark = pytest.mark.data

PROBE = """
import sys
from intent_graph.cli import build, load_config
adapter, executor = build("webshop", load_config())
executor._load()                                  # installs the stubs, loads goal.py
import json
real = []
for name in ("pyserini", "pyserini.search", "pyserini.search.lucene"):
    mod = sys.modules.get(name)
    if mod is None:
        continue
    # a stub has no __file__; the real package is installed on disk
    if getattr(mod, "__file__", None):
        real.append((name, mod.__file__))
lucene = sys.modules.get("pyserini.search.lucene")
print(json.dumps({
    "real": real,
    "has_searcher": hasattr(lucene, "LuceneSearcher") if lucene else False,
    "jvm_loaded": any(k.startswith(("jnius", "pyjnius", "jpype")) for k in sys.modules),
}))
"""


def _probe():
    out = subprocess.run([sys.executable, "-c", PROBE], capture_output=True, text=True,
                         timeout=900)
    assert out.returncode == 0, out.stderr[-3000:]
    import json
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_no_real_pyserini_is_imported():
    res = _probe()
    assert res["real"] == [], f"real pyserini modules imported: {res['real']}"


def test_stub_searcher_is_present_but_refuses_construction():
    res = _probe()
    assert res["has_searcher"], "the stub must provide LuceneSearcher or engine.py fails"


def test_no_jvm_bridge_is_loaded():
    """A JVM bridge in sys.modules means something imported the real search stack."""
    assert _probe()["jvm_loaded"] is False


def test_constructing_the_stub_searcher_raises():
    from intent_graph.executors.webshop_inproc import _install_stubs
    _install_stubs()
    from pyserini.search.lucene import LuceneSearcher
    with pytest.raises(RuntimeError, match="ground truth does not use it"):
        LuceneSearcher("whatever")
