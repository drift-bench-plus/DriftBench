"""Auto-mark tests by directory so nobody has to remember a decorator."""
import os
import sys
from pathlib import Path

import pytest

# Pin `import intent_graph` to the unified pipeline package, and the config home
# to this benchmark, no matter where pytest is invoked from.
_BENCH = Path(__file__).resolve().parents[1]
_SRC = Path(__file__).resolve().parents[3] / "pipeline" / "src"
sys.path.insert(0, str(_SRC))
os.environ.setdefault("IG_HOME", str(_BENCH))
for _m in [m for m in sys.modules if m == "intent_graph" or m.startswith("intent_graph.")]:
    del sys.modules[_m]
import intent_graph  # noqa: E402
assert Path(intent_graph.__file__).resolve().is_relative_to(_SRC), (
    f"intent_graph resolved to {intent_graph.__file__}, not the pipeline package")

_DIR_MARKS = {"unit": "unit", "data": "data", "docker": "docker", "battery": "battery"}


def pytest_collection_modifyitems(config, items):
    for item in items:
        for part in Path(item.fspath).parts:
            if part in _DIR_MARKS:
                item.add_marker(getattr(pytest.mark, _DIR_MARKS[part]))
                break
