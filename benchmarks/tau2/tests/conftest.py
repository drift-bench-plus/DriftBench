"""Pin `import intent_graph` to the unified pipeline package.

Both packages are called `intent_graph`, and the perturbation copy is pip-installed, so
without this the suite would silently test the wrong tree -- which is exactly the kind of
invisible mismatch these tests exist to catch.
"""
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[3] / "pipeline" / "src"
import os
os.environ.setdefault("IG_HOME", str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(SRC))

for name in [m for m in sys.modules if m == "intent_graph" or m.startswith("intent_graph.")]:
    del sys.modules[name]

import intent_graph  # noqa: E402
assert Path(intent_graph.__file__).resolve().is_relative_to(SRC), (
    f"intent_graph resolved to {intent_graph.__file__}, not this fork's src")


# Auto-mark by directory, matching the perturbation suite so the same test files behave
# identically in both forks.
from pathlib import Path  # noqa: E402

import pytest  # noqa: E402

_DIR_MARKS = {"unit": "unit", "data": "data", "docker": "docker", "battery": "battery"}


def pytest_collection_modifyitems(config, items):
    for item in items:
        for part in Path(item.fspath).parts:
            if part in _DIR_MARKS:
                item.add_marker(getattr(pytest.mark, _DIR_MARKS[part]))
                break
