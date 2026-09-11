"""Freeze one WebShop cluster's generated graphs as a byte-exact fixture.

Run this ONLY when a graph-content change is intentional, and commit the diff alongside the
code change so the change is visible in review.

    python tools/make_golden.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "pipeline" / "src"))

from intent_graph.golden import GOLDEN_CLUSTER, golden_path, regenerate  # noqa: E402
from intent_graph.ids import canonical_dumps  # noqa: E402


def main() -> int:
    payload = regenerate()
    golden_path().write_text(canonical_dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {golden_path()}")
    print(f"cluster={GOLDEN_CLUSTER!r}  graphs={len(payload['graphs'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
