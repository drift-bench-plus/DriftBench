"""R-dimension judges over the v9 rational cells (GRIP v2).

Identification only: the campaign is single-persona, so Consistency is undefined by
construction (no different-persona pairs exist) and run_judges now skips it.
judge_a rides ROLE_JUDGE (agent-side -> deepseek pool) and judge_b ROLE_SELECT
(user-side -> doubao pool): cross-family pairing from this session's own allocation,
never the 2-1-* models that belong to the WebShop/Airline sessions.
"""
import json
import os
import sys
from pathlib import Path

WS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(WS, "..", "..", "pipeline", "src"))
sys.path.insert(0, WS)

import perturb_run  # noqa: E402  (env, key, LLM_CFG with allocation asserts)
from intent_graph.runtime.llm import LLMClient  # noqa: E402
from intent_graph.runtime.judges import run_judges  # noqa: E402

CELLS = ["B0_s1", "A0_s1", "A2_s1", "A3_s1", "A2_s2", "A2r_s2", "A3_s2",
         "A2_s3", "A3_s3"]


def main() -> None:
    cfg = dict(perturb_run.LLM_CFG)
    cfg["cache_dir"] = os.path.join(WS, "artifacts", "llm_cache", "judges_v9")
    llm = LLMClient({"llm": cfg})
    for c in CELLS:
        d = Path(WS) / "artifacts" / "rational_v9" / c
        out = d / "judgments"
        if (out / "judgments.json").exists():
            print(f"{c}: judgments exist, skipping", flush=True)
            continue
        if not d.exists():
            print(f"{c}: no cell dir", flush=True)
            continue
        r = run_judges(d, out, llm, per_persona=20, n_pairs=100)
        print(f"{c}: {json.dumps(r.get('identification', {}))[:160]}", flush=True)
    print("JUDGES_DONE", flush=True)


if __name__ == "__main__":
    main()
