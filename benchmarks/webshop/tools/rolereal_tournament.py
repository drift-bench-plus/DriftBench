#!/usr/bin/env python
"""Judge tournament (ruling 2026-08-28): evaluate 5 top-tier candidate judges on a
FIXED calibration slice of the role-realism tasks; the two best become the judge
pair for the full re-run.

Every candidate sees the IDENTICAL task set (fixed rng): 100 identification items
(20/persona) + 100 consistency pairs (balanced) drawn from the WebShop A0 r1 pool.
Scores: identification accuracy (dummy 20) and consistency balanced accuracy
(dummy 50, with the degenerate-marginal validity gate). Selection = rank by
ident_acc + cons_bal_acc among non-degenerate candidates.

Candidates (all with thinking off via the runtime's per-family switches):
  ark:     doubao-seed-2-1-pro-260628, deepseek-v4-pro-260425
  gateway: gateway/gpt-5.5-2026-04-24, gateway/openai_qwen3.7-max, gateway/glm-4.7
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1] / "pipeline" / "src"))

GATEWAY = "https://YOUR-GATEWAY.example/v1"
# NOTE: cross-provider gateway access may require local TLS trust configuration;
# the tournament can also run on the primary provider's top tiers alone.
CANDIDATES = [
    "doubao-seed-2-1-pro-260628",
    "deepseek-v4-pro-260425",
    "doubao-seed-2-0-pro-260215",
    "doubao-seed-2-1-turbo-260628",
]


def make_judge_llm(model: str):
    from intent_graph.runtime.cli import load_runtime, make_llm
    cfg = load_runtime(None)
    cfg["llm"]["agent_model"] = model
    if model.startswith("gateway/"):
        cfg["llm"]["agent_base_url"] = GATEWAY
        cfg["llm"]["agent_protocol"] = "chat"
        cfg["llm"]["agent_api_key_env"] = "GPT_GATEWAY_KEY"
    cfg["llm"].pop("agent_model_fallback", None)
    return make_llm(cfg, fake=False)


def build_tasks(run: Path, n_ident: int, n_pairs: int, seed: int):
    import glob
    from intent_graph.runtime.judges import _user_turns, _persona_cards
    rng = random.Random(seed)
    by_persona: dict[str, list] = {}
    for f in sorted(run.glob("*.json")):
        try:
            t = json.loads(f.read_text())
        except Exception:
            continue
        turns = _user_turns(t)
        if not turns:
            continue
        pid = t["header"].get("persona")
        by_persona.setdefault(pid, []).append(
            {"file": f.name, "persona": pid, "turns": turns})
    ident = []
    per = n_ident // len(by_persona)
    for pid in sorted(by_persona):
        pool = sorted(by_persona[pid], key=lambda x: x["file"])
        rng.shuffle(pool)
        for item in pool[:per]:
            cards, mapping = _persona_cards(rng)
            ident.append((item, cards, mapping))
    pids = sorted(by_persona)
    pairs = []
    for _ in range(n_pairs):
        same = rng.random() < 0.5
        if same:
            pid = rng.choice(pids)
            a, b = rng.sample(by_persona[pid], 2)
        else:
            pa, pb = rng.sample(pids, 2)
            a = rng.choice(by_persona[pa])
            b = rng.choice(by_persona[pb])
        pairs.append((a, b, same))
    return ident, pairs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="experiments/rolereal/ws_A0_r1")
    ap.add_argument("--n-ident", type=int, default=100)
    ap.add_argument("--n-pairs", type=int, default=100)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--out", default="experiments/rolereal/tournament.json")
    args = ap.parse_args()

    from intent_graph.runtime.judges import (identification_prompt,
                                             consistency_prompt, _parse_letter,
                                             _parse_same)
    from intent_graph.runtime.llm import ROLE_JUDGE

    run = ROOT / args.run
    ident, pairs = build_tasks(run, args.n_ident, args.n_pairs, args.seed)
    print(f"calibration set: {len(ident)} ident + {len(pairs)} pairs", flush=True)

    results = {}
    for model in CANDIDATES:
        llm = make_judge_llm(model)

        def ask_ident(task):
            item, cards, mapping = task
            try:
                raw = llm.complete(identification_prompt(cards, item["turns"][:8]),
                                   role=ROLE_JUDGE)
                letter = _parse_letter(raw)
                return mapping.get(letter) == item["persona"]
            except Exception:
                return None

        def ask_pair(task):
            a, b, same = task
            try:
                raw = llm.complete(consistency_prompt(a["turns"][:6], b["turns"][:6]),
                                   role=ROLE_JUDGE)
                return _parse_same(raw), same
            except Exception:
                return None, same

        with ThreadPoolExecutor(max_workers=args.threads) as ex:
            iv = list(ex.map(ask_ident, ident))
            pv = list(ex.map(ask_pair, pairs))
        i_ok = [v for v in iv if v is not None]
        ident_acc = sum(i_ok) / len(i_ok) if i_ok else 0.0
        same_t = [(g, s) for g, s in pv if g is not None]
        s_pairs = [(g, s) for g, s in same_t if s]
        d_pairs = [(g, s) for g, s in same_t if not s]
        acc_s = sum(1 for g, _ in s_pairs if g) / len(s_pairs) if s_pairs else 0.0
        acc_d = sum(1 for g, _ in d_pairs if not g) / len(d_pairs) if d_pairs else 0.0
        bal = (acc_s + acc_d) / 2
        same_rate = (sum(1 for g, _ in same_t if g) / len(same_t)) if same_t else 0.0
        valid = 0.10 <= same_rate <= 0.90
        results[model] = {
            "ident_acc": round(ident_acc, 4), "ident_n": len(i_ok),
            "ident_errors": len(iv) - len(i_ok),
            "cons_bal_acc": round(bal, 4), "cons_n": len(same_t),
            "cons_errors": len(pv) - len(same_t),
            "same_rate": round(same_rate, 4), "valid": valid,
            "combined": round(ident_acc + (bal if valid else 0.5), 4),
        }
        print(f"{model}: {json.dumps(results[model])}", flush=True)

    ranked = sorted(results.items(), key=lambda kv: -kv[1]["combined"])
    top2 = [m for m, _ in ranked[:2]]
    out = {"results": results, "ranked": [m for m, _ in ranked], "top2": top2}
    (ROOT / args.out).write_text(json.dumps(out, indent=1))
    print("\nRANKING:")
    for m, r in ranked:
        print(f"  {m}: combined={r['combined']} ident={r['ident_acc']} "
              f"cons={r['cons_bal_acc']} valid={r['valid']}")
    print("TOP2:", top2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
