#!/usr/bin/env python
"""Measure the ORACLE path length on the faithful WebShop site.

WHY. `runtime.max_turns` has never been derived from anything. The config claimed
">= worst-case oracle (10) + headroom" and said 16, while every campaign overrode it to 40
on the command line -- and both numbers described the OLD environment, where a purchase was
`search` + one synthesized `buy`, i.e. two actions. On the faithful site a purchase needs
search -> click item -> click each option -> Buy Now, and a REFUSED purchase sends the
shopper back to the store to navigate again. So the budget has to be re-derived.

WHAT THIS MEASURES. For each sampled graph we know the target asin and the goal options.
The oracle does the minimum an agent could possibly do: one search using the product's own
cluster words, page through results until the target appears, click it, click each required
option, then Buy Now. That is the FLOOR -- a real agent must also read, compare and
sometimes back out, and must pay for the conversation on top.

The number to take from this is not the mean but the tail: max_turns has to sit well above
the 95th percentile of the floor, or the cap decides episodes instead of the economy.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import urllib.request
from pathlib import Path

_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def post(base, path, **body):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with _OPENER.open(req, timeout=180) as r:
        return json.loads(r.read())


def oracle_episode(base, sid, instruction, targets, options, cluster="", source="", max_pages=6):
    """Return (n_actions, found, bought_asin) for the minimum navigation path.

    THE ORACLE SEARCHES THE CLUSTER, NOT THE QUERY. The episode's query is deliberately
    PERTURBED -- it opens with hedging ("Since you probably don't have...") and may misstate
    requirements, so its leading words are not search terms. A first version searched them
    and reached 0/30 targets, which measured the perturbation, not the environment. The
    oracle is allowed to know the target's own product cluster (WebShop's own goal `query`
    field), which is the search a shopper who knew what they wanted would type.
    """
    def P(path, **kw):
        return post(base, path, **kw)
    n = 0
    P("/reset", session=sid, instruction=instruction)
    # MEASURED 2026-08-20: a bare cluster string surfaces an acceptable product 0% of the
    # time, the source instruction 83%. The oracle searches the way a shopper who knows what
    # they want would speak, which is what the reachability gate admits samples on.
    terms = (source or cluster or instruction)[:220]
    out = P("/step", session=sid, action=f"search[{terms}]")
    n += 1
    want = {str(t).lower() for t in targets}   # ANY acceptable product will do

    for _ in range(max_pages):
        clickables = [c.lower() for c in out["actions"]["clickables"]]
        hit = next((c for c in clickables if c in want), None)
        if hit:
            out = P("/step", session=sid, action=f"click[{hit}]")
            n += 1
            # select every option the goal names, when the item page offers it
            avail = [c.lower() for c in out["actions"]["clickables"]]
            for val in (options or []):
                v = str(val).lower()
                if v in avail:
                    out = P("/step", session=sid, action=f"click[{val}]")
                    n += 1
            out = P("/step", session=sid, action="click[Buy Now]")
            n += 1
            m = re.search(r"asin\s*\[SEP\]\s*([A-Za-z0-9]+)", str(out.get("obs") or ""))
            return n, True, (m.group(1) if m else None)
        if "next >" not in clickables:
            break
        out = P("/step", session=sid, action="click[Next >]")
        n += 1
    return n, False, None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:3020")
    ap.add_argument("--samples", default="artifacts/samples_v2/samples/webshop")
    ap.add_argument("--n", type=int, default=40)
    a = ap.parse_args()

    files = sorted(Path(a.samples).glob("*.json"))[: a.n]
    lens, found, bought_right = [], 0, 0
    for i, f in enumerate(files):
        s = json.loads(f.read_text())
        gt = s.get("ground_truth") or {}
        hidden = s.get("hidden_intent") or {}
        conds = hidden.get("conditions") or []
        # ANY member of the purchaseset is acceptable -- the ground truth is a SET of
        # (asin, options) that all satisfy the intent (median 34 products, up to 4049).
        # Requiring one specific asin measured the wrong thing entirely.
        targets = []
        for v in (gt.get("value") or []):
            try:
                targets.append(json.loads(v)[0])
            except Exception:
                pass
        base_asin = (s.get("base") or {}).get("asin")
        if base_asin:
            targets.append(base_asin)
        if not targets:
            continue
        target = targets[0]
        options = [v for (slot, _op, v) in conds if str(slot).startswith("option:")]
        n, ok, got = oracle_episode(a.base, f"oracle-{i}", s.get("query") or "", targets,
                                    options, cluster=str(s.get("cluster") or ""),
                                    source=str(s.get("source_instruction") or ""))
        lens.append(n)
        found += int(ok)
        bought_right += int(got is not None and got.lower() in {str(t).lower() for t in targets})

    if not lens:
        print("no samples measured")
        return 2
    lens.sort()
    p = lambda q: lens[min(len(lens) - 1, int(q * len(lens)))]  # noqa: E731
    print(f"episodes measured : {len(lens)}")
    print(f"target reachable  : {found}/{len(lens)} ({100*found/len(lens):.0f}%)")
    print(f"bought the target : {bought_right}/{len(lens)}")
    print(f"ORACLE ACTIONS    : mean={statistics.mean(lens):.1f} median={p(0.5)} "
          f"p90={p(0.9)} p95={p(0.95)} max={lens[-1]}")
    print()
    print("A real arm additionally pays for: reading Description/Features, comparing "
          "candidates, backing out of wrong items, the ASK turns, and -- after a refused "
          "purchase -- a COMPLETE re-navigation. Budget well above this floor.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
