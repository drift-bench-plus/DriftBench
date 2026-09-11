"""Export the admitted samples as the released dataset, with a datasheet.

One file per sample. `query` is the ENTIRE agent-visible surface; everything else is
scoring material and must never be shown to an agent under test.
"""
import json
import os
import sys
from collections import Counter
from pathlib import Path

WS = os.path.dirname(os.path.abspath(__file__))
OUT = Path(WS + "/artifacts/dataset")


def load(path):
    p = Path(WS + "/" + path)
    return json.loads(p.read_text()) if p.exists() else None


def export(domain: str, files: list[str]) -> dict:
    # Priority walk per key (files are listed newest-run first): the first row that is
    # either admitted or rejected_collision decides the key -- admitted ships,
    # rejected_collision kills the key (falling back to an older attempt would ship a
    # query whose flaw class failed its final audit). Ordinary generation-time rejections
    # keep the historical fallback semantics.
    decided, rows = {}, []
    for f in files:
        data = load(f)
        if not data:
            continue
        for r in data["results"]:
            key = (r["tree"], r["strategy"])
            if key in decided:
                continue
            if r["status"] == "admitted":
                decided[key] = True
                rows.append(r)
            elif r["status"] == "rejected_collision":
                decided[key] = False
    d = OUT / domain
    if d.exists():
        import shutil
        shutil.rmtree(d)     # a key rejected since the last export must not survive as a stale file
    d.mkdir(parents=True, exist_ok=True)
    for r in rows:
        sid = f"{r['tree']}__{r['strategy']}"
        mask = r.get("mask") or {}
        sig = r.get("signature") or {}
        sample = {
            "sample_id": sid,
            "domain": domain,
            "tree_id": r["tree"],
            # ---- the only field an agent may see ----
            "query": r["query"],
            # ---- scoring material ----
            "strategy_ids": [r["strategy"]],
            "mask": mask,
            "signature": sig,
            "executable_admission": bool(sig.get("executable", True)),
            "hidden_intent": {
                "conditions": r.get("root_conditions") or [],
                # ambiguous placements ARE hidden slots (see episodes_run_retail)
                "hidden_slots": sorted(set(mask.get("withheld") or [])
                                       | {m[0] for m in (mask.get("marked") or [])}
                                       | {s[0] for s in (mask.get("substituted") or [])}
                                       | {pl[0] for a in (mask.get("ambiguous") or [])
                                          for pl in (a.get("placements") or []) if pl}
                                       | {p["about"][0] for p in (mask.get("presupposed") or [])
                                          if p.get("about")}),
            },
            "verifier": {"kind": "adapter_accepts",
                         "adapter": f"tau2_{domain}",
                         "note": "acceptance re-executes the intent's own tau2 checks "
                                 "against the live environment"},
        }
        (d / f"{sid}.json").write_text(json.dumps(sample, indent=1))
    return {
        "domain": domain,
        "samples": len(rows),
        "by_strategy": dict(Counter(r["strategy"] for r in rows)),
        "non_executable_admissions": sum(
            1 for r in rows if not (r.get("signature") or {}).get("executable", True)),
        "trees_covered": len({r["tree"] for r in rows}),
    }


def main() -> None:
    stats = {
        # later runs of a strategy supersede earlier ones (see export(): first-wins on a
        # (tree, strategy) key, so the recovered-type runs are listed FIRST)
        "telecom": export("telecom", ["artifacts/telecom_fe_v2.json",
                                      "artifacts/telecom_vag_v2.json",
                                      "artifacts/telecom_fp_v2.json",
                                      "artifacts/telecom_irrel_v2.json",
                                      "artifacts/telecom_samples_v2.json"]),
        "retail": export("retail", ["artifacts/retail_vag_v2.json",
                                    "artifacts/retail_fp_v2.json",
                                    "artifacts/retail_irrel_v2.json",
                                    "artifacts/retail_samples_v2.json"]),
        # airline (2026-08-18): per-type recovery reruns, when they exist, go FIRST
        "airline": export("airline", ["artifacts/airline_v3.json"]),
    }
    (OUT / "datasheet.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
