"""Close the pragmatic-extras contract loosening (flagged 2026-08-13).

`irrelevant_information` extras on tau2 cannot compile into the closed condition
vocabulary, so they were admitted with verdict `ok_pragmatic_extra` and no executable
check. The risk that loosening left open: an "extra" could accidentally assert a REAL
domain condition (a telecom fault symptom, a retail shop request) that is not part of the
true intent -- then the query is not noise, it is a factual/scope perturbation mislabeled
as noise, and the fault-type taxonomy is contaminated.

This audit runs the missing check, per sample, with the pipeline's dual cross-family
extractor convention (doubao ROLE_SELECT + deepseek ROLE_JUDGE), fail-closed: if EITHER
extractor reads a domain-real assertion beyond the true conditions, the sample is
rejected. LLM use here is perturbation admission, not tree construction, so the code-only
rule does not apply.
"""
import json
import os
import sys
import threading
from collections import Counter
from pathlib import Path

WS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, WS + "/../../pipeline/src")
os.environ["TAU2_SRC"] = WS + "/tau2-bench/src"
os.environ["TAU2_DATA"] = WS + "/tau2-bench/data/tau2"
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
_envf = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env")
if "ARK_API_KEY" not in os.environ and os.path.exists(_envf):
    os.environ["ARK_API_KEY"] = [l.split("=", 1)[1].strip() for l in open(_envf)
                                 if l.startswith("ARK_API_KEY=")][0]
from loguru import logger as _l          # noqa: E402
_l.remove()

from intent_tree import storage                                   # noqa: E402
from intent_tree.adapters.tau2_retail import Tau2RetailAdapter    # noqa: E402
from intent_tree.adapters.tau2_telecom import Tau2TelecomAdapter  # noqa: E402
from intent_tree.runtime.genverify import _first_json             # noqa: E402
from intent_tree.runtime.llm import LLMClient, ROLE_JUDGE, ROLE_SELECT  # noqa: E402
from perturb_run import LLM_CFG                                   # noqa: E402

SCHEMA = ('Reply with ONLY a JSON object: {"collides": true or false, '
          '"evidence": "<the colliding phrase, or empty>"}')


def telecom_prompt(query: str, true_problems: list[str], vocabulary: list[str]) -> str:
    return (
        "A customer message to phone support is below. The customer's REAL problems are "
        "exactly these:\n  - " + "\n  - ".join(true_problems) +
        "\n\nThe message deliberately also contains irrelevant chatter (purchase stories, "
        "accessories, compliments, weather...). That is fine. Your ONLY job: check whether "
        "the message ALSO claims, implies, or asks about any OTHER phone/service problem "
        "beyond the real ones -- for example any of these known problem kinds:\n  - "
        + "\n  - ".join(vocabulary) +
        "\n\nMentioning a device feature or a past purchase is NOT a problem claim. "
        "Saying something is broken, not working, or asking to fix/check it IS.\n\n"
        "Message:\n\"\"\"\n" + query + "\n\"\"\"\n\n" + SCHEMA)


def retail_prompt(query: str, true_requests: list[str]) -> str:
    return (
        "A customer message to an online shop is below. The customer's REAL requests are "
        "exactly these:\n  - " + "\n  - ".join(true_requests) +
        "\n\nThe message deliberately also contains irrelevant chatter (certifications, "
        "compliments, stories...). That is fine. Your ONLY job: check whether the message "
        "ALSO asks for or asserts any OTHER shop operation beyond the real ones -- another "
        "cancellation, return, exchange, modification, address or payment change, or a "
        "claim about a different order/item that reads as a request.\n\n"
        "Merely mentioning a product, a certification, or a past purchase is NOT a "
        "request. Asking the shop to DO something with it IS.\n\n"
        "Message:\n\"\"\"\n" + query + "\n\"\"\"\n\n" + SCHEMA)


def describe_true(dom: str, adapter, conds) -> list[str]:
    if dom == "telecom":
        return [adapter.fault_description(str(v)) for _s, _o, v in conds]
    by_action: dict[str, list[str]] = {}
    for s, _o, v in conds:
        body = str(s)
        if body.startswith("arg:"):
            _a, idx, tool, key = body.split(":", 3)
            by_action.setdefault(f"{idx}:{tool}", []).append(f"{key}={v}")
        elif body.startswith("also:"):
            _a, idx, tool = body.split(":", 2)
            by_action.setdefault(f"{idx}:{tool}", []).append(f"args={v}")
    return [f"{k.split(':', 1)[1]} with {', '.join(vv)}" for k, vv in sorted(by_action.items())]


_local = threading.local()


def check(job):
    dom, r, prompt = job
    llm = getattr(_local, "llm", None)
    if llm is None:
        llm = LLMClient({"llm": LLM_CFG})
        _local.llm = llm
    verdicts = {}
    for tag, role in (("e1", ROLE_SELECT), ("e2", ROLE_JUDGE)):
        try:
            obj = _first_json(llm.complete(prompt, role=role)) or {}
        except Exception as e:
            obj = {"collides": True, "evidence": f"extractor_error:{type(e).__name__}"}
        verdicts[tag] = {"collides": bool(obj.get("collides")),
                         "evidence": str(obj.get("evidence") or "")[:200]}
    collides = verdicts["e1"]["collides"] or verdicts["e2"]["collides"]
    return {"domain": dom, "tree": r["tree"], "strategy": r["strategy"],
            "source": r.get("_source"), "collides": collides, "verdicts": verdicts}


def main() -> None:
    tel = Tau2TelecomAdapter()
    ret = Tau2RetailAdapter()
    fault_vocab = sorted({str(v)
                          for t in storage.iter_trees(Path(WS + "/artifacts/trees/tau2_telecom"))
                          for n in (t.root, *t.children)
                          for _s, _o, v in n.conditions})
    vocab_desc = [tel.fault_description(f) for f in fault_vocab]

    # EXACTLY the rows the export ships: same file lists, same first-wins key rule as
    # export_dataset.py (a strategy rerun in an earlier-listed file supersedes later ones,
    # so auditing a single artifact file can hit superseded rows -- measured: retail's
    # exported irrelevant_information rows come from retail_fp_v2.json, not the irrel file)
    FILES = {"telecom": ["artifacts/telecom_fe_v2.json", "artifacts/telecom_vag_v2.json",
                         "artifacts/telecom_fp_v2.json", "artifacts/telecom_irrel_v2.json",
                         "artifacts/telecom_samples_v2.json"],
             "retail": ["artifacts/retail_vag_v2.json", "artifacts/retail_fp_v2.json",
                        "artifacts/retail_irrel_v2.json", "artifacts/retail_samples_v2.json"]}
    # ROW-exhaustive: every admitted pragmatic row in every file, winners AND superseded.
    # Export decides winners separately; auditing all rows means no fallback can ever
    # ship a query this audit has not seen. (Previously-rejected rows stay rejected --
    # their status is no longer "admitted".)
    jobs = []
    for dom, flist in FILES.items():
        for f in flist:
            p = Path(WS + "/" + f)
            if not p.exists():
                continue
            d = json.loads(p.read_text())
            for r in d["results"]:
                if r["status"] != "admitted":
                    continue
                if (r.get("signature") or {}).get("verdict") != "ok_pragmatic_extra":
                    continue
                conds = r.get("root_conditions") or []
                if dom == "telecom":
                    pr = telecom_prompt(r["query"], describe_true(dom, tel, conds), vocab_desc)
                else:
                    pr = retail_prompt(r["query"], describe_true(dom, ret, conds))
                jobs.append((dom, dict(r, _source=f), pr))
    print(f"{len(jobs)} pragmatic-extra samples to audit", flush=True)

    import concurrent.futures as cf
    rows = []
    with cf.ThreadPoolExecutor(max_workers=16) as ex:
        for i, out in enumerate(ex.map(check, jobs), 1):
            rows.append(out)
            if i % 40 == 0:
                print(f"  {i}/{len(jobs)} | collisions so far "
                      f"{sum(r['collides'] for r in rows)}", flush=True)

    summary = {"audited": len(rows),
               "collisions": sum(r["collides"] for r in rows),
               "by_domain": {d: {"audited": sum(1 for r in rows if r["domain"] == d),
                                 "collisions": sum(1 for r in rows
                                                   if r["domain"] == d and r["collides"])}
                             for d in ("telecom", "retail")},
               "only_one_extractor": sum(1 for r in rows if r["collides"] and not
                                         (r["verdicts"]["e1"]["collides"]
                                          and r["verdicts"]["e2"]["collides"]))}
    Path(WS + "/artifacts/pragmatic_collision_audit.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=1))
    print(json.dumps(summary, indent=1))
    ex_rows = [r for r in rows if r["collides"]][:5]
    for r in ex_rows:
        ev = r["verdicts"]["e1"]["evidence"] or r["verdicts"]["e2"]["evidence"]
        print(f"  COLLIDES {r['domain']} {r['tree'][:8]}: {ev[:120]}")


if __name__ == "__main__":
    main()
