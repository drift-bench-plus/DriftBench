"""R -- Role realism: judge-based metrics over STORED transcripts (GRIP v2).

Identification: two independent LLM judges see the user's turns only and pick which of
the five personas produced them (five-way; chance 0.2). Consistency: the judges see two
transcripts' user turns and answer whether the same persona produced both.

Everything runs offline over trajectory files, is cached by the normal LLM cache, and
every judgment is written to disk -- future metrics recompute from the stored record.
"""

from __future__ import annotations

import json
import random
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .llm import ROLE_JUDGE, ROLE_SELECT
from .persona import PERSONAS

LETTERS = "ABCDE"


def _user_turns(traj: dict) -> list[str]:
    out = []
    for rec in traj.get("turns", []):
        reply = rec.get("reply")
        if reply:
            out.append(str(reply))
        acc = rec.get("acceptance")
        if acc and not acc.get("ok"):
            reason = str(acc.get("reason") or "")
            if reason and not reason.startswith("reward="):
                out.append(reason)
    return out


def _persona_cards(rng: random.Random) -> tuple[str, dict]:
    """The five persona descriptions, shuffled and lettered, plus letter->id mapping."""
    ids = sorted(PERSONAS)
    rng.shuffle(ids)
    lines, mapping = [], {}
    for letter, pid in zip(LETTERS, ids, strict=False):
        bio = PERSONAS[pid].bio.strip().splitlines()[0]
        lines.append(f"{letter}. {bio}")
        mapping[letter] = pid
    return "\n".join(lines), mapping


def identification_prompt(cards: str, turns: list[str]) -> str:
    quoted = "\n".join(f"- {t}" for t in turns)
    return (
        "Below are messages written by ONE simulated shopping customer during a "
        "conversation with an assistant (the assistant's messages are omitted).\n\n"
        f"Customer messages:\n{quoted}\n\n"
        "Five customer personalities:\n"
        f"{cards}\n\n"
        "Which personality wrote these messages? Answer with the single letter only."
    )


def consistency_prompt(turns_a: list[str], turns_b: list[str]) -> str:
    qa = "\n".join(f"- {t}" for t in turns_a)
    qb = "\n".join(f"- {t}" for t in turns_b)
    return (
        "Below are two sets of messages, each written by a shopping customer in a "
        "different conversation. The two conversations concern DIFFERENT products and "
        "tasks -- that is expected and tells you nothing. Judge ONLY the customer's "
        "manner: how fully they answer, how much they volunteer unprompted, how they "
        "phrase refusals, and how they react when pressed or disappointed.\n\n"
        f"Customer 1:\n{qa}\n\nCustomer 2:\n{qb}\n\n"
        "Based on manner alone, were these written by the SAME customer personality or "
        "by two DIFFERENT personalities? Answer with the single word SAME or DIFFERENT."
    )


def _parse_letter(raw: str) -> str | None:
    m = re.search(r"\b([A-E])\b", str(raw).upper())
    return m.group(1) if m else None


def _parse_same(raw: str) -> bool | None:
    up = str(raw).upper()
    if "DIFFERENT" in up:
        return False
    if "SAME" in up:
        return True
    return None


def run_judges(traj_dir: Path, out_dir: Path, llm, *, per_persona: int = 20,
               n_pairs: int = 100, min_turns: int = 2, seed: int = 20260811,
               threads: int = 12) -> dict:
    """Judge a run's transcripts; write every judgment; return the R summary."""
    rng = random.Random(seed)
    by_persona: dict[str, list[dict]] = {}
    for f in sorted(Path(traj_dir).glob("*.json")):
        t = json.loads(f.read_text(encoding="utf-8"))
        turns = _user_turns(t)
        if len(turns) < min_turns or t.get("outcome") == "ERROR":
            continue
        pid = t["header"].get("persona")
        by_persona.setdefault(pid, []).append(
            {"file": f.name, "persona": pid, "turns": turns})

    out_dir.mkdir(parents=True, exist_ok=True)
    judgments = {"identification": [], "consistency": []}

    # ---- Identification (parallel: judge calls are network-bound) ---------
    ident_tasks = []
    for pid in sorted(by_persona):
        pool = sorted(by_persona[pid], key=lambda x: x["file"])
        rng.shuffle(pool)
        for item in pool[:per_persona]:
            cards, mapping = _persona_cards(rng)
            ident_tasks.append((item, cards, mapping))

    def _judge_ident(task):
        item, cards, mapping = task
        prompt = identification_prompt(cards, item["turns"][:8])
        votes = {}
        for name, role in (("judge_a", ROLE_JUDGE), ("judge_b", ROLE_SELECT)):
            raw = llm.complete(prompt, role=role)
            letter = _parse_letter(raw)
            votes[name] = {"raw": str(raw)[:200],
                           "guess": mapping.get(letter) if letter else None}
        return {"file": item["file"], "persona": item["persona"],
                "mapping": mapping, **votes}

    with ThreadPoolExecutor(max_workers=threads) as ex:
        judgments["identification"] = list(ex.map(_judge_ident, ident_tasks))

    # ---- Consistency ------------------------------------------------------
    # Needs BOTH pair kinds to be constructible: a single-persona run has no
    # different-persona pairs, so the metric is undefined there (not zero) -- and
    # rng.sample(pids, 2) would raise. Skip; summarize_judgments reports it as absent.
    pids = sorted(by_persona)
    pairs = []
    for _ in range(n_pairs if len(pids) >= 2 else 0):
        same = rng.random() < 0.5
        if same:
            pid = rng.choice(pids)
            if len(by_persona[pid]) < 2:
                continue
            a, b = rng.sample(by_persona[pid], 2)
        else:
            pa, pb = rng.sample(pids, 2)
            a = rng.choice(by_persona[pa])
            b = rng.choice(by_persona[pb])
        pairs.append((a, b, same))
    def _judge_pair(task):
        a, b, same = task
        prompt = consistency_prompt(a["turns"][:6], b["turns"][:6])
        votes = {}
        for name, role in (("judge_a", ROLE_JUDGE), ("judge_b", ROLE_SELECT)):
            raw = llm.complete(prompt, role=role)
            votes[name] = {"raw": str(raw)[:200], "guess_same": _parse_same(raw)}
        return {"a": a["file"], "b": b["file"], "truth_same": same,
                "persona_a": a["persona"], "persona_b": b["persona"], **votes}

    with ThreadPoolExecutor(max_workers=threads) as ex:
        judgments["consistency"] = list(ex.map(_judge_pair, pairs))

    (out_dir / "judgments.json").write_text(
        json.dumps(judgments, indent=1, ensure_ascii=False), encoding="utf-8")
    return summarize_judgments(judgments)


def summarize_judgments(judgments: dict) -> dict:
    """Accuracy per judge, per persona, and combined -- pure arithmetic, recomputable."""
    out: dict = {"identification": {}, "consistency": {}}
    ident = judgments["identification"]
    if ident:
        for judge in ("judge_a", "judge_b"):
            ok = sum(1 for j in ident if j[judge]["guess"] == j["persona"])
            out["identification"][judge] = round(ok / len(ident), 4)
        agree = sum(1 for j in ident
                    if j["judge_a"]["guess"] == j["judge_b"]["guess"])
        out["identification"]["agreement"] = round(agree / len(ident), 4)
        out["identification"]["n"] = len(ident)
        by_p: dict[str, list] = {}
        for j in ident:
            by_p.setdefault(j["persona"], []).append(j)
        out["identification"]["by_persona"] = {
            p: round((sum(1 for j in js if j["judge_a"]["guess"] == p)
                      + sum(1 for j in js if j["judge_b"]["guess"] == p))
                     / (2 * len(js)), 4)
            for p, js in sorted(by_p.items())}
    cons = judgments["consistency"]
    if cons:
        # Balanced accuracy per judge: accuracy on SAME pairs and on DIFFERENT pairs,
        # averaged -- a judge's fixed prior (one answered DIFFERENT to all 100 pairs;
        # the other leaned SAME 65/49) gains on one half exactly what it loses on the
        # other, so only genuine discrimination raises the score.
        same = [j for j in cons if j["truth_same"]]
        diff = [j for j in cons if not j["truth_same"]]
        # Validity gate: a judge is scoreable only if its answer marginal on the
        # (balanced) pair set is non-degenerate -- minority answer >= 10%. A judge that
        # gives one answer regardless of input (judge_a: 993/1000 DIFFERENT across the
        # nine stored runs) carries no information about the pair; balanced accuracy
        # pins it at 0.5 by construction, and averaging it in only dilutes the
        # informative judge. Excluded judges keep their individual score in the record.
        valid_judges = []
        same_rates = {}
        for judge in ("judge_a", "judge_b"):
            ans = [j[judge]["guess_same"] for j in cons
                   if j[judge]["guess_same"] is not None]
            rate = (sum(1 for a in ans if a) / len(ans)) if ans else 0.0
            same_rates[judge] = round(rate, 4)
            if 0.10 <= rate <= 0.90:
                valid_judges.append(judge)
        for judge in ("judge_a", "judge_b"):
            acc_s = (sum(1 for j in same if j[judge]["guess_same"] is True) / len(same)
                     if same else 0.0)
            acc_d = (sum(1 for j in diff if j[judge]["guess_same"] is False) / len(diff)
                     if diff else 0.0)
            out["consistency"][judge] = round((acc_s + acc_d) / 2, 4)
        out["consistency"]["same_rate"] = same_rates
        out["consistency"]["valid_judges"] = valid_judges
        out["consistency"]["score"] = (
            round(sum(out["consistency"][j] for j in valid_judges)
                  / len(valid_judges), 4) if valid_judges else None)
        out["consistency"]["n"] = len(cons)
        # Per-persona: all pairs INVOLVING p (its same-pairs as positives, its
        # cross-pairs as negatives), so a SAME-biased judge cannot score well.
        # Judges are stateless per call, so the skewed per-persona base rate does
        # not bias individual judgments (ruling 2026-08-11).
        by_p: dict[str, list] = {}
        for j in cons:
            for pid in {j["persona_a"], j["persona_b"]}:
                by_p.setdefault(pid, []).append(j)
        def _balanced(js):
            s_ = [j for j in js if j["truth_same"]]
            d_ = [j for j in js if not j["truth_same"]]
            accs = []
            for judge in valid_judges:
                a_s = (sum(1 for j in s_ if j[judge]["guess_same"] is True) / len(s_)
                       if s_ else None)
                a_d = (sum(1 for j in d_ if j[judge]["guess_same"] is False) / len(d_)
                       if d_ else None)
                pair = [x for x in (a_s, a_d) if x is not None]
                if pair:
                    accs.append(sum(pair) / len(pair))
            return round(sum(accs) / len(accs), 4) if accs else None
        out["consistency"]["by_persona"] = {
            pid: {"acc": _balanced(js), "n_pairs": len(js)}
            for pid, js in sorted(by_p.items())}
    return out
