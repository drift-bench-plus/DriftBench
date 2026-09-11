"""A3's lessonbook: online, generation-based self-evolution (design of record:
docs/a3v2-lessonbook-design.md, replacing the raw-example askbook after the 2026-08-20
review docs/a3-self-evolving-method-review.md).

The mechanism in one paragraph: a run starts with an EMPTY book. The runner executes the
episode list in deterministic generations; when a generation finishes, the agent model
reflects on the generation's episodes -- projected onto what the agent could observe and
mechanically redacted -- and distills at most eight abstract lessons about clarification
strategy (when to ask, what kind of thing to ask, how to phrase it, when not to ask).
Validators reject any lesson that could carry task content; the surviving lessons are the
whole playbook the next generation sees. Nothing crosses runs, personas, seeds, or
benchmarks: the snapshot chain lives inside the run's own output directory and is bound to
its cell.

INTEGRITY: the summarizer's input is the observable projection ONLY -- opening query, the
agent's own questions, user replies, proposal reactions, acceptance, outcome. Hidden
intent, masks and ground truth are never read. On top of the askbook's observable-only
contract, the OUTPUT is also constrained: abstract, bounded, and checked against the run's
own text so no concrete entity or reusable answer survives into a later prompt.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from .askbook import DomainProfile, _content_words, get_profile, observable_view

FORMAT = "lessonbook_v2"   # v4 harness (2026-08-20): outcome brake + text dedupe + id-hijack
                           # guard + numberless rendering; v1 books fail closed at binding
MAX_LESSONS = 8
MAX_FIELD = 200
MAX_WHEN = 280   # "when" must fit its full applicability condition (q2: too_long:when starved early books)
UTILITIES = ("high", "medium", "low")

# ------------------------------------------------------------------ redaction
# Layer (a): typed placeholders replace concrete entities BEFORE the summarizer reads
# anything. Regex cannot catch everything (review doc: "regex-only redaction will not be
# sufficient by itself") -- layers (b) and (c) in validate_lesson stand behind it.
_REDACTIONS = (
    (re.compile(r"#W\d+", re.I), "<ORDER_ID>"),        # abbreviated mentions too (#W999)
    (re.compile(r"\bW\d{6,}\b", re.I), "<ORDER_ID>"),
    (re.compile(r"\b[a-z]+_[a-z]+_\d+\b", re.I), "<USER_ID>"),   # noah_brown_6181
    (re.compile(r"\b(gift_card|credit_card|paypal)_\w+", re.I), "<PAYMENT_METHOD>"),
    (re.compile(r"\$\s?\d[\d,]*(\.\d+)?"), "<PRICE>"),
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b"), "<EMAIL>"),
    (re.compile(r"\d{6,}"), "<ID>"),            # long digit runs, boundaries or not --
    (re.compile(r"\b\d{5}\b"), "<ZIP>"),        # review: \b misses digits glued to '_'
    (re.compile(r"\d+(\.\d+)?"), "<NUM>"),      # FINAL PASS: no digit survives redaction
)


def redact(text: str) -> str:
    t = text or ""
    for rx, tok in _REDACTIONS:
        t = rx.sub(tok, t)
    return t


# ------------------------------------------------------------------- digests
def episode_digest(traj: dict, *, max_text: int = 220) -> dict:
    """One episode as the summarizer sees it: observable, redacted, compact.

    Alongside the interaction sequence it carries the outcome features the review said the
    askbook ignored (§3.3): asks, commit attempts, rejections, turns -- so "useful" can be
    judged by decision value and patience cost rather than reply length. The sequence
    interleaves the agent's own non-conversational moves as bare {"did": kind} markers
    (no tool, no arguments): without them the summarizer cannot evaluate its central
    criterion, whether an answer CHANGED what the agent did next (2026-08-20 review).

    `sample_id` is provenance for the snapshot lineage ONLY -- summarize() strips it
    before anything reaches the reflection prompt, because on tau2 the id embeds the
    ground-truth fault strategy (`<tree>__false_presupposition`), which the agent must
    never see (2026-08-20 review, critical).
    """
    # ---- move-level credit labels (ruling 2026-08-20: "does the agent know which are
    # good moves and which are bad moves?"). Two labels, both mechanical and computed
    # ONLY from what the agent could observe -- never from acceptance.reason, which is
    # grader-internal and stays excluded:
    #   answer_used  (per ask): content the reply newly introduced later appears in one
    #                of the agent's own commands/proposals -- the answer changed what it
    #                did. An ask whose answer never got used was patience spent for
    #                nothing.
    #   grounded     (per command-carrying proposal): every argument value traces to the
    #                user's words or a prior tool observation. A value from neither is a
    #                guess -- the autopsied core failure.
    turns = traj.get("turns", [])
    _V = re.compile(r'"([^"]{2,60})"|\b(\d{5,})\b')     # quoted values + bare ids

    def _vals(cmd: str) -> list[str]:
        return [a or b for a, b in _V.findall(cmd or "")]

    later_agent_text: list[str] = [""] * (len(turns) + 1)
    for i in range(len(turns) - 1, -1, -1):
        act = turns[i].get("action") or {}
        own = f"{act.get('command') or ''} {act.get('proposal_raw') or ''}"
        later_agent_text[i] = later_agent_text[i + 1] + " " + own.lower()

    seq = []
    seen_props: list[str] = []
    patience_end = None
    seen_user = ((traj.get("header") or {}).get("query") or "").lower()
    seen_obs = ""
    for i, rec in enumerate(turns):
        if rec.get("patience_after") is not None:
            patience_end = rec.get("patience_after")
        act = rec.get("action") or {}
        kind = act.get("kind")
        if kind == "ASK":
            reply = rec.get("reply") or ""
            new_words = _content_words(reply) - _content_words(seen_user)
            used = any(w in later_agent_text[i + 1] for w in new_words) if new_words else False
            seq.append({"ask": redact(act.get("question") or "")[:max_text],
                        "reply": redact(reply)[:max_text],
                        "answer_used": used})
            seen_user += " " + reply.lower()
        elif kind == "PROPOSE":
            sig = json.dumps(act, sort_keys=True)
            entry = {"proposed": True,
                     "accepted": bool((rec.get("acceptance") or {}).get("ok")),
                     "reaction": redact(rec.get("reply") or "")[:max_text]}
            cmd = act.get("command")
            if cmd:
                vals = _vals(cmd)
                known = seen_user + " " + seen_obs
                entry["grounded"] = all(v.lower() in known for v in vals) if vals else True
            if sig in seen_props:
                entry["identical_resubmission"] = True   # observable; the reflection
            seen_props.append(sig)                       # checklist asks about exactly this
            seq.append(entry)
            seen_user += " " + (rec.get("reply") or "").lower()
        elif kind:
            obs = rec.get("observation")
            entry = {"did": str(kind)}
            low = str(obs or "").lower()
            if obs and any(m in low[:200] for m in ("error", "invalid", "not found")):
                # FAILED CALLS ARE VISIBLE (2026-08-21): several autopsied losses were
                # pure format fumbles (an enum needing the customer's exact wording) that
                # reflection could never see -- digests carried only bare did-markers.
                # The error text, redacted, lets the book learn what the fix was.
                entry["error"] = redact(str(obs))[:120]
            seq.append(entry)
            if obs:
                seen_obs += " " + low
    view = observable_view(traj)
    return {"sample_id": (traj.get("header") or {}).get("sample_id") or "",
            "query": redact(view["query"])[:max_text],
            "outcome": view.get("outcome"),
            "turns_used": len(traj.get("turns") or []),
            "patience_end": patience_end,                # the meter is published to this arm
            "asks": sum(1 for s in seq if "ask" in s),
            "proposals": sum(1 for s in seq if s.get("proposed")),
            "rejected": sum(1 for s in seq if s.get("proposed") and not s["accepted"]),
            "sequence": seq}


# ------------------------------------------------------------------ validation
_HAS_DIGIT = re.compile(r"\d")
_HAS_MARKER = re.compile(r"[#$@]")
_PLACEHOLDER_OK = re.compile(r"<(ORDER_ID|PAYMENT_METHOD|PRICE|EMAIL|ID|ZIP|NUM)>")


def _ngrams(text: str, n: int = 5) -> set[tuple[str, ...]]:
    words = re.findall(r"[a-z']+", (text or "").lower())
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def source_ngrams(digests: list[dict]) -> set[tuple[str, ...]]:
    """5-word-grams of every source text the run has produced so far. A lesson that shares
    one is quoting its source, not abstracting from it."""
    grams: set[tuple[str, ...]] = set()
    for d in digests:
        grams |= _ngrams(d.get("query") or "")
        for s in d.get("sequence") or []:
            for k in ("ask", "reply", "reaction"):
                grams |= _ngrams(s.get(k) or "")
    return grams


# A tiny exemption list so the entity guard does not ban ordinary strategy vocabulary
# that happens to double as a first name or city somewhere in the DB. Deliberately short:
# a false-positive drop is visible and cheap, a false-negative leak is neither.
_COMMON = frozenset("""young walker parker hunter mason taylor summer daisy june april may
grace hope faith bill will frank harry jack rose lily ivy amber crystal chase reed""".split())


def build_entity_lexicon(db_path: str | Path | None) -> dict:
    """Names, cities, street lines and product names harvested from the benchmark's OWN
    database -- exactly the universe of concrete entities an episode can surface. The
    validator rejects any lesson naming one (review invariant 7.2: regex classes alone
    cannot catch a person or product name)."""
    if not db_path or not Path(db_path).exists():
        return {"tokens": frozenset(), "phrases": tuple()}
    db = json.loads(Path(db_path).read_text(encoding="utf-8"))
    users = db.get("users") or []
    users = users.values() if isinstance(users, dict) else users
    products = db.get("products") or []
    products = products.values() if isinstance(products, dict) else products
    tokens: set[str] = set()
    phrases: set[str] = set()
    for u in users:
        nm = u.get("name") or {}
        first = (nm.get("first_name") or "").lower()
        last = (nm.get("last_name") or "").lower()
        for t in (first, last):
            if len(t) >= 3 and t not in _COMMON:
                tokens.add(t)
        if first and last:
            phrases.add(f"{first} {last}")
        addr = u.get("address") or {}
        city = (addr.get("city") or "").lower()
        if city and city not in _COMMON:
            (phrases if " " in city else tokens).add(city)
        street = (addr.get("address1") or "").lower()
        street = re.sub(r"^\d+\s+", "", street)
        if street:
            phrases.add(street)
    for p in products:
        name = (p.get("name") or "").lower()
        if name:
            # products ban as PHRASES only: "portable charger" identifies a catalogue
            # product, but its single words are ordinary strategy vocabulary -- measured
            # 2026-08-20: token-banning them rejected valid lessons for containing
            # "action" (Action Camera) and "portable". Person names and cities stay
            # token-banned; they are identity, not vocabulary.
            phrases.add(name)
    return {"tokens": frozenset(tokens), "phrases": tuple(sorted(phrases, key=len, reverse=True))}


def validate_lesson(lesson: dict, *, grams: set, tool_names: tuple[str, ...] = (),
                    lexicon: dict | None = None) -> str | None:
    """None if the lesson is admissible; else the reason it is rejected. Fail-closed and
    per-lesson: one bad lesson never poisons the batch, and every drop is counted."""
    if not isinstance(lesson, dict):
        return "not_a_dict"
    extra = set(lesson) - {"when", "do", "avoid", "evidence", "utility",
                        "confidence", "id"}
    if extra:
        return f"unknown_fields:{sorted(extra)}"        # no escape hatch for free-form memory
    for k in ("when", "do"):
        if not isinstance(lesson.get(k), str) or not lesson[k].strip():
            return f"missing:{k}"
    if not isinstance(lesson.get("avoid", ""), str):
        return "bad:avoid"
    if lesson.get("utility") not in UTILITIES:
        return "bad:utility"
    ev = lesson.get("evidence")
    if not isinstance(ev, int) or isinstance(ev, bool) or ev < 1:
        return "bad:evidence"                            # review: True passes bare isinstance
    c = lesson.get("confidence")
    if not isinstance(c, (int, float)) or isinstance(c, bool) or not (0.0 < float(c) <= 1.0):
        return "bad:confidence"
    joined = " ".join((lesson["when"], lesson["do"], lesson.get("avoid") or ""))
    for k in ("when", "do", "avoid"):
        v = lesson.get(k) or ""
        if len(v) > (MAX_WHEN if k == "when" else MAX_FIELD):
            return f"too_long:{k}"
    scrubbed = _PLACEHOLDER_OK.sub("", joined)          # typed placeholders are abstract
    if _HAS_DIGIT.search(scrubbed):
        return "contains_digits"
    if _HAS_MARKER.search(scrubbed):
        return "contains_id_marker"
    low = joined.lower()
    for name in tool_names:
        if name and name.lower() in low:
            return f"names_tool:{name}"                 # no executable answers
    if lexicon:
        words = set(re.findall(r"[a-z']+", low))
        hit = words & lexicon.get("tokens", frozenset())
        if hit:
            return f"names_entity:{sorted(hit)[0]}"     # a DB name/city/product token
        for ph in lexicon.get("phrases", ()):
            if ph in low:
                return f"names_entity:{ph}"
    if _ngrams(joined) & grams:
        return "quotes_source"
    return None


# ------------------------------------------------------------------ summarize
_SUMMARIZE_PROMPT = """You are the reflection step of a {domain} service agent that is \
evaluated on completing the {actor}'s true request with as few wasted interactions as \
possible. Below are (1) the strategy lessons you currently hold and (2) digests of your own \
recently finished episodes: the opening request, each question you asked with the {actor}'s \
reply, each submission with its reaction, and the outcome. Concrete values were replaced by \
placeholders like <ORDER_ID>; treat them as opaque.

Update the lessons. A lesson is a reusable rule about CLARIFICATION STRATEGY. Do these \
three tasks explicitly, in order: (A) From asks labeled "answer_used": true, identify the \
TYPES of question that paid off -- what kind of information they sought and in what \
situation -- and write or strengthen lessons naming those question types. (B) From asks \
labeled "answer_used": false and from failed episodes, identify the types of question NOT \
to ask and the situations where asking wasted patience; write those as explicit \
do-not-ask lessons. (C) From proposals labeled "grounded": false that were rejected, \
write lessons naming the check or the question that would have grounded the guessed value \
first.

EDIT THE BOOK INCREMENTALLY -- never rewrite it wholesale. Each lesson carries an "id" \
and a "confidence" (0-1, your estimate that following it improves outcomes). Keep every \
existing lesson whose confidence holds up, raising its confidence and evidence when new \
digests confirm it and lowering them when contradicted. REPLACE or RETIRE only lessons \
whose confidence has fallen below 0.4. Add new lessons only into free slots (at most \
{max_lessons} total), starting them at modest confidence (0.4-0.6) until repeated \
evidence earns more. Return every kept lesson WITH its existing id; new lessons get no id. Judge a question by whether its answer CHANGED what you did next \
and whether the episode then succeeded -- a long reply that changed nothing is not success. \
Asking is a cost: each question spends the {actor}'s limited patience, and episodes that \
ran out of patience FAILED. But a wrong state-changing action costs MORE: a rejected \
submission spends more of the {actor}'s patience than a question does, resubmitting an \
identical proposal or repeating a failed question spends it for nothing, and some actions \
cannot be retried once taken. Weigh both failure modes: compare episodes that lost by \
asking too much with episodes that lost by acting on a guess, and episodes that succeeded \
with either.

Moves in the digests carry credit labels you should trust over your own impression: an \
ask marked "answer_used": true changed what you did next (a good spend); "answer_used": \
false means the reply never entered a later action (patience spent for nothing, or a \
question that needed no answer). A proposal marked "grounded": false contains at least one \
argument value that appears in NO user statement and NO tool observation before it -- a \
guess; "grounded": true proposals that were still rejected point at a missing requirement \
rather than a wrong value. Rejected proposals and "identical_resubmission" are bad moves by \
definition. Cite these labels when you weigh a lesson's evidence. Evidence is \
OUTCOME-WEIGHTED, never a match counter: a digest supports a lesson only if the episode \
FOLLOWED the lesson's advice and succeeded, or departed from it and failed. An episode \
that matched the lesson's situation, followed its advice, and still FAILED is evidence \
AGAINST the lesson: lower its confidence, and consider narrowing its "when" to exclude \
the situation that failed. A lesson's evidence may never grow from a digest whose episode \
failed while complying with it.

For every FAILED episode, locate the FIRST state-changing step and answer three questions \
before writing lessons: (1) Was every argument of that step a value the {actor} stated \
literally or a tool result confirmed -- or was some argument filled in from a vague, \
comparative, or merely assumed description? A request can supply many exact values and \
still leave the one that matters unstated. (2) Did any tool result contradict something \
the request took for granted? When your own fresh observation and the {actor}'s stated \
assumption disagree about the current state of things, the observation is the fact -- a \
lesson should say what to do then. (3) Would one question at that step, offering the \
{actor} concrete options to choose between, have replaced the guess? If yes, the lesson to \
write is about that question. For failed episodes that DID ask, check the mirror image: \
did the question have an answer that could still change anything (or was everything \
already given, or already done), could the {actor} plausibly answer it, and was it \
repeated after failing once? A question with nothing left to learn, or re-asked after the \
{actor} could not answer it, only spends patience. (4) Look at what each question CAUSED: \
did the {actor}'s reply stay on the original request, or did it veer into new or changed \
demands? Compare the questions that preceded direction changes with the ones that did \
not -- their timing, their breadth, what they were about -- and if a pattern separates \
them IN THESE EPISODES, write that pattern as a lesson. (5) Note what preceded each \
direction change: had any patience just been spent (a question or a rejected \
submission)? If direction changes reliably follow the first spend, weigh what that \
implies about when spending is safe in these episodes, and write that as a lesson. \
(6) For every REJECTED submission, read its reaction: did it request a step-by-step \
account of what was done, voice doubt, or name a requirement? Compare what happened next \
in episodes that ANSWERED the reaction with a question against episodes that resubmitted \
an identical or near-identical proposal, and write what separated the recoveries as a \
lesson. The reaction to a rejection is the {actor} saying what is missing: a rule against \
re-asking or against following changed demands must never be writable over answering it.

(7) Failed tool calls appear in the sequence with their error text: compare a failing \
call with the retry that finally worked -- the difference (often the {actor}'s exact \
wording or casing, or an allowed-values hint inside the error) is a lesson about how to \
fill that KIND of field.

Hard rules for every lesson:
- a "when" that licenses acting WITHOUT asking must state what makes the situation fully \
specified -- every value the next state-changing step needs was given literally or \
confirmed by observation -- and must never be writable over a situation where a needed \
value is only described vaguely or by comparison; a "when" that calls for asking must name \
exactly what KIND of value is missing AND must be false whenever that value is already \
derivable from the {actor}'s own words or a prior observation, whenever a failed step has \
an untried retry using the {actor}'s own wording, and whenever every stated task has \
already been carried out -- a question is only worth writing into a lesson if no free \
move could learn the same thing.
- if two lessons could both match the same situation with opposite advice, tighten their \
"when" clauses until they cannot, or retire the broader one.
- a lesson that treats a tool observation as decisive must say WHICH observations qualify: \
only a successful result from a step suited to producing that kind of value. An error or \
not-found reply is an observation about the attempted step, never about the world -- it \
must not satisfy any lesson's "observation contradicts the {actor}" condition, because the \
step itself may have been the wrong one.
- where it matters, say whether the rule applies before a step that can be retried or one \
that cannot: the bar for acting on inference is highest just before an action that cannot \
be undone.
- abstract and transferable: NO ids, prices, product names, addresses, dates, or any value \
copied from an episode; placeholder tokens like <ORDER_ID> are allowed as type names.
- strategy only: never name a tool or prescribe a specific action or value to submit.
- "when" = the situation cue, stating its FULL applicability condition (max {max_when} chars); \
"do" = the behaviour (max {max_field} chars) -- if a field would run long, split the situation \
into two lessons rather than cramming; "avoid" = the failure it prevents, may be empty; "evidence" = how many digests \
support it (integer >= 1); "utility" = high | medium | low; "confidence" = 0-1.
- keep, strengthen, revise, or retire the existing lessons based on the new evidence; at \
most {max_lessons} lessons total, ranked most useful first.

Return ONLY a JSON object: {{"lessons": [...]}}.

CURRENT LESSONS:
{prior}

EPISODE DIGESTS ({n} episodes):
{digests}"""


def _norm_text(les: dict) -> str:
    """Content-word normalization of a lesson's text; two lessons with the same _norm_text
    are the same lesson however their ids or counts differ (the gen-6..9 books carried a
    verbatim L6==L8 pair that split one lesson's evidence across two ranks)."""
    return " ".join(re.findall(r"[a-z']+", " ".join(
        str(les.get(k) or "") for k in ("when", "do", "avoid")).lower()))


def _json_candidates(text: str) -> list[str]:
    r"""Every balanced {...} block in the reply, longest first.

    WHY (2026-08-25): the old extractor was `re.search(r"\{.*\}", text, re.S)` --
    greedy from the FIRST brace to the LAST. With thinking enabled a model prefixes
    its answer with reasoning prose, and any brace in that prose became the start of
    the "JSON", so the match swallowed narration and json.loads failed at some
    interior character (measured: gemini cell, JSONDecodeError at char 1313, book
    silently empty). Scanning for BALANCED blocks and trying each in turn reads the
    model faithfully instead. Same class of fix as the action-parser repairs (D28):
    no mechanism or method changes, only how honestly we read a model's output.
    """
    out, depth, start = [], 0, None
    for i, ch in enumerate(text or ""):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                out.append(text[start:i + 1])
                start = None
    return sorted(out, key=len, reverse=True)


def summarize(llm, *, prior: list[dict], digests: list[dict],
              profile: DomainProfile | str, tool_names: tuple[str, ...] = (),
              extra_grams: set | None = None, lexicon: dict | None = None,
              reflector_role: str = "agent_audit",
              prior_gen_success: list[float] | None = None) -> dict:
    """One generation's reflection: digests in, validated lessons out.

    Deterministic by construction -- digests sorted by sample_id, temperature 0, and the
    response cache keys on the full prompt -- so re-deriving a snapshot (resume, replay,
    audit) is byte-identical.
    """
    prof = get_profile(profile)
    digests = sorted(digests, key=lambda d: d.get("sample_id") or "")
    # stable ids let the model edit incrementally and let us hold it to that
    prior = [dict(l, id=l.get("id") or f"L{i+1}") for i, l in enumerate(prior or [])]
    # sample_id is lineage provenance and NEVER prompt material: on tau2 it embeds the
    # injected fault strategy (`<tree>__false_presupposition`), which is ground truth the
    # agent must not observe (2026-08-20 review, critical)
    slim = [{k: v for k, v in d.items() if k != "sample_id"} for d in digests]
    prompt = _SUMMARIZE_PROMPT.format(
        domain=prof.name.replace("_", " "), actor=prof.actor,
        max_field=MAX_FIELD, max_when=MAX_WHEN, max_lessons=MAX_LESSONS,
        prior=json.dumps(prior, ensure_ascii=False, indent=1) if prior else "(none yet)",
        n=len(slim), digests=json.dumps(slim, ensure_ascii=False, indent=1))
    # reflector_role selects WHICH model reflects: "agent_audit" = the agent model
    # (self-reflection); "select" = the user-side model (v7: stronger reflector, same
    # allocation, deterministic temp-0 role). Cache keys include the role, so the two
    # reflector variants never share summaries.
    text = llm.complete(prompt, role=reflector_role, temperature=0.0, max_tokens=4000)
    raw: list = []
    parse_error = None
    cands = _json_candidates(text)
    if not cands:
        # no JSON at all is a FAILED reflection, never "the model retired every lesson"
        parse_error = "no_json_object"
    else:
        for cand in cands:
            try:
                obj = json.loads(cand)
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(obj, dict) and "lessons" in obj:
                raw = list(obj.get("lessons") or [])
                parse_error = None
                break
            parse_error = "no_lessons_key"
        else:
            parse_error = parse_error or "unparseable_json"
    # the overlap guard covers the WHOLE run so far, not just this generation: a lesson
    # must not quote anything the run has ever seen (the runner passes the accumulation)
    grams = source_ngrams(digests) | (extra_grams or set())
    prior_by_id = {pl["id"]: pl for pl in prior}
    kept, dropped, seen_ids = [], {}, set()
    seen_text: dict[str, dict] = {}
    _FIELDS = ("when", "do", "avoid", "evidence", "utility", "confidence", "id")
    for les in raw[:MAX_LESSONS * 2]:
        why = validate_lesson(les, grams=grams, tool_names=tool_names, lexicon=lexicon)
        if why is None and len(kept) < MAX_LESSONS:
            lid = les.get("id")
            if lid and lid in seen_ids:
                dropped["duplicate_id"] = dropped.get("duplicate_id", 0) + 1
                continue
            # ID-HIJACK GUARD (v4): reusing a prior id for essentially unrelated text is a
            # silent delete of the prior lesson (gen 4-5 killed the act-bias L6 this way).
            # Strip the id: the text is admitted as a NEW lesson and the prior one must
            # exit through the honest path (demotion below 0.6, else stability guard).
            pl = prior_by_id.get(lid) if lid else None
            if pl is not None:
                a, b = set(_norm_text(les).split()), set(_norm_text(pl).split())
                if a and b and len(a & b) < 0.3 * len(a | b):
                    dropped["id_hijack_stripped"] = dropped.get("id_hijack_stripped", 0) + 1
                    les = {k: v for k, v in les.items() if k != "id"}
                    lid = None
            # TEXT DEDUPE (v4): a lesson whose text already stands in the book is merged
            # into its twin (evidence summed, confidence maxed), never rendered twice --
            # duplicates doubled a lesson's rendered weight while SPLITTING its evidence
            # across two ranks, and the extra slot squeezed out the L4 recovery lesson.
            key = _norm_text(les)
            twin = seen_text.get(key)
            if twin is not None:
                if isinstance(twin.get("evidence"), int) and isinstance(les.get("evidence"), int):
                    twin["evidence"] = twin["evidence"] + les["evidence"]
                ca, cb = twin.get("confidence"), les.get("confidence")
                if isinstance(ca, (int, float)) and isinstance(cb, (int, float)):
                    twin["confidence"] = max(float(ca), float(cb))
                dropped["duplicate_text"] = dropped.get("duplicate_text", 0) + 1
                continue
            if lid:
                seen_ids.add(lid)
            entry = {k: les.get(k, "") for k in _FIELDS if k != "id" or les.get(k)}
            kept.append(entry)
            seen_text[key] = entry
        elif why is None:
            dropped["over_cap"] = dropped.get("over_cap", 0) + 1   # valid but beyond MAX_LESSONS
        else:
            dropped[why] = dropped.get(why, 0) + 1
    # STABILITY GUARD (ruling 2026-08-20: "replace only the bad ones"): a prior lesson with
    # confidence >= 0.6 may not silently vanish. If the model dropped one without demoting
    # it first, it is re-inserted while room remains -- retirement must happen the honest
    # way, by lowering its confidence below 0.4 in one update and replacing it in the next.
    for pl in prior:
        conf = pl.get("confidence")
        if (isinstance(conf, (int, float)) and float(conf) >= 0.6
                and pl["id"] not in seen_ids and len(kept) < MAX_LESSONS):
            if _norm_text(pl) in seen_text:
                # the model moved this text under another id; re-inserting the old copy
                # is what fossilized the verbatim L6==L8 duplicate pair from gen 6 on
                dropped["stability_dedup"] = dropped.get("stability_dedup", 0) + 1
                continue
            kept.append(pl)
            seen_text[_norm_text(pl)] = pl
            dropped["stability_reinserted"] = dropped.get("stability_reinserted", 0) + 1
    # assign ids to genuinely new lessons; a retired lesson's id is never recycled, so
    # lineage across snapshots stays unambiguous
    used = {l.get("id") for l in kept if l.get("id")} | {pl["id"] for pl in prior}
    nxt = 1
    for l in kept:
        if not l.get("id"):
            while f"L{nxt}" in used:
                nxt += 1
            l["id"] = f"L{nxt}"
            used.add(l["id"])
    if parse_error and prior:
        kept = prior                                     # fail closed: keep what we had
    # OUTCOME BRAKE (v4): the harness knows how each generation actually scored; the model
    # does not get to inflate its own weights through a losing streak. The book's evidence
    # counter tracked episode MATCHES, not wins (the top anti-ask lesson grew ev 18->38
    # across the losing gens 5-9 while its text never changed), so when this generation's
    # success rate fell below the trailing mean of the previous three, upward drift is
    # frozen (evidence and confidence clamped to their prior values) and every surviving
    # prior lesson's confidence is shrunk toward neutral (0.5 + 0.8*(c-0.5)). Two losing
    # generations pull a 0.97 hard rule back to ~0.80 -- soft advice again -- and a lesson
    # that is genuinely good re-earns its weight in the next winning generation. Purely
    # mechanical, computed from the same observable outcomes the digests already carry.
    n_out = sum(1 for d in digests if d.get("outcome"))
    rate = (sum(1 for d in digests if d.get("outcome") == "SUCCESS") / n_out) if n_out else None
    hist = [float(x) for x in (prior_gen_success or [])]
    window = hist[-3:]
    declined = bool(window) and rate is not None and rate < sum(window) / len(window)
    if declined:
        for l in kept:
            pl = prior_by_id.get(l.get("id"))
            if pl is None:
                continue                                 # new this gen: starts modest anyway
            if isinstance(l.get("evidence"), int) and isinstance(pl.get("evidence"), int):
                l["evidence"] = min(l["evidence"], pl["evidence"])
            c, pc = l.get("confidence"), pl.get("confidence")
            if isinstance(c, (int, float)) and isinstance(pc, (int, float)):
                c = min(float(c), float(pc))
                l["confidence"] = round(0.5 + 0.8 * (c - 0.5), 3)
    if rate is not None:
        hist.append(round(rate, 4))
    return {"lessons": kept, "dropped_by_validator": dropped, "parse_error": parse_error,
            "gen_success": hist, "outcome_brake": declined}


# ------------------------------------------------------------------ snapshots
def snapshot_sha(snap: dict) -> str:
    body = {k: v for k, v in snap.items() if k != "sha"}
    return hashlib.sha1(json.dumps(body, sort_keys=True).encode()).hexdigest()[:12]


def make_snapshot(*, lessons_result: dict, profile: str, run_id: str, arm: str,
                  persona: str, seed: int, generation: int, parent: dict | None,
                  source_episode_ids: list[str], n_seen: int,
                  gen_size: int = 0, samples_sha: str = "") -> dict:
    snap = {"format": FORMAT, "profile": get_profile(profile).name, "run_id": run_id,
            "arm": arm, "persona": persona, "seed": int(seed), "generation": int(generation),
            "gen_size": int(gen_size), "samples_sha": samples_sha,
            "parent_sha": (parent or {}).get("sha") or "",
            "source_episode_ids": sorted(source_episode_ids), "n_episodes_seen": int(n_seen),
            "lessons": lessons_result["lessons"],
            "dropped_by_validator": lessons_result.get("dropped_by_validator") or {},
            "parse_error": lessons_result.get("parse_error"),
            # v4 outcome brake: per-generation success rates and whether the brake fired,
            # so the drift of every book is auditable from its own snapshot chain
            "gen_success": lessons_result.get("gen_success") or [],
            "outcome_brake": bool(lessons_result.get("outcome_brake"))}
    snap["sha"] = snapshot_sha(snap)
    return snap


class BindingError(ValueError):
    """A snapshot outside its declared cell. Always fail closed (review doc §3.8/§6)."""


def check_binding(snap: dict, *, profile: str, run_id: str, arm: str, persona: str,
                  seed: int, generation: int | None = None,
                  gen_size: int | None = None, samples_sha: str | None = None) -> dict:
    if snap.get("format") != FORMAT:
        raise BindingError(f"lessonbook format {snap.get('format')!r}, expected {FORMAT!r} "
                           "-- legacy askbooks are quarantined and cannot back A3")
    want = {"profile": get_profile(profile).name, "run_id": run_id, "arm": arm,
            "persona": persona, "seed": int(seed)}
    if gen_size is not None:
        want["gen_size"] = int(gen_size)     # resume with a different partition = a
    if samples_sha is not None:              # different method, not a resume
        want["samples_sha"] = samples_sha    # ditto for a different dataset/split
    for k, v in want.items():
        if snap.get(k) != v:
            raise BindingError(f"lessonbook {k}={snap.get(k)!r} does not match this cell's "
                               f"{k}={v!r} -- one independent playbook per benchmark AND run")
    if generation is not None and snap.get("generation") != int(generation):
        raise BindingError(f"lessonbook generation {snap.get('generation')} != {generation}")
    if snap.get("sha") != snapshot_sha(snap):
        raise BindingError("lessonbook sha mismatch -- snapshot edited after writing")
    return snap


def save_snapshot(snap: dict, path: str | Path) -> None:
    import os
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(f".tmp{os.getpid()}")            # unique: double-launches never race
    tmp.write_text(json.dumps(snap, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(p)                                       # atomic: readers never see a torn file


def load_snapshot(path: str | Path | None) -> dict | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


# ------------------------------------------------------------------ rendering
def render(snap: dict | None, profile: DomainProfile | str) -> str:
    """The lessons as one compact prompt block. Empty book -> empty string: generation 0
    runs on the scaffold alone, and the trajectory header says so."""
    if not snap or not snap.get("lessons"):
        return ""
    prof = get_profile(profile)
    lines = [f"\nWHAT THIS RUN'S EARLIER EPISODES TAUGHT (your own reflection over "
             f"{snap['n_episodes_seen']} finished episodes; generation {snap['generation']}):",
             "  Apply a lesson only if EVERY clause of its When-condition is verifiably "
             "true of this conversation right now -- checked against what was actually "
             "said and observed, not a surface resemblance. If two lessons pull in "
             "opposite directions, follow the one whose When-condition describes THIS "
             "situation more specifically -- a condition naming the situation's "
             "particulars beats a broader rule that merely also matches, no matter how "
             "prominent the broader one is; and set a matched lesson aside only on the "
             "strength of something observed in THIS episode, never a general prior. "
             "Re-check which lessons apply before every state-changing action -- that is "
             "the step lessons exist to protect.",
             f"  A question SPENDS the {prof.actor}'s limited patience; your own tool use "
             "does not. No lesson licenses asking for what you can resolve for free: a "
             f"value already present in the {prof.actor}'s own words or a prior tool "
             "result, a failed call you have not yet retried using the "
             f"{prof.actor}'s own wording, or confirmation of work already done -- when "
             "every stated task is executed and verified, submit; acceptance is the free "
             "confirmation, and a confirmation question is a spend that buys nothing. "
             f"Never present the {prof.actor} with a fixed list of options that no tool "
             "output or statement of theirs contains.",
             "  After a rejected submission, never resubmit the same thing unchanged: the "
             f"reaction is the {prof.actor} telling you what is missing. If it asks you "
             "to walk through what you did, or voices doubt, give exactly that "
             "walkthrough as your next question and act on the correction it draws."]
    ordered = sorted(snap["lessons"],
                     key=lambda l: -float(l.get("confidence") or 0.5))
    for les in ordered:
        avoid = f" Avoid: {les['avoid']}" if les.get("avoid") else ""
        # v4: no confidence/evidence numerals in the block. The autopsied mature-gen
        # losses had lesson TEXT verbatim-identical to winning gens while only the counts
        # grew (ev 5->38, conf .90->.97, and %.1f rendered 0.97 as "conf 1.0") -- the
        # numbers alone hardened soft advice into literal compliance. Weight still orders
        # the list; it no longer argues inside the prompt.
        lines.append(f"  - [{les['utility']}] When {les['when']}: {les['do']}.{avoid}")
    # intent-authority vs world-state-authority: the old footer ("what this customer has
    # said always wins") taught A3 to defer to FALSIFIED premises against its own tool
    # observations -- both false_presupposition losses in the 2026-08-20 autopsy
    lines.append(f"  These are strategy lessons from OTHER tasks, never facts about this "
                 f"{prof.actor}. What this {prof.actor} ASKS FOR always wins over any "
                 f"lesson; but about the current state of things, your own fresh tool "
                 f"results outrank both these lessons and any assumption inside the "
                 f"{prof.actor}'s request. A FAILED call outranks nothing: an error or "
                 f"not-found result is evidence about your own attempted step, not about "
                 f"the world -- never treat it as proof that the {prof.actor} is wrong, "
                 f"and before acting on it, re-derive the fact through a different kind "
                 f"of check or take the {prof.actor}'s direct confirmation over it.")
    return "\n".join(lines) + "\n"
