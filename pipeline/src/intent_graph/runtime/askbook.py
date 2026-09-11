"""A3's cross-task ask-knowledge: learning WHAT to ask, from experience.

The idea (ruling 2026-08-13): a summariser watches finished episodes, notices which kinds of
requirement tend to be missing or misstated for a given kind of request, and which questions
actually pulled information out of the user. That distilled experience becomes a playbook
injected into later episodes, so the agent asks about the right elements from the start and
keeps improving as it runs.

EVERY DATASET HAS ITS OWN PLAYBOOK (ruling 2026-08-20). A playbook is knowledge about a
*domain*, so the vocabulary it is written in belongs to that domain. This module once
hard-coded one taxonomy -- WebShop's `budget/option/feature/quantity/purpose` -- and applied
it to every benchmark. On tau2 retail nothing matched, `classify_question` fell through to
"feature" for a third of questions, incidental price mentions bucketed half of them as
"budget", and the rendered playbook instructed the agent to interrogate retail customers
about "purpose" and "feature", which are not things a retail task has. The taxonomy is now a
`DomainProfile` chosen per dataset, a book records the profile it was built under, and
rendering a book under a different profile is a hard error rather than a silent mismatch.

BACK-COMPAT (the WebShop default): every `profile` parameter below defaults to None, and
None reproduces the original WebShop pipeline exactly -- the WebShop taxonomy with its
historical "feature" fall-through, the comma-based audit heuristic, unprefixed query
signatures, version-2 books that carry no `profile` field, and rendering without the
profile guard. The WebShop executor calls with no profile and must keep matching the books
it has already built; every profile-aware pipeline names its profile explicitly (see
ADAPTER_PROFILES), which selects the stricter semantics above.

INTEGRITY (non-negotiable): everything here is distilled ONLY from what the agent itself
could observe -- the opening query, its own questions, the user's replies, and whether a
submission was accepted. The sample's hidden intent, mask and ground truth are never read.
`distill_dir` enforces this by projecting each trajectory onto an observable view before any
statistics are computed.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------- domain profiles
# A dimension is a KIND OF REQUIREMENT the user can be silent or wrong about. Deliberately
# coarse: the playbook has to be actionable in one line ("ask which order"), not a taxonomy.
# Each dimension's pattern is grounded in the domain's own vocabulary -- for the tau2
# domains, in the argument names of the tools an agent actually has to fill.

OTHER = "other"  # the honest bucket for a question that matches no dimension


@dataclass(frozen=True)
class DomainProfile:
    """One dataset's ask-vocabulary and the nouns the playbook is written in."""

    name: str
    dimensions: tuple[str, ...]
    patterns: dict[str, str]
    actor: str = "user"            # "the shopper" / "the customer"
    actors: str = "users"          # "shoppers" / "customers"
    commit: str = "submit"         # the committing verb: "buy" / "submit"
    commit_noun: str = "submission"
    scope_noun: str = "past tasks"  # "past shopping tasks" / "past retail tasks"
    _rx: dict = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_rx",
                           {d: re.compile(p, re.IGNORECASE) for d, p in self.patterns.items()})

    def match(self, low: str) -> str:
        """Which dimension this text targets, judged from its own words."""
        for dim in self.dimensions:                 # declaration order = priority
            rx = self._rx.get(dim)
            if rx and rx.search(low):
                return dim
        return OTHER

    def mentioned(self, low: str) -> list[str]:
        return [d for d in self.dimensions if self._rx[d].search(low)]


WEBSHOP = DomainProfile(
    name="webshop",
    # order matters: the first match wins, so the more specific dimensions come first
    dimensions=("budget", "option", "feature", "quantity", "purpose"),
    patterns={
        "budget": r"\b(budget|price|cost|spend|afford|cheap|expensive|\$|dollar|usd|£|under \d)",
        "option": r"\b(size|colou?r|shade|flavou?r|scent|style|variant|model|pack|width|length|inch|ft|ml|oz)\b",
        "feature": r"\b(feature|material|made of|waterproof|organic|wireless|certif|type|kind of|specification|spec)\b",
        "quantity": r"\b(how many|quantity|count|pieces|pairs|set of|number of)\b",
        "purpose": r"\b(for (a|an|my|the)\b|use it|used for|purpose|occasion|room|who is it for|gift)\b",
    },
    actor="shopper", actors="shoppers", commit="buy", commit_noun="purchase",
    scope_noun="past shopping tasks",
)

# tau2 retail. Grounded in the retail write-tool arguments: order_id, item_ids,
# new_item_ids, payment_method_id, address1/city/state/zip, reason.
TAU2_RETAIL = DomainProfile(
    name="tau2_retail",
    dimensions=("order", "item", "replacement", "payment", "address", "reason", "scope"),
    patterns={
        "order": r"(\border\s*(id|number|#)|#w\d|which order|what order|the order you|"
                 r"\bplaced\b|\border\b.*\b(this|that|which)\b)",
        "item": r"\b(which item|what item|item\s*(id|#)|product\s*(id|#)|sku|"
                r"the item you|which (one|product)|line item)\b",
        "replacement": r"\b(exchange|replace|replacement|swap|instead of|change it to|"
                       r"new item|different (size|colou?r|variant|option|version)|"
                       r"\bmodify\b.*\bto\b)",
        "payment": r"\b(payment method|pay(ment)? (id|with)|credit card|gift card|paypal|"
                   r"refund (to|method|destination)|card ending|original payment)\b",
        "address": r"\b(address|shipping|ship (it )?to|deliver(y| to)|zip( code)?|"
                   r"postal|city|state|street)\b",
        "reason": r"\b(reason|why (are|do|did) you|no longer needed|ordered by mistake|"
                  r"defective|doesn'?t fit|not as described|changed your mind)\b",
        "scope": r"\b(all (of the )?items|whole order|entire order|every item|just the|"
                 r"only the|both items|each item|part of the order)\b",
    },
    actor="customer", actors="customers", commit="submit", commit_noun="submission",
    scope_noun="past retail tasks",
)

# tau2 airline. Grounded in: reservation_id, origin/destination/date, cabin, flights,
# passengers, total_baggages/nonfree_baggages, payment_id/payment_methods, insurance.
TAU2_AIRLINE = DomainProfile(
    name="tau2_airline",
    dimensions=("reservation", "itinerary", "cabin", "passenger", "baggage", "payment",
                "insurance"),
    patterns={
        "reservation": r"\b(reservation\s*(id|code|number)?|booking|confirmation (code|number)|"
                       r"which (trip|reservation|booking))\b",
        "itinerary": r"\b(flight (number|s)?|origin|destination|depart|arriv|"
                     r"outbound|return leg|layover|nonstop|direct|route|travel date|"
                     r"which date|what date|one.?way|round.?trip)\b",
        "cabin": r"\b(cabin|economy|business( class)?|first class|basic economy|seat class|"
                 r"upgrade)\b",
        "passenger": r"\b(passenger|traveler|traveller|who (is|are) (flying|travelling|traveling)|"
                     r"date of birth|\bdob\b|how many people)\b",
        "baggage": r"\b(bag(s|gage)?|luggage|checked bag|carry.?on|suitcase)\b",
        "payment": r"\b(payment|pay with|credit card|gift card|certificate|travel credit|"
                   r"refund (to|method)|card ending)\b",
        "insurance": r"\b(insurance|travel protection|coverage|covered)\b",
    },
    actor="traveler", actors="travelers", commit="submit", commit_noun="submission",
    scope_noun="past airline tasks",
)

# tau2 telecom. Grounded in: customer_id, line_id, expected_status, app_name, phone_number,
# overdue_bill_id, permission, data_used_gb.
TAU2_TELECOM = DomainProfile(
    name="tau2_telecom",
    dimensions=("account", "line", "billing", "service", "device", "reason"),
    patterns={
        "account": r"\b(customer\s*(id|number)|account (id|number|holder)|verify your identity|"
                   r"full name|date of birth|\bdob\b)\b",
        "line": r"\b(line\s*(id|number)?|phone number|which line|\bsim\b|mobile number)\b",
        "billing": r"\b(bill|invoice|overdue|charge|balance|payment|past due|refund)\b",
        "service": r"\b(plan|data (usage|cap|limit)|roaming|suspend|activate|status|"
                   r"throttl|speed|signal|network|coverage|contract)\b",
        "device": r"\b(app|permission|setting|device|airplane mode|wi.?fi|restart|reboot|"
                  r"toggle|phone model)\b",
        "reason": r"\b(reason|why (are|do|did|is)|what happened|the (issue|problem)|complaint)\b",
    },
    actor="customer", actors="customers", commit="resolve", commit_noun="resolution",
    scope_noun="past telecom tasks",
)

PROFILES: dict[str, DomainProfile] = {
    p.name: p for p in (WEBSHOP, TAU2_RETAIL, TAU2_AIRLINE, TAU2_TELECOM)
}

# adapter key -> profile name, so the runner never has to name the profile by hand
ADAPTER_PROFILES = {
    "webshop": "webshop",
    "tau2_retail": "tau2_retail", "retail": "tau2_retail",
    "tau2_airline": "tau2_airline", "airline": "tau2_airline",
    "tau2_telecom": "tau2_telecom", "telecom": "tau2_telecom",
}

DEFAULT_PROFILE = "webshop"


def get_profile(profile: DomainProfile | str | None) -> DomainProfile:
    if isinstance(profile, DomainProfile):
        return profile
    key = profile or DEFAULT_PROFILE
    key = ADAPTER_PROFILES.get(key, key)
    if key not in PROFILES:
        raise ValueError(
            f"unknown askbook profile {profile!r}; known: {sorted(PROFILES)}. "
            f"Every dataset needs its own profile -- add one rather than borrowing another's.")
    return PROFILES[key]


# back-compat for callers that still read the module-level WebShop taxonomy
DIMENSIONS = WEBSHOP.dimensions
_DIM_PATTERNS = WEBSHOP.patterns

_DECLINE = re.compile(
    r"(don'?t (really )?(understand|know)|not sure what you mean|either (one )?works|"
    r"i'?m not (too )?picky|no (strong )?preference|whatever you think|doesn'?t matter)",
    re.IGNORECASE)
_CONCRETE = re.compile(r"(\d|\byes\b|\bno\b|\bexactly\b|\bmust\b|\bneed\b|\bwant\b)", re.IGNORECASE)


def classify_question(text: str, profile: DomainProfile | str | None = None) -> str:
    """Which dimension a question targets, judged from its own words.

    With an explicit profile this returns OTHER when nothing matches; falling through to
    "feature" -- a WebShop dimension -- silently invented evidence for a dimension nobody
    had asked about. The no-profile default keeps that historical WebShop fall-through, so
    the original pipeline and the books it has already built stay exactly reproducible.
    """
    low = (text or "").lower()
    if profile is None:
        for dim, pat in _DIM_PATTERNS.items():
            if re.search(pat, low):
                return dim
        return "feature"
    return get_profile(profile).match(low)


_AUDIT = re.compile(
    r"(so far i have|here'?s what i have|my understanding|to confirm.*:|"
    r"anything (else )?(off|wrong|missing)|did i miss|have i got|is that right\?|"
    r"correct me if)", re.IGNORECASE)


def classify_style(text: str, profile: DomainProfile | str | None = None) -> str:
    """AUDIT (lay out the understanding, invite corrections) vs TARGETED (one thing).

    Which style pays is itself learnable: it is visible in the question's own words, and
    the user's reply says whether it worked.

    With an explicit profile the test is an explicit audit cue, or genuinely bundling
    several questions at once (two or more question marks). The no-profile WebShop default
    also fires on `>=2 commas and a "?"`, which is a proxy for LENGTH, not for style: on
    retail that labelled 89% of questions "audit" merely because they quoted comma-rich
    order lines, and the playbook then recommended the style it had mismeasured. The
    profile path therefore drops the comma rule; the default keeps it for reproducibility.
    """
    t = text or ""
    if profile is None:
        if _AUDIT.search(t) or (t.count(",") >= 2 and "?" in t):
            return "audit"
        return "targeted"
    return "audit" if (_AUDIT.search(t) or t.count("?") >= 2) else "targeted"


def query_signature(query: str, profile: DomainProfile | str | None = None) -> str:
    """A coarse, observable description of the request the agent is looking at.

    Only surface features of the query text: which dimensions it already speaks to, and
    how much detail it carries. Two requests with the same signature get the same prior.
    With an explicit profile, the profile name is part of the key so one dataset's contexts
    can never be looked up with another's; the no-profile default keeps the original
    unprefixed WebShop form, which is what existing WebShop books are keyed by.
    """
    low = (query or "").lower()
    words = len(low.split())
    size = "short" if words < 20 else ("medium" if words < 40 else "long")
    if profile is None:
        said = [d for d, pat in _DIM_PATTERNS.items() if re.search(pat, low)]
        return f"{size}|" + (",".join(sorted(said)) if said else "bare")
    prof = get_profile(profile)
    said = prof.mentioned(low)
    return f"{prof.name}|{size}|" + (",".join(sorted(said)) if said else "bare")


_STOP = frozenset("""about actually already alright also always another anything around because
been before being between both bring could didn't doesn't don't down each either else even
ever every everything from give going gonna have having here honestly however i'd i'll i'm
into it's just kind know like little look looking maybe mean might more most much must need
needs never nothing only other over prefer probably quite rather really right same should
some something sort still such sure take than that that's their them then there these they
thing things think this those though thought through very want wanted well what whatever
when where which while will with without would yeah your you're""".split())
_WORD = re.compile(r"[a-z0-9'&$.-]+")


def _content_words(text: str) -> set[str]:
    return {w for w in _WORD.findall((text or "").lower())
            if len(w) >= 4 and w not in _STOP} | {
        w for w in _WORD.findall((text or "").lower()) if any(c.isdigit() for c in w)}


def _informative(reply: str, seen: str = "") -> bool:
    """Did the reply add content the agent had NOT already seen?

    Length and enthusiasm are not information: a user who restates the request in more
    words has taught nothing. What counts is new content words -- a value, a constraint, a
    correction -- that appear in the reply and nowhere in the request or the earlier
    replies. Purely observable, and it separates a real answer from "either works for me".
    """
    r = (reply or "").strip()
    if len(r) < 12 or _DECLINE.search(r):
        return False
    return len(_content_words(r) - _content_words(seen)) >= 2


def observable_view(traj: dict) -> dict:
    """Project a trajectory onto ONLY what the agent could see. The gate that keeps the
    knowledge base honest: nothing outside this view reaches the statistics."""
    turns = []
    for rec in traj.get("turns", []):
        kind = (rec.get("action") or {}).get("kind")
        if kind == "ASK":
            turns.append({"kind": "ASK",
                          "question": (rec.get("action") or {}).get("question") or "",
                          "reply": rec.get("reply") or ""})
        elif kind == "PROPOSE":
            turns.append({"kind": "PROPOSE",
                          "accepted": bool((rec.get("acceptance") or {}).get("ok")),
                          "reaction": rec.get("reply") or ""})
    return {"query": (traj.get("header") or {}).get("query") or "",
            "outcome": traj.get("outcome"), "turns": turns}


def observe(view: dict, profile: DomainProfile | str | None = None) -> list[dict]:
    """One record per question asked, plus the episode's own outcome."""
    prof = None if profile is None else get_profile(profile)
    out = []
    sig = query_signature(view["query"], prof)
    won = view.get("outcome") == "SUCCESS"
    seen = view["query"]
    for t in view["turns"]:
        if t["kind"] != "ASK":
            continue
        out.append({"sig": sig, "dim": classify_question(t["question"], prof),
                    "style": classify_style(t["question"], prof),
                    "informative": _informative(t["reply"], seen), "won": won,
                    "question": t["question"][:160]})
        seen += " " + (t["reply"] or "")
    # rejections are experience too: what the user corrected is what to ask about
    for t in view["turns"]:
        if t["kind"] == "PROPOSE" and not t["accepted"] and t.get("reaction"):
            out.append({"sig": sig, "dim": classify_question(t["reaction"], prof),
                        "informative": True, "won": won, "question": None,
                        "from_rejection": True})
    return out


def distill(records: list[dict], *, min_n: int = 4,
            profile: DomainProfile | str | None = None) -> dict:
    """Aggregate records into per-signature dimension priors plus exemplar questions.

    With an explicit profile the book is stamped version 3 and records the profile it was
    built under; the no-profile default writes the original version-2 WebShop book, with
    no `profile` field, exactly as the pre-profile pipeline did.
    """
    prof = None if profile is None else get_profile(profile)
    ctx: dict[str, dict] = {}
    glob: dict[str, dict] = {}
    styles: dict[str, dict] = {}

    def bump(bucket: dict, dim: str, rec: dict) -> None:
        cell = bucket.setdefault(dim, {"asked": 0, "hit": 0, "won": 0, "corrected": 0,
                                       "examples": []})
        if rec.get("from_rejection"):
            # not a question: the user CORRECTED this dimension after committing, which
            # is evidence about what to ask, not evidence about a question's yield
            cell["corrected"] += 1
            return
        cell["asked"] += 1
        cell["hit"] += int(bool(rec["informative"]))
        cell["won"] += int(bool(rec["won"]))
        q = (rec.get("question") or "").strip()
        if (q and rec["informative"] and q.endswith("?") and len(q) <= 160
                and len(cell["examples"]) < 3 and q not in cell["examples"]):
            cell["examples"].append(q)

    for r in records:
        bump(ctx.setdefault(r["sig"], {}), r["dim"], r)
        bump(glob, r["dim"], r)
        if r.get("style"):
            bump(styles, r["style"], r)
    if prof is None:
        return {"version": 2, "n_records": len(records), "min_n": min_n,
                "contexts": ctx, "global": glob, "styles": styles}
    return {"version": 3, "profile": prof.name, "n_records": len(records), "min_n": min_n,
            "contexts": ctx, "global": glob, "styles": styles}


def _rank(bucket: dict, min_n: int) -> list[tuple[str, float, int]]:
    """Dimensions ranked by how often a question about them got a real answer. A cell with
    too little evidence is left out rather than ranked on noise. OTHER is never ranked:
    "ask about other" is not an instruction anyone can follow."""
    rows = []
    for dim, cell in bucket.items():
        if dim == OTHER or cell["asked"] < min_n:
            continue
        rows.append((dim, cell["hit"] / cell["asked"], cell["asked"]))
    rows.sort(key=lambda r: (-r[1], -r[2]))
    return rows


def _corrections(bucket: dict, min_n: int) -> list[tuple[str, int]]:
    """What users turned out to correct AFTER committing -- the requirements the agent
    had wrong and could have asked about."""
    rows = [(dim, cell.get("corrected", 0)) for dim, cell in bucket.items()
            if dim != OTHER and cell.get("corrected", 0) >= min_n]
    rows.sort(key=lambda r: -r[1])
    return rows


def book_profile(kb: dict | None) -> str | None:
    return (kb or {}).get("profile")


def check_profile(kb: dict | None, profile: DomainProfile | str | None) -> DomainProfile:
    """A book may only be rendered under the profile it was built with.

    This is the guard that makes "every dataset has its own playbook" a property of the
    code rather than a convention. A WebShop book served to a retail run is now a loud
    failure at load time instead of silently-wrong advice in every prompt.
    """
    prof = get_profile(profile)
    if not kb:
        return prof
    got = kb.get("profile")
    if got is None:
        raise ValueError(
            "askbook has no `profile` field: it was built by the pre-2026-08-20 code under "
            "the hard-coded WebShop taxonomy. Rebuild it for this dataset "
            f"(expected profile {prof.name!r}).")
    if got != prof.name:
        raise ValueError(
            f"askbook was built for profile {got!r} but this run is {prof.name!r}. "
            f"Playbooks are domain knowledge and are not transferable -- build "
            f"{prof.name!r} its own book.")
    return prof


def render_playbook(kb: dict | None, query: str, *, top: int = 3,
                    profile: DomainProfile | str | None = None) -> str:
    """The learned prior, as one compact block of prompt text.

    With an explicit profile the book must have been built under that profile
    (`check_profile` is a hard error otherwise); the no-profile default renders the
    original WebShop book with the original WebShop wording, unguarded, as the
    pre-profile pipeline did.
    """
    if not kb:
        return ""
    legacy = profile is None
    prof = WEBSHOP if legacy else check_profile(kb, profile)
    sig = query_signature(query, None if legacy else prof)
    bucket = (kb.get("contexts") or {}).get(sig) or {}
    scope = "requests like this one"
    rows = _rank(bucket, kb.get("min_n", 4))
    if len(rows) < 2:
        bucket = kb.get("global") or {}
        rows = _rank(bucket, kb.get("min_n", 4))
        scope = prof.scope_noun
    if not rows:
        return ""
    if legacy:
        said = set(sig.split("|", 1)[1].split(",")) if "|" in sig else set()
    else:
        said = set(sig.split("|", 2)[2].split(",")) if sig.count("|") >= 2 else set()
    silent = [d for d in prof.dimensions if d not in said]
    lines = [f"\nWHAT EXPERIENCE SAYS TO ASK ABOUT (learned from earlier {prof.actors}, not "
             f"from this one):"]
    if silent:
        ranked_silent = [d for d, _r, _n in rows if d in silent] or silent
        if legacy:
            # the original WebShop wording, byte-for-byte
            lines.append(f"  This request says nothing about: {', '.join(ranked_silent[:3])}. "
                         f"A requirement a shopper never mentions is the one most likely to be "
                         f"hidden -- put those in your audit before you buy.")
        else:
            lines.append(f"  This request says nothing about: {', '.join(ranked_silent[:3])}. "
                         f"A requirement a {prof.actor} never mentions is the one most likely to "
                         f"be hidden -- settle those before you {prof.commit}.")
    for dim, rate, n in rows[:top]:
        verdict = ("usually pays off" if rate >= 0.6 else
                   "sometimes pays off" if rate >= 0.35 else "rarely pays off")
        lines.append(f"  - {dim}: {verdict} ({int(round(rate * 100))}% of questions about it "
                     f"got a real answer in {scope}, n={n})")
    corr = _corrections(bucket, max(kb.get("min_n", 4), 3))
    if corr:
        total = sum(c for _d, c in corr)
        worst = ", ".join(f"{d} ({int(round(100 * c / total))}%)" for d, c in corr[:3])
        lines.append(f"  What {prof.actors} turned out to CORRECT after a {prof.commit_noun}, "
                     f"in {scope}: {worst}. Those are the requirements agents get wrong -- "
                     f"settle them before you {prof.commit}, not after.")
    weak = [d for d, rate, n in rows if rate < 0.35 and n >= kb.get("min_n", 4)]
    if weak:
        lines.append(f"  Do not spend questions on: {', '.join(weak)} -- {prof.actors} rarely "
                     f"add anything useful there.")
    ex = []
    for dim, _rate, _n in rows[:top]:
        ex += (bucket.get(dim, {}) or {}).get("examples", [])[:1]
    if ex:
        lines.append("  Question wordings that worked before:")
        lines += [f"    \"{e}\"" for e in ex[:3]]
    st = kb.get("styles") or {}
    ranked = [(name, cell["hit"] / max(cell["asked"], 1), cell["asked"])
              for name, cell in st.items() if cell["asked"] >= kb.get("min_n", 4)]
    if len(ranked) >= 2:
        ranked.sort(key=lambda r: -r[1])
        best, brate, bn = ranked[0]
        worst, wrate, _wn = ranked[-1]
        if brate - wrate >= 0.05:
            how = ("laying out everything you have and asking what is wrong or missing"
                   if best == "audit" else "asking about one specific thing at a time")
            lines.append(f"  Style that works better: {best.upper()} questions -- {how} "
                         f"({int(round(brate * 100))}% got a real answer against "
                         f"{int(round(wrate * 100))}% for {worst}, n={bn}).")
    lines.append(f"  This is a prior, not a script: if THIS {prof.actor} already told you "
                 f"something, never ask it again.")
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------- storage
def load(path: str | Path | None) -> dict | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def save(kb: dict, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(kb, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(p)


def distill_dir(run_dirs: list[str | Path], *, min_n: int = 4,
                profile: DomainProfile | str | None = None) -> dict:
    """Learn from every finished episode in these directories, through the observable view."""
    prof = None if profile is None else get_profile(profile)
    records: list[dict] = []
    for d in run_dirs:
        for f in Path(d).rglob("*.json"):
            try:
                traj = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if traj.get("outcome") in (None, "ERROR"):
                continue
            records += observe(observable_view(traj), prof)
    return distill(records, min_n=min_n, profile=prof)
