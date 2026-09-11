"""WebShop adapter.

The recipe here is not a program but a *goal spec* -- the structured shopping list
WebShop's reward function is a pure function of.  Editing the spec and re-scoring the
catalog is the same move as editing SQL and re-running it.

Conditions:
    attr:<text>     a required product attribute   ("machine wash")
    option:<i>      a requested option value       ("heather charcoal", "small")
    price_upper     the price ceiling

Environment: WebShop has exactly one world -- the whole catalog is searchable at all
times.  The scrape query ("Men's Shorts") is therefore not a separate environment but a
topical cluster, used to keep pivots from jumping to an unrelated product category (D6).

Determinism: WebShop samples ``price_upper`` at goal-construction time and prices for
ranged products with ``random.uniform``.  We do not touch the reward function; we derive
the price ceiling deterministically from the target product's own price instead of
sampling, so the same config always produces the same ground truth.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path

from ..executors.webshop_inproc import WebShopExecutor
from ..models import Condition, GroundTruth, Seed
from . import base as registry

# The sentinel the live executor uses for `click[Buy Now]`: what was purchased is a
# property of the browsing session, resolved during acceptance.
FROM_ENV = "__FROM_ENV__"

log = logging.getLogger(__name__)

PRICE_RANGE = [10.0 * i for i in range(1, 100)]


def _is_speakable_attribute(value) -> bool:
    """Would a shopper plausibly say this as a product feature?

    Many WebShop attributes are junk tokens -- "1", "#1", "5x3fu", bare measurements. They
    are fine as data but useless as a FALSE PREMISE: "I need the product feature '1'" does not
    read as a person who is mistaken, it reads as corrupted text. A false premise only tests
    anything if it is believable enough to be worth correcting.
    """
    if not value or value == "DUMMY_ATTR":
        return False
    text = str(value).strip()
    if len(text) < 5 or len(text) > 40:
        return False
    words = [w for w in re.split(r"[\s/_-]+", text) if w]
    alpha = [w for w in words if len(w) >= 3 and w.isalpha()]
    if not alpha:
        return False
    # mostly-digit strings ("10 inches", "5x3fu") are measurements or codes, not features
    digits = sum(c.isdigit() for c in text)
    return digits / len(text) <= 0.15


def deterministic_price_upper(price: float) -> float:
    """A ceiling above the target's price, chosen without RNG.

    WebShop picks one at random from the next few 10-dollar steps; we always take the
    second step so a seed's ground truth never depends on interpreter state.
    """
    steps = [p for p in PRICE_RANGE if p > price][:4]
    if len(steps) < 2:
        return 1_000_000.0
    return steps[1]


class WebShopAdapter:
    name = "webshop"
    version = "1"
    executor_name = "webshop_inproc"

    def __init__(self, config: dict) -> None:
        self.repo = Path(config["paths"]["webshop_repo"])
        self.config = config
        self._clusters: dict[str, dict] = {}
        self._prices: dict[str, float] = {}

    # ------------------------------------------------------------------ loading
    def load(self) -> list[Seed]:
        from ..executors.webshop_inproc import _price_of

        human = json.loads((self.repo / "data" / "items_human_ins.json").read_text(encoding="utf-8"))
        products = {}
        for line in (Path(self.config["paths"]["webshop_derived"]) /
                     "human_ins_products_full.jsonl").open(encoding="utf-8"):
            p = json.loads(line)
            products[p["asin"]] = p

        # Iterate in WebShop's own goal order rather than dict order, so every seed carries
        # its official goal_idx and the train/eval/test split can be honoured. Without this
        # the adapter cannot tell a test goal from a training goal, and anything generated
        # silently mixes the two. See webshop/build_goal_index.py.
        wcfg = self.config.get("webshop") or {}
        root_splits = set(wcfg.get("root_splits") or ["test"])
        retrieval_splits = set(wcfg.get("retrieval_splits") or ["test", "eval"])
        goals = self._goal_index()

        seeds, seen = [], set()
        for goal_idx, asin, entry_pos, split in goals:
            if split not in retrieval_splits and split not in root_splits:
                continue
            entries = human.get(asin) or []
            if entry_pos >= len(entries):
                continue
            entry = entries[entry_pos]
            product = products.get(asin)
            if product is None:
                continue
            cluster = (product.get("query") or "").lower().strip()
            if not cluster:
                continue
            price = _price_of(product.get("pricing"))
            self._prices[asin] = price
            env_key = cluster
            self._clusters[env_key] = {
                "env_id": "webshop_" + hashlib.sha256(cluster.encode()).hexdigest()[:16],
                "kind": "webshop",
                "cluster": cluster,
            }
            attrs = entry.get("instruction_attributes") or []
            if not attrs:
                continue  # goal.py skips these too, and they consume no goal_idx
            options = entry.get("instruction_options") or []
            conds = [(f"attr:{a}", "=", a) for a in attrs]
            conds += [(f"option:{i}", "=", o) for i, o in enumerate(options)]
            conds.append(("price_upper", "<=", deterministic_price_upper(price)))
            rid = hashlib.sha256(
                (asin + "||" + entry.get("instruction", "")).encode()
            ).hexdigest()[:16]
            if rid in seen:
                continue
            seen.add(rid)
            seeds.append(Seed(
                record_id=rid, env_key=env_key,
                base={"asin": asin, "cluster": cluster,
                      "name": product.get("name") or "",
                      "product_category": product.get("product_category") or ""},
                conditions=tuple(conds),
                shipped_answer=asin,     # the product the instruction was written for
                meta={"instruction": entry.get("instruction", ""),
                      "goal_idx": int(goal_idx), "split": split},
                root_eligible=split in root_splits,
            ))
        return seeds

    def _goal_index(self) -> list[tuple[int, str, int, str]]:
        """WebShop's official goal ordering: (goal_idx, asin, entry_pos, split)."""
        import pandas as pd

        path = Path(self.config["paths"]["webshop_derived"]) / "goal_index.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing. Run webshop/build_goal_index.py -- without it the "
                f"official train/eval/test split cannot be honoured."
            )
        df = pd.read_parquet(path, columns=["goal_idx", "asin", "entry_pos", "split"])
        df = df.sort_values("goal_idx")
        return list(df.itertuples(index=False, name=None))

    # -------------------------------------------------------- user-facing text
    @staticmethod
    def slot_encodes_value(slot: str) -> bool:
        """Does naming this slot give its value away?

        On WebShop an attribute slot is `attr:<the attribute>` -- the name IS the value. So a
        strategy that says "mention this requirement but stay vague about it" cannot be
        applied to one: mentioning it reveals exactly what the mask claims is unbound, and
        the literal reading (which drops the slot) would then be wrong about what the query
        conveys. Options and the price limit are safe: "the size" and "my budget" name the
        dimension without naming the choice.
        """
        return slot.startswith("attr:")

    @staticmethod
    def render_context(base: dict) -> str:
        """Topical context the renderer may see, with no answer in it.

        Without this the model has no idea what is being shopped for and invents a category:
        a tonic-water intent came out as "reusable stainless steel drinking straws". That is
        false content the MASK does not record, so the literal reading stops describing the
        query and the signature check measures the wrong thing.

        The cluster is safe to reveal: thousands of products share it, `describe_pivot`
        already announces it to the user, and it is the search scope rather than the answer.
        The asin, product name and full category path stay hidden.
        """
        cluster = (base or {}).get("cluster")
        return f"The user is shopping in the \"{cluster}\" section." if cluster else ""

    @staticmethod
    def slot_phrase(slot: str) -> str:
        """How a shopper would refer to this requirement, without internal syntax."""
        if slot == "price_upper":
            return "the most I want to spend"
        if slot.startswith("option:"):
            return "the product option (size, flavour, colour and so on)"
        if slot.startswith("attr:"):
            # value-free on purpose: this phrase is also used to say "do not mention it",
            # and an attribute slot's name is its value
            return "a product feature you require"
        return slot

    def env_spec(self, env_key: str) -> dict:
        return self._clusters[env_key]

    # ------------------------------------------------------------- the contract
    def compile(self, base: dict, conditions: tuple[Condition, ...]):
        attrs = [v for s, _, v in conditions if s.startswith("attr:")]
        options = [v for s, _, v in sorted(conditions) if s.startswith("option:")]
        price = next((v for s, _, v in conditions if s == "price_upper"), 1_000_000.0)
        goal = {
            "asin": base["asin"],
            "category": "",
            "query": base["cluster"],
            "name": base["name"],
            "product_category": base["product_category"],
            "attributes": attrs,
            "goal_options": options,
            "price_upper": float(price),
        }
        return {"goal": goal, "cluster": base["cluster"]}

    def is_mutating(self, recipe) -> bool:
        return False

    def is_valid_intent(self, base: dict, conditions: tuple[Condition, ...]) -> bool:
        """WebShop goals must name at least one attribute.

        ``get_reward`` computes ``num_attr_matches / len(goal['attributes'])``, so an
        attribute-free goal raises ZeroDivisionError -- it is not a task the shipped
        verifier can score.  goal.py agrees: it skips human instructions with no
        attributes and asserts a non-empty list for synthetic ones.  So a relaxation
        that would remove the last attribute is not a legal intent, and we never
        propose it.
        """
        return any(s.startswith("attr:") for s, _, _ in conditions)

    def execute(self, recipe, session) -> GroundTruth:
        return GroundTruth.purchaseset(session.run(recipe))

    def validate(self, seed: Seed, session) -> bool:
        """The instruction's own target product must be a maximal-scoring purchase.

        If it is not, the human annotation and the reward function disagree and the
        record cannot anchor a graph.
        """
        recipe = self.compile(seed.base, seed.conditions)
        hits = session.run(recipe)
        return any(asin == seed.shipped_answer for asin, _ in hits)

    # ------------------------------------------------------------- acceptance
    def describe_pivot(self, base: dict) -> str:
        """Announce a pivot WITHOUT naming the target.

        `base` carries the asin and the full product name; repeating either would hand the
        agent the answer, so only the shared topical frame is mentioned.
        """
        cluster = (base or {}).get("cluster") or "this shop"
        return f"I've changed my mind, I need something else from the {cluster} instead."

    def act_is_mutating(self, raw_action: str) -> bool:
        """Only buying is irreversible; searching and viewing are free."""
        low = str(raw_action or "").strip().lower()
        if not low:
            return True
        return low.startswith("buy") or "buy now" in low or low.startswith("purchase")

    def is_proposal_command(self, command: str) -> bool:
        """Does this environment Operation actually constitute a PROPOSAL?

        Real WebShop makes buying an environment action, so models naturally emit
        ``buy <asin> {...}`` inside ``Action: Operation``.  That is a proposal in this
        adapter's surface syntax, and recognising it belongs here rather than in the
        episode loop -- the loop knows only *proposal* and *submission*, which are the
        terms that hold for every benchmark; ``buy`` is WebShop's spelling of one of them.
        """
        import re as _re
        cmd = str(command or "")
        # THE REAL BUTTON (2026-08-20). On the faithful site a purchase is `click[Buy Now]`
        # pressed on an item page after the options have been selected by clicking them.
        # The legacy `buy <asin> {...}` spelling is still accepted so replays of the old
        # trajectories parse, but it is no longer how an agent buys anything.
        if _re.match(r"\s*click\s*\[\s*buy\s*now\s*\]\s*$", cmd, _re.IGNORECASE):
            return True
        return bool(_re.match(r"\s*(buy|purchase)\b", cmd, _re.IGNORECASE))

    def parse_proposal(self, raw):
        import re as _re
        # `click[Buy Now]` names no product: WHAT is being bought is a property of the live
        # browsing session (which item page, which options were clicked), not of the command
        # text. Defer to acceptance, which holds the session and commits the purchase.
        if _re.match(r"\s*click\s*\[\s*buy\s*now\s*\]\s*$", str(raw or ""), _re.IGNORECASE):
            return (FROM_ENV, {})
        parsed = self._parse_proposal_impl(raw)
        if parsed is not None:
            return parsed
        # loose key: value pairs -- `buy B09X flavor name: lemon, size: 1 pack`. A real
        # store's buy form does not require JSON; rejecting this shape scored real purchases
        # as unparseable_proposal.
        m = re.match(r"\s*(?:buy|purchase)\s+([A-Za-z0-9]{6,})\s+(.+)", str(raw or ""),
                     re.IGNORECASE | re.DOTALL)
        if m:
            asin, rest = m.group(1), m.group(2)
            opts = {}
            for part in re.split(r"[,;]", rest):
                if ":" in part:
                    k, v = part.split(":", 1)
                    if k.strip() and v.strip():
                        opts[k.strip().lower()] = v.strip().lower()
            if opts:
                return asin, opts
        return None

    def _parse_proposal_impl(self, raw):
        """``buy <asin> {"size": "small"}`` or a ready-made ``(asin, options)`` pair."""
        if isinstance(raw, (tuple, list)) and len(raw) == 2:
            return (str(raw[0]), dict(raw[1]))
        text = str(raw).strip()
        text = re.sub(r"^(buy|purchase)\s+", "", text, flags=re.I)
        m = re.match(r"([A-Za-z0-9]+)\s*(\{.*\})?$", text, re.S)
        if not m:
            return None
        options = {}
        if m.group(2):
            try:
                options = {str(k).lower(): str(v).lower() for k, v in
                           json.loads(m.group(2)).items()}
            except Exception:
                options = {}
        return (m.group(1), options)

    def accepts(self, proposal, node, session, *, executor=None) -> tuple[bool, str]:
        """Score the agent's ACTUAL purchase with WebShop's own reward function.

        Not set membership: the stored options come from a fuzzy `_best_options` pick, so
        membership would reject an agent that bought the right product with a different but
        equally valid option set -- stricter than WebShop's own reward, and a change to how
        the benchmark evaluates.
        """
        parsed = proposal if isinstance(proposal, tuple) else self.parse_proposal(proposal)
        if not parsed:
            return False, "unparseable_proposal"
        asin, options = parsed
        # The agent pressed the real Buy Now. Commit it in the environment and read back
        # what WebShop actually recorded as purchased -- the asin of the item page it was
        # on, and the options it selected by clicking them. If the agent never opened an
        # item page there is nothing to buy, and that is a legitimate refusal.
        if asin == FROM_ENV:
            commit = getattr(session, "commit_purchase", None)
            bought = commit() if commit is not None else None
            if not bought:
                return False, "nothing_selected_to_buy"
            asin, options = bought
        ex = session.executor
        ex._load()
        product = ex.get_product(asin, node.base["cluster"])
        if product is None:
            return False, "unknown_asin"
        goal = node.recipe["goal"] if isinstance(node.recipe, dict) else None
        if goal is None:
            return False, "node_has_no_goal"
        price = product["price"]
        if goal["price_upper"] and price > goal["price_upper"]:
            return False, "over_price_limit"
        chosen = options or _best_options_for(product, goal)
        score = ex._reward(product, goal, price, chosen)
        ok = score >= 0.999
        return ok, f"reward={score:.3f}"

    def witness(self, base: dict, conditions: tuple[Condition, ...], session) -> list[dict]:
        """Attributes and price band of the products that currently qualify."""
        ex = session.executor
        hits = session.run(self.compile(base, conditions))
        rows = []
        for asin, _ in hits[:200]:
            product = ex.products[asin]
            row = {f"attr:{a}": a for a in product["Attributes"] if a != "DUMMY_ATTR"}
            row["price_band"] = _price_band(product["price"])
            rows.append(row)
        return rows

    def foreign_domains(self, slot: str, base: dict, session) -> list:
        """Real values for this slot drawn from OTHER product categories.

        These make the best false premises: very unlikely to hold in this cluster, yet real
        phrases, so the rendered query reads like a shopper who is simply mistaken rather
        than like corrupted text. Without them the FALSIFY strategies produced a sample on
        only 4 of 14 graphs, because WebShop's fuzzy reward means a value taken from THIS
        cluster usually still satisfies the intent.

        Costs no extra cluster load: `items_ins_v2.json` is already resident, and the local
        vocabulary comes from the cluster already in memory.
        """
        import itertools

        if not slot.startswith("attr:"):
            return []
        ex = session.executor
        pool = ex.load_cluster(base["cluster"])
        local = {a.lower() for asin in pool[:400]
                 for a in ex.products[asin]["Attributes"] if a != "DUMMY_ATTR"}
        # also exclude anything sharing a word with a local attribute: WebShop scores text
        # similarity, so a near-miss would still satisfy the intent and be rejected
        local_words = {w for a in local for w in a.split() if len(w) > 3}
        out: list[str] = []
        for rec in itertools.islice(ex._attributes().values(), 40_000):
            for a in (rec or {}).get("attributes") or []:
                if not _is_speakable_attribute(a):
                    continue
                low = a.lower()
                if low in local or any(w in local_words for w in low.split() if len(w) > 3):
                    continue
                out.append(a)
                if len(out) >= 60:
                    return sorted(set(out))
        return sorted(set(out))

    def domains(self, slot: str, base: dict, session) -> list:
        ex = session.executor
        pairs = ex.snapshot_cluster(base["cluster"])
        if slot.startswith("option:"):
            values: set[str] = set()
            for _asin, prod in pairs[:300]:
                for vals in prod["options"].values():
                    values.update(vals)
            return sorted(values)[:30]
        if slot == "price_upper":
            prices = sorted({prod["price"] for _a, prod in pairs})
            if not prices:
                return []
            return [p for p in PRICE_RANGE if prices[0] < p < prices[-1]][:10]
        return []


def _best_options_for(product, goal):
    from ..executors.webshop_inproc import _best_options
    return _best_options(product["options"], goal["goal_options"])


def _price_band(price: float) -> str:
    for limit in (10, 25, 50, 100, 250):
        if price < limit:
            return f"under_{limit}"
    return "over_250"


# THE LIVE EXECUTOR IS THE DEFAULT (2026-08-20). The inproc executor's browsing surface
# was a reimplementation that skipped WebShop's navigation entirely; it survives only as
# the base class supplying ground-truth scoring, which was always faithful. Registering the
# live one here means no run can accidentally fall back to the shortcut.
# Presence-gated: a distribution that does not ship the live executor (it is not part of
# every benchmark's install) registers the inproc base class instead, exactly as before.
try:
    from ..executors.webshop_live import WebShopLiveExecutor  # noqa: E402
except ImportError:
    registry.register("webshop", WebShopAdapter, WebShopExecutor)
else:
    registry.register("webshop", WebShopAdapter, WebShopLiveExecutor)
