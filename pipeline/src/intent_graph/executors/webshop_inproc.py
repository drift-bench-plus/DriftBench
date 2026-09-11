"""WebShop executor: the benchmark's own reward function, in-process.

WebShop's ``engine.py`` imports pyserini (Lucene, hence a JVM) at module level purely to
build the *search* index, which ground-truth computation never touches.  Java is not
available here, so we stub that import -- and only that import -- before loading the
module.  ``get_reward`` itself is used exactly as shipped (north-star #2).

Two caches make an otherwise minute-scale scan tractable:
  * spaCy parses, memoized by string (``get_type_reward`` re-parses the same product
    names for every candidate goal)
  * ``get_type_reward`` results, memoized by (asin, goal frame)
Both are pure functions of their inputs, so caching cannot change a reward -- there is a
unit test that asserts cached and uncached rewards agree.
"""

from __future__ import annotations

import logging
import re
import sys
import threading
import types
from pathlib import Path

log = logging.getLogger(__name__)


def _install_stubs() -> None:
    """Satisfy engine.py's module-level imports that ground truth does not need."""
    for name in ("pyserini", "pyserini.search", "pyserini.search.lucene"):
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
    lucene = sys.modules["pyserini.search.lucene"]
    if not hasattr(lucene, "LuceneSearcher"):
        class LuceneSearcher:  # pragma: no cover - never constructed
            def __init__(self, *a, **k):
                raise RuntimeError("search index unavailable; ground truth does not use it")
        lucene.LuceneSearcher = LuceneSearcher
    sys.modules["pyserini"].search = sys.modules["pyserini.search"]
    sys.modules["pyserini.search"].lucene = lucene

    for name in ("cleantext", "rank_bm25"):
        if name not in sys.modules:
            mod = types.ModuleType(name)
            if name == "rank_bm25":
                mod.BM25Okapi = object
            sys.modules[name] = mod

    # TORCH MUST NOT LOAD HERE (2026-08-20). WebShop's modules import torch only for the
    # image-feature path, which ground truth never touches. But torch initialises OpenMP
    # thread pools at import, and those do not survive fork() on macOS: every forked worker
    # died from a signal with no Python traceback, so the run reported only
    # `failed_workers: N`. This surfaced the day torch arrived as a pyserini dependency --
    # installing a package silently broke the fork path of code that had not changed.
    # Stubbing it restores the state every prior campaign actually ran under.
    if "torch" not in sys.modules:
        torch_stub = types.ModuleType("torch")
        torch_stub.load = lambda *a, **k: {}      # FEAT_CONV / FEAT_IDS are never used
        sys.modules["torch"] = torch_stub


class WebShopSession:
    """Read-only: the catalog cannot be mutated, so nothing can leak between graphs."""

    def __init__(self, executor: WebShopExecutor, env_spec: dict) -> None:
        self.executor = executor
        self.env_spec = env_spec

    def run(self, recipe, *, mutating: bool = False):
        """Score a goal spec, or -- when handed a command instead -- search the catalogue.

        WebShop's own affordance is search-and-click, and without it an agent has no way to
        discover an ASIN: PROPOSE can never name a real product, so the only rational
        behaviour left is to keep asking until patience runs out. That is exactly what every
        arm did before this existed.
        """
        if isinstance(recipe, dict) and "goal" not in recipe:
            query = recipe.get("command") or recipe.get("sql") or ""
            return self.executor.search(str(query), self.env_spec)
        return self.executor.evaluate(recipe, self.env_spec)

    def rematerialize(self) -> None:
        pass

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class WebShopExecutor:
    name = "webshop_inproc"

    def __init__(self, config: dict) -> None:
        self.repo = Path(config["paths"]["webshop_repo"])
        self.derived = Path(config["paths"]["webshop_derived"])
        self.clusters_dir = self.derived / "clusters"
        self.seed = int(config.get("seed", 42))
        # Raised for threaded runs: with T threads on different clusters a cap of 2 would
        # thrash, reloading a 7k-product cluster every time another thread interleaved.
        self.max_resident_clusters = int(
            (config.get("webshop") or {}).get("max_resident_clusters", 2))
        self._loaded = False
        self._reward = None
        self._attrs: dict | None = None
        self._nlp_cache: dict[str, object] = {}
        self._type_cache: dict[tuple, dict] = {}
        self.products: dict[str, dict] = {}
        self.by_query: dict[str, list[str]] = {}
        self._order: list[str] = []
        # load_cluster/_evict mutate self.products, self.by_query, self._order and
        # self._type_cache. The experiment runner puts several episodes in one process on
        # separate threads, so those mutations must be serialised or one thread can evict a
        # cluster another is mid-scan over -- silently scoring against a partial pool.
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ loading
    def _load(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            self._load_locked()

    def _load_locked(self) -> None:
        _install_stubs()
        sys.path.insert(0, str(self.repo))
        import web_agent_site.engine.goal as goalmod

        # memoize spaCy: the same product names are parsed for every candidate goal
        original_nlp = goalmod.nlp

        def cached_nlp(text):
            if text not in self._nlp_cache:
                self._nlp_cache[text] = original_nlp(text)
            return self._nlp_cache[text]

        goalmod.nlp = cached_nlp

        original_type = goalmod.get_type_reward

        def cached_type(product, goal):
            key = (product["asin"], goal.get("name"), goal.get("query"),
                   goal.get("product_category"))
            if key not in self._type_cache:
                self._type_cache[key] = original_type(product, goal)
            return self._type_cache[key]

        goalmod.get_type_reward = cached_type
        self._reward = goalmod.get_reward
        self._loaded = True

    def _attributes(self) -> dict:
        """items_ins_v2.json, keyed by asin — covers all 1,181,436 catalog products."""
        if self._attrs is None:
            import json
            path = self.repo / "data" / "items_ins_v2.json"
            self._attrs = json.loads(path.read_text(encoding="utf-8"))
        return self._attrs

    def preload(self) -> None:
        """Force every lazy global load NOW, so a fork-after-load parent shares it.

        Without this, _load() and the 1,181,436-product attributes JSON load lazily at
        first use -- i.e. AFTER the fork, once per child (10 children x ~1.5G, observed
        as 15G and minutes of parallel JSON parsing before the first episode). Cluster
        shards stay lazy on purpose: they are per-episode, small, and evicted.
        """
        self._load()
        self._attributes()

    def load_cluster(self, cluster: str) -> list[str]:
        """Every product sharing this scrape query — the plan's candidate pool.

        Loaded on demand from `clusters/<slug>.jsonl` and evicted, because the full set is
        2.8GB: keeping 308 clusters resident would cost more memory than the machine has,
        while one cluster is at most a few thousand products.

        A missing shard is a hard error on purpose. The defect this replaced was a silent
        fallback to the human-instruction subset, which quietly computed every ground truth
        over ~0.8% of the cluster.
        """
        key = (cluster or "").lower().strip()
        with self._lock:
            if key in self.by_query:
                self._touch(key)
                return self.by_query[key]
        return self._load_cluster_locked(key)

    def _load_cluster_locked(self, key: str) -> list[str]:
        import json

        path = self.clusters_dir / f"{_slug(key)}.jsonl"
        if not path.exists():
            raise FileNotFoundError(
                f"no cluster shard for {key!r} at {path}. Run "
                f"webshop/extract_full_clusters.py to build them."
            )
        attributes = self._attributes()
        asins: list[str] = []
        records = []
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                records.append(json.loads(line))
        with self._lock:
            if key in self.by_query:            # another thread won the race; reuse its work
                self._touch(key)
                return self.by_query[key]
            for p in records:
                asin = p["asin"]
                sd = p.get("small_description") or []
                self.products[asin] = {
                    "asin": asin,
                    "query": (p.get("query") or "").lower().strip(),
                    "product_category": p.get("product_category") or "",
                    "name": p.get("name") or "",
                    "Title": p.get("name") or "",
                    "Description": p.get("full_description") or "",
                    "BulletPoints": sd if isinstance(sd, list) else [sd],
                    "Attributes": (attributes.get(asin, {}) or {}).get("attributes")
                    or ["DUMMY_ATTR"],
                    "options": _option_values(p.get("customization_options")),
                    "price": _price_of(p.get("pricing")),
                }
                asins.append(asin)
            self.by_query[key] = asins
            self._touch(key)
            self._evict()
        log.debug("loaded cluster %r: %d products", key, len(asins))
        return asins

    def snapshot_cluster(self, cluster: str) -> list:
        """(asin, product) pairs safe to scan without the lock.

        Readers used to iterate `self.products` directly, and eviction on another thread
        popped entries mid-scan: KeyError in `evaluate`, and `_evict`'s cache rebuild raced
        concurrent inserts (RuntimeError: dict changed size during iteration). Product dicts
        are never mutated in place -- eviction only unlinks them -- so holding references in
        a snapshot stays valid even if the cluster is evicted mid-scan.
        """
        for _ in range(3):
            asins = self.load_cluster(cluster)
            with self._lock:
                try:
                    return [(a, self.products[a]) for a in asins]
                except KeyError:
                    continue          # evicted between load and snapshot; reload
        raise RuntimeError(f"cluster {cluster!r} kept being evicted; raise "
                           f"webshop.max_resident_clusters above the thread count")

    def get_product(self, asin: str, cluster: str) -> dict | None:
        """One product, eviction-safe."""
        with self._lock:
            p = self.products.get(asin)
            if p is not None:
                return p
        self.load_cluster(cluster)
        with self._lock:
            return self.products.get(asin)

    def _touch(self, key: str) -> None:
        if key in self._order:
            self._order.remove(key)
        self._order.append(key)

    def _evict(self) -> None:
        """Drop least-recently-used clusters, and the product records only they held."""
        while len(self._order) > self.max_resident_clusters:
            old = self._order.pop(0)
            for asin in self.by_query.pop(old, ()):
                self.products.pop(asin, None)
            self._type_cache = {k: v for k, v in list(self._type_cache.items())
                                if k[0] in self.products}

    # ------------------------------------------------------------------ scoring
    def evaluate(self, recipe: dict, env_spec: dict) -> list[tuple[str, dict]]:
        """Every purchase in the cluster that scores maximally under WebShop's reward.

        The pool is every product sharing the scrape query, per `load_cluster`.
        """
        self._load()
        goal = recipe["goal"]
        hits = []
        for asin, product in self.snapshot_cluster(env_spec["cluster"]):
            price = product["price"]
            if goal["price_upper"] and price > goal["price_upper"]:
                continue
            options = _best_options(product["options"], goal["goal_options"])
            if self._reward(product, goal, price, options) >= 0.999:
                hits.append((asin, options))
        return hits

    # ------------------------------------------------------------------ search
    def search(self, command: str, env_spec: dict, *, limit: int = 12) -> str:
        """Top matching products in this cluster, as the store would list them.

        Token-overlap ranking, no model involved -- this is an environment affordance, not a
        judgement, and it must stay deterministic so an episode replays identically. Reveals
        only what a shopper browsing a results page would see.
        """
        self._load()
        query = re.sub(r"^\s*(search|click)\s*\[?|\]?\s*$", "", str(command),
                       flags=re.IGNORECASE).strip().lower()
        pairs = self.snapshot_cluster(env_spec["cluster"])
        if not query:
            return "Type a search query, e.g. search[machine washable socks]."
        terms = {t for t in re.split(r"[^a-z0-9]+", query) if len(t) > 2}
        if not terms:
            return "Type a search query, e.g. search[machine washable socks]."
        # Rank by the FRACTION of query terms matched, and require most of them. Counting any
        # single overlap reported "4414 matches" in a 5,000-product cluster -- in a
        # hair-products cluster the word "hair" matches nearly everything -- so the results
        # were noise and no agent could pick from them.
        scored = []
        by_asin = dict(pairs)
        for asin, p in pairs:
            attrs = [a for a in (p.get("Attributes") or []) if a != "DUMMY_ATTR"]
            hay = f"{p['name']} {p['product_category']} {' '.join(attrs)}".lower()
            hits = sum(1 for t in terms if t in hay)
            if hits >= max(1, (len(terms) + 1) // 2):
                scored.append((-hits / len(terms), p["price"], asin))
        scored.sort()
        if not scored:
            return (f"No product matches most of {sorted(terms)}. Search fewer or different "
                    f"words -- try the single most important feature.")
        lines = [f"{len(scored)} products match most of those words; best {min(limit, len(scored))}:"]
        for frac, price, asin in scored[:limit]:
            p = by_asin[asin]
            # Attributes are what the store's product page lists, and what deciding between
            # candidates actually requires. Without them an agent can see names only and can
            # never tell which product HAS the feature the shopper asked for.
            attrs = [a for a in (p.get("Attributes") or []) if a != "DUMMY_ATTR"][:6]
            opts = {k: v[:4] for k, v in (p["options"] or {}).items()}
            lines.append(f"  {asin} | ${price:.2f} | match {-frac:.0%} | {p['name'][:64]}")
            if attrs:
                lines.append(f"      features: {', '.join(attrs)}")
            if opts:
                lines.append(f"      options: {opts}")
        return "\n".join(lines)

    def open(self, env_spec: dict) -> WebShopSession:
        self._load()
        return WebShopSession(self, env_spec)

    def shutdown(self) -> None:
        pass


def _slug(query: str) -> str:
    """Filename for a cluster shard. Must match webshop/extract_full_clusters.py:slug."""
    s = re.sub(r"[^a-z0-9]+", "_", (query or "").lower().strip()).strip("_")
    return s or "_blank"


def _price_of(pricing) -> float:
    nums = re.findall(r"[\d.]+", str(pricing or ""))
    return float(nums[0]) if nums else 100.0


def _option_values(customization_options) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for name, contents in (customization_options or {}).items():
        if not contents:
            continue
        vals = [c["value"].strip().replace("/", " | ").lower()
                for c in contents if isinstance(c, dict) and c.get("value")]
        if vals:
            out[name.lower()] = vals
    return out


def _best_options(available: dict[str, list[str]], goal_options) -> dict[str, str]:
    """Pick, for each requested option, the closest value the product offers."""
    from thefuzz import fuzz

    chosen: dict[str, str] = {}
    for want in goal_options:
        best, best_score = None, -1
        for name, values in available.items():
            for value in values:
                score = fuzz.token_set_ratio(value, want)
                if score > best_score:
                    best, best_score = (name, value), score
        if best:
            chosen[best[0]] = best[1]
    return chosen
