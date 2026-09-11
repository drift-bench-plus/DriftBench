"""WebShop executor that drives the REAL WebShop site.

WHAT WAS WRONG BEFORE (2026-08-20). `webshop_inproc.WebShopExecutor` scored goals with
WebShop's shipped `get_reward` -- faithful -- but its *browsing* surface was a
reimplementation: `search` scanned a derived cluster shard with a local keyword matcher and
rendered a listing that already contained each product's features and OPTIONS. WebShop's
`click[...]` was never implemented at all. Consequences, measured over ~148k episodes:
agents issued 99.4% `search` and ZERO clicks, and could win by reading options off the
listing and emitting one `buy` -- the item page, the option buttons and the Buy Now button
were never touched. That is not WebShop; it is an easier environment wearing its name.

WHAT THIS DOES. Browsing is delegated verbatim to WebShop's own `SimServer`/`SimBrowser`
via the shared env server (`tools/webshop_server.py`): real BM25 over the Lucene index,
real results pages, real item pages, real option buttons, real Buy Now. No page rendering,
retrieval or ranking is reimplemented here.

WHAT IS ADAPTED, AND WHY IT IS ADAPTATION RATHER THAN A SHORTCUT. WebShop's Buy Now is
terminal; our research question needs an episode to survive a REJECTED purchase, because
the patience economy and the intent shifts are the object of study. So Buy Now is routed
to our proposal channel: the purchase really happens in the environment, we adjudicate it
against the episode's intent graph with WebShop's own `get_reward`, and on rejection the
session returns to the store -- the agent must NAVIGATE AGAIN to try something else
(ruling 2026-08-20, option (a)). Nothing the agent must be *able to do* is removed; only
who judges the purchase, and what happens after a refusal, differ.

Ground-truth scoring (`evaluate`) still goes through the inproc executor, unchanged.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
import time
import urllib.error
import urllib.request
import uuid

from ..errors import EnvTransportError
from .webshop_inproc import WebShopExecutor, WebShopSession as _InprocSession

log = logging.getLogger(__name__)

# The adapter emits this when the agent presses Buy Now: the purchased (asin, options) are
# a property of the LIVE SESSION, not of the command text, so acceptance resolves it by
# committing the purchase through the environment (see `accepts` in adapters/webshop.py).
FROM_ENV = "__FROM_ENV__"

# FORK-SAFE HTTP (macOS, 2026-08-20). `urllib.request.urlopen` calls `getproxies()`, which
# on macOS goes through `_scproxy` -> SystemConfiguration, an Objective-C framework that is
# NOT fork-safe: a forked worker dies with a signal and no Python traceback, so the run
# reports `failed_workers: N` and nothing else. The episode runner forks by design
# (fork-after-load), so every worker hit this. An opener with an explicit empty ProxyHandler
# never consults the system proxy and is safe across fork. Localhost needs no proxy anyway.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

# Transport retry budget (see LiveSession._post). Three attempts with a 0.4s/0.8s backoff
# costs at most ~1.2s on a doomed request while recovering the transient backlog rejections
# that were destroying up to 24% of a cell's episodes.
_POST_ATTEMPTS = 6
_POST_BACKOFF_S = 0.5



_BUY = re.compile(r"^\s*click\s*\[\s*buy\s*now\s*\]\s*$", re.I)
_PURCHASED = re.compile(r"asin\s*\[SEP\]\s*([A-Za-z0-9]+)\s*\[SEP\]\s*options\s*\[SEP\]\s*(\{.*?\})",
                        re.I | re.S)


def is_buy_now(command: str) -> bool:
    return bool(_BUY.match(str(command or "")))


class LiveSession(_InprocSession):
    """One WebShop browsing session, backed by the shared env server.

    Session state lives on the server (SimServer is session-scoped), so this object holds
    only an id. `run()` forwards environment actions; `evaluate` inherits from the inproc
    session so ground-truth computation is untouched.
    """

    def __init__(self, executor: "WebShopLiveExecutor", env_spec: dict, session_id: str,
                 instruction: str) -> None:
        super().__init__(executor, env_spec)
        # PIN THE SAMPLE TO ONE SERVER (ruling 2026-08-21: "for one sample, it's always in
        # one server"). Session state lives on the server that created it, so every call of
        # a session must go to one server -- but the pinning KEY is the sample's env_id,
        # not the session id. Same sample => same server, across methods, repeats and
        # retries. Two things follow:
        #   * any state drift between servers becomes a FIXED property of the sample, so
        #     paired cross-method deltas cancel it instead of inheriting it as noise;
        #   * repeats are reproducible -- a rerun of a sample replays against the same
        #     server state trajectory.
        # Session IDENTITY stays globally unique (pid+counter+uuid below): uniqueness is
        # what fixed the 400 "unknown session" collisions, and it must hold even when the
        # same sample runs concurrently in two method pools on the same server.
        # sha1, not hash(): PYTHONHASHSEED randomises hash() per process, which would pin
        # the same sample differently in different workers.
        pool = getattr(executor, "base_urls", None) or [executor.base_url]
        pin = str(env_spec.get("env_id") or env_spec.get("sample_id") or session_id)
        idx = int(hashlib.sha1(pin.encode()).hexdigest()[:8], 16) % len(pool)
        self.base = pool[idx]
        self.session_id = session_id
        self.instruction = instruction
        self.last_purchase: tuple[str, dict] | None = None
        self._done = False          # the env terminated (a purchase was committed)
        self._open = False
        # DELIBERATELY LAZY: ground-truth work (generation, `witness`, `evaluate`) opens
        # sessions but never browses, and it must not require the env server to be running.
        # The HTTP connection is made on the first actual browsing action.

    # ------------------------------------------------------------------ transport
    def _post(self, path: str, **body):
        """POST with bounded retry.

        WHY RETRY (2026-08-21). This had no retry, and a SINGLE transient transport failure
        was scored as if the environment had answered: `run()` turned it into an
        "error: environment unreachable" observation the agent then had to reason about, and
        `commit_purchase()` returned None, silently discarding a purchase the agent had
        navigated to correctly. Both corrupt the episode rather than delaying it.

        MEASURED contamination before this fix, over completed cells:
            B0 8.5% of episodes / A0 9.2% / A1 15.6% / A2 24.3%
        and episodes touched by one succeeded at 3-13% against 24-32% for clean ones. The
        rate tracked worker width, because the failure is the dev server's listen backlog
        overflowing under ~96 concurrent workers -- the errno is 60 (timed out) after the
        full timeout_s, i.e. a connection that never got served, not a slow page render.
        Since it is a QUEUEING failure it clears on its own in milliseconds, so a short
        backoff recovers what was previously a lost episode.

        Bounded deliberately: retrying forever would park a worker on a genuinely dead
        server and stall the cell. After the last attempt the caller still sees the
        exception and keeps its original error handling.
        """
        req = urllib.request.Request(
            self.base + path, data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        last: Exception | None = None
        for attempt in range(_POST_ATTEMPTS):
            try:
                with _OPENER.open(req, timeout=self.executor.timeout_s) as r:
                    return json.loads(r.read())
            except (urllib.error.URLError, TimeoutError, OSError,
                    json.JSONDecodeError) as exc:
                last = exc
                if attempt + 1 < _POST_ATTEMPTS:
                    # wait-and-retry with JITTER (2026-08-21 overnight): 96 workers
                    # synchronise on cell starts and post-LLM waves, overflowing the
                    # kernel SYN backlog in bursts -- an unlucky worker's connects then
                    # drop on EVERY aligned retry (measured: ~100 husks/hr all dying on
                    # their FIRST env call while neighbours flowed). Randomising the
                    # backoff breaks the alignment; a deterministic ladder re-collides.
                    time.sleep(min(_POST_BACKOFF_S * (2 ** attempt), 15.0)
                               + random.uniform(0.0, 3.0))
        raise EnvTransportError(
            f"env server gave no answer after {_POST_ATTEMPTS} attempts: "
            f"{type(last).__name__}: {last}") from last

    def _reset(self) -> None:
        out = self._post("/reset", session=self.session_id, instruction=self.instruction)
        self._done = False
        self._open = True
        return out

    def _ensure_open(self) -> None:
        if not self._open:
            # spread the first contact: at cell start ~96 fresh sessions would otherwise
            # fire /reset in the same instant (the SYN-burst source above). 0-8s: the
            # 0-2.5s version still lost ~16% of a cell-start window when the freshly
            # recycled server's pages were cold and every first render was slow.
            time.sleep(random.uniform(0.0, 8.0))
            self._reset()

    # ------------------------------------------------------------------ actions
    def run(self, recipe, *, mutating: bool = False):
        # a goal spec is ground truth, not a browsing action -- unchanged path
        if not (isinstance(recipe, dict) and "goal" not in recipe):
            return self.executor.evaluate(recipe, self.env_spec)

        command = str(recipe.get("command") or recipe.get("sql") or "").strip()

        # A REFUSED purchase sends the shopper back to the store (ruling 2026-08-20). The
        # env is terminal after Buy Now, so the next action re-opens a fresh session: the
        # agent keeps everything it LEARNED from the conversation but must find the product
        # again. Re-navigation is real work and is meant to cost turns.
        self._ensure_open()
        if self._done:
            self._reset()

        if is_buy_now(command):
            # Adjudication commits the purchase (see commit_purchase); pressing the button
            # here as well would buy twice. Report the page the agent is looking at.
            return "You are about to buy this item. Awaiting the shopper's decision."

        # No catch here: _post waits-and-retries generously, and if the server is truly
        # dead it raises EnvTransportError -- the episode dies loudly as a husk and is
        # refilled, never scored on a fabricated observation (ruling 2026-08-21).
        out = self._post("/step", session=self.session_id, action=command)
        self._done = bool(out.get("done"))
        return out.get("obs") or ""

    def commit_purchase(self) -> tuple[str, dict] | None:
        """Press Buy Now for real and read back what WebShop recorded as purchased."""
        if not self._open:
            return None          # never browsed: there is no item page to buy from
        # EnvTransportError propagates: a transport failure here used to return None,
        # scoring a correctly-navigated purchase as a failure. Husk instead.
        out = self._post("/step", session=self.session_id, action="click[Buy Now]")
        self._done = bool(out.get("done"))
        m = _PURCHASED.search(str(out.get("obs") or ""))
        if not m:
            return None
        try:
            options = {str(k).lower(): str(v).lower()
                       for k, v in json.loads(m.group(2)).items()}
        except json.JSONDecodeError:
            options = {}
        self.last_purchase = (m.group(1), options)
        return self.last_purchase

    def close(self) -> None:
        if not self._open:
            return
        try:
            self._post("/close", session=self.session_id)
        except Exception:                      # a lost session is harmless; the server GCs
            pass


class WebShopLiveExecutor(WebShopExecutor):
    """Ground truth from the inproc executor; browsing from the real WebShop server."""

    def get_product(self, asin: str, cluster: str) -> dict | None:
        """Resolve a purchased product against the ENVIRONMENT's catalog.

        The cluster shard holds only the products our generator scanned. Agents on the
        faithful site retrieve over all 1,181,430 via BM25, so a correct purchase from
        outside the shard came back None and acceptance called it `unknown_asin` -- 58 of
        ~170 proposals in one subset, which made Success mostly noise. Fall back to the env
        server, which holds WebShop's own normalised catalog. Cached per process; the HTTP
        call happens once per unseen asin.
        """
        local = super().get_product(asin, cluster)
        if local is not None:
            return local
        cache = getattr(self, "_prod_cache", None)
        if cache is None:
            cache = self._prod_cache = {}
        if asin in cache:
            return cache[asin]
        # /product is stateless (a read of the shared catalog), so unlike a browsing session
        # it may go to ANY server; spread by asin so acceptance lookups do not all pile onto
        # the first one. Retried for the same reason /step is: a backlog rejection here
        # silently returns None, which acceptance scores as `unknown_asin` -- a correct
        # purchase recorded as a failure.
        pool = getattr(self, "base_urls", None) or [self.base_url]
        base = pool[int(hashlib.sha1(asin.encode()).hexdigest()[:8], 16) % len(pool)]
        out = None
        for attempt in range(_POST_ATTEMPTS):
            try:
                req = urllib.request.Request(f"{base}/product/{asin}")
                with _OPENER.open(req, timeout=self.timeout_s) as r:
                    data = json.loads(r.read())
                out = data.get("product") if data.get("found") else None
                break
            except urllib.error.HTTPError as exc:
                # 404 is the server DEFINITIVELY saying this asin is not in the catalog.
                # Retrying it would burn the whole budget (~1.2s) on every genuine miss.
                if exc.code == 404:
                    break
                if attempt + 1 >= _POST_ATTEMPTS:
                    log.warning("product lookup failed for %s: %s", asin, exc)
                else:
                    time.sleep(_POST_BACKOFF_S * (attempt + 1))
            except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
                if attempt + 1 >= _POST_ATTEMPTS:
                    log.warning("product lookup failed for %s: %s", asin, exc)
                else:
                    time.sleep(_POST_BACKOFF_S * (attempt + 1))
        cache[asin] = out
        return out

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        ws = (config.get("webshop") or {})
        # WEBSHOP_SERVER_URL lets each concurrent stream target its OWN forked server; one
        # server saturates at ~24 workers (GIL-bound page rendering), so throughput comes
        # from more servers, not more workers.
        # MULTIPLE SERVERS (2026-08-21). One server is GIL-bound on page render+parse, and
        # under ~96 workers a real /step took 5.3s while /health still read 114ms -- the
        # backlog overflow that was corrupting up to 22% of a cell's episodes. Capacity has
        # to come from more SERVERS, not more workers. Fork-after-load cannot provide them
        # (the Lucene JVM does not survive fork()), so each is a separate process; at
        # --limit-goals 2000 one costs ~3GB, which the machine has room for.
        #
        # WEBSHOP_SERVER_URLS takes a comma-separated list. WEBSHOP_SERVER_URL (singular)
        # still works and wins if set, so existing callers are unaffected.
        urls = (os.environ.get("WEBSHOP_SERVER_URLS")
                or os.environ.get("WEBSHOP_SERVER_URL")
                or ws.get("server_urls") or ws.get("server_url")
                or "http://127.0.0.1:3020")
        if isinstance(urls, str):
            urls = urls.split(",")
        self.base_urls = [str(u).strip().rstrip("/") for u in urls if str(u).strip()]
        self.base_url = self.base_urls[0]
        # 120s, PROVEN -- DO NOT "FAIL FAST" (2026-08-21, twice burned). The 10k-episode
        # zero-contamination campaign ran at 120s. Cutting this to 25s converted ordinary
        # queue spikes (worker LLM-retries synchronise, then everyone clicks at once; the
        # single-core Python renderer briefly backs up) into transport failures: the SAME
        # width that was clean at 120s produced up to 100% dirty episodes at 25s. A long
        # timeout costs nothing when the server is healthy and absorbs the bursts when it
        # is busy; a short one manufactures contamination out of normal queueing.
        self.timeout_s = int(ws.get("server_timeout_s") or 120)
        self._n = 0

    def open(self, env_spec: dict):
        self._n += 1
        # SESSION IDS MUST BE UNIQUE ACROSS PROCESSES (2026-08-21).
        #
        # This was f"{env_id}-{self._n}-{id(self):x}", and every component of it repeats
        # under fork-after-load: each worker inherits the SAME executor at the SAME address
        # (so `id(self)` is identical), starts from the same `self._n`, and advances in
        # near-lockstep; `env_id` is only ~106-valued across the sample set (26 episodes
        # share one), so concurrent workers regularly hold the same one. Session ids
        # therefore collided routinely.
        #
        # Two episodes sharing an id share the server's session: one worker's /close
        # deletes the other's live session, and the victim's next /step returns
        # HTTP 400 "unknown session" -- 490 such errors in a 15-minute window, which the
        # agent then saw as an "environment unreachable" observation. Worse, before the
        # close they can READ EACH OTHER'S PAGES, which is silent rather than an error.
        # MEASURED: 8-22% of episodes per cell were touched, and they succeeded at 3-13%
        # against 24-32% for clean ones.
        #
        # pid distinguishes workers, the counter distinguishes sessions within a worker,
        # and uuid4 (os.urandom-backed, so fork-safe and not reseeded identically) closes
        # the remaining window when pids are recycled across cells.
        sid = (f"{env_spec.get('env_id', 'ws')}-{os.getpid()}-{self._n}"
               f"-{uuid.uuid4().hex[:8]}")
        return LiveSession(self, env_spec, sid, str(env_spec.get("instruction") or ""))
