"""The only place the runtime talks to a language model.

Three call sites are permitted, and no others: ``perturb.render``, ``user.select_slots``,
``user.phrase``.  Ground truth and the acceptance decision never pass through here.

Reproducibility comes from the cache, not from the API: Ark's Responses endpoint exposes no
``seed`` parameter, so an identical prompt is only guaranteed to give an identical answer
because we stored the first one.  That is why the cache is load-bearing rather than an
optimisation, and why ``offline=True`` (a miss is an error) is how tests and CI run.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import logging
import os
import random
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..ids import canonical_dumps

log = logging.getLogger(__name__)

_TMP_SEQ = itertools.count()

# roles map to their own temperature so a deterministic call (slot selection) cannot be
# accidentally rendered creative
ROLE_RENDER = "render"
ROLE_PHRASE = "phrase"
ROLE_SELECT = "select"
# agent-side roles. Separate from the user-side ones so an experiment arm's cache entries can
# never collide with the simulated user's, and so each can carry its own temperature.
ROLE_ACT = "agent_act"          # the agent's next action
ROLE_AUDIT = "agent_audit"      # structured belief audit (A1, A3)
ROLE_SAMPLE = "agent_sample"    # sampled intent hypotheses (A2)
ROLE_JUDGE = "agent_judge"      # scoring simulated futures (A5)
ROLES = (ROLE_RENDER, ROLE_PHRASE, ROLE_SELECT,
         ROLE_ACT, ROLE_AUDIT, ROLE_SAMPLE, ROLE_JUDGE)


class LLMError(RuntimeError):
    pass


class CacheMiss(LLMError):
    """Raised in offline mode when a prompt has no stored response."""


class ConfigError(LLMError):
    """Misconfiguration -- a missing key, an unusable base URL.

    Terminal and deliberately not fallback-able. Treating it as a model failure produced the
    worst possible diagnostic: 500 episodes logged "agent model X failed; falling back to Y",
    each doing twice the work, when the real cause was an unset environment variable.
    """


class QuotaExhausted(LLMError):
    """The account is out of credit / rate-limited past the point of retrying.

    Terminal, and deliberately NOT a subclass of anything the retry loop catches: retrying a
    drained account just burns wall clock across every shard. Callers stop cleanly and leave a
    resume marker, because the runner is idempotent and relaunching costs nothing.
    """


# Substrings that mean "stop", not "wait". Matched case-insensitively against the exception
# text because Ark returns these as generic API errors rather than typed exceptions.
_TERMINAL_SIGNS = (
    "insufficient balance", "insufficient_balance", "insufficient credit",
    "quota", "exceeded your current", "billing", "arrears", "account is suspended",
    "no available capacity", "free tier", "payment required",
)


# A per-minute throttle is "wait", not "stop": the window reopens within the minute. Ark
# spells these as a 429 whose body names the limit that was hit.
_TRANSIENT_SIGNS = (
    "tpmratelimitexceeded", "rpmratelimitexceeded", "tokens per minute",
    "requests per minute", "toomanyrequests", "server_busy", "please try again later",
)


def is_terminal_api_error(exc: BaseException) -> bool:
    """Does this error mean the account is done, rather than the request being unlucky?

    Status codes are read from the SDK where it exposes one, and otherwise matched only as
    explicit "error code: NNN" tokens, never bare substrings: provider request ids are
    long digit strings, and a request id containing "402" made a plain TPM 429 read as
    payment-required (measured 2026-08-17 -- an episode was killed as 'terminal' mid-run
    for exactly this).
    """
    text = f"{type(exc).__name__}: {exc}".lower()
    # Decided FIRST, and deliberately (2026-08-19): a per-minute rate limit is transient by
    # definition, and the status tests below used to be bare substring matches against the
    # whole error text -- which embeds Ark's request id. A TPM throttle whose id ended
    # "...d8432a02402b2" matched the "402" test and was discarded as a billing failure, so
    # a retryable episode became a husk. Measured: 3 of 20 sanity episodes lost this way.
    if any(sign in text for sign in _TRANSIENT_SIGNS):
        return False
    # Read the real status code where the SDK exposes one, instead of grepping digits out
    # of free text that also carries a request id.
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(getattr(exc, "response", None), "status_code", None)
    if status in (401, 402, 403):
        return True
    for code in ("401", "402", "403"):
        if f"error code: {code}" in text or f"status {code}" in text:
            return True
    if "invalid api key" in text or "unauthorized" in text or "payment required" in text:
        return True
    return any(sign in text for sign in _TERMINAL_SIGNS)


def cache_key(*, model: str, prompt_version: str, role: str, system: str | None,
              temperature: float, max_tokens: int, prompt: str, reasoning: str = "") -> str:
    """Identity of a completion request.

    ``system`` and ``max_tokens`` are part of the key: two personas asking the same
    question differ only in their system text, and without it they would share one cache
    entry -- silently making per-persona results identical.
    """
    payload = canonical_dumps({
        "model": model, "prompt_version": prompt_version, "role": role,
        "system": system or "", "temperature": round(float(temperature), 4),
        "max_tokens": int(max_tokens), "reasoning": reasoning or "", "prompt": prompt,
    })
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class Call:
    """A recorded request, for test assertions (notably the leakage test)."""

    role: str
    system: str | None
    prompt: str
    response: str
    cached: bool


class _RateLimiter:
    """Token bucket shared by every worker thread in the process.

    Blunt concurrency bursts: N workers finish a turn together and fire N requests in the
    same instant, which trips the provider's per-second gate even when the per-minute
    quota has room (measured: 28% of calls throttled at 128 processes while the minute
    quota was barely half used). Pacing the same volume smoothly keeps the aggregate rate
    under the gate, so retries stop competing with first attempts.
    """

    def __init__(self, rps: float) -> None:
        self.rps = float(rps or 0)
        self._lock = threading.Lock()
        self._next = 0.0

    def acquire(self) -> None:
        if self.rps <= 0:
            return
        gap = 1.0 / self.rps
        with self._lock:                 # reserve exactly ONE slot, then wait for it
            now = time.monotonic()
            start = max(now, self._next)
            self._next = start + gap
        wait = start - now
        if wait > 0:
            time.sleep(wait)



# Roles that are HARNESS BOOKKEEPING, never the subject of measurement: reflection
# (lessonbook) and slot selection. See the note in the request builder. Applied only
# when the config opts in (llm.no_think_bookkeeping: true): forcing thinking off for
# these roles changes their cache identities, so a benchmark that never adopted the
# ruling must keep its cache and request shapes unchanged.
_NO_THINK_ROLES = {"agent_audit", "select"}


# Models that 400 on `temperature`. Shared across every client in the process: the
# rejection is a model property, and rediscovering it per thread costs a real call.
_TEMP_REJECTED: set = set()


def thinking_off_params(model: str, protocol: str) -> dict:
    """Request parameters that force hidden reasoning OFF, per model family.

    "None of the thinking mode everywhere" (ruling 2026-08-19). One switch does not fit all
    providers, and each of these shapes was verified live against the model gateway on
    2026-08-19 (probe: a render-style prompt, checking completion tokens and content):

      qwen3.7-max   default: 1,193 hidden reasoning tokens, 23s  -> flag: 20 tokens, 1.3s
      glm-4.7       default:   749 hidden reasoning tokens, 15s  -> flag: 23 tokens, 2.2s
      gpt-5.x       accepts reasoning effort "none" (NOT "minimal" -- rejected with a 400)
      gemini-flash  emits no hidden reasoning; needs nothing
      doubao/deepseek (Ark): the existing "thinking: disabled" shape, measured 2026-08-16
                    (deepseek-v4-pro 6.8s -> 1.0s)

    The doubao/deepseek shape is the default branch so unknown models keep the pre-existing
    behaviour rather than silently getting no switch at all.
    """
    m = model.lower()
    if "qwen" in m:
        return {"extra_body": {"enable_thinking": False}}
    if "gpt-5" in m:
        if protocol == "chat":
            return {"extra_body": {"reasoning_effort": "none"}}
        return {"reasoning": {"effort": "none"}}
    if "gemini" in m:
        return {}
    return {"extra_body": {"thinking": {"type": "disabled"}}}


class LLMClient:
    """Cached wrapper over Ark's Responses API."""

    def __init__(self, config: dict) -> None:
        cfg = config["llm"]
        self.model = cfg["model"]
        self.base_url = cfg["base_url"]
        self.api_key_env = cfg.get("api_key_env", "ARK_API_KEY")
        self.prompt_version = cfg.get("prompt_version", "v1")
        self.max_output_tokens = int(cfg.get("max_output_tokens", 1024))
        # see config/runtime.yaml: at default effort this model returns empty output_text
        self.reasoning_effort = str(cfg.get("reasoning_effort", "minimal") or "")
        self.timeout_s = int(cfg.get("timeout_s", 60))
        self.sdk_max_retries = int(cfg.get("sdk_max_retries", 3))
        self.retry_attempts = max(1, int(cfg.get("retry_attempts", 8)))
        self.disable_thinking = bool(cfg.get("disable_thinking", False))
        # llm.no_think_bookkeeping: force thinking OFF for the bookkeeping roles in
        # _NO_THINK_ROLES even when the agent under test keeps thinking ON. Off by
        # default because it re-keys those roles' cache entries.
        self.no_think_roles = (frozenset(_NO_THINK_ROLES)
                               if cfg.get("no_think_bookkeeping") else frozenset())
        # "responses" = Ark's Responses API (the default, unchanged). "chat" = OpenAI
        # chat-completions, required for the model gateway: it serves Responses only for
        # gpt-5.x, so gemini/qwen/glm are reachable via chat alone.
        self.protocol = str(cfg.get("protocol", "responses") or "responses")
        # SPLIT ENDPOINTS (2026-08-23, for the backbone experiment): the agent under test
        # may live on a different endpoint/protocol/key than the user simulator -- the four
        # gateway backbones (gpt/gemini/qwen/glm) speak chat on the model gateway while the
        # user simulator stays on Ark responses. Each unset agent_* falls back to the
        # user-side value, so every existing config is unchanged.
        self.agent_base_url = cfg.get("agent_base_url") or self.base_url
        self.agent_api_key_env = cfg.get("agent_api_key_env") or self.api_key_env
        self.agent_protocol = str(cfg.get("agent_protocol") or self.protocol)
        self.retry_backoff_cap_s = float(cfg.get("retry_backoff_cap_s", 60))
        # llm.max_rps: smooth the aggregate request rate across every worker thread.
        # 0 disables it (the forked-process path, where each child has its own client
        # and no shared clock, keeps the old bursty behaviour).
        self.limiter = _RateLimiter(float(cfg.get("max_rps", 0) or 0))
        self.offline = bool(cfg.get("offline", False))
        self.temperatures = {
            ROLE_RENDER: float(cfg.get("temperature_render", 0.7)),
            ROLE_PHRASE: float(cfg.get("temperature_phrase", 0.3)),
            ROLE_SELECT: float(cfg.get("temperature_select", 0.0)),
            ROLE_ACT: float(cfg.get("temperature_agent_act", 0.7)),
            ROLE_AUDIT: float(cfg.get("temperature_agent_audit", 0.0)),
            ROLE_SAMPLE: float(cfg.get("temperature_agent_sample", 1.0)),
            ROLE_JUDGE: float(cfg.get("temperature_agent_judge", 0.0)),
        }
        # An agent arm may run on a DIFFERENT model from the simulated user -- that is how the
        # self-play confound is removed. The model is part of the cache key already.
        self.agent_model = cfg.get("agent_model") or self.model
        self.agent_model_fallback = cfg.get("agent_model_fallback")
        # ---- WIRE POOLS (2026-08-19) ----
        # A dedicated endpoint id addresses the SAME model as its public name but carries
        # its own TPM allowance, so alternating between the two identities doubles usable
        # throughput. The rotation is a TRANSPORT detail only: the cache is still keyed on
        # the canonical model name (see complete()), because two identities of one model
        # must not produce two cache entries for one prompt -- that would halve the hit
        # rate and make hits depend on which identity happened to serve the call.
        self.model_pool = list(cfg.get("model_pool") or []) or [self.model]
        self.agent_model_pool = list(cfg.get("agent_model_pool") or []) or [self.agent_model]
        self._pool_lock = threading.Lock()
        self._pool_i = {"user": 0, "agent": 0}

        self._agent_roles = {ROLE_ACT, ROLE_AUDIT, ROLE_SAMPLE, ROLE_JUDGE}
        self.cache_dir = Path(cfg.get("cache_dir", "artifacts/llm_cache"))
        self.calls: list[Call] = []
        self._client = None
        self._agent_client = None
        # models that rejected a non-default temperature (the self-heal below); remembered
        # so later calls skip the parameter up front instead of paying a 400 round-trip
        # per call. Under the shipped config this stays empty: with reasoning effort
        # "none", both gpt-5.4 and gpt-5.5 ACCEPT our temperatures (matrix, 2026-08-23).
        # PROCESS-WIDE (2026-08-25): the driver builds a fresh thread-local client for
        # every worker in every generation, so a per-instance set made each of them
        # re-discover the same 400 -- measured 200 wasted round-trips in one gpt-5.5
        # cell, every one a doubled call. The fact "this model rejects temperature" is
        # a property of the model, not of a client object.
        self._temp_rejected: set = _TEMP_REJECTED
        self._usage = {"requests": 0, "cached": 0, "input_tokens": 0, "output_tokens": 0}

    # ------------------------------------------------------------------ cache
    def _path(self, key: str) -> Path:
        return self.cache_dir / key[:2] / f"{key}.json"

    def _read_cache(self, key: str) -> str | None:
        p = self._path(key)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))["text"]
        except Exception as exc:
            log.warning("corrupt cache entry %s: %s", key[:8], exc)
            return None

    def _write_cache(self, key: str, text: str, meta: dict) -> None:
        """Write via temp-and-rename.

        The cache is the only source of reproducibility, so a half-written entry is worse
        than no entry: `_read_cache` logs it as corrupt and re-pays for the call, forever.
        `os.replace` is atomic, so a crash or a concurrent writer leaves either the old
        content or the new one, never a prefix.
        """
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        body = json.dumps({"text": text, "meta": meta}, ensure_ascii=False)
        # pid alone is not unique enough: two THREADS in one process writing the same key
        # built the same temp name, so one renamed it and the other's rename raised
        # FileNotFoundError. Thread id plus a counter makes every writer's temp private.
        tmp = p.with_name(f".{p.name}.{os.getpid()}.{threading.get_ident()}."
                          f"{next(_TMP_SEQ)}.tmp")
        try:
            tmp.write_text(body, encoding="utf-8")
            os.replace(tmp, p)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise

    # ------------------------------------------------------------------ client
    def _ensure_client(self, agent_side: bool = False):
        if agent_side and (self.agent_base_url != self.base_url
                           or self.agent_api_key_env != self.api_key_env):
            if self._agent_client is not None:
                return self._agent_client
            key = os.environ.get(self.agent_api_key_env)
            if not key:
                raise ConfigError(f"agent api key env {self.agent_api_key_env!r} is not set")
            from openai import OpenAI
            self._agent_client = OpenAI(base_url=self.agent_base_url, api_key=key,
                                        timeout=self.timeout_s,
                                        max_retries=self.sdk_max_retries)
            return self._agent_client
        if self._client is not None:
            return self._client
        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise ConfigError(
                f"${self.api_key_env} is not set. Either export it, or run with "
                f"llm.offline=true to use only cached responses."
            )
        from openai import OpenAI

        self._client = OpenAI(base_url=self.base_url, api_key=api_key,
                              timeout=self.timeout_s, max_retries=self.sdk_max_retries)
        return self._client

    def _request(self, *, system: str | None, prompt: str, temperature: float,
                 max_tokens: int, model: str | None = None,
                 no_think: bool = False) -> tuple[str, dict]:
        from openai import (APIConnectionError, APITimeoutError,
                            BadRequestError, RateLimitError)

        model = model or self.model
        # side follows the MODEL, not the role: an explicit per-call model must reach the
        # endpoint that serves it, whatever role asked for it
        agent_side = bool(self.agent_model) and model == self.agent_model
        client = self._ensure_client(agent_side)
        protocol = self.agent_protocol if agent_side else self.protocol
        chat = protocol == "chat"
        if chat:
            messages = ([{"role": "system", "content": system}] if system else []) + \
                       [{"role": "user", "content": prompt}]
            kwargs: dict[str, Any] = {"model": model, "messages": messages,
                                      "temperature": temperature, "max_tokens": max_tokens}
            if model in self._temp_rejected:
                kwargs.pop("temperature", None)
        else:
            kwargs = {
                "model": model,
                "input": [{"role": "user",
                           "content": [{"type": "input_text", "text": prompt}]}],
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            }
            if system:
                kwargs["instructions"] = system
        # THINKING OFF (llm.disable_thinking): measured 2026-08-16, hidden chain-of-thought
        # was 82% of everything these models generate, and decode is the run's bottleneck --
        # deepseek-v4-flash 3.1s -> 1.3s, deepseek-v4-pro 6.8s -> 1.0s with it off. The two
        # parameters are MUTUALLY EXCLUSIVE: sending `thinking` alongside `reasoning` is a
        # 400, which is why an earlier probe wrongly concluded it could not be disabled.
        # The switch SHAPE is per model family -- see thinking_off_params().
        # BOOKKEEPING ROLES NEVER THINK (2026-08-25). The D28 ruling put thinking ON
        # for the AGENT UNDER TEST -- that is what a capability comparison means. The
        # reflector is harness machinery, not the subject: it summarises finished
        # episodes into a playbook. Thinking there is pure cost and pure fragility --
        # measured, it (a) hung glm-4.7's first reflection until the retry budget
        # escalated to 45s backoff with 0 episodes progressing, and (b) wrapped
        # doubao's reflection in reasoning prose that corrupted the JSON read. Forcing
        # it off for audit roles keeps the book reliable and changes nothing about the
        # model being measured.
        if self.disable_thinking or no_think:
            kwargs.update(thinking_off_params(model, protocol))
        elif self.reasoning_effort:
            if chat:
                kwargs["extra_body"] = {"reasoning_effort": self.reasoning_effort}
            else:
                kwargs["reasoning"] = {"effort": self.reasoning_effort}

        last: Exception | None = None
        # PROGRESSIVE TOKEN CEILING (ruling 2026-08-27: "it should be a progressively
        # longer process"). A reasoning model that overflows max_tokens returns EMPTY
        # text; retrying at the SAME cap re-rolls the dice until an attempt happens to
        # reason briefly enough to fit -- which both wastes a full generation per failed
        # attempt (measured: 577 retries in 45min at cap 1024) and SELECTS the surviving
        # completions for minimal reasoning, biasing the run. Escalating the cap only on
        # empty-output retries fixes both while leaving the first attempt at the
        # configured cap, so admission-controlled quotas are not charged for headroom
        # that is almost never needed (84 of ~1,400 calls overflowed 4000).
        cap = int(max_tokens)
        for attempt in range(self.retry_attempts):
            try:
                self.limiter.acquire()
                kwargs["max_tokens" if chat else "max_output_tokens"] = cap
                if chat:
                    resp = client.chat.completions.create(**kwargs)
                    _choice = (getattr(resp, "choices", None) or [None])[0]
                    _msg = getattr(_choice, "message", None)
                    text = (getattr(_msg, "content", None) or "").strip()
                else:
                    resp = client.responses.create(**kwargs)
                    text = (resp.output_text or "").strip()
                if not text:
                    # an empty completion must never become a query or an utterance.
                    # The usual cause is reasoning consuming the whole token budget.
                    if cap < 16000:
                        nxt = min(cap * 2, 16000)
                        log.warning("empty output at max_tokens=%d; escalating to %d",
                                    cap, nxt)
                        cap = nxt
                    raise LLMError(
                        f"empty output_text (status={getattr(resp, 'status', '?')}, "
                        f"effort={self.reasoning_effort or 'default'}, max_tokens={cap})")
                usage = getattr(resp, "usage", None)
                # Ark does AUTOMATIC prefix caching (measured 2026-08-16: a repeated 2,013-token
                # prefix reports cached_tokens=1024 on the second call, in 1024-token blocks, with
                # no request parameter needed -- prompt_cache_key changed nothing). Recording the
                # hit rate is what tells us whether our prompts are actually shaped to exploit it.
                det = (getattr(usage, "input_tokens_details", None)
                       or getattr(usage, "prompt_tokens_details", None)) if usage else None
                cached_tok = getattr(det, "cached_tokens", None) if det else None
                meta = {
                    "id": getattr(resp, "id", None),
                    "status": getattr(resp, "status", None),
                    "model": getattr(resp, "model", None),
                    "input_tokens": (getattr(usage, "input_tokens", None)
                                     or getattr(usage, "prompt_tokens", None)) if usage else None,
                    "output_tokens": (getattr(usage, "output_tokens", None)
                                      or getattr(usage, "completion_tokens", None)) if usage else None,
                    "cached_tokens": cached_tok,
                }
                if cached_tok:
                    self._usage["cached_input_tokens"] = (
                        self._usage.get("cached_input_tokens", 0) + int(cached_tok))
                return text, meta
            except BadRequestError as exc:
                msg = str(exc)
                # SELF-HEAL (verified live 2026-08-19): gpt-5.5 accepts only the default
                # temperature and rejects anything else with a 400. Strip the parameter and
                # retry, once, rather than hard-coding a per-version model list that will
                # be stale by the next release. Any other 400 is a misconfiguration and is
                # raised as ConfigError -- terminal, never fallback-able.
                if "temperature" in msg and "temperature" in kwargs:
                    log.warning("model %s rejects temperature; retrying without it", model)
                    kwargs.pop("temperature", None)
                    self._temp_rejected.add(model)
                    continue
                raise ConfigError(f"bad request for model {model}: {msg[:200]}") from exc
            except (RateLimitError, APITimeoutError, APIConnectionError, LLMError) as exc:
                if is_terminal_api_error(exc):
                    raise QuotaExhausted(
                        f"terminal API error, not retrying: {type(exc).__name__}: {exc}"
                    ) from exc
                last = exc
                # WAIT, don't substitute (ruling 2026-08-16). A degraded provider recovers on a
                # timescale of minutes, so the backoff is capped rather than doubling forever:
                # riding it out costs wall-clock, while swapping models costs comparability.
                sleep = min(2 ** attempt, self.retry_backoff_cap_s) + random.random()
                log.warning("llm retry %d/%d after %s (%.1fs)", attempt + 1,
                            self.retry_attempts, type(exc).__name__, sleep)
                time.sleep(sleep)
        if last is not None and is_terminal_api_error(last):
            raise QuotaExhausted(f"terminal API error after retries: {last}") from last
        # Persistent rate limiting that never resolves is functionally terminal for a long
        # run: a sustained 429 across every retry means the budget window is closed.
        if last is not None and ("429" in str(last) or "rate limit" in str(last).lower()):
            raise QuotaExhausted(f"rate limited through all retries: {last}") from last
        raise LLMError(f"llm request failed after retries: {last}")

    # ------------------------------------------------------------------ public
    def _wire_model(self, canonical: str, *, agent: bool) -> str:
        """Next identity for this logical model, round-robin and thread-safe.

        A pool aliases the CONFIGURED model only: an explicit per-call model is a
        different logical model and reaches the wire untouched.
        """
        if canonical != (self.agent_model if agent else self.model):
            return canonical
        pool = self.agent_model_pool if agent else self.model_pool
        if len(pool) <= 1:
            return pool[0] if pool else canonical
        with self._pool_lock:
            k = "agent" if agent else "user"
            m = pool[self._pool_i[k] % len(pool)]
            self._pool_i[k] += 1
        return m

    def complete(self, prompt: str, *, role: str, system: str | None = None,
                 temperature: float | None = None, max_tokens: int | None = None,
                 model: str | None = None) -> str:
        if role not in ROLES:
            raise ValueError(f"unknown role {role!r}; expected one of {ROLES}")
        temp = self.temperatures[role] if temperature is None else float(temperature)
        toks = self.max_output_tokens if max_tokens is None else int(max_tokens)
        # an explicit per-call model wins; else agent roles use the agent model
        model = model or (self.agent_model if role in self._agent_roles else self.model)
        key = cache_key(model=model, prompt_version=self.prompt_version, role=role,
                        system=system, temperature=temp, max_tokens=toks, prompt=prompt,
                        reasoning=("off" if (self.disable_thinking
                                             or role in self.no_think_roles)
                                   else self.reasoning_effort))

        hit = self._read_cache(key)
        if hit is not None:
            self._usage["cached"] += 1
            self.calls.append(Call(role, system, prompt, hit, cached=True))
            return hit
        if self.offline:
            raise CacheMiss(
                f"offline mode: no cached response for role={role} key={key[:12]}. "
                f"Run once online to populate artifacts/llm_cache, or use FakeLLM in tests."
            )

        try:
            # cache key used the canonical name above; the WIRE identity rotates
            wire = self._wire_model(model, agent=(role in self._agent_roles))
            text, meta = self._request(system=system, prompt=prompt, temperature=temp,
                                       max_tokens=toks, model=wire,
                                       no_think=role in self.no_think_roles)
        except (QuotaExhausted, ConfigError):
            raise
        except LLMError:
            # One cross-model fallback for agent roles: a model that is unavailable or refuses
            # the request should not end a night's run. Recorded in the trajectory via usage.
            if role in self._agent_roles and self.agent_model_fallback and \
                    model != self.agent_model_fallback:
                log.warning("agent model %s failed; falling back to %s", model,
                            self.agent_model_fallback)
                self._usage["agent_fallbacks"] = self._usage.get("agent_fallbacks", 0) + 1
                model = self.agent_model_fallback
                key = cache_key(model=model, prompt_version=self.prompt_version, role=role,
                                system=system, temperature=temp, max_tokens=toks,
                                prompt=prompt, reasoning=self.reasoning_effort)
                hit = self._read_cache(key)
                if hit is not None:
                    self._usage["cached"] += 1
                    self.calls.append(Call(role, system, prompt, hit, cached=True))
                    return hit
                text, meta = self._request(system=system, prompt=prompt, temperature=temp,
                                           max_tokens=toks, model=model,
                                           no_think=role in self.no_think_roles)
            else:
                raise
        self._write_cache(key, text, meta)
        self._usage["requests"] += 1
        for k in ("input_tokens", "output_tokens"):
            if meta.get(k):
                self._usage[k] += meta[k]
        self.calls.append(Call(role, system, prompt, text, cached=False))
        return text

    def usage(self) -> dict:
        return dict(self._usage)


@dataclass
class FakeLLM:
    """Offline stand-in with the same surface as :class:`LLMClient`.

    ``responder`` receives ``(role, system, prompt)`` and returns the completion, so a test
    can react to the prompt instead of pre-computing keys.  Every call is recorded, which is
    what the leakage test inspects.
    """

    responder: Any = None
    default: str = "FAKE"
    calls: list[Call] = field(default_factory=list)
    model: str = "fake-model"
    prompt_version: str = "test"
    offline: bool = True

    def complete(self, prompt: str, *, role: str, system: str | None = None,
                 temperature: float | None = None, max_tokens: int | None = None,
                 model: str | None = None) -> str:
        if role not in ROLES:
            raise ValueError(f"unknown role {role!r}")
        if self.responder is None:
            text = self.default
        else:
            text = self.responder(role, system, prompt)
        if text is None:
            raise LLMError("fake responder returned None")
        self.calls.append(Call(role, system, prompt, text, cached=True))
        return text

    # -- helpers for assertions -------------------------------------------
    def prompts(self, role: str | None = None) -> list[str]:
        return [c.prompt for c in self.calls if role is None or c.role == role]

    def systems(self, role: str | None = None) -> list[str]:
        return [c.system or "" for c in self.calls if role is None or c.role == role]

    def all_text(self, role: str | None = None) -> str:
        """Every prompt + system + response, for leak scanning."""
        return "\n".join(
            f"{c.system or ''}\n{c.prompt}\n{c.response}"
            for c in self.calls if role is None or c.role == role
        )

    def usage(self) -> dict:
        return {"requests": 0, "cached": len(self.calls)}
