"""LLM wrapper: cache identity, offline behaviour, and the fake."""

import pytest

from intent_graph.runtime.llm import (
    ROLE_PHRASE,
    ROLE_RENDER,
    ROLE_SELECT,
    CacheMiss,
    FakeLLM,
    LLMClient,
    LLMError,
    cache_key,
)


def cfg(tmp_path, **over):
    base = {
        "llm": {
            "base_url": "http://x", "model": "m", "api_key_env": "NOPE_NOT_SET",
            "prompt_version": "v1", "max_output_tokens": 32, "timeout_s": 5,
            "temperature_render": 0.7, "temperature_phrase": 0.3, "temperature_select": 0.0,
            "cache_dir": str(tmp_path / "cache"), "offline": True,
        }
    }
    base["llm"].update(over)
    return base


# ------------------------------------------------------------------ cache key
def _k(**over):
    args = dict(model="m", prompt_version="v1", role=ROLE_RENDER, system="sys",
                temperature=0.7, max_tokens=32, prompt="hello")
    args.update(over)
    return cache_key(**args)


def test_cache_key_is_stable():
    assert _k() == _k()


@pytest.mark.parametrize("field,value", [
    ("model", "other"), ("prompt_version", "v2"), ("role", ROLE_PHRASE),
    ("system", "different"), ("temperature", 0.1), ("max_tokens", 64), ("prompt", "bye"),
])
def test_cache_key_changes_with_every_field(field, value):
    """`system` and `max_tokens` matter: two personas differ only in system text, and
    sharing one cache entry would make per-persona results silently identical."""
    assert _k() != _k(**{field: value})


# --------------------------------------------------------------- offline mode
def test_offline_miss_is_an_explicit_error(tmp_path):
    c = LLMClient(cfg(tmp_path))
    with pytest.raises(CacheMiss, match="offline mode"):
        c.complete("anything", role=ROLE_RENDER)


def test_cache_hit_avoids_the_network(tmp_path):
    c = LLMClient(cfg(tmp_path))
    key = cache_key(model="m", prompt_version="v1", role=ROLE_RENDER, system=None,
                    temperature=0.7, max_tokens=32, prompt="p",
                    reasoning=c.reasoning_effort)
    c._write_cache(key, "stored answer", {})
    assert c.complete("p", role=ROLE_RENDER) == "stored answer"
    assert c.usage()["cached"] == 1 and c.usage()["requests"] == 0
    assert c.calls[-1].cached is True


def test_missing_api_key_is_reported_clearly(tmp_path):
    c = LLMClient(cfg(tmp_path, offline=False))
    with pytest.raises(LLMError, match="NOPE_NOT_SET"):
        c.complete("p", role=ROLE_RENDER)


def test_role_temperature_participates_in_the_key(tmp_path):
    """The same prompt asked as `select` (temp 0) must not reuse a `render` answer."""
    c = LLMClient(cfg(tmp_path))
    k_sel = cache_key(model="m", prompt_version="v1", role=ROLE_SELECT, system=None,
                      temperature=0.0, max_tokens=32, prompt="p",
                      reasoning=c.reasoning_effort)
    c._write_cache(k_sel, "selected", {})
    assert c.complete("p", role=ROLE_SELECT) == "selected"
    with pytest.raises(CacheMiss):
        c.complete("p", role=ROLE_RENDER)


def test_unknown_role_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown role"):
        LLMClient(cfg(tmp_path)).complete("p", role="whatever")


def test_corrupt_cache_entry_is_treated_as_a_miss(tmp_path):
    c = LLMClient(cfg(tmp_path))
    key = cache_key(model="m", prompt_version="v1", role=ROLE_RENDER, system=None,
                    temperature=0.7, max_tokens=32, prompt="p",
                    reasoning=c.reasoning_effort)
    p = c._path(key)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("not json", encoding="utf-8")
    with pytest.raises(CacheMiss):
        c.complete("p", role=ROLE_RENDER)


# ------------------------------------------------------------------- the fake
def test_fake_records_calls_and_serves_responder():
    fake = FakeLLM(responder=lambda role, system, prompt: f"{role}:{len(prompt)}")
    assert fake.complete("abc", role=ROLE_SELECT, system="s") == "select:3"
    assert fake.calls[0].role == ROLE_SELECT and fake.calls[0].system == "s"


def test_fake_all_text_exposes_prompts_for_leak_scanning():
    fake = FakeLLM(default="ok")
    fake.complete("the secret is 42", role=ROLE_PHRASE, system="persona")
    blob = fake.all_text()
    assert "the secret is 42" in blob and "persona" in blob and "ok" in blob


def test_fake_rejects_none_response():
    fake = FakeLLM(responder=lambda *a: None)
    with pytest.raises(LLMError):
        fake.complete("p", role=ROLE_RENDER)


def test_cache_write_is_atomic(tmp_path):
    """A partial entry would be logged as corrupt and re-paid for on every future run,
    so the write must leave either the old content or the new one."""
    import json as _json

    from intent_graph.runtime.llm import LLMClient

    c = LLMClient({"llm": {"model": "m", "base_url": "u", "cache_dir": str(tmp_path)}})
    key = "ab" + "0" * 62
    c._write_cache(key, "first", {})
    assert c._read_cache(key) == "first"
    c._write_cache(key, "second", {})
    assert c._read_cache(key) == "second"
    assert _json.loads(c._path(key).read_text())["text"] == "second"
    # no temp files survive a successful write
    assert not list(c._path(key).parent.glob(".*tmp"))


def test_reasoning_effort_participates_in_the_key(tmp_path):
    """MEASURED: at default effort this model spends the whole token budget on reasoning and
    returns empty text, so effort changes the answer -- two efforts must not share an entry."""
    base = dict(model="m", prompt_version="v1", role=ROLE_RENDER, system=None,
                temperature=0.7, max_tokens=32, prompt="p")
    assert cache_key(**base, reasoning="minimal") != cache_key(**base, reasoning="high")
    assert cache_key(**base, reasoning="minimal") != cache_key(**base, reasoning="")


def test_reasoning_effort_is_sent_and_omittable(tmp_path):
    c = LLMClient(cfg(tmp_path))
    assert c.reasoning_effort == "minimal", "default must keep output_text non-empty"
    d = cfg(tmp_path)
    d["llm"]["reasoning_effort"] = ""
    assert LLMClient(d).reasoning_effort == ""


# ------------------------------------------------- thinking off, per family
def test_thinking_off_shape_per_model_family():
    """"None of the thinking mode everywhere" (ruling 2026-08-19). One switch does not fit
    all providers; each shape was verified live against the gateway. The default branch is
    the Ark shape so unknown models keep the pre-existing behaviour."""
    from intent_graph.runtime.llm import thinking_off_params as t
    assert t("openai_qwen3.7-max", "chat") == {"extra_body": {"enable_thinking": False}}
    assert t("gateway/gpt-5.5-2026-04-24", "chat") == \
        {"extra_body": {"reasoning_effort": "none"}}
    assert t("gateway/gpt-5.5-2026-04-24", "responses") == {"reasoning": {"effort": "none"}}
    assert t("gemini-3.5-flash", "chat") == {}
    for m in ("doubao-seed-2-1-pro-260628", "deepseek-v4-pro-260425", "glm-4.7"):
        assert t(m, "responses") == {"extra_body": {"thinking": {"type": "disabled"}}}, m


class _FakeChatClient:
    """chat.completions.create stub; records kwargs, optionally fails first."""
    def __init__(self, text="hello", fail_first=None):
        self.kwargs = []
        self._fail = fail_first
        from types import SimpleNamespace as NS
        self._resp = NS(id="r1", model="m", status=None,
                        choices=[NS(message=NS(content=text))],
                        usage=NS(prompt_tokens=7, completion_tokens=3,
                                 prompt_tokens_details=None, input_tokens=None,
                                 output_tokens=None, input_tokens_details=None))
        self.chat = NS(completions=NS(create=self._create))
    def _create(self, **kw):
        self.kwargs.append(kw)
        if self._fail is not None:
            exc, self._fail = self._fail, None
            raise exc
        return self._resp


def _client_with(monkeypatch, tmp_path, fake, **over):
    c = LLMClient(cfg(tmp_path, offline=False, protocol="chat",
                      api_key_env="PATH", **over))
    monkeypatch.setattr(c, "_ensure_client", lambda agent_side=False: fake)
    return c


def test_chat_protocol_sends_messages_and_parses_content(monkeypatch, tmp_path):
    fake = _FakeChatClient(text="a reply")
    c = _client_with(monkeypatch, tmp_path, fake, disable_thinking=True,
                     model="gateway/glm-4.7")
    out = c.complete("hi", role=ROLE_RENDER, system="sys")
    assert out == "a reply"
    kw = fake.kwargs[0]
    assert kw["messages"][0] == {"role": "system", "content": "sys"}
    assert kw["messages"][1] == {"role": "user", "content": "hi"}
    assert kw["extra_body"] == {"thinking": {"type": "disabled"}}, \
        "glm must get the thinking-disabled shape"
    assert "input" not in kw and "instructions" not in kw


def test_chat_empty_content_is_an_error_not_an_utterance(monkeypatch, tmp_path):
    fake = _FakeChatClient(text="")
    c = _client_with(monkeypatch, tmp_path, fake, retry_attempts=1)
    with pytest.raises(LLMError):
        c.complete("hi", role=ROLE_RENDER)


def test_rejected_temperature_self_heals(monkeypatch, tmp_path):
    """Verified live 2026-08-19: gpt-5.5 rejects any non-default temperature with a 400.
    The client strips the parameter and retries instead of dying or falling back."""
    import httpx
    from openai import BadRequestError
    req = httpx.Request("POST", "http://x")
    resp = httpx.Response(400, request=req)
    exc = BadRequestError(
        "Error code: 400 - Unsupported value: 'temperature' does not support 0.7 "
        "with this model. Only the default (1) value is supported.",
        response=resp, body=None)
    fake = _FakeChatClient(text="healed", fail_first=exc)
    c = _client_with(monkeypatch, tmp_path, fake, model="gateway/gpt-5.5-2026-04-24",
                     disable_thinking=True)
    assert c.complete("hi", role=ROLE_RENDER) == "healed"
    assert "temperature" in fake.kwargs[0], "first attempt carries temperature"
    assert "temperature" not in fake.kwargs[1], "retry must strip it"
    # the rejection is REMEMBERED: a later call must not pay the 400 again
    assert c.complete("hi again", role=ROLE_RENDER) == "healed"
    assert "temperature" not in fake.kwargs[2], "later calls must skip it up front"


def test_agent_roles_route_to_the_agent_endpoint(monkeypatch, tmp_path):
    """Split endpoints (2026-08-23, backbone experiment): the agent under test may live on
    a different endpoint/protocol than the user simulator. Side follows the resolved MODEL:
    user roles must reach the responses client, agent roles the chat client."""
    from types import SimpleNamespace as NS
    from intent_graph.runtime.llm import ROLE_ACT
    seen = {"user": [], "agent": []}
    resp_client = NS(responses=NS(create=lambda **kw: (seen["user"].append(kw),
                     NS(id="u", status=None, model="m", usage=None,
                        output_text="user ok"))[1]))
    chat_client = NS(chat=NS(completions=NS(create=lambda **kw: (seen["agent"].append(kw),
                     NS(id="a", status=None, model="m", usage=None,
                        choices=[NS(message=NS(content="agent ok"))]))[1])))
    c = LLMClient(cfg(tmp_path, offline=False, api_key_env="PATH",
                      model="doubao-x", agent_model="gateway/gpt-5.4-2026-03-05",
                      agent_base_url="http://gw", agent_api_key_env="PATH",
                      agent_protocol="chat", disable_thinking=True,
                      temperature_agent_act=0.7))
    c._client, c._agent_client = resp_client, chat_client
    assert c.complete("q", role=ROLE_RENDER) == "user ok"
    assert c.complete("q", role=ROLE_ACT) == "agent ok"
    assert seen["user"][0].get("input"), "user side must use the responses protocol"
    assert seen["agent"][0].get("messages"), "agent side must use the chat protocol"
    assert seen["agent"][0]["extra_body"] == {"reasoning_effort": "none"}, \
        "gpt on chat gets effort=none as its thinking-off shape"
