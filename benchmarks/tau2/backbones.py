"""Backbone registry for the D27 backbone experiment.

Only the AGENT side varies; the user simulator's model, identity, and protocol are
never touched (protocol rule 1). Each entry lists exactly the agent_* keys to overlay
on perturb_run.LLM_CFG. `deepseek` is the campaign default — its cells already exist
and are REUSED, never re-run.

SELF-PLAY (doubao backbone): agent model == user model. The delicate part is quota:
both roles on one identity would halve throughput and collide. The two provisioned
identities are split BY ROLE -- user wire = canonical name, agent wire = our ep-
endpoint -- so each side owns a full TPM allowance. Cache keys are canonical and
role-scoped, so the split cannot fragment the cache (llm.py identity-pool rules).
"""

GATEWAY = "https://YOUR-GATEWAY.example/v1"

BACKBONES = {
    # campaign default -- cells exist in r3a_rational/ and r3_rational/; never re-run
    "deepseek": {},
    # same-model self-play, identity-split by role (see module docstring)
    "doubao": {
        "agent_model": "doubao-seed-2-0-pro-260215",
        "agent_model_pool": ["ep-YOUR-ENDPOINT-A"],
        "model_pool": ["doubao-seed-2-0-pro-260215"],
        "agent_model_fallback": None,
    },
    "gpt55": {
        "agent_model": "gateway/gpt-5.5-2026-04-24",
        "agent_model_pool": ["gateway/gpt-5.5-2026-04-24"],
        "agent_base_url": GATEWAY, "agent_protocol": "chat",
        "agent_api_key_env": "GPT_GATEWAY_KEY", "agent_model_fallback": None,
    },
    "gemini": {
        "agent_model": "gateway/gemini-3.5-flash",
        "agent_model_pool": ["gateway/gemini-3.5-flash"],
        "agent_base_url": GATEWAY, "agent_protocol": "chat",
        "agent_api_key_env": "GPT_GATEWAY_KEY", "agent_model_fallback": None,
    },
    "qwen": {
        "agent_model": "gateway/openai_qwen3.7-max",
        "agent_model_pool": ["gateway/openai_qwen3.7-max"],
        "agent_base_url": GATEWAY, "agent_protocol": "chat",
        "agent_api_key_env": "GPT_GATEWAY_KEY", "agent_model_fallback": None,
    },
    "glm": {
        "agent_model": "gateway/glm-4.7",
        "agent_model_pool": ["gateway/glm-4.7"],
        "agent_base_url": GATEWAY, "agent_protocol": "chat",
        "agent_api_key_env": "GPT_GATEWAY_KEY", "agent_model_fallback": None,
    },
}


def apply(llm_cfg: dict, name: str) -> dict:
    """Overlay a backbone onto a COPY of llm_cfg; refuse unknown names."""
    import os
    if name not in BACKBONES:
        raise SystemExit(f"unknown backbone {name!r}; known: {sorted(BACKBONES)}")
    out = dict(llm_cfg)
    out.update(BACKBONES[name])
    # the gateway key lives in project/.env like ARK's; drivers only export ARK's
    env = out.get("agent_api_key_env")
    if env and env not in os.environ:
        for line in open("../.env"):
            if line.startswith(env + "="):
                os.environ[env] = line.split("=", 1)[1].strip()
                break
        else:
            raise SystemExit(f"{env} not found in project/.env")
    # protocol rule 1: the user-side model itself is untouchable
    assert out["model"] == llm_cfg["model"], "backbone must not change the user model"
    return out
