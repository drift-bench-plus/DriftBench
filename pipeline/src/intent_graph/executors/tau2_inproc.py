"""In-process executor for tau2-bench domains.

A tau2 environment is a pair of pydantic databases plus a tool surface, constructed fresh
from the domain's shipped data.  Construction is cheap (measured: telecom 2.3 ms, retail
14.5 ms), so every mutating recipe simply gets a new environment -- the same isolation
discipline the MySQL executor buys with DROP/CREATE.

The session is deliberately thin: it owns the live environment and exposes it, because a
tau2 "recipe" is a list of typed tool calls that must run against *this* environment, and
episode acceptance needs to interrogate the same live world the agent has been acting in.
"""

from __future__ import annotations

import json
import os
import re
import sys
from typing import Any

from .base import BaseSession

# tau2 lives outside the pipeline; the path is configuration, not an import-time constant.
_TAU2_SRC_ENV = "TAU2_SRC"


def _ensure_tau2_importable() -> None:
    src = os.environ.get(_TAU2_SRC_ENV)
    if src and src not in sys.path:
        sys.path.insert(0, src)
    import tau2  # noqa: F401  (raises ImportError with a useful message if absent)


class Tau2Session(BaseSession):
    """One live tau2 environment.

    ``run`` executes a recipe: ``{"actions": [ {name, requestor, arguments}, ... ]}``.
    Initialization (the fault injection, for telecom) is part of materialization, not of
    the recipe, because it defines the *world the intent is about* rather than the answer.
    """

    def __init__(self, executor: "Tau2Executor", env_spec: dict) -> None:
        super().__init__(executor, env_spec)
        self.env = None

    # -- materialization ------------------------------------------------------
    def _materialize(self) -> None:
        _ensure_tau2_importable()
        from tau2.registry import registry

        domain = self.env_spec["domain"]
        get_env = registry.get_env_constructor(domain)
        # solo_mode=True gives the agent the union of agent+user tools, which is the ruled
        # design for this project (ruling 2026-08-12): no dual control, voice-only user.
        try:
            self.env = get_env(solo_mode=bool(self.env_spec.get("solo_mode", True)))
        except TypeError:
            self.env = get_env()
        init = self.env_spec.get("initialization") or {}
        self.env.set_state(
            initialization_data=init.get("initialization_data"),
            initialization_actions=self._as_calls(init.get("initialization_actions")),
            message_history=[],
        )

    @staticmethod
    def _uncanon(args):
        """Undo canonical float stringification for execution.

        The graph writer renders non-integral floats as their repr STRING (a deliberate
        hashing rule in ids._canonical), so a stored recipe's `data_used_gb: "28.7"` breaks
        tau2's numeric comparisons on re-execution -- the battery's reproduce check caught
        this on every telecom graph. Only strings that are exactly a canonical float repr are
        converted back; anything else ("555-123-2002", names) is left untouched.
        """
        if not isinstance(args, dict):
            return args
        out = {}
        for k, v in args.items():
            if isinstance(v, str):
                try:
                    f = float(v)
                    if repr(f) == v:
                        v = f
                except (TypeError, ValueError):
                    pass
            elif isinstance(v, dict):
                v = Tau2Session._uncanon(v)
            out[k] = v
        return out

    # tau2's modify/exchange applies the LAST pair's price to every item (upstream
    # stale-variant bug, spike 2026-08-12), so the db hash depends on pair ORDER.
    # Measured 2026-08-17: 10 of 28 multi-pair nodes in our corpus change state under a
    # reordering of the same pairs. Canonical order at the execution boundary makes the
    # bug deterministic -- GT and agent always execute the same order, so a semantically
    # correct call can never fail the hash for listing items differently.
    _PAIRED_TOOLS = ("exchange_delivered_order_items", "modify_pending_order_items")

    @staticmethod
    def _canon_pairs(name: str, kwargs: dict) -> dict:
        if name not in Tau2Session._PAIRED_TOOLS or not isinstance(kwargs, dict):
            return kwargs
        old = kwargs.get("item_ids")
        new = kwargs.get("new_item_ids")
        if isinstance(old, list) and isinstance(new, list) and len(old) == len(new) and len(old) >= 2:
            pairs = sorted(zip([str(x) for x in old], [str(x) for x in new]))
            kwargs = dict(kwargs)
            kwargs["item_ids"] = [p[0] for p in pairs]
            kwargs["new_item_ids"] = [p[1] for p in pairs]
        return kwargs

    @staticmethod
    def _as_calls(actions):
        if not actions:
            return None
        from tau2.data_model.tasks import EnvFunctionCall

        out = []
        for a in actions:
            if isinstance(a, dict):
                a = dict(a)
                a["arguments"] = Tau2Session._uncanon(a.get("arguments") or {})
                a = EnvFunctionCall(**a)
            out.append(a)
        return out

    # -- execution ------------------------------------------------------------
    _CALL_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*\((.*)\)\s*$", re.S)

    def _run_agent_command(self, command: str) -> str:
        """Execute one agent-issued tool call and return its result as readable text.

        The episode loop hands ACT a raw command line; without this the call never reaches
        the environment, the agent receives no observation, and it repeats the same probe
        until the turn cap -- which is exactly what the first telecom smoke run did.
        """
        text = (command or "").strip().strip("`").strip()
        m = self._CALL_RE.match(text)
        if not m:
            return f"error: could not parse a tool call from {text[:120]!r}"
        name, argstr = m.group(1), m.group(2).strip()
        kwargs = {}
        if argstr:
            try:
                import ast
                node = ast.parse(f"_f({argstr})", mode="eval").body
                for kw in node.keywords:
                    kwargs[kw.arg] = ast.literal_eval(kw.value)
                if node.args:
                    return "error: pass arguments by name, e.g. tool(arg=\"value\")"
            except Exception as e:
                return f"error: could not read arguments ({type(e).__name__})"
        # Format tolerance, not semantics (AUTOPSY 2026-08-14): tau2 order ids are
        # '#W1234567'; agents write 'W1234567' and 90.4% of write attempts died on
        # "Order not found" (97.5% fixable by exactly this). A real order system
        # accepts the id its own receipt prints, with or without the display prefix.
        oid = kwargs.get("order_id")
        if isinstance(oid, str) and re.fullmatch(r"W\d+", oid):
            kwargs["order_id"] = "#" + oid
        # AIRLINE cabin spelling (2026-08-20): the enum token is `basic_economy`, but the
        # policy prose the agent is given writes it "basic economy", so agents echo the
        # prose and the tool raises KeyError. Same class as the '#W' tolerance above --
        # a display spelling of a value the agent already chose correctly, not semantics.
        cab = kwargs.get("cabin")
        if isinstance(cab, str) and cab.strip().lower().replace(" ", "_") in (
                "basic_economy", "economy", "business"):
            kwargs["cabin"] = cab.strip().lower().replace(" ", "_")
        kwargs = self._canon_pairs(name, kwargs)
        try:
            out = self.env.make_tool_call(tool_name=name, requestor="assistant", **kwargs)
            self.env.sync_tools()
        except Exception as e:
            return f"error: {type(e).__name__}: {e}"
        # golden-action grading (ruling 2026-08-17): acceptance may require the expected
        # tool calls to have actually been made, not just their end state reached
        if not hasattr(self, "executed_calls"):
            self.executed_calls = []
        self.executed_calls.append((name, dict(kwargs)))
        if out is None:
            return "ok"
        try:
            if hasattr(out, "model_dump"):
                return json.dumps(out.model_dump(), default=str)[:1800]
            if isinstance(out, (dict, list)):
                return json.dumps(out, default=str)[:1800]
        except Exception:
            pass
        return str(out)[:1800]

    def _execute(self, recipe: Any) -> Any:
        """Replay a recipe's typed tool calls; return the live env for inspection."""
        if isinstance(recipe, dict) and recipe.get("command") is not None:
            return self._run_agent_command(str(recipe["command"]))
        actions = (recipe or {}).get("actions") or []
        errors = []
        for a in actions:
            name = a["name"] if isinstance(a, dict) else a.name
            solo = bool(self.env_spec.get("solo_mode", True))
            requestor = "assistant" if solo else (
                (a.get("requestor") if isinstance(a, dict) else a.requestor) or "assistant")
            args = (a.get("arguments") if isinstance(a, dict) else a.arguments) or {}
            args = Tau2Session._uncanon(args)
            args = Tau2Session._canon_pairs(name, args)
            try:
                self.env.make_tool_call(tool_name=name, requestor=requestor, **args)
                self.env.sync_tools()
            except Exception as e:  # a recipe action may legitimately fail (tau2 does this too)
                errors.append({"action": name, "error": f"{type(e).__name__}: {e}"})
        return {"env": self.env, "action_errors": errors}

    def close(self) -> None:
        self.env = None
        super().close()


class Tau2Executor:
    name = "tau2_inproc"

    def open(self, env_spec: dict) -> Tau2Session:
        return Tau2Session(self, env_spec)

    def shutdown(self) -> None:  # nothing process-wide to release
        return None
