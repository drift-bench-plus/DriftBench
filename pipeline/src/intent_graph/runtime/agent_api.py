"""What the agent may say, and how we read it — by rule, never by judgement.

The wire format is Drift-Bench's, so an agent written for that harness works here, and the
parsing regexes are theirs verbatim.  One inherited bug is fixed: their
``Action:\\s*Clarify\\s*\\n`` requires a trailing newline, so a clarification on the final
line of a response was silently missed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum

CLARIFY_STRATEGIES = ("Report_Blocker", "Ask_Parameter", "Disambiguate",
                      "Propose_Solution", "Confirm_Risk")

# verbatim from drift-bench/AgentBench/src/server/tasks/dbbench/__init__.py:203-235,
# except the trailing-newline fix noted above
RE_CLARIFY = re.compile(r"Action:\s*Clarify\s*(?:\n|$)", re.IGNORECASE)
RE_CLARIFY_OS = re.compile(r"Act:\s*clarify\s*(?:\n|$)", re.IGNORECASE)
RE_STRATEGY = re.compile(r"Strategy:\s*(.+?)(?:\n|$)", re.IGNORECASE)
RE_CONTENT = re.compile(r"Content:\s*(.+?)(?:\nCandidates:|\nAction:|\Z)",
                        re.IGNORECASE | re.DOTALL)
RE_CANDIDATES = re.compile(r"Candidates:\s*\[(.+?)\]", re.IGNORECASE | re.DOTALL)
RE_ANSWER = re.compile(r"Action:\s*Answer\s*(?:\n|$)", re.IGNORECASE)
RE_FINAL = re.compile(r"Final Answer:\s*(.+?)(?:\n\n|\Z)", re.IGNORECASE | re.DOTALL)
RE_PREDICTED = re.compile(r"Predicted user question:\s*(.+?)(?:\n|$)", re.IGNORECASE)
RE_OPERATION = re.compile(r"Action:\s*Operation\s*(?:\n|$)", re.IGNORECASE)
RE_SQL_BLOCK = re.compile(r"```(?:sql|bash|sh)?\s*(.+?)```", re.IGNORECASE | re.DOTALL)
# `Action: <tool_name>` -- a tool name where the wire format expects the literal
# word Operation (see the reassembly note in parse_action).
RE_OPERATION_NAMED = re.compile(r"Action:\s*([a-z_][a-z0-9_]{2,})\s*(?:\n|$)", re.IGNORECASE)
# a command that already looks like a call: name(...)
RE_CALLABLE = re.compile(r"\s*[A-Za-z_][A-Za-z0-9_.]*\s*\(", re.DOTALL)


class ActionKind(StrEnum):
    ASK = "ASK"
    ACT = "ACT"
    PROPOSE = "PROPOSE"
    MALFORMED = "MALFORMED"


@dataclass(frozen=True, slots=True)
class AgentAction:
    kind: ActionKind
    raw: str
    question: str | None = None
    strategy: str | None = None
    candidates: tuple[str, ...] = ()
    command: str | None = None
    proposal_raw: str | None = None
    predicted_question: str | None = None
    problems: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {"kind": self.kind.value, "raw": self.raw, "question": self.question,
                "strategy": self.strategy, "candidates": list(self.candidates),
                "command": self.command, "proposal_raw": self.proposal_raw,
                "predicted_question": self.predicted_question,
                "problems": list(self.problems)}


def parse_action(text: str) -> AgentAction:
    """Turn a raw agent response into exactly one action."""
    raw = text or ""

    # --- clarification --------------------------------------------------
    if RE_CLARIFY.search(raw) or RE_CLARIFY_OS.search(raw):
        # Strategy is OPTIONAL metadata (ruling 2026-08-13: "Don't use those 5 strategy.
        # Just a asking skill.") -- the five-token menu was a standing list of reasons
        # to message the user, and the token names alone activated asking. A Clarify is
        # valid with Content alone; a Strategy line, if an arm still emits one, is
        # recorded but never required and never judged.
        content = RE_CONTENT.search(raw)
        if not content:
            return AgentAction(ActionKind.MALFORMED, raw, problems=("missing_content",))
        cands = ()
        cm = RE_CANDIDATES.search(raw)
        if cm:
            cands = tuple(c.strip().strip("'\"") for c in cm.group(1).split(",") if c.strip())
        strategy = RE_STRATEGY.search(raw)
        name = strategy.group(1).strip() if strategy else None
        return AgentAction(ActionKind.ASK, raw, question=content.group(1).strip(),
                           strategy=name, candidates=cands)

    # --- final answer ---------------------------------------------------
    ans = RE_ANSWER.search(raw)
    if ans:
        final = RE_FINAL.search(raw)
        pred = RE_PREDICTED.search(raw)
        if final:
            body = final.group(1).strip()
        else:
            # UNLABELLED ANSWER (harness fidelity fix, 2026-08-25 -- same family as the
            # tool-name reassembly below). The model committed to `Action: Answer` and
            # then wrote its answer as prose, omitting only the literal `Final Answer:`
            # label. The answer is right there; discarding it scores the label rather
            # than the work. Measured on retail B0: 48 complete answers discarded across
            # 19 glm-4.7 episodes, against 0 for gpt-5.5 and 0 for qwen3.7-max -- so the
            # requirement was deciding WHICH MODEL got scored, which is precisely the
            # unfairness the D28 parser fixes removed.
            body = raw[ans.end():]
            if pred:                       # the prediction line is not part of the answer
                body = body.replace(pred.group(0), "", 1)
            body = body.strip()
            # too short to be an answer -- treat as before rather than invent one
            if len(body) < 8:
                return AgentAction(ActionKind.MALFORMED, raw,
                                   problems=("missing_final_answer",))
        return AgentAction(ActionKind.PROPOSE, raw, proposal_raw=body,
                           predicted_question=pred.group(1).strip() if pred else None)

    # --- an environment action ------------------------------------------
    block = RE_SQL_BLOCK.search(raw)
    named = RE_OPERATION_NAMED.search(raw)
    if RE_OPERATION.search(raw) or named or block:
        if not block:
            return AgentAction(ActionKind.MALFORMED, raw, problems=("missing_code_block",))
        cmd = block.group(1).strip()
        # TOOL-NAME-ON-THE-ACTION-LINE (harness fidelity fix, 2026-08-24). Some models
        # write `Action: get_reservation_details` + a block holding only the ARGUMENTS,
        # instead of `Action: Operation` + `tool(args)`. Both express the same call, but
        # the old parser kept the block verbatim, silently dropping the function name --
        # the call then failed forever and the model looped. Measured on glm-4.7: 63.5
        # tool calls per episode, 52% of episodes hitting the turn cap, Success 24%
        # while its intent and arguments were correct. Reassemble the call instead;
        # this changes NO mechanism and no method, only how faithfully we read a model's
        # output. A block that already names a callable is untouched.
        # NEVER treat the wire format's own keywords as a tool name (regression fixed
        # 2026-08-25): `Action: Operation` + a non-callable body (a WebShop-style
        # `search[...]`, a bare SQL statement) was being wrapped into
        # `Operation(search[...])`, which can only fail. Measured on glm-4.7 retail:
        # 602 "could not read arguments" errors across 555 episodes.
        _RESERVED = {"operation", "answer", "clarify"}
        if named and named.group(1).strip().lower() not in _RESERVED \
                and not RE_CALLABLE.match(cmd):
            # arguments may be one per line with no commas (measured on glm-4.7:
            # `origin="IAH"\ndestination="JFK"` -- joining verbatim produced invalid
            # call syntax, the call failed, and one episode retried it 91 times).
            args = ", ".join(l.strip().rstrip(",") for l in cmd.splitlines() if l.strip())
            cmd = f"{named.group(1).strip()}({args})"
        return AgentAction(ActionKind.ACT, raw, command=cmd)

    return AgentAction(ActionKind.MALFORMED, raw, problems=("no_recognised_action",))


# WIRE FORMAT ONLY -- deliberately strategy-free (2026-08-12).
#
# This text is prepended to EVERY arm's system prompt, so anything in it is given to the
# plain baseline as well. It previously taught clarification: "If the request is ambiguous,
# incomplete, or looks wrong, ask the user", plus a five-way menu of named strategies WITH
# DEFINITIONS ("Ask_Parameter: ask for a missing detail", "Disambiguate: offer options and
# ask the user to choose", ...). That is a compact clarification playbook handed to an arm
# whose whole point is having no strategy -- and it pre-installs most of what AT-CoT (B1)
# contributes, for everyone.
#
# What CANNOT be removed: `parse_action` requires the `Strategy:` line and requires its value
# to be one of CLARIFY_STRATEGIES (a Clarify without it is MALFORMED and costs patience), and
# the format is inherited verbatim from Drift-Bench on purpose. So the field survives as a
# bare syntactic requirement with no definitions. The token NAMES remain mildly suggestive;
# that residual is identical for every arm and is disclosed rather than fixed.
PROMPT_FORMAT = """You are helping a user. Reply with exactly ONE action per turn.

1. To interact with the environment:
Action: Operation
```
<one command>
```

2. To send the user a message:
Action: Clarify
Content: <your message>
Candidates: [<option1>, <option2>]   # optional

3. To give your final answer:
Action: Answer
Predicted user question: <what you believe the user actually wants>
Final Answer: <your answer>
"""

# B0's menu (ruling 2026-08-13): a REGULAR agent does not ask -- the user-message channel is
# not in its action space at all, not merely un-coached. Its purchases serve as proposals
# (the buy-shaped Operation path adjudicates them), the user's rejection is the only
# feedback it ever receives, and when it cannot afford another rejection the next purchase
# is the final submission. Should the model hallucinate a Clarify anyway, the Strategy-token
# requirement it has never seen makes that reply MALFORMED rather than a message.
PROMPT_FORMAT_NO_ASK = """You are helping a user. Reply with exactly ONE action per turn.

1. To interact with the environment:
Action: Operation
```
<one command>
```

2. To give your final answer:
Action: Answer
Predicted user question: <what you believe the user actually wants>
Final Answer: <your answer>
"""


@dataclass
class Transcript:
    """What the agent has been told. The graph is never in here."""

    system: str = PROMPT_FORMAT
    turns: list[dict] = field(default_factory=list)

    def user(self, text: str) -> None:
        self.turns.append({"role": "user", "content": text})

    def agent(self, text: str) -> None:
        self.turns.append({"role": "agent", "content": text})

    def env(self, text: str) -> None:
        self.turns.append({"role": "environment", "content": text})

    def as_messages(self) -> list[dict]:
        return [{"role": "system", "content": self.system}, *self.turns]
