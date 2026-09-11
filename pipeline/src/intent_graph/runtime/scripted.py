"""Deterministic agents, and the deterministic user stubs they need.

These are how the runtime is validated.  Each agent embodies one behaviour whose required
outcome is known in advance, so a silently-broken loop cannot pass: an oracle must win, an
agent that ignores a real intent change must lose, and an agent whose answer was made stale
by a *non*-moving change must still win.

The battery is not LLM-free -- the user still selects and phrases -- so a deterministic
selector stub is supplied here that maps these agents' fixed questions to slot ids.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from ..models import GroundTruth, Node
from .agent_api import Transcript
from .llm import ROLE_SELECT, FakeLLM

ASK_MARKER = "ASK_SLOT"


def stub_llm() -> FakeLLM:
    """A user whose slot selection is mechanical and whose phrasing echoes the value.

    The scripted agents ask ``"... ASK_SLOT slot_3 ..."``, so selection needs no
    understanding -- which is what makes battery outcomes deterministic.
    """
    def responder(role, system, prompt):
        if role == ROLE_SELECT:
            marker = f"{ASK_MARKER} "
            if marker in prompt:
                wanted = prompt.split(marker, 1)[1].split()[0].strip("'\".,")
                return wanted if wanted in prompt else "NONE"
            return "NONE"
        return "ok: " + prompt.split("What to tell it:")[-1].strip()[:200] if "What to tell it:" in prompt else "ok"
    return FakeLLM(responder=responder)


def answer_block(value: Any, predicted: str = "the user's request") -> str:
    return (f"Action: Answer\nPredicted user question: {predicted}\n"
            f"Final Answer: {value}\n")


def clarify_block(slot_id: str, strategy: str = "Ask_Parameter") -> str:
    return (f"Action: Clarify\nStrategy: {strategy}\n"
            f"Content: Could you tell me about {ASK_MARKER} {slot_id} please?\n")


def operation_block(command: str) -> str:
    return f"Action: Operation\n```\n{command}\n```\n"


def _gt_member(node: Node) -> Any:
    """A perfect answer for this node, formatted the way an agent would submit it.

    Note the two different notions of "the answer", which follow the benchmarks:
      * rowset (DBBench reads): the answer is the COMPLETE result set -- `compare_results`
        compares every value, so submitting one row of many is wrong.
      * purchaseset (WebShop): the answer is ONE purchase, and many are equally valid.
    """
    gt: GroundTruth = node.ground_truth
    if gt.kind == "rowset":
        rows = gt.rows()
        return json.dumps([v for row in rows for v in row])
    if gt.kind == "purchaseset":
        first = json.loads(sorted(gt.value)[0])
        asin, opts = first[0], dict(first[1])
        return f"buy {asin} {json.dumps(opts)}" if opts else f"buy {asin}"
    if gt.kind == "scalar":
        return gt.value
    if gt.kind == "statehash":
        # For a write task the answer IS the statement: executing it is what produces the
        # ground-truth state hash. The node's own recipe is therefore the perfect answer.
        recipe = node.recipe
        if isinstance(recipe, dict) and recipe.get("sql"):
            return recipe["sql"]
        return None
    return None


def has_multiple_valid_answers(node: Node) -> bool:
    """Only WebShop admits several genuinely different correct answers.

    A DBBench read has exactly one correct answer (the whole result set), even when that
    set has many rows -- so `second_valid` does not apply there.
    """
    gt = node.ground_truth
    return gt.kind == "purchaseset" and gt.cardinality() > 1


def _second_gt_member(node: Node) -> Any:
    """A DIFFERENT valid answer, where more than one exists."""
    gt = node.ground_truth
    if gt.kind == "purchaseset" and len(gt.value) > 1:
        first = json.loads(sorted(gt.value)[1])
        asin, opts = first[0], dict(first[1])
        return f"buy {asin} {json.dumps(opts)}" if opts else f"buy {asin}"
    return None


@dataclass
class ScriptedAgent:
    """Base: subclasses implement :meth:`plan` over the *hidden* state.

    A scripted agent is allowed to see the graph -- it is a test fixture standing in for a
    policy, not a policy under evaluation.
    """

    graph: Any
    name: str = "scripted"
    asked: list[str] = field(default_factory=list)
    _turn: int = 0

    # the episode calls this; the graph/nodes are injected for scripting purposes
    def act(self, transcript: Transcript) -> str:
        self._turn += 1
        return self.plan(transcript, self._turn)

    def plan(self, transcript: Transcript, turn: int) -> str:  # pragma: no cover
        raise NotImplementedError

    # -- helpers -----------------------------------------------------------
    _current: Node | None = None

    def current_node(self) -> Node:
        """The node the *user* currently holds.

        A scripted agent is permitted to know this: it is a fixture standing in for a
        policy, not a policy under evaluation.  The episode supplies it via observe_node
        before each turn; without that the oracle would answer the ROOT intent forever and
        would fail every episode in which the intent moved -- i.e. it would silently become
        the intent-ignorer.
        """
        return self._current or self.graph.root

    def observe_node(self, node: Node) -> None:
        self._current = node


class Oracle(ScriptedAgent):
    """Asks about every hidden slot, then answers the current node. Must always win.

    It is *persistent*: a declined slot is asked again, which is what a competent agent
    would do and what the `min_reveal_after_n_asks` override exists to reward.  Without
    persistence an avoidant user costs the oracle its recovery score even though it still
    wins, which would misreport recovery as a property of the environment.
    """

    name = "oracle"
    max_retries_per_slot: int = 3

    def __init__(self, graph, hidden_slot_ids: list[str], **kw):
        super().__init__(graph=graph, **kw)
        self.slot_ids = list(hidden_slot_ids)
        self._pending = list(hidden_slot_ids)
        self._tries: dict[str, int] = {}

    def plan(self, transcript, turn):
        while self._pending:
            sid = self._pending[0]
            n = self._tries.get(sid, 0)
            if n >= self.max_retries_per_slot:
                self._pending.pop(0)          # give up on this one, move on
                continue
            self._tries[sid] = n + 1
            self.asked.append(sid)
            return clarify_block(sid)
        member = _gt_member(self.current_node())
        return answer_block(member if member is not None else "unknown")

    def observe_reveal(self, slots) -> None:
        """The episode tells us which slots were actually revealed."""
        for s in slots:
            if s in self._pending:
                self._pending.remove(s)


class IntentIgnorer(ScriptedAgent):
    """Answers the ROOT intent regardless of any shift. Must lose when the answer moved."""

    name = "intent_ignorer"

    def observe_node(self, node):
        pass          # deliberately deaf to the shift -- that is the behaviour under test

    def plan(self, transcript, turn):
        if turn == 1:
            return clarify_block("slot_0")
        return answer_block(_gt_member(self.graph.root))


class NoOp(ScriptedAgent):
    """Never proposes. Must never succeed."""

    name = "no_op"

    def plan(self, transcript, turn):
        return "I am thinking about it."      # malformed: no recognised action


class PrematureActor(ScriptedAgent):
    """Mutates the environment before asking anything. Must be flagged."""

    name = "premature_actor"

    def __init__(self, graph, command: str, **kw):
        super().__init__(graph=graph, **kw)
        self.command = command

    def plan(self, transcript, turn):
        if turn == 1:
            return operation_block(self.command)
        return answer_block(_gt_member(self.current_node()) or "unknown")


class SecondValid(ScriptedAgent):
    """Proposes a DIFFERENT member of the ground-truth set. Must be accepted."""

    name = "second_valid"

    def plan(self, transcript, turn):
        alt = _second_gt_member(self.current_node())
        return answer_block(alt if alt is not None else _gt_member(self.current_node()))


class DestructiveRejected(ScriptedAgent):
    """Proposes a wrong, destructive write. Must be rejected AND leave the env untouched."""

    name = "destructive_rejected"

    def __init__(self, graph, statement: str, **kw):
        super().__init__(graph=graph, **kw)
        self.statement = statement

    def plan(self, transcript, turn):
        return answer_block(self.statement)
