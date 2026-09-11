"""AgentBench os_interaction adapter.

The recipe is the shell command AgentBench ships as ``evaluation.example``.  That command
is the reference solution: at eval time the official harness runs it and compares its
stdout against the agent's answer, so its output *is* the ground truth.  Editing the
command therefore moves the answer exactly the way editing SQL does.

Conditions are the command's predicate flags:

    find:  ("find:type", "=", "f") ("find:mtime", "<", "7") ("find:size", ">", "100k")
    grep:  ("grep:pattern", "=", "Linux") ("grep:word", "=", "true") ...

Environment = the filesystem the ``create.init`` script builds (identity = hash of that
script), so two tasks with the same setup share a world and can pivot in place.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shlex
from pathlib import Path

from ..executors.os_docker import OSExecutor
from ..models import Condition, GroundTruth, Seed
from . import base as registry

log = logging.getLogger(__name__)

# comparator semantics, ported from data/os_interaction/scripts/*/check/*.py
_SIZE_UNITS = {"B": 1, "K": 1024, "KB": 1024, "M": 1024**2, "MB": 1024**2,
               "G": 1024**3, "GB": 1024**3, "T": 1024**4, "TB": 1024**4}


def integer_match(answer: str, reference: str) -> bool:
    try:
        return int(str(answer).strip()) == int(str(reference).strip())
    except (ValueError, TypeError):
        return False


def string_match(answer: str, reference: str) -> bool:
    def norm(s):
        return str(s).replace("\r\n", "\n").replace("\r", "\n").strip()
    return norm(answer) == norm(reference)


def size_match(answer: str, reference: str) -> bool:
    def parse(s):
        m = re.match(r"^\s*([\d.]+)\s*([A-Za-z]*)\s*$", str(s))
        if not m:
            return None
        return float(m.group(1)) * _SIZE_UNITS.get(m.group(2).upper() or "B", 1)
    a, b = parse(answer), parse(reference)
    return a is not None and b is not None and a == b


COMPARATORS = {"integer-match.py": integer_match, "string-match.py": string_match,
               "size-match.py": size_match}

# find/grep predicate flags we know how to move
FIND_FLAGS = {"-type": ("find:type", "="), "-name": ("find:name", "="),
              "-iname": ("find:iname", "="), "-mtime": ("find:mtime", "@"),
              "-size": ("find:size", "@"), "-maxdepth": ("find:maxdepth", "="),
              "-perm": ("find:perm", "="), "-user": ("find:user", "=")}
GREP_SWITCHES = {"-i": "grep:ignorecase", "-w": "grep:word", "-v": "grep:invert",
                 "-r": "grep:recursive", "-c": "grep:count", "-l": "grep:fileslist"}


def parse_command(cmd: str) -> tuple[dict, tuple[Condition, ...]] | None:
    """Split a reference command into an immutable base and mutable flag conditions."""
    if not cmd or "\n" in cmd.strip() or ";" in cmd or "&&" in cmd:
        return None  # multi-statement scripts: keep as pivot targets, do not guess
    try:
        stages = [shlex.split(p.strip()) for p in cmd.split("|")]
    except ValueError:
        return None
    if not stages or not stages[0]:
        return None
    tool = stages[0][0]
    if tool not in ("find", "grep"):
        return None

    conds: list[Condition] = []
    tail = [" ".join(s) for s in stages[1:]]

    if tool == "find":
        tokens, paths, i = stages[0][1:], [], 0
        while i < len(tokens):
            t = tokens[i]
            if t in FIND_FLAGS and i + 1 < len(tokens):
                slot, op = FIND_FLAGS[t]
                value = tokens[i + 1]
                if op == "@":  # -mtime -7 / -size +100k carry their comparator inline
                    sign = value[0] if value and value[0] in "+-" else "="
                    conds.append((slot, {"+": ">", "-": "<", "=": "="}[sign],
                                  value.lstrip("+-")))
                else:
                    conds.append((slot, "=", value))
                i += 2
                continue
            if t.startswith("-"):
                return None  # an option we do not model: refuse rather than mangle
            paths.append(t)
            i += 1
        if not paths:
            return None
        return {"tool": "find", "path": paths[0], "tail": tail}, tuple(conds)

    # grep
    tokens, operands, i = stages[0][1:], [], 0
    while i < len(tokens):
        t = tokens[i]
        if t in GREP_SWITCHES:
            conds.append((GREP_SWITCHES[t], "=", "true"))
            i += 1
            continue
        if t.startswith("-") and len(t) > 1 and all(f"-{c}" in GREP_SWITCHES for c in t[1:]):
            for c in t[1:]:
                conds.append((GREP_SWITCHES[f"-{c}"], "=", "true"))
            i += 1
            continue
        if t.startswith("-"):
            return None
        operands.append(t)
        i += 1
    if len(operands) < 2:
        return None
    conds.append(("grep:pattern", "=", operands[0]))
    return {"tool": "grep", "path": " ".join(operands[1:]), "tail": tail}, tuple(conds)


def compile_command(base: dict, conditions: tuple[Condition, ...]) -> str:
    tool, tail = base["tool"], base.get("tail") or []
    if tool == "find":
        parts = ["find", base["path"]]
        for slot, op, value in sorted(conditions):
            flag = next((f for f, (s, _) in FIND_FLAGS.items() if s == slot), None)
            if flag is None:
                continue
            if slot in ("find:mtime", "find:size"):
                sign = {">": "+", "<": "-", "=": ""}[op]
                parts += [flag, f"{sign}{value}"]
            else:
                parts += [flag, shlex.quote(str(value))]
    else:
        switches = [f for f, s in ((f, s) for f, s in GREP_SWITCHES.items())
                    if any(c[0] == s and str(c[2]).lower() == "true" for c in conditions)]
        pattern = next((c[2] for c in conditions if c[0] == "grep:pattern"), "")
        parts = ["grep", *sorted(switches), shlex.quote(str(pattern)), base["path"]]
    cmd = " ".join(parts)
    return " | ".join([cmd, *tail]) if tail else cmd


class OSBenchAdapter:
    name = "osbench"
    version = "1"
    executor_name = "os_docker"

    def __init__(self, config: dict) -> None:
        self.root = Path(config["paths"]["agentbench_repo"]) / "data" / "os_interaction"
        self._envs: dict[str, dict] = {}

    # ------------------------------------------------------------------ loading
    def _records(self):
        train = self.root / "train_0317" / "training.json"
        if train.exists():
            yield from json.load(train.open(encoding="utf-8"))
        # numbered eval dirs only: dev.json is the official dev split (and holds the
        # match-type tasks we cannot use), 6-backup.json is a duplicate
        for path in sorted((self.root / "data").glob("*/*.json")):
            data = json.load(path.open(encoding="utf-8"))
            yield from (data if isinstance(data, list) else [data])

    def load(self) -> list[Seed]:
        seeds, seen = [], set()
        for entry in self._records():
            ev = entry.get("evaluation") or {}
            example = ev.get("example")
            if not isinstance(example, str) or not example.strip():
                continue  # no reference command -> ground truth is a bare literal (Tier D)
            comparator = None
            for c in (ev.get("check") if isinstance(ev.get("check"), list) else [ev.get("check")]):
                if isinstance(c, dict) and c.get("file"):
                    comparator = Path(c["file"]).name
            if comparator not in COMPARATORS:
                continue

            init = (entry.get("create") or {}).get("init") or ""
            init_text = init if isinstance(init, str) else json.dumps(init, sort_keys=True)
            env_key = hashlib.sha256(re.sub(r"\s+", " ", init_text).strip().encode()).hexdigest()[:24]
            self._envs[env_key] = {"env_id": env_key, "kind": "os", "init": init_text}

            record_id = hashlib.sha256(
                (entry.get("description", "") + "||" + example).encode()
            ).hexdigest()[:16]
            if record_id in seen:
                continue
            seen.add(record_id)

            parsed = parse_command(example)
            base, conds = parsed if parsed else ({"tool": "raw", "command": example}, None)
            # the comparator must live ON THE NODE: Node has no `meta`, and synthetic
            # children have no source_record to look it up from
            base = {**base, "comparator": comparator}
            seeds.append(Seed(
                record_id=record_id, env_key=env_key, base=base, conditions=conds,
                shipped_answer=None,   # ground truth is computed, never shipped
                meta={"comparator": comparator, "description": entry.get("description", ""),
                      "raw_example": example},
            ))
        return seeds

    def env_spec(self, env_key: str) -> dict:
        return self._envs[env_key]

    # ------------------------------------------------------------- the contract
    def compile(self, base: dict, conditions: tuple[Condition, ...]):
        cmd = base["command"] if base["tool"] == "raw" else compile_command(base, conditions)
        return {"command": cmd, "tool": base["tool"]}

    def is_mutating(self, recipe) -> bool:
        # reference commands are read-only queries (find/grep); anything else is refused
        return False

    READ_ONLY_TOOLS = frozenset({"find", "grep", "ls", "cat", "wc", "stat", "du", "head",
                                 "tail", "sort", "uniq", "cut", "tr", "file", "which",
                                 "echo", "test", "awk", "sed", "basename", "dirname"})
    MUTATING_MARKERS = (">", ">>", "|tee", "| tee", "rm ", "mv ", "cp ", "touch ", "mkdir ",
                        "chmod ", "chown ", "ln ", "dd ", "truncate ", "$(", "`", "-delete",
                        "-exec", "sed -i", "tee ")

    def act_is_mutating(self, raw_action: str) -> bool:
        """Fail closed: anything unrecognised is treated as mutating.

        Being wrong this way costs a shift opportunity; being wrong the other way corrupts
        ground truth. Note is_mutating() cannot be reused -- it describes OUR reference
        commands and is always False.
        """
        cmd = str(raw_action or "").strip()
        if not cmd:
            return True
        low = cmd.lower()
        if any(m in low for m in self.MUTATING_MARKERS):
            return True
        first = re.split(r"[\s|;&]+", cmd)[0]
        return first not in self.READ_ONLY_TOOLS

    def describe_pivot(self, base: dict) -> str:
        tool = (base or {}).get("tool", "something")
        return ("I've changed my mind — I want to search the file contents instead."
                if tool == "grep" else
                "I've changed my mind — I want to know something else about these files.")

    def parse_proposal(self, raw: str):
        return str(raw).strip()

    def accepts(self, proposal, node, session, *, executor=None) -> tuple[bool, str]:
        """Compare with the task's OWN matcher, read off the node's base."""
        name = (node.base or {}).get("comparator") or "string-match.py"
        cmp_fn = COMPARATORS.get(name, string_match)
        ok = cmp_fn(str(proposal), str(node.ground_truth.value))
        return ok, name if ok else f"{name}_mismatch"

    def gt_extensional(self, recipe) -> bool:
        # the answer is a count/size/string produced by an aggregation stage, never the
        # matching files themselves
        return False

    def execute(self, recipe, session) -> GroundTruth:
        return GroundTruth.scalar(session.run(recipe))

    def validate(self, seed: Seed, session) -> bool:
        """No shipped answer exists, so validity has to be established three ways:

        1. FIDELITY  our recompiled command must agree with the dataset's original.  We
           reorder flags and requote when compiling, so this is the check that proves the
           parser did not quietly change the task -- without it a parser bug becomes
           wrong ground truth with no symptom.
        2. USABLE    the answer is non-empty and the task's own comparator can parse it.
        3. STABLE    a fresh materialization gives the same answer, or the task is
           time/randomness sensitive and its ground truth would drift under us.
        """
        if not seed.parseable:
            return False
        cmp_fn = COMPARATORS[seed.meta["comparator"]]
        recipe = self.compile(seed.base, seed.conditions)
        try:
            ours = session.run(recipe)
            theirs = session.run({"command": seed.meta["raw_example"], "tool": "raw"})
        except Exception:
            return False
        if not str(ours).strip() or not cmp_fn(ours, ours):
            return False
        if not cmp_fn(ours, theirs):
            log.debug("fidelity mismatch for %s: %r vs %r", seed.record_id, ours, theirs)
            return False

        session.rematerialize()
        try:
            again = session.run(recipe)
        except Exception:
            return False
        return cmp_fn(ours, again)

    def gt_equal(self, a: GroundTruth, b: GroundTruth) -> bool:
        return a.hash == b.hash

    def witness(self, base: dict, conditions: tuple[Condition, ...], session) -> list[dict]:
        """The objects behind the count: strip the aggregation and inspect what is left."""
        if base["tool"] != "find":
            return self._grep_witness(base, conditions, session)
        listing = compile_command({**base, "tail": []}, conditions)
        out = session.probe(f"{listing} -printf '%p\\t%s\\t%T@\\n' 2>/dev/null || "
                            f"{listing} -exec stat -c '%n\\t%s\\t%Y' {{}} \\; 2>/dev/null")
        rows = []
        for line in out.splitlines()[:200]:
            parts = line.split("\t")
            if len(parts) != 3:
                continue
            path, size, _ = parts
            suffix = Path(path).suffix
            rows.append({
                "find:name": f"*{suffix}" if suffix else "*",
                "find:size": _size_bucket(int(float(size or 0))),
            })
        return rows

    def _grep_witness(self, base, conditions, session) -> list[dict]:
        """Word frequencies in the searched files: substitution/refinement material."""
        out = session.probe(
            f"cat {base['path']} 2>/dev/null | tr -cs '[:alnum:]' '\\n' | sort | uniq -c | "
            f"sort -rn | head -40"
        )
        rows = []
        for line in out.splitlines():
            m = re.match(r"\s*(\d+)\s+(\S+)$", line)
            if m and len(m.group(2)) > 2:
                rows.append({"grep:pattern": m.group(2)})
        return rows

    def domains(self, slot: str, base: dict, session) -> list:
        if slot == "grep:pattern":
            return [w["grep:pattern"] for w in self._grep_witness(base, (), session)][:20]
        if slot == "find:mtime":
            return ["1", "3", "7", "14", "30"]
        if slot == "find:size":
            return ["1k", "10k", "100k", "1M"]
        if slot == "find:type":
            return ["f", "d"]
        if slot.startswith("grep:"):
            return ["true", "false"]
        return []


def _size_bucket(nbytes: int) -> str:
    for limit, label in ((1024, "1k"), (10 * 1024, "10k"), (100 * 1024, "100k"),
                         (1024**2, "1M")):
        if nbytes < limit:
            return label
    return "big"


registry.register("osbench", OSBenchAdapter, OSExecutor)
