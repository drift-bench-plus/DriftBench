"""AgentBench DBBench adapter.

The recipe is SQL.  Read tasks ship a reference query whose result must reproduce the
``label``; write tasks ship a gold statement whose post-execution table state must
reproduce ``answer_md5``.  Both are re-executable, which is what makes their ground truth
move mechanically when the intent moves.

Condition slots are namespaced so the two halves of a write statement can both be
mutated without ambiguity:

    where:<col>   a WHERE predicate      (which rows the intent is about)
    set:<col>     an UPDATE assignment   (what it changes them to)
    value:<col>   an INSERT column value (what it adds)

Environment = one loaded table (identity includes the rows, so two tasks over the same
schema but different data are different worlds).
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path

from ..executors.mysql_docker import MySQLExecutor
from ..models import Condition, GroundTruth, Seed
from . import base as registry
from .dbbench_compare import combine_table_hashes, compare_results, table_hash_sql

log = logging.getLogger(__name__)

WRITE_TYPES = {"INSERT", "UPDATE", "DELETE"}

SELECT_RE = re.compile(
    r"^\s*SELECT\s+(?P<sel>.+?)\s+FROM\s+(?P<tbl>`[^`]+`|\"[^\"]+\"|\[[^\]]+\]|[\w$]+)"
    r"(?:\s+WHERE\s+(?P<where>.*?))?\s*;?\s*$",
    re.I | re.S,
)
UPDATE_RE = re.compile(
    r"^\s*UPDATE\s+(?P<tbl>`[^`]+`|[\w$]+)\s+SET\s+(?P<set>.+?)"
    r"(?:\s+WHERE\s+(?P<where>.*?))?\s*;?\s*$",
    re.I | re.S,
)
DELETE_RE = re.compile(
    r"^\s*DELETE\s+FROM\s+(?P<tbl>`[^`]+`|[\w$]+)"
    r"(?:\s+WHERE\s+(?P<where>.*?))?\s*;?\s*$",
    re.I | re.S,
)
INSERT_RE = re.compile(
    r"^\s*INSERT\s+INTO\s+(?P<tbl>`[^`]+`|[\w$]+)\s*\((?P<cols>[^)]*)\)\s*"
    r"VALUES\s*\((?P<vals>.*)\)\s*;?\s*$",
    re.I | re.S,
)

# one simple comparison: `col` op literal   (no OR, no nesting -- those stay unparsed)
PRED_RE = re.compile(
    r"^\s*(?P<col>`[^`]+`|\"[^\"]+\"|[\w$ ()/%+.-]+?)\s*"
    r"(?P<op><=|>=|!=|<>|=|<|>|(?:NOT\s+)?LIKE)\s*"
    r"(?P<val>'(?:[^']|'')*'|\"[^\"]*\"|-?[\d.]+)\s*$",
    re.I,
)


def _unquote_ident(s: str) -> str:
    return s.strip().strip("`\"[]").strip()


def _unquote_literal(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "'\"":
        return s[1:-1].replace("''", "'")
    return s


def _quote_literal(v) -> str:
    return "'" + str(v).replace("'", "''") + "'"


def _split_top_level(text: str, sep: str) -> list[str]:
    """Split on ``sep`` outside quotes and parentheses."""
    out, depth, in_q, cur, i = [], 0, None, "", 0
    pat = sep.lower()
    while i < len(text):
        ch = text[i]
        if in_q:
            cur += ch
            if ch == in_q:
                in_q = None
            i += 1
            continue
        if ch in "'\"":
            in_q, cur = ch, cur + ch
            i += 1
            continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if depth == 0 and text[i:i + len(sep)].lower() == pat:
            out.append(cur)
            cur = ""
            i += len(sep)
            continue
        cur += ch
        i += 1
    out.append(cur)
    return [p for p in (s.strip() for s in out) if p]


def parse_predicates(where: str | None, prefix: str) -> tuple[Condition, ...] | None:
    """WHERE text -> conditions, or None when it is beyond simple AND-of-comparisons."""
    if not where or not where.strip():
        return ()
    if re.search(r"\bOR\b|\bSELECT\b|\bBETWEEN\b|\bIN\s*\(|\bIS\b", where, re.I):
        return None  # unparseable: keep as a pivot target rather than guess
    conds = []
    for part in _split_top_level(where, " AND "):
        m = PRED_RE.match(part)
        if not m:
            return None
        conds.append((f"{prefix}{_unquote_ident(m['col'])}",
                      m["op"].upper().replace("<>", "!="),
                      _unquote_literal(m["val"])))
    return tuple(conds)


def parse_sql(sql: str) -> tuple[dict, tuple[Condition, ...]] | None:
    """Split a statement into an immutable base frame and mutable conditions."""
    if not sql:
        return None
    text = sql.strip()

    m = SELECT_RE.match(text)
    if m and not re.match(r"^\s*(INSERT|UPDATE|DELETE)", text, re.I):
        conds = parse_predicates(m["where"], "where:")
        if conds is None:
            return None
        return {"kind": "SELECT", "table": _unquote_ident(m["tbl"]),
                "select": m["sel"].strip()}, conds

    m = UPDATE_RE.match(text)
    if m:
        where = parse_predicates(m["where"], "where:")
        if where is None:
            return None
        sets = []
        for part in _split_top_level(m["set"], ","):
            sm = PRED_RE.match(part)
            if not sm or sm["op"] != "=":
                return None
            sets.append((f"set:{_unquote_ident(sm['col'])}", "=", _unquote_literal(sm["val"])))
        return {"kind": "UPDATE", "table": _unquote_ident(m["tbl"])}, tuple(sets) + where

    m = DELETE_RE.match(text)
    if m:
        where = parse_predicates(m["where"], "where:")
        if where is None or not where:
            return None  # an unconditional DELETE has no intent to move
        return {"kind": "DELETE", "table": _unquote_ident(m["tbl"])}, where

    m = INSERT_RE.match(text)
    if m:
        cols = [_unquote_ident(c) for c in _split_top_level(m["cols"], ",")]
        vals = [_unquote_literal(v) for v in _split_top_level(m["vals"], ",")]
        if not cols or len(cols) != len(vals):
            return None
        return ({"kind": "INSERT", "table": _unquote_ident(m["tbl"])},
                tuple((f"value:{c}", "=", v) for c, v in zip(cols, vals, strict=True)))
    return None


def compile_sql(base: dict, conditions: tuple[Condition, ...]) -> str:
    kind, table = base["kind"], base["table"]
    where = [(s.split(":", 1)[1], o, v) for s, o, v in conditions if s.startswith("where:")]
    sets = [(s.split(":", 1)[1], o, v) for s, o, v in conditions if s.startswith("set:")]
    values = [(s.split(":", 1)[1], o, v) for s, o, v in conditions if s.startswith("value:")]

    def where_sql() -> str:
        if not where:
            return ""
        return " WHERE " + " AND ".join(f"`{c}` {o} {_quote_literal(v)}" for c, o, v in where)

    if kind == "SELECT":
        return f"SELECT {base['select']} FROM `{table}`{where_sql()};"
    if kind == "UPDATE":
        assigns = ", ".join(f"`{c}` = {_quote_literal(v)}" for c, _, v in sets)
        return f"UPDATE `{table}` SET {assigns}{where_sql()};"
    if kind == "DELETE":
        return f"DELETE FROM `{table}`{where_sql()};"
    if kind == "INSERT":
        cols = ", ".join(f"`{c}`" for c, _, _ in values)
        vals = ", ".join(_quote_literal(v) for _, _, v in values)
        return f"INSERT INTO `{table}` ({cols}) VALUES ({vals});"
    raise ValueError(f"unknown statement kind {kind!r}")


class DBBenchAdapter:
    name = "dbbench"
    version = "1"
    executor_name = "mysql_docker"

    FILES = ("data/dbbench/db_out_new.jsonl", "data/dbbench/standard.jsonl",
             "data/dbbench/extend_std.json")

    def __init__(self, config: dict) -> None:
        self.root = Path(config["paths"]["agentbench_repo"])
        self.config = config
        self._envs: dict[str, dict] = {}

    # ------------------------------------------------------------------ loading
    def _records(self):
        for rel in self.FILES:
            path = self.root / rel
            if not path.exists():
                log.warning("missing dataset file %s", path)
                continue
            if path.suffix == ".jsonl":
                for line in path.open(encoding="utf-8"):
                    if line.strip():
                        yield json.loads(line)
            else:
                yield from json.load(path.open(encoding="utf-8"))

    def load(self) -> list[Seed]:
        seeds, seen = [], set()
        for entry in self._records():
            tables = entry["table"] if isinstance(entry["table"], list) else [entry["table"]]
            if len(tables) != 1:
                continue  # multi-table environments: out of scope for v1
            types = set(entry.get("type") or [])
            is_write = bool(types & WRITE_TYPES)
            gold = (entry.get("sql") or {}).get("query") or ""
            label = (entry.get("label") or [None])[0]
            statement = gold if gold else (label if is_write else "")
            if not statement:
                continue

            dedup_key = hashlib.sha256(
                (entry.get("description", "") + "||" + statement).encode()
            ).hexdigest()
            if dedup_key in seen:
                continue
            seen.add(dedup_key)

            table = tables[0]
            columns = [c["name"] for c in table["table_info"]["columns"]]
            rows = table["table_info"]["rows"]
            env_key = hashlib.sha256(
                json.dumps([table["table_name"], columns, rows], sort_keys=True,
                           default=str).encode()
            ).hexdigest()[:24]
            self._envs[env_key] = {
                "env_id": env_key,
                "kind": "mysql",
                "tables": [{"table_name": table["table_name"], "columns": columns,
                            "rows": rows}],
            }

            parsed = parse_sql(statement)
            base, conds = (parsed if parsed else ({"kind": "UNPARSED",
                                                   "table": table["table_name"]}, None))
            seeds.append(Seed(
                record_id=dedup_key[:16],
                env_key=env_key,
                base=base,
                conditions=conds,
                shipped_answer=(entry.get("answer_md5") if is_write else entry.get("label")),
                meta={"type": sorted(types), "is_write": is_write,
                      "description": entry.get("description", "")},
            ))
        return seeds

    def env_spec(self, env_key: str) -> dict:
        return self._envs[env_key]

    # ------------------------------------------------------------- the contract
    def compile(self, base: dict, conditions: tuple[Condition, ...]):
        return {"sql": compile_sql(base, conditions), "kind": base["kind"],
                "table": base["table"]}

    def is_mutating(self, recipe) -> bool:
        return recipe["kind"] in WRITE_TYPES

    AGGREGATE_RE = re.compile(r"\b(COUNT|SUM|AVG|MIN|MAX|GROUP_CONCAT)\s*\(", re.I)

    def gt_extensional(self, recipe) -> bool:
        """Is the answer the matching rows themselves, or a value computed from them?

        ``SELECT name FROM t WHERE ...`` returns the rows, so narrowing the intent must
        shrink the answer.  ``SELECT COUNT(*) ...`` returns one row whose *value* moves --
        no subset relation holds, and asserting one would flag correct graphs.
        """
        if recipe["kind"] != "SELECT":
            return False
        sql = recipe["sql"]
        return not (self.AGGREGATE_RE.search(sql) or re.search(r"\bLIMIT\b|\bDISTINCT\b", sql, re.I))

    def execute(self, recipe, session) -> GroundTruth:
        if recipe["kind"] in WRITE_TYPES:
            session.run(recipe, mutating=True)
            return GroundTruth.statehash(self._state_hash(session))
        rows = session.run(recipe)
        return GroundTruth.rowset(rows)

    def _state_hash(self, session) -> str:
        hashes = []
        for table in session.env_spec["tables"]:
            h = session.scalar(table_hash_sql(table["table_name"], table["columns"]))
            hashes.append(str(h) if h is not None else "")
        return combine_table_hashes(hashes)

    def validate(self, seed: Seed, session) -> bool:
        if not seed.parseable or seed.shipped_answer in (None, "", []):
            return False
        recipe = self.compile(seed.base, seed.conditions)
        if seed.meta["is_write"]:
            gt = self.execute(recipe, session)
            expected = str(seed.shipped_answer)
            got = gt.value
            # answer_md5 is stored as the raw cursor repr, e.g. "[('abc12',)]"
            m = re.search(r"[0-9a-f]{32}", expected)
            return bool(m) and m.group(0) == got if m else expected.find(got) >= 0
        rows = session.run(recipe)
        if not rows:
            return False
        return compare_results([list(r) for r in rows], seed.shipped_answer, "SELECT")

    def gt_equal(self, a: GroundTruth, b: GroundTruth) -> bool:
        """Use DBBench's own comparison (1e-2 float tolerance) for read answers."""
        if a.kind != b.kind:
            return False
        if a.kind == "rowset":
            return compare_results(a.value, b.value, "SELECT")
        return a.hash == b.hash

    # ------------------------------------------------------------- acceptance
    WRITE_FIRST_KEYWORDS = frozenset({"INSERT", "UPDATE", "DELETE", "DROP", "ALTER",
                                      "CREATE", "TRUNCATE", "REPLACE", "GRANT"})

    def act_is_mutating(self, raw_action: str) -> bool:
        """Does an AGENT-issued statement change state? Unknown => True (fail closed).

        Distinct from is_mutating(), which describes OUR reference recipes.
        """
        text = re.sub(r"/\*.*?\*/", " ", str(raw_action or ""), flags=re.S)
        text = re.sub(r"--[^\n]*", " ", text).strip()
        m = re.match(r"[(\s]*([A-Za-z]+)", text)
        if not m:
            return True
        head = m.group(1).upper()
        if head in ("SELECT", "SHOW", "DESCRIBE", "DESC", "EXPLAIN", "WITH"):
            # a SELECT can still write, via INTO OUTFILE
            return bool(re.search(r"\bINTO\s+(OUT|DUMP)FILE\b", text, re.I))
        return True   # a write keyword, or something unrecognised: fail closed either way

    def describe_pivot(self, base: dict) -> str:
        kind = (base or {}).get("kind", "question")
        return ("I've changed my mind — I want you to change the data instead of just "
                "looking it up." if kind in WRITE_TYPES else
                "I've changed my mind — I want to know something else about this table.")

    STATEMENT_HEAD = re.compile(
        r"^\s*(INSERT|UPDATE|DELETE|REPLACE|TRUNCATE|DROP|ALTER|CREATE|SELECT|WITH)\b", re.I)

    def parse_proposal(self, raw: str):
        """Read tasks propose values; write tasks propose a statement.

        The split must be POSITIVE ("does this look like SQL?"), not the fail-closed
        `act_is_mutating` used for actions: that returns True for anything unrecognised, so
        a perfectly good JSON answer would be misrouted as a statement and rejected.
        """
        text = str(raw).strip()
        if self.STATEMENT_HEAD.match(text):
            return {"sql": text}
        try:
            val = json.loads(text)
            return val if isinstance(val, list) else [val]
        except Exception:
            return [t.strip().strip("'\"") for t in text.split(",") if t.strip()]

    def accepts(self, proposal, node, session, *, executor=None) -> tuple[bool, str]:
        gt = node.ground_truth
        if gt.kind == "rowset":
            # gt.value holds canonical-JSON row strings; comparing raw objects against them
            # rejects every correct answer, so decode first and use DBBench's own comparison
            ok = compare_results(proposal, gt.rows(), "SELECT")
            return ok, "compare_results" if ok else "compare_results_mismatch"
        if gt.kind == "statehash":
            stmt = proposal.get("sql") if isinstance(proposal, dict) else str(proposal)
            if executor is None:
                return False, "write_proposal_needs_executor_for_scratch_session"
            # evaluate on a throwaway database so a rejected (possibly destructive) proposal
            # cannot dirty the episode's environment
            with executor.open(session.env_spec) as scratch:
                try:
                    scratch.run({"sql": stmt, "kind": "UPDATE", "table": node.base["table"]})
                except Exception as exc:
                    return False, f"write_failed:{type(exc).__name__}"
                got = self._state_hash(scratch)
            ok = str(got) == str(gt.value)
            return ok, "state_hash" if ok else "state_hash_mismatch"
        return False, f"unsupported_gt_kind:{gt.kind}"

    def witness(self, base: dict, conditions: tuple[Condition, ...], session) -> list[dict]:
        """Rows the intent is about -- for writes, the rows the statement would touch (D13)."""
        where = [c for c in conditions if c[0].startswith("where:")]
        probe = compile_sql({"kind": "SELECT", "table": base["table"], "select": "*"},
                            tuple(where))
        rows = session.run({"sql": probe, "kind": "SELECT", "table": base["table"]})
        columns = session.env_spec["tables"][0]["columns"]
        prefix = "value:" if base["kind"] == "INSERT" else "where:"
        return [{f"{prefix}{c}": v for c, v in zip(columns, row, strict=False)}
                for row in rows[:200]]

    def domains(self, slot: str, base: dict, session) -> list:
        column = slot.split(":", 1)[1]
        sql = f"SELECT DISTINCT `{column}` FROM `{base['table']}` LIMIT 50;"
        try:
            rows = session.run({"sql": sql, "kind": "SELECT", "table": base["table"]})
        except Exception:
            return []
        return [r[0] for r in rows if r and r[0] is not None]


registry.register("dbbench", DBBenchAdapter, MySQLExecutor)
