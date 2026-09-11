"""Port of AgentBench's DBResultProcessor comparison semantics.

Faithful to ``src/server/tasks/dbbench/result_processor.py`` at AgentBench d1e4a10,
minus its debug ``print`` calls.  We reuse the benchmark's own notion of "same answer"
rather than substituting ours (north-star #2): it treats None/null/""/nan as "0", strips
percent signs and thousands separators, compares floats with a 1e-2 tolerance, and falls
back to set equality for multi-row results.
"""

from __future__ import annotations

import ast

FLOAT_TOL = 1e-2

_SPECIAL = {
    "none": "0", "null": "0", "undefined": "0", "nan": "0",
    "inf": "0", "infinity": "0", "-inf": "0", "-infinity": "0", "": "0",
}


def normalize_special_values(value) -> str:
    if value is None:
        return "0"
    s = str(value).strip()
    if s.endswith("%"):
        s = s[:-1].strip()
    if "," in s and not (s.startswith("[") or s.endswith("]")):
        s = s.replace(",", "")
    return _SPECIAL.get(s.lower(), s)


def is_float(value) -> bool:
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False


def float_equal(a, b, tol: float = FLOAT_TOL) -> bool:
    try:
        return abs(float(a) - float(b)) <= tol
    except (ValueError, TypeError):
        return False


def _literal(text: str):
    """``eval`` in the original; ``literal_eval`` here -- same result for data, no exec."""
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return None


def clean_answer(answer) -> list[str]:
    """Flatten whatever shape an answer arrives in into a list of normalized strings."""
    if answer is None:
        return ["0"]
    if isinstance(answer, (list, tuple)):
        out = []
        for item in answer:
            if isinstance(item, (list, tuple)):
                out.extend(normalize_special_values(x) for x in item)
            else:
                out.append(normalize_special_values(item))
        return out
    if isinstance(answer, str):
        s = answer.strip()
        if s.startswith("[") and s.endswith("]"):
            parsed = _literal(s)
            if isinstance(parsed, list):
                out = []
                for item in parsed:
                    if isinstance(item, tuple) and len(item) == 1:
                        out.append(normalize_special_values(str(item[0]).strip().strip("'\"")))
                    elif isinstance(item, (tuple, list)):
                        out.extend(normalize_special_values(str(x).strip().strip("'\"")) for x in item)
                    else:
                        out.append(normalize_special_values(str(item).strip().strip("'\"")))
                return out
            # unparseable bracket text: split on commas outside quotes
            inner, items, current, in_quotes = s[1:-1], [], "", False
            for ch in inner:
                if ch in "\"'":
                    in_quotes = not in_quotes
                elif ch == "," and not in_quotes:
                    if current:
                        items.append(normalize_special_values(current.strip().strip("'\"")))
                    current = ""
                    continue
                current += ch
            if current:
                items.append(normalize_special_values(current.strip().strip("'\"")))
            return items
        return [normalize_special_values(s)]
    return [normalize_special_values(answer)]


def compare_results(answer, ground_truth, query_type: str) -> bool:
    """True iff DBBench would call these the same answer."""
    try:
        a = clean_answer(answer)
        g = clean_answer(ground_truth)

        if query_type in ("INSERT", "DELETE", "UPDATE"):
            return a == g

        if len(a) == 1 and len(g) == 1:
            av, gv = a[0], g[0]
            if av == "0" and gv == "0":
                return True
            if is_float(av) and is_float(gv):
                return float_equal(av, gv)
            return av == gv

        if all(is_float(x) for x in a) and all(is_float(x) for x in g):
            if len(a) != len(g):
                return False
            matched = [False] * len(g)
            for ans in a:
                for i, gt in enumerate(g):
                    if not matched[i] and float_equal(ans, gt):
                        matched[i] = True
                        break
                else:
                    return False
            return all(matched)

        return set(a) == set(g)
    except Exception:
        return False


def table_hash_sql(table_name: str, columns: list[str]) -> str:
    """DBBench's per-table state hash, verbatim in MySQL SQL.

    Run in MySQL rather than re-implemented in Python so the digest matches the
    ``answer_md5`` the dataset ships (MD5, CONCAT_WS and GROUP_CONCAT semantics are the
    server's, not ours).
    """
    cols = ",".join(f"`{c}`" for c in columns)
    return (
        "select md5(group_concat(rowhash order by rowhash)) as hash from"
        f"( SELECT substring(MD5(CONCAT_WS(',', {cols})), 1, 5) AS rowhash "
        f"FROM `{table_name}`) as sub;"
    )


def combine_table_hashes(hashes: list[str]) -> str:
    """Multi-table state = sorted per-table hashes joined by '_' (task.py semantics)."""
    return "_".join(sorted(h if h is not None else "" for h in hashes))
