"""The canonical GRIP v2 table for the v9 rational campaign.

The v2 protocol table WITHOUT the role-realism pair (ruling 2026-08-21: "NO NEED for
role-realism parts... But you don't have PostShift"):

    | Success | Earned | Inferred | Aim | Recovery | Patience | Staleness | Reaction |
    | PostShift |

One row per cell (the campaign is single-persona rational). Below the table: each
cell's conditioning n's (undefined is never zero), then paired A3-A2 deltas on the v2
metrics over the clean pairs. Note: under the at-or-below shift physics every episode
carries a moving shift, so PostShift coincides with Success; the column stays because
the coincidence is a property of this regime, not of the metric.
"""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pipeline" / "src"))
from loguru import logger as _l  # noqa: E402
_l.remove()
from intent_graph.runtime import grip  # noqa: E402

_SET = sys.argv[1] if len(sys.argv) > 1 else "rational_v9"
BASE = Path(__file__).parent / "artifacts" / _SET
_ARM_ORDER = ["B0", "A0", "A1", "A2", "A2r", "A3",
              "B1v3", "B2v2", "B3", "B4v2", "B5", "B6v2"]


def _cell_key(name: str):
    arm, _, tail = name.partition("_")
    return (tail, _ARM_ORDER.index(arm) if arm in _ARM_ORDER else 99)


if _SET == "rational_v9":
    CELLS = ["B0_s1", "A0_s1", "A1_s1", "A2_s1", "A3_s1",
             "B0_s2", "A0_s2", "A1_s2", "A2_s2", "A2r_s2", "A3_s2",
             "B0_s3", "A0_s3", "A1_s3", "A2_s3", "A3_s3"]
    # Clean pairs only: s2's original A2 ran single-identity while A3 s2 ran dual.
    PAIRS = [("s1", "A2_s1", "A3_s1"), ("s2", "A2r_s2", "A3_s2"),
             ("s3", "A2_s3", "A3_s3")]
else:
    CELLS = sorted((d.name for d in BASE.iterdir() if d.is_dir()), key=_cell_key)
    PAIRS = [(f"s{n}", f"A2_s{n}", f"A3_s{n}") for n in (1, 2, 3)
             if (BASE / f"A2_s{n}").is_dir() and (BASE / f"A3_s{n}").is_dir()]
UND = "--"


def fmt(v, pct=True):
    """0-1 metrics print x100 with two decimals (ruling 2026-08-21); raw-unit metrics
    (Patience in budget points, Reaction in turns) keep two decimals unscaled."""
    if v in (None, "undefined"):
        return UND
    if isinstance(v, bool):
        v = float(v)
    if isinstance(v, float):
        return f"{v * 100:.2f}" if pct else f"{v:.2f}"
    return str(v)


def load(cell: str):
    d = BASE / cell
    rows, by = [], {}
    if not d.exists():
        return rows, by, None
    for p in sorted(d.glob("*__*.json")):
        t = json.loads(p.read_text())
        if t.get("outcome") in (None, "ERROR"):
            continue
        r = grip.score_v2(t)
        rows.append(r)
        by[t["header"]["sample_id"]] = r
    return rows, by


def paired(vals, pct=True):
    n = len(vals)
    if not n:
        return UND
    m = sum(vals) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in vals) / max(n - 1, 1))
    se = sd / math.sqrt(n)
    s = 100.0 if pct else 1.0
    return f"{m * s:+.2f} (z={m / se if se else 0:+.1f}, n={n})"


def main() -> None:
    cells = {}
    for c in CELLS:
        rows, by = load(c)
        if rows:
            cells[c] = (rows, by, grip.report_v2(rows))

    cols = ["Success", "Earned", "Inferred", "Aim", "Recovery", "Patience",
            "Staleness", "Reaction", "PostShift"]
    print("| Cell | n | " + " | ".join(cols) + " |")
    print("|" + "---|" * (len(cols) + 2))
    for c, (rows, by, rep) in cells.items():
        print(f"| {c} | {rep['n']} | "
              f"{fmt(rep['success'])} | {fmt(rep['earned'])} | {fmt(rep['inferred'])} | "
              f"{fmt(rep['aim'])} | {fmt(rep['recovery'])} | "
              f"{fmt(rep['patience_left'], pct=False)} | "
              f"{fmt(rep['staleness'])} | {fmt(rep['reaction'], pct=False)} | "
              f"{fmt(rep['post_shift_success'])} |")

    print("\nconditioning n's per cell:")
    for c, (rows, by, rep) in cells.items():
        print(f"  {c}: hidden={rep['n_hidden']} asked={rep['n_asked']} "
              f"stale-domain={rep['n_stale_dom']} reaction={rep['n_reaction']} "
              f"shifted={rep['n_shifted']}")

    print("\npaired A3-A2 deltas on the v2 metrics (clean pairs, same samples):")
    for seed, ca2, ca3 in PAIRS:
        if ca2 not in cells or ca3 not in cells:
            print(f"  {seed}: missing cell")
            continue
        b2, b3 = cells[ca2][1], cells[ca3][1]
        common = sorted(set(b2) & set(b3))

        def d(field, indicator=None, sub=common):
            out = []
            for k in sub:
                a, b = b2[k][field], b3[k][field]
                if indicator is not None:
                    a, b = int(a == indicator), int(b == indicator)
                if a in (None, "undefined") or b in (None, "undefined"):
                    continue
                out.append(b - a)
            return out

        hid = [k for k in common if b2[k]["hidden"] and b3[k]["hidden"]]
        print(f"  {seed} ({ca2} vs {ca3}, {len(common)} samples)")
        print(f"    Success  {paired(d('success'))}")
        print(f"    Earned   {paired(d('bucket', grip.EARNED, hid))}")
        print(f"    Recovery {paired(d('recovery'))}")
        print(f"    Aim      {paired(d('aim'))}")
        print(f"    Patience {paired(d('patience_left'), pct=False)}")


if __name__ == "__main__":
    main()
