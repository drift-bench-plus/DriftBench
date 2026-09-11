#!/bin/bash
# Campaign v2 (ruling 2026-08-11 design):
#   L0 = same agent as B0, user never speaks (mute floor; real API costs)
#   B0 = raw baseline (asking permitted, never encouraged)
#   A0 = free-form asking; A1 = slot-audit asking
#   max_turns 40 (the tool loop + conversation does not fit WebShop's bare 10-16)
#   fork-after-load workers: parent preloads the catalog once, workers share it
set -euo pipefail
: "${ARK_API_KEY:?export ARK_API_KEY first}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$PWD/../../pipeline/src${PYTHONPATH:+:$PYTHONPATH}"
PY="${PY:-python3}"
SAMPLES="$ROOT/artifacts/samples_v2"
OUT="$ROOT/experiments/runs_v2"
MAN="$OUT/manifest.json"
PROCS="${PROCS:-20}"
MAXTURNS="${MAXTURNS:-40}"

# gate chunk first (B0), then everything
echo "== gate chunk: B0, 40 rows"
"$PY" -m intent_graph.cli episode exp run --arm B0 --manifest "$MAN" --samples "$SAMPLES" \
    --out "$OUT" --procs 8 --max-turns "$MAXTURNS" --limit 40
"$PY" "$ROOT/tools/validity_gate.py" --out "$OUT" --min 30 --halt
echo "== gate passed"

for ARM in B0 L0 A0 A1; do
  if [ -f "$OUT/HALT" ]; then echo "HALT present; stopping"; exit 2; fi
  echo "== arm $ARM ($PROCS forked workers, max_turns=$MAXTURNS)"
  "$PY" -m intent_graph.cli episode exp run --arm "$ARM" --manifest "$MAN" \
      --samples "$SAMPLES" --out "$OUT" --procs "$PROCS" --max-turns "$MAXTURNS"
done

"$PY" "$ROOT/tools/validity_gate.py" --out "$OUT" --min 60 || true
"$PY" -m intent_graph.cli episode exp report --arms L0,B0,A0,A1 --out "$OUT" \
    --baseline B0 --report-to "$OUT/report.json"
echo "CAMPAIGN DONE"
