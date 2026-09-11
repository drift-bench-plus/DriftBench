#!/bin/bash
# Airline pre-flight (register-mandated): all 11 arms x 8 eps + clean control.
# R3 physics (coin 0.6), policy-ON (stamped), rational.
cd "$(dirname "$0")"
PY=${PY:-python3}
mkdir -p logs artifacts
: > logs/air_smoke_progress.log
$PY episodes_run_airline.py \
  --arms B0,A0,A1,B1v3,B2v2,B3,B4v2,B5,B6v2 --persona rational --limit 8 \
  --shift --shift-fire-prob 0.6 --workers 32 --max-rps 7 --seed 1 \
  --cache-tag air_smoke --out artifacts/air_smoke > logs/air_smoke_a.log 2>&1
echo "SMOKE_PLAIN_DONE" >> logs/air_smoke_progress.log
$PY episodes_run_airline.py \
  --arms A2,A3 --persona rational --limit 8 --show-patience \
  --shift --shift-fire-prob 0.6 --workers 32 --max-rps 7 --seed 1 \
  --cache-tag air_smoke --out artifacts/air_smoke > logs/air_smoke_b.log 2>&1
echo "SMOKE_A2A3_DONE" >> logs/air_smoke_progress.log
$PY episodes_run_airline.py \
  --arms A0 --persona rational --limit 40 --clean \
  --shift --shift-fire-prob 0.6 --workers 32 --max-rps 7 --seed 1 \
  --cache-tag air_clean --out artifacts/air_clean > logs/air_clean.log 2>&1
echo "SMOKE_ALL_DONE $(date +%s)" >> logs/air_smoke_progress.log
