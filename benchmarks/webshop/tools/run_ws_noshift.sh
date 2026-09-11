#!/bin/bash
# WEBSHOP BACKBONE EXPERIMENT (ruling 2026-08-26, D27 on the third benchmark).
#
# Scope: 4 backbones x {A0, A3} x 3 seeds x the k=1 p500 rational subset = 24 cells,
#        12,000 episodes.
#
# REGIME -- matched to the night3 coin campaign and the persona sweep, because those are
# the numbers this must reproduce for deepseek (validated: control A0 scored 51.6 at n=80
# against the 54.11+-1.00 p500 anchor, Aim 35.93 vs 36.11):
#   shift_fire_prob 0.6   <- DEFAULTS TO 1.0 IN CODE. Unset, every crossing fires and the
#                            regime is strictly harder. This is the setting that made the
#                            first smoke look broken.
#   thinking OFF          <- runtime.yaml `disable_thinking: true`, and the reported runs
#                            are the "w48/nothink" ones. NOT tau2's thinking-ON regime;
#                            thinking-on/off never share a table.
#   max_turns 40, patience 10 x persona mult, user_v2, shift_scheduled -- all from
#   config/runtime.yaml, which now declares what the campaigns actually ran.
#   retry 40/45 (up from the shipped 8/60): overnight runs must ride out a rate-limit
#   storm rather than leave husks.
#
# API ALLOCATION -- two independent pools, run concurrently:
#   gateway (one shared GPT_GATEWAY_KEY): gpt55, qwen, glm agents  -> GWPER cells each
#   ark: doubao's AGENT on ep-YOUR-ENDPOINT-A, AND every cell's user simulator
#        (doubao-seed-2-1-pro-260628). doubao is therefore self-play; pointing its
#        agent_model at the ep- endpoint splits the two roles across the two ark
#        identities, which is what tau2's model_pool does with one identity per role.
# ENV SERVER: 3 forks on 3020-3022, COW-shared catalog. One server saturates at ~24
#   workers (measured 28 eps/min at 24, 7 eps/min at 48), so widths are sized to that.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/../../pipeline/src${PYTHONPATH:+:$PYTHONPATH}"
PY=${PY:-python3}
export ARK_API_KEY=$(grep '^ARK_API_KEY=' ../.env | cut -d= -f2-)
export GPT_GATEWAY_KEY=$(grep '^GPT_GATEWAY_KEY=' ../.env | cut -d= -f2-)
export WEBSHOP_SERVER_URLS=http://127.0.0.1:3020
MAN=experiments/persona2/views/p500_rational.json
SAMPLES=artifacts/samples_v2
OUT=experiments/ws_noshift
GW=${LLM_GATEWAY_URL:?export LLM_GATEWAY_URL=<your cross-provider gateway>/v1}
GWPER=${GWPER:-1}
DBPER=${DBPER:-1}
W=${W:-6}
# GLOBAL cell cap. Throughput tracks TOTAL worker processes, and this machine kills the
# env server somewhere between 36 and 48 (measured: 24w=13 eps/min, 36w=~17, 48w=19 but
# the server stopped answering and had to be load-shed). Cap the whole campaign, not just
# per-backbone, so the footprint is a decision rather than an accident.
MAXCELLS=${MAXCELLS:-8}   # see RESERVE SLOTS PER BACKBONE above
mkdir -p $OUT logs

backbone_flags () {
  case "$1" in
    gpt55)  echo "--set llm.agent_model=gateway/gpt-5.5-2026-04-24 --set llm.agent_base_url=$GW --set llm.agent_protocol=chat --set llm.agent_api_key_env=GPT_GATEWAY_KEY" ;;
    qwen)   echo "--set llm.agent_model=gateway/openai_qwen3.7-max --set llm.agent_base_url=$GW --set llm.agent_protocol=chat --set llm.agent_api_key_env=GPT_GATEWAY_KEY" ;;
    glm)    echo "--set llm.agent_model=gateway/glm-4.7 --set llm.agent_base_url=$GW --set llm.agent_protocol=chat --set llm.agent_api_key_env=GPT_GATEWAY_KEY" ;;
    # self-play: agent on the SECOND ark identity so it does not contend with the
    # doubao user simulator on the canonical name
    doubao) echo "--set llm.agent_model=ep-YOUR-ENDPOINT-A" ;;
  esac
}

# COUNT CELLS, NOT PROCESSES. --procs forks W workers per cell, so a plain grep -c
# returns ~13x the cell count and the per-backbone cap trips on the first launch.
# Counting DISTINCT --cache-tag values is the cell count.
running_for () {
  ps aux | grep "[e]pisode exp run" | grep -o -- "$1[A-Za-z0-9_]*" \
    | sort -u | wc -l | tr -d " "
}
cell_running () { ps aux | grep "[e]pisode exp run" | grep -q -- "--cache-tag wsns_$1"; }
complete () { [ "$(ls $1/*/*.json 2>/dev/null | wc -l | tr -d ' ')" -ge 495 ]; }

launch () {
  local BB=$1 ARM=$2 SEED=$3
  local TAG=wsns_${BB}_${ARM}_s${SEED}
  local DIR=$OUT/${BB}_${ARM}_s${SEED}
  local OFFSET=$(( (SEED - 1) * 1000 ))
  ( $PY -m intent_graph.cli episode exp run --arm $ARM \
      --manifest $MAN --samples $SAMPLES --out $DIR --threads $W \
      --seed-offset $OFFSET --cache-tag $TAG \
      --set shift_scheduled=false --set p_shift=0 \
      --set llm.retry_attempts=40 --set llm.retry_backoff_cap_s=45 \
      $(backbone_flags $BB) > logs/${TAG}.log 2>&1
    N=$(ls $DIR/*/*.json 2>/dev/null | wc -l | tr -d ' ')
    if grep -qiE "insufficient balance|no available capacity|arrears|QuotaExhausted" logs/${TAG}.log; then
      echo "QUOTA_HALT $DIR $(date +%s)" >> logs/wsns_progress.log; touch logs/HALT_WSNS
    fi
    if [ "$N" -ge 495 ]; then echo "CELL_DONE $DIR ($N)" >> logs/wsns_progress.log
    else echo "CELL_PARTIAL $DIR ($N/500)" >> logs/wsns_progress.log; fi ) &
}

: > logs/wsns_progress.log
echo "WSB_START $(date +%s)" >> logs/wsns_progress.log
while true; do
  [ -f logs/HALT_WSNS ] && { echo "HALTED $(date +%s)" >> logs/wsns_progress.log; break; }
  left=0
  for BB in deepseek; do
    LIM=$GWPER; [ "$BB" = "doubao" ] && LIM=$DBPER
    for SEED in 1 2 3; do
      for ARM in B0 A0 A3; do
        DIR=$OUT/${BB}_${ARM}_s${SEED}
        complete $DIR && continue
        left=$((left+1))
        cell_running ${BB}_${ARM}_s${SEED} && continue
        [ "$(running_for "--cache-tag wsns_${BB}_")" -ge $LIM ] && continue
        [ "$(running_for "--cache-tag wsns_")" -ge $MAXCELLS ] && continue
        launch $BB $ARM $SEED
        sleep 4
      done
    done
  done
  [ "$left" -eq 0 ] && break
  sleep 60
done
echo "WSB_ALL_DONE $(date +%s)" >> logs/wsns_progress.log
