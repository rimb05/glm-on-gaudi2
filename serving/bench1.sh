#!/bin/bash
# Launch one single-server configuration, wait for it, run the speed check, append one line to results/bench.tsv.
#   bench1.sh NAME "ENV1=a ENV2=b" MODEL_DIR [extra vllm args...]
# Leaves the server running (the next bench1.sh call replaces it).
set -uo pipefail
NAME=$1; ENVS=$2; MODEL=$3; shift 3
ROOT=/workspace/glm53
for c in $(sudo -n docker ps -a --format '{{.Names}}' | grep '^glm53-' | grep -v twin-gen); do sudo -n docker rm -f "$c" >/dev/null 2>&1; done
sleep 6
BASE_ENV="VLLM_SKIP_WARMUP=true VLLM_HPU_FSDPA_SOFTMAX_MODE=None VLLM_SCALE_ADJUSTMENT=false VLLM_GRAPH_RESERVED_MEM=0.2 PT_HPU_QKV_SLICE_SEQ_LEN_THLD=128"
# PP=n runs n pipeline stages on this server (PARTITION = layers per stage, e.g. 6,4)
PPARGS=""
if [ "${PP:-1}" != 1 ]; then
  PPARGS="--pipeline-parallel-size $PP --no-async-scheduling"
  BASE_ENV="$BASE_ENV VLLM_PP_LAYER_PARTITION=$PARTITION VLLM_HPU_PP_STATS_EVERY=500 VLLM_HOST_IP=127.0.0.1 GLOO_SOCKET_IFNAME=lo"
fi
t0=$(date +%s)
$ROOT/serving/run1.sh "$NAME" "${DEVICES:-all}" "${IMAGE:-glm53-serve:dev1}" "$BASE_ENV $ENVS" "$MODEL" --served-model-name twin --dtype bfloat16 \
  --max-model-len "${MAXLEN:-131072}" --max-num-seqs 4 --block-size 128 --max-num-batched-tokens 8192 --enable-chunked-prefill \
  --long-prefill-token-threshold 4096 --gpu-memory-utilization "${UTIL:-0.86}" --max-cudagraph-capture-size 2048 \
  --trust-remote-code --tensor-parallel-size "${TP:-8}" --distributed-executor-backend mp --port 8000 $PPARGS "$@" >/dev/null
ok=0
for i in $(seq 1 240); do
  curl -fsS http://127.0.0.1:8000/health >/dev/null 2>&1 && { ok=1; break; }
  sudo -n docker ps --format '{{.Names}}' | grep -q "^glm53-$NAME$" || break
  sleep 5
done
load=$(( $(date +%s) - t0 ))
if [ $ok != 1 ]; then
  echo "$NAME: server did not come up (${load}s)"; sudo -n docker logs "glm53-$NAME" 2>&1 | grep -E "Error|error|Traceback" | grep -v "commit hash" | tail -5 | cut -c1-300
  exit 1
fi
cd $ROOT/results
python3 $ROOT/code/serving/evalkit.py speed --prompt-tokens 2000 --max-tokens 256 --runs 1 --salt 900 >/dev/null 2>&1   # compile
r=$(python3 $ROOT/code/serving/evalkit.py speed --prompt-tokens 2000 --max-tokens "${GEN:-384}" --runs "${RUNS:-8}" 2>&1 | tail -1)
kv=$(sudo -n docker logs "glm53-$NAME" 2>&1 | grep -o "HPU KV cache size: [0-9,]* tokens" | tail -1)
w=$(sudo -n docker logs "glm53-$NAME" 2>&1 | grep -o "Model profiling run took.*device memory ([0-9.]* GiB" | tail -1 | grep -o "[0-9.]* GiB$")
echo "$(date -u +%FT%TZ)	$NAME	load=${load}s	weights/card=$w	$kv	$r	env=[$ENVS]	args=[$*]" | tee -a $ROOT/results/bench.tsv
