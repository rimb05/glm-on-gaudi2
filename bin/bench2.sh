#!/bin/bash
# Two-server benchmark, run from the workstation: start one pipeline configuration on node 0 (API) and node 1,
# wait for it, run the speed check on node 0, append one summary line to results/bench.tsv there. Leaves the
# servers running. It removes every other glm53-* container first: do not use it while the deployment serves.
#   bench2.sh NAME "ENV1=a ENV2=b" MODEL_DIR PP TP PARTITION [extra vllm args...]
# Env: IFN (interface for the stage link and rendezvous; default FABRIC_IF of deploy/site.env), RUNS, GEN, MAXLEN, UTIL
set -uo pipefail
NAME=$1; ENVS=$2; MODEL=$3; PP=$4; TP=$5; PART=$6; shift 6
. "$(dirname "$0")/lib.sh"
IFN=${IFN:-$FABRIC_IF}
MASTER=$(node_ssh 0 "ip -4 -o addr show dev $IFN | awk '{print \$4}' | cut -d/ -f1 | head -1")
BASE_ENV="VLLM_SKIP_WARMUP=true VLLM_HPU_FSDPA_SOFTMAX_MODE=None VLLM_SCALE_ADJUSTMENT=false VLLM_GRAPH_RESERVED_MEM=${GRAPH_MEM:-0.2} PT_HPU_QKV_SLICE_SEQ_LEN_THLD=128 VLLM_PP_LAYER_PARTITION=$PART VLLM_HPU_PP_STATS_EVERY=${STATS:-0}"
ARGS="$MODEL --served-model-name ${SERVED:-twin} --dtype bfloat16 --max-model-len ${MAXLEN:-131072} --max-num-seqs ${SEQS:-4} --block-size 128 --max-num-batched-tokens 8192 --enable-chunked-prefill --long-prefill-token-threshold 4096 --gpu-memory-utilization ${UTIL:-0.86} --max-cudagraph-capture-size 2048 --trust-remote-code --tensor-parallel-size $TP $*"
CLEAN='for c in $(sudo -n docker ps -a --format "{{.Names}}" | grep "^glm53-" | grep -v twin-gen); do sudo -n docker rm -f $c >/dev/null 2>&1; done'
node_ssh 0 "$CLEAN" & node_ssh 1 "$CLEAN" & wait
sleep 6
t0=$(date +%s)
node_ssh 0 "PP=$PP /workspace/glm53/serving/run2.sh 0 $NAME ${IMAGE:-glm53-serve:dev1} $IFN $MASTER '$BASE_ENV $ENVS' $ARGS" >/dev/null || { echo "node 0 failed to start"; exit 1; }
sleep 3
node_ssh 1 "PP=$PP /workspace/glm53/serving/run2.sh 1 $NAME ${IMAGE:-glm53-serve:dev1} $IFN $MASTER '$BASE_ENV $ENVS' $ARGS" >/dev/null || { echo "node 1 failed to start"; exit 1; }
node_ssh 0 "ok=0; for i in \$(seq 1 ${WAIT:-150}); do curl -fsS http://127.0.0.1:8000/health >/dev/null 2>&1 && { ok=1; break; }; sudo -n docker ps --format '{{.Names}}' | grep -q '^glm53-$NAME\$' || break; sleep 4; done; [ \$ok = 1 ]" || {
  echo "$NAME: server did not come up after $(( $(date +%s) - t0 ))s"
  for h in 0 1; do echo "--- node $h ---"; node_ssh $h "sudo -n docker logs glm53-$NAME 2>&1 | grep -E 'Error|error|Traceback|rror:' | grep -v 'commit hash' | tail -6 | cut -c1-300"; done
  exit 1; }
load=$(( $(date +%s) - t0 ))
node_ssh 0 "cd /workspace/glm53/results && python3 ../serving/evalkit.py speed --prompt-tokens 2000 --max-tokens 256 --runs 1 --salt 900 >/dev/null 2>&1; r=\$(python3 ../serving/evalkit.py speed --prompt-tokens ${PROMPT:-2000} --max-tokens ${GEN:-384} --runs ${RUNS:-8} 2>&1 | tail -1); kv=\$(sudo -n docker logs glm53-$NAME 2>&1 | grep -o 'HPU KV cache size: [0-9,]* tokens' | tail -1); w=\$(sudo -n docker logs glm53-$NAME 2>&1 | grep -o 'Model profiling run took.*device memory ([0-9.]* GiB' | tail -1 | grep -o '[0-9.]* GiB\$'); echo \"\$(date -u +%FT%TZ)	$NAME	PP$PP x TP$TP [$PART]	load=${load}s	weights/card(node0)=\$w	\$kv	\$r	env=[$ENVS]	args=[$*]\" | tee -a /workspace/glm53/results/bench.tsv"
