#!/bin/bash
# Two-server pipeline launcher (PP 2 x TP n). Run the SAME command on both servers, NODE 0 first:
#   run2.sh NODE NAME IMAGE IFNAME MASTER_IP "ENV1=a ENV2=b ..." vllm-serve-args...
# NODE 0 runs the API server (127.0.0.1:8000, loopback only) and pipeline stage 0; NODE 1 is headless.
# IFNAME is the interface the stage-to-stage link uses on this server (a Gaudi scale-out port with an IPv4
# address, or an ordinary network port); MASTER_IP is NODE 0's address on that network. HOST_IP (environment)
# names this server's address on IFNAME if it has more than one.
set -euo pipefail
NODE=$1; NAME=$2; IMAGE=$3; IFN=$4; MASTER_IP=$5; EXTRA_ENV=$6; shift 6
ROOT=/workspace/glm53
. "$ROOT/deploy/site.env"   # HCCL_IF
HOST_IP=${HOST_IP:-$(ip -4 -o addr show dev "$IFN" | awk '{print $4}' | cut -d/ -f1 | head -1)}
[ -n "$HOST_IP" ] || { echo "no IPv4 address on $IFN"; exit 1; }
mkdir -p $ROOT/recipe-cache/$NAME-node$NODE $ROOT/logs
ENVS=()
for kv in $EXTRA_ENV; do ENVS+=(-e "$kv"); done
if [ "$NODE" = 0 ]; then TAIL=(--host 127.0.0.1 --port 8000); else TAIL=(--headless); fi
sudo -n docker rm -f glm53-$NAME >/dev/null 2>&1 || true
exec sudo -n docker run -d --name glm53-$NAME --runtime=habana --network=host --ipc=host \
  --cap-add=SYS_NICE --ulimit memlock=-1 --ulimit core=0 \
  -e HABANA_VISIBLE_DEVICES=all -e PT_HPU_LAZY_MODE=1 -e PT_HPU_ENABLE_LAZY_COLLECTIVES=true -e PYTHONUNBUFFERED=1 \
  -e HF_HUB_OFFLINE=1 -e VLLM_NO_USAGE_STATS=1 -e DO_NOT_TRACK=1 \
  -e HCCL_SOCKET_IFNAME="$HCCL_IF" -e GLOO_SOCKET_IFNAME="$IFN" -e VLLM_HOST_IP="$HOST_IP" \
  -e PT_HPU_RECIPE_CACHE_CONFIG=$ROOT/recipe-cache/$NAME-node$NODE,false,16384 \
  -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600 -e VLLM_ENGINE_ITERATION_TIMEOUT_S=3600 \
  "${ENVS[@]}" \
  -v $ROOT:$ROOT \
  --entrypoint $ROOT/code/serving/entry.sh "$IMAGE" "$@" --pipeline-parallel-size "${PP:-2}" --no-async-scheduling \
  --distributed-executor-backend mp --nnodes 2 --node-rank "$NODE" --master-addr "$MASTER_IP" --master-port 29555 \
  "${TAIL[@]}"
