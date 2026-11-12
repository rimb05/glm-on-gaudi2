#!/bin/bash
# Launcher for one node of the two-server GLM-5.3 deployment. Runs the container in the foreground, so a systemd
# unit (deploy/glm53-node0.service on node 0, glm53-node1.service on node 1) can supervise it.
#   serve_node.sh 0     on node 0: pipeline stage 0 + the API server
#   serve_node.sh 1     on node 1: pipeline stage 1 (headless); waits until node 0 listens on the rendezvous port
# Settings: /workspace/glm53/deploy/glm53.env (the active profile) and /workspace/glm53/deploy/site.env (this
# installation; see deploy/site.env.example).
set -euo pipefail
NODE=${1:?node rank 0 or 1}
ROOT=/workspace/glm53
set -a; . "$ROOT/deploy/glm53.env"; set +a
[ -f "$ROOT/deploy/site.env" ] || { echo "missing $ROOT/deploy/site.env (see deploy/site.env.example)"; exit 1; }
. "$ROOT/deploy/site.env"
: "${FABRIC_IF:?set in deploy/site.env}" "${HCCL_IF:?set in deploy/site.env}"
: "${NODE0_ADDR:?set in deploy/site.env}" "${NODE1_ADDR:?set in deploy/site.env}"

# this node's and the other node's address on the fabric interface
MASTER_ADDR=$NODE0_ADDR
if [ "$NODE" = 0 ]; then HOST_IP=$NODE0_ADDR; PEER=$NODE1_ADDR; else HOST_IP=$NODE1_ADDR; PEER=$NODE0_ADDR; fi
ADDRS=$(ip -4 -o addr show dev "$FABRIC_IF" | awk '{print $4}' | cut -d/ -f1)
case $'\n'"$ADDRS"$'\n' in
  *$'\n'"$HOST_IP"$'\n'*) ;;
  *) echo "$HOST_IP is not an address of $FABRIC_IF on this server (is this node $NODE?)"; exit 1 ;;
esac
case "$API_HOST" in
  127.*|10.*|192.168.*|172.1[6-9].*|172.2[0-9].*|172.3[01].*) ;;
  *) echo "API_HOST=$API_HOST is not a private address: refusing to start"; exit 1 ;;
esac
mkdir -p "$ROOT/recipe-cache/$NAME-node$NODE" "$ROOT/logs"

if [ "$NODE" = 0 ]; then
  TAIL=(--host "$API_HOST" --port "$API_PORT")
  [ -z "$API_KEY" ] || TAIL+=(--api-key "$API_KEY")
else
  TAIL=(--headless)
  echo "waiting for node 0 at $MASTER_ADDR:$MASTER_PORT"
  until timeout 2 bash -c "exec 3<>/dev/tcp/$MASTER_ADDR/$MASTER_PORT" 2>/dev/null; do sleep 2; done
fi

sudo -n docker rm -f "$NAME" >/dev/null 2>&1 || true

# Pair watchdog. The two containers are one engine: if the other server's processes disappear (its unit was
# restarted, or it crashed), this side would keep running and answer nothing. Once connections to the peer's fabric
# address have been seen, their absence stops this container; systemd restarts the unit and the pair forms again.
(
  seen=0
  while sleep 10; do
    if ! sudo -n docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
      [ "$seen" = 1 ] && exit 0
      continue
    fi
    if [ "$(ss -Htn state established | grep -c "$PEER")" -gt 0 ]; then
      seen=1
    elif [ "$seen" = 1 ]; then
      echo "pair watchdog: no connection to $PEER any more, stopping $NAME"
      sudo -n docker stop -t 10 "$NAME" >/dev/null 2>&1
      exit 0
    fi
  done
) &
# page cache left by an earlier load or by other jobs competes with the 773 GB checkpoint read
sync; echo 1 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null || true

exec sudo -n docker run --rm --name "$NAME" --runtime=habana --network=host --ipc=host \
  --cap-add=SYS_NICE --ulimit memlock=-1 --ulimit core=0 \
  -e HABANA_VISIBLE_DEVICES=all -e PT_HPU_LAZY_MODE=1 -e PT_HPU_ENABLE_LAZY_COLLECTIVES=true -e PYTHONUNBUFFERED=1 \
  -e HF_HUB_OFFLINE=1 -e VLLM_NO_USAGE_STATS=1 -e DO_NOT_TRACK=1 \
  -e HCCL_SOCKET_IFNAME="$HCCL_IF" -e GLOO_SOCKET_IFNAME="$FABRIC_IF" -e VLLM_HOST_IP="$HOST_IP" \
  -e PT_HPU_RECIPE_CACHE_CONFIG="$ROOT/recipe-cache/$NAME-node$NODE,false,16384" \
  -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600 -e VLLM_ENGINE_ITERATION_TIMEOUT_S=3600 \
  -e VLLM_SKIP_WARMUP=true -e VLLM_HPU_FSDPA_SOFTMAX_MODE=None -e VLLM_SCALE_ADJUSTMENT=false \
  -e VLLM_GRAPH_RESERVED_MEM=0.2 -e PT_HPU_QKV_SLICE_SEQ_LEN_THLD=128 -e VLLM_PP_LAYER_PARTITION="$PARTITION" \
  -e VLLM_BUCKETING_FROM_FILE="$BUCKETS" \
  -e GLM_ENGINE_SKIP_EMPTY=1 -e GLM_ENGINE_PREFILL_WAVE=1 -e GLM_PP_FASTLINK=1 \
  -e GLM_MTP="$MTP" -e GLM_DRAFT_CHAIN=1 -e GLM_FAST_META=1 -e GLM_ROUTER_FAST=1 -e GLM_EARLY_DRAFT=1 \
  -e GLM_SPEC_DEVICE_REJECT=1 \
  -e GLM_DSA_SPARSE=1 -e GLM_DSA_V2=1 -e GLM_DSA_MIN_KEYS_DECODE="${SPARSE_DECODE_FROM:-12288}" -e GLM_PARSERS=1 \
  -e GLM_KV_FP8="${KV_CODEC:-0}" \
  -v "$ROOT:$ROOT" \
  --entrypoint "$ROOT/code/serving/entry.sh" "$IMAGE" "$MODEL" \
  --served-model-name "$SERVED_NAME" --dtype bfloat16 --max-model-len "$MAX_MODEL_LEN" --max-num-seqs "$MAX_SEQS" \
  --block-size 128 --max-num-batched-tokens 8192 --enable-chunked-prefill --long-prefill-token-threshold 4096 \
  --gpu-memory-utilization 0.86 --num-gpu-blocks-override "$KV_BLOCKS" --max-cudagraph-capture-size 2048 \
  --trust-remote-code --tensor-parallel-size "$TP" --enable-expert-parallel --enable-ep-weight-filter \
  --pipeline-parallel-size 2 --no-async-scheduling --distributed-executor-backend mp --nnodes 2 \
  --node-rank "$NODE" --master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT" "${TAIL[@]}"
