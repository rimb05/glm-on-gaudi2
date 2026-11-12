#!/bin/bash
# Single-server test launcher. Runs `vllm serve` in a detached container named glm53-NAME on this server.
#   run1.sh NAME DEVICES IMAGE "ENV1=a ENV2=b ..." vllm-serve-args...
# DEVICES is the HABANA_VISIBLE_DEVICES list (hl-smi indices, or "all"). The API binds to 127.0.0.1 only.
set -euo pipefail
NAME=$1; DEVICES=$2; IMAGE=$3; EXTRA_ENV=$4; shift 4
ROOT=/workspace/glm53
. "$ROOT/deploy/site.env"   # HCCL_IF
mkdir -p $ROOT/recipe-cache/$NAME $ROOT/logs
ENVS=()
for kv in $EXTRA_ENV; do ENVS+=(-e "$kv"); done
sudo -n docker rm -f glm53-$NAME >/dev/null 2>&1 || true
exec sudo -n docker run -d --name glm53-$NAME --runtime=habana --network=host --ipc=host \
  --cap-add=SYS_NICE --ulimit memlock=-1 --ulimit core=0 \
  -e HABANA_VISIBLE_DEVICES="$DEVICES" -e PT_HPU_LAZY_MODE=1 -e PYTHONUNBUFFERED=1 \
  -e HF_HUB_OFFLINE=1 -e VLLM_NO_USAGE_STATS=1 -e DO_NOT_TRACK=1 \
  -e HCCL_SOCKET_IFNAME="$HCCL_IF" -e GLOO_SOCKET_IFNAME="$HCCL_IF" \
  -e PT_HPU_RECIPE_CACHE_CONFIG=$ROOT/recipe-cache/$NAME,false,16384 \
  "${ENVS[@]}" \
  -v $ROOT:$ROOT \
  --entrypoint $ROOT/code/serving/entry.sh "$IMAGE" "$@" --host 127.0.0.1
