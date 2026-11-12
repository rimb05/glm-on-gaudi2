#!/bin/bash
# run_chain.sh N [ENV=VAL ...]   run hccl_chain.py with N ranks on cards 0..N-1 of this server
N=$1; shift
. /workspace/glm53/deploy/site.env   # HCCL_IF
DEVS=$(seq -s, 0 $((N-1)))
sudo -n docker rm -f glm53-hccl-chain >/dev/null 2>&1
sudo -n docker run --rm --name glm53-hccl-chain --runtime=habana --network=host --ipc=host --cap-add=SYS_NICE --ulimit memlock=-1 \
  -e HABANA_VISIBLE_DEVICES=$DEVS -e PT_HPU_LAZY_MODE=1 -e HCCL_SOCKET_IFNAME=$HCCL_IF $(for kv in "$@"; do echo -n "-e $kv "; done) \
  -v /workspace/glm53:/workspace/glm53 --entrypoint bash glm53-serve:dev1 -c "
cd /workspace/glm53/code/tools; export WORLD_SIZE=$N MASTER_ADDR=127.0.0.1 MASTER_PORT=29600
for r in \$(seq 0 $((N-2))); do RANK=\$r python3 hccl_chain.py > /tmp/r\$r.log 2>&1 & done
RANK=$((N-1)) timeout 240 python3 hccl_chain.py 2>&1 | grep -E '^RESULT|^rank|Error|error' ; sleep 1
grep -h -E '^rank|Error|rror' /tmp/r*.log | sort | head -12" 2>&1 | tail -14
