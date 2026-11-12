#!/bin/bash
# run_scan_bench.sh N [ENV=VAL ...]   run scan_bench.py with N ranks on cards 0..N-1 of this server (cards must be free)
N=$1; shift
. /workspace/glm53/deploy/site.env   # HCCL_IF
DEVS=$(seq -s, 0 $((N-1)))
sudo -n docker rm -f glm53-scan-bench >/dev/null 2>&1
sudo -n docker run --rm --name glm53-scan-bench --runtime=habana --network=host --ipc=host --cap-add=SYS_NICE --ulimit memlock=-1 \
  -e HABANA_VISIBLE_DEVICES=$DEVS -e PT_HPU_LAZY_MODE=1 -e PT_HPU_ENABLE_LAZY_COLLECTIVES=true -e HCCL_SOCKET_IFNAME=$HCCL_IF \
  $(for kv in "$@"; do echo -n "-e $kv "; done) \
  -v /workspace/glm53:/workspace/glm53 --entrypoint bash glm53-serve:dev1 -c "
cd /workspace/glm53/code/tools; export WORLD_SIZE=$N MASTER_ADDR=127.0.0.1 MASTER_PORT=29600
for r in \$(seq 1 $((N-1))); do RANK=\$r python3 scan_bench.py > /tmp/r\$r.log 2>&1 & done
RANK=0 timeout ${TIMEOUT:-900} python3 scan_bench.py 2>&1 | grep -E '^RESULT|^CHECK|^TIME|^world|^done|Error|error|Traceback' ; sleep 1
grep -h -E 'Error|rror|Traceback' /tmp/r*.log 2>/dev/null | sort | uniq -c | head -8" 2>&1 | tail -60
