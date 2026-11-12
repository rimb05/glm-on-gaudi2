#!/bin/bash
# Shorter check of the 720K profile on the running deployment (run on node 0). Results in results/gates-720k/.
# It loads the endpoint: check vllm:num_requests_running in /metrics first.
cd /workspace/glm53/results && mkdir -p gates-720k
OUT=gates-720k
G="python3 ../serving/cookbook_gates/glm5_gates.py"
( while true; do echo "$(date -u +%T) $(hl-smi --query-aip=memory.used --format=csv,noheader | head -1)"; sleep 5; done ) > $OUT/mem0.log 2>&1 &
MEM=$!
{
  for g in "tools" "gsm8k" "needles 16000,64000,120000"; do echo "== $g"; $G $g 2>&1 | tail -8; done
  echo GATES_DONE
} > $OUT/gates.log 2>&1
python3 ../serving/longctx.py needle --tokens 700000 --followup 300 --batch-id p700 --out $OUT/long.jsonl > $OUT/run700k.log 2>&1
python3 ../serving/longctx.py needle --tokens 700000 --followup 400 --batch-id p700 --out $OUT/long.jsonl >> $OUT/run700k.log 2>&1
python3 ../serving/longctx.py needle --tokens 700000 --followup 400 --batch-id p700 --temperature 1.0 --top-p 0.95 --out $OUT/long.jsonl >> $OUT/run700k.log 2>&1
kill $MEM
echo ALL_DONE >> $OUT/gates.log
