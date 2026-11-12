#!/bin/bash
# Validation of the 1M profile on the running deployment (run on node 0). Results in results/gates-1m/.
# It loads the endpoint for about an hour: check vllm:num_requests_running in /metrics first.
cd /workspace/glm53/results && mkdir -p gates-1m
OUT=gates-1m
G="python3 ../serving/cookbook_gates/glm5_gates.py"
( while true; do echo "$(date -u +%T) $(hl-smi --query-aip=memory.used --format=csv,noheader | head -1)"; sleep 5; done ) > $OUT/mem0.log 2>&1 &
MEM=$!
{
  /workspace/glm53/deploy/warmup.sh
  python3 ../serving/crosscheck.py
  python3 ../serving/crosscheck.py --prompt-tokens 1500 --essay-words 1600
  for g in "tools" "mtp" "logprobs 20" "gsm8k" "needles 16000,64000,120000" "speed" "stress"; do echo "== $g"; $G $g 2>&1 | tail -8; done
  echo GATES_DONE
} > $OUT/gates.log 2>&1
python3 ../serving/longctx.py needle --tokens 700000 --followup 300 --batch-id p700 --out $OUT/long.jsonl > $OUT/run700k.log 2>&1
python3 ../serving/longctx.py needle --tokens 700000 --followup 400 --batch-id p700 --out $OUT/long.jsonl >> $OUT/run700k.log 2>&1
python3 ../serving/longctx.py needle --tokens 1036000 --followup 300 --batch-id m1 --out $OUT/long.jsonl > $OUT/run1m.log 2>&1
python3 ../serving/longctx.py needle --tokens 1036000 --followup 400 --batch-id m1 --out $OUT/long.jsonl >> $OUT/run1m.log 2>&1
python3 ../serving/longctx.py needle --tokens 1036000 --followup 400 --batch-id m1 --temperature 1.0 --top-p 0.95 --out $OUT/long.jsonl >> $OUT/run1m.log 2>&1
kill $MEM
echo ALL_DONE >> $OUT/gates.log
