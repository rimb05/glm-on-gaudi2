#!/bin/bash
# Touch the common request shapes after a (re)start so the first real requests do not wait for graph compiles.
# Run on node 0. Compiled recipes are cached on disk (recipe-cache/), so later restarts warm up much faster.
#   warmup.sh               short prompts, an 8K and a 40K prompt with a follow-up turn (about 1-3 minutes)
#   warmup.sh full [TOKENS] also one long prompt with a follow-up turn: TOKENS, or 97 % of the served window
#                           (5-10 minutes at 700K, 12-17 minutes at 1M)
# Environment: API_KEY if the server was started with one, BASE (default http://127.0.0.1:8000).
set -uo pipefail
BASE=${BASE:-http://127.0.0.1:8000}
cd /workspace/glm53/results
echo "waiting for $BASE/health"
until curl -fsS "$BASE/health" >/dev/null 2>&1; do sleep 5; done
python3 ../serving/chatcheck.py | tail -3
for p in 300 2000 6000; do
  python3 ../serving/evalkit.py speed --prompt-tokens $p --max-tokens 96 --runs 1 --salt 77 | tail -1
done
python3 ../serving/evalkit.py natspeed --prompts 2 --max-tokens 200 --temperature 1.0 --top-p 0.95 | tail -1
for t in 8000 40000; do
  python3 ../serving/longctx.py needle --tokens $t --followup 150 | cut -c1-240
done
if [ "${1:-}" = full ]; then
  WINDOW=$(grep -E '^MAX_MODEL_LEN=' /workspace/glm53/deploy/glm53.env | cut -d= -f2)
  TOKENS=${2:-$(( ${WINDOW:-720000} * 97 / 100 ))}
  python3 ../serving/longctx.py needle --tokens "$TOKENS" --followup 200 | cut -c1-240
fi
echo "warm-up done"
