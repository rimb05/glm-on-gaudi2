#!/bin/bash
# Convert the full BF16 checkpoint to FP8 with parallel workers, then validate. Runs on the CPU in one container.
#   convert_full.sh SRC_DIR DST_DIR [WORKERS=12] [THREADS=8] [VALIDATE_STRIDE=8]
# Steps: expert layers 3..78 split over WORKERS ranges + one --rest worker, --finalize, then validation of every
# VALIDATE_STRIDE-th expert of every layer, the non-expert tensors and the index. Exit 0 only if all of it passed.
# A full validation (stride 1) can be run later with validate_fp8.py.
set -uo pipefail
SRC=$1; DST=$2; W=${3:-12}; TH=${4:-8}; STRIDE=${5:-8}
C=/workspace/glm53/code/convert
LOG=/workspace/glm53/logs/convert
mkdir -p "$DST" "$LOG"
[ -f "$SRC/.VERIFIED" ] || { echo "source is not verified (no $SRC/.VERIFIED)"; exit 3; }
L=$(python3 $C/bf16_to_fp8.py --src "$SRC" --list 2>/dev/null | tail -1)
echo "$(date -u +%FT%TZ) source: $L"
LAYERS=($(python3 -c "import json,sys; print(' '.join(map(str, json.loads(sys.argv[1])['expert_layers'])))" "$L"))
N=${#LAYERS[@]}
echo "$(date -u +%FT%TZ) converting $N expert layers with $W workers x $TH threads"
pids=()
for w in $(seq 0 $((W-1))); do
  lo=$(( w * N / W )); hi=$(( (w + 1) * N / W - 1 ))
  [ $hi -ge $lo ] || continue
  r="${LAYERS[$lo]}-${LAYERS[$hi]}"
  python3 $C/bf16_to_fp8.py --src "$SRC" --dst "$DST" --layers "$r" --threads "$TH" --shard-gb 2.5 > "$LOG/L$r.log" 2>&1 &
  pids+=($!)
done
python3 $C/bf16_to_fp8.py --src "$SRC" --dst "$DST" --rest --shard-gb 2.5 > "$LOG/rest.log" 2>&1 &
pids+=($!)
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
[ $fail = 0 ] || { echo "$(date -u +%FT%TZ) a conversion worker failed"; grep -l -E "Error|Traceback" "$LOG"/*.log; exit 1; }
python3 $C/bf16_to_fp8.py --src "$SRC" --dst "$DST" --finalize 2>&1 | grep -v "^Calling" || exit 1
echo "$(date -u +%FT%TZ) validating (every ${STRIDE}th expert, all other tensors, index)"
pids=()
for w in $(seq 0 $((W-1))); do
  lo=$(( w * N / W )); hi=$(( (w + 1) * N / W - 1 ))
  [ $hi -ge $lo ] || continue
  r="${LAYERS[$lo]}-${LAYERS[$hi]}"
  python3 $C/validate_fp8.py --src "$SRC" --dst "$DST" --layers "$r" --threads "$TH" --stride "$STRIDE" > "$LOG/V$r.log" 2>&1 &
  pids+=($!)
done
python3 $C/validate_fp8.py --src "$SRC" --dst "$DST" --rest > "$LOG/Vrest.log" 2>&1 &
pids+=($!)
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
python3 $C/validate_fp8.py --src "$SRC" --dst "$DST" --index 2>&1 | grep -v "^Calling" || fail=1
grep -h -E "^PASS|^FAIL" "$LOG"/V*.log | sort | uniq -c
[ $fail = 0 ] && { echo "$(date -u +%FT%TZ) CONVERTED_AND_VALIDATED"; touch "$DST/.VALIDATED-stride$STRIDE"; exit 0; }
echo "$(date -u +%FT%TZ) VALIDATION_FAILED"; grep -h "^FAIL" "$LOG"/V*.log | head -20; exit 2
