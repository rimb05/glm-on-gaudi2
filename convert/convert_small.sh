#!/bin/bash
# Convert a small test checkpoint (one worker for all expert layers) and validate it. Runs the tools in the serving
# image on the CPU.   convert_small.sh MODEL_DIR_NAME LAYERS      (under /workspace/glm53/models; output NAME-FP8)
set -euo pipefail
M=models/$1; L=$2
cd /workspace/glm53
rm -rf "$M-FP8"
sudo -n docker run --rm --entrypoint bash -v /workspace/glm53:/workspace/glm53 -w /workspace/glm53 glm53-serve:dev1 -c "
set -e
C=code/convert
python3 \$C/bf16_to_fp8.py --src $M --dst $M-FP8 --layers $L --threads 16
python3 \$C/bf16_to_fp8.py --src $M --dst $M-FP8 --rest
python3 \$C/bf16_to_fp8.py --src $M --dst $M-FP8 --finalize
python3 \$C/validate_fp8.py --src $M --dst $M-FP8 --layers $L --threads 16
python3 \$C/validate_fp8.py --src $M --dst $M-FP8 --rest
python3 \$C/validate_fp8.py --src $M --dst $M-FP8 --index
chown -R 1000:1000 $M-FP8
" 2>&1 | grep -v "^Calling" | grep -E "finalized|PASS|FAIL|Error|error" 
( cd "$M-FP8" && sha256sum *.safetensors | sha256sum | cut -c1-16 )
