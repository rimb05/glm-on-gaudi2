#!/bin/bash
# Container entry point: apply this project's hooks to the image's vllm_gaudi (idempotent), then run vllm serve.
set -euo pipefail
CODE=${GLM53_CODE:-/workspace/glm53/code}
export PYTHONPATH="$CODE${PYTHONPATH:+:$PYTHONPATH}"
python3 "$CODE/patches/fastlink_hooks.py"
python3 "$CODE/patches/engine_skip_empty.py"
python3 "$CODE/patches/engine_single_batch.py"
python3 "$CODE/patches/engine_wave.py"
python3 "$CODE/patches/dsa_override.py"
# GLM_MTP=n: speculative decoding with the model's own MTP layer, n draft tokens per step
if [ -n "${GLM_MTP:-}" ]; then
  set -- "$@" --speculative-config "{\"method\": \"mtp\", \"num_speculative_tokens\": ${GLM_MTP}}"
fi
# GLM_PARSERS=1: GLM's reasoning and tool-call parsers, and the model card's sampling defaults
if [ "${GLM_PARSERS:-0}" = 1 ]; then
  set -- "$@" --reasoning-parser glm47 --tool-call-parser glm47 --enable-auto-tool-choice \
    --override-generation-config '{"temperature": 1.0, "top_p": 0.95}'
fi
exec vllm serve "$@"
