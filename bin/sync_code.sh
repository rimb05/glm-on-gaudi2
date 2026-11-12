#!/bin/bash
# Push this repository's code and settings to /workspace/glm53/code on both servers (deploy/, including the
# untracked deploy/site.env, also goes to /workspace/glm53/deploy; serving/ to /workspace/glm53/serving).
# Files are replaced, never rewritten in place, so a running script keeps the version it started with.
# Prints a checksum of the synced tree per server: the two must match. Restart the units to use the new code.
set -euo pipefail
. "$(dirname "$0")/lib.sh"
cd "$REPO"
find convert fetch image meta serving patches glm53_hpu tools deploy -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
for n in 0 1; do
  COPYFILE_DISABLE=1 tar -cf - --exclude .DS_Store convert fetch image meta serving patches glm53_hpu tools deploy | node_ssh $n 'mkdir -p /workspace/glm53/code && tar -xf - -C /workspace/glm53/code && cd /workspace/glm53/code && mkdir -p /workspace/glm53/serving /workspace/glm53/deploy /workspace/glm53/meta /workspace/glm53/results /workspace/glm53/logs && cp --remove-destination meta/* /workspace/glm53/meta/ && cp -r --remove-destination serving/*.sh serving/*.py serving/*.txt serving/cookbook_gates /workspace/glm53/serving/ && chmod +x /workspace/glm53/serving/*.sh serving/*.sh fetch/*.sh && cp --remove-destination deploy/* /workspace/glm53/deploy/ && chmod +x /workspace/glm53/deploy/*.sh && find . -type f -not -path "*/__pycache__/*" | sort | xargs md5sum | md5sum | cut -c1-12'
done
