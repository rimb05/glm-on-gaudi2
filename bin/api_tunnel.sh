#!/bin/bash
# Forward the GLM-5.3 API, which listens on node 0's loopback, to this machine:
#   bin/api_tunnel.sh [local_port]      default 8000; then: curl http://127.0.0.1:8000/v1/models
# Stays in the foreground; Ctrl-C closes the forward. Uses NODE0_SSH from deploy/site.env.
set -euo pipefail
. "$(dirname "$0")/lib.sh"
PORT=${1:-8000}
exec ssh -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=4 \
  -N -L "127.0.0.1:$PORT:127.0.0.1:8000" "${NODE0_SSH:?set NODE0_SSH in deploy/site.env}"
