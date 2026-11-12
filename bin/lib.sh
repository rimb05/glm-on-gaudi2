#!/bin/bash
# Sourced by the bin/ scripts: reads deploy/site.env and defines node_ssh.
#   node_ssh N 'command'     run a command on node N (0 or 1); stdin is passed through
# GLM53_SITE points at another settings file if deploy/site.env is not the one to use.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SITE_ENV="${GLM53_SITE:-$REPO/deploy/site.env}"
[ -f "$SITE_ENV" ] || { echo "missing $SITE_ENV: copy deploy/site.env.example to deploy/site.env and fill it in" >&2; exit 1; }
# shellcheck disable=SC1090
. "$SITE_ENV"
if ! declare -F node_ssh >/dev/null; then
  node_ssh() {
    local n=$1; shift
    local var="NODE${n}_SSH"
    [ -n "${!var:-}" ] || { echo "set $var in $SITE_ENV" >&2; return 1; }
    ssh "${!var}" "$@"
  }
fi
