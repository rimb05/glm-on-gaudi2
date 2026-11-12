#!/bin/bash
# Unattended chain for the server that holds the BF16 download, so that nothing depends on a workstation staying
# connected. Run it as root, for example:
#   sudo systemd-run --unit glm53-convert-chain --setenv=REV=<commit> \
#        /bin/bash -c '/workspace/glm53/code/fetch/convert_after_download.sh >> /workspace/glm53/logs/convert-chain.log 2>&1'
# Steps: wait for the download unit -> require the checksum-verified marker (re-run the resumable downloader up to
# 3 times if it is missing) -> convert BF16 to FP8 with 12 workers -> validate EVERY expert tensor against the
# independent reference, every other tensor byte for byte, and the index.
# Markers: logs/CONVERSION_OK or logs/CONVERSION_FAILED. Worker logs: logs/convert/*.log.
# Required: REV (the commit fetch/make_manifest.py printed). Optional: REPO SRC DST MAN WAIT_UNIT WORKERS STRIDE
# OKMARK, RUN_AS (account that owns the download), OWNER (uid:gid for the converted files), MEM, CPUS, IMAGE.
set -uo pipefail
ROOT=/workspace/glm53
REPO=${REPO:-zai-org/GLM-5.3-BF16}
REV=${REV:?the commit hash of the revision that was downloaded}
SRC=${SRC:-$ROOT/src/GLM-5.3-BF16}
DST=${DST:-$ROOT/models/GLM-5.3-FP8}
MAN=${MAN:-$ROOT/meta/glm-5.3-bf16.manifest.tsv}
WAIT_UNIT=${WAIT_UNIT:-glm53-download}
WORKERS=${WORKERS:-12}
STRIDE=${STRIDE:-1}
OKMARK=${OKMARK:-$ROOT/logs/CONVERSION_OK}
OWNER=${OWNER:-1000:1000}
IMAGE=${IMAGE:-glm53-serve:dev1}
log() { echo "$(date -u +%FT%TZ) $*"; }
mkdir -p "$ROOT/logs"
log "chain started; waiting for unit $WAIT_UNIT"
while systemctl is-active --quiet "$WAIT_UNIT"; do sleep 30; done
tries=0
while [ ! -f "$SRC/.VERIFIED" ] && [ $tries -lt 3 ]; do
  tries=$((tries+1))
  log "no $SRC/.VERIFIED after the download unit ended; re-running the downloader, attempt $tries (it resumes and re-verifies)"
  ${RUN_AS:+sudo -u "$RUN_AS"} "$ROOT/code/fetch/download.sh" "$MAN" "$SRC" "$REPO" "$REV" 8 >> "$ROOT/logs/download.log" 2>&1
done
[ -f "$SRC/.VERIFIED" ] || { log "GIVING UP: the download did not verify"; touch "$ROOT/logs/CONVERSION_FAILED"; exit 1; }
avail=$(df --output=avail -B1 "$ROOT" | tail -1)
log "download verified; free space under $ROOT: $((avail / 1024 / 1024 / 1024)) GiB"
[ "$avail" -gt $((900 * 1024 * 1024 * 1024)) ] || [ -n "${SKIP_SPACE_CHECK:-}" ] || {
  log "STOP: less than 900 GiB free for the 773 GB FP8 checkpoint; nothing was deleted"; touch "$ROOT/logs/CONVERSION_FAILED"; exit 1; }
sync; echo 3 > /proc/sys/vm/drop_caches
rm -f "$ROOT/logs/CONVERSION_FAILED" "$OKMARK"
log "starting conversion: $SRC -> $DST ($WORKERS workers, validation stride $STRIDE)"
docker rm -f glm53-convert >/dev/null 2>&1
docker run --rm --name glm53-convert --memory "${MEM:-200g}" --cpus "${CPUS:-$(nproc)}" --entrypoint bash -v $ROOT:$ROOT -w $ROOT "$IMAGE" \
  -c "code/convert/convert_full.sh $SRC $DST $WORKERS 8 $STRIDE; rc=\$?; chown -R $OWNER $DST $ROOT/logs/convert; exit \$rc" 2>&1 | grep -v "^Calling"
rc=${PIPESTATUS[0]}
log "conversion container exit code $rc"
if [ "$rc" -eq 0 ]; then log "DONE: converted and validated: $DST ($(du -sh "$DST" | cut -f1))"; touch "$OKMARK"
else log "FAILED: see $ROOT/logs/convert/"; touch "$ROOT/logs/CONVERSION_FAILED"; fi
exit "$rc"
