#!/bin/bash
# Fetch one file of the pinned revision with resume; exit 0 only when the size matches the manifest.
# usage: dl_one.sh SIZE KIND HASH PATH   (env: REPO REV DEST)
set -u
size="$1"; path="$4"
out="$DEST/$path"; part="$out.partial"
mkdir -p "$(dirname "$out")"
if [ -f "$out" ] && [ "$(stat -c %s "$out")" = "$size" ]; then echo "$(date -u +%FT%TZ) SKIP $path"; exit 0; fi
attempt=0
while [ $attempt -lt 60 ]; do
  attempt=$((attempt+1))
  # RATE (optional, e.g. 9M) caps each stream so other traffic on the shared line is not starved
  curl -fsSL --connect-timeout 30 --speed-limit 200000 --speed-time 120 -C - ${RATE:+--limit-rate "$RATE"} \
       -A "curl/7.81.0" -o "$part" "https://huggingface.co/$REPO/resolve/$REV/$path"
  rc=$?
  got=$(stat -c %s "$part" 2>/dev/null || echo 0)
  if [ "$got" = "$size" ]; then mv "$part" "$out"; echo "$(date -u +%FT%TZ) DONE $path $size"; exit 0; fi
  if [ "$got" -gt "$size" ]; then echo "$(date -u +%FT%TZ) OVERSIZE $path got=$got want=$size (restart)"; rm -f "$part"; fi
  echo "$(date -u +%FT%TZ) RETRY $path attempt=$attempt rc=$rc got=$got want=$size"
  if [ $attempt -lt 10 ]; then sleep 15; else sleep 90; fi
done
echo "$(date -u +%FT%TZ) FAIL $path"; exit 1
