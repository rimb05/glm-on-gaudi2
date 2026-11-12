#!/bin/bash
# Download the files of one pinned revision of a public Hugging Face repository (no token, nothing from the
# repository is executed), then verify every hash. Resumable: run it again after an interruption.
# usage: download.sh MANIFEST.tsv DEST_DIR REPO REV [PARALLEL]      (MANIFEST.tsv and REV: fetch/make_manifest.py)
# Refuses to start if fewer than 300 GiB would stay free. Writes DEST_DIR/.VERIFIED when everything matches.
# Needs GNU coreutils and curl (it runs on the servers). RATE=9M caps each stream.
set -u
MAN="$1"; export DEST="$2"; export REPO="$3"; export REV="$4"; PAR="${5:-8}"
BIN="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$DEST"
need=$(awk -F'\t' '{s+=$1} END{print s}' "$MAN")
have=$(find "$DEST" -type f ! -name '*.partial' -printf '%s\n' 2>/dev/null | awk '{s+=$1} END{print s+0}')
avail=$(df --output=avail -B1 "$DEST" | tail -1)
reserve=$((300*1024*1024*1024))
echo "$(date -u +%FT%TZ) START repo=$REPO rev=$REV need=$need have=$have avail=$avail parallel=$PAR"
if [ $((need-have+reserve)) -gt "$avail" ]; then echo "$(date -u +%FT%TZ) ABORT not enough free space (keeps a 300 GiB reserve)"; exit 3; fi
tr '\t' '\n' < "$MAN" | xargs -d '\n' -n 4 -P "$PAR" "$BIN/dl_one.sh"
rc=$?
echo "$(date -u +%FT%TZ) TRANSFER_EXIT rc=$rc"
[ $rc -eq 0 ] || exit 1
echo "$(date -u +%FT%TZ) VERIFY sha256 of LFS files, git blob hash of the small ones"
awk -F'\t' '$2=="sha256"{print $3"  "$4}' "$MAN" > "$DEST/.sha256.expected"
( cd "$DEST" && awk '{print $2}' .sha256.expected | xargs -d '\n' -n 1 -P 24 sha256sum > .sha256.actual.unsorted )
sort -k2 "$DEST/.sha256.actual.unsorted" > "$DEST/.sha256.actual"; sort -k2 "$DEST/.sha256.expected" > "$DEST/.sha256.expected.sorted"
bad=0
if ! cmp -s "$DEST/.sha256.actual" "$DEST/.sha256.expected.sorted"; then bad=1; diff "$DEST/.sha256.actual" "$DEST/.sha256.expected.sorted" | head -40; fi
while IFS=$'\t' read -r size kind hash path; do
  [ "$kind" = gitblob ] || continue
  got=$( (printf 'blob %s\0' "$size"; cat "$DEST/$path") | sha1sum | awk '{print $1}')
  [ "$got" = "$hash" ] || { echo "GITBLOB MISMATCH $path"; bad=1; }
done < "$MAN"
n=$(find "$DEST" -type f ! -name '.sha256*' ! -name '.VERIFIED' | wc -l); want=$(wc -l < "$MAN")
[ "$n" = "$want" ] || { echo "FILE COUNT $n != $want"; bad=1; }
if [ $bad -eq 0 ]; then echo "$(date -u +%FT%TZ) VERIFIED files=$n bytes=$need"; touch "$DEST/.VERIFIED"; exit 0; fi
echo "$(date -u +%FT%TZ) VERIFY_FAILED"; exit 2
