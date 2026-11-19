#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# neo-j2k 1a: per-prefix output hashes of a list of codestreams with one probe build, in parallel.
# usage: run_cuthash.sh <probe exe> <out dir> <list file> [jobs]
#   list lines: <codestream path> <reduce list> <step>
# Writes <out dir>/<basename>.txt (one line per prefix: reduce len stage hash) and <out dir>/log.txt.
# Temporary files go to OPJ_TEST_TMP (a scratch folder excluded from malware scanning), else TEMP.
set -u
P=$1; OUT=$2; LIST=$3; J=${4:-6}
mkdir -p "$OUT"
export TEMP="${OPJ_TEST_TMP:-$TEMP}" TMP="${OPJ_TEST_TMP:-$TEMP}"
tr -d '\r' < "$LIST" | grep -v '^#' | grep -v '^$' | \
  xargs -P "$J" -L 1 sh -c 'f=$1; r=$2; s=$3; b=$(basename "$f"); "$0" cuthash "$f" "$r" "$s" "'"$OUT"'/$b.txt" > "'"$OUT"'/$b.log" 2>&1; echo "rc=$? $b"' "$P" > "$OUT/log.txt"
cat "$OUT"/*.log >> "$OUT/log.txt" 2>/dev/null
rm -f "$OUT"/*.log
