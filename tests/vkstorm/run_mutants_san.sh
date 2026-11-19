#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# neo-j2k 1a: decode every damaged codestream of a list under ASan+UBSan (non-strict, reduce 0 and 5), one process
# per file; collect sanitizer reports and non-zero exits. Aggregates only (no file names in the summary line).
# usage: run_mutants_san.sh <list> <probe exe> <out> [jobs]     (PROBE_PROFILE=sl in the environment for the profile)
# Needs OPJ_LLVM_RT (the folder of the clang runtime DLLs); temporary files go to OPJ_TEST_TMP, else TEMP.
LIST=$1; PROBE=$2; OUT=$3; J=${4:-3}
export ASAN_OPTIONS=allocator_may_return_null=1:detect_leaks=0
export PATH="${OPJ_LLVM_RT:?set OPJ_LLVM_RT to the folder of the clang runtime DLLs}:$PATH"
export TEMP="${OPJ_TEST_TMP:-$TEMP}" TMP="${OPJ_TEST_TMP:-$TEMP}"
export TMPD="$TEMP/mut-tmp"
mkdir -p "$TMPD"
: > "$OUT"
tr -d '\r' < "$LIST" | xargs -P "$J" -I{} sh -c 't="$TMPD/$(basename "$1").$$.txt"; "$0" hash 0,5 "$1" > "$t.out" 2> "$t"; rc=$?; if [ -s "$t" ] || [ $rc -ne 0 ]; then echo "rc=$rc $(basename "$1")"; grep -m3 -E "runtime error|AddressSanitizer|SUMMARY" "$t"; fi; grep -c FAILED "$t.out" >> "$TMPD/failed.count"; rm -f "$t" "$t.out"' "$PROBE" {} >> "$OUT"
n=$(tr -d '\r' < "$LIST" | grep -c .)
f=$(awk '{s+=$1} END {print s+0}' "$TMPD/failed.count")
rm -f "$TMPD/failed.count"
echo "files $n, decodes $((n * 2)), refused or failed decodes $f, reports $(grep -c '^rc=' "$OUT")" >> "$OUT"
