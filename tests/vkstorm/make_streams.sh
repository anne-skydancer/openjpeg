#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# fork-audit: encode the synthetic test codestreams with the fork's opj_compress (release build).
# usage: make_streams.sh <opj_compress.exe> <work/synth dir>   (after make_synth.py)
set -u
C=$1; W=$2; cd "$W"
enc() { out=$1; shift; "$C" -i "$@" -o "$out" > /dev/null 2>&1 || echo "FAIL $out"; }
enc lrcp97_256.j2k        synth_256x256.ppm -I -p LRCP -n 6 -b 64,64 -r 80,40,20,10,5
enc lrcp97_64.j2k         synth_64x64.ppm   -I -p LRCP -n 6 -b 64,64 -r 40,20,10,5,2.5
enc lrcp53_256.j2k        synth_256x256.ppm -p LRCP -n 6 -b 64,64 -r 64,32,16,8,1
enc lrcp53_64.j2k         synth_64x64.ppm   -p LRCP -n 6 -b 64,64 -r 64,32,16,8,1
enc lrcp53_32.j2k         synth_32x32.ppm   -p LRCP -n 6 -b 64,64 -r 32,16,8,4,1
enc rlcp97_256.j2k        synth_256x256.ppm -I -p RLCP -n 6 -b 64,64 -r 80,40,20,10,5
enc lrcp97eph_256.j2k     synth_256x256.ppm -I -p LRCP -n 6 -b 64,64 -r 80,40,20,10,5 -EPH
enc lrcp97sop_256.j2k     synth_256x256.ppm -I -p LRCP -n 6 -b 64,64 -r 80,40,20,10,5 -SOP
enc lrcp53eph_64.j2k      synth_64x64.ppm   -p LRCP -n 6 -b 64,64 -r 64,32,16,8,1 -EPH
enc lrcp97byp_256.j2k     synth_256x256.ppm -I -p LRCP -n 6 -b 64,64 -r 80,40,20,10,5 -M 1
enc lrcp97bypterm_256.j2k synth_256x256.ppm -I -p LRCP -n 6 -b 64,64 -r 80,40,20,10,5 -M 5
enc lrcp53all_64.j2k      synth_64x64.ppm   -p LRCP -n 6 -b 64,64 -r 64,32,16,8,1 -M 63
enc w2_16_n5.j2k          synth_16x16.ppm   -p LRCP -n 5
