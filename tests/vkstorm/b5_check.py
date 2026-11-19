# SPDX-License-Identifier: BSD-2-Clause
# neo-j2k 1a (audit B5): every whole-file decode limited to k layers (probe "layhash") must appear among the
# cut decodes of the same file at the same reduce (probe "cuthash"): in a layer-progressive (LRCP) single-tile
# stream, the cut at the end of layer k decodes to exactly the first k layers. Prints one line per file and the
# missing (reduce, layers) pairs.
# usage: py -3 -I b5_check.py <layhash file> <cuthash file> [label]
import sys


def read(path):
    rows = []
    for line in open(path, encoding="ascii", errors="replace"):
        f = line.split()
        if len(f) == 4:
            rows.append(f)
    return rows


lay, cut = read(sys.argv[1]), read(sys.argv[2])
label = sys.argv[3] if len(sys.argv) > 3 else sys.argv[2]
seen = {}
for r, length, stage, h in cut:
    if stage == "2":
        seen.setdefault((r, h), int(length))
missing = []
for r, k, stage, h in lay:
    if stage != "2" or (r, h) not in seen:
        missing.append(f"r{r} {k}")
print(f"{label}: {len(lay) - len(missing)} of {len(lay)} layer-limited decodes found among the cut decodes"
      + (f"; missing {', '.join(missing)}" if missing else ""))
