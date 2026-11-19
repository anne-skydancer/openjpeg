# SPDX-License-Identifier: BSD-2-Clause
# neo-j2k 1a (audit B5): writes the prefixes of a layer-progressive (LRCP), single-tile, single-tile-part
# codestream that end exactly where each layer's packet data ends. The layer ends come from a whole-file
# decode by a "pktlog" variant (make_variant_src.py), so they also hold when the packet headers are packed
# in PPM or PPT, where the tile data carries the packet bodies only. A cut decode of the prefix of layer k
# must equal the whole file decoded with k layers (probe "layhash"): probe "cuthash" of each prefix with a
# step longer than the file gives its hash at full length, in the format b5_check.py reads.
# usage: py -3 -I layer_cuts.py <codestream> <pktlog output> <out dir>
import struct
import sys
from pathlib import Path

src, log, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
b = src.read_bytes()
pos = 2
while pos + 4 <= len(b):  # main header
    m, L = struct.unpack_from(">HH", b, pos)
    if m == 0xFF90:
        break
    pos += 2 + L
tile_parts = 0
data_start = None
p = pos
while p + 4 <= len(b) and struct.unpack_from(">H", b, p)[0] == 0xFF90:
    psot = struct.unpack_from(">I", b, p + 6)[0]
    tile_parts += 1
    q = p + 12
    while struct.unpack_from(">H", b, q)[0] != 0xFF93:
        q += 2 + struct.unpack_from(">H", b, q + 2)[0]
    data_start = q + 2
    if psot == 0:
        break
    p += psot
if tile_parts != 1:
    raise SystemExit(f"{src}: {tile_parts} tile-parts; this check needs one tile in one tile-part")
ends = {}
for line in log.read_text(encoding="ascii", errors="replace").splitlines():
    f = line.split()
    if len(f) == 7 and f[0] == "PKT":
        layer, end = int(f[2]), int(f[6])
        ends[layer] = max(ends.get(layer, 0), end)
out.mkdir(parents=True, exist_ok=True)
best = 0
for k in range(1, max(ends) + 2):
    best = max(best, ends.get(k - 1, 0))
    if k - 1 not in ends:
        break
    cut = out / f"{src.stem}_L{k}{src.suffix}"
    cut.write_bytes(b[:data_start + best])
    print(f"{cut.name} {data_start + best}")
