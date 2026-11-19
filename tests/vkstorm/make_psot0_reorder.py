# SPDX-License-Identifier: GPL-3.0-or-later
# neo-j2k 1a: from a multi-tile codestream with one tile-part per tile, writes a copy whose tile-parts are in the
# order 1, 2, ..., n-1, 0 (tile 0 last, a tile that is not the last index), with TNsot = 0 (count unknown) in
# every SOT, and Psot = 0 in the last tile-part. All of this is allowed by ISO/IEC 15444-1 A.4.2.
# usage: py -3 -I make_psot0_reorder.py <in.j2k> <out.j2k>
import struct
import sys

src, dst = sys.argv[1], sys.argv[2]
b = open(src, "rb").read()
pos = 2
while True:
    m, L = struct.unpack_from(">HH", b, pos)
    if m == 0xFF90:
        break
    pos += 2 + L
main = b[:pos]
parts = []
while pos + 12 <= len(b) and struct.unpack_from(">H", b, pos)[0] == 0xFF90:
    isot, psot, tpsot, tnsot = struct.unpack_from(">HIBB", b, pos + 4)
    if psot == 0 or tpsot != 0 or tnsot != 1:
        raise SystemExit("needs one tile-part per tile with Psot > 0")
    parts.append((isot, bytearray(b[pos:pos + psot])))
    pos += psot
if b[pos:] != b"\xff\xd9" or len(parts) < 2:
    raise SystemExit("unexpected end")
parts.sort(key=lambda p: (p[0] == 0, p[0]))  # tile 0 last
out = bytearray(main)
for i, (isot, tp) in enumerate(parts):
    tp[11] = 0  # TNsot unknown
    if i == len(parts) - 1:
        struct.pack_into(">I", tp, 6, 0)
    out += tp
out += b"\xff\xd9"
open(dst, "wb").write(bytes(out))
print(f"{dst}: {len(parts)} tile-parts, last is tile {parts[-1][0]} with Psot = 0")
