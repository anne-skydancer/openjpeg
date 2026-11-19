# SPDX-License-Identifier: GPL-3.0-or-later
# neo-j2k 1a: writes a copy of a J2K codestream whose last tile-part has Psot = 0 (ISO/IEC 15444-1 A.4.2:
# the last tile-part may run to the EOC marker). Nothing else changes, so the copy decodes like the
# original, whole and at every cut point, when the decoder treats a missing EOC correctly.
# usage: py -3 -I make_psot0.py <in.j2k> <out.j2k>
import struct
import sys

src, dst = sys.argv[1], sys.argv[2]
b = bytearray(open(src, "rb").read())
if b[:2] != b"\xff\x4f":
    raise SystemExit("not a raw codestream")
pos = 2
while pos + 4 <= len(b):  # main header
    m, L = struct.unpack_from(">HH", b, pos)
    if m == 0xFF90:
        break
    pos += 2 + L
last = None
while pos + 12 <= len(b) and struct.unpack_from(">H", b, pos)[0] == 0xFF90:
    psot = struct.unpack_from(">I", b, pos + 6)[0]
    last = pos
    if psot == 0:
        break
    pos += psot
if last is None:
    raise SystemExit("no tile-part")
if b[-2:] != b"\xff\xd9" or struct.unpack_from(">I", b, last + 6)[0] + last != len(b) - 2:
    raise SystemExit("the last tile-part does not end at the final EOC")
struct.pack_into(">I", b, last + 6, 0)
open(dst, "wb").write(bytes(b))
print(f"{dst}: last tile-part at {last}, Psot set to 0")
