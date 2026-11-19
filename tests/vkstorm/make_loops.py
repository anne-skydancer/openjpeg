# SPDX-License-Identifier: GPL-3.0-or-later
# neo-j2k 1a: writes the fork-audit probe's "loops" codestream (probe.c cmd_loops) to a file: W x W, 3 components,
# one tile, LRCP, <layers> layers, Scod flags, optional 2^pp precincts, and one empty packet of data.
# usage: py -3 -I make_loops.py <out> <W> <layers> <scod> [pp]
import struct
import sys

out, W, layers, scod = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4], 0)
pp = int(sys.argv[5]) if len(sys.argv) > 5 else 0
b = bytearray()
b += struct.pack(">H", 0xFF4F)
b += struct.pack(">HHHIIIIIIIIH", 0xFF51, 38 + 3 * 3, 0, W, W, 0, 0, W, W, 0, 0, 3) + bytes([7, 1, 1]) * 3
b += struct.pack(">HHBBHB", 0xFF52, 12 + (6 if pp else 0), scod | (1 if pp else 0), 0, layers, 0) + bytes([5, 4, 4, 0, 1])
if pp:
    b += bytes([pp | (pp << 4)]) * 6
b += struct.pack(">HHB", 0xFF5C, 3 + 16, 0x40) + bytes([9 << 3]) * 16
b += struct.pack(">HHHIBB", 0xFF90, 10, 0, 12 + 2 + 1, 0, 1) + struct.pack(">H", 0xFF93) + b"\x00"
b += struct.pack(">H", 0xFFD9)
open(out, "wb").write(bytes(b))
print(f"{out}: {len(b)} bytes")
