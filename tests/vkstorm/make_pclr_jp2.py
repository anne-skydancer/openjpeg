# SPDX-License-Identifier: BSD-2-Clause
# neo-j2k 1a: a JP2 file whose palette (pclr + cmap boxes) maps one W x W component to <npc> output
# components, around a codestream with one empty packet (audit 1a finding 4: palette expansion and the
# decoder's limits).
# usage: py -3 -I make_pclr_jp2.py <out.jp2> <W> <npc>
import struct
import sys


def box(kind, payload):
    return struct.pack(">I", 8 + len(payload)) + kind + payload


def codestream(w):
    b = bytearray(b"\xff\x4f")
    b += struct.pack(">HHHIIIIIIIIH", 0xFF51, 38 + 3, 0, w, w, 0, 0, w, w, 0, 0, 1) + bytes([7, 1, 1])
    # COD: no SOP/EPH, LRCP, 1 layer, no MCT, 5 decompositions, 64 x 64 code-blocks, 5/3
    b += struct.pack(">HHBBHB", 0xFF52, 12, 0, 0, 1, 0) + bytes([5, 4, 4, 0, 1])
    b += struct.pack(">HHB", 0xFF5C, 3 + 16, 0x40) + bytes([9 << 3]) * 16
    b += struct.pack(">HHHIBB", 0xFF90, 10, 0, 12 + 2 + 1, 0, 1) + b"\xff\x93" + b"\x00"
    b += b"\xff\xd9"
    return bytes(b)


def main():
    out, w, npc = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    ihdr = box(b"ihdr", struct.pack(">IIHBBBB", w, w, 1, 7, 7, 0, 0))
    colr = box(b"colr", struct.pack(">BBBI", 1, 0, 0, 16))
    pclr = box(b"pclr", struct.pack(">HB", 1, npc) + bytes([7]) * npc + bytes(range(npc)))
    cmap = box(b"cmap", b"".join(struct.pack(">HBB", 0, 1, i) for i in range(npc)))
    data = box(b"jP  ", b"\r\n\x87\n") + box(b"ftyp", b"jp2 " + struct.pack(">I", 0) + b"jp2 ")
    data += box(b"jp2h", ihdr + colr + pclr + cmap) + box(b"jp2c", codestream(w))
    open(out, "wb").write(data)
    print(f"{out}: {len(data)} bytes, {w} x {w}, palette of {npc} components")


main()
