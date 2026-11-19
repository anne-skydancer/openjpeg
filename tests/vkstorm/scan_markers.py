# SPDX-License-Identifier: GPL-3.0-or-later
# neo-j2k 1a: list the markers, progression, tiles, layers and Psot=0 tile-parts of J2K codestreams (read only).
# usage: py -3 -I scan_markers.py <files...>
import struct
import sys
NAMES = {0xFF52: "COD", 0xFF53: "COC", 0xFF5C: "QCD", 0xFF5D: "QCC", 0xFF5E: "RGN", 0xFF5F: "POC",
         0xFF55: "TLM", 0xFF57: "PLM", 0xFF58: "PLT", 0xFF60: "PPM", 0xFF61: "PPT", 0xFF63: "CRG", 0xFF64: "COM"}
PROG = ["LRCP", "RLCP", "RPCL", "PCRL", "CPRL"]


def scan(path):
    b = open(path, "rb").read()
    if b[:4] == b"\x00\x00\x00\x0c":  # JP2: find jp2c
        i = b.find(b"jp2c")
        if i < 0:
            return None
        b = b[i + 4:]
    if b[:2] != b"\xff\x4f":
        return None
    pos, seen, info = 2, set(), {}
    while pos + 4 <= len(b):
        m, L = struct.unpack_from(">HH", b, pos)
        if m == 0xFF90:
            break
        if m == 0xFF51:
            xs, ys, xo, yo, xt, yt, xto, yto, cs = struct.unpack_from(">IIIIIIIIH", b, pos + 6)
            info["size"] = f"{xs - xo}x{ys - yo}"
            info["tiles"] = ((xs - xto + xt - 1) // xt) * ((ys - yto + yt - 1) // yt)
            info["comps"] = cs
        if m == 0xFF52:
            scod, prog, layers = struct.unpack_from(">BBH", b, pos + 4)
            info["prog"] = PROG[prog] if prog < 5 else prog
            info["layers"] = layers
            info["sop"] = bool(scod & 2)
            info["eph"] = bool(scod & 4)
            info["res"] = b[pos + 9] + 1
            info["cblksty"] = b[pos + 12]
        if m in NAMES:
            seen.add(NAMES[m])
        pos += 2 + L
    # tile-parts
    psot0 = 0
    tps = 0
    while pos + 12 <= len(b) and struct.unpack_from(">H", b, pos)[0] == 0xFF90:
        Lsot, isot, psot, tpsot, tnsot = struct.unpack_from(">HHIBB", b, pos + 2)
        tps += 1
        # tile-part header markers
        q = pos + 12
        while q + 4 <= len(b):
            m, L = struct.unpack_from(">HH", b, q)
            if m == 0xFF93:
                break
            if m in NAMES:
                seen.add("t" + NAMES[m])
            q += 2 + L
        if psot == 0:
            psot0 += 1
            break
        pos += psot
    info["tileparts"] = tps
    info["psot0"] = psot0
    info["markers"] = ",".join(sorted(seen))
    return info


for p in sys.argv[1:]:
    print(p.split("/")[-1].split("\\")[-1], scan(p))
