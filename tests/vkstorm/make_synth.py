# SPDX-License-Identifier: GPL-3.0-or-later
# fork-audit: deterministic synthetic test images (PPM/PGM) for the truncation probes.
# usage: py -3 -I make_synth.py <out dir>
import math
import random
import sys
import os
out = sys.argv[1]


def img(w, h, ch, seed):
    rnd = random.Random(seed)
    px = bytearray()
    for y in range(h):
        for x in range(w):
            for c in range(ch):
                v = 128 + 60 * math.sin((x + 3 * c) * 0.21) * math.cos((y - 2 * c) * 0.13)
                v += 40 * ((x // 8 + y // 8 + c) % 2)          # edges
                v += (x * 37 + y * 11 + c * 5) % 23 - 11       # texture
                v += rnd.gauss(0, 6)                           # noise
                px.append(max(0, min(255, int(round(v)))))
    return px


for (w, h, ch) in [(256, 256, 3), (64, 64, 3), (32, 32, 3), (16, 16, 3), (128, 128, 4)]:
    data = img(w, h, ch, w * 1000 + ch)
    if ch == 3:
        name = os.path.join(out, f"synth_{w}x{h}.ppm")
        with open(name, "wb") as f:
            f.write(f"P6\n{w} {h}\n255\n".encode())
            f.write(bytes(data))
    else:
        # RGBA as four PGM planes is awkward for opj_compress; write RGB + separate alpha PGM
        rgb = bytearray()
        a = bytearray()
        for i in range(w * h):
            rgb += data[i * 4:i * 4 + 3]
            a.append(data[i * 4 + 3])
        with open(os.path.join(out, f"synth_{w}x{h}_rgb.ppm"), "wb") as f:
            f.write(f"P6\n{w} {h}\n255\n".encode())
            f.write(bytes(rgb))
print("ok")
