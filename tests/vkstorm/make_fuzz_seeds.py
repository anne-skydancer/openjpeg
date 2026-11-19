# SPDX-License-Identifier: GPL-3.0-or-later
# neo-j2k 1a: seed corpus for fuzz1a (3 control bytes + codestream). Codestreams up to <max bytes>.
# usage: py -3 -I make_fuzz_seeds.py <out dir> <max bytes> <codestreams...>
import os
import random
import sys

out, maxb = sys.argv[1], int(sys.argv[2])
os.makedirs(out, exist_ok=True)
rnd = random.Random(1)
n = 0
for path in sys.argv[3:]:
    data = open(path, "rb").read()
    if len(data) > maxb or data[:2] != b"\xff\x4f":
        continue
    base = os.path.basename(path).replace(".", "_")
    # control: reduce 0/3/5, profile, header-once order, decode area; cut: whole and two random cuts
    for ctl in (0x00, 0x03, 0x05, 0x08, 0x0B, 0x25, 0x40):
        for cut in (0, rnd.randrange(1, 65536), rnd.randrange(1, 65536)):
            with open(os.path.join(out, f"{base}_{ctl:02x}_{cut:05d}"), "wb") as f:
                f.write(bytes([ctl, cut & 0xFF, cut >> 8]) + data)
            n += 1
print(f"{n} seeds")
