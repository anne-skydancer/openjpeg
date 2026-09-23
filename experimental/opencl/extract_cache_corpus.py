# SPDX-License-Identifier: BSD-2-Clause
"""Read-only extraction of complete J2C samples from the viewer texture cache.

The Windows cache layout is verified against Vulkanstorm lltexturecache.h/cpp:
44-byte EntriesInfo, 28-byte Entry, 600-byte prefix per entry, UUID body file.
Copies are private test inputs; never publish cache contents with this source.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import struct
import uuid


def extract(cache, output, limit):
    output.mkdir(parents=True, exist_ok=False)
    entries_path, prefix_path = cache / "texture.entries", cache / "texture.cache"
    before = (entries_path.stat(), prefix_path.stat())
    entries = entries_path.read_bytes()
    version, address, encoder, count = struct.unpack_from("<fI32sI",entries)
    if not (abs(version-1.71)<0.001 and address==64 and len(entries)==44+28*count):
        raise ValueError("Unrecognized cache format")
    order = list(range(count))
    random.Random(53097).shuffle(order)
    buckets = {}
    records = []
    with prefix_path.open("rb") as prefixes:
        for index in order:
            identity, image_size, body_size, timestamp = struct.unpack_from("<16siiI",entries,44+28*index)
            if not (0 < image_size <= 8*1024*1024 and body_size >= max(0,image_size-600)):
                continue
            prefixes.seek(index*600)
            prefix = prefixes.read(min(600,image_size))
            if not prefix.startswith(b"\xff\x4f\xff\x51") or len(prefix)<45:
                continue
            # SIZ: marker, Lsiz, Rsiz, Xsiz,Ysiz,XOsiz,YOsiz,...,Csiz.
            xs,ys,xo,yo = struct.unpack_from(">IIII",prefix,8)
            components = struct.unpack_from(">H",prefix,40)[0]
            w,h=xs-xo,ys-yo
            if not (0<w<=1024 and 0<h<=1024 and components in (1,3,4)):
                continue
            if any(prefix[42+3*c] != 7 for c in range(components)):
                continue
            key=(components,max(w,h).bit_length())
            if buckets.get(key,0)>=8:
                continue
            body=b""
            if image_size>600:
                name=str(uuid.UUID(bytes=identity))
                path=cache/name[0]/(name+".texture")
                try:
                    with path.open("rb") as f: body=f.read(image_size-600)
                except FileNotFoundError:
                    continue
            data=prefix+body
            if len(data)!=image_size or not data.endswith(b"\xff\xd9"):
                continue
            digest=hashlib.sha256(data).hexdigest()
            name=digest[:16]+".j2k"
            (output/name).write_bytes(data)
            records.append(dict(file=name,sha256=digest,bytes=len(data),width=w,height=h,components=components))
            buckets[key]=buckets.get(key,0)+1
            if len(records)>=limit:
                break
    after=(entries_path.stat(),prefix_path.stat())
    if any((a.st_size,a.st_mtime_ns)!=(b.st_size,b.st_mtime_ns) for a,b in zip(before,after)):
        raise RuntimeError("Cache index changed during extraction; copies are not a verified snapshot")
    if not records: raise RuntimeError("No complete eligible J2C inputs")
    (output/"manifest.json").write_text(json.dumps(dict(cache_version=version,records=records),indent=2))
    print(f"Copied {len(records)} complete textures; {sum(r['bytes'] for r in records)} compressed bytes")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--limit",type=int,default=32)
    args=parser.parse_args()
    extract(args.cache.resolve(),args.out.resolve(),args.limit)
