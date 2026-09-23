# SPDX-License-Identifier: BSD-2-Clause
"""Compare repeated decoding through reused GPU buffers, including CPU fallback."""
import argparse
import json
import os
from pathlib import Path
import tempfile
from test_complete import execute, SUFFIX


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--driver", default="")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="run-", dir=args.out)).resolve()
    binary = args.bin.resolve()
    cpu = os.environ.copy()
    for key in ("OPJ_OPENCL_DEVICE", "OPJ_OPENCL_DRIVER", "OPJ_OPENCL_PROFILE", "OPJ_T1_CAPTURE_FILE"):
        cpu.pop(key, None)
    cpu["OPJ_OPENCL_DEVICE"]="off"
    cpu["OPJ_OPENCL_ENCODE_DEVICE"]="off"
    sources = []
    for size in (32, 129, 4096, 65):
        pgm = root / f"{size}.pgm"
        pixels = (bytes([127]) * (size*size) if size == 4096 else
                  bytes((x*17 + y*23) % 256 for y in range(size) for x in range(size)))
        pgm.write_bytes(f"P5\n{size} {size}\n255\n".encode() + pixels)
        source = root / f"{size}.j2k"
        execute([binary / ("opj_compress" + SUFFIX), "-i", pgm, "-o", source, "-n", 3], cpu)
        sources.append(source)
    results = []
    for env in (cpu, dict(cpu, OPJ_OPENCL_DEVICE=args.device, OPJ_OPENCL_DRIVER=args.driver)):
        results.append(json.loads(execute([binary / ("bench_decode" + SUFFIX), 3, *sources], env)))
    expected, actual = results
    assert actual["images"] == expected["images"] == 16, results
    assert actual["gpu_tiles"] == 12 and expected["gpu_tiles"] == 0, results
    assert actual["checksum"] == expected["checksum"], results
    print("PASS: repeated small/large/fallback/small sequence retains identical pixels")


if __name__ == "__main__":
    main()
