# SPDX-License-Identifier: BSD-2-Clause
"""Generate deterministic JPEG2000 inputs and capture CPU Tier-1 reference blocks."""
import argparse
import json
import os
from pathlib import Path
import random
import subprocess


def generate(binary_dir, directory):
    directory.mkdir(parents=True, exist_ok=True)
    capture = directory / "blocks.jsonl"
    capture.write_text("", encoding="utf-8")
    rng = random.Random(53007)
    suffix = ".exe" if os.name == "nt" else ""
    compressor = binary_dir / ("opj_compress" + suffix)
    decoder = binary_dir / ("opj_decompress" + suffix)
    env = os.environ.copy()
    env.pop("OPJ_T1_CAPTURE_FILE", None)
    env["OPJ_OPENCL_DEVICE"]="off"
    env["OPJ_OPENCL_ENCODE_DEVICE"]="off"
    logs = []
    for size_index, (w, h) in enumerate(((17, 19), (64, 64), (129, 129))):
        patterns = [bytes([0]) * (w * h),
                    bytes((x * 7 + y * 13) % 256 for y in range(h) for x in range(w)),
                    bytes(rng.randrange(256) for _ in range(w * h))]
        for pattern, pixels in enumerate(patterns):
            pgm = directory / f"source-{size_index}-{pattern}.pgm"
            pgm.write_bytes(f"P5\n{w} {h}\n255\n".encode() + pixels)
            for style in range(64):
                for irreversible in (False, True):
                    name = f"{size_index}-{pattern}-{style}-{int(irreversible)}"
                    encoded = directory / (name + ".j2k")
                    args = [str(compressor), "-i", str(pgm), "-o", str(encoded),
                            "-n", "3", "-b", "64,64" if size_index == 2 else "16,16",
                            "-M", str(style), "-r", "8,1"]
                    if irreversible:
                        args.append("-I")
                    # Exercise component ROI upshift across all coding styles;
                    # the random pattern retains the non-ROI control path.
                    if pattern < 2:
                        args.extend(("-ROI", f"c=0,U={3 if pattern == 0 else 5}"))
                    result = subprocess.run(args, env=env, capture_output=True, text=True, check=True)
                    logs.append(result.stdout + result.stderr)
                    for discard, layer in ((0, 0), (1, 0), (0, 1)):
                        decode_env = dict(env, OPJ_T1_CAPTURE_FILE=str(capture.resolve()))
                        result = subprocess.run([str(decoder), "-i", str(encoded),
                                                 "-o", str(directory / "decoded.pgm"),
                                                 "-r", str(discard), "-l", str(layer), "-threads", "1"],
                                                env=decode_env, capture_output=True, text=True, check=True)
                        logs.append(result.stdout + result.stderr)
    (directory / "generation.log").write_text("\n".join(logs), encoding="utf-8")
    records = [json.loads(line) for line in capture.read_text().splitlines()]
    if not records:
        raise RuntimeError("No reference blocks: build with OPJ_CAPTURE_T1_REFERENCES=ON")
    print(f"Captured {len(records)} blocks / {sum(len(b['coefficients']) for b in records)} coefficients")
    return capture


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    generate(args.bin.resolve(), args.out.resolve())
