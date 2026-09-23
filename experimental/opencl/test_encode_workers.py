# SPDX-License-Identifier: BSD-2-Clause
"""Exact concurrent encoded streams with shared transform/Tier-1 admission."""
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
    references = root / "references"
    references.mkdir()
    binary = args.bin.resolve()
    cpu = os.environ.copy()
    for key in ("OPJ_OPENCL_DEVICE", "OPJ_OPENCL_ENCODE_DEVICE", "OPJ_OPENCL_DRIVER", "OPJ_OPENCL_WORKERS", "OPJ_OPENCL_PROFILE", "OPJ_T1_CAPTURE_FILE"):
        cpu.pop(key, None)
    sources = []
    # Three large RGB jobs cannot simultaneously fit in the 64 MiB device pool.
    # Smaller, heterogeneous jobs exercise reuse and independent kernel arguments.
    cases = [(1024,1024,3)]*3 + [(17,19,1),(129,97,3),(65,67,4),(1,17,2)]
    for index, (width,height,channels) in enumerate(cases):
        raw = root / f"{index}.raw"
        raw.write_bytes(b"".join(bytes([(index*31+c*47)%256])*(width*height)
                                 for c in range(channels)))
        source = root / f"{index}.j2k"
        command = [binary / ("opj_compress"+SUFFIX), "-i", raw, "-o", source,
                   "-F", f"{width},{height},{channels},8,u", "-n", min(3,min(width,height).bit_length()),
                   "-mct", int(channels>=3)]
        if index%2:
            command.append("-I")
        execute(command, cpu)
        sources.append(source)
    def run(mode, env, callers):
        return json.loads(execute([binary / ("bench_encode"+SUFFIX), mode, references,
                                   callers, 1, 2, *sources], env))
    expected = run("--write-reference", cpu, 4)
    for slots in (1,2,4,8):
        gpu = dict(cpu, OPJ_OPENCL_ENCODE_DEVICE=args.device, OPJ_OPENCL_DRIVER=args.driver,
                   OPJ_OPENCL_WORKERS=str(slots))
        if slots==4:
            gpu["OPJ_OPENCL_PROFILE"]="1"
            gpu["OPJ_OPENCL_DEVICE"]=args.device
        actual = run("--verify-reference", gpu, 8)
        assert actual["checksums"] == expected["checksums"], actual
        if slots<=4:assert 0 < actual["gpu_fused_tiles"] <= actual["tiles"],actual
        else:assert actual["gpu_fused_tiles"]==0,actual
        assert actual["gpu_tiles"] == actual["gpu_transform_tiles"] == actual["tiles"] == 3*len(sources), actual
        assert 0 < actual["peak_pool_bytes"] <= 64*1024*1024, actual
        assert 1 <= actual["peak_active"] <= slots, actual
        if slots>1:
            assert actual["peak_active"]>1, actual
    print("PASS: exact concurrent encoded streams at 1/2/4/8 slots; both stages on GPU; pool <=64 MiB")


if __name__ == "__main__":
    main()
