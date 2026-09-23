# SPDX-License-Identifier: BSD-2-Clause
"""End-to-end API/CLI comparison: GPU backend must run and match CPU pixels."""
import argparse
import os
from pathlib import Path
import subprocess
import random
import time
import json
import tempfile
import struct

SUFFIX = ".exe" if os.name == "nt" else ""


def execute(args, env):
    result = subprocess.run([str(a) for a in args], env=env, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    return result.stdout + result.stderr


def compare(source, binary, directory, gpu_env, cpu_env, discard=0, layer=0):
    cpu_dir, gpu_dir = directory / "cpu", directory / "gpu"
    cpu_dir.mkdir(parents=True, exist_ok=True)
    gpu_dir.mkdir(parents=True, exist_ok=True)
    logs = []
    outputs = []
    for target, env in ((cpu_dir, cpu_env), (gpu_dir, gpu_env)):
        # Each case has a unique directory; old files cannot mask missing output.
        log = execute([binary / ("opj_decompress" + SUFFIX), "-i", source, "-o", target / "image.pgx",
                       "-r", discard, "-l", layer, "-threads", "1"], env)
        logs.append(log)
        outputs.append({p.name: p.read_bytes() for p in target.glob("*.pgx")})
    (directory / "decode.log").write_text("\n".join(logs))
    if (not logs[0].count("Header of tile ") or
            logs[1].count("OpenCL decoded tile") != logs[0].count("Header of tile ")):
        raise AssertionError(f"GPU backend did not run: {source}\n{logs[1]}")
    if not outputs[0] or outputs[0] != outputs[1]:
        mismatches = []
        for name, expected in outputs[0].items():
            actual = outputs[1].get(name, b"")
            differences = [(i, a, b) for i, (a, b) in enumerate(zip(expected, actual)) if a != b]
            if differences or len(expected) != len(actual):
                mismatches.append((name, len(expected), len(actual), differences[:8]))
        raise AssertionError(f"Pixel mismatch for {source}: {mismatches}")
    return len(outputs[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--driver", default="")
    parser.add_argument("--cache-corpus", type=Path)
    args = parser.parse_args()
    binary = args.bin.resolve()
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="run-",dir=output))
    cpu_env = os.environ.copy()
    for key in ("OPJ_OPENCL_DEVICE", "OPJ_OPENCL_DRIVER", "OPJ_T1_CAPTURE_FILE"):
        cpu_env.pop(key, None)
    gpu_env = dict(cpu_env, OPJ_OPENCL_DEVICE=args.device, OPJ_OPENCL_DRIVER=args.driver)
    rng = random.Random(97053)
    count = components = 0
    started = time.perf_counter()
    if args.cache_corpus:
        sources = sorted(args.cache_corpus.glob("*.j2k"))
        if not sources:
            raise ValueError("Empty cache corpus")
        for source in sources:
            for discard in (0, 1, 2):
                case = root / f"{source.stem}-{discard}"
                case.mkdir()
                components += compare(source,binary,case,gpu_env,cpu_env,discard)
                count += 1
    else:
        for w,h in ((17,19),(129,97),(1,1),(1,17),(2,3),(65,67)):
            for channels in (1,2,3,4):
                raw = root / f"{w}-{h}-{channels}.raw"
                raw.write_bytes(bytes(rng.randrange(256) for _ in range(w*h*channels)))
                for irreversible in (0,1):
                    case = root / f"{w}-{h}-{channels}-{irreversible}"
                    case.mkdir()
                    source = case / "source.j2k"
                    levels = min(3,min(w,h).bit_length())
                    command = [binary / ("opj_compress" + SUFFIX), "-i", raw, "-o", source,
                               "-F", f"{w},{h},{channels},8,u", "-n", levels, "-b", "16,16",
                               "-M", 63, "-r", "8,1", "-mct", int(channels >= 3)]
                    if irreversible: command.append("-I")
                    if w > 2 and h > 2: command.extend(("-d","1,1","-t","64,64"))
                    execute(command,cpu_env)
                    for discard,layer in ((0,0),(min(1,levels-1),0),(0,1)):
                        run = case / f"run-{count}"
                        run.mkdir()
                        components += compare(source,binary,run,gpu_env,cpu_env,discard,layer)
                        count += 1
        # Packed stripe scratch must cover partial four-row columns even when
        # an image is narrower/shorter than its nominal code-block geometry.
        for w,h,block in ((1024,1,"1024,4"),(1024,3,"1024,4"),
                          (1,1024,"4,1024"),(3,1024,"4,1024"),
                          (512,5,"512,8"),(5,512,"8,512")):
            raw = root / f"edge-{w}-{h}.raw"
            raw.write_bytes(bytes(rng.randrange(256) for _ in range(w*h)))
            for style in (0,63):
                case=root / f"edge-{w}-{h}-{style}"
                case.mkdir()
                source=case / "source.j2k"
                execute([binary / ("opj_compress"+SUFFIX), "-i", raw, "-o", source,
                         "-F", f"{w},{h},1,8,u", "-n", 1, "-b", block,
                         "-M", style],cpu_env)
                components += compare(source,binary,case,gpu_env,cpu_env)
                count += 1
        # Exercise the upper admitted precision and signed output independently
        # of the larger 8-bit geometry/style matrix above.
        for signed in (False, True):
            for channels in (1,3,4):
                w,h=17,19
                values=[rng.randrange(65536) - (32768 if signed else 0)
                        for _ in range(w*h*channels)]
                raw=root / f"precision16-{signed}-{channels}.raw"
                raw.write_bytes(struct.pack(">" + ("h" if signed else "H")*len(values),*values))
                for irreversible in (0,1):
                    case=root / f"precision16-{signed}-{channels}-{irreversible}"
                    case.mkdir()
                    source=case / "source.j2k"
                    command=[binary / ("opj_compress" + SUFFIX), "-i",raw,"-o",source,
                             "-F",f"{w},{h},{channels},16,{'s' if signed else 'u'}",
                             "-n",3,"-b","16,16","-mct",int(channels>=3)]
                    if irreversible: command.append("-I")
                    execute(command,cpu_env)
                    components+=compare(source,binary,case,gpu_env,cpu_env)
                    count+=1
    summary = dict(cases=count,component_images=components,seconds=time.perf_counter()-started)
    (root / "summary.json").write_text(json.dumps(summary,indent=2))
    (output / "summary.json").write_text(json.dumps(dict(summary,run_directory=str(root)),indent=2))
    print("PASS:",summary)


if __name__ == "__main__": main()
