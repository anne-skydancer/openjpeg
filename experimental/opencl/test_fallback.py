# SPDX-License-Identifier: BSD-2-Clause
"""Verify CPU fallback with an invalid selector and an over-budget image."""
import argparse
import os
from pathlib import Path
import tempfile
from test_complete import execute,SUFFIX


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--device",required=True)
    parser.add_argument("--driver",default="")
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    root=Path(tempfile.mkdtemp(prefix="run-",dir=args.out)).resolve()
    binary=args.bin.resolve()
    cpu=os.environ.copy()
    for key in ("OPJ_OPENCL_DEVICE","OPJ_OPENCL_DRIVER","OPJ_T1_CAPTURE_FILE"):
        cpu.pop(key,None)
    cases=((32,"__nonexistent_opencl_gpu__"),(4096,args.device))
    for size,device in cases:
        pgm=root/f"source-{size}.pgm"
        pgm.write_bytes(f"P5\n{size} {size}\n255\n".encode()+bytes([127])*(size*size))
        source=root/f"source-{size}.j2k"
        execute([binary/("opj_compress"+SUFFIX),"-i",pgm,"-o",source,"-n",3],cpu)
        results=[]
        for label,env in (("cpu",cpu),("fallback",dict(cpu,OPJ_OPENCL_DEVICE=device,OPJ_OPENCL_DRIVER=args.driver))):
            target=root/f"{label}-{size}.pgm"
            log=execute([binary/("opj_decompress"+SUFFIX),"-i",source,"-o",target],env)
            if "OpenCL decoded tile" in log:
                raise AssertionError("Unexpected GPU execution for fallback case")
            results.append(target.read_bytes())
        if results[0]!=results[1]: raise AssertionError("CPU fallback changed output")
    print("PASS: missing device and over-budget input use CPU with identical pixels")


if __name__=="__main__": main()
