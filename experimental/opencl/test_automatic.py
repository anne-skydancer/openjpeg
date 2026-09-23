# SPDX-License-Identifier: BSD-2-Clause
"""Verify no-variable GPU activation, independent CPU overrides and shared selection."""
import argparse
import json
import os
from pathlib import Path
import tempfile
from test_complete import execute, SUFFIX


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--device",required=True)
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    root=Path(tempfile.mkdtemp(prefix="run-",dir=args.out)).resolve()
    binary=args.bin.resolve()
    automatic={k:v for k,v in os.environ.items() if not k.startswith("OPJ_OPENCL_") and k!="OPJ_T1_CAPTURE_FILE"}
    cpu=dict(automatic,OPJ_OPENCL_DEVICE="off",OPJ_OPENCL_ENCODE_DEVICE="off")
    raw=root/"source.pgm"
    raw.write_bytes(b"P5\n65 67\n255\n"+bytes((x*19+y*31)%256 for y in range(67) for x in range(65)))
    source=root/"cpu.j2k"
    execute([binary/("opj_compress"+SUFFIX),"-i",raw,"-o",source,"-n",4],cpu)
    reference=source.read_bytes()
    policies=[("automatic",automatic,True,True),
              ("auto-alias",dict(automatic,OPJ_OPENCL_DEVICE="auto",OPJ_OPENCL_ENCODE_DEVICE=""),True,True),
              ("decode-off",dict(automatic,OPJ_OPENCL_DEVICE="off"),False,True),
              ("encode-off",dict(automatic,OPJ_OPENCL_ENCODE_DEVICE="0"),True,False),
              ("cpu",cpu,False,False),
              ("missing",dict(automatic,OPJ_OPENCL_DEVICE="__missing_gpu__",OPJ_OPENCL_ENCODE_DEVICE="__missing_gpu__"),False,False),
              ("missing-driver",dict(automatic,OPJ_OPENCL_DRIVER="__missing_driver__"),False,False)]
    for name,env,decode_gpu,encode_gpu in policies:
        encoded=root/f"{name}.j2k"
        log=execute([binary/("opj_compress"+SUFFIX),"-i",raw,"-o",encoded,"-n",4],env)
        assert ("OpenCL encoded Tier-1 tile" in log)==encode_gpu,(name,log)
        assert encoded.read_bytes()==reference,name
        decoded=root/f"{name}.pgm"
        log=execute([binary/("opj_decompress"+SUFFIX),"-i",source,"-o",decoded],env)
        assert ("OpenCL decoded tile" in log)==decode_gpu,(name,log)
        # PGM writer may add a comment; compare the final sample plane.
        assert decoded.read_bytes()[-65*67:]==raw.read_bytes()[-65*67:],name
    references=root/"references";references.mkdir()
    command=[binary/("bench_encode"+SUFFIX)]
    expected=json.loads(execute(command+["--write-reference",references,1,1,1,source],cpu))
    # One process first decodes the source, then encodes repeatedly. Automatic
    # and matching explicit selectors must share the same runtime. A failed
    # decoder selector must not poison the encoder automatic initialization.
    for env in (automatic,
                dict(automatic,OPJ_OPENCL_DEVICE=args.device),
                dict(automatic,OPJ_OPENCL_ENCODE_DEVICE=args.device),
                dict(automatic,OPJ_OPENCL_DEVICE="__missing_gpu__"),
                dict(automatic,OPJ_OPENCL_DEVICE="x"*512)):
        result=json.loads(execute(command+["--verify-reference",references,1,1,1,source],env))
        assert result["checksums"]==expected["checksums"],result
        assert result["gpu_tiles"]==result["gpu_transform_tiles"]==result["tiles"]==2,result
    print("PASS: automatic GPU encode/decode with no variables, independent CPU overrides, missing-device/driver fallback, and shared runtime selection")


if __name__=="__main__":
    main()
