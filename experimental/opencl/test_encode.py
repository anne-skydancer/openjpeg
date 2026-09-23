# SPDX-License-Identifier: BSD-2-Clause
"""Compare CPU and opt-in GPU JPEG2000 encoding, including rate allocation."""
import argparse
import os
from pathlib import Path
import random
import subprocess
import tempfile
import struct


def run(args, env):
    result = subprocess.run(list(map(str,args)), env=env, capture_output=True, text=True, timeout=60)
    if result.returncode:
        raise RuntimeError(result.stdout+result.stderr)
    return result.stdout+result.stderr


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--device",required=True)
    parser.add_argument("--driver",default="")
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    root=Path(tempfile.mkdtemp(prefix="run-",dir=args.out)).resolve()
    binary=args.bin.resolve()/("opj_compress.exe" if os.name=="nt" else "opj_compress")
    cpu=os.environ.copy()
    for key in ("OPJ_OPENCL_DEVICE","OPJ_OPENCL_ENCODE_DEVICE","OPJ_OPENCL_DRIVER","OPJ_OPENCL_WORKERS","OPJ_OPENCL_PROFILE","OPJ_T1_CAPTURE_FILE"):
        cpu.pop(key,None)
    gpu=dict(cpu,OPJ_OPENCL_ENCODE_DEVICE=args.device,OPJ_OPENCL_DRIVER=args.driver)
    rng=random.Random(2392026)
    count=0
    for w,h in ((1,1),(2,3),(17,19),(64,64),(129,97),(257,259)):
        for channels in (1,3,4):
            raw=root/f"{w}-{h}-{channels}.raw"
            raw.write_bytes(bytes(rng.randrange(256) for _ in range(w*h*channels)))
            for irreversible in (False,True):
                for rates in ("1","16,8,1","12"):
                    name=f"{w}-{h}-{channels}-{int(irreversible)}-{rates}"
                    common=[binary,"-i",raw,"-F",f"{w},{h},{channels},8,u","-n",min(5,min(w,h).bit_length()),
                            "-b","64,64","-r",rates,"-threads",1,"-mct",int(channels>=3)]
                    if irreversible:common.append("-I")
                    if w>64:common += ["-d","1,1","-t","128,128"]
                    logs=[];data=[]
                    for label,env in (("cpu",cpu),("gpu",gpu)):
                        dest=root/(name+"-"+label+".j2k")
                        logs.append(run(common+["-o",dest],env));data.append(dest.read_bytes())
                    (root/(name+".log")).write_text("\n".join(logs))
                    if ("OpenCL fused encoding tile" not in logs[1] or
                            "OpenCL encoded Tier-1 tile" not in logs[1] or
                            logs[1].count("OpenCL forward transformed tile") != logs[1].count("OpenCL encoded Tier-1 tile")):
                        raise AssertionError(f"GPU encoder did not run: {name}\n{logs[1]}")
                    if data[0]!=data[1]:
                        offset=next((i for i,(a,b) in enumerate(zip(*data)) if a!=b),min(map(len,data)))
                        raise AssertionError(f"Different codestream: {name}, offset {offset}, sizes {list(map(len,data))}; {root}")
                    count+=1
    # Precision limits, signed samples, zero blocks and fallback controls.
    for signed in (False,True):
        for channels in (1,3,4):
            w,h=65,67
            raw=root/f"wide-{int(signed)}-{channels}.raw"
            values=[rng.randrange(-32768,32768) if signed else rng.randrange(65536)
                    for _ in range(w*h*channels)]
            raw.write_bytes(b"".join(struct.pack(">h" if signed else ">H",v) for v in values))
            for irreversible in (False,True):
                common=[binary,"-i",raw,"-F",f"{w},{h},{channels},16,{'s' if signed else 'u'}",
                        "-n",4,"-r","8,1","-threads",1,"-mct",int(channels>=3)]
                if irreversible:common.append("-I")
                outputs=[]
                for label,env in (("cpu",cpu),("gpu",gpu)):
                    dest=root/f"wide-{int(signed)}-{channels}-{int(irreversible)}-{label}.j2k"
                    log=run(common+["-o",dest],env)
                    if label=="gpu":
                        assert "OpenCL fused encoding tile" in log and "OpenCL forward transformed tile" in log and "OpenCL encoded Tier-1 tile" in log,log
                    outputs.append(dest.read_bytes())
                assert outputs[0]==outputs[1],(signed,channels,irreversible)
                count+=1
    for value in (0,128,255):
        raw=root/f"constant-{value}.raw";raw.write_bytes(bytes([value])*17*19)
        common=[binary,"-i",raw,"-F","17,19,1,8,u","-n",3,"-threads",1]
        outputs=[]
        for label,env in (("cpu",cpu),("gpu",gpu)):
            dest=root/f"constant-{value}-{label}.j2k"
            log=run(common+["-o",dest],env);outputs.append(dest.read_bytes())
            if label=="gpu":assert "OpenCL encoded Tier-1 tile" in log,log
        assert outputs[0]==outputs[1]
        count+=1
    # Non-default block shapes stress packed stripes and coefficient gathering.
    raw=root/"shapes.raw";raw.write_bytes(bytes(rng.randrange(256) for _ in range(137*139*3)))
    for shape in ("16,16","32,128","128,32","4,1024","1024,4","32,32"):
        for irreversible in (False,True):
            common=[binary,"-i",raw,"-F","137,139,3,8,u","-n",4,"-b",shape,"-r","16,8,1","-threads",1]
            if irreversible:common.append("-I")
            outputs=[]
            for label,env in (("cpu",cpu),("gpu",gpu)):
                dest=root/f"shape-{shape}-{irreversible}-{label}.j2k"
                log=run(common+["-o",dest],env);outputs.append(dest.read_bytes())
                if label=="gpu":assert "OpenCL fused encoding tile" in log,log
            assert outputs[0]==outputs[1],(shape,irreversible)
            count+=1
    # The complete pipeline exceeds 64 MiB. Separate stages/CPU must still
    # produce the exact codestream without keeping a worker lease or partial state.
    raw=root/"budget.raw";raw.write_bytes(bytes(range(256))*(2048*2048*3//256))
    outputs=[]
    common=[binary,"-i",raw,"-F","2048,2048,3,8,u","-n",5,"-threads",1]
    for label,env in (("cpu",cpu),("gpu",gpu)):
        dest=root/f"budget-{label}.j2k"
        log=run(common+["-o",dest],env);outputs.append(dest.read_bytes())
        if label=="gpu":assert "OpenCL fused encoding tile" not in log,log
    assert outputs[0]==outputs[1]
    count+=1
    raw=root/"fallback.raw";raw.write_bytes(bytes(rng.randrange(256) for _ in range(33*35)))
    for style,device,expected_transform in ((63,args.device,True),(0,"no-such-OpenCL-device",False)):
        common=[binary,"-i",raw,"-F","33,35,1,8,u","-n",3,"-M",style,"-threads",1]
        outputs=[]
        for label,env in (("cpu",cpu),("gpu",dict(gpu,OPJ_OPENCL_ENCODE_DEVICE=device))):
            dest=root/f"fallback-{style}-{label}.j2k"
            log=run(common+["-o",dest],env);outputs.append(dest.read_bytes())
            if label=="gpu":
                assert "OpenCL encoded Tier-1 tile" not in log,log
                assert ("OpenCL forward transformed tile" in log)==expected_transform,log
        assert outputs[0]==outputs[1]
        count+=1
    print(f"PASS: {count-3} fused GPU codestreams and 3 fallback cases byte-identical to CPU; {root}")


if __name__=="__main__":
    main()
