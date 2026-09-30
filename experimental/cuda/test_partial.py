# SPDX-License-Identifier: BSD-2-Clause
"""Partial downloads, ROI coding, reduced resolutions and area/component fallback."""
import argparse
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"opencl"))
from test_complete import execute,SUFFIX
p=argparse.ArgumentParser(description=__doc__)
p.add_argument("--bin",type=Path,required=True)
p.add_argument("--out",type=Path,required=True)
a=p.parse_args()
a.out.mkdir(parents=True,exist_ok=True)
root=Path(tempfile.mkdtemp(prefix="run-",dir=a.out)).resolve()
b=a.bin.resolve()
cpu=dict(os.environ,OPJ_DECODE_BACKEND="cpu",OPJ_OPENCL_ENCODE_DEVICE="off")
gpu=dict(cpu,OPJ_DECODE_BACKEND="cuda",OPJ_CUDA_DEVICE="auto")
raw=root/"source.raw"
rng=random.Random(591)
raw.write_bytes(bytes(rng.randrange(256) for _ in range(129*97*3)))
source=root/"source.j2k"
execute([b/("opj_compress"+SUFFIX),"-i",raw,"-o",source,"-F","129,97,3,8,u",
         "-n",4,"-r","8,4,1","-ROI","c=0,U=3","-M",63],cpu)
data=source.read_bytes()
cases=[]
for label,length in (("full",len(data)),("no-eoc",len(data)-2),("half",len(data)//2),("three-quarters",len(data)*3//4)):
    f=root/(label+".j2k");f.write_bytes(data[:length])
    for discard in (0,1,2): cases.append((label+str(discard),f,["-allow-partial","-r",str(discard)],None))
cases.extend((("area",source,["-d","1,1,32,32"],False),("component",source,["-c","0"],False)))
partial_gpu=0
for label,f,options,require in cases:
    outputs=[];codes=[];logs=[]
    for mode,env in (("cpu",cpu),("cuda",gpu)):
        folder=root/label/mode;folder.mkdir(parents=True)
        run=subprocess.run([str(b/("opj_decompress"+SUFFIX)),"-i",str(f),"-o",str(folder/"out.pgx"),*options],env=env,capture_output=True,text=True)
        codes.append(run.returncode);logs.append(run.stdout+run.stderr)
        outputs.append({x.name:x.read_bytes() for x in folder.glob("*.pgx")})
    assert codes[0]==codes[1],(label,codes,logs)
    assert outputs[0]==outputs[1],(label,"pixel mismatch",logs)
    accelerated="CUDA decoded tile" in logs[1]
    if require is not None: assert accelerated==require,(label,logs)
    if label.startswith("no-eoc") and accelerated: partial_gpu+=1
assert partial_gpu>0,"No truncated input exercised CUDA"
print("PASS:",len(cases),"partial/ROI/fallback comparisons;",partial_gpu,"incomplete downloads decoded on CUDA")
