# SPDX-License-Identifier: BSD-2-Clause
"""Measure whole public-API decode throughput, including output readback/checksum."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess

p=argparse.ArgumentParser(description=__doc__)
p.add_argument("--bin",type=Path,required=True)
p.add_argument("--corpus",type=Path,required=True)
p.add_argument("--out",type=Path,required=True)
p.add_argument("--opencl-device",required=True)
p.add_argument("--callers",type=int,nargs="+",default=[1,4,8])
p.add_argument("--rounds",type=int,default=3)
p.add_argument("--repeat",type=int,default=3)
a=p.parse_args()
sources=sorted(a.corpus.resolve().glob("*.j2k"))
assert sources,"Empty corpus"
exe=a.bin.resolve()/("bench_workers.exe" if os.name=="nt" else "bench_workers")
base={k:v for k,v in os.environ.items() if not k.startswith(("OPJ_OPENCL_","OPJ_CUDA_","OPJ_DECODE_"))}
base["OPJ_OPENCL_ENCODE_DEVICE"]="off"
rows=[];reference=None
for callers in a.callers:
    for round in range(a.rounds):
        modes=("cpu","opencl","cuda") if round%2==0 else ("cuda","opencl","cpu")
        for backend in modes:
            env=dict(base,OPJ_DECODE_BACKEND=backend,OPJ_OPENCL_DEVICE=a.opencl_device)
            run=subprocess.run([str(exe),str(callers),"1",str(a.repeat),*[str(s) for s in sources]],
                               env=env,capture_output=True,text=True,check=True)
            row=json.loads(run.stdout)
            if reference is None: reference=row["checksums"]
            assert row["checksums"]==reference,(backend,"checksum mismatch")
            if backend!="cpu": assert row["gpu_tiles"]>0,(backend,"GPU did not execute")
            row.update(backend=backend,round=round)
            rows.append(row)
            print(backend,callers,round,row["warm_ms_per_image"],flush=True)
summary=[]
for callers in a.callers:
    for backend in ("cpu","opencl","cuda"):
        selected=[r for r in rows if r["backend"]==backend and r["workers"]==callers]
        summary.append(dict(backend=backend,callers=callers,
             median_ms_per_image=statistics.median(r["warm_ms_per_image"] for r in selected),
             gpu_tiles=[r["gpu_tiles"] for r in selected],tiles=[r["tiles"] for r in selected]))
a.out.parent.mkdir(parents=True,exist_ok=True)
a.out.write_text(json.dumps(dict(images=len(sources),summary=summary,runs=rows),indent=2))
print(json.dumps(summary,indent=2))
