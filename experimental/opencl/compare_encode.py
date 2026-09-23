# SPDX-License-Identifier: BSD-2-Clause
"""Paired encoder benchmark; input JPEG2000 files are decoded before timing."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline",type=Path,required=True,help="Baseline bench_encode executable")
    parser.add_argument("--candidate",type=Path,required=True)
    parser.add_argument("--device",required=True)
    parser.add_argument("--driver",default="")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--trials",type=int,default=3)
    parser.add_argument("--rounds",type=int,default=3)
    parser.add_argument("inputs",type=Path,nargs="+")
    args=parser.parse_args()
    if args.trials<1 or args.rounds<1:parser.error("trials and rounds must be positive")
    env=os.environ.copy()
    for key in ("OPJ_OPENCL_DEVICE","OPJ_OPENCL_ENCODE_DEVICE","OPJ_OPENCL_PROFILE","OPJ_OPENCL_WORKERS","OPJ_T1_CAPTURE_FILE"):
        env.pop(key,None)
    env["OPJ_OPENCL_DEVICE"]="off"
    env["OPJ_OPENCL_ENCODE_DEVICE"]="off"
    modes=[("cpu",1,1),("cpu",1,4),("cpu",4,1),("cpu",8,1)]
    modes += [(mode,workers,1) for mode in ("baseline","candidate") for workers in (1,2,4,8)]
    results=[];expected={}
    metadata={"inputs":[{"sha256":hashlib.sha256(p.read_bytes()).hexdigest(),"bytes":p.stat().st_size} for p in args.inputs],
              "device":args.device,"driver":args.driver,"rounds":args.rounds,"trials":args.trials}
    metadata["binaries"]={}
    for label,binary in (("baseline",args.baseline),("candidate",args.candidate)):
        library=binary.parent/("openjp2.dll" if os.name=="nt" else "libopenjp2.so")
        metadata["binaries"][label]={"executable_sha256":hashlib.sha256(binary.read_bytes()).hexdigest(),
            "library_sha256":hashlib.sha256(library.read_bytes()).hexdigest() if library.exists() else None}
    for trial in range(args.trials):
        cases=[(lossy,*mode) for lossy in (False,True) for mode in modes]
        if trial%2:cases.reverse()
        for lossy,mode,workers,threads in cases:
            binary=args.baseline if mode=="baseline" else args.candidate
            runenv=env.copy()
            if mode!="cpu":runenv.update(OPJ_OPENCL_ENCODE_DEVICE=args.device,OPJ_OPENCL_DRIVER=args.driver,OPJ_OPENCL_WORKERS=str(workers))
            command=[binary]+(["--lossy","8"] if lossy else [])+[workers,threads,args.rounds,*args.inputs]
            proc=subprocess.run(list(map(str,command)),env=runenv,text=True,capture_output=True,timeout=300,check=True)
            result=json.loads(proc.stdout)
            tiles=(args.rounds+1)*len(args.inputs)
            if mode!="cpu":
                assert result["gpu_tiles"]==result["gpu_transform_tiles"]==result["tiles"]==tiles,result
                if mode=="candidate":
                    if workers<=4:assert 0<result["gpu_fused_tiles"]<=tiles,result
                    else:assert result["gpu_fused_tiles"]==0,result
            else:assert result["gpu_tiles"]==0,result
            if lossy not in expected:expected[lossy]=result["checksums"]
            assert result["checksums"]==expected[lossy],(mode,lossy,workers)
            result.update(trial=trial,mode=mode,lossy=lossy)
            results.append(result)
            args.output.write_text(json.dumps({"metadata":metadata,"results":results},indent=2))
            print(trial,lossy,mode,workers,threads,result["images_per_second"],flush=True)
    for lossy in (False,True):
        for mode,workers,threads in modes:
            rates=[r["images_per_second"] for r in results if r["lossy"]==lossy and r["mode"]==mode and r["workers"]==workers and r["cpu_threads"]==threads]
            print("SUMMARY",lossy,mode,workers,threads,round(statistics.mean(rates),3),min(rates),max(rates),flush=True)


if __name__=="__main__":
    main()
