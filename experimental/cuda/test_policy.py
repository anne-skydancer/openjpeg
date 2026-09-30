# SPDX-License-Identifier: BSD-2-Clause
"""Backend selection and conservative CPU fallback preserve exact pixels."""
import argparse
import os
from pathlib import Path
import sys
import tempfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"opencl"))
from test_complete import execute, SUFFIX

p=argparse.ArgumentParser(description=__doc__)
p.add_argument("--bin",type=Path,required=True)
p.add_argument("--out",type=Path,required=True)
a=p.parse_args()
a.out.mkdir(parents=True,exist_ok=True)
root=Path(tempfile.mkdtemp(prefix="run-",dir=a.out)).resolve()
b=a.bin.resolve()
base={k:v for k,v in os.environ.items() if not k.startswith(("OPJ_OPENCL_","OPJ_CUDA_","OPJ_DECODE_"))}
base["OPJ_OPENCL_ENCODE_DEVICE"]="off"
cpu=dict(base,OPJ_DECODE_BACKEND="cpu")
for size in (65,4096):
    raw=root/f"{size}.pgm"
    raw.write_bytes(f"P5\n{size} {size}\n255\n".encode()+bytes([127])*(size*size))
    source=root/f"{size}.j2k"
    execute([b/("opj_compress"+SUFFIX),"-i",raw,"-o",source,"-n",3],cpu)
    policies=[("cpu",cpu,False),
      ("cuda",dict(base,OPJ_DECODE_BACKEND="cuda"),size==65),
      ("auto",base,size==65),
      ("missing",dict(base,OPJ_DECODE_BACKEND="cuda",OPJ_CUDA_DEVICE="__missing__"),False),
      ("off",dict(base,OPJ_DECODE_BACKEND="cuda",OPJ_CUDA_DEVICE="off"),False),
      ("bad-workers",dict(base,OPJ_DECODE_BACKEND="cuda",OPJ_CUDA_WORKERS="9"),False),
      ("legacy-off",dict(base,OPJ_OPENCL_DEVICE="off"),False)]
    reference=None
    for label,env,on_gpu in policies:
        target=root/f"{size}-{label}.pgm"
        log=execute([b/("opj_decompress"+SUFFIX),"-i",source,"-o",target],env)
        assert ("CUDA decoded tile" in log)==on_gpu,(label,log)
        data=target.read_bytes()
        if reference is None: reference=data
        assert data==reference,label
print("PASS: automatic/explicit selection, CPU overrides, unavailable device, invalid configuration, over-budget fallback")
