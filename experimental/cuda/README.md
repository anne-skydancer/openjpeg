# NVIDIA CUDA decoding

This backend accelerates OpenJPEG itself and is enabled by default for native
64-bit Windows (MSVC/x86-64) and Linux (x86-64/AArch64) builds. It does not use nvJPEG2000.
The public synchronous decoder API and encoder behaviour are unchanged. Explicit
`OPJ_ENABLE_CUDA=OFF` builds retain the existing CPU/OpenCL paths.

## Build

The validated Windows build uses CUDA **12.9.1** (nvcc 12.9.86), Visual Studio
2022/MSVC 19.44, and `RelWithDebInfo`. A CUDA toolkit and its supported host C++
compiler are build dependencies; applications only need a compatible NVIDIA
driver at runtime. Neither cudart nor nvJPEG is linked or distributed.

```powershell
cmake -S . -B build/cuda -G "Visual Studio 17 2022" -A x64 `
  -DOPJ_CUDA_ROOT="C:/path/to/cuda-12.9" `
  -DBUILD_SHARED_LIBS=ON -DBUILD_CODEC=ON -DBUILD_TESTING=OFF `
  -DOPJ_BUILD_CUDA_EXPERIMENTS=ON
cmake --build build/cuda --config RelWithDebInfo --parallel
ctest --test-dir build/cuda -C RelWithDebInfo -L cuda --output-on-failure
```

`OPJ_ENABLE_OPENCL` can stay enabled for other vendors and for encoding. It can
also be disabled independently. The host backend contains a Linux Driver API
loader, but Linux compilation and execution have not yet been qualified.

Default native kernel targets are SM 50, 52, 60, 61, 70, 75, 80, 86, 89, 90 and
120, plus compute-50 PTX for driver JIT. Override `OPJ_CUDA_ARCHITECTURES` and
`OPJ_CUDA_PTX_ARCHITECTURE` together when using a different toolkit support range.
The default architecture set requires a toolkit that supports both Maxwell and
Blackwell; CUDA 12.9 was used to compile every listed target. Architecture
compilation is not a claim of measured performance on those physical GPUs.

## Selection

Set environment variables before decoding starts. CUDA device/worker selection
is fixed on the first initialization attempt; do not change process environment
concurrently with codec calls.

| Variable | Behaviour |
| --- | --- |
| `OPJ_DECODE_BACKEND=auto` | Prefer CUDA when available, otherwise OpenCL, then CPU. Default. |
| `OPJ_DECODE_BACKEND=cuda` | Attempt CUDA; unsupported or failed jobs fall back directly to CPU. |
| `OPJ_DECODE_BACKEND=opencl` | Use the existing OpenCL selector and CPU fallback. |
| `OPJ_DECODE_BACKEND=cpu` | Bypass both GPU decoders. |
| `OPJ_CUDA_DEVICE=auto` | Select the only eligible NVIDIA GPU. Ambiguous selection declines CUDA. |
| `OPJ_CUDA_DEVICE=GPU-...` | Exact CUDA UUID, or an exact PCI bus ID returned by `cuDeviceGetPCIBusId`. |
| `OPJ_CUDA_DEVICE=off` | Disable CUDA initialization (`0` is an alias). |
| `OPJ_CUDA_WORKERS=1..8` | Concurrent CUDA streams; default clamps SM-count / 20 to 1..4. |
| `OPJ_CUDA_PROFILE=1` | Diagnostic host timings with an extra stream synchronization; do not use for normal benchmarks. |

An explicit legacy `OPJ_OPENCL_DEVICE` override, including `off`, retains its
meaning when `OPJ_DECODE_BACKEND` is unset. Encoding continues to use its existing
OpenCL settings. A CUDA job does not retry through OpenCL after failure.

## Pipeline and resource limits

CPU code retains codestream parsing, packet/Tier-2 decoding, range validation,
resource admission, and the public API. CUDA performs Part-1 MQ/RAW entropy
decoding, ROI undo, coefficient placement/dequantization, inverse 5/3 or 9/7
wavelets, component transforms, DC shift and clamping.

Each serial entropy coder owns one CUDA block and at most 4 KiB of shared packed
flags. MQ transition data fits in one 32-bit load, and CLZ-based renormalization
advances several bits at once. Wavelet kernels use 64 threads and at most 16 KiB
shared memory per block. The implementation uses no tensor cores or newer
architecture-only instructions; it is designed within GTX 1650 resource limits.

Reusable device buffers share a **64 MiB aggregate allocation budget** with the
OpenCL backend. CUDA pinned staging has a separate **64 MiB aggregate budget**.
These figures cover backend buffers, not driver/context/module overhead or
OpenJPEG's normal CPU image allocations. Idle buffers are reclaimed within a
backend as needed. Admission never waits for a busy CUDA stream: excess work
returns to CPU, so concurrent throughput can include both processors.

Admitted tiles have 1..4 matching components, 1..16-bit samples, up to 4096 pixels
per dimension, and must fit the budgets. Part-1 coding styles, partial quality
layers, reduced resolutions, ROI coding and compatible truncated codestreams
are supported. HTJ2K, differing component geometry, arbitrary component
selection, partial-area decoding and other unsupported cases retain CPU decode.
The planner also caps code blocks, packet segments and entropy input sizes.

GPU output is copied to tile samples only after stream completion and successful
block statuses. Pending DMA retains its pinned storage on a fatal driver error;
future CUDA use is disabled. The backend never resets the device. It retains
the primary context and restores the calling thread's previous CUDA context.
Callbacks run after releasing the worker slot, including recursive decodes.
CUDA resources are retained until process exit; the Windows shared library is
pinned to avoid driver calls during DLL detach.

## Validation and measured performance

On the RTX 5070 Ti, the generated matrix matched CPU pixels byte-for-byte for
168 cases (404 component images). A 32-texture cache corpus at three resolutions
added 96 cases (336 component images). Fourteen further comparisons cover ROI,
truncated downloads, reduced resolutions, area and component selection.
Concurrent exact-pixel tests at 1/2/4/8 CUDA slots, recursive callbacks, automatic
selection, invalid selectors/configuration and over-budget fallback also pass.
All eight existing OpenCL-labelled decode/encode tests passed. CPU-only build
and installed-package consumer checks passed. Forced compute-50 PTX JIT on the
5070 Ti produced the same checksums as CPU.

Isolated measurements on 2026-09-30 used the RTX 5070 Ti, driver 617.14 and a
Ryzen 9 9950X3D, with the viewer closed. Values are median warm wall time per
image across three runs of the same 32-texture corpus, with one CPU decoder
thread per caller. Transfer, reconstruction, CPU readback and checksum costs
are included; initialization is reported separately by the benchmark.

| Concurrent callers | CPU (ms/image) | NVIDIA OpenCL | CUDA with bounded CPU fallback |
| ---: | ---: | ---: | ---: |
| 1 | 26.027 | 11.274 | 9.718 |
| 4 | 7.049 | 6.957 | 5.947 |
| 8 | 4.765 | 6.754 | 4.404 |

All single-caller CUDA tiles used the GPU. At four callers, 100/128 tiles per run
used CUDA; at eight, 75..78/128 did. Those concurrent rows are hybrid throughput,
not pure GPU throughput or individual-image latency. Correctness checksums
matched in every run. See `results-2026-09-30.json` for the summary and corpus
fingerprint. The corpus itself is local test data and is not distributed here.

```powershell
python experimental/cuda/benchmark.py --bin build/cuda/bin/RelWithDebInfo `
  --corpus path/to/j2k-corpus --out build/cuda/timings.json `
  --opencl-device "RTX 5070 Ti" --callers 1 4 8
```

NVIDIA Compute Sanitizer was attempted but could not initialize the Windows
WDDM debugger interface. **Memory-sanitizer validation remains outstanding**;
the run is not recorded as a pass, and no system debugger setting was changed.
No GTX 1650 hardware measurements or Linux runtime measurements are claimed.
Viewer integration, rendering contention and frame-time effects remain separate
qualification work. CUDA is included by default on supported build targets and selected automatically
on an eligible NVIDIA GPU. Unsupported targets default to CPU/OpenCL. Existing
CMake build directories retain cached options: change a cached
`OPJ_ENABLE_CUDA=OFF` to `ON` when reconfiguring. Build machines need the CUDA
toolkit even when no NVIDIA GPU is installed; set `OPJ_CUDA_ROOT`, `CUDA_PATH`,
or provide the compiler and headers through normal CMake search paths. A missing
toolkit is a configuration error, rather than silently producing a binary without
CUDA. CPU-only CI/fuzzing jobs must explicitly use `OPJ_ENABLE_CUDA=OFF`.
