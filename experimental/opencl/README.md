# OpenCL decoder development

This branch starts at OpenJPEG **v2.5.4**, commit
`6c4a29b00211eb0430fa0e5e890f1ce5c80f409f`. It implements an optional native
OpenCL decoding backend without changing the public OpenJPEG API.

**Status: functional experimental backend, not a performance win.** On the
current RX 9070 XT corpus, the optimized GPU path is about three times slower
than the single-thread CPU path. It remains disabled by default. Vulkanstorm's
package and shipping decoder have not been changed.

## Decoding path

The CPU parses the codestream and performs Tier-2 packet processing. After that,
`opencl_backend.c` validates and copies a bounded tile plan, then runs:

1. Tier-1 MQ/RAW entropy decoding, context resets, vertical causal contexts,
   segmentation symbols, ROI recovery and predictable-termination diagnostics.
2. Subband placement and dequantization.
3. Multilevel reversible 5/3 or irreversible 9/7 inverse wavelet reconstruction.
4. Reversible/irreversible component transform, level shift and output clamping.

Coefficients and reconstructed planes remain on the GPU between these stages.
The final planar integer samples return through the existing OpenJPEG API.
This is not direct GPU texture upload or asynchronous viewer integration.

A persistent context, in-order queue, program and kernels are reused. Job
buffers currently allocate per tile; pooling and cross-image batching are still
future work. The entropy kernel assigns one work-item per code block. Cached
neighbor flags, packed MQ contexts and 32-lane interleaved scratch reduce its
cost; the layout does not require vendor extensions or a particular wave size.

## Build and select a device

The normal build needs no OpenCL headers or link library. Explicitly enabling
`OPJ_ENABLE_OPENCL` requires Khronos headers; it dynamically loads the installed
ICD at runtime. On Windows the DLL is loaded from System32.

The development build used Khronos OpenCL-Headers revision
`e6060189f4ebe8b52d885c37af71b9a50c272154`, cloned into the ignored
`build/OpenCL-Headers` directory. Set the include path to your own checkout.

```powershell
python experimental/opencl/test_idwt53.py --list
cmake -S . -B build/gpu -G "Visual Studio 17 2022" -A x64 -DBUILD_CODEC=ON -DBUILD_TESTING=OFF -DBUILD_THIRDPARTY=ON -DBUILD_SHARED_LIBS=OFF -DOPJ_ENABLE_OPENCL=ON -DOPJ_OPENCL_INCLUDE_DIR=C:/Dev/openjpeg/build/OpenCL-Headers -DOPJ_BUILD_OPENCL_EXPERIMENTS=ON -DOPJ_OPENCL_TEST_DEVICE=gfx1201 -DOPJ_OPENCL_TEST_DRIVER=3679
cmake --build build/gpu --config RelWithDebInfo --parallel 8
ctest --test-dir build/gpu -C RelWithDebInfo --output-on-failure
$env:OPJ_OPENCL_DEVICE = 'gfx1201'
$env:OPJ_OPENCL_DRIVER = '3679'
build/gpu/bin/RelWithDebInfo/opj_decompress.exe -i input.j2k -o output.pgx
```

Device/driver selectors are case-sensitive substrings and must select exactly
one GPU. Two ICD versions expose gfx1201 on this machine, so the driver selector
is material. Selection is cached on first initialization; change it in a new
process. Unset `OPJ_OPENCL_DEVICE` for the CPU path. Set `OPJ_OPENCL_PROFILE`
before initialization for experimental per-stage event timings.

Hardware qualification uses the existing **AMD RX 9070 XT / driver 3679.0**.
NVIDIA, Intel and Linux execution remain unverified; no additional hardware
purchase is a prerequisite. The implementation uses standard OpenCL C 1.2.

## Eligibility and fallback

The current backend accepts whole-tile decoding with one to four components,
matching component geometry/resolutions/precision/signedness, precision up to
16 bits, standard component transforms, and traditional Part 1 code blocks.
Reduced resolutions and reduced quality layers are supported. Component
selection, partial-area decoding, heterogeneous components, custom transforms
and HTJ2K use CPU decoding. Tile dimensions are limited to 4096 and combined
per-job device buffers to **64 MiB**; host staging is additional.

A nonblocking process-wide lock admits one GPU job. Other decoder workers use
the CPU if it is busy. Unsupported input, unavailable/ambiguous devices,
allocation refusal and OpenCL errors fall back to CPU. Host tile output is
committed only after the entire GPU job and final readback succeed. This is
not yet complete malformed-input, concurrency or device-loss qualification.

## Validation

Tests compare decoded PGX component files byte for byte and require a GPU
completion record for **every** tile, preventing silent CPU fallback from
passing a GPU correctness test. The native suite exercises 8-bit grayscale,
two-channel, RGB and RGBA data, odd origins, singleton dimensions, tiled
images, both transforms, all style flags together, reduced resolution and
first-layer decoding. Separate cases exercise signed and unsigned 16-bit data. All **156 cases /
392 component images** passed on the qualified device.

`test_fallback.py` checks exact CPU output for an invalid device selector and
for a 4096-square image exceeding the device working budget. A separate clean
RelWithDebInfo CPU-only library builds without either the OpenCL selector or
reference-capture environment-variable string. The OpenCL-enabled
RelWithDebInfo shared DLL also builds successfully; 17 selected upstream
self-contained CPU regression tests pass (not the full external-data suite).

The Tier-1 reference corpus covers all 64 combinations of
BYPASS/RESET/TERMALL/VSC/PTERM/SEGSYM, multiple block/image sizes, two transforms,
ROI shifts, full/reduced resolutions and first-layer-only decoding:

- 1,152 synthetic codestreams, 29,184 code blocks.
- **18,224,256 exact CPU/GPU coefficient matches**, including 19,456 ROI blocks.
- All 8,448 enabled PTERM checks agree, including 24 synthesized-marker and
  nine remaining-byte diagnostics.

`OPJ_CAPTURE_T1_REFERENCES=ON` requires the development experiments option and
adds the CPU hook selected by `OPJ_T1_CAPTURE_FILE`. Capture version 2 records
coefficients after ROI recovery and before scaling, plus the CPU's PTERM check
decision and diagnostic. Records are serialized under the decoder mutex;
separate codecs/processes must use separate files. Normal builds have no hook.

```powershell
cmake -S . -B build/cpu-reference -G "Visual Studio 17 2022" -A x64 -DBUILD_TESTING=ON -DBUILD_CODEC=ON -DBUILD_SHARED_LIBS=OFF -DBUILD_THIRDPARTY=ON -DOPJ_BUILD_OPENCL_EXPERIMENTS=ON -DOPJ_CAPTURE_T1_REFERENCES=ON -DOPJ_OPENCL_TEST_DEVICE=gfx1201 -DOPJ_OPENCL_TEST_DRIVER=3679
cmake --build build/cpu-reference --config RelWithDebInfo --parallel 8
ctest --test-dir build/cpu-reference -C RelWithDebInfo --output-on-failure -R "^(opencl_.*|tte[0-5]|ttd[0-2]|rta[1-5]|testempty[0-2])$"
```

The standalone 5/3 tests cover 68 batches / 80,784 exact samples. Descriptor
validation tests reject inconsistent geometry, modes and unsupported styles.
PTERM diagnostics (512/1024) and segmentation-symbol diagnostics (256) remain
nonfatal to match upstream; bits 0-7 indicate fatal kernel errors.

With the owner's permission, `extract_cache_corpus.py` reconstructed **32 real
textures** from read-only copies of the viewer's indexed 600-byte prefixes and
body files. The source cache was not modified. The private corpus and manifest
remain under ignored `build/cache-corpus`, not in Git. All **96 full/reduced
resolution decodes / 336 component images** matched the CPU exactly with every
tile decoded on the GPU. These tests do not establish complete Part 1
conformance, independent-encoder interoperability, arbitrary truncated-payload
handling or viewer image quality across every texture.

## Performance evidence (2026-09-23)

`bench_decode` is a development-only public-API benchmark. It decodes the 32
cached textures in one process, excludes its first corpus pass from the warm
mean, uses one CPU decoding thread and includes identical output checksumming
in both modes. Files are warm; context/program initialization is excluded from
the GPU warm mean. Three measured rounds give 96 warm decodes per mode.

| Path | Mean time per texture |
| --- | ---: |
| CPU | 25.514 ms |
| Initial complete GPU backend | 142.002 ms |
| Cached neighbor contexts | 89.653 ms |
| Packed MQ contexts | 81.154 ms |
| Interleaved flag scratch | **75.311 ms** |

CPU and GPU output checksums agree. The optimized GPU path takes about 47%
less time than the initial GPU version, but remains **2.95 times slower than
CPU**. Profiling after the neighbor-context optimization attributed about
74.8 ms to Tier-1 and 7.2 ms to DWT per warm texture; placement and finishing
were much smaller. This identifies Tier-1 as the main optimization target,
without proving a specific hardware cause. These are codec timings, not viewer
FPS or texture-arrival measurements.

```powershell
$inputs = Get-ChildItem build/cache-corpus/*.j2k | ForEach-Object FullName
Remove-Item Env:OPJ_OPENCL_DEVICE -ErrorAction SilentlyContinue
build/gpu/bin/RelWithDebInfo/bench_decode.exe 3 @inputs
$env:OPJ_OPENCL_DEVICE = 'gfx1201'
$env:OPJ_OPENCL_DRIVER = '3679'
build/gpu/bin/RelWithDebInfo/bench_decode.exe 3 @inputs
```

The next performance work is entropy execution/scheduling and larger batches
across independent textures, followed by buffer reuse and more parallel wavelet
reconstruction. Further qualification must cover malformed/truncated streams,
contention and device failure before package/viewer integration. Do not enable
this backend by default based on functionality alone.

## Provenance

Kernel behavior follows OpenJPEG v2.5.4 `t1.c`, `mqc.c`, `dwt.c`, `mct.c` and
`tcd.c`. Upstream BSD notices are retained in adapted kernel sources and the
embedded bundle. No Roger or xxjjss decoder code is copied.
