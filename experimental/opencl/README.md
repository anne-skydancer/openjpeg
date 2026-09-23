# OpenCL decoder development

This branch starts at OpenJPEG **v2.5.4**, commit
`6c4a29b00211eb0430fa0e5e890f1ce5c80f409f`. It implements an optional native
OpenCL decoding backend without changing the public OpenJPEG API.

**Status: experimental backend with a measured gain over one CPU thread.**
The current RX 9070 XT corpus averages 20.124 ms per texture on the GPU versus
27.541 ms with one CPU decoding thread. CPU decoding with two or more threads
is still faster on this corpus. The backend remains disabled by default;
Vulkanstorm's package and shipping decoder have not been changed.

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

A persistent context, in-order queue, program, kernels and device buffers are
reused. Buffer capacity across the pool stays within the 64 MiB aggregate cap;
independent high-water marks cannot accumulate beyond it. Failed jobs drain the
queue and discard their pooled buffers before freeing host staging.

Tier-1 assigns one code block to each one-item work-group. Its significance and
cardinal-sign flags fit in 16 bits per coefficient and reside in dynamically
sized local memory (at most 8 KiB per group). Independent streams therefore do
not share divergent instruction execution within a wave. Packed MQ contexts
remain in private storage. Coefficients and compressed bytes stay in global
memory; experiments moving these into local memory were slower.

Placement and wavelet reconstruction use 64-item work-groups. Each wavelet group
cooperates on one row or column, with barriers between dependent lifting phases
and dynamically sized local scratch. This removes the global flags and wavelet
scratch buffers entirely. Kernel status and pixels are checked at the final
readback, avoiding the previous host wait between entropy and reconstruction.

The host backend is thread-safe, but admits one GPU tile job at a time; concurrent
callers use CPU fallback. It does not yet batch images or keep multiple GPU jobs
in flight. That requires separate job arguments/buffers and a shared total memory
budget, rather than allowing each worker its own 64 MiB pool.

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
for a 4096-square image exceeding the device working budget.
`test_reuse.py` also exercises a small/large/CPU-fallback/small sequence four
times in one process, verifying matching output checksums and 12 GPU tile jobs. A separate clean
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
mean, and includes identical output checksumming in both modes. Files are warm;
context/program initialization is excluded from the GPU warm mean. This is
codec latency, not viewer FPS, total texture-arrival time or multi-image throughput.

Historical measurements on the same corpus:

| GPU implementation | Mean time per texture |
| --- | ---: |
| Initial complete backend | 142.002 ms |
| Previous committed optimized backend | 75.311 ms |
| Previous backend remeasured before this pass | 75.410 ms |
| Local Tier-1 flags | 29.421 ms |
| Parallel wavelet reconstruction | 23.426 ms |
| Compact 16-bit flags | 21.934 ms |
| Dynamic local scratch; removed global scratch | 20.433 ms |

Final comparison uses three separate runs per mode, five warm corpus rounds
per run: **480 measured decodes per mode**, plus warm-ups. Execution order was
CPU/GPU, GPU/CPU, CPU/GPU. Every GPU decode used the backend and output checksums
matched the CPU across all runs.

| Decoder | Mean | Range of run means |
| --- | ---: | ---: |
| CPU, one thread | 27.541 ms | 27.170-27.872 ms |
| Optimized GPU | **20.124 ms** | **20.092-20.168 ms** |

This is about **73% less GPU decode time / 3.75x throughput** than the remeasured
previous commit, and **27% less time / 1.37x throughput** than one CPU thread.
The throughput ratios describe this sequential warm benchmark only.

A separate five-round CPU thread-count comparison, with matching checksums:

| CPU decoder threads | Mean per texture |
| --- | ---: |
| 1 | 27.379 ms |
| 2 | 16.816 ms |
| 4 | 12.415 ms |
| 8 | 9.703 ms |

**The GPU backend does not beat the multithreaded CPU decoder here.** These runs
process images sequentially with multiple CPU workers inside each decode; they
do not measure several viewer texture workers decoding independent images.

Optional profiling on the final kernels measured average device execution of
9.669 ms for Tier-1, 0.020 ms for placement, 0.215 ms for DWT and 0.014 ms for
finishing. Kernel time excludes CPU parsing, planning, API submission, transfers,
checksumming and other host work, so it must not be substituted for end-to-end
latency. Tier-1 remains the main device execution cost.

Rejected experiments included local coefficient storage, two entropy streams
per group, bulk MQ renormalization, explicit neighbor-update unrolling, indexed
MQ context arrays, compressed-input local caching, and combined component DWT
dispatch. They did not improve the measured result enough to keep.

```powershell
$inputs = Get-ChildItem build/cache-corpus/*.j2k | ForEach-Object FullName
Remove-Item Env:OPJ_OPENCL_DEVICE -ErrorAction SilentlyContinue
build/gpu/bin/RelWithDebInfo/bench_decode.exe --threads 1 5 @inputs
build/gpu/bin/RelWithDebInfo/bench_decode.exe --threads 4 5 @inputs
$env:OPJ_OPENCL_DEVICE = 'gfx1201'
$env:OPJ_OPENCL_DRIVER = '3679'
build/gpu/bin/RelWithDebInfo/bench_decode.exe 5 @inputs
```

The remaining opportunities are improving entropy execution and overlapping
independent images with CPU parsing and transfers. Cross-image batching needs
an explicit scheduling design; adding CPU decoder threads alone does not make
the current single-job GPU queue concurrent. Further qualification must cover
malformed/truncated streams, contention and device failure before package/viewer
integration. Keep the backend opt-in until workload-level results justify it.

## Provenance

Kernel behavior follows OpenJPEG v2.5.4 `t1.c`, `mqc.c`, `dwt.c`, `mct.c` and
`tcd.c`. Upstream BSD notices are retained in adapted kernel sources and the
embedded bundle. No Roger or xxjjss decoder code is copied.
