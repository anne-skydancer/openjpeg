# OpenCL decoder development

This branch starts at OpenJPEG **v2.5.4**, commit
`6c4a29b00211eb0430fa0e5e890f1ce5c80f409f`. It implements an optional native
OpenCL decoding backend without changing the public OpenJPEG API.

**Status: experimental concurrent GPU backend with optimized Tier-1 decoding.**
On the qualified RX 9070 XT, the latest paired GPU comparison measured about
**138 textures/s with four image workers**, versus **114 before this change**;
one worker improved from **52 to 61 textures/s**. These are warm codec throughput
measurements, not viewer FPS. The backend remains disabled by default;
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

One persistent context and compiled program are shared. Each GPU worker owns
its own in-order queue, kernel objects/arguments, events and reusable buffers.
Buffer capacity across **all workers** stays within the 64 MiB aggregate cap;
independent high-water marks cannot accumulate beyond it. Failed jobs drain the
queue and discard their pooled buffers before freeing host staging.

Tier-1 assigns one code block to each one-item work-group. Four vertically
adjacent coefficients share one 32-bit significance/sign/refinement/visit word,
following OpenJPEG's stripe layout. Flags reside in dynamically sized local
memory: `width * ceil(height / 4) * 4` bytes, including partial stripes. A 64x64
block uses 4 KiB rather than the previous 8 KiB. Independent streams therefore
do not share divergent instruction execution within a wave. Context lookup
tables come directly from upstream `t1_luts.h`; constant-pass calls and a
64x64/style-zero fast path remove invariant choices from inner loops. Other
block geometries and all 64 coding styles retain GPU decoding.

Packed MQ contexts remain in private storage. Coefficients and compressed bytes
stay in global memory; experiments moving these into local memory were slower.

Placement and wavelet reconstruction use 64-item work-groups. Each wavelet group
cooperates on one row or column, with barriers between dependent lifting phases
and dynamically sized local scratch. This removes the global flags and wavelet
scratch buffers entirely. Kernel status and pixels are checked at the final
readback, avoiding the previous host wait between entropy and reconstruction.

Concurrent callers, each using its own OpenJPEG codec, lease independent GPU
slots. The default is **four slots**, configurable from one to eight through
`OPJ_OPENCL_WORKERS` before first initialization. This does not spawn texture
threads inside OpenJPEG: caller workers perform parsing and feed the GPU slots.
The synchronous public API is unchanged; multiple calls can now remain in flight.
Cross-image fusion into one kernel dispatch remains future work.

## Build and select a device

The normal build needs no OpenCL headers or link library. Explicitly enabling
`OPJ_ENABLE_OPENCL` requires Khronos headers; it dynamically loads the installed
ICD at runtime. On Windows the DLL is loaded from System32.

On Windows DLL builds, successful opt-in initialization pins the module and
keeps the shared OpenCL runtime for the process lifetime. The OS reclaims it at
process exit; explicit DLL unload/reload is not a resource-reset mechanism.
Calling the GPU driver from a DLL `atexit` handler hung during shutdown in our
runtime test. DLL CRT teardown executes those handlers under loader-lock
restrictions, so this path deliberately avoids driver teardown there. Static
EXE builds retain their ordinary exit cleanup. See Microsoft's
[DLL CRT lifecycle](https://learn.microsoft.com/en-us/cpp/build/run-time-library-behavior)
and [DllMain restrictions](https://learn.microsoft.com/en-us/windows/win32/dlls/dllmain).
This behavior applies only after explicitly enabling the experimental backend.

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
`OPJ_OPENCL_WORKERS` is also cached on first initialization (default 4, range
1-8); invalid values leave decoding on the CPU.

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
pooled device buffers across all jobs to **64 MiB**; host staging, local memory
and driver allocations are additional.

A short admission lock protects initialization, slot leasing and the shared
allocation budget. It is released before uploads or decoding. A condition
variable parks eligible callers when slots or budget are busy; this contention
does not silently turn GPU benchmarks into mixed CPU/GPU runs. Idle buffers can
be reclaimed to admit larger jobs; active buffers are never evicted.

Unsupported input, unavailable/ambiguous devices, allocation refusal and OpenCL
errors still fall back to CPU. Host output is committed only after successful
GPU decoding/readback. Completion and diagnostic callbacks run after returning
the slot, allowing a callback to decode another image without self-deadlocking.
This is not yet complete malformed-input or device-loss qualification.

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

`test_workers.py` verifies every decoded pixel against CPU reference files with
eight callers sharing 1/2/4/8 GPU slots. Heterogeneous small images and three
1024-square RGB images exercise independent kernel arguments and admission
pressure: the three large jobs cannot simultaneously fit within the budget.
The test requires every eligible tile to run on the GPU and checks observed
active slots and pooled byte counts. `test_reentrant.c` performs a nested decode
from a completion callback with only one slot, checking exact output and return.
The native six-test CTest suite passes, including the 156-case image matrix.

The 32 private real cache textures also passed byte-for-byte concurrent checks
with eight callers at all four slot counts, covering 384 GPU tile decodes.
An additional 96 cached-texture decodes passed through the shared DLL with
concurrent profiling enabled, followed by normal process exit. The shared DLL
also passed the one-slot nested-callback check. No cached images or reference
pixels are included in Git.

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

## Sequential performance evidence (2026-09-23, before concurrent slots)

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

## Concurrent throughput (2026-09-23)

`bench_workers` creates independent public-API codecs on caller threads. Its
arguments are image-worker count, CPU threads **per image**, warm corpus rounds,
then input files. Each round schedules every input exactly once. Worker startup,
join, file reads, parsing, GPU work/readback and output checksumming are included.
The first round is excluded from warm throughput. Reference file I/O is disabled
for timing runs. The reported `warm_ms_per_image` is wall time divided by image
count, **not individual-image latency**.

The comparison below uses the same 32 cached textures, three separate processes
per setting, five warm rounds per process (480 measured images per setting).
Run order was reversed in the middle trial. Each decoder has one CPU thread per
image; GPU slot count matches the image-worker count. Checksums agree across
all modes and trials. GPU completion count equals total tile count in every run.

| Concurrent image workers | CPU textures/s | GPU textures/s | GPU run range |
| --- | ---: | ---: | ---: |
| 1 | 36.34 | 49.11 | 47.86-49.85 |
| 2 | 71.35 | 87.02 | 86.42-87.78 |
| 4 | 135.93 | **110.76** | 108.16-112.40 |
| 8 | 200.97 | 116.84 | 111.53-121.00 |

Four GPU workers deliver **2.26x** the single-worker throughput. Eight add only
about **5.5%** beyond four, so the default remains four. The 64 MiB buffer cap
limited peak admitted jobs to five or six in the eight-worker runs; peak pooled
capacity stayed below 66,511,000 bytes (about 63.43 MiB). Kernel/program/driver
memory and CPU staging are not part of this buffer accounting.

The GPU wins at one/two image workers in this workload, but CPU image workers
scale better at four/eight. Concurrent GPU submission alone has not removed the
remaining entropy-decoding/device bottleneck. These results do not establish a
viewer FPS gain or behavior while the GPU is simultaneously rendering a busy sim.

```powershell
$inputs = Get-ChildItem build/cache-corpus/*.j2k | ForEach-Object FullName
Remove-Item Env:OPJ_OPENCL_DEVICE -ErrorAction SilentlyContinue
build/gpu/bin/RelWithDebInfo/bench_workers.exe 4 1 5 @inputs
$env:OPJ_OPENCL_DEVICE = 'gfx1201'
$env:OPJ_OPENCL_DRIVER = '3679'
$env:OPJ_OPENCL_WORKERS = '4'
build/gpu/bin/RelWithDebInfo/bench_workers.exe 4 1 5 @inputs
```

For exact concurrency checks, run once with `--write-reference DIRECTORY` on the
CPU, then `--verify-reference DIRECTORY` on the GPU with the identical ordered
input list. The directory must exist. Both options compare/write all component
metadata and sample bytes; use neither option for throughput measurements.

Remaining work includes improving entropy execution, evaluating cross-image
batching, and qualification under rendering load, malformed/truncated input and
device failure. Keep the backend opt-in until workload-level results justify it.

## Within-block entropy experiments (2026-09-23)

Implemented and correctness-tested all four proposed approaches, in sequence:

1. **Packed four-row stripe state:** exact, but slower in isolation (27.153 ms
   per texture in the initial one-worker screening versus 20.253 before).
2. **Context lookup tables and specialization:** retained with packed state.
   Tables recovered much of the initial loss; separate pass loops, packed
   cleanup aggregation and a common 64x64/style-zero fast path produced the gain.
3. **Cooperative refinement:** tested eight and 32 lanes. Each lane prepares
   contexts and updates its own four-row column; lane zero advances MQ in exact
   scan order. Three barriers per batch, none per decision. The 32-lane version
   was the better candidate, but repeated runs did not favor it over scalar
   specialization at one/four workers and were effectively tied at eight.
4. **Two-decision speculative MQ lookahead:** tested both arithmetic exchange
   paths and the subsequent refinement decision, committing only the actual
   path. It remained exact but cost 20.693 ms per texture and 99.68 textures/s
   at four workers in the initial screening. It is not active.

The last two implementations are preserved as explicitly applied research
[patches](variants/README.md), outside the compiled kernel bundle. This records
negative results without paying their runtime cost or adding runtime selectors.

The repeated comparison used the same 32 private cached textures, three trials,
five warm corpus rounds per process: **480 measured images per table cell**, plus
warmups. Each process used one CPU thread per image; GPU slot count matched the
image-worker count. The middle trial reversed execution order. Every GPU tile
count matched the total and all output checksums agreed. Baseline is `e8765369`.

| Image workers | Previous GPU textures/s | Retained scalar textures/s | 32-lane refinement textures/s |
| --- | ---: | ---: | ---: |
| 1 | 51.86 (51.65-51.99) | **61.43 (61.06-61.85)** | 59.01 (56.08-60.89) |
| 4 | 113.79 (107.61-118.67) | **138.01 (133.62-141.54)** | 134.79 (127.34-139.78) |
| 8 | 123.48 (118.84-129.31) | **146.65 (142.38-153.26)** | 146.46 (140.43-150.40) |

Parentheses show the range of process means. The retained implementation raises
throughput by about **18.4%, 21.3% and 18.8%**, respectively. Timing varies between
sessions; use these paired results rather than dividing by older table entries.
This experiment does not establish a gain while the GPU is rendering the viewer.
The default remains four slots and the aggregate device buffer cap remains 64 MiB.

Each candidate passed all 29,184 captured blocks / 18,224,256 coefficients,
including all 64 styles, 19,456 ROI blocks and 8,448 PTERM checks. Final validation:

- Six native CTest tests passed; the expanded complete-image matrix covers
  **168 cases / 404 component images**, including long one/three-row or column
  images and partial stripes with large nominal code blocks.
- **96 real-cache full/reduced decodes / 336 component images** were byte-exact.
- Eight callers at 1/2/4/8 slots produced **384 exact real-cache GPU tile decodes**.
- The rebuilt RelWithDebInfo shared DLL produced another **96 exact cached GPU
  tile decodes** with profiling enabled, exited normally, and passed nested
  decoding from a completion callback with one slot.

Both static and shared RelWithDebInfo builds were rebuilt. Private texture
payloads, reference pixels and benchmark executables remain in ignored `build/`.

## Provenance

Kernel behavior follows OpenJPEG v2.5.4 `t1.c`, `mqc.c`, `dwt.c`, `mct.c` and
`tcd.c`. Upstream BSD notices are retained in adapted kernel sources and the
embedded bundle. No Roger or xxjjss decoder code is copied.

## GPU encoding (2026-09-23)

The optional encoder accelerates level shifting, reversible/irreversible color
transforms, forward 5/3 or 9/7 wavelets, and style-zero Tier-1 MQ entropy coding.
Quantization, distortion weighting, rate allocation and Tier-2 packet assembly
remain on the CPU. Both lossy and lossless encoding use the existing OpenJPEG
API. Unsupported configurations or failed GPU stages use the original CPU stage;
results are committed only after the entire GPU stage succeeds.

Encoding is disabled by default and enabled independently of decoding:

```powershell
$env:OPJ_OPENCL_ENCODE_DEVICE = 'gfx1201'
$env:OPJ_OPENCL_DRIVER = '3679'
$env:OPJ_OPENCL_WORKERS = '4'
```

If decoding is also enabled, use the same literal device selector for
`OPJ_OPENCL_DEVICE`. Both paths share the worker pool and its aggregate 64 MiB
buffer limit. Driver allocations, kernels and CPU staging are additional.
The transform path supports matching component geometry, 1-4 components,
1-16-bit samples and dimensions up to 4096, subject to the buffer limit.
Tier-1 currently requires coding style zero without ROI; other styles retain
CPU entropy coding even when GPU transforms are eligible. Intermediate CPU
quantization currently requires a device readback and another upload.

`bench_encode` reuses the concurrent benchmark harness. It decodes input J2K
files once before timing, then measures image cloning, encoder setup, encoding,
memory-stream output and checksum generation. Use `--lossy 8` before positional
arguments for an 8:1 target; otherwise encoding is lossless. Its
`--write-reference DIRECTORY` and `--verify-reference DIRECTORY` options compare
complete encoded codestream bytes. Reference directories must already exist.

Repeated RelWithDebInfo results on RX 9070 XT / gfx1201, driver 3679, used 32
private cached textures, three trials and five warm rounds per trial: 480
measured encodes per configuration, excluding warmup. The middle trial reversed
execution order. All output checksums agreed, and both GPU stage counts matched
the encoded tile count. Values are mean textures/second across trials.

| Mode | Image workers | CPU threads/image | Lossless | Lossy 8:1 |
| --- | ---: | ---: | ---: | ---: |
| CPU | 1 | 1 | 23.59 | 28.19 |
| GPU | 1 | 1 | 33.80 | 30.45 |
| CPU | 1 | 4 | 72.20 | 76.47 |
| CPU | 4 | 1 | 81.79 | 105.94 |
| GPU | 4 | 1 | 74.71 | 76.90 |

GPU throughput improves about 43% losslessly and 8% lossily over one CPU thread,
but multithreaded CPU encoding remains faster. These are current desktop
measurements, not a controlled viewer-rendering-load qualification or a viewer
FPS claim. Keep GPU encoding opt-in while reducing transfers and entropy cost.

The encoder regression matrix requires byte-identical codestreams for 123 GPU
cases and two fallback controls, including odd/tiled origins, short dimensions,
8/16-bit signed/unsigned samples, 1/3/4 components, multiple rate layers and
constant images. Concurrent exact-output tests exercise eight callers with
1/2/4/8 GPU slots and the shared buffer cap. Hardware qualification is currently
limited to this AMD device/driver.

The separate optimized shared Release build (`build/gpu-encode/bin/Release`)
passed all eight native CTest tests, the 29,184-block decoder reference corpus,
and 128 exact real-cache GPU encodes (lossless/lossy, eight callers, four slots).
The CPU-only Release library also builds. The existing `build/gpu-release`
decoder DLL was preserved unchanged. Release uses /O2, /Ob3 and link-time
optimization with precise floating-point semantics; the performance table above
remains the measured RelWithDebInfo result rather than an inferred Release gain.
