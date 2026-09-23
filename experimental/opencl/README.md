# OpenCL decoder development

This branch starts from upstream **v2.5.4**, commit
`6c4a29b00211eb0430fa0e5e890f1ce5c80f409f`, matching Vulkanstorm's current
OpenJPEG source version. The goal is GPU Tier-1 entropy decoding and full
reconstruction inside OpenJPEG, retaining CPU parsing and compatibility.

## Implemented first slice

- OpenCL C 1.2 inverse reversible 5/3 lifting kernels, with separate update and
  prediction dispatches. Both origin parities and single-sample cases are handled.
- Strided rows/columns and multiple independent lines per dispatch. Coefficients
  currently arrive interleaved, not in OpenJPEG's packed subband layout.
- A standard-library Python runner using the installed OpenCL ICD through ctypes.
  It compiles once and reuses the context, queue and kernels across cases. It
  requires explicit device selection and rejects ambiguous selections.
- Exact round-trip tests using independent CPU forward lifting. Signals include
  impulses, ramps, constants, alternating extremes and deterministic random data.
  Oversized dispatches and sentinel padding check bounds handling.

The runner is development infrastructure, not the production C backend. It uses
an in-order queue to establish the dependency between the two kernels. There is
no kernel timing or speedup claim. The existing library decoding path is unchanged.

## Reproduce on this machine

```powershell
python experimental/opencl/test_idwt53.py --list
python experimental/opencl/test_idwt53.py --device gfx1201 --driver 3679

cmake -S . -B build/cpu-reference -G "Visual Studio 17 2022" -A x64 -DBUILD_TESTING=ON -DBUILD_CODEC=ON -DBUILD_SHARED_LIBS=OFF -DBUILD_THIRDPARTY=ON -DOPJ_BUILD_OPENCL_EXPERIMENTS=ON -DOPJ_OPENCL_TEST_DEVICE=gfx1201 -DOPJ_OPENCL_TEST_DRIVER=3679
cmake --build build/cpu-reference --config RelWithDebInfo --parallel 8
ctest --test-dir build/cpu-reference -C RelWithDebInfo --output-on-failure -R "^(opencl_idwt53|tte[0-5]|ttd[0-2]|rta[1-5]|testempty[0-2])$"
```

Device and driver strings are local selections, not requirements for the kernels.
Two driver versions expose gfx1201 on this machine; do not silently choose one.
Hardware validation currently uses the existing **RX 9070 XT only**. No NVIDIA or
Intel purchase/testing prerequisite is part of this project. Other devices remain
unverified; the code does not use a vendor whitelist or vendor extensions.

OpenCL experiments are off by default and nothing here is installed with openjp2.
The host checks geometry, non-overlapping line layouts, buffer bounds and a
restricted coefficient range before dispatch. This is not yet a hardened interface
for arbitrary JPEG2000 bitstreams.

## Next implementation work

Verified initial checkpoint (2026-09-23): the RelWithDebInfo reference build
completed, and all 18 selected CTest cases passed (17 upstream self-contained
tests plus the OpenCL test). The GPU test passed 68 batches / 80,784 exact samples
on gfx1201, driver 3679.0. This is a selected test set, not the full external
OpenJPEG regression corpus. Test logs are in the ignored build directory.

1. Extract a validated code-block decode plan after Tier-2, including segment/pass
   metadata, coding style, bitplanes, subband geometry and request resolution.
2. Capture reference Tier-1 coefficients from OpenJPEG and port MQ/raw entropy
   decoding across independent blocks. Require exact intermediate parity.
3. Add subband placement, multilevel 2-D reconstruction, 9/7 and final component
   conversion. Compare against the actual OpenJPEG decoder, including reduced
   resolutions, boundaries and truncated input.
4. Implement the production C runtime with bounded pools, event ownership and
   CPU retry, then integrate the complete path in the third-party package.
5. Add asynchronous viewer integration and evaluate texture readiness, frame
   pacing and working-set use on the RX 9070 XT.

The current 1-D lifting tests do not prove full JPEG2000 compatibility, multilevel
transform correctness, entropy throughput or an end-to-end performance gain.

## Provenance

Kernel lifting semantics follow the reversible JPEG2000 transform, with boundary
behavior checked against OpenJPEG v2.5.4 `src/lib/openjp2/dwt.c`. New files use the
repository's BSD-2-Clause licensing. No Roger or xxjjss decoder code is copied.
