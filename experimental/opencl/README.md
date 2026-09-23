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

1. Broaden ROI coverage beyond component-wide upshift and exercise additional
   precision, origin and malformed-input cases against the CPU reference.
2. Add independently produced/real texture inputs, progressive quality layers and
   failure cases, retaining exact intermediate parity.
3. Add subband placement, multilevel 2-D reconstruction, 9/7 and final component
   conversion. Compare against the actual OpenJPEG decoder, including reduced
   resolutions, boundaries and truncated input.
4. Implement the production C runtime with bounded pools, event ownership and
   CPU retry, then integrate the complete path in the third-party package.
5. Add asynchronous viewer integration and evaluate texture readiness, frame
   pacing and working-set use on the RX 9070 XT.

The current tests do not prove full JPEG2000 compatibility, multilevel transform
correctness, entropy throughput or an end-to-end performance gain.

## Tier-1 MQ/RAW checkpoint

`t1_decode.cl` now decodes compressed code blocks on the GPU, with one work-item
per block. It implements the MQ arithmetic decoder, significance/sign decoding,
magnitude refinement, cleanup aggregation, context reset, terminated segments,
RAW bypass with byte stuffing, vertical causal contexts, segmentation symbols,
ROI bitplane decoding/shift reversal and predictable-termination diagnostics.
MQ states are generated from the pinned upstream table as
integer indices, retaining upstream notices. Tests verify the generated file is
current. HTJ2K remains outside this kernel's accepted domain. This does not
establish complete Part 1 conformance or production-ready error handling.

`OPJ_CAPTURE_T1_REFERENCES=ON` adds an opt-in development hook to the CPU decoder.
`OPJ_T1_CAPTURE_FILE` selects a per-process JSONL file. Each record contains
geometry, subband/component/resolution metadata, style, quantization parameters,
compressed segments and **CPU coefficients after ROI undo, before scaling**.
Capture version 2 explicitly labels this boundary `post_roi`; older captures must
be regenerated. It also records whether OpenJPEG performed its PTERM check and
the resulting diagnostic category. Records
are serialized under the existing decoder job mutex. Separate processes/codecs
must use separate output files. This option has no effect unless OpenCL experiments
are enabled; normal builds contain no capture hook.

`make_t1_corpus.py` creates deterministic grayscale codestreams using constant,
ramp and random signals, three image sizes (including odd sizes), all 64
combinations of BYPASS/RESET/TERMALL/VSC/PTERM/SEGSYM, both wavelet transforms and
full/reduced resolution decoding. Each codestream has two quality layers and is
also decoded with only the first quality layer. Code blocks include partial
blocks, 16x16 and up to 64x64.
Constant/ramp patterns exercise component-wide ROI shifts 3/5, while random
patterns retain the non-ROI path. OpenJPEG's command-line ROI option upshifts an
entire component; spatial ROI masks from other encoders still need testing.
This is a synthetic corpus generated by OpenJPEG, not yet a captured viewer corpus
or an independent encoder interoperability suite.

`test_t1.py` validates and flattens capture records into experimental descriptors,
then compares every GPU coefficient directly to the CPU reference in bounded
128-block batches. It checks GPU status, initializes output/scratch to sentinels,
and deliberately over-dispatches to exercise the block-count guard. Synthetic
malformed descriptors are rejected before submission by `test_t1_validation.py`.
The descriptor format is not yet a public ABI or a hardened untrusted-input API.

Verified on RX 9070 XT / driver 3679.0: **29,184 blocks and 18,224,256 coefficients
match the CPU exactly**, including full 64x64 blocks and 19,456 ROI blocks.
All **8,448 enabled PTERM checks** agree with the CPU, including 24 synthesized
marker diagnostics and 9 remaining-byte diagnostics. The corpus contains 1,152
codestreams, each decoded at full
resolution, reduced resolution, and first-layer-only quality. Host validation
rejects a descriptor that mixes MQ and RAW passes within one segment. A separate
RelWithDebInfo build with experiments disabled previously succeeded and contains
no capture environment-variable string. No throughput or end-to-end speedup has
been measured.

```powershell
cmake -S . -B build/cpu-reference -DOPJ_CAPTURE_T1_REFERENCES=ON
cmake --build build/cpu-reference --config RelWithDebInfo --parallel 8
ctest --test-dir build/cpu-reference -C RelWithDebInfo --output-on-failure -R "^opencl_"
```

CTest generates its corpus as a fixture before running Tier-1 parity. Captures
and logs remain under the ignored build directory. A capture failure fails the
experimental CPU decode. The GPU kernel consumes an unexpected segmentation
symbol and records diagnostic bit 256 without failing, matching upstream's
non-rejection behavior (upstream's warning code is commented out). PTERM
diagnostics are also nonfatal: 512 means remaining bytes, 1024 means more than
two synthesized markers, with the same precedence and check gating as the CPU.
Partial-quality decoding follows the captured CPU check decision. Status bits
0-7 remain fatal errors. This corrects the earlier
prototype's overly strict rejection. Arbitrarily truncated network payloads and
malformed codestream interoperability still require separate qualification.

Next: extend entropy mode coverage, add independently produced/real texture
inputs, then connect coefficient placement to multilevel reconstruction. The
OpenJPEG decoding API and Vulkanstorm package still execute their CPU path.

## Provenance

Kernel lifting semantics follow the reversible JPEG2000 transform, with boundary
behavior checked against OpenJPEG v2.5.4 `src/lib/openjp2/dwt.c`. New files use the
repository's BSD-2-Clause licensing. No Roger or xxjjss decoder code is copied.
