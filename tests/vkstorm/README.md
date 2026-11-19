# Gate harness of the Vulkanstorm decoder fixes

The tools here check the fixes this fork carries for decoding cut codestreams in non-strict mode
and for the decoder's resource limits (`opj_decoder_set_limits`, `OPJ_DECODER_LIMITS_SL_TEXTURE`).
They drive the library the way the Vulkanstorm viewer does (memory stream with read, skip and seek,
strict mode off, the reduce factor set before the header) and compare builds of different commits.
They run on Windows with Visual Studio 2026 (MSVC and clang-cl), Ninja, CMake, Git Bash and Python 3.
Nothing here is built by the project's CMake; the portable tests are `tests/test_partial_decode_limits.c`
(CTest cases `tpdl_*`).

The tools copied from the Vulkanstorm survey and stage 1a work keep their `GPL-3.0-or-later` notice;
the files added for this folder are `BSD-2-Clause`, like the rest of the fork's own files.

## Builds

| Script | What it builds |
|---|---|
| `build_lib.cmd rel <src> <build>` | MSVC Release static library and codec tools with the package's CPU flags |
| `build_lib.cmd dll <src> <build>` | the same as a DLL |
| `build_lib.cmd pkgdll <src> <build>` | the DLL with link-time code generation, as the Windows package builds it |
| `build_lib.cmd san <src> <build>` | clang-cl ASan + UBSan, NDEBUG as shipped |
| `build_lib.cmd sandbg <src> <build>` | the same with assertions on, which check that `opj_tcd_decode_need` covers what `opj_tcd_init_tile` builds |
| `build_tools.cmd probe rel\|san <src> <build> <exe> [/DPROBE_LIMITS]` | `probe.c` against a `rel` or `san`/`sandbg` build |
| `build_tools.cmd fuzz <src> <build> <exe>` | the libFuzzer target `fuzz_partial.c` with its own library build |

`<src>` is a source tree, best a `git archive` of the commit under test. The sanitizer and fuzzer builds
need `OPJ_LLVM_RT`, a folder without spaces that holds the clang runtime libraries (copy them from
`VC\Tools\Llvm\x64\lib\clang\<version>\lib\windows`); add it to `PATH` when running those builds.
`VCVARS` overrides the location of `vcvars64.bat`.

## Streams

| Script | Writes |
|---|---|
| `make_synth.py <dir>`, then `make_streams.sh <opj_compress> <dir>` | the synthetic test images and their codestreams (LRCP, RLCP, SOP, EPH, BYPASS, TERMALL, all mode switches; 16 to 256 px) |
| `make_loops.py <out> <W> <layers> <Scod> [pp]` | a one-packet codestream with many layers (the packet-loop rows, CVE-2023-39327) |
| `make_psot0.py <in> <out>` | a copy whose last tile-part has Psot = 0 |
| `make_psot0_reorder.py <in> <out>` | a copy with tile-parts reordered, TNsot = 0 and Psot = 0 in the last |
| `make_pclr_jp2.py <out> <W> <npc>` | a JP2 whose palette expands one component to `npc` |
| `make_fuzz_seeds.py <dir> <max bytes> <codestreams...>` | seeds for `fuzz_partial` (3 control bytes, then the codestream) |
| `scan_markers.py <files...>` | markers, progression, tiles, layers and Psot = 0 tile-parts of codestreams |

The ISO/IEC 15444-4 conformance streams come from `uclouvain/openjpeg-data` (`input/conformance`).

## Runs

- `probe <mode> ...` lists its modes in `probe.c`: conformance, cut sweeps (`cut`, `cuthash`), single
  cuts, whole-file hashes, peak memory of one decode, the synthetic header and packet-loop rows, the
  reduce setter order of upstream issue #1631, the allocation census, layer-limited hashes (`layhash`)
  and the header check of a viewer texture cache. `PROBE_PROFILE=sl`, `PROBE_LAYERS=<n>` and
  `PROBE_JP2=1` in the environment select the SL limits profile, a layer limit and the JP2 format.
- `run_cuthash.sh <probe> <out dir> <list> [jobs]`: per-prefix hashes of every stream of a list (lines:
  `<codestream> <reduce list> <step>`). Two builds agree on cut decodes when their outputs are equal.
- `b5_check.py <layhash output> <cuthash output>`: every layer-limited whole-file decode of a
  layer-progressive single-tile stream must be one of its cut decodes.
- `run_mutants_san.sh <list> <probe> <out> [jobs]`: one sanitizer process per damaged codestream.

`OPJ_TEST_TMP` names the scratch folder for temporary files (excluded from malware scanning); without
it the scripts use `TEMP`.
