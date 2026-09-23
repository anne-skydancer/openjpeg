# GPU encoder optimization plan

Baseline: 34974482, optional GPU encoding, exact OpenJPEG codestreams.

1. Preserve baseline binaries for paired measurements. Fuse GPU transforms,
   quantization/packing and Tier-1 under one worker lease. Retain atomic CPU
   fallback and the aggregate 64 MiB admission limit. Require correctly rounded
   floating-point division for GPU lossy quantization.
2. Extend exact-output tests to require the fused path, including concurrent
   callers, odd/tiled images, multiple layers, signed samples and fallback.
3. Profile the fused path. Measure MQ renormalization and coefficient-access
   optimizations independently; retain only exact candidates with repeatable gains.
4. Compare CPU and old/new GPU at multiple image-worker counts on the same cached
   textures. Publish limits as well as gains. Validate decoder and CPU-only builds.
5. Produce a separate optimized Release DLL; preserve the deployed decoder.

## Implementation and decisions

- Implemented one-lease GPU transforms, quantization and Tier-1, with exact
  floating-point division and round-to-even conversion for lossy quantization.
- Quantization writes each sample in place. Code blocks partition the planes;
  the entropy kernel reads the original row stride, avoiding a second full
  coefficient buffer. CPU tile samples stay untouched until staged fallback.
- Retained MQ renormalization that shifts to a byte boundary rather than one bit
  at a time. Three paired trials on the intermediate fused path improved mean
  throughput about 3-4% at one/four workers for lossless and lossy encoding.
- Rejected a local coefficient cache: exact but slower. Measurements are recorded
  in variants/README.md; its intermediate patch/build remains in ignored build/.
- Initial fusion reduced eight-worker throughput because admission waited for
  larger per-job buffers. Nonblocking fused admission now retries smaller GPU
  stages under pressure; decoder and staged admission retain their waiting policy.
- Expanded exact tests cover 135 fused cases and three fallback cases, including
  unusual code-block shapes and a tile exceeding the combined 64 MiB budget.
  Concurrent tests require complete GPU stage coverage and some fused execution
  while allowing the staged route selected by memory pressure. Pools above four
  slots use the staged route throughout: the final sweep found an 11% lossy
  regression with eight fused-capable slots, so that configuration was not kept.

Final benchmarks and validation evidence are recorded in README.md. The reusable
compare_encode.py records input and binary hashes, requires matching outputs and
complete GPU stage coverage, and reverses execution order on alternating trials.
