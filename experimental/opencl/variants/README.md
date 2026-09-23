# Reproducible entropy experiments

These research patches apply independently to the optimized scalar implementation
shipped beside this document. They are not loaded or compiled in a normal build.
Both preserve ordinary JPEG2000 codestream compatibility and passed the full
29,184-block reference corpus. Neither was selected as the active decoder.

- `cooperative-refinement.patch`: 32 work-items per code block, batched refinement
  context preparation and coefficient updates, with one ordered MQ owner. It also
  changes the native and reference-test launch geometry. Eight-lane screening was
  slower; this patch retains the better 32-lane experiment.
- `mq-lookahead.patch`: evaluate both possible first arithmetic exchange paths
  and a second refinement decision; commit only the actual state. RAW refinement
  uses the ordinary path. Extra work/state outweighed the benefit in screening.

Apply only one patch at a time, from the repository root, with a clean checkout:

```powershell
git apply --check experimental/opencl/variants/cooperative-refinement.patch
git apply experimental/opencl/variants/cooperative-refinement.patch
python experimental/opencl/test_t1.py --corpus build/cpu-reference/experimental/opencl/corpus/blocks.jsonl --device gfx1201 --driver 3679
cmake --build build/gpu --config RelWithDebInfo --parallel 8
ctest --test-dir build/gpu -C RelWithDebInfo --output-on-failure
```

Use the `bench_workers` commands in the parent README for measurements. Require
`gpu_tiles == tiles` and matching checksums; CPU fallback invalidates a GPU timing.
Exclude warmup and repeat in alternating order against a saved baseline executable.
Restore the scalar sources before testing the other patch:

```powershell
git apply -R experimental/opencl/variants/cooperative-refinement.patch
cmake --build build/gpu --config RelWithDebInfo --parallel 8
```

Substitute `mq-lookahead.patch` for the second experiment. The patches are tied to
this source revision; `git apply --check` should fail if future changes conflict.
The runtime remains opt-in and no viewer package is changed by these experiments.
