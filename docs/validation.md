# Validation

This document records the v0.1 validation and its numerical conventions.

## Acceptance checks

- Native analytic and seeded tests cover dense, CSR and CSC layouts; row-major
  strides; integer summaries; explicit sparse zeros; all-zero rows and
  matrices; sample statistics; intersecting filters; repeated mask use;
  normalization factors; logarithms; scaling; batch groups; bin behavior; HVG
  score order and ties; invalid shapes/options; and explicit materialization.
- The optional independent comparison uses Scanpy 1.11.5. It checks feature
  alignment and metadata, and compares floating-point results with tolerances:
  FP32 uses rtol=1e-5, atol=1e-6; FP64 uses rtol=1e-8, atol=1e-10. Non-finite
  classifications must match. Selection-mask differences are reported per
  fixture; exact mask agreement is expected for cases without tied or
  near-cutoff scores, while tie behavior is reviewed with score and ordering
  context. The automatic-target rule for zero-total input is tested
  independently because it is deliberately layout-independent here.
- Native tests run without Python. ASan/UBSan, static and shared installed
  consumers, two-executable module sharing, and documented examples are part
  of the package checks.
- Runtime, working-memory, allocation, module import/build and private
  implementation rebuild measurements use bounded identical workloads. Each
  initial probe is limited to 30 seconds. Measurements identify sources,
  binaries, inputs, compiler and conditions.

## Results

The current integration source was built with Clang 18.1.3, CMake 3.28.3 and
Ninja 1.11.1, against anndata.cppm commit
dc946845e1457299351b0ffc4ba36ef91d49db7b. Debug/native, Release/static, shared
Release and ASan/UBSan CTest runs each passed 4/4: numerical tests, HVG tests, the compiled
example and the installed consumer. The installed-consumer check builds and
runs two executables against one installed module provider. Native tests and
consumer checks do not require Python.

The sanitizer configuration instruments preprocess and its consumers; the
separately built anndata dependency is an ordinary Release build.

The independent [Scanpy report](../tests/results/reference.json) passes
**302/302 cases**. It records production-source hashes, native source and driver
hashes, pinned dependencies, fixture hashes, and per-operation errors. The
fixtures exercise dense/CSR/CSC/reversed strides, both precisions, integer
inputs, QC, intersecting filters, statistics, normalization/exclusion, three
logarithm bases, scaling/masks/clipping, both HVG methods, batches/cutoff ties,
annotations, and large FP64 inputs transformed into FP32 destinations.

No HVG selection or batch-summary disagreements occurred in these fixtures.
Maximum normalized-dispersion absolute error was 1.83e-5 for FP32 and 2.66e-15
for FP64, within the declared combined relative/absolute tolerances. The goal
is numerical similarity. These results do not promise universal identity of
intermediates, selections at ambiguous boundaries, or metadata storage dtypes.

The [runtime observations](../benchmarks/results/runtime.json) cover 28
operation/layout/precision combinations, each repeated three times, on seeded
2,048 by 512 matrices. Processes were pinned to CPU 2 on an Intel Xeon
E5-2698 v4. Every subprocess had a 30-second timeout. The report records native
time, requested C++ allocation counts/bytes, and process RSS high-water marks.
Timings include allocation-counter instrumentation; RSS includes preparation,
code pages, and allocator retention. These are native characterizations, not
comparisons with another library or speedup claims.

The [build observation](../benchmarks/results/build.json) used a disposable
source copy, two build jobs, and a 30-second timeout per command:

| Observation | Seconds |
|---|---:|
| Clean package build | 27.71 |
| Cold installed-consumer build, including module interfaces | 8.94 |
| Warm consumer rebuild | 0.29 |
| Private implementation rebuild | 19.42 |
| Consumer rebuild after private implementation edit | 0.14 |

A private implementation edit preserved package BMIs and all consumer BMIs and
objects. The consumer only needed relinking. These are single observations on
the recorded toolchain, rather than general compilation-time guarantees.

Numerical agreement is software evidence. It is not biological validation or a
universal performance claim.
