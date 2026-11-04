# Coding style

Write compact and explicit numerical code in the tradition of classic C and
Fortran scientific libraries. Keep the calculation, storage layout, and
execution visible.

- Keep related mathematics and invariants together. Use concise conventional
  notation and direct loops; extract helpers for real reusable calculations
  or distinct contracts.
- Use arrays, pointers, references, dimensions, and strides that show where
  values come from and where they go. Describe ownership, aliasing, and byte
  versus element strides precisely.
- Make allocation, conversion, copying, and materialization deliberate.
  Reuse caller-provided storage where it helps.
- Use small structs and ordinary free functions for data and computations.
  Introduce classes for ownership or real invariants. Do not add callback
  layers, wrapper hierarchies, or generic frameworks without a concrete user.
- Validate types, layouts, ownership, and dimensions at operation boundaries.
  Keep acquired numeric loops free of repeated checks, type erasure and
  library synchronization.
- Keep C++20 named-module interfaces small and independently compiled.
  Benchmark runtime or build-cost claims on identical workloads.

Follow the written
[Cellerator C++ coding style](/home/tumlinson/Cellerator/coding_style.md),
not the implementation of another project. Keep that guide unchanged. Fast
imports and consumer recompilation remain important.
