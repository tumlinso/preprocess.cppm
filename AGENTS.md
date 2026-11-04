# Working on preprocess.cppm

## Purpose and boundaries

preprocess.cppm is a small C++20 library of reusable biological preprocessing
calculations built on anndata.cppm. Its public functions provide an explicit
CPU reference for Cellerator and other native consumers. Scanpy supplies pinned
numerical comparison behavior; it does not dictate Python-style call semantics.
Numerical similarity is required; bitwise identity and universal equality of
intermediates are not requirements. Explain consequential differences and
qualify them with scientific quantities and tolerances.

The dependency direction is downstream: Cellerator may use this package.
preprocess depends on anndata. It does not depend on Python, Scanpy at runtime,
Torch, CUDA, or Cellerator. Keep preprocessing, fitting, PCA, neighbors, GPU
execution, and dataset orchestration as separate concerns.

## C++ and numerical style

- Follow STYLE.md and the protected
  [Cellerator C++ coding style](/home/tumlinson/Cellerator/coding_style.md).
  Keep its reference unchanged.
- Use C++20 named modules and .cppm interfaces. Public consumption is
  import preprocess;. Do not require import std; or add a public umbrella
  header.
- Keep exported module interfaces and consumer recompilation small. Put
  implementation-only code in compiled source files.
- Use direct arrays, pointers, extents, strides, sparse offsets and indices.
  Make allocation, conversion, mutation, and materialization explicit.
- Use anndata's views directly. Keep shape, layout, lifetime and buffer
  representation with anndata; do not maintain a parallel matrix abstraction.
  Preprocess owns numerical operations and their additional contracts.
- Resolve validation and representation dispatch at acquisition boundaries.
  Keep element loops free of type-erased dispatch, repeated validation and
  hidden synchronization. Callers own parallel execution.
- Keep numerical types and accumulation explicit. Preserve the operation's
  declared mathematical method and test against the pinned reference cases.

## Build and validation

Use Linux, Clang 18 with matching clang-scan-deps, CMake 3.28 or newer, and
Ninja 1.11 or newer.

~~~sh
sh tools/build_anndata.sh /path/to/anndata.cppm
cmake --workflow --preset native-check

cmake --preset sanitizers
cmake --build --preset sanitizers
ctest --preset sanitizers
~~~

Optional Python interoperability is development-only. Its reference environment
pins Scanpy 1.11.5 and the packages in tests/requirements-reference.txt.
See docs/development.md and docs/validation.md.

Performance probes should be bounded to 30 seconds and compare identical
workloads. Keep source and toolchain identity with results. Do not claim speed
or memory improvements from source inspection alone.

## Project Control and preservation

Project Control workspace and repository alias: preprocess-cppm. The repository
is registered as a source project. This repository does not bootstrap a Todo
authority, credentials, task, or host runtime.

Preserve unknown work and supplied inputs. Do not reset or clean Git state,
force-push, replace authority data, or edit sibling anndata or Cellerator
repositories. Keep generated builds, local reference environments, data and
raw reports out of Git.

For the initial delivery, the user explicitly authorized implementation without
a Project Control workflow handle and prohibited restarting the service.

Use the configured Codex agent profiles for useful independent work. Root
retains user intent, interface decisions, integration, and acceptance. Use
Codex subagents for engineering; preserve concurrent edits and never use
Project Control local coding workers as engineering delegates.
