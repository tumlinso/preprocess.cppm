# preprocess.cppm

preprocess.cppm is a small native C++20 library of reusable preprocessing
calculations for annotated biological data. It requires
[AnnData++](https://github.com/tumlinso/anndatapp) and provides a direct
CPU reference based on Scanpy's numerical behavior.

The interface is C++-native: typed options and results, explicit ownership,
mutable versus read-only array access, and free functions. Cellerator may use
the calculations or compare specialized implementations with the reference.
The package has no runtime dependency on Python, Scanpy, Torch, CUDA, or
Cellerator.

The example below assumes an existing `anndatapp::AnnData` named `data`; a
complete compiled example is in [examples/basic.cpp](examples/basic.cpp).

~~~cpp
#include <utility>
import preprocess;

auto result = preprocess::qc(preprocess::source(data));
auto cells = preprocess::filter_cells(result, {.min_detected = 200});

auto normalized = preprocess::floating_copy(
    preprocess::source(data), anndatapp::DType::float32);
auto factors = preprocess::normalize_total(
    normalized.edit(), {.target_sum = 1e4});
preprocess::log1p(normalized.edit());
preprocess::annotate_qc(data, std::move(result));
preprocess::annotate_normalization(data, std::move(factors));
data.set_layer("normalized", std::move(normalized));
~~~

The numerical scope includes quality-control summaries, filter masks, total
normalization, log1p, feature scaling, and Seurat or Cell Ranger dispersion
based highly variable gene selection. It does not include PCA, neighborhood
construction, Seurat v3, dataset orchestration, or device execution. See
[operations](docs/operations.md) for numerical conventions and
[development](docs/development.md) for builds and qualification.

## Build

The qualified toolchain is Linux, Clang 18 with matching clang-scan-deps,
CMake 3.28 or newer, and Ninja 1.11 or newer. Build the dependency-free AnnData++
core from a supplied checkout, then build and test preprocess:

~~~sh
sh tools/build_anndatapp.sh /path/to/anndatapp
cmake --workflow --preset native-check
~~~

The installed CMake package provides target preprocess::preprocess and supports
find_package(preprocess CONFIG REQUIRED). Static, shared, and sanitizer
presets are available. Python and Scanpy are used only for optional independent
qualification. Version 0.1 uses the ordinary compiler and standard-library
ABI, so consumers should use a compatible toolchain configuration; no
cross-toolchain stable ABI is promised.

The pinned Scanpy comparison passes 302 cases across the supported operations,
layouts, precisions, annotations, and declared numerical conventions. See
[validation](docs/validation.md) for the evidence and coverage limits.

## License

Original code is MIT licensed. See [third-party notices](THIRD_PARTY_NOTICES.md)
for the independent Scanpy reference.
