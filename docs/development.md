# Development and packaging

## Toolchain

Use Linux with Clang 18 and its matching clang-scan-deps, CMake 3.28 or newer,
and Ninja 1.11 or newer. CMake's CXX_MODULES file sets and dependency scanning
manage module build order. C++20 named modules are required; import std is
not.

The reference local dependency was anndata.cppm commit
dc946845e1457299351b0ffc4ba36ef91d49db7b. The helper accepts a path to a
checkout, builds anndata's dependency-free core in build/deps, and installs
it in build/deps/install. It does not change the supplied checkout.

~~~sh
sh tools/build_anndata.sh /path/to/anndata.cppm
cmake --workflow --preset native-check
~~~

An installed anndata package may also be selected by setting
CMAKE_PREFIX_PATH. The package installs its named-module interface sources,
CMake target, version file and third-party notice. Consumers require compatible
compiler and standard-library module artifacts; they build their own BMIs.

Version 0.1 follows the ordinary C++ ABI of the compiler, standard library,
target platform and ABI-affecting build settings used for the installed
library. Build and consume it with a compatible toolchain configuration. The
package does not promise a stable ABI across compilers or standard libraries.

## Package configurations

~~~sh
cmake --preset release
cmake --build --preset release
ctest --preset release

cmake --preset sanitizers
cmake --build --preset sanitizers
ctest --preset sanitizers

cmake --preset shared
cmake --build --preset shared
ctest --preset shared
~~~

The installed-consumer check uses find_package(preprocess CONFIG REQUIRED) in a
separate build and builds two executables against one installed module
provider. The native package build and this consumer build do not require Python.
Consumer projects should disable compiler-specific language extensions so the
module provider and importer use the same C++20 mode:

~~~cmake
cmake_minimum_required(VERSION 3.28)
project(example LANGUAGES CXX)
set(CMAKE_CXX_EXTENSIONS OFF)
find_package(preprocess CONFIG REQUIRED)
add_executable(example main.cpp)
target_link_libraries(example PRIVATE preprocess::preprocess)
~~~

With Clang, leaving extensions enabled selects gnu++20 and can make the BMI
incompatible with the package's c++20 module interface.

## Optional Python comparison

Install the pinned development dependencies from
tests/requirements-reference.txt, build the qualification driver, and compare
against the pinned Scanpy 1.11.5 implementation:

Use an isolated environment for these test dependencies. For example,
`python3 -m venv .venv-reference` creates one when the system supplies ensurepip;
`uv venv --python python3 .venv-reference` is an alternative. Activate it before
running the commands below.

~~~sh
python -m pip install -r tests/requirements-reference.txt
cmake --preset release -DPREPROCESS_BUILD_QUALIFICATION=ON
cmake --build build/release --target preprocess_interop_driver --parallel 2
python tools/qualify.py --driver build/release/preprocess_interop_driver
~~~

Python and Scanpy are test tools, not package runtime dependencies. Fixtures are
seeded and generated locally. The reference tests do not fetch biological
datasets.

## Bounded native measurements

~~~sh
cmake --preset release -DPREPROCESS_BUILD_QUALIFICATION=ON
cmake --build --preset release
python3 tools/measure_runtime.py --driver build/release/preprocess_benchmark --cpu 2
python3 tools/measure_build.py
~~~

Choose an allowed idle CPU for `--cpu`, and avoid simultaneous compilation or
other measurements. Runtime observations are medians of three fresh processes;
build observations run in disposable source copies. Every subprocess is capped
at 30 seconds. Compact results are committed under benchmarks/results; neither
tool changes the original numerical source. See [validation](validation.md)
for the measured limits.

## Project Control

The repository is registered under the Project Control alias preprocess-cppm.
This registration supplies source context only. It does not create a Todo
authority, credentials, tasks, execution handles or host runtime.
