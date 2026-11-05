#!/bin/sh
set -eu

dependency_source=${1:?Usage: sh tools/build_anndatapp.sh /path/to/anndatapp}
project_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cmake -S "$dependency_source" -B "$project_root/build/deps/anndatapp" -G Ninja \
  -DCMAKE_CXX_COMPILER=clang++-18 -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$project_root/build/deps/install" \
  -DANNDATAPP_ENABLE_HDF5=OFF -DANNDATAPP_ENABLE_ZARR=OFF \
  -DANNDATAPP_BUILD_TESTS=OFF -DANNDATAPP_BUILD_EXAMPLES=OFF
cmake --build "$project_root/build/deps/anndatapp" --parallel 2
cmake --install "$project_root/build/deps/anndatapp"
