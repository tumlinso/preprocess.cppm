#!/bin/sh
set -eu

dependency_source=${1:?Usage: sh tools/build_anndata.sh /path/to/anndata.cppm}
project_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cmake -S "$dependency_source" -B "$project_root/build/deps/anndata" -G Ninja \
  -DCMAKE_CXX_COMPILER=clang++-18 -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$project_root/build/deps/install" \
  -DANNDATA_ENABLE_HDF5=OFF -DANNDATA_ENABLE_ZARR=OFF \
  -DANNDATA_BUILD_TESTS=OFF -DANNDATA_BUILD_EXAMPLES=OFF
cmake --build "$project_root/build/deps/anndata" --parallel 2
cmake --install "$project_root/build/deps/anndata"
