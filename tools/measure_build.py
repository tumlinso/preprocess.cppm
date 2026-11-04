#!/usr/bin/env python3
"""Measure module/consumer iteration in a disposable source copy."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import time


def timed(argv, cwd=None):
    start = time.monotonic()
    r = subprocess.run([str(v) for v in argv], cwd=cwd, text=True, capture_output=True, timeout=30)
    if r.returncode:
        raise RuntimeError(f"Command failed: {argv}\n{r.stdout}\n{r.stderr}")
    return time.monotonic() - start


def artifacts(root):
    return {str(p.relative_to(root)): {"mtime_ns": p.stat().st_mtime_ns,
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
            for p in root.rglob("*") if p.is_file() and p.suffix in (".pcm", ".o")}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--anndata-prefix", type=Path, default=Path("build/deps/install"))
    p.add_argument("--output", type=Path, default=Path("benchmarks/results/build.json"))
    a = p.parse_args()
    source, dependency = a.source.resolve(), a.anndata_prefix.resolve()
    digest = hashlib.sha256()
    for base in (source / "src", source / "cmake"):
        for path in sorted(base.rglob("*")):
            if path.is_file():
                digest.update(str(path.relative_to(source)).encode()); digest.update(path.read_bytes())
    digest.update((source / "CMakeLists.txt").read_bytes())
    with tempfile.TemporaryDirectory(prefix="preprocess-build-probe-") as tmp:
        root = Path(tmp)
        tree, build, prefix, consumer = (root / n for n in ("source", "build", "install", "consumer"))
        shutil.copytree(source, tree, ignore=shutil.ignore_patterns(".git", "build", ".venv*", ".agents", ".aws", ".codex", "__pycache__"))
        measures = {}
        measures["configure_seconds"] = timed(["cmake", "-S", tree, "-B", build, "-G", "Ninja",
            "-DCMAKE_CXX_COMPILER=clang++-18", "-DCMAKE_BUILD_TYPE=Release", f"-DCMAKE_PREFIX_PATH={dependency}",
            "-DPREPROCESS_BUILD_TESTS=OFF", "-DPREPROCESS_BUILD_QUALIFICATION=OFF"])
        measures["clean_package_build_seconds"] = timed(["cmake", "--build", build, "-j2"])
        measures["noop_package_build_seconds"] = timed(["cmake", "--build", build, "-j2"])
        measures["install_seconds"] = timed(["cmake", "--install", build, "--prefix", prefix])
        consumer.mkdir()
        (consumer / "CMakeLists.txt").write_text("cmake_minimum_required(VERSION 3.28)\nproject(probe LANGUAGES CXX)\nset(CMAKE_CXX_EXTENSIONS OFF)\nfind_package(preprocess CONFIG REQUIRED)\nadd_executable(probe main.cpp)\ntarget_link_libraries(probe PRIVATE preprocess::preprocess)\n")
        (consumer / "main.cpp").write_text("import preprocess;\nint main() { auto x = anndata::Array::zeros<double>({2, 2}); return preprocess::feature_stats(x.view()).observations != 2; }\n")
        cb = root / "consumer-build"
        measures["consumer_configure_seconds"] = timed(["cmake", "-S", consumer, "-B", cb, "-G", "Ninja",
            "-DCMAKE_CXX_COMPILER=clang++-18", "-DCMAKE_BUILD_TYPE=Release", f"-DCMAKE_PREFIX_PATH={prefix};{dependency}"])
        measures["cold_consumer_build_seconds"] = timed(["cmake", "--build", cb, "-j2"])
        timed([cb / "probe"])
        with (consumer / "main.cpp").open("a") as out: out.write("\n// Consumer-local edit.\n")
        measures["warm_consumer_rebuild_seconds"] = timed(["cmake", "--build", cb, "-j2"])
        package_before, consumer_before = artifacts(build), artifacts(cb)
        with (tree / "src" / "preprocess.cpp").open("a") as out: out.write("\n// Private implementation probe.\n")
        measures["private_implementation_rebuild_seconds"] = timed(["cmake", "--build", build, "-j2"])
        timed(["cmake", "--install", build, "--prefix", prefix])
        measures["consumer_after_private_edit_seconds"] = timed(["cmake", "--build", cb, "-j2"])
        package_after, consumer_after = artifacts(build), artifacts(cb)
        pcm = [name for name in package_before if name.endswith(".pcm")]
        checks = {"package_bmis_preserved": bool(pcm) and all(package_before[k] == package_after.get(k) for k in pcm),
                  "consumer_bmis_and_objects_preserved": bool(consumer_before) and consumer_before == consumer_after}
        if not all(checks.values()):
            raise RuntimeError(f"Implementation isolation failed: {checks}")
    report = {"schema": 1, "platform": platform.platform(), "source_sha256": digest.hexdigest(),
              "compiler": subprocess.check_output(["clang++-18", "--version"], text=True).splitlines()[0],
              "cmake": subprocess.check_output(["cmake", "--version"], text=True).splitlines()[0],
              "ninja": subprocess.check_output(["ninja", "--version"], text=True).strip(),
              "anndata_prefix": str(dependency), "timeout_seconds_per_command": 30,
              "jobs": 2, "measurements": measures, "checks": checks,
              "limits": ["one bounded observation, not a comparative performance claim",
                         "cold consumer build includes consumer-local module interface compilation",
                         "warm consumer rebuild includes compiler, scanner, build-system and linker costs"]}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(a.output), "checks": checks, "measurements": measures}))


if __name__ == "__main__":
    main()
