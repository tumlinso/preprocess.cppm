#!/usr/bin/env python3
"""Bounded native allocation/runtime observations; no comparative speed claim."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--driver", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("benchmarks/results/runtime.json"))
    p.add_argument("--rows", type=int, default=2048)
    p.add_argument("--columns", type=int, default=512)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--cpu", type=int, help="pin this probe and its children to an allowed CPU")
    a = p.parse_args()
    if a.repeats < 1 or a.repeats > 5:
        p.error("repeats must be between 1 and 5")
    driver = a.driver.resolve()
    if a.cpu is not None:
        if a.cpu not in os.sched_getaffinity(0):
            p.error("requested CPU is outside this process affinity")
        os.sched_setaffinity(0, {a.cpu})
    observations = []
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    for layout in ("dense", "csr"):
        for dtype in ("float32", "float64"):
            for op in ("baseline", "qc", "stats", "normalize", "log1p", "scale", "hvg"):
                runs = []
                for _ in range(a.repeats):
                    r = subprocess.run([str(driver), op, layout, dtype, str(a.rows), str(a.columns)],
                                       text=True, capture_output=True, check=True, timeout=30, env=env)
                    runs.append(json.loads(r.stdout))
                item = {k: v for k, v in runs[0].items() if k not in (
                    "milliseconds", "peak_rss_kib", "baseline_peak_rss_kib", "incremental_high_water_kib")}
                for key in ("milliseconds", "peak_rss_kib", "baseline_peak_rss_kib", "incremental_high_water_kib"):
                    item["median_" + key] = statistics.median(r[key] for r in runs)
                observations.append(item)
    report = {"schema": 1, "observed_at_unix": time.time(), "platform": platform.platform(),
              "cpu_affinity": sorted(os.sched_getaffinity(0)),
              "cpu_model": next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")), "unknown"),
              "driver_sha256": hashlib.sha256(driver.read_bytes()).hexdigest(), "repeats": a.repeats,
              "timeout_seconds_per_process": 30, "threads": 1,
              "limits": ["native observations only; no speedup comparison",
                         "RSS high-water includes preparation, code pages and allocator retention",
                         "allocation counters cover requested C++ new/new[] bytes during the operation, not direct mmap/malloc",
                         "reported time includes allocation-counter instrumentation",
                         "CSR uses int32 indices and int64 offsets; sparse scaling omits centering",
                         "HVG preparation normalizes to 10000 and applies log1p before measurement"],
              "observations": observations}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Recorded {len(observations)} native observations in {a.output}")


if __name__ == "__main__":
    main()
