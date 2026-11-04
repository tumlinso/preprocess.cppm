#!/usr/bin/env python3
"""Compare the C++ reference with Scanpy 1.11.5 production operations.

The driver uses a simple text protocol; this script constructs both its input
and the independent Python AnnData input from the same seeded logical matrix.
No Python dependency is needed by the C++ package itself.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import warnings

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "preprocess-matplotlib"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "preprocess-numba-cache"))
for name in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scanpy.preprocessing._utils import _get_mean_var

ROOT = Path(__file__).resolve().parents[1]
VERSION = "1.11.5"
FLOAT_TOL = {32: (1e-5, 1e-6), 64: (1e-8, 1e-10), 0: (1e-5, 1e-6)}
EXACT_FIELDS = {"cell_detected", "gene_detected", "cells", "genes", "selected", "included", "nbatches", "intersection",
                "obs_n_genes_by_counts", "var_n_cells_by_counts", "var_highly_variable", "metadata",
                "axis_alignment", "obs_sentinel", "var_sentinel"}
LAYOUTS = {"dense": 0, "csr": 1, "csc": 2, "reverse-stride": 3}


def native(driver, operation, x, layout, dtype, mode, groups, observations, batches):
    n, g = x.shape
    as_input = np.asarray(x, dtype=np.int32 if dtype == 0 else np.float32 if dtype == 32 else np.float64)
    parts = [f"{n} {g} {dtype} {LAYOUTS[layout]} {mode}"]
    parts.append(" ".join(format(float(v), ".17g") for v in as_input.flat))
    parts.append(" ".join(str(int(v)) for v in groups))
    parts.append(" ".join(str(int(v)) for v in observations))
    parts.append(" ".join(str(int(v)) for v in batches))
    result = subprocess.run([str(driver), operation], input="\n".join(parts) + "\n",
                            capture_output=True, text=True, timeout=30)
    if result.returncode:
        raise RuntimeError(f"native {operation} exited {result.returncode}: {result.stderr.strip()}")
    out = {}
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) < 2 or fields[0] in out:
            raise RuntimeError(f"invalid or repeated native output line: {line[:120]}")
        rank = int(fields[1])
        shape = tuple(int(v) for v in fields[2:2+rank])
        values = np.array([float(v) for v in fields[2+rank:]], dtype=np.float64)
        if values.size != int(np.prod(shape)):
            raise RuntimeError(f"native output {fields[0]} has wrong value count for {shape}")
        out[fields[0]] = values.reshape(shape)
    return out


def scanpy_matrix(x, layout, dtype):
    t = np.int32 if dtype == 0 else np.float32 if dtype == 32 else np.float64
    array = np.asarray(x, dtype=t).copy()
    return sparse.csr_matrix(array) if layout == "csr" else sparse.csc_matrix(array) if layout == "csc" else array


def dense(x):
    return x.toarray() if sparse.issparse(x) else np.asarray(x)


def qc_oracle(x, groups):
    a = ad.AnnData(X=x.copy())
    a.var["control"] = groups.astype(bool)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        obs, var = sc.pp.calculate_qc_metrics(a, qc_vars=["control"], percent_top=[2, 4, 8],
                                               log1p=True, inplace=False)
    return {
        "cell_total": obs["total_counts"].to_numpy(),
        "cell_detected": obs["n_genes_by_counts"].to_numpy(),
        "gene_total": var["total_counts"].to_numpy(),
        "gene_detected": var["n_cells_by_counts"].to_numpy(),
        "gene_mean": var["mean_counts"].to_numpy(),
        "gene_dropout": var["pct_dropout_by_counts"].to_numpy(),
        "log_cell_total": obs["log1p_total_counts"].to_numpy(),
        "log_cell_detected": obs["log1p_n_genes_by_counts"].to_numpy(),
        "log_gene_total": var["log1p_total_counts"].to_numpy(),
        "log_gene_mean": var["log1p_mean_counts"].to_numpy(),
        "group_total": obs["total_counts_control"].to_numpy(),
        "group_percent": obs["pct_counts_control"].to_numpy(),
        "group_log_total": obs["log1p_total_counts_control"].to_numpy(),
        "top_percent": obs[[f"pct_counts_in_top_{k}_genes" for k in (2, 4, 8)]].to_numpy(),
    }


def filter_oracle(x):
    n, g = x.shape
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cells = np.logical_and.reduce([
            sc.pp.filter_cells(x, min_counts=4, inplace=False)[0],
            sc.pp.filter_cells(x, max_counts=10000, inplace=False)[0],
            sc.pp.filter_cells(x, min_genes=2, inplace=False)[0],
            sc.pp.filter_cells(x, max_genes=g, inplace=False)[0],
        ])
        genes = np.logical_and.reduce([
            sc.pp.filter_genes(x, min_counts=5, inplace=False)[0],
            sc.pp.filter_genes(x, max_counts=10000, inplace=False)[0],
            sc.pp.filter_genes(x, min_cells=2, inplace=False)[0],
            sc.pp.filter_genes(x, max_cells=n, inplace=False)[0],
        ])
    return {"cells": cells, "genes": genes}


def norm_oracle(x, mode):
    values = dense(x).astype(np.float64)
    totals = values.sum(axis=1)
    included = ~(values > (.2 * totals[:, None])).any(axis=0) if mode == 2 else np.ones(values.shape[1], dtype=bool)
    eligible = values[:, included].sum(axis=1)
    if mode == 1 and np.any(eligible == 0):
        # Declared cross-layout behavior, independently calculated. Scanpy 1.11.5
        # uses different automatic targets for dense and sparse zero rows.
        positive = eligible[eligible > 0]
        target = float(np.median(positive)) if positive.size else 1.0
        factors = eligible / target
        out = values / np.where(factors > 0, factors, 1)[:, None]
    else:
        target = float(np.median(eligible[eligible > 0])) if mode == 1 else 100.0
        a = ad.AnnData(X=x.copy())
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            sc.pp.normalize_total(a, target_sum=None if mode == 1 else target,
                                  exclude_highly_expressed=mode == 2, max_fraction=.2,
                                  key_added="factor")
        factors = a.obs["factor"].to_numpy()
        out = dense(a.X)
    return {"matrix": out, "factors": factors, "included": included, "target": np.array([target])}


def scale_oracle(x, mode, observations):
    a = ad.AnnData(X=x.copy())
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        sc.pp.scale(a, zero_center=mode != 1, max_value=3 if mode else None,
                    mask_obs=observations.astype(bool) if mode == 2 else None)
    suffix = " with mask" if mode == 2 else ""
    return {"matrix": dense(a.X), "mean": a.var["mean" + suffix].to_numpy(),
            "stddev": a.var["std" + suffix].to_numpy()}


def hvg_oracle(x, mode, batches):
    a = ad.AnnData(X=x.copy())
    # Scanpy groups batched results by feature label before sorting ties. These
    # labels sort in positional order, the documented native tie convention.
    a.var_names = [f"gene_{i:04d}" for i in range(a.n_vars)]
    if mode & (8 | 16):
        a.uns["log1p"] = {"base": 2 if mode & 8 else .5}
    if mode & 2:
        a.obs["batch"] = pd.Categorical(batches.astype(str))
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        result = sc.pp.highly_variable_genes(a, flavor="cell_ranger" if mode & 1 else "seurat", n_bins=10,
                                             n_top_genes=None if mode & 4 else 12,
                                             batch_key="batch" if mode & 2 else None, inplace=False)
    # The returned DataFrame retains the calculation's FP64 precision; the
    # alternate inplace path casts `var['dispersions_norm']` to FP32.
    out = {"mean": result["means"].to_numpy(), "dispersion": result["dispersions"].to_numpy(),
           "normalized_dispersion": result["dispersions_norm"].to_numpy(),
           "selected": result["highly_variable"].to_numpy()}
    if mode & 2:
        out["nbatches"] = result["highly_variable_nbatches"].to_numpy()
        out["intersection"] = result["highly_variable_intersection"].to_numpy()
    return out


def annotations_oracle(x, groups, mode):
    a = ad.AnnData(X=x.copy())
    original_obs_names = [f"sample_{a.n_obs-i}" for i in range(a.n_obs)]
    original_var_names = [f"gene_{a.n_vars-i}" for i in range(a.n_vars)]
    a.obs_names = original_obs_names
    a.var_names = original_var_names
    a.obs["sentinel"] = np.arange(a.n_obs, dtype=np.int64)
    a.var["sentinel"] = np.arange(a.n_vars, dtype=np.int64)
    a.layers["counts"] = a.X.copy()
    a.var["control"] = groups.astype(bool)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        sc.pp.calculate_qc_metrics(a, qc_vars=["control"], percent_top=[2, 4, 8], log1p=True, inplace=True)
        sc.pp.normalize_total(a, target_sum=100, key_added="size_factors")
        sc.pp.log1p(a, base=2 if mode == 1 else .5 if mode == 2 else None)
        hvg_result = sc.pp.highly_variable_genes(a, flavor="seurat", n_bins=10, n_top_genes=12, inplace=False)
        sc.pp.highly_variable_genes(a, flavor="seurat", n_bins=10, n_top_genes=12)
        sc.pp.scale(a)
    obs_names = ("total_counts", "n_genes_by_counts", "log1p_total_counts", "log1p_n_genes_by_counts",
                 "total_counts_control", "pct_counts_control", "log1p_total_counts_control",
                 "pct_counts_in_top_2_genes", "pct_counts_in_top_4_genes", "pct_counts_in_top_8_genes",
                 "size_factors")
    var_names = ("total_counts", "n_cells_by_counts", "mean_counts", "pct_dropout_by_counts",
                 "log1p_total_counts", "log1p_mean_counts", "means", "dispersions", "dispersions_norm",
                 "highly_variable", "mean", "std")
    result = {"obs_" + name: a.obs[name].to_numpy() for name in obs_names}
    result.update({"var_" + name: a.var[name].to_numpy() for name in var_names})
    result["var_dispersions_norm_fp32"] = result["var_dispersions_norm"]
    result["var_dispersions_norm"] = hvg_result["dispersions_norm"].to_numpy()
    expected_base = 2 if mode == 1 else .5 if mode == 2 else None
    result["metadata"] = np.array([a.uns["log1p"]["base"] == expected_base, a.uns["hvg"]["flavor"] == "seurat"])
    result["axis_alignment"] = np.array([list(a.obs_names) == original_obs_names,
                                         list(a.var_names) == original_var_names])
    result["obs_sentinel"] = a.obs["sentinel"].to_numpy()
    result["var_sentinel"] = a.var["sentinel"].to_numpy()
    result["layer_counts"] = dense(a.layers["counts"])
    result["matrix"] = dense(a.X)
    return result


def oracle(operation, x, layout, dtype, mode, groups, observations, batches):
    matrix = scanpy_matrix(x, layout, dtype)
    if operation == "annotations": return annotations_oracle(matrix, groups, mode)
    if operation == "qc": return qc_oracle(matrix, groups)
    if operation == "filters": return filter_oracle(matrix)
    if operation == "stats":
        sample = matrix[observations.astype(bool)] if mode else matrix
        mean, variance = _get_mean_var(sample, axis=0)
        return {"mean": np.asarray(mean).ravel(), "variance": np.asarray(variance).ravel()}
    if operation in ("normalize", "normalize_into"):
        result = norm_oracle(matrix, mode)
        if operation == "normalize_into": result["matrix"] = result["matrix"].astype(np.float32)
        return result
    if operation in ("log1p", "log1p_into"):
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            out = sc.pp.log1p(matrix.copy(), base=2 if mode == 1 else .5 if mode == 2 else None)
        result = dense(out)
        return {"matrix": result.astype(np.float32) if operation == "log1p_into" else result}
    if operation == "scale": return scale_oracle(matrix, mode, observations)
    if operation == "hvg": return hvg_oracle(matrix, mode, batches)
    raise ValueError(f"unknown operation {operation}")


def compare(expected, actual, dtype, field_dtypes=None):
    if expected.keys() != actual.keys():
        return {"pass": False, "error": f"fields differ: expected {sorted(expected)}, got {sorted(actual)}"}
    fields, passed = {}, True
    for name, expected_values in expected.items():
        rtol, atol = FLOAT_TOL[(field_dtypes or {}).get(name, dtype)]
        e, a = np.asarray(expected_values), np.asarray(actual[name])
        if e.shape != a.shape:
            fields[name] = {"pass": False, "error": f"shape {a.shape}, expected {e.shape}"}
            passed = False
            continue
        if name in EXACT_FIELDS:
            ok = np.equal(e, a)
        else:
            ok = np.isclose(e, a, rtol=rtol, atol=atol, equal_nan=True)
        field_pass = bool(np.all(ok))
        detail = {"pass": field_pass, "mismatches": int(np.count_nonzero(~ok))}
        if name not in EXACT_FIELDS:
            finite = np.isfinite(e) & np.isfinite(a)
            if np.any(finite):
                difference = np.abs(a[finite] - e[finite])
                detail["max_abs_error"] = float(np.max(difference))
                detail["max_rel_error"] = float(np.max(difference / np.maximum(np.abs(e[finite]), atol)))
        if not field_pass:
            positions = np.flatnonzero(~ok)[:5]
            detail["examples"] = [{"index": int(i), "expected": str(e.flat[i]), "actual": str(a.flat[i])}
                                  for i in positions]
        fields[name] = detail
        passed &= field_pass
    return {"pass": passed, "fields": fields}


def inputs():
    rng = np.random.default_rng(821047)
    n, g = 36, 96
    means = np.exp(np.linspace(-1.0, 2.4, g))
    x = rng.poisson(means, size=(n, g)).astype(np.float64)
    x[:, 0] += 1
    # Selected equal observations exercise constant and zero features.
    x[:, -1] = 0
    groups = (np.arange(g) % 7 == 0)
    observations = (np.arange(n) % 3 != 0)
    batches = (np.arange(n) % 3).astype(np.int32)
    return x, groups, observations, batches


def source_info():
    base = Path(sc.__file__).resolve().parent / "preprocessing"
    files = ("_qc.py", "_simple.py", "_normalization.py", "_scale.py", "_highly_variable_genes.py", "_utils.py")
    versions = {p: importlib.metadata.version(p) for p in
                ("scanpy", "anndata", "numpy", "scipy", "pandas", "numba", "statsmodels")}
    if versions["scanpy"] != VERSION:
        raise RuntimeError(f"Scanpy {VERSION} is required; found {versions['scanpy']}")
    hashes = {f: hashlib.sha256((base / f).read_bytes()).hexdigest() for f in files}
    native_sources = [str(p.relative_to(ROOT)) for p in sorted((ROOT / "src").rglob("*"))
                      if p.suffix in (".cppm", ".cpp", ".hpp")]
    native_sources.append("tests/interop_driver.cpp")
    native_hashes = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in native_sources}
    return {"versions": versions, "scanpy_source_sha256": hashes, "native_source_sha256": native_hashes,
            "qualifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "requirements_sha256": hashlib.sha256((ROOT / "tests/requirements-reference.txt").read_bytes()).hexdigest(),
            "python": sys.version.split()[0], "fixture_seed": 821047}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--driver", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "tests/results/reference.json")
    args = parser.parse_args()
    driver = args.driver.resolve(strict=True)
    provenance = source_info()
    x, groups, observations, batches = inputs()
    with np.errstate(all="ignore"):
        hvg_x = np.log1p(x / x.sum(axis=1)[:, None] * 100)
    cases = []
    def exercise(operation, values, dtype, layout, mode, tag="ordinary"):
        name = f"{operation}:{dtype}:{layout}:{mode}:{tag}"
        try:
            expected = oracle(operation, values, layout, dtype, mode, groups, observations, batches)
            actual = native(driver, operation, values, layout, dtype, mode, groups, observations, batches)
            if operation == "annotations":
                # Verify the values Scanpy would store after its FP32 cast as
                # well as preserving the native annotation's FP64 result.
                actual["var_dispersions_norm_fp32"] = actual["var_dispersions_norm"].astype(np.float32)
            output_dtype = {"matrix": 32} if operation.endswith("_into") else None
            result = compare(expected, actual, dtype, output_dtype)
        except Exception as exc:
            result = {"pass": False, "error": f"{type(exc).__name__}: {exc}"}
        cases.append({"case": name, **result})
        if not result["pass"]: print(f"FAIL {name}: {json.dumps(result, sort_keys=True)}", file=sys.stderr)

    for dtype in (32, 64, 0):
        for layout in LAYOUTS:
            for operation in ("qc", "filters", "stats", "normalize", "log1p", "scale"):
                modes = {"qc": (0,), "filters": (0,), "stats": (0, 1), "normalize": (0, 1, 2),
                         "log1p": (0, 1, 2), "scale": (0, 1, 2)}[operation]
                for mode in modes: exercise(operation, x, dtype, layout, mode)
    for dtype in (32, 64):
        for layout in LAYOUTS:
            for mode in range(8): exercise("hvg", hvg_x, dtype, layout, mode)
            exercise("hvg", hvg_x / np.log(2), dtype, layout, 8, "base-2")
            exercise("hvg", hvg_x / np.log(.5), dtype, layout, 16, "base-half")
        for mode in (0, 1, 2): exercise("annotations", x, dtype, "dense", mode)
    # Scanpy selects all genes tied at the single-batch dispersion threshold;
    # the batched top-N cutoff instead breaks equal scores by feature position.
    # Padded feature labels above make Scanpy's label ordering match that policy.
    single_tie = hvg_x.copy(); single_tie[:, 82] = single_tie[:, 81]
    batch_tie = hvg_x.copy(); batch_tie[:, 0] = batch_tie[:, 53]
    for dtype in (32, 64):
        for layout in LAYOUTS:
            single = hvg_oracle(scanpy_matrix(single_tie, layout, dtype), 0, batches)
            if int(single["selected"].sum()) <= 12 or not single["selected"][[81, 82]].all():
                raise RuntimeError("single-batch tie fixture no longer exercises the cutoff")
            batched = hvg_oracle(scanpy_matrix(batch_tie, layout, dtype), 2, batches)
            if (batched["selected"][0] == batched["selected"][53] or
                batched["nbatches"][0] != batched["nbatches"][53] or
                batched["normalized_dispersion"][0] != batched["normalized_dispersion"][53]):
                raise RuntimeError("batched tie fixture no longer exercises positional ordering")
            exercise("hvg", single_tie, dtype, layout, 0, "single-tie")
            exercise("hvg", batch_tie, dtype, layout, 2, "batch-tie")
    # Ensure highly expressed-gene exclusion actually removes one feature.
    for dtype in (32, 64):
        for layout in LAYOUTS:
            outlier = x.copy(); outlier[1, 1] = 20000
            exercise("normalize", outlier, dtype, layout, 2, "excluded-feature")
    # Exercise the deliberate zero-row/fully-zero automatic normalization contract.
    for dtype in (32, 64):
        for layout in LAYOUTS:
            z = x.copy(); z[0] = 0
            exercise("normalize", z, dtype, layout, 1, "zero-row")
            exercise("normalize", np.zeros_like(x), dtype, layout, 1, "all-zero")
    # Casting 1e200 to FP32 before transforming overflows, even though both
    # normalized values and log1p values are representable in the destination.
    extreme = x * 1e200; extreme[1, 1] = 1e205
    for layout in LAYOUTS:
        for mode in (0, 2): exercise("normalize_into", extreme, 64, layout, mode, "fp64-to-fp32")
        for mode in (0, 1, 2): exercise("log1p_into", extreme, 64, layout, mode, "fp64-to-fp32")
    passed = sum(c["pass"] for c in cases)
    compact, coverage, accuracy = [], {}, {}
    for case in cases:
        operation, dtype = case["case"].split(":", 2)[:2]
        coverage.setdefault(operation, set()).update(case.get("fields", {}))
        fields = case.get("fields", {})
        for name, measurement in fields.items():
            target = accuracy.setdefault(operation, {}).setdefault(dtype, {}).setdefault(
                name, {"cases": 0, "mismatches": 0, "max_abs_error": 0.0, "max_rel_error": 0.0})
            target["cases"] += 1
            target["mismatches"] += measurement.get("mismatches", 0)
            target["max_abs_error"] = max(target["max_abs_error"], measurement.get("max_abs_error", 0.0))
            target["max_rel_error"] = max(target["max_rel_error"], measurement.get("max_rel_error", 0.0))
        if not case["pass"]:
            compact.append(case)
            continue
        absolute = [v["max_abs_error"] for v in fields.values() if "max_abs_error" in v]
        relative = [v["max_rel_error"] for v in fields.values() if "max_rel_error" in v]
        compact.append({"case": case["case"], "pass": True, "checks": len(fields),
                        "max_abs_error": max(absolute, default=0.0),
                        "max_rel_error": max(relative, default=0.0)})
    result = {"format": "preprocess.scanpy-reference.v1", "provenance": provenance,
              "driver_sha256": hashlib.sha256(driver.read_bytes()).hexdigest(),
              "fixture_sha256": hashlib.sha256(x.astype("<f8").tobytes() + groups.tobytes() +
                                                observations.tobytes() + batches.astype("<i4").tobytes()).hexdigest(),
              "tolerances": {"float32": {"rtol": 1e-5, "atol": 1e-6},
                             "float64": {"rtol": 1e-8, "atol": 1e-10}},
              "coverage_fields": {name: sorted(fields) for name, fields in sorted(coverage.items())},
              "accuracy_by_operation_and_input_dtype": accuracy,
              "comparison_notes": [
                  "Scanpy inplace=False retains FP64 HVG normalized dispersions; inplace AnnData annotations cast them to FP32. Native results and annotations retain FP64. Both native FP64 and its FP32 representation are checked.",
                  "Scanpy 1.11.5 derives different automatic normalization targets for dense and CSR inputs with zero rows. Such cases use an independent median-positive oracle for the documented representation-independent native contract.",
                  "Batched HVG score ties use native feature position. Scanpy groups by feature label; fixtures use zero-padded labels that sort in positional order. Scanpy single-batch selection includes ties at the threshold.",
                  "These are bounded seeded software fixtures, not biological validation. Exact masks passed for these cases; this is not a universal identity claim."
              ],
              "case_count": len(cases), "passed": passed, "failed": len(cases) - passed,
              "cases": compact}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"Scanpy {VERSION} reference: {passed}/{len(cases)} cases passed; report: {args.output}")
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
