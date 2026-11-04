#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <new>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <vector>

import preprocess;

// Count requested C++ allocation bytes only during the operation. RSS includes
// preparation and allocator retention; these counters do not count direct mmap.
static std::atomic<bool> counting{false};
static std::atomic<std::size_t> calls{0}, bytes{0};
static void count(std::size_t n) {
    if (!counting.load(std::memory_order_relaxed)) return;
    calls.fetch_add(1, std::memory_order_relaxed);
    bytes.fetch_add(n, std::memory_order_relaxed);
}
void* operator new(std::size_t n) {
    void* p = std::malloc(n ? n : 1);
    if (!p) throw std::bad_alloc();
    count(n); return p;
}
void* operator new[](std::size_t n) { return ::operator new(n); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }
void* operator new(std::size_t n, std::align_val_t a) {
    void* p = nullptr;
    if (posix_memalign(&p, std::size_t(a), n ? n : 1)) throw std::bad_alloc();
    count(n); return p;
}
void* operator new[](std::size_t n, std::align_val_t a) { return ::operator new(n, a); }
void operator delete(void* p, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void* p, std::align_val_t) noexcept { std::free(p); }
void operator delete(void* p, std::size_t, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t, std::align_val_t) noexcept { std::free(p); }

static long peak_rss() { rusage r{}; getrusage(RUSAGE_SELF, &r); return r.ru_maxrss; }

template<class T>
static anndata::Array input(std::size_t n, std::size_t m, bool sparse, std::size_t& nnz) {
    std::minstd_rand rng(1729);
    std::uniform_real_distribution<double> u(0, 1);
    std::vector<T> values;
    std::vector<std::int32_t> indices;
    std::vector<std::int64_t> offsets(n + 1);
    anndata::Array x;
    if (!sparse) x = anndata::Array::zeros<T>({n, m});
    T* dense = sparse ? nullptr : x.edit().template dense<T>().data();
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < m; ++j) {
            if (u(rng) > .03 + .17 * double(j % 53) / 52) continue;
            T v = T(1 + (rng() % (3 + j % 37)));
            ++nnz;
            if (sparse) { values.push_back(v); indices.push_back(std::int32_t(j)); }
            else dense[i*m + j] = v;
        }
        offsets[i + 1] = std::int64_t(values.size());
    }
    if (!sparse) return x;
    x = anndata::Array::csr<T, std::int32_t, std::int64_t>(n, m, nnz);
    auto s = x.edit().template csr<T, std::int32_t, std::int64_t>();
    for (std::size_t k = 0; k < nnz; ++k) { s.values()[k] = values[k]; s.unsafe_indices()[k] = indices[k]; }
    for (std::size_t i = 0; i <= n; ++i) s.unsafe_offsets()[i] = offsets[i];
    return x;
}

template<class T>
static int run(const std::string& op, const std::string& layout, std::size_t n, std::size_t m) {
    bool sparse = layout == "csr";
    std::size_t nnz = 0;
    auto x = input<T>(n, m, sparse, nnz);
    if (op == "hvg") {
        preprocess::normalize_total(x.edit(), {.target_sum = 1e4});
        preprocess::log1p(x.edit());
    }
    std::optional<preprocess::QcResult> qc;
    std::optional<preprocess::FeatureStats> stats;
    std::optional<preprocess::Normalization> norm;
    std::optional<preprocess::Scaling> scale;
    std::optional<preprocess::HvgResult> hvg;
    long before = peak_rss();
    auto t = std::chrono::steady_clock::now();
    counting.store(true, std::memory_order_relaxed);
    if (op == "qc") qc.emplace(preprocess::qc(x.view()));
    else if (op == "stats") stats.emplace(preprocess::feature_stats(x.view()));
    else if (op == "normalize") norm.emplace(preprocess::normalize_total(x.edit(), {.target_sum = 1e4}));
    else if (op == "log1p") preprocess::log1p(x.edit());
    else if (op == "scale") scale.emplace(preprocess::scale(x.edit(), {.zero_center = !sparse}));
    else if (op == "hvg") hvg.emplace(preprocess::highly_variable_genes(x.view(), {.n_top_genes = m / 4}));
    else if (op != "baseline") throw std::invalid_argument("unknown operation");
    counting.store(false, std::memory_order_relaxed);
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
    auto allocated_calls = calls.load(), allocated_bytes = bytes.load();
    long peak = peak_rss();
    std::size_t input_bytes = sparse ? nnz*(sizeof(T) + sizeof(std::int32_t)) + (n + 1)*sizeof(std::int64_t) : n*m*sizeof(T);
    std::cout << std::setprecision(12) << "{\"operation\":\"" << op << "\",\"layout\":\"" << layout
              << "\",\"dtype\":\"" << (sizeof(T) == 4 ? "float32" : "float64") << "\",\"rows\":" << n
              << ",\"columns\":" << m << ",\"seed\":1729,\"nnz\":" << nnz << ",\"input_bytes\":" << input_bytes
              << ",\"milliseconds\":" << ms << ",\"allocation_calls\":" << allocated_calls
              << ",\"requested_allocation_bytes\":" << allocated_bytes << ",\"baseline_peak_rss_kib\":" << before
              << ",\"peak_rss_kib\":" << peak << ",\"incremental_high_water_kib\":" << peak - before << "}\n";
    return 0;
}

int main(int argc, char** argv) {
    try {
        if (argc != 6) throw std::invalid_argument("usage: preprocess_benchmark OP dense|csr float32|float64 ROWS COLS");
        std::string layout = argv[2], dtype = argv[3];
        std::size_t n = std::stoull(argv[4]), m = std::stoull(argv[5]);
        if ((layout != "dense" && layout != "csr") || !n || m < 20 || n > 100000 || m > 100000 || n > 100000000 / m)
            throw std::invalid_argument("invalid layout or bounded geometry");
        if (dtype == "float32") return run<float>(argv[1], layout, n, m);
        if (dtype == "float64") return run<double>(argv[1], layout, n, m);
        throw std::invalid_argument("invalid dtype");
    } catch (const std::exception& e) { counting = false; std::cerr << e.what() << '\n'; return 1; }
}
