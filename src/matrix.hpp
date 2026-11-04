// Included after `module preprocess;`; standard headers belong to the global fragment.
namespace preprocess::detail {

inline void require(bool ok, const char* message) {
    if (!ok) throw std::invalid_argument(message);
}
template<class T> inline T load(const std::byte* p) {
    T x; std::memcpy(&x, p, sizeof(T)); return x;
}
template<class T> inline void store(std::byte* p, T x) { std::memcpy(p, &x, sizeof(T)); }

template<class F> decltype(auto) numeric(anndata::DType type, F&& f) {
    using D = anndata::DType;
    switch (type) {
        case D::boolean: return f.template operator()<bool>();
        case D::int8: return f.template operator()<std::int8_t>();
        case D::uint8: return f.template operator()<std::uint8_t>();
        case D::int16: return f.template operator()<std::int16_t>();
        case D::uint16: return f.template operator()<std::uint16_t>();
        case D::int32: return f.template operator()<std::int32_t>();
        case D::uint32: return f.template operator()<std::uint32_t>();
        case D::int64: return f.template operator()<std::int64_t>();
        case D::uint64: return f.template operator()<std::uint64_t>();
        case D::float32: return f.template operator()<float>();
        case D::float64: return f.template operator()<double>();
    }
    throw std::invalid_argument("unsupported numerical dtype");
}
template<class F> decltype(auto) floating(anndata::DType type, F&& f) {
    if (type == anndata::DType::float32) return f.template operator()<float>();
    if (type == anndata::DType::float64) return f.template operator()<double>();
    throw std::invalid_argument("destination must have FP32 or FP64 storage");
}

struct Matrix {
    anndata::ArrayView owner;
    anndata::Layout layout;
    std::size_t rows, columns, nnz = 0, width, iw = 0, ow = 0;
    std::ptrdiff_t sr = 0, sc = 0;
    const std::byte* values = nullptr;
    const std::byte* indices = nullptr;
    const std::byte* offsets = nullptr;

    inline explicit Matrix(anndata::ArrayView a, bool allow_unsorted = false) : owner(a), layout(a.layout()), rows(0), columns(0), width(anndata::dtype_size(a.dtype())) {
        require(a.rank() == 2, "preprocessing requires a matrix");
        a.validate();
        rows = a.extent(0); columns = a.extent(1);
        if (layout == anndata::Layout::dense) {
            auto b = a.bytes(); values = b.data(); sr = b.stride_bytes(0); sc = b.stride_bytes(1);
        } else {
            auto b = a.sparse_bytes(); values = b.values(); indices = b.indices(); offsets = b.offsets();
            nnz = b.nnz(); iw = b.index_size(); ow = b.offset_size();
            require((iw == 4 || iw == 8) && (ow == 4 || ow == 8), "sparse indices and offsets must be 32 or 64 bit");
            structural([&]<class I, class O>() {
                auto major = layout == anndata::Layout::csr ? rows : columns;
                for (std::size_t i = 0; i < major; ++i) {
                    auto p = std::size_t(load<O>(offsets + i*sizeof(O))), end = std::size_t(load<O>(offsets + (i+1)*sizeof(O)));
                    if (allow_unsorted) {
                        std::vector<I> sorted(end-p);
                        for (auto k = p; k < end; ++k) sorted[k-p] = load<I>(indices+k*sizeof(I));
                        std::sort(sorted.begin(),sorted.end());
                        require(std::adjacent_find(sorted.begin(),sorted.end()) == sorted.end(), "materialized sparse selection has duplicate entries");
                    } else for (std::size_t k = p + (p != end); k < end; ++k)
                        require(load<I>(indices + (k-1)*sizeof(I)) < load<I>(indices + k*sizeof(I)), "sparse input must have sorted unique indices");
                }
            });
        }
    }
    template<class F> void structural(F&& f) const {
        if (iw == 4 && ow == 4) return f.template operator()<std::uint32_t, std::uint32_t>();
        if (iw == 4) return f.template operator()<std::uint32_t, std::uint64_t>();
        if (ow == 4) return f.template operator()<std::uint64_t, std::uint32_t>();
        return f.template operator()<std::uint64_t, std::uint64_t>();
    }
    template<class T, class F> void each(F&& f) const {
        if (layout == anndata::Layout::dense) {
            for (std::size_t i = 0; i < rows; ++i)
                for (std::size_t j = 0; j < columns; ++j)
                    f(i, j, double(load<T>(values + std::ptrdiff_t(i)*sr + std::ptrdiff_t(j)*sc)), i*columns+j);
        } else structural([&]<class I, class O>() {
            auto major = layout == anndata::Layout::csr ? rows : columns;
            for (std::size_t i = 0; i < major; ++i) {
                auto p = std::size_t(load<O>(offsets + i*sizeof(O))), end = std::size_t(load<O>(offsets + (i+1)*sizeof(O)));
                for (std::size_t k = p; k < end; ++k) {
                    auto j = std::size_t(load<I>(indices + k*sizeof(I)));
                    f(layout == anndata::Layout::csr ? i : j, layout == anndata::Layout::csr ? j : i,
                      double(load<T>(values + k*sizeof(T))), k);
                }
            }
        });
    }
    template<class T> void row(std::size_t i, double* output) const {
        std::fill_n(output, columns, 0.0);
        if (layout == anndata::Layout::dense) {
            for (std::size_t j = 0; j < columns; ++j) output[j] = double(load<T>(values + std::ptrdiff_t(i)*sr + std::ptrdiff_t(j)*sc));
        } else structural([&]<class I, class O>() {
            if (layout == anndata::Layout::csr) {
                auto p = std::size_t(load<O>(offsets + i*sizeof(O))), end = std::size_t(load<O>(offsets + (i+1)*sizeof(O)));
                for (auto k = p; k < end; ++k) output[load<I>(indices+k*sizeof(I))] = double(load<T>(values+k*sizeof(T)));
            } else for (std::size_t j = 0; j < columns; ++j) {
                auto lo = std::size_t(load<O>(offsets+j*sizeof(O))), hi = std::size_t(load<O>(offsets+(j+1)*sizeof(O)));
                const auto end = hi;
                while (lo < hi) { auto k = lo+(hi-lo)/2; load<I>(indices+k*sizeof(I)) < i ? lo = k+1 : hi = k; }
                if (lo < end && load<I>(indices+lo*sizeof(I)) == i) output[j] = double(load<T>(values+lo*sizeof(T)));
            }
        });
    }
    template<class T> void write(std::byte* p, std::size_t i, std::size_t j, std::size_t k, double v) const {
        auto offset = layout == anndata::Layout::dense ? std::ptrdiff_t(i)*sr + std::ptrdiff_t(j)*sc : std::ptrdiff_t(k)*std::ptrdiff_t(sizeof(T));
        store<T>(p+offset,T(v));
    }
};

inline std::vector<std::uint8_t> mask(std::optional<anndata::ArrayView> x, std::size_t n) {
    std::vector<std::uint8_t> out(n, 1);
    if (!x) return out;
    require(x->rank() == 1 && x->extent(0) == n && x->layout() == anndata::Layout::dense, "mask must be a dense vector with matching length");
    x->validate();
    auto b = x->bytes();
    numeric(x->dtype(), [&]<class T>() {
        for (std::size_t i = 0; i < n; ++i) {
            auto v = double(load<T>(b.data()+std::ptrdiff_t(i)*b.stride_bytes(0)));
            require(v == 0 || v == 1, "mask values must be zero or one"); out[i] = v != 0;
        }
    });
    return out;
}
template<class T> anndata::Array array(const std::vector<T>& values) {
    auto out = anndata::Array::empty<T>({values.size()});
    std::copy(values.begin(), values.end(), out.edit().template dense<T>().data()); return out;
}
inline anndata::Array booleans(const std::vector<std::uint8_t>& values) {
    auto out = anndata::Array::empty<bool>({values.size()}); auto* p = out.edit().dense<bool>().data();
    for (std::size_t i = 0; i < values.size(); ++i) p[i] = values[i] != 0;
    return out;
}
inline std::vector<double> doubles(anndata::ArrayView a, std::size_t n) {
    require(a.rank() == 1 && a.extent(0) == n && a.layout() == anndata::Layout::dense, "expected matching numerical vector");
    a.validate();
    auto b = a.bytes(); std::vector<double> out(n);
    numeric(a.dtype(), [&]<class T>() { for (std::size_t i = 0; i < n; ++i) out[i] = double(load<T>(b.data()+std::ptrdiff_t(i)*b.stride_bytes(0))); });
    return out;
}

} // namespace preprocess::detail
