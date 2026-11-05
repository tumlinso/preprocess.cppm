// Stateless numerical helpers; anndata owns every view and buffer descriptor.
namespace preprocess::detail {

inline void require(bool ok, const char* message) { if (!ok) throw std::invalid_argument(message); }
template<class T> inline T load(const std::byte* p) { T x; std::memcpy(&x,p,sizeof(T)); return x; }
template<class T> inline void store(std::byte* p, T x) { std::memcpy(p,&x,sizeof(T)); }

template<class F> decltype(auto) numeric(anndatapp::DType type, F&& f) {
    using D = anndatapp::DType;
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
template<class F> decltype(auto) floating(anndatapp::DType type, F&& f) {
    if (type == anndatapp::DType::float32) return f.template operator()<float>();
    if (type == anndatapp::DType::float64) return f.template operator()<double>();
    throw std::invalid_argument("destination must have FP32 or FP64 storage");
}
template<class Byte, class F> void indices(const anndatapp::SparseBytesView<Byte>& x, F&& f) {
    if (x.index_size() == 4 && x.offset_size() == 4) return f.template operator()<std::uint32_t,std::uint32_t>();
    if (x.index_size() == 4) return f.template operator()<std::uint32_t,std::uint64_t>();
    if (x.offset_size() == 4) return f.template operator()<std::uint64_t,std::uint32_t>();
    return f.template operator()<std::uint64_t,std::uint64_t>();
}

inline void check_matrix(anndatapp::ArrayView x, bool allow_unsorted = false) {
    require(x.rank() == 2,"preprocessing requires a matrix"); x.validate();
    if (x.layout() == anndatapp::Layout::dense) { (void)x.bytes(); return; }
    auto b = x.sparse_bytes();
    require((b.index_size() == 4 || b.index_size() == 8) && (b.offset_size() == 4 || b.offset_size() == 8),
            "sparse indices and offsets must be 32 or 64 bit");
    auto major = x.layout() == anndatapp::Layout::csr ? b.rows() : b.columns();
    indices(b,[&]<class I, class O>() {
        const auto* ix = b.indices(); const auto* ptr = b.offsets();
        for (std::size_t i = 0; i < major; ++i) {
            auto p = std::size_t(load<O>(ptr+i*sizeof(O))), end = std::size_t(load<O>(ptr+(i+1)*sizeof(O)));
            if (allow_unsorted) {
                std::vector<I> sorted(end-p);
                for (auto k = p; k < end; ++k) sorted[k-p] = load<I>(ix+k*sizeof(I));
                std::sort(sorted.begin(),sorted.end());
                require(std::adjacent_find(sorted.begin(),sorted.end()) == sorted.end(),"materialized sparse selection has duplicate entries");
            } else for (std::size_t k = p+(p != end); k < end; ++k)
                require(load<I>(ix+(k-1)*sizeof(I)) < load<I>(ix+k*sizeof(I)),"sparse input must have sorted unique indices");
        }
    });
}

// Visit stored entries only. The existing anndata view retains their lifetime;
// dtype/index dispatch and descriptor acquisition happen outside element loops.
template<class T, class F> void entries(anndatapp::ArrayView x, F&& f) {
    if (x.layout() == anndatapp::Layout::dense) {
        auto b = x.bytes(); const auto* p = b.data();
        auto rows = b.extent(0), columns = b.extent(1); auto sr = b.stride_bytes(0), sc = b.stride_bytes(1);
        for (std::size_t i = 0; i < rows; ++i)
            for (std::size_t j = 0; j < columns; ++j)
                f(i,j,double(load<T>(p+std::ptrdiff_t(i)*sr+std::ptrdiff_t(j)*sc)),i*columns+j);
    } else {
        auto b = x.sparse_bytes(); bool csr = x.layout() == anndatapp::Layout::csr;
        auto major = csr ? b.rows() : b.columns();
        indices(b,[&]<class I, class O>() {
            const auto* p = b.values(); const auto* ix = b.indices(); const auto* ptr = b.offsets();
            for (std::size_t i = 0; i < major; ++i) {
                auto first = std::size_t(load<O>(ptr+i*sizeof(O))), end = std::size_t(load<O>(ptr+(i+1)*sizeof(O)));
                for (auto k = first; k < end; ++k) {
                    auto j = std::size_t(load<I>(ix+k*sizeof(I)));
                    f(csr ? i : j,csr ? j : i,double(load<T>(p+k*sizeof(T))),k);
                }
            }
        });
    }
}
template<class T> void dense_row(const anndatapp::DenseBytesView<const std::byte>& x, std::size_t i, double* output) {
    const auto* p = x.data()+std::ptrdiff_t(i)*x.stride_bytes(0); auto sc = x.stride_bytes(1); auto n = x.extent(1);
    for (std::size_t j = 0; j < n; ++j) output[j] = double(load<T>(p+std::ptrdiff_t(j)*sc));
}
template<class T, class I, class O> void sparse_row(const anndatapp::SparseBytesView<const std::byte>& x, bool csr, std::size_t i, double* output) {
    auto columns = x.columns(); std::fill_n(output,columns,0.0);
    const auto* p = x.values(); const auto* ix = x.indices(); const auto* ptr = x.offsets();
    if (csr) {
        auto first = std::size_t(load<O>(ptr+i*sizeof(O))), end = std::size_t(load<O>(ptr+(i+1)*sizeof(O)));
        for (auto k = first; k < end; ++k) output[load<I>(ix+k*sizeof(I))] = double(load<T>(p+k*sizeof(T)));
    } else for (std::size_t j = 0; j < columns; ++j) {
        auto lo = std::size_t(load<O>(ptr+j*sizeof(O))), hi = std::size_t(load<O>(ptr+(j+1)*sizeof(O))); const auto end = hi;
        while (lo < hi) { auto k = lo+(hi-lo)/2; load<I>(ix+k*sizeof(I)) < i ? lo = k+1 : hi = k; }
        if (lo < end && load<I>(ix+lo*sizeof(I)) == i) output[j] = double(load<T>(p+lo*sizeof(T)));
    }
}

inline std::vector<std::uint8_t> mask(std::optional<anndatapp::ArrayView> x, std::size_t n) {
    std::vector<std::uint8_t> out(n,1); if (!x) return out;
    require(x->rank() == 1 && x->extent(0) == n && x->layout() == anndatapp::Layout::dense,"mask must be a dense vector with matching length");
    x->validate(); auto b = x->bytes(); const auto* p = b.data(); auto stride = b.stride_bytes(0);
    numeric(x->dtype(),[&]<class T>() {
        for (std::size_t i = 0; i < n; ++i) {
            auto v = double(load<T>(p+std::ptrdiff_t(i)*stride));
            require(v == 0 || v == 1,"mask values must be zero or one"); out[i] = v != 0;
        }
    });
    return out;
}
template<class T> anndatapp::Array array(const std::vector<T>& values) {
    auto out = anndatapp::Array::empty<T>({values.size()});
    std::copy(values.begin(),values.end(),out.edit().template dense<T>().data()); return out;
}
inline anndatapp::Array booleans(const std::vector<std::uint8_t>& values) {
    auto out = anndatapp::Array::empty<bool>({values.size()}); auto* p = out.edit().dense<bool>().data();
    for (std::size_t i = 0; i < values.size(); ++i) p[i] = values[i] != 0; return out;
}
inline std::vector<double> doubles(anndatapp::ArrayView a, std::size_t n) {
    require(a.rank() == 1 && a.extent(0) == n && a.layout() == anndatapp::Layout::dense,"expected matching numerical vector");
    a.validate(); auto b = a.bytes(); const auto* p = b.data(); auto stride = b.stride_bytes(0); std::vector<double> out(n);
    numeric(a.dtype(),[&]<class T>() { for (std::size_t i = 0; i < n; ++i) out[i] = double(load<T>(p+std::ptrdiff_t(i)*stride)); }); return out;
}

} // namespace preprocess::detail
