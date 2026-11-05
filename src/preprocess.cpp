module;
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

module preprocess;
#include "numeric.hpp"

namespace preprocess {
namespace {
using namespace detail;

void vector_shape(const anndatapp::Array& a, std::size_t n) {
    require(a.rank() == 1 && a.extent(0) == n, "annotation vector does not match dataset axis");
}
bool present(const anndatapp::Array& a) { return a.rank() != 0; }
std::uint64_t magnitude(std::ptrdiff_t n) { return n < 0 ? std::uint64_t(-(n+1))+1 : std::uint64_t(n); }
void writable(anndatapp::ArrayView x) {
    if (x.layout() != anndatapp::Layout::dense || !x.extent(0) || !x.extent(1)) return;
    auto b = x.bytes(); auto rows = b.extent(0), columns = b.extent(1);
    auto r = magnitude(b.stride_bytes(0)), c = magnitude(b.stride_bytes(1)), g = std::gcd(r,c);
    require(rows <= 1 || r != 0, "writable matrix rows overlap");
    require(columns <= 1 || c != 0, "writable matrix columns overlap");
    require(!g || !(c/g < rows && r/g < columns), "writable matrix elements overlap");
}
std::pair<std::uintptr_t,std::uintptr_t> interval(anndatapp::ArrayView x) {
    if (!x.extent(0) || !x.extent(1)) return {0,0};
    if (x.layout() != anndatapp::Layout::dense) {
        auto b = x.sparse_bytes(); if (!b.nnz()) return {0,0};
        auto p = std::uintptr_t(b.values()); return {p,p+b.nnz()*b.value_size()};
    }
    auto b = x.bytes(); auto p = std::uintptr_t(b.data());
    auto r = std::ptrdiff_t(b.extent(0)-1)*b.stride_bytes(0), c = std::ptrdiff_t(b.extent(1)-1)*b.stride_bytes(1);
    return {p+std::min<std::ptrdiff_t>(r,0)+std::min<std::ptrdiff_t>(c,0),
            p+std::max<std::ptrdiff_t>(r,0)+std::max<std::ptrdiff_t>(c,0)+b.element_size()};
}
bool same_storage(anndatapp::ArrayView a, anndatapp::ArrayView b) {
    if (a.layout() != b.layout() || a.dtype() != b.dtype() || a.extent(0) != b.extent(0) || a.extent(1) != b.extent(1)) return false;
    if (a.layout() == anndatapp::Layout::dense) {
        auto x = a.bytes(), y = b.bytes();
        return x.data() == y.data() && x.stride_bytes(0) == y.stride_bytes(0) && x.stride_bytes(1) == y.stride_bytes(1);
    }
    auto x = a.sparse_bytes(), y = b.sparse_bytes();
    return x.values() == y.values() && x.indices() == y.indices() && x.offsets() == y.offsets();
}
void output_contract(anndatapp::ArrayView a, anndatapp::ArrayView b) {
    check_matrix(b);
    require(a.extent(0) == b.extent(0) && a.extent(1) == b.extent(1), "destination shape differs from source");
    require(b.dtype() == anndatapp::DType::float32 || b.dtype() == anndatapp::DType::float64, "destination must have FP32 or FP64 storage");
    writable(b); auto [a0,a1] = interval(a); auto [b0,b1] = interval(b);
    require(a1 <= b0 || b1 <= a0 || same_storage(a,b), "source and destination partially overlap");
    if (b.layout() != anndatapp::Layout::dense) {
        require(a.layout() == b.layout(), "sparse destination layout differs from source");
        auto x = a.sparse_bytes(), y = b.sparse_bytes();
        require(x.nnz() == y.nnz() && x.index_size() == y.index_size() && x.offset_size() == y.offset_size(),
                "sparse destination must have the same structure widths");
        auto major = a.layout() == anndatapp::Layout::csr ? x.rows() : x.columns();
        require((!x.nnz() || std::memcmp(x.indices(),y.indices(),x.nnz()*x.index_size()) == 0)
                && std::memcmp(x.offsets(),y.offsets(),(major+1)*x.offset_size()) == 0, "sparse destination structure differs from source");
    }
}
template<class F> void transform_into(anndatapp::ArrayView a, anndatapp::ArrayEdit output, F&& f) {
    auto b = output.view(); output_contract(a,b); bool dense = b.layout() == anndatapp::Layout::dense;
    std::byte* p; std::ptrdiff_t sr = 0, sc = 0;
    if (dense) { auto d = output.bytes(); p = d.data(); sr = d.stride_bytes(0); sc = d.stride_bytes(1); }
    else p = output.sparse_bytes().values();
    auto rows = b.extent(0), columns = b.extent(1);
    floating(b.dtype(), [&]<class U>() {
        if (dense && a.layout() != anndatapp::Layout::dense)
            for (std::size_t i = 0; i < rows; ++i) for (std::size_t j = 0; j < columns; ++j)
                store<U>(p+std::ptrdiff_t(i)*sr+std::ptrdiff_t(j)*sc,U(f(i,j,0)));
        numeric(a.dtype(), [&]<class T>() {
            entries<T>(a,[&](std::size_t i, std::size_t j, double v, std::size_t k) {
                auto offset = dense ? std::ptrdiff_t(i)*sr+std::ptrdiff_t(j)*sc : std::ptrdiff_t(k)*std::ptrdiff_t(sizeof(U));
                store<U>(p+offset,U(f(i,j,v)));
            });
        });
    });
}
void copy_values(anndatapp::ArrayView x, anndatapp::ArrayEdit output) {
    auto y = output.view(); output_contract(x,y);
    if (same_storage(x,y)) return;
    bool dense = y.layout() == anndatapp::Layout::dense;
    if (dense && x.layout() == anndatapp::Layout::dense && x.dtype() == y.dtype()) { x.copy_into(output); return; }
    std::byte* p; std::ptrdiff_t sr = 0, sc = 0;
    if (dense) { auto b = output.bytes(); p = b.data(); sr = b.stride_bytes(0); sc = b.stride_bytes(1); }
    else p = output.sparse_bytes().values();
    auto rows = y.extent(0), columns = y.extent(1);
    floating(y.dtype(),[&]<class U>() {
        if (dense && x.layout() != anndatapp::Layout::dense)
            for (std::size_t i = 0; i < rows; ++i) for (std::size_t j = 0; j < columns; ++j)
                store<U>(p+std::ptrdiff_t(i)*sr+std::ptrdiff_t(j)*sc,U(0));
        numeric(x.dtype(),[&]<class T>() {
            entries<T>(x,[&](std::size_t i, std::size_t j, double v, std::size_t k) {
                auto offset = dense ? std::ptrdiff_t(i)*sr+std::ptrdiff_t(j)*sc : std::ptrdiff_t(k)*std::ptrdiff_t(sizeof(U));
                store<U>(p+offset,U(v));
            });
        });
    });
}
template<class F> void transform(anndatapp::ArrayEdit a, F&& f) {
    auto x = a.view(); check_matrix(x); writable(x);
    floating(x.dtype(), [&]<class T>() {
        if (x.layout() == anndatapp::Layout::dense) {
            auto b = a.bytes(); auto* p = b.data(); auto rows = b.extent(0), columns = b.extent(1);
            auto sr = b.stride_bytes(0), sc = b.stride_bytes(1);
            for (std::size_t i = 0; i < rows; ++i) for (std::size_t j = 0; j < columns; ++j) {
                auto* q = p+std::ptrdiff_t(i)*sr+std::ptrdiff_t(j)*sc; store<T>(q,T(f(i,j,double(load<T>(q)))));
            }
        } else {
            auto b = a.sparse_bytes(); auto* p = b.values();
            entries<T>(x,[&](std::size_t i, std::size_t j, double v, std::size_t k) { store<T>(p+k*sizeof(T),T(f(i,j,v))); });
        }
    });
}
anndatapp::Array filter(const AxisSummary& a, const FilterBounds& b) {
    auto n = a.total.extent(0);
    auto totals = doubles(a.total.view(),n), counts = doubles(a.detected.view(),n);
    require(!b.min_counts || std::isfinite(*b.min_counts), "minimum count must be finite");
    require(!b.max_counts || std::isfinite(*b.max_counts), "maximum count must be finite");
    require(!b.min_counts || !b.max_counts || *b.min_counts <= *b.max_counts, "count bounds are reversed");
    require(!b.min_detected || !b.max_detected || *b.min_detected <= *b.max_detected, "detection bounds are reversed");
    std::vector<std::uint8_t> keep(n);
    for (std::size_t i = 0; i < n; ++i)
        keep[i] = (!b.min_counts || totals[i] >= *b.min_counts) && (!b.max_counts || totals[i] <= *b.max_counts)
            && (!b.min_detected || counts[i] >= double(*b.min_detected)) && (!b.max_detected || counts[i] <= double(*b.max_detected));
    return booleans(keep);
}
double log_divisor(std::optional<double> base) {
    require(!base || (std::isfinite(*base) && *base > 0 && *base != 1), "logarithm base must be positive, finite, and different from one");
    return base ? std::log(*base) : 1;
}
anndatapp::Array log_array(const std::vector<double>& x) {
    std::vector<double> y(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) y[i] = std::log1p(x[i]);
    return array(y);
}
Normalization factors(anndatapp::ArrayView x, const NormalizationOptions& options) {
    auto rows = x.extent(0), columns = x.extent(1);
    require(!options.target_sum || (std::isfinite(*options.target_sum) && *options.target_sum > 0), "normalization target must be positive and finite");
    require(std::isfinite(options.max_fraction) && options.max_fraction >= 0 && options.max_fraction <= 1, "max_fraction must be in [0,1]");
    std::vector<double> totals(rows,0);
    std::vector<std::uint8_t> included(columns,1);
    numeric(x.dtype(), [&]<class T>() {
        entries<T>(x,[&](std::size_t i, std::size_t, double v, std::size_t) {
            require(std::isfinite(v) && v >= 0, "normalization requires finite nonnegative counts"); totals[i] += v;
        });
        if (options.exclude_highly_expressed) {
            entries<T>(x,[&](std::size_t i, std::size_t j, double v, std::size_t) {
                if (v > options.max_fraction*totals[i]) included[j] = 0;
            });
            std::fill(totals.begin(),totals.end(),0);
            entries<T>(x,[&](std::size_t i, std::size_t j, double v, std::size_t) { if (included[j]) totals[i] += v; });
        }
    });
    double target = options.target_sum.value_or(1);
    if (!options.target_sum) {
        std::vector<double> positive;
        for (auto x : totals) if (x > 0) positive.push_back(x);
        if (!positive.empty()) {
            std::sort(positive.begin(),positive.end()); auto k = positive.size()/2;
            target = positive.size()%2 ? positive[k] : .5*positive[k-1]+.5*positive[k];
        }
    }
    for (auto& x : totals) { require(std::isfinite(x), "normalization total overflowed"); x /= target; }
    return {array(totals),booleans(included),target};
}
void scale_options(const ScaleOptions& o) {
    require(!o.max_value || (std::isfinite(*o.max_value) && *o.max_value >= 0), "scale clipping limit must be nonnegative and finite");
}
Scaling scaling(anndatapp::ArrayView x, const ScaleOptions& options) {
    auto stats = feature_stats(x,options.observations);
    auto n = stats.variance.extent(0); auto out = anndatapp::Array::empty<double>({n});
    auto* p = out.edit().dense<double>().data(); auto* v = stats.variance.view().dense<double>().data();
    for (std::size_t j = 0; j < n; ++j) p[j] = v[j] == 0 ? 1 : std::sqrt(v[j]);
    return {std::move(stats.mean),std::move(out)};
}
using Columns = std::vector<std::pair<std::string,anndatapp::Column>>;
void column(Columns& columns, std::string name, anndatapp::Array a) {
    columns.emplace_back(std::move(name),anndatapp::Column::numeric(std::move(a)));
}
void attach_obs(anndatapp::AnnData& a, Columns& columns) { for (auto& [name,c] : columns) a.set_obs_column(std::move(name),std::move(c)); }
void attach_var(anndatapp::AnnData& a, Columns& columns) { for (auto& [name,c] : columns) a.set_var_column(std::move(name),std::move(c)); }
} // namespace

QcResult qc(anndatapp::ArrayView x, const QcOptions& options) {
    check_matrix(x); auto rows = x.extent(0), columns = x.extent(1);
    std::vector<double> cell_total(rows,0), gene_total(columns,0), mean(columns), dropout(columns);
    std::vector<std::uint64_t> cell_detected(rows,0), gene_detected(columns,0);
    std::vector<std::vector<std::uint8_t>> masks;
    std::set<std::string> names;
    for (const auto& g : options.groups) {
        require(!g.name.empty() && names.insert(g.name).second, "QC group names must be nonempty and unique");
        masks.push_back(mask(g.mask,columns));
    }
    std::set<std::size_t> cutoffs;
    for (auto n : options.percent_top)
        require(n && n <= columns && cutoffs.insert(n).second, "QC top counts must be distinct and between one and the number of genes");
    std::vector<std::vector<double>> groups(options.groups.size(),std::vector<double>(rows,0));
    numeric(x.dtype(), [&]<class T>() {
        entries<T>(x,[&](std::size_t i, std::size_t j, double v, std::size_t) {
            require(std::isfinite(v) && v >= 0, "QC requires finite nonnegative counts");
            cell_total[i] += v; gene_total[j] += v;
            cell_detected[i] += v != 0; gene_detected[j] += v != 0;
            for (std::size_t g = 0; g < masks.size(); ++g) if (masks[g][j]) groups[g][i] += v;
        });
    });
    for (std::size_t j = 0; j < columns; ++j) {
        mean[j] = rows ? gene_total[j]/double(rows) : std::numeric_limits<double>::quiet_NaN();
        dropout[j] = rows ? 100*(1-double(gene_detected[j])/double(rows)) : std::numeric_limits<double>::quiet_NaN();
    }
    QcResult out;
    out.cells = {array(cell_total),array(cell_detected)}; out.genes = {array(gene_total),array(gene_detected)};
    out.mean = array(mean); out.dropout = array(dropout); out.percent_top = options.percent_top;
    if (options.log1p) {
        out.log_cell_total = log_array(cell_total); out.log_gene_total = log_array(gene_total); out.log_gene_mean = log_array(mean);
        std::vector<double> detected(cell_detected.begin(),cell_detected.end()); out.log_cell_detected = log_array(detected);
    }
    for (std::size_t g = 0; g < groups.size(); ++g) {
        std::vector<double> percent(rows);
        for (std::size_t i = 0; i < rows; ++i) percent[i] = 100*groups[g][i]/cell_total[i];
        GroupQc result{options.groups[g].name,array(groups[g]),array(percent),{}};
        if (options.log1p) result.log_total = log_array(groups[g]); out.groups.push_back(std::move(result));
    }
    out.top_percent = anndatapp::Array::empty<double>({rows,options.percent_top.size()});
    if (!options.percent_top.empty()) {
        auto* p = out.top_percent.edit().dense<double>().data(); std::vector<double> row(columns);
        auto emit = [&](std::size_t i) {
            std::sort(row.begin(),row.end(),std::greater<double>()); std::partial_sum(row.begin(),row.end(),row.begin());
            for (std::size_t k = 0; k < options.percent_top.size(); ++k)
                p[i*options.percent_top.size()+k] = 100*row[options.percent_top[k]-1]/cell_total[i];
        };
        numeric(x.dtype(), [&]<class T>() {
            if (x.layout() == anndatapp::Layout::dense) {
                auto b = x.bytes();
                for (std::size_t i = 0; i < rows; ++i) { dense_row<T>(b,i,row.data()); emit(i); }
            } else {
                auto b = x.sparse_bytes(); bool csr = x.layout() == anndatapp::Layout::csr;
                indices(b,[&]<class I, class O>() {
                    for (std::size_t i = 0; i < rows; ++i) { sparse_row<T,I,O>(b,csr,i,row.data()); emit(i); }
                });
            }
        });
    }
    return out;
}

anndatapp::Array filter_cells(const QcResult& q, const FilterBounds& bounds) { return filter(q.cells,bounds); }
anndatapp::Array filter_genes(const QcResult& q, const FilterBounds& bounds) { return filter(q.genes,bounds); }
anndatapp::Array filter_cells(anndatapp::ArrayView x, const FilterBounds& bounds) { QcOptions o; o.log1p = false; return filter_cells(qc(x,o),bounds); }
anndatapp::Array filter_genes(anndatapp::ArrayView x, const FilterBounds& bounds) { QcOptions o; o.log1p = false; return filter_genes(qc(x,o),bounds); }

FeatureStats feature_stats(anndatapp::ArrayView x, std::optional<anndatapp::ArrayView> observations) {
    check_matrix(x); auto rows = x.extent(0), columns = x.extent(1); auto selected = mask(observations,rows);
    auto n = std::size_t(std::count(selected.begin(),selected.end(),std::uint8_t(1)));
    std::vector<double> mean(columns,0), variance(columns,0);
    std::vector<std::size_t> stored(columns,0);
    numeric(x.dtype(), [&]<class T>() {
        entries<T>(x,[&](std::size_t i, std::size_t j, double v, std::size_t) {
            require(std::isfinite(v), "feature statistics require finite values");
            if (selected[i]) mean[j] += v;
        });
        for (auto& v : mean) v = n ? v/double(n) : std::numeric_limits<double>::quiet_NaN();
        entries<T>(x,[&](std::size_t i, std::size_t j, double v, std::size_t) {
            if (selected[i]) { variance[j] += (v-mean[j])*(v-mean[j]); ++stored[j]; }
        });
    });
    for (std::size_t j = 0; j < columns; ++j)
        variance[j] = !n ? std::numeric_limits<double>::quiet_NaN()
            : (variance[j]+double(n-stored[j])*mean[j]*mean[j])/double(n > 1 ? n-1 : 1);
    return {array(mean),array(variance),n};
}

Normalization normalize_total(anndatapp::ArrayEdit x, const NormalizationOptions& options) {
    auto view = x.view(); check_matrix(view); floating(view.dtype(), []<class T>() {}); writable(view);
    auto out = factors(view,options); auto f = out.factors.view().dense<double>();
    transform(x,[&](std::size_t i, std::size_t, double v) { return f.data()[i] > 0 ? v/f.data()[i] : v; }); return out;
}
Normalization normalize_total_into(anndatapp::ArrayView x, anndatapp::ArrayEdit destination, const NormalizationOptions& options) {
    check_matrix(x); output_contract(x,destination.view()); auto out = factors(x,options);
    auto f = out.factors.view().dense<double>();
    transform_into(x,destination,[&](std::size_t i, std::size_t, double v) { return f.data()[i] > 0 ? v/f.data()[i] : v; }); return out;
}
void log1p(anndatapp::ArrayEdit x, std::optional<double> base) {
    auto d = log_divisor(base); transform(x,[&](std::size_t, std::size_t, double v) { return std::log1p(v)/d; });
}
void log1p_into(anndatapp::ArrayView x, anndatapp::ArrayEdit destination, std::optional<double> base) {
    auto d = log_divisor(base); check_matrix(x);
    transform_into(x,destination,[&](std::size_t, std::size_t, double v) { return std::log1p(v)/d; });
}

anndatapp::Array floating_copy(anndatapp::ArrayView x, anndatapp::DType type) {
    check_matrix(x); auto rows = x.extent(0), columns = x.extent(1);
    return floating(type,[&]<class T>() -> anndatapp::Array {
        if (x.layout() == anndatapp::Layout::dense) {
            auto out = anndatapp::Array::empty<T>({rows,columns}); copy_values(x,out.edit()); return out;
        }
        auto b = x.sparse_bytes(); auto values = anndatapp::Array::empty<T>({b.nnz()});
        auto copy_structure = [&](anndatapp::DType dtype, const std::byte* src, std::size_t n) {
            return numeric(dtype,[&]<class I>() {
                auto out = anndatapp::Array::empty<I>({n}); if (n) std::memcpy(out.edit().bytes().data(),src,n*sizeof(I)); return out;
            });
        };
        auto ix = copy_structure(x.index_dtype(),b.indices(),b.nnz());
        auto ptr = copy_structure(x.offset_dtype(),b.offsets(),(x.layout() == anndatapp::Layout::csr ? rows : columns)+1);
        auto out = anndatapp::Array::from_sparse_buffers(x.layout(),std::move(values),std::move(ix),std::move(ptr),rows,columns);
        copy_values(x,out.edit()); return out;
    });
}
anndatapp::Array dense_copy(anndatapp::ArrayView x, anndatapp::DType type) {
    floating(type,[]<class T>() {});
    x.validate(); anndatapp::Array retained; bool materialized = false;
    try {
        if (x.layout() == anndatapp::Layout::dense) (void)x.bytes();
        else (void)x.sparse_bytes();
    } catch (const anndatapp::Error& e) {
        if (e.code() != anndatapp::ErrorCode::unsupported_layout && e.code() != anndatapp::ErrorCode::non_contiguous) throw;
        retained = x.copy(); x = retained.view(); materialized = true;
    }
    check_matrix(x,materialized); auto rows = x.extent(0), columns = x.extent(1);
    return floating(type,[&]<class T>() { auto out = anndatapp::Array::empty<T>({rows,columns}); copy_values(x,out.edit()); return out; });
}

void apply_scale(anndatapp::ArrayEdit x, const Scaling& parameters, const ScaleOptions& options) {
    scale_options(options); auto view = x.view(); check_matrix(view); writable(view);
    floating(x.view().dtype(),[]<class T>() {});
    require(!options.zero_center || view.layout() == anndatapp::Layout::dense, "centered sparse scaling requires an explicit dense destination");
    auto columns = view.extent(1); auto mean = doubles(parameters.mean.view(),columns), sd = doubles(parameters.stddev.view(),columns);
    auto selected = mask(options.observations,view.extent(0));
    if (std::find(selected.begin(),selected.end(),std::uint8_t(1)) != selected.end())
        for (std::size_t j = 0; j < columns; ++j)
            require(std::isfinite(mean[j]) && std::isfinite(sd[j]) && sd[j] > 0, "scaling parameters must be finite with positive standard deviation");
    transform(x,[&](std::size_t i, std::size_t j, double v) {
        if (!selected[i]) return v;
        v = (v-(options.zero_center ? mean[j] : 0))/sd[j];
        return !options.max_value ? v : options.zero_center ? std::clamp(v,-*options.max_value,*options.max_value) : std::min(v,*options.max_value);
    });
}
Scaling scale(anndatapp::ArrayEdit x, const ScaleOptions& options) {
    scale_options(options); auto view = x.view(); check_matrix(view); writable(view);
    require(!options.zero_center || view.layout() == anndatapp::Layout::dense, "centered sparse scaling requires an explicit dense destination");
    floating(x.view().dtype(),[]<class T>() {});
    auto out = scaling(x.view(),options); apply_scale(x,out,options); return out;
}
Scaling scale_into(anndatapp::ArrayView x, anndatapp::ArrayEdit destination, const ScaleOptions& options) {
    scale_options(options); check_matrix(x); output_contract(x,destination.view());
    require(!options.zero_center || destination.view().layout() == anndatapp::Layout::dense, "centered sparse scaling requires an explicit dense destination");
    auto out = scaling(x,options); auto columns = x.extent(1); auto mean = doubles(out.mean.view(),columns), sd = doubles(out.stddev.view(),columns);
    auto selected = mask(options.observations,x.extent(0));
    transform_into(x,destination,[&](std::size_t i, std::size_t j, double v) {
        if (!selected[i]) return v;
        v = (v-(options.zero_center ? mean[j] : 0))/sd[j];
        return !options.max_value ? v : options.zero_center ? std::clamp(v,-*options.max_value,*options.max_value) : std::min(v,*options.max_value);
    });
    return out;
}

anndatapp::ArrayView source(const anndatapp::AnnData& data, std::string_view layer) { return layer.empty() ? data.x() : data.layer(layer); }
anndatapp::ArrayEdit edit_source(anndatapp::AnnData& data, std::string_view layer) { return layer.empty() ? data.edit_x() : data.edit_layer(layer); }
anndatapp::Array batch_codes(const anndatapp::AnnData& data, std::string_view column_name) {
    auto c = data.obs_column(column_name); std::vector<std::int64_t> codes(c.size()); std::map<std::string,std::int64_t> labels;
    for (std::size_t i = 0; i < c.size(); ++i) require(c.valid(i), "batch labels must not be missing");
    if (c.kind() == anndatapp::ColumnKind::categorical) {
        for (std::size_t i = 0; i < c.size(); ++i) { codes[i] = c.code(i); require(codes[i] >= 0, "batch labels must not be missing"); }
    } else if (c.kind() == anndatapp::ColumnKind::string) {
        for (std::size_t i = 0; i < c.size(); ++i) {
            auto [it,inserted] = labels.emplace(std::string(c.string(i)),std::int64_t(labels.size())); codes[i] = it->second;
        }
    } else {
        auto values = c.values(); require(values.dtype() != anndatapp::DType::float32 && values.dtype() != anndatapp::DType::float64, "numeric batch labels must be integers");
        values.validate(); auto b = values.bytes(); const auto* p = b.data(); auto stride = b.stride_bytes(0);
        numeric(values.dtype(),[&]<class T>() {
            for (std::size_t i = 0; i < c.size(); ++i) {
                auto v = load<T>(p+std::ptrdiff_t(i)*stride);
                auto label = std::to_string(v); auto [it,inserted] = labels.emplace(std::move(label),std::int64_t(labels.size())); codes[i] = it->second;
            }
        });
    }
    return array(codes);
}

void annotate_qc(anndatapp::AnnData& data, QcResult&& q) {
    auto n = data.n_obs(), p = data.n_vars();
    for (auto* a : {&q.cells.total,&q.cells.detected}) vector_shape(*a,n);
    for (auto* a : {&q.genes.total,&q.genes.detected,&q.mean,&q.dropout}) vector_shape(*a,p);
    for (auto* a : {&q.log_cell_total,&q.log_cell_detected}) if (present(*a)) vector_shape(*a,n);
    for (auto* a : {&q.log_gene_total,&q.log_gene_mean}) if (present(*a)) vector_shape(*a,p);
    std::set<std::string> names;
    for (auto& g : q.groups) {
        require(!g.name.empty() && names.insert(g.name).second, "QC group names must be nonempty and unique");
        vector_shape(g.total,n); vector_shape(g.percent,n); if (present(g.log_total)) vector_shape(g.log_total,n);
    }
    require(q.top_percent.rank() == 2 && q.top_percent.extent(0) == n && q.top_percent.extent(1) == q.percent_top.size(), "QC top fractions have the wrong shape");
    std::set<std::size_t> top;
    for (auto k : q.percent_top) require(k && k <= p && top.insert(k).second, "QC top counts are invalid or duplicate");
    Columns obs, var;
    column(obs,"total_counts",std::move(q.cells.total)); column(obs,"n_genes_by_counts",std::move(q.cells.detected));
    column(var,"total_counts",std::move(q.genes.total)); column(var,"n_cells_by_counts",std::move(q.genes.detected));
    column(var,"mean_counts",std::move(q.mean)); column(var,"pct_dropout_by_counts",std::move(q.dropout));
    if (present(q.log_cell_total)) column(obs,"log1p_total_counts",std::move(q.log_cell_total));
    if (present(q.log_cell_detected)) column(obs,"log1p_n_genes_by_counts",std::move(q.log_cell_detected));
    if (present(q.log_gene_total)) column(var,"log1p_total_counts",std::move(q.log_gene_total));
    if (present(q.log_gene_mean)) column(var,"log1p_mean_counts",std::move(q.log_gene_mean));
    for (auto& g : q.groups) {
        column(obs,"total_counts_"+g.name,std::move(g.total)); column(obs,"pct_counts_"+g.name,std::move(g.percent));
        if (present(g.log_total)) column(obs,"log1p_total_counts_"+g.name,std::move(g.log_total));
    }
    auto values = q.top_percent.view().dense<double>();
    for (std::size_t k = 0; k < q.percent_top.size(); ++k) {
        auto a = anndatapp::Array::empty<double>({n}); auto* dst = a.edit().dense<double>().data();
        for (std::size_t i = 0; i < n; ++i) dst[i] = values.data()[std::ptrdiff_t(i)*values.stride(0)+std::ptrdiff_t(k)*values.stride(1)];
        column(obs,"pct_counts_in_top_"+std::to_string(q.percent_top[k])+"_genes",std::move(a));
    }
    attach_obs(data,obs); attach_var(data,var);
}
void annotate_normalization(anndatapp::AnnData& data, Normalization&& result, std::string_view key) {
    vector_shape(result.factors,data.n_obs()); vector_shape(result.included_genes,data.n_vars());
    require(!key.empty(), "normalization annotation key must not be empty");
    data.set_obs_column(std::string(key),anndatapp::Column::numeric(std::move(result.factors)));
}
void annotate_log1p(anndatapp::AnnData& data, std::optional<double> base) {
    log_divisor(base); auto e = anndatapp::Element::mapping();
    e.set("base",base ? anndatapp::Element(*base) : anndatapp::Element(nullptr)); data.set_uns("log1p",std::move(e));
}
void annotate_scale(anndatapp::AnnData& data, Scaling&& parameters) {
    vector_shape(parameters.mean,data.n_vars()); vector_shape(parameters.stddev,data.n_vars());
    Columns columns; column(columns,"mean",std::move(parameters.mean)); column(columns,"std",std::move(parameters.stddev)); attach_var(data,columns);
}
void annotate_hvg(anndatapp::AnnData& data, HvgResult&& result) {
    auto p = data.n_vars();
    for (auto* a : {&result.mean,&result.dispersion,&result.normalized_dispersion,&result.selected}) vector_shape(*a,p);
    require(present(result.nbatches) == present(result.intersection), "HVG batch annotations must occur together");
    if (present(result.nbatches)) { vector_shape(result.nbatches,p); vector_shape(result.intersection,p); }
    Columns columns;
    column(columns,"means",std::move(result.mean)); column(columns,"dispersions",std::move(result.dispersion));
    column(columns,"dispersions_norm",std::move(result.normalized_dispersion)); column(columns,"highly_variable",std::move(result.selected));
    if (present(result.nbatches)) {
        column(columns,"highly_variable_nbatches",std::move(result.nbatches)); column(columns,"highly_variable_intersection",std::move(result.intersection));
    }
    auto e = anndatapp::Element::mapping(); e.set("flavor",anndatapp::Element(result.method == HvgMethod::seurat ? "seurat" : "cell_ranger"));
    attach_var(data,columns); data.set_uns("hvg",std::move(e));
}

} // namespace preprocess
