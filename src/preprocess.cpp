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
#include "matrix.hpp"

namespace preprocess {
namespace {
using namespace detail;

void vector_shape(const anndata::Array& a, std::size_t n) {
    require(a.rank() == 1 && a.extent(0) == n, "annotation vector does not match dataset axis");
}
bool present(const anndata::Array& a) { return a.rank() != 0; }
std::uint64_t magnitude(std::ptrdiff_t n) { return n < 0 ? std::uint64_t(-(n+1))+1 : std::uint64_t(n); }
void writable(const Matrix& m) {
    if (m.layout != anndata::Layout::dense || !m.rows || !m.columns) return;
    auto r = magnitude(m.sr), c = magnitude(m.sc), g = std::gcd(r, c);
    require(m.rows <= 1 || r != 0, "writable matrix rows overlap");
    require(m.columns <= 1 || c != 0, "writable matrix columns overlap");
    require(!g || !(c/g < m.rows && r/g < m.columns), "writable matrix elements overlap");
}
std::pair<std::uintptr_t, std::uintptr_t> interval(const Matrix& m) {
    if (!m.rows || !m.columns || (m.layout != anndata::Layout::dense && !m.nnz)) return {0,0};
    auto p = std::uintptr_t(m.values);
    if (m.layout != anndata::Layout::dense) return {p,p+m.nnz*m.width};
    auto r = std::ptrdiff_t(m.rows-1)*m.sr, c = std::ptrdiff_t(m.columns-1)*m.sc;
    return {p+std::min<std::ptrdiff_t>(r,0)+std::min<std::ptrdiff_t>(c,0),
            p+std::max<std::ptrdiff_t>(r,0)+std::max<std::ptrdiff_t>(c,0)+m.width};
}
bool same_storage(const Matrix& a, const Matrix& b) {
    return a.values == b.values && a.layout == b.layout && a.width == b.width && a.sr == b.sr && a.sc == b.sc
        && a.rows == b.rows && a.columns == b.columns && a.owner.dtype() == b.owner.dtype()
        && (a.layout == anndata::Layout::dense || (a.indices == b.indices && a.offsets == b.offsets));
}
void output_contract(const Matrix& a, const Matrix& b) {
    require(a.rows == b.rows && a.columns == b.columns, "destination shape differs from source");
    require(b.owner.dtype() == anndata::DType::float32 || b.owner.dtype() == anndata::DType::float64,
            "destination must have FP32 or FP64 storage");
    writable(b);
    auto [a0,a1] = interval(a); auto [b0,b1] = interval(b);
    require(a1 <= b0 || b1 <= a0 || same_storage(a,b), "source and destination partially overlap");
    if (b.layout != anndata::Layout::dense) {
        require(a.layout == b.layout && a.nnz == b.nnz && a.iw == b.iw && a.ow == b.ow,
                "sparse destination must have the same layout and structure widths");
        auto major = a.layout == anndata::Layout::csr ? a.rows : a.columns;
        require((!a.nnz || std::memcmp(a.indices,b.indices,a.nnz*a.iw) == 0)
                && std::memcmp(a.offsets,b.offsets,(major+1)*a.ow) == 0,
                "sparse destination structure differs from source");
    }
}
void copy_values(const Matrix& a, anndata::ArrayEdit output) {
    Matrix b(output.view()); output_contract(a,b);
    if (same_storage(a,b)) return;
    auto* p = b.layout == anndata::Layout::dense ? output.bytes().data() : output.sparse_bytes().values();
    floating(b.owner.dtype(), [&]<class U>() {
        if (b.layout == anndata::Layout::dense)
            for (std::size_t i = 0; i < b.rows; ++i)
                for (std::size_t j = 0; j < b.columns; ++j) b.write<U>(p,i,j,0,0);
        numeric(a.owner.dtype(), [&]<class T>() {
            a.each<T>([&](std::size_t i, std::size_t j, double v, std::size_t k) { b.write<U>(p,i,j,k,v); });
        });
    });
}
template<class F> void transform_into(const Matrix& a, anndata::ArrayEdit output, F&& f) {
    Matrix b(output.view()); output_contract(a,b);
    auto* p = b.layout == anndata::Layout::dense ? output.bytes().data() : output.sparse_bytes().values();
    floating(b.owner.dtype(), [&]<class U>() {
        if (a.layout != anndata::Layout::dense && b.layout == anndata::Layout::dense)
            for (std::size_t i = 0; i < b.rows; ++i)
                for (std::size_t j = 0; j < b.columns; ++j) b.write<U>(p,i,j,0,f(i,j,0));
        numeric(a.owner.dtype(), [&]<class T>() {
            a.each<T>([&](std::size_t i, std::size_t j, double v, std::size_t k) { b.write<U>(p,i,j,k,f(i,j,v)); });
        });
    });
}
template<class F> void transform(anndata::ArrayEdit a, F&& f) {
    Matrix m(a.view()); writable(m);
    auto* p = m.layout == anndata::Layout::dense ? a.bytes().data() : a.sparse_bytes().values();
    floating(m.owner.dtype(), [&]<class T>() {
        m.each<T>([&](std::size_t i, std::size_t j, double v, std::size_t k) { m.write<T>(p,i,j,k,f(i,j,v)); });
    });
}
anndata::Array filter(const AxisSummary& a, const FilterBounds& b) {
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
anndata::Array log_array(const std::vector<double>& x) {
    std::vector<double> y(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) y[i] = std::log1p(x[i]);
    return array(y);
}
Normalization factors(const Matrix& m, const NormalizationOptions& options) {
    require(!options.target_sum || (std::isfinite(*options.target_sum) && *options.target_sum > 0), "normalization target must be positive and finite");
    require(std::isfinite(options.max_fraction) && options.max_fraction >= 0 && options.max_fraction <= 1, "max_fraction must be in [0,1]");
    std::vector<double> totals(m.rows,0);
    std::vector<std::uint8_t> included(m.columns,1);
    numeric(m.owner.dtype(), [&]<class T>() {
        m.each<T>([&](std::size_t i, std::size_t, double v, std::size_t) {
            require(std::isfinite(v) && v >= 0, "normalization requires finite nonnegative counts"); totals[i] += v;
        });
        if (options.exclude_highly_expressed) {
            m.each<T>([&](std::size_t i, std::size_t j, double v, std::size_t) {
                if (v > options.max_fraction*totals[i]) included[j] = 0;
            });
            std::fill(totals.begin(),totals.end(),0);
            m.each<T>([&](std::size_t i, std::size_t j, double v, std::size_t) { if (included[j]) totals[i] += v; });
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
Scaling scaling(anndata::ArrayView x, const ScaleOptions& options) {
    auto stats = feature_stats(x,options.observations);
    auto n = stats.variance.extent(0); auto out = anndata::Array::empty<double>({n});
    auto* p = out.edit().dense<double>().data(); auto* v = stats.variance.view().dense<double>().data();
    for (std::size_t j = 0; j < n; ++j) p[j] = v[j] == 0 ? 1 : std::sqrt(v[j]);
    return {std::move(stats.mean),std::move(out)};
}
using Columns = std::vector<std::pair<std::string,anndata::Column>>;
void column(Columns& columns, std::string name, anndata::Array a) {
    columns.emplace_back(std::move(name),anndata::Column::numeric(std::move(a)));
}
void attach_obs(anndata::AnnData& a, Columns& columns) { for (auto& [name,c] : columns) a.set_obs_column(std::move(name),std::move(c)); }
void attach_var(anndata::AnnData& a, Columns& columns) { for (auto& [name,c] : columns) a.set_var_column(std::move(name),std::move(c)); }
} // namespace

QcResult qc(anndata::ArrayView x, const QcOptions& options) {
    Matrix m(x);
    std::vector<double> cell_total(m.rows,0), gene_total(m.columns,0), mean(m.columns), dropout(m.columns);
    std::vector<std::uint64_t> cell_detected(m.rows,0), gene_detected(m.columns,0);
    std::vector<std::vector<std::uint8_t>> masks;
    std::set<std::string> names;
    for (const auto& g : options.groups) {
        require(!g.name.empty() && names.insert(g.name).second, "QC group names must be nonempty and unique");
        masks.push_back(mask(g.mask,m.columns));
    }
    std::set<std::size_t> cutoffs;
    for (auto n : options.percent_top)
        require(n && n <= m.columns && cutoffs.insert(n).second, "QC top counts must be distinct and between one and the number of genes");
    std::vector<std::vector<double>> groups(options.groups.size(),std::vector<double>(m.rows,0));
    numeric(x.dtype(), [&]<class T>() {
        m.each<T>([&](std::size_t i, std::size_t j, double v, std::size_t) {
            require(std::isfinite(v) && v >= 0, "QC requires finite nonnegative counts");
            cell_total[i] += v; gene_total[j] += v;
            cell_detected[i] += v != 0; gene_detected[j] += v != 0;
            for (std::size_t g = 0; g < masks.size(); ++g) if (masks[g][j]) groups[g][i] += v;
        });
    });
    for (std::size_t j = 0; j < m.columns; ++j) {
        mean[j] = m.rows ? gene_total[j]/double(m.rows) : std::numeric_limits<double>::quiet_NaN();
        dropout[j] = m.rows ? 100*(1-double(gene_detected[j])/double(m.rows)) : std::numeric_limits<double>::quiet_NaN();
    }
    QcResult out;
    out.cells = {array(cell_total),array(cell_detected)}; out.genes = {array(gene_total),array(gene_detected)};
    out.mean = array(mean); out.dropout = array(dropout); out.percent_top = options.percent_top;
    if (options.log1p) {
        out.log_cell_total = log_array(cell_total); out.log_gene_total = log_array(gene_total); out.log_gene_mean = log_array(mean);
        std::vector<double> detected(cell_detected.begin(),cell_detected.end()); out.log_cell_detected = log_array(detected);
    }
    for (std::size_t g = 0; g < groups.size(); ++g) {
        std::vector<double> percent(m.rows);
        for (std::size_t i = 0; i < m.rows; ++i) percent[i] = 100*groups[g][i]/cell_total[i];
        GroupQc result{options.groups[g].name,array(groups[g]),array(percent),{}};
        if (options.log1p) result.log_total = log_array(groups[g]); out.groups.push_back(std::move(result));
    }
    out.top_percent = anndata::Array::empty<double>({m.rows,options.percent_top.size()});
    if (!options.percent_top.empty()) {
        auto* p = out.top_percent.edit().dense<double>().data(); std::vector<double> row(m.columns);
        numeric(x.dtype(), [&]<class T>() {
            for (std::size_t i = 0; i < m.rows; ++i) {
                m.row<T>(i,row.data()); std::sort(row.begin(),row.end(),std::greater<double>());
                std::partial_sum(row.begin(),row.end(),row.begin());
                for (std::size_t k = 0; k < options.percent_top.size(); ++k)
                    p[i*options.percent_top.size()+k] = 100*row[options.percent_top[k]-1]/cell_total[i];
            }
        });
    }
    return out;
}

anndata::Array filter_cells(const QcResult& q, const FilterBounds& bounds) { return filter(q.cells,bounds); }
anndata::Array filter_genes(const QcResult& q, const FilterBounds& bounds) { return filter(q.genes,bounds); }
anndata::Array filter_cells(anndata::ArrayView x, const FilterBounds& bounds) { QcOptions o; o.log1p = false; return filter_cells(qc(x,o),bounds); }
anndata::Array filter_genes(anndata::ArrayView x, const FilterBounds& bounds) { QcOptions o; o.log1p = false; return filter_genes(qc(x,o),bounds); }

FeatureStats feature_stats(anndata::ArrayView x, std::optional<anndata::ArrayView> observations) {
    Matrix m(x); auto selected = mask(observations,m.rows);
    auto n = std::size_t(std::count(selected.begin(),selected.end(),std::uint8_t(1)));
    std::vector<double> mean(m.columns,0), variance(m.columns,0);
    std::vector<std::size_t> stored(m.columns,0);
    numeric(x.dtype(), [&]<class T>() {
        m.each<T>([&](std::size_t i, std::size_t j, double v, std::size_t) {
            require(std::isfinite(v), "feature statistics require finite values");
            if (selected[i]) mean[j] += v;
        });
        for (auto& v : mean) v = n ? v/double(n) : std::numeric_limits<double>::quiet_NaN();
        m.each<T>([&](std::size_t i, std::size_t j, double v, std::size_t) {
            if (selected[i]) { variance[j] += (v-mean[j])*(v-mean[j]); ++stored[j]; }
        });
    });
    for (std::size_t j = 0; j < m.columns; ++j)
        variance[j] = !n ? std::numeric_limits<double>::quiet_NaN()
            : (variance[j]+double(n-stored[j])*mean[j]*mean[j])/double(n > 1 ? n-1 : 1);
    return {array(mean),array(variance),n};
}

Normalization normalize_total(anndata::ArrayEdit x, const NormalizationOptions& options) {
    Matrix m(x.view()); floating(x.view().dtype(), []<class T>() {}); writable(m);
    auto out = factors(m,options); auto f = out.factors.view().dense<double>();
    transform(x,[&](std::size_t i, std::size_t, double v) { return f.data()[i] > 0 ? v/f.data()[i] : v; }); return out;
}
Normalization normalize_total_into(anndata::ArrayView x, anndata::ArrayEdit destination, const NormalizationOptions& options) {
    Matrix m(x), d(destination.view()); output_contract(m,d); auto out = factors(m,options);
    auto f = out.factors.view().dense<double>();
    transform_into(m,destination,[&](std::size_t i, std::size_t, double v) { return f.data()[i] > 0 ? v/f.data()[i] : v; }); return out;
}
void log1p(anndata::ArrayEdit x, std::optional<double> base) {
    auto d = log_divisor(base); transform(x,[&](std::size_t, std::size_t, double v) { return std::log1p(v)/d; });
}
void log1p_into(anndata::ArrayView x, anndata::ArrayEdit destination, std::optional<double> base) {
    auto d = log_divisor(base); Matrix m(x);
    transform_into(m,destination,[&](std::size_t, std::size_t, double v) { return std::log1p(v)/d; });
}

anndata::Array floating_copy(anndata::ArrayView x, anndata::DType type) {
    Matrix m(x);
    return floating(type,[&]<class T>() -> anndata::Array {
        if (m.layout == anndata::Layout::dense) {
            auto out = anndata::Array::empty<T>({m.rows,m.columns}); copy_values(m,out.edit()); return out;
        }
        auto values = anndata::Array::empty<T>({m.nnz});
        auto copy_structure = [&](anndata::DType dtype, const std::byte* src, std::size_t n) {
            return numeric(dtype,[&]<class I>() {
                auto out = anndata::Array::empty<I>({n}); if (n) std::memcpy(out.edit().bytes().data(),src,n*sizeof(I)); return out;
            });
        };
        auto indices = copy_structure(x.index_dtype(),m.indices,m.nnz);
        auto offsets = copy_structure(x.offset_dtype(),m.offsets,(m.layout == anndata::Layout::csr ? m.rows : m.columns)+1);
        auto out = anndata::Array::from_sparse_buffers(m.layout,std::move(values),std::move(indices),std::move(offsets),m.rows,m.columns);
        copy_values(m,out.edit()); return out;
    });
}
anndata::Array dense_copy(anndata::ArrayView x, anndata::DType type) {
    floating(type,[]<class T>() {});
    x.validate(); anndata::Array retained; bool materialized = false;
    try {
        if (x.layout() == anndata::Layout::dense) (void)x.bytes();
        else (void)x.sparse_bytes();
    } catch (const anndata::Error& e) {
        if (e.code() != anndata::ErrorCode::unsupported_layout && e.code() != anndata::ErrorCode::non_contiguous) throw;
        retained = x.copy(); x = retained.view(); materialized = true;
    }
    Matrix m(x,materialized);
    return floating(type,[&]<class T>() { auto out = anndata::Array::empty<T>({m.rows,m.columns}); copy_values(m,out.edit()); return out; });
}

void apply_scale(anndata::ArrayEdit x, const Scaling& parameters, const ScaleOptions& options) {
    scale_options(options); Matrix m(x.view()); writable(m);
    floating(x.view().dtype(),[]<class T>() {});
    require(!options.zero_center || m.layout == anndata::Layout::dense, "centered sparse scaling requires an explicit dense destination");
    auto mean = doubles(parameters.mean.view(),m.columns), sd = doubles(parameters.stddev.view(),m.columns);
    auto selected = mask(options.observations,m.rows);
    if (std::find(selected.begin(),selected.end(),std::uint8_t(1)) != selected.end())
        for (std::size_t j = 0; j < m.columns; ++j)
            require(std::isfinite(mean[j]) && std::isfinite(sd[j]) && sd[j] > 0, "scaling parameters must be finite with positive standard deviation");
    transform(x,[&](std::size_t i, std::size_t j, double v) {
        if (!selected[i]) return v;
        v = (v-(options.zero_center ? mean[j] : 0))/sd[j];
        return !options.max_value ? v : options.zero_center ? std::clamp(v,-*options.max_value,*options.max_value) : std::min(v,*options.max_value);
    });
}
Scaling scale(anndata::ArrayEdit x, const ScaleOptions& options) {
    scale_options(options); Matrix m(x.view()); writable(m);
    require(!options.zero_center || m.layout == anndata::Layout::dense, "centered sparse scaling requires an explicit dense destination");
    floating(x.view().dtype(),[]<class T>() {});
    auto out = scaling(x.view(),options); apply_scale(x,out,options); return out;
}
Scaling scale_into(anndata::ArrayView x, anndata::ArrayEdit destination, const ScaleOptions& options) {
    scale_options(options); Matrix m(x), d(destination.view()); output_contract(m,d);
    require(!options.zero_center || d.layout == anndata::Layout::dense, "centered sparse scaling requires an explicit dense destination");
    auto out = scaling(x,options); auto mean = doubles(out.mean.view(),m.columns), sd = doubles(out.stddev.view(),m.columns);
    auto selected = mask(options.observations,m.rows);
    transform_into(m,destination,[&](std::size_t i, std::size_t j, double v) {
        if (!selected[i]) return v;
        v = (v-(options.zero_center ? mean[j] : 0))/sd[j];
        return !options.max_value ? v : options.zero_center ? std::clamp(v,-*options.max_value,*options.max_value) : std::min(v,*options.max_value);
    });
    return out;
}

anndata::ArrayView source(const anndata::AnnData& data, std::string_view layer) { return layer.empty() ? data.x() : data.layer(layer); }
anndata::ArrayEdit edit_source(anndata::AnnData& data, std::string_view layer) { return layer.empty() ? data.edit_x() : data.edit_layer(layer); }
anndata::Array batch_codes(const anndata::AnnData& data, std::string_view column_name) {
    auto c = data.obs_column(column_name); std::vector<std::int64_t> codes(c.size()); std::map<std::string,std::int64_t> labels;
    for (std::size_t i = 0; i < c.size(); ++i) require(c.valid(i), "batch labels must not be missing");
    if (c.kind() == anndata::ColumnKind::categorical) {
        for (std::size_t i = 0; i < c.size(); ++i) { codes[i] = c.code(i); require(codes[i] >= 0, "batch labels must not be missing"); }
    } else if (c.kind() == anndata::ColumnKind::string) {
        for (std::size_t i = 0; i < c.size(); ++i) {
            auto [it,inserted] = labels.emplace(std::string(c.string(i)),std::int64_t(labels.size())); codes[i] = it->second;
        }
    } else {
        auto values = c.values(); require(values.dtype() != anndata::DType::float32 && values.dtype() != anndata::DType::float64, "numeric batch labels must be integers");
        values.validate(); auto b = values.bytes(); numeric(values.dtype(),[&]<class T>() {
            for (std::size_t i = 0; i < c.size(); ++i) {
                auto v = load<T>(b.data()+std::ptrdiff_t(i)*b.stride_bytes(0));
                auto label = std::to_string(v); auto [it,inserted] = labels.emplace(std::move(label),std::int64_t(labels.size())); codes[i] = it->second;
            }
        });
    }
    return array(codes);
}

void annotate_qc(anndata::AnnData& data, QcResult&& q) {
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
        auto a = anndata::Array::empty<double>({n}); auto* dst = a.edit().dense<double>().data();
        for (std::size_t i = 0; i < n; ++i) dst[i] = values.data()[std::ptrdiff_t(i)*values.stride(0)+std::ptrdiff_t(k)*values.stride(1)];
        column(obs,"pct_counts_in_top_"+std::to_string(q.percent_top[k])+"_genes",std::move(a));
    }
    attach_obs(data,obs); attach_var(data,var);
}
void annotate_normalization(anndata::AnnData& data, Normalization&& result, std::string_view key) {
    vector_shape(result.factors,data.n_obs()); vector_shape(result.included_genes,data.n_vars());
    require(!key.empty(), "normalization annotation key must not be empty");
    data.set_obs_column(std::string(key),anndata::Column::numeric(std::move(result.factors)));
}
void annotate_log1p(anndata::AnnData& data, std::optional<double> base) {
    log_divisor(base); auto e = anndata::Element::mapping();
    e.set("base",base ? anndata::Element(*base) : anndata::Element(nullptr)); data.set_uns("log1p",std::move(e));
}
void annotate_scale(anndata::AnnData& data, Scaling&& parameters) {
    vector_shape(parameters.mean,data.n_vars()); vector_shape(parameters.stddev,data.n_vars());
    Columns columns; column(columns,"mean",std::move(parameters.mean)); column(columns,"std",std::move(parameters.stddev)); attach_var(data,columns);
}
void annotate_hvg(anndata::AnnData& data, HvgResult&& result) {
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
    auto e = anndata::Element::mapping(); e.set("flavor",anndata::Element(result.method == HvgMethod::seurat ? "seurat" : "cell_ranger"));
    attach_var(data,columns); data.set_uns("hvg",std::move(e));
}

} // namespace preprocess
