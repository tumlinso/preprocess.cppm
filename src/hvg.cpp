module;
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <functional>
#include <map>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <vector>

module preprocess;
#include "numeric.hpp"

namespace preprocess {
namespace {
using namespace detail;
constexpr double nan = std::numeric_limits<double>::quiet_NaN();

struct HvgBatch {
    std::vector<double> mean, dispersion, norm;
    std::vector<std::uint8_t> selected;
};

double median(std::vector<double> x) {
    if (x.empty()) return nan;
    auto n = x.size();
    std::nth_element(x.begin(), x.begin()+n/2, x.end());
    double a = x[n/2];
    if (n%2) return a;
    return (a + *std::max_element(x.begin(), x.begin()+n/2))/2;
}

HvgBatch calculate(const std::vector<double>& sum, const std::vector<double>& square,
                   const std::vector<std::uint8_t>& expressed, std::size_t n,
                   const HvgOptions& o, bool filter) {
    auto g = sum.size();
    HvgBatch out{std::vector<double>(g), std::vector<double>(g), std::vector<double>(g), std::vector<std::uint8_t>(g)};
    std::vector<std::size_t> included;
    for (std::size_t j = 0; j < g; ++j) {
        if (filter && !expressed[j]) continue;
        included.push_back(j);
        double mean = sum[j]/double(n), var = square[j]/double(n) - mean*mean;
        if (n > 1) var *= double(n)/double(n-1);
        if (mean == 0) mean = 1e-12;
        double disp = var/mean;
        if (o.method == HvgMethod::seurat) {
            disp = disp == 0 ? nan : std::log(disp);
            mean = std::log1p(mean);
        }
        require(std::isfinite(mean), "HVG mean is not finite");
        out.mean[j] = mean; out.dispersion[j] = disp; out.norm[j] = nan;
    }
    require(!included.empty(), "HVG batch has no expressed genes");
    std::vector<double> edges;
    if (o.method == HvgMethod::seurat) {
        double lo = out.mean[included[0]], hi = lo;
        for (auto j : included) { lo = std::min(lo, out.mean[j]); hi = std::max(hi, out.mean[j]); }
        bool constant = lo == hi;
        if (constant) { double d = lo == 0 ? .001 : std::abs(lo)*.001; lo -= d; hi += d; }
        double step = (hi-lo)/double(o.n_bins);
        edges.resize(o.n_bins+1);
        for (std::size_t k = 0; k <= o.n_bins; ++k) edges[k] = lo + step*double(k);
        edges.back() = hi;
        if (!constant) edges[0] -= (hi-lo)*.001;
    } else {
        std::vector<double> means; means.reserve(included.size());
        for (auto j : included) means.push_back(out.mean[j]);
        std::sort(means.begin(), means.end());
        edges.push_back(-std::numeric_limits<double>::infinity());
        for (int p = 10; p <= 100; p += 5) {
            double pos = (means.size()-1)*double(p)/100;
            auto k = std::size_t(pos); double f = pos-double(k);
            edges.push_back(k+1 < means.size() ? means[k]*(1-f) + means[k+1]*f : means[k]);
        }
        edges.push_back(std::numeric_limits<double>::infinity());
        for (std::size_t k = 1; k < edges.size(); ++k)
            require(edges[k] > edges[k-1], "Cell Ranger percentile bin edges must be unique");
    }
    std::vector<std::vector<std::size_t>> bins(edges.size()-1);
    for (auto j : included) {
        auto it = std::lower_bound(edges.begin()+1, edges.end(), out.mean[j]);
        require(it != edges.end(), "HVG mean outside bin edges");
        bins[std::size_t(it-edges.begin()-1)].push_back(j);
    }
    for (const auto& bin : bins) {
        std::vector<double> d;
        for (auto j : bin) if (!std::isnan(out.dispersion[j])) d.push_back(out.dispersion[j]);
        if (d.empty()) continue;
        double avg, dev;
        if (o.method == HvgMethod::seurat) {
            avg = std::accumulate(d.begin(), d.end(), 0.0)/double(d.size());
            if (d.size() == 1) { dev = avg; avg = 0; }
            else {
                dev = 0;
                for (double v : d) dev += (v-avg)*(v-avg);
                dev = std::sqrt(dev/double(d.size()-1));
            }
        } else {
            avg = median(d);
            for (auto& v : d) v = std::abs(v-avg);
            dev = median(d)/0.6744897501960817;
        }
        for (auto j : bin) out.norm[j] = (out.dispersion[j]-avg)/dev;
    }
    if (o.n_top_genes) {
        std::vector<double> valid;
        for (auto j : included) if (!std::isnan(out.norm[j])) valid.push_back(out.norm[j]);
        require(!valid.empty(), "no normalized dispersions available for HVG selection");
        auto n_top = std::min(*o.n_top_genes, valid.size());
        std::nth_element(valid.begin(), valid.begin()+n_top-1, valid.end(), std::greater<double>());
        double cutoff = valid[n_top-1];
        for (auto j : included) out.selected[j] = !std::isnan(out.norm[j]) && out.norm[j] >= cutoff;
    } else for (auto j : included) {
        double d = std::isnan(out.norm[j]) ? 0 : out.norm[j];
        out.selected[j] = out.mean[j] > o.min_mean && out.mean[j] < o.max_mean && d > o.min_disp && d < o.max_disp;
    }
    return out;
}
} // namespace

HvgResult highly_variable_genes(anndatapp::ArrayView input, const HvgOptions& o) {
    check_matrix(input); auto rows = input.extent(0), columns = input.extent(1);
    require(rows && columns, "HVG requires a nonempty matrix");
    require(o.method == HvgMethod::seurat || o.method == HvgMethod::cell_ranger, "unknown HVG method");
    require(o.n_bins > 0 && o.n_bins < std::numeric_limits<std::size_t>::max(), "HVG bin count must be positive");
    require(!o.n_top_genes || *o.n_top_genes > 0, "HVG top count must be positive");
    require(!o.log_base || (std::isfinite(*o.log_base) && *o.log_base > 0 && *o.log_base != 1), "log base must be finite, positive, and unequal to one");
    if (!o.n_top_genes) require(!std::isnan(o.min_mean) && !std::isnan(o.max_mean) && !std::isnan(o.min_disp) && !std::isnan(o.max_disp) && o.min_mean < o.max_mean && o.min_disp < o.max_disp, "invalid HVG cutoffs");
    std::vector<std::size_t> batch(rows, 0), counts(1, rows);
    if (o.batches) {
        auto b = *o.batches;
        b.validate();
        require(b.rank() == 1 && b.extent(0) == rows && b.layout() == anndatapp::Layout::dense, "batch codes must be a dense vector matching cells");
        auto bytes = b.bytes(); const auto* p = bytes.data(); auto stride = bytes.stride_bytes(0);
        std::map<std::uint64_t, std::size_t> codes;
        numeric(b.dtype(), [&]<class T>() {
            if constexpr (!std::is_integral_v<T> || std::is_same_v<T, bool>) throw std::invalid_argument("batch codes must be nonnegative integers");
            else for (std::size_t i = 0; i < rows; ++i) {
                T v = load<T>(p+std::ptrdiff_t(i)*stride);
                if constexpr (std::is_signed_v<T>) require(v >= 0, "batch codes must be nonnegative integers");
                auto key = std::uint64_t(v); auto [it, inserted] = codes.emplace(key, codes.size());
                batch[i] = it->second;
            }
        });
        counts.assign(codes.size(), 0); for (auto b : batch) ++counts[b];
    }
    auto B = counts.size(), g = columns;
    require(B <= std::numeric_limits<std::size_t>::max()/g, "HVG batch workspace overflow");
    std::vector<double> sum(B*g), square(B*g);
    std::vector<std::uint8_t> expressed(B*g);
    double multiplier = o.log_base ? std::log(*o.log_base) : 1;
    numeric(input.dtype(), [&]<class T>() {
        entries<T>(input,[&](std::size_t i, std::size_t j, double v, std::size_t) {
            require(std::isfinite(v), "HVG input must be finite");
            auto k = batch[i]*g+j; expressed[k] |= v != 0;
            if (o.method == HvgMethod::seurat) v = std::expm1(v*multiplier);
            require(std::isfinite(v) && v >= 0 && std::isfinite(v*v), "HVG transformed values must be finite and nonnegative");
            sum[k] += v; square[k] += v*v;
        });
    });
    for (std::size_t k = 0; k < B*g; ++k)
        require(std::isfinite(sum[k]) && std::isfinite(square[k]), "HVG accumulated moments are not finite");
    std::vector<double> means(g), dispersions(g), norms(g);
    std::vector<std::size_t> mc(g), dc(g), nc(g), nb(g);
    std::vector<std::uint8_t> selected(g), intersection(g);
    for (std::size_t b = 0; b < B; ++b) {
        auto first = b*g;
        auto r = calculate(std::vector<double>(sum.begin()+first, sum.begin()+first+g),
                           std::vector<double>(square.begin()+first, square.begin()+first+g),
                           std::vector<std::uint8_t>(expressed.begin()+first, expressed.begin()+first+g), counts[b], o, bool(o.batches));
        for (std::size_t j = 0; j < g; ++j) {
            if (!std::isnan(r.mean[j])) { means[j] += r.mean[j]; ++mc[j]; }
            if (!std::isnan(r.dispersion[j])) { dispersions[j] += r.dispersion[j]; ++dc[j]; }
            if (!std::isnan(r.norm[j])) { norms[j] += r.norm[j]; ++nc[j]; }
            nb[j] += r.selected[j]; selected[j] = r.selected[j];
        }
    }
    for (std::size_t j = 0; j < g; ++j) {
        means[j] = mc[j] ? means[j]/double(mc[j]) : nan;
        dispersions[j] = dc[j] ? dispersions[j]/double(dc[j]) : nan;
        norms[j] = nc[j] ? norms[j]/double(nc[j]) : nan;
        intersection[j] = nb[j] == B;
    }
    if (o.batches) {
        if (o.n_top_genes) {
            std::vector<std::size_t> order(g); std::iota(order.begin(), order.end(), 0);
            std::stable_sort(order.begin(), order.end(), [&](auto a, auto b) {
                if (nb[a] != nb[b]) return nb[a] > nb[b];
                if (std::isnan(norms[a])) return false;
                if (std::isnan(norms[b])) return true;
                return norms[a] > norms[b];
            });
            std::fill(selected.begin(), selected.end(), 0);
            for (std::size_t k = 0; k < std::min(g, *o.n_top_genes); ++k) selected[order[k]] = 1;
        } else for (std::size_t j = 0; j < g; ++j) {
            if (std::isnan(norms[j])) norms[j] = 0;
            selected[j] = means[j] > o.min_mean && means[j] < o.max_mean && norms[j] > o.min_disp && norms[j] < o.max_disp;
        }
    }
    HvgResult result;
    result.mean = array(means); result.dispersion = array(dispersions); result.normalized_dispersion = array(norms);
    result.selected = booleans(selected); result.method = o.method;
    if (o.batches) { result.nbatches = array(nb); result.intersection = booleans(intersection); }
    return result;
}
} // namespace preprocess
