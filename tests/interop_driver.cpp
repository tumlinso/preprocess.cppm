#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

import preprocess;

namespace {

template<class T> void emit_values(anndata::ArrayView a) {
    auto v = a.dense<T>();
    if (v.rank() == 1) {
        for (std::size_t i = 0; i < v.extent(0); ++i) std::cout << ' ' << double(v.data()[i*v.stride(0)]);
    } else if (v.rank() == 2) {
        for (std::size_t i = 0; i < v.extent(0); ++i)
            for (std::size_t j = 0; j < v.extent(1); ++j)
                std::cout << ' ' << double(v.data()[i*v.stride(0) + j*v.stride(1)]);
    } else throw std::runtime_error("unsupported output rank");
}

void emit(const char* name, anndata::ArrayView a) {
    std::cout << name << ' ' << a.rank();
    for (std::size_t i = 0; i < a.rank(); ++i) std::cout << ' ' << a.extent(i);
    switch (a.dtype()) {
    case anndata::DType::boolean: emit_values<bool>(a); break;
    case anndata::DType::int8: emit_values<std::int8_t>(a); break;
    case anndata::DType::uint8: emit_values<std::uint8_t>(a); break;
    case anndata::DType::int32: emit_values<std::int32_t>(a); break;
    case anndata::DType::uint32: emit_values<std::uint32_t>(a); break;
    case anndata::DType::int64: emit_values<std::int64_t>(a); break;
    case anndata::DType::uint64: emit_values<std::uint64_t>(a); break;
    case anndata::DType::float32: emit_values<float>(a); break;
    case anndata::DType::float64: emit_values<double>(a); break;
    default: throw std::runtime_error("unsupported output dtype");
    }
    std::cout << '\n';
}

void emit(const char* name, const anndata::Array& a) { emit(name, a.view()); }

void emit_matrix(anndata::ArrayView a) {
    auto dense = preprocess::dense_copy(a, anndata::DType::float64);
    emit("matrix", dense);
}

template<class T>
void run(const std::string& operation, int layout, int mode, std::size_t n, std::size_t g,
         const std::vector<double>& input, const std::vector<int>& group_values,
         const std::vector<int>& observation_values, const std::vector<std::int32_t>& batch_values) {
    std::vector<T> values;
    std::vector<std::int32_t> indices;
    std::vector<std::int64_t> offsets;
    anndata::Array a;
    if (layout == 0 || layout == 3) {
        values.resize(n*g);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < g; ++j)
                values[(layout == 3 ? n - 1 - i : i)*g + j] = T(input[i*g + j]);
        T* p = values.data();
        if (layout == 3 && n) p += (n-1)*g;
        a = anndata::Array::borrow_dense(p, {n,g}, {layout == 3 ? -std::ptrdiff_t(g) : std::ptrdiff_t(g), 1});
    } else {
        const bool csr = layout == 1;
        const std::size_t major = csr ? n : g, minor = csr ? g : n;
        offsets.push_back(0);
        for (std::size_t i = 0; i < major; ++i) {
            for (std::size_t j = 0; j < minor; ++j) {
                const std::size_t row = csr ? i : j, col = csr ? j : i;
                T x = T(input[row*g + col]);
                if (x != T(0) || (row*g + col) % 31 == 0) {
                    values.push_back(x); indices.push_back(std::int32_t(j));
                }
            }
            offsets.push_back(std::int64_t(values.size()));
        }
        a = anndata::Array::borrow_sparse(csr ? anndata::Layout::csr : anndata::Layout::csc,
            n, g, values.size(), values.data(), indices.data(), offsets.data());
    }
    auto group = std::make_unique<bool[]>(g), observations = std::make_unique<bool[]>(n);
    for (std::size_t i = 0; i < g; ++i) group[i] = group_values[i] != 0;
    for (std::size_t i = 0; i < n; ++i) observations[i] = observation_values[i] != 0;
    auto ga = anndata::Array::borrow_dense(group.get(), {g}, {1});
    auto oa = anndata::Array::borrow_dense(observations.get(), {n}, {1});
    auto ba = anndata::Array::borrow_dense(batch_values.data(), {n}, {1});

    if (operation == "annotations") {
        if (layout != 0 || std::is_integral_v<T>) throw std::invalid_argument("annotations require dense floating input");
        auto data = anndata::AnnData::empty(n, g);
        data.set_x(std::move(a));
        std::vector<std::string> obs_names_order(n), var_names_order(g);
        for (std::size_t i = 0; i < n; ++i) obs_names_order[i] = "sample_" + std::to_string(n-i);
        for (std::size_t j = 0; j < g; ++j) var_names_order[j] = "gene_" + std::to_string(g-j);
        data.set_obs(anndata::Table::with_index(obs_names_order));
        data.set_var(anndata::Table::with_index(var_names_order));
        auto obs_sentinel = anndata::Array::empty<std::int64_t>({n});
        auto var_sentinel = anndata::Array::empty<std::int64_t>({g});
        auto* obs_value = obs_sentinel.edit().dense<std::int64_t>().data();
        auto* var_value = var_sentinel.edit().dense<std::int64_t>().data();
        for (std::size_t i = 0; i < n; ++i) obs_value[i] = std::int64_t(i);
        for (std::size_t j = 0; j < g; ++j) var_value[j] = std::int64_t(j);
        data.set_obs_column("sentinel", anndata::Column::numeric(std::move(obs_sentinel)));
        data.set_var_column("sentinel", anndata::Column::numeric(std::move(var_sentinel)));
        data.set_layer("counts", preprocess::dense_copy(data.x()));
        preprocess::QcOptions qc_options;
        qc_options.log1p = true; qc_options.percent_top = {2, 4, 8};
        qc_options.groups.push_back({"control", ga.view()});
        auto q = preprocess::qc(preprocess::source(data), qc_options);
        preprocess::annotate_qc(data, std::move(q));
        preprocess::NormalizationOptions norm_options; norm_options.target_sum = 100;
        auto norm = preprocess::normalize_total(preprocess::edit_source(data), norm_options);
        preprocess::annotate_normalization(data, std::move(norm));
        const auto base = mode ? std::optional<double>(mode == 1 ? 2.0 : .5) : std::nullopt;
        preprocess::log1p(preprocess::edit_source(data), base);
        preprocess::annotate_log1p(data, base);
        preprocess::HvgOptions hvg_options; hvg_options.n_top_genes = 12; hvg_options.n_bins = 10;
        hvg_options.log_base = base;
        auto hvg = preprocess::highly_variable_genes(preprocess::source(data), hvg_options);
        preprocess::annotate_hvg(data, std::move(hvg));
        auto scaling = preprocess::scale(preprocess::edit_source(data));
        preprocess::annotate_scale(data, std::move(scaling));
        const char* obs_names[] = {"total_counts", "n_genes_by_counts", "log1p_total_counts",
            "log1p_n_genes_by_counts", "total_counts_control", "pct_counts_control",
            "log1p_total_counts_control", "pct_counts_in_top_2_genes", "pct_counts_in_top_4_genes",
            "pct_counts_in_top_8_genes", "size_factors"};
        const char* var_names[] = {"total_counts", "n_cells_by_counts", "mean_counts",
            "pct_dropout_by_counts", "log1p_total_counts", "log1p_mean_counts",
            "means", "dispersions", "dispersions_norm", "highly_variable", "mean", "std"};
        for (auto name : obs_names) { std::string key = std::string("obs_") + name; emit(key.c_str(), data.obs_column(name).values()); }
        for (auto name : var_names) { std::string key = std::string("var_") + name; emit(key.c_str(), data.var_column(name).values()); }
        auto uns = data.uns();
        bool log_ok = uns.contains("log1p") && uns.at("log1p").contains("base");
        if (log_ok) {
            auto b = uns.at("log1p").at("base");
            log_ok = base ? b.kind() == anndata::ElementKind::floating && b.floating() == *base
                          : b.kind() == anndata::ElementKind::null;
        }
        const bool hvg_ok = uns.contains("hvg") && uns.at("hvg").contains("flavor") &&
                            uns.at("hvg").at("flavor").string() == "seurat";
        std::cout << "metadata 1 2 " << log_ok << ' ' << hvg_ok << '\n';
        bool obs_aligned = true, var_aligned = true;
        for (std::size_t i = 0; i < n; ++i) obs_aligned &= data.obs().name(i) == obs_names_order[i];
        for (std::size_t j = 0; j < g; ++j) var_aligned &= data.var().name(j) == var_names_order[j];
        std::cout << "axis_alignment 1 2 " << obs_aligned << ' ' << var_aligned << '\n';
        emit("obs_sentinel", data.obs_column("sentinel").values());
        emit("var_sentinel", data.var_column("sentinel").values());
        emit("layer_counts", preprocess::source(data, "counts"));
        emit_matrix(data.x());
    } else if (operation == "qc" || operation == "filters") {
        preprocess::QcOptions options;
        options.log1p = true; options.percent_top = {2, 4, 8};
        options.groups.push_back({"control", ga.view()});
        auto q = preprocess::qc(a.view(), options);
        if (operation == "filters") {
            preprocess::FilterBounds cells, genes;
            cells.min_counts = 4; cells.max_counts = 10000;
            cells.min_detected = 2; cells.max_detected = g;
            genes.min_counts = 5; genes.max_counts = 10000;
            genes.min_detected = 2; genes.max_detected = n;
            emit("cells", preprocess::filter_cells(q, cells));
            emit("genes", preprocess::filter_genes(q, genes));
            return;
        }
        emit("cell_total", q.cells.total); emit("cell_detected", q.cells.detected);
        emit("gene_total", q.genes.total); emit("gene_detected", q.genes.detected);
        emit("gene_mean", q.mean); emit("gene_dropout", q.dropout);
        emit("log_cell_total", q.log_cell_total); emit("log_cell_detected", q.log_cell_detected);
        emit("log_gene_total", q.log_gene_total); emit("log_gene_mean", q.log_gene_mean);
        emit("group_total", q.groups.at(0).total); emit("group_percent", q.groups.at(0).percent);
        emit("group_log_total", q.groups.at(0).log_total); emit("top_percent", q.top_percent);
    } else if (operation == "stats") {
        auto s = mode ? preprocess::feature_stats(a.view(), oa.view()) : preprocess::feature_stats(a.view());
        emit("mean", s.mean); emit("variance", s.variance);
    } else if (operation == "normalize" || operation == "normalize_into") {
        preprocess::NormalizationOptions options;
        if (mode != 1) options.target_sum = 100;
        options.exclude_highly_expressed = mode == 2; options.max_fraction = .2;
        if (operation == "normalize_into") {
            auto x = anndata::Array::empty<float>({n,g});
            auto result = preprocess::normalize_total_into(a.view(), x.edit(), options);
            emit_matrix(x.view()); emit("factors", result.factors); emit("included", result.included_genes);
            std::cout << "target 1 1 " << result.target_sum << '\n';
        } else if constexpr (std::is_integral_v<T>) {
            auto x = preprocess::floating_copy(a.view(), anndata::DType::float64);
            auto result = preprocess::normalize_total(x.edit(), options);
            emit_matrix(x.view()); emit("factors", result.factors); emit("included", result.included_genes);
            std::cout << "target 1 1 " << result.target_sum << '\n';
        } else {
            auto result = preprocess::normalize_total(a.edit(), options);
            emit_matrix(a.view()); emit("factors", result.factors); emit("included", result.included_genes);
            std::cout << "target 1 1 " << result.target_sum << '\n';
        }
    } else if (operation == "log1p" || operation == "log1p_into") {
        const auto base = mode ? std::optional<double>(mode == 1 ? 2.0 : .5) : std::nullopt;
        if (operation == "log1p_into") {
            auto x = anndata::Array::empty<float>({n,g});
            preprocess::log1p_into(a.view(), x.edit(), base); emit_matrix(x.view());
        } else if constexpr (std::is_integral_v<T>) {
            auto x = preprocess::floating_copy(a.view(), anndata::DType::float64);
            preprocess::log1p(x.edit(), base); emit_matrix(x.view());
        } else {
            preprocess::log1p(a.edit(), base); emit_matrix(a.view());
        }
    } else if (operation == "scale") {
        preprocess::ScaleOptions options;
        options.zero_center = mode != 1;
        if (mode) options.max_value = 3;
        if (mode == 2) options.observations = oa.view();
        if (layout == 1 || layout == 2 || std::is_integral_v<T>) {
            auto out = anndata::Array::empty<double>({n,g});
            auto result = preprocess::scale_into(a.view(), out.edit(), options);
            emit_matrix(out.view()); emit("mean", result.mean); emit("stddev", result.stddev);
        } else {
            auto result = preprocess::scale(a.edit(), options);
            emit_matrix(a.view()); emit("mean", result.mean); emit("stddev", result.stddev);
        }
    } else if (operation == "hvg") {
        preprocess::HvgOptions options;
        options.method = (mode & 1) ? preprocess::HvgMethod::cell_ranger : preprocess::HvgMethod::seurat;
        options.n_bins = 10;
        if (!(mode & 4)) options.n_top_genes = 12;
        if (mode & 2) options.batches = ba.view();
        if (mode & 8) options.log_base = 2;
        if (mode & 16) options.log_base = .5;
        auto result = preprocess::highly_variable_genes(a.view(), options);
        emit("mean", result.mean); emit("dispersion", result.dispersion);
        emit("normalized_dispersion", result.normalized_dispersion); emit("selected", result.selected);
        if (mode & 2) { emit("nbatches", result.nbatches); emit("intersection", result.intersection); }
    } else throw std::invalid_argument("unknown operation");
}

} // namespace

int main(int argc, char** argv) try {
    if (argc != 2) throw std::invalid_argument("usage: preprocess_interop_driver operation < input");
    std::size_t n, g; int dtype, layout, mode;
    if (!(std::cin >> n >> g >> dtype >> layout >> mode) || n > 10000 || g > 10000)
        throw std::invalid_argument("invalid header");
    std::vector<double> values(n*g);
    std::vector<int> groups(g), observations(n);
    std::vector<std::int32_t> batches(n);
    for (auto& value : values) std::cin >> value;
    for (auto& value : groups) std::cin >> value;
    for (auto& value : observations) std::cin >> value;
    for (auto& value : batches) std::cin >> value;
    if (!std::cin) throw std::invalid_argument("truncated input");
    std::cout << std::setprecision(17);
    if (dtype == 32) run<float>(argv[1], layout, mode, n, g, values, groups, observations, batches);
    else if (dtype == 64) run<double>(argv[1], layout, mode, n, g, values, groups, observations, batches);
    else if (dtype == 0) run<std::int32_t>(argv[1], layout, mode, n, g, values, groups, observations, batches);
    else throw std::invalid_argument("unsupported dtype");
    return 0;
} catch (const std::exception& e) {
    std::cerr << "interop: " << e.what() << '\n';
    return 1;
}
