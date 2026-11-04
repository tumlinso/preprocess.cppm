module;
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

export module preprocess;
export import anndata.core;

export namespace preprocess {

struct FeatureGroup { std::string name; anndata::ArrayView mask; };
struct QcOptions {
    std::vector<FeatureGroup> groups;
    std::vector<std::size_t> percent_top;
    bool log1p = true;
};
struct AxisSummary { anndata::Array total, detected; };
struct GroupQc { std::string name; anndata::Array total, percent, log_total; };
struct QcResult {
    AxisSummary cells, genes;
    anndata::Array mean, dropout, log_cell_total, log_cell_detected, log_gene_total, log_gene_mean;
    std::vector<GroupQc> groups;
    std::vector<std::size_t> percent_top;
    anndata::Array top_percent;
};
struct FilterBounds {
    std::optional<double> min_counts, max_counts;
    std::optional<std::uint64_t> min_detected, max_detected;
};
struct FeatureStats { anndata::Array mean, variance; std::size_t observations = 0; };

QcResult qc(anndata::ArrayView, const QcOptions& = {});
anndata::Array filter_cells(const QcResult&, const FilterBounds&);
anndata::Array filter_genes(const QcResult&, const FilterBounds&);
anndata::Array filter_cells(anndata::ArrayView, const FilterBounds&);
anndata::Array filter_genes(anndata::ArrayView, const FilterBounds&);
FeatureStats feature_stats(anndata::ArrayView, std::optional<anndata::ArrayView> observations = {});

struct NormalizationOptions {
    std::optional<double> target_sum;
    bool exclude_highly_expressed = false;
    double max_fraction = .05;
};
struct Normalization { anndata::Array factors, included_genes; double target_sum = 1; };
Normalization normalize_total(anndata::ArrayEdit, const NormalizationOptions& = {});
Normalization normalize_total_into(anndata::ArrayView, anndata::ArrayEdit, const NormalizationOptions& = {});
void log1p(anndata::ArrayEdit, std::optional<double> base = {});
void log1p_into(anndata::ArrayView, anndata::ArrayEdit, std::optional<double> base = {});

// Copies and conversions are explicit. floating_copy preserves sparse layout
// and structure; dense_copy also materializes sparse input or logical gathers.
anndata::Array floating_copy(anndata::ArrayView, anndata::DType = anndata::DType::float64);
anndata::Array dense_copy(anndata::ArrayView, anndata::DType = anndata::DType::float64);

struct ScaleOptions {
    bool zero_center = true;
    std::optional<double> max_value;
    std::optional<anndata::ArrayView> observations;
};
struct Scaling { anndata::Array mean, stddev; };
Scaling scale(anndata::ArrayEdit, const ScaleOptions& = {});
Scaling scale_into(anndata::ArrayView, anndata::ArrayEdit, const ScaleOptions& = {});
void apply_scale(anndata::ArrayEdit, const Scaling&, const ScaleOptions& = {});

enum class HvgMethod { seurat, cell_ranger };
struct HvgOptions {
    HvgMethod method = HvgMethod::seurat;
    std::optional<std::size_t> n_top_genes;
    double min_mean = .0125, max_mean = 3, min_disp = .5;
    double max_disp = std::numeric_limits<double>::infinity();
    std::size_t n_bins = 20;
    std::optional<double> log_base;
    std::optional<anndata::ArrayView> batches;
};
struct HvgResult {
    anndata::Array mean, dispersion, normalized_dispersion, selected, nbatches, intersection;
    HvgMethod method = HvgMethod::seurat;
};
HvgResult highly_variable_genes(anndata::ArrayView, const HvgOptions& = {});

anndata::ArrayView source(const anndata::AnnData&, std::string_view layer = {});
anndata::ArrayEdit edit_source(anndata::AnnData&, std::string_view layer = {});
anndata::Array batch_codes(const anndata::AnnData&, std::string_view column);
void annotate_qc(anndata::AnnData&, QcResult&&);
void annotate_normalization(anndata::AnnData&, Normalization&&, std::string_view key = "size_factors");
void annotate_log1p(anndata::AnnData&, std::optional<double> base = {});
void annotate_scale(anndata::AnnData&, Scaling&&);
void annotate_hvg(anndata::AnnData&, HvgResult&&);

} // namespace preprocess
