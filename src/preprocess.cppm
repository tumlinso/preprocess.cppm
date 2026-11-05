module;
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

export module preprocess;
export import anndatapp.core;

export namespace preprocess {

struct FeatureGroup { std::string name; anndatapp::ArrayView mask; };
struct QcOptions {
    std::vector<FeatureGroup> groups;
    std::vector<std::size_t> percent_top;
    bool log1p = true;
};
struct AxisSummary { anndatapp::Array total, detected; };
struct GroupQc { std::string name; anndatapp::Array total, percent, log_total; };
struct QcResult {
    AxisSummary cells, genes;
    anndatapp::Array mean, dropout, log_cell_total, log_cell_detected, log_gene_total, log_gene_mean;
    std::vector<GroupQc> groups;
    std::vector<std::size_t> percent_top;
    anndatapp::Array top_percent;
};
struct FilterBounds {
    std::optional<double> min_counts, max_counts;
    std::optional<std::uint64_t> min_detected, max_detected;
};
struct FeatureStats { anndatapp::Array mean, variance; std::size_t observations = 0; };

QcResult qc(anndatapp::ArrayView, const QcOptions& = {});
anndatapp::Array filter_cells(const QcResult&, const FilterBounds&);
anndatapp::Array filter_genes(const QcResult&, const FilterBounds&);
anndatapp::Array filter_cells(anndatapp::ArrayView, const FilterBounds&);
anndatapp::Array filter_genes(anndatapp::ArrayView, const FilterBounds&);
FeatureStats feature_stats(anndatapp::ArrayView, std::optional<anndatapp::ArrayView> observations = {});

struct NormalizationOptions {
    std::optional<double> target_sum;
    bool exclude_highly_expressed = false;
    double max_fraction = .05;
};
struct Normalization { anndatapp::Array factors, included_genes; double target_sum = 1; };
Normalization normalize_total(anndatapp::ArrayEdit, const NormalizationOptions& = {});
Normalization normalize_total_into(anndatapp::ArrayView, anndatapp::ArrayEdit, const NormalizationOptions& = {});
void log1p(anndatapp::ArrayEdit, std::optional<double> base = {});
void log1p_into(anndatapp::ArrayView, anndatapp::ArrayEdit, std::optional<double> base = {});

// Copies and conversions are explicit. floating_copy preserves sparse layout
// and structure; dense_copy also materializes sparse input or logical gathers.
anndatapp::Array floating_copy(anndatapp::ArrayView, anndatapp::DType = anndatapp::DType::float64);
anndatapp::Array dense_copy(anndatapp::ArrayView, anndatapp::DType = anndatapp::DType::float64);

struct ScaleOptions {
    bool zero_center = true;
    std::optional<double> max_value;
    std::optional<anndatapp::ArrayView> observations;
};
struct Scaling { anndatapp::Array mean, stddev; };
Scaling scale(anndatapp::ArrayEdit, const ScaleOptions& = {});
Scaling scale_into(anndatapp::ArrayView, anndatapp::ArrayEdit, const ScaleOptions& = {});
void apply_scale(anndatapp::ArrayEdit, const Scaling&, const ScaleOptions& = {});

enum class HvgMethod { seurat, cell_ranger };
struct HvgOptions {
    HvgMethod method = HvgMethod::seurat;
    std::optional<std::size_t> n_top_genes;
    double min_mean = .0125, max_mean = 3, min_disp = .5;
    double max_disp = std::numeric_limits<double>::infinity();
    std::size_t n_bins = 20;
    std::optional<double> log_base;
    std::optional<anndatapp::ArrayView> batches;
};
struct HvgResult {
    anndatapp::Array mean, dispersion, normalized_dispersion, selected, nbatches, intersection;
    HvgMethod method = HvgMethod::seurat;
};
HvgResult highly_variable_genes(anndatapp::ArrayView, const HvgOptions& = {});

anndatapp::ArrayView source(const anndatapp::AnnData&, std::string_view layer = {});
anndatapp::ArrayEdit edit_source(anndatapp::AnnData&, std::string_view layer = {});
anndatapp::Array batch_codes(const anndatapp::AnnData&, std::string_view column);
void annotate_qc(anndatapp::AnnData&, QcResult&&);
void annotate_normalization(anndatapp::AnnData&, Normalization&&, std::string_view key = "size_factors");
void annotate_log1p(anndatapp::AnnData&, std::optional<double> base = {});
void annotate_scale(anndatapp::AnnData&, Scaling&&);
void annotate_hvg(anndatapp::AnnData&, HvgResult&&);

} // namespace preprocess
