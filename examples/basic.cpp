import preprocess;

#include <cstdint>
#include <utility>
#include <vector>

int main() {
    auto counts = anndatapp::Array::empty<float>({3, 3});
    auto x = counts.edit().dense<float>();
    float* p = x.data();
    p[0] = 2; p[1] = 0; p[2] = 1;
    p[3] = 4; p[4] = 2; p[5] = 0;
    p[6] = 0; p[7] = 1; p[8] = 3;

    auto data = anndatapp::AnnData::empty(3, 3);
    data.set_x(std::move(counts));
    auto summary = preprocess::qc(preprocess::source(data));
    auto keep = preprocess::filter_cells(summary, {.min_detected = 2});
    auto selected = keep.view().dense<bool>();
    std::vector<std::uint8_t> mask(3);
    for (std::size_t i = 0; i < mask.size(); ++i) mask[i] = selected.data()[i];

    auto normalized = preprocess::floating_copy(preprocess::source(data), anndatapp::DType::float32);
    auto norm = preprocess::normalize_total(normalized.edit(), {.target_sum = 1e4});
    preprocess::log1p(normalized.edit());
    preprocess::annotate_qc(data, std::move(summary));
    preprocess::annotate_normalization(data, std::move(norm));
    data.set_layer("normalized", std::move(normalized));
    auto hvg = preprocess::highly_variable_genes(preprocess::source(data, "normalized"));
    preprocess::annotate_hvg(data, std::move(hvg));
    return mask.size() == data.n_obs() ? 0 : 1;
}
