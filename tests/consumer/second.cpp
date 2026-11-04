import preprocess;

int main() {
    auto x = anndata::Array::empty<double>({2, 1});
    auto v = x.edit().dense<double>();
    v.data()[0] = 0; v.data()[1] = 2;
    const auto keep = preprocess::filter_cells(x.view(), {.min_detected = 1});
    auto m = keep.view().dense<bool>();
    return m.data()[0] == 0 && m.data()[1] == 1 ? 0 : 1;
}
