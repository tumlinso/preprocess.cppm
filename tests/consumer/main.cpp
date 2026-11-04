import preprocess;

int main() {
    auto x = anndata::Array::empty<double>({2, 2});
    auto v = x.edit().dense<double>();
    double* p = v.data();
    p[0] = 1; p[1] = 2; p[2] = 3; p[3] = 4;
    auto result = preprocess::feature_stats(x.view());
    auto mean = result.mean.view().dense<double>();
    return result.observations == 2 && mean.data()[0] == 2 && mean.data()[1] == 3 ? 0 : 1;
}
