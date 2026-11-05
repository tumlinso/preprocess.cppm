#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <source_location>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
import preprocess;

namespace {
void check(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
bool close(double a, double b, double tolerance = 1e-10) { return std::isnan(a) ? std::isnan(b) : std::abs(a-b) <= tolerance*(1+std::abs(a)); }
template<class F> void rejects(F f, std::source_location where = std::source_location::current()) {
    try { f(); } catch (const std::exception&) { return; }
    throw std::runtime_error("expected rejection at line "+std::to_string(where.line()));
}
template<class T> anndatapp::Array matrix(std::size_t n, std::size_t p, const std::vector<T>& values) {
    auto a = anndatapp::Array::empty<T>({n,p}); std::copy(values.begin(),values.end(),a.edit().template dense<T>().data()); return a;
}
anndatapp::Array data() { return matrix<double>(4,4,{1,0,3,0, 0,0,0,0, 2,4,0,0, 3,2,1,0}); }
template<class T, class I, class O> anndatapp::Array sparse(anndatapp::Layout layout) {
    auto d = data(); auto x = d.view().dense<double>();
    auto a = layout == anndatapp::Layout::csr ? anndatapp::Array::csr<T,I,O>(4,4,16) : anndatapp::Array::csc<T,I,O>(4,4,16);
    auto s = layout == anndatapp::Layout::csr ? a.edit().template csr<T,I,O>() : a.edit().template csc<T,I,O>();
    auto* ptr = s.unsafe_offsets(); auto* idx = s.unsafe_indices(); auto* v = s.values();
    for (int i = 0; i <= 4; ++i) ptr[i] = O(i*4);
    for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) {
        idx[4*i+j] = I(j); v[4*i+j] = T(x.data()[layout == anndatapp::Layout::csr ? 4*i+j : 4*j+i]);
    }
    return a;
}
void equal(anndatapp::ArrayView a, anndatapp::ArrayView b) {
    auto x = a.dense<double>(), y = b.dense<double>();
    check(a.rank() == b.rank() && a.extent(0) == b.extent(0),"array shape mismatch");
    auto n = a.rank() == 1 ? a.extent(0) : a.extent(0)*a.extent(1);
    for (std::size_t i = 0; i < n; ++i) check(close(x.data()[i],y.data()[i]),"numeric arrays differ");
}
void qc_and_statistics() {
    auto a = data(); auto c = sparse<double,std::uint32_t,std::int64_t>(anndatapp::Layout::csr);
    auto s = sparse<double,std::int64_t,std::uint32_t>(anndatapp::Layout::csc);
    auto group = anndatapp::Array::empty<bool>({4}); auto* g = group.edit().dense<bool>().data();
    g[0] = true; g[1] = false; g[2] = true; g[3] = false;
    preprocess::QcOptions options{.groups={{"mito",group.view()}},.percent_top={1,3}};
    auto q = preprocess::qc(a.view(),options), qc = preprocess::qc(c.view(),options), qs = preprocess::qc(s.view(),options);
    equal(q.cells.total.view(),qc.cells.total.view()); equal(q.cells.total.view(),qs.cells.total.view());
    equal(q.mean.view(),qc.mean.view()); equal(q.mean.view(),qs.mean.view());
    equal(q.top_percent.view(),qc.top_percent.view()); equal(q.top_percent.view(),qs.top_percent.view());
    auto* counts = q.cells.detected.view().dense<std::uint64_t>().data();
    check(counts[0] == 2 && counts[1] == 0 && counts[2] == 2 && counts[3] == 3,"stored zeros counted as detection");
    auto stats = preprocess::feature_stats(a.view()), cs = preprocess::feature_stats(c.view()), ss = preprocess::feature_stats(s.view());
    equal(stats.mean.view(),cs.mean.view()); equal(stats.variance.view(),ss.variance.view());
    auto* mu = stats.mean.view().dense<double>().data(); auto* var = stats.variance.view().dense<double>().data();
    check(close(mu[0],1.5) && close(var[0],5.0/3) && mu[3] == 0 && var[3] == 0,"incorrect sample moments");
    auto cells = preprocess::filter_cells(q,{.min_counts=5,.max_counts=6,.min_detected=3});
    auto* keep = cells.view().dense<bool>().data(); check(!keep[0] && !keep[1] && !keep[2] && keep[3],"filter bounds did not intersect");
    auto genes = preprocess::filter_genes(q,{.min_detected=2}); auto* kg = genes.view().dense<bool>().data();
    check(kg[0] && kg[1] && kg[2] && !kg[3],"gene filter wrong");
    rejects([&] { preprocess::qc(a.view(),{.groups={{"x",group.view()},{"x",group.view()}}}); });
    rejects([&] { preprocess::qc(a.view(),{.percent_top={0}}); });
    rejects([&] { preprocess::qc(a.view(),{.percent_top={5}}); });
    auto empty = anndatapp::Array::empty<double>({0,4});
    check(preprocess::feature_stats(empty.view()).observations == 0,"empty observations wrong");
}
void normalization_and_logarithms() {
    auto a = data(); auto b = sparse<double,std::int32_t,std::uint64_t>(anndatapp::Layout::csr);
    auto norm = preprocess::normalize_total(a.edit()), sparse_norm = preprocess::normalize_total(b.edit());
    check(norm.target_sum == 6 && sparse_norm.target_sum == 6,"median positive totals incorrect");
    auto bd = preprocess::dense_copy(b.view()); equal(a.view(),bd.view());
    auto* f = norm.factors.view().dense<double>().data(); check(close(f[0],2.0/3) && f[1] == 0 && f[2] == 1,"normalization factors wrong");
    preprocess::log1p(a.edit(),2); preprocess::log1p(b.edit(),2); bd = preprocess::dense_copy(b.view()); equal(a.view(),bd.view());
    auto zero = anndatapp::Array::zeros<float>({3,5}); auto z = preprocess::normalize_total(zero.edit());
    check(z.target_sum == 1 && z.factors.view().dense<double>().data()[0] == 0,"all-zero normalization convention wrong");
    auto integer = matrix<int>(2,2,{1,3,2,2}); auto output = anndatapp::Array::empty<double>({2,2});
    auto n = preprocess::normalize_total_into(integer.view(),output.edit(),{.target_sum=10});
    check(output.view().dense<double>().data()[0] == 2.5 && integer.view().dense<int>().data()[0] == 1,"explicit integer normalization wrong");
    rejects([&] { preprocess::normalize_total(integer.edit()); });
    rejects([&] { preprocess::log1p(a.edit(),1); });
    rejects([&] { preprocess::normalize_total(a.edit(),{.target_sum=0}); });
    auto raw = matrix<double>(2,2,{99,1,1,9}); auto r = preprocess::normalize_total(raw.edit(),{.target_sum=1,.exclude_highly_expressed=true,.max_fraction=.8});
    check(!r.included_genes.view().dense<bool>().data()[0] && !r.included_genes.view().dense<bool>().data()[1],"highly expressed gene exclusion wrong");
    auto fp = preprocess::floating_copy(integer.view(),anndatapp::DType::float32); check(fp.dtype() == anndatapp::DType::float32,"explicit dtype conversion failed");
    auto huge = matrix<double>(1,1,{1e100}); auto narrow = anndatapp::Array::empty<float>({1,1});
    preprocess::log1p_into(huge.view(),narrow.edit());
    check(close(narrow.view().dense<float>().data()[0],std::log1p(1e100),1e-6),"log1p narrowed before evaluation");
    preprocess::normalize_total_into(huge.view(),narrow.edit(),{.target_sum=1});
    check(narrow.view().dense<float>().data()[0] == 1,"normalization narrowed before evaluation");
    preprocess::scale_into(huge.view(),narrow.edit());
    check(narrow.view().dense<float>().data()[0] == 0,"scaling narrowed before centering");
    auto sc = sparse<int,std::uint64_t,std::uint32_t>(anndatapp::Layout::csc);
    auto converted = preprocess::floating_copy(sc.view(),anndatapp::DType::float32);
    check(converted.layout() == anndatapp::Layout::csc && converted.view().index_dtype() == anndatapp::DType::uint64 && converted.view().offset_dtype() == anndatapp::DType::uint32,"sparse conversion did not preserve structure widths");
}
void scaling_and_strides() {
    auto a = data(); auto b = sparse<double,std::uint32_t,std::uint64_t>(anndatapp::Layout::csr);
    auto out = anndatapp::Array::empty<double>({4,4}); auto p = preprocess::scale(a.edit());
    auto q = preprocess::scale_into(b.view(),out.edit()); equal(a.view(),out.view()); equal(p.stddev.view(),q.stddev.view());
    for (int i = 0; i < 4; ++i) check(a.view().dense<double>().data()[4*i+3] == 0,"constant feature not centered to zero");
    rejects([&] { preprocess::scale(b.edit()); });
    auto selection = anndatapp::Array::empty<bool>({4}); auto* mask = selection.edit().dense<bool>().data();
    mask[0] = true; mask[1] = false; mask[2] = false; mask[3] = true;
    auto masked = data(); auto fitted = preprocess::scale(masked.edit(),{.observations=selection.view()});
    check(close(fitted.mean.view().dense<double>().data()[0],2) && masked.view().dense<double>().data()[8] == 2,"masked scaling fit/application wrong");
    std::vector<double> values{1,0,3,0,0,0,0,0,2,4,0,0,3,2,1,0};
    auto reversed = anndatapp::Array::borrow_dense(values.data()+12,{4,4},{-4,1});
    auto normal = data(); auto rs = preprocess::feature_stats(reversed.view()), ns = preprocess::feature_stats(normal.view()); equal(rs.mean.view(),ns.mean.view());
    preprocess::log1p(reversed.edit()); check(close(values[0],std::log(2.0)),"negative strides not written correctly");
    auto logarithm_input = matrix<double>(2,2,{1,3,7,15}); double reversed_output[4]{};
    auto destination = anndatapp::Array::borrow_dense(reversed_output+2,{2,2},{-2,1});
    preprocess::log1p_into(logarithm_input.view(),destination.edit(),2);
    check(close(reversed_output[0],3) && close(reversed_output[1],4) && close(reversed_output[2],1) && close(reversed_output[3],2),"negative-stride destination changed logical result order");
    auto gathered = normal.view().select({anndatapp::Axis::gather({2,0,3}),anndatapp::Axis::all()});
    rejects([&] { preprocess::qc(gathered); });
    auto materialized = preprocess::dense_copy(gathered); check(materialized.extent(0) == 3 && materialized.view().dense<double>().data()[0] == 2,"explicit gather materialization failed");
    auto sparse_rows = b.view().select({anndatapp::Axis::gather({2,0}),anndatapp::Axis::all()});
    auto rows = preprocess::dense_copy(sparse_rows); check(rows.extent(0) == 2 && rows.view().dense<double>().data()[1] == 4,"sparse row gather materialization failed");
    auto sparse_columns = b.view().select({anndatapp::Axis::all(),anndatapp::Axis::gather({2,0,2})});
    auto cols = preprocess::dense_copy(sparse_columns); auto* cp = cols.view().dense<double>().data();
    check(cols.extent(1) == 3 && cp[0] == 3 && cp[1] == 1 && cp[2] == 3,"sparse minor gather order/repetitions failed");
}
void contracts_and_annotations() {
    auto invalid = anndatapp::Array::borrow_dense<double>(nullptr,{1,1},{1,1},anndatapp::Validation::none);
    rejects([&] { preprocess::qc(invalid.view()); });
    double storage[8]{1,2,3,4,5,6,7,8};
    auto src = anndatapp::Array::borrow_dense(storage,{2,2},{2,1});
    auto overlap = anndatapp::Array::borrow_dense(storage+1,{2,2},{2,1});
    rejects([&] { preprocess::log1p_into(src.view(),overlap.edit()); }); check(storage[0] == 1 && storage[1] == 2,"overlap rejection changed data");
    double empty_output[4]{1,2,3,4}; long empty_offsets[]{0,0,0};
    auto empty_alias = anndatapp::Array::borrow_sparse(anndatapp::Layout::csr,2,2,0,empty_output+1,static_cast<int*>(nullptr),empty_offsets);
    auto empty_destination = anndatapp::Array::borrow_dense(empty_output,{2,2},{2,1});
    preprocess::log1p_into(empty_alias.view(),empty_destination.edit());
    for (auto v : empty_output) check(v == 0,"zero-entry sparse source falsely overlapped destination");
    auto repeated = anndatapp::Array::borrow_dense(storage,{2,2},{0,1}); rejects([&] { preprocess::log1p(repeated.edit()); });
    double v[]{1,2}; int idx[]{0,0}; long off[]{0,2};
    auto dup = anndatapp::Array::borrow_sparse(anndatapp::Layout::csr,1,2,2,v,idx,off);
    rejects([&] { preprocess::qc(dup.view()); });
    auto d = anndatapp::AnnData::empty(4,4); d.set_x(data());
    auto q = preprocess::qc(d.x(),{.percent_top={1}}); preprocess::annotate_qc(d,std::move(q));
    check(d.obs().contains("total_counts") && d.var().contains("pct_dropout_by_counts"),"QC annotation absent");
    auto malformed = preprocess::qc(d.x()); malformed.mean = anndatapp::Array::zeros<double>({1});
    rejects([&] { preprocess::annotate_qc(d,std::move(malformed)); });
    check(d.obs_column("total_counts").values().dense<double>().data()[0] == 4,"failed annotation changed prior dataset");
    auto norm = preprocess::normalize_total(d.edit_x()); preprocess::annotate_normalization(d,std::move(norm));
    preprocess::log1p(d.edit_x()); preprocess::annotate_log1p(d);
    check(d.obs().contains("size_factors") && d.uns().contains("log1p"),"transform annotation absent");
    auto params = preprocess::scale(d.edit_x()); preprocess::annotate_scale(d,std::move(params));
    check(d.var().contains("mean") && d.var().contains("std"),"scale annotation absent");
    d.set_obs_column("batch",anndatapp::Column::strings({"b","a","b","a"}));
    auto codes = preprocess::batch_codes(d,"batch"); auto* c = codes.view().dense<std::int64_t>().data();
    check(c[0] == c[2] && c[1] == c[3] && c[0] != c[1],"batch label encoding wrong");
}
} // namespace

int main() {
    try { qc_and_statistics(); normalization_and_logarithms(); scaling_and_strides(); contracts_and_annotations(); }
    catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
