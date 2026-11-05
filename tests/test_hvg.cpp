#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

import preprocess;

void check(bool x, const char* message) { if (!x) throw std::runtime_error(message); }
bool near(double a, double b, double tol = 2e-5) { return std::abs(a-b) < tol; }

int main() try {
    const double counts[] = {0,1,2,0,4, 1,1,1,0,2, 0,2,4,3,1, 2,0,1,0,1, 0,3,1,1,2, 1,0,0,0,5};
    std::vector<double> x(30);
    for (int k = 0; k < 30; ++k) x[k] = std::log1p(counts[k]);
    auto a = anndatapp::Array::borrow_dense(x.data(), {6,5}, {5,1});
    preprocess::HvgOptions o; o.n_bins = 1; o.n_top_genes = 2;
    auto r = preprocess::highly_variable_genes(a.view(), o);
    const double means[] = {.510826,.773190,.916291,.510826,1.252763};
    const double norm[] = {-.806067,-.299972,-.049954,1.715893,-.559900};
    const bool selected[] = {false,false,true,true,false};
    for (int j = 0; j < 5; ++j) {
        check(near(r.mean.view().dense<double>().data()[j], means[j]), "single mean");
        check(near(r.normalized_dispersion.view().dense<double>().data()[j], norm[j]), "single normalized dispersion");
        check(r.selected.view().dense<bool>().data()[j] == selected[j], "single selection");
    }
    std::int32_t batches[] = {0,0,0,1,1,1};
    auto ba = anndatapp::Array::borrow_dense(batches, {6}, {1}); o.batches = ba.view();
    r = preprocess::highly_variable_genes(a.view(), o);
    const double batch_means[] = {.490415,.770223,.857399,.490415,1.251628};
    const double batch_norm[] = {-.101539,-.060268,-.623373,.521206,.263974};
    const bool batch_selected[] = {false,false,false,true,true};
    for (int j = 0; j < 5; ++j) {
        check(near(r.mean.view().dense<double>().data()[j], batch_means[j]), "batch mean");
        check(near(r.normalized_dispersion.view().dense<double>().data()[j], batch_norm[j]), "batch normalized dispersion");
        check(r.selected.view().dense<bool>().data()[j] == batch_selected[j], "batch selection");
    }
    check(r.nbatches.view().dense<std::uint64_t>().data()[2] == 0, "batch counts");
    check(!r.intersection.view().dense<bool>().data()[3], "batch intersection");
    x[23] = 0;
    r = preprocess::highly_variable_genes(a.view(), o);
    check(near(r.mean.view().dense<double>().data()[3], .346574), "batch unexpressed gene mean");
    check(near(r.normalized_dispersion.view().dense<double>().data()[3], .655359), "batch unexpressed gene normalized dispersion");
    x[23] = std::log1p(1.0);
    o.batches.reset();
    std::vector<double> based(30), reversed(30);
    for (int i = 0; i < 6; ++i) for (int j = 0; j < 5; ++j) {
        based[i*5+j] = x[i*5+j]/std::log(2.0); reversed[(5-i)*5+j] = x[i*5+j];
    }
    auto base = anndatapp::Array::borrow_dense(based.data(), {6,5}, {5,1});
    o.log_base = 2; r = preprocess::highly_variable_genes(base.view(), o);
    for (int j = 0; j < 5; ++j) check(near(r.normalized_dispersion.view().dense<double>().data()[j], norm[j]), "log base");
    for (int k = 0; k < 30; ++k) based[k] = x[k]/std::log(.5);
    o.log_base = .5; r = preprocess::highly_variable_genes(base.view(), o);
    for (int j = 0; j < 5; ++j) check(near(r.normalized_dispersion.view().dense<double>().data()[j], norm[j]), "log base below one");
    o.log_base.reset();
    auto rev = anndatapp::Array::borrow_dense(reversed.data()+25, {6,5}, {-5,1});
    r = preprocess::highly_variable_genes(rev.view(), o);
    for (int j = 0; j < 5; ++j) check(near(r.normalized_dispersion.view().dense<double>().data()[j], norm[j]), "negative stride");
    std::vector<double> values; std::vector<std::int32_t> indices; std::vector<std::int64_t> offsets{0};
    for (int j = 0; j < 5; ++j) {
        for (int i = 0; i < 6; ++i) if (x[i*5+j] != 0) { values.push_back(x[i*5+j]); indices.push_back(i); }
        offsets.push_back(values.size());
    }
    auto csc = anndatapp::Array::borrow_sparse(anndatapp::Layout::csc, 6, 5, values.size(), values.data(), indices.data(), offsets.data());
    r = preprocess::highly_variable_genes(csc.view(), o);
    for (int j = 0; j < 5; ++j) check(near(r.normalized_dispersion.view().dense<double>().data()[j], norm[j]), "CSC normalized dispersion");
    values.clear(); indices.clear(); offsets.assign(1,0);
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 5; ++j) if (x[i*5+j] != 0) { values.push_back(x[i*5+j]); indices.push_back(j); }
        offsets.push_back(values.size());
    }
    auto csr = anndatapp::Array::borrow_sparse(anndatapp::Layout::csr, 6, 5, values.size(), values.data(), indices.data(), offsets.data());
    r = preprocess::highly_variable_genes(csr.view(), o);
    for (int j = 0; j < 5; ++j) check(near(r.normalized_dispersion.view().dense<double>().data()[j], norm[j]), "CSR normalized dispersion");
    double tied[] = {0,0,1, 1,1,1, 0,0,2, 2,2,2};
    for (auto& v : tied) v = std::log1p(v);
    auto t = anndatapp::Array::borrow_dense(tied, {4,3}, {3,1}); o.n_top_genes = 1;
    r = preprocess::highly_variable_genes(t.view(), o);
    auto* chosen = r.selected.view().dense<bool>().data();
    check(chosen[0] && chosen[1] && !chosen[2], "single batch ties select all at threshold");
    double tied_batches[24];
    for (int k = 0; k < 24; ++k) tied_batches[k] = tied[k%12];
    auto tb = anndatapp::Array::borrow_dense(tied_batches, {8,3}, {3,1});
    std::int32_t tie_codes[] = {0,0,0,0,1,1,1,1};
    auto tie_batch = anndatapp::Array::borrow_dense(tie_codes, {8}, {1});
    o.batches = tie_batch.view();
    r = preprocess::highly_variable_genes(tb.view(), o);
    chosen = r.selected.view().dense<bool>().data();
    check(chosen[0] && !chosen[1] && !chosen[2], "batched top ties use feature order");
    check(r.nbatches.view().dense<std::uint64_t>().data()[0] == 2 && r.nbatches.view().dense<std::uint64_t>().data()[1] == 2, "batched tie counts");
    o.batches.reset();
    auto invalid_batch = anndatapp::Array::borrow_dense(static_cast<std::int32_t*>(nullptr), {6}, {1}, anndatapp::Validation::none);
    o.batches = invalid_batch.view();
    bool invalid_rejected = false;
    try { preprocess::highly_variable_genes(a.view(), o); } catch (const std::exception&) { invalid_rejected = true; }
    check(invalid_rejected, "malformed batch vector rejected");
    o.batches.reset();
    std::vector<double> cr(600);
    for (int i = 0; i < 6; ++i) for (int j = 0; j < 100; ++j)
        cr[i*100+j] = (j+1)*.05 + ((j*17+i*11)%13)*.01;
    auto ca = anndatapp::Array::borrow_dense(cr.data(), {6,100}, {100,1});
    o.method = preprocess::HvgMethod::cell_ranger; o.n_top_genes = 5;
    r = preprocess::highly_variable_genes(ca.view(), o);
    const int probe[] = {0,5,25,50,75,95};
    const double crnorm[] = {4.309008168623969,-.08200719092867405,.37191876678591346,1.7138946054453674,0,1.0214361900158766};
    for (int k = 0; k < 6; ++k) check(near(r.normalized_dispersion.view().dense<double>().data()[probe[k]], crnorm[k], 1e-8), "Cell Ranger normalized dispersion");
    for (int j = 0; j < 100; ++j) check(r.selected.view().dense<bool>().data()[j] == (j == 0 || j == 1 || j == 40 || j == 66 || j == 92), "Cell Ranger selection");
    o.method = preprocess::HvgMethod::seurat;
    bool rejected = false; o.n_top_genes = 0;
    try { preprocess::highly_variable_genes(a.view(), o); } catch (const std::invalid_argument&) { rejected = true; }
    check(rejected, "reject zero top count");
    rejected = false; o.n_top_genes.reset(); o.method = preprocess::HvgMethod::cell_ranger;
    try { preprocess::highly_variable_genes(t.view(), o); } catch (const std::invalid_argument&) { rejected = true; }
    check(rejected, "reject duplicate Cell Ranger percentile edges");
    std::cout << "HVG tests passed\n";
    return 0;
} catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
