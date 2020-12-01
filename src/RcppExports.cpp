// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppEigen.h>
#include <Rcpp.h>

using namespace Rcpp;

// test_null
void test_null(Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> G, Rcpp::Nullable<Rcpp::List> l);
RcppExport SEXP _BGSL_test_null(SEXP GSEXP, SEXP lSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> >::type G(GSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<Rcpp::List> >::type l(lSEXP);
    test_null(G, l);
    return R_NilValue;
END_RCPP
}
// rGwish
Rcpp::List rGwish(Eigen::Matrix<unsigned int, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> G, double const& b, Eigen::MatrixXd const& D, Rcpp::String norm, Rcpp::Nullable<Rcpp::List> groups, unsigned int const& max_iter, long double const& threshold, int seed);
RcppExport SEXP _BGSL_rGwish(SEXP GSEXP, SEXP bSEXP, SEXP DSEXP, SEXP normSEXP, SEXP groupsSEXP, SEXP max_iterSEXP, SEXP thresholdSEXP, SEXP seedSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Eigen::Matrix<unsigned int, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> >::type G(GSEXP);
    Rcpp::traits::input_parameter< double const& >::type b(bSEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXd const& >::type D(DSEXP);
    Rcpp::traits::input_parameter< Rcpp::String >::type norm(normSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<Rcpp::List> >::type groups(groupsSEXP);
    Rcpp::traits::input_parameter< unsigned int const& >::type max_iter(max_iterSEXP);
    Rcpp::traits::input_parameter< long double const& >::type threshold(thresholdSEXP);
    Rcpp::traits::input_parameter< int >::type seed(seedSEXP);
    rcpp_result_gen = Rcpp::wrap(rGwish(G, b, D, norm, groups, max_iter, threshold, seed));
    return rcpp_result_gen;
END_RCPP
}
// log_Gconstant
long double log_Gconstant(Eigen::Matrix<unsigned int, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> G, double const& b, Eigen::MatrixXd const& D, unsigned int const& MCiteration, Rcpp::Nullable<Rcpp::List> groups, int seed);
RcppExport SEXP _BGSL_log_Gconstant(SEXP GSEXP, SEXP bSEXP, SEXP DSEXP, SEXP MCiterationSEXP, SEXP groupsSEXP, SEXP seedSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Eigen::Matrix<unsigned int, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> >::type G(GSEXP);
    Rcpp::traits::input_parameter< double const& >::type b(bSEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXd const& >::type D(DSEXP);
    Rcpp::traits::input_parameter< unsigned int const& >::type MCiteration(MCiterationSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<Rcpp::List> >::type groups(groupsSEXP);
    Rcpp::traits::input_parameter< int >::type seed(seedSEXP);
    rcpp_result_gen = Rcpp::wrap(log_Gconstant(G, b, D, MCiteration, groups, seed));
    return rcpp_result_gen;
END_RCPP
}
// GGM_sim_sampling
Rcpp::List GGM_sim_sampling(int const& p, int const& n, int const& niter, int const& burnin, double const& thin, Eigen::MatrixXd const& D, double const& b, int const& MCprior, int const& MCpost, Rcpp::String form, Rcpp::String prior, Rcpp::String algo, int const& n_groups, int seed, double sparsity, double const& Gprior, double const& sigmaG, double const& paddrm);
RcppExport SEXP _BGSL_GGM_sim_sampling(SEXP pSEXP, SEXP nSEXP, SEXP niterSEXP, SEXP burninSEXP, SEXP thinSEXP, SEXP DSEXP, SEXP bSEXP, SEXP MCpriorSEXP, SEXP MCpostSEXP, SEXP formSEXP, SEXP priorSEXP, SEXP algoSEXP, SEXP n_groupsSEXP, SEXP seedSEXP, SEXP sparsitySEXP, SEXP GpriorSEXP, SEXP sigmaGSEXP, SEXP paddrmSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int const& >::type p(pSEXP);
    Rcpp::traits::input_parameter< int const& >::type n(nSEXP);
    Rcpp::traits::input_parameter< int const& >::type niter(niterSEXP);
    Rcpp::traits::input_parameter< int const& >::type burnin(burninSEXP);
    Rcpp::traits::input_parameter< double const& >::type thin(thinSEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXd const& >::type D(DSEXP);
    Rcpp::traits::input_parameter< double const& >::type b(bSEXP);
    Rcpp::traits::input_parameter< int const& >::type MCprior(MCpriorSEXP);
    Rcpp::traits::input_parameter< int const& >::type MCpost(MCpostSEXP);
    Rcpp::traits::input_parameter< Rcpp::String >::type form(formSEXP);
    Rcpp::traits::input_parameter< Rcpp::String >::type prior(priorSEXP);
    Rcpp::traits::input_parameter< Rcpp::String >::type algo(algoSEXP);
    Rcpp::traits::input_parameter< int const& >::type n_groups(n_groupsSEXP);
    Rcpp::traits::input_parameter< int >::type seed(seedSEXP);
    Rcpp::traits::input_parameter< double >::type sparsity(sparsitySEXP);
    Rcpp::traits::input_parameter< double const& >::type Gprior(GpriorSEXP);
    Rcpp::traits::input_parameter< double const& >::type sigmaG(sigmaGSEXP);
    Rcpp::traits::input_parameter< double const& >::type paddrm(paddrmSEXP);
    rcpp_result_gen = Rcpp::wrap(GGM_sim_sampling(p, n, niter, burnin, thin, D, b, MCprior, MCpost, form, prior, algo, n_groups, seed, sparsity, Gprior, sigmaG, paddrm));
    return rcpp_result_gen;
END_RCPP
}
// GGM_sampling
Rcpp::List GGM_sampling(Eigen::MatrixXd const& data, int const& p, int const& n, int const& niter, int const& burnin, double const& thin, Eigen::MatrixXd const& D, double const& b, int const& MCprior, int const& MCpost, Rcpp::String form, Rcpp::String prior, Rcpp::String algo, int const& n_groups, int seed, double const& Gprior, double const& sigmaG, double const& paddrm);
RcppExport SEXP _BGSL_GGM_sampling(SEXP dataSEXP, SEXP pSEXP, SEXP nSEXP, SEXP niterSEXP, SEXP burninSEXP, SEXP thinSEXP, SEXP DSEXP, SEXP bSEXP, SEXP MCpriorSEXP, SEXP MCpostSEXP, SEXP formSEXP, SEXP priorSEXP, SEXP algoSEXP, SEXP n_groupsSEXP, SEXP seedSEXP, SEXP GpriorSEXP, SEXP sigmaGSEXP, SEXP paddrmSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Eigen::MatrixXd const& >::type data(dataSEXP);
    Rcpp::traits::input_parameter< int const& >::type p(pSEXP);
    Rcpp::traits::input_parameter< int const& >::type n(nSEXP);
    Rcpp::traits::input_parameter< int const& >::type niter(niterSEXP);
    Rcpp::traits::input_parameter< int const& >::type burnin(burninSEXP);
    Rcpp::traits::input_parameter< double const& >::type thin(thinSEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXd const& >::type D(DSEXP);
    Rcpp::traits::input_parameter< double const& >::type b(bSEXP);
    Rcpp::traits::input_parameter< int const& >::type MCprior(MCpriorSEXP);
    Rcpp::traits::input_parameter< int const& >::type MCpost(MCpostSEXP);
    Rcpp::traits::input_parameter< Rcpp::String >::type form(formSEXP);
    Rcpp::traits::input_parameter< Rcpp::String >::type prior(priorSEXP);
    Rcpp::traits::input_parameter< Rcpp::String >::type algo(algoSEXP);
    Rcpp::traits::input_parameter< int const& >::type n_groups(n_groupsSEXP);
    Rcpp::traits::input_parameter< int >::type seed(seedSEXP);
    Rcpp::traits::input_parameter< double const& >::type Gprior(GpriorSEXP);
    Rcpp::traits::input_parameter< double const& >::type sigmaG(sigmaGSEXP);
    Rcpp::traits::input_parameter< double const& >::type paddrm(paddrmSEXP);
    rcpp_result_gen = Rcpp::wrap(GGM_sampling(data, p, n, niter, burnin, thin, D, b, MCprior, MCpost, form, prior, algo, n_groups, seed, Gprior, sigmaG, paddrm));
    return rcpp_result_gen;
END_RCPP
}
// Create_RandomGraph
Eigen::Matrix<unsigned int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Create_RandomGraph(int const& p, int const& n_groups, Rcpp::String form, Rcpp::Nullable<Rcpp::List> groups, double sparsity, int seed);
RcppExport SEXP _BGSL_Create_RandomGraph(SEXP pSEXP, SEXP n_groupsSEXP, SEXP formSEXP, SEXP groupsSEXP, SEXP sparsitySEXP, SEXP seedSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int const& >::type p(pSEXP);
    Rcpp::traits::input_parameter< int const& >::type n_groups(n_groupsSEXP);
    Rcpp::traits::input_parameter< Rcpp::String >::type form(formSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<Rcpp::List> >::type groups(groupsSEXP);
    Rcpp::traits::input_parameter< double >::type sparsity(sparsitySEXP);
    Rcpp::traits::input_parameter< int >::type seed(seedSEXP);
    rcpp_result_gen = Rcpp::wrap(Create_RandomGraph(p, n_groups, form, groups, sparsity, seed));
    return rcpp_result_gen;
END_RCPP
}
// GraphTest
void GraphTest();
RcppExport SEXP _BGSL_GraphTest() {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    GraphTest();
    return R_NilValue;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_BGSL_test_null", (DL_FUNC) &_BGSL_test_null, 2},
    {"_BGSL_rGwish", (DL_FUNC) &_BGSL_rGwish, 8},
    {"_BGSL_log_Gconstant", (DL_FUNC) &_BGSL_log_Gconstant, 6},
    {"_BGSL_GGM_sim_sampling", (DL_FUNC) &_BGSL_GGM_sim_sampling, 18},
    {"_BGSL_GGM_sampling", (DL_FUNC) &_BGSL_GGM_sampling, 18},
    {"_BGSL_Create_RandomGraph", (DL_FUNC) &_BGSL_Create_RandomGraph, 6},
    {"_BGSL_GraphTest", (DL_FUNC) &_BGSL_GraphTest, 0},
    {NULL, NULL, 0}
};

RcppExport void R_init_BGSL(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
