// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppEigen.h>
#include <Rcpp.h>

using namespace Rcpp;

// rGwish
Rcpp::List rGwish(Eigen::Matrix<unsigned int, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> const& G, double const& b, Eigen::MatrixXd& D, Rcpp::String norm, Rcpp::String form, Rcpp::Nullable<Rcpp::List> groups, bool check_structure, unsigned int const& max_iter, long double const& threshold_check, long double const& threshold_conv, int seed);
RcppExport SEXP _BGSL_rGwish(SEXP GSEXP, SEXP bSEXP, SEXP DSEXP, SEXP normSEXP, SEXP formSEXP, SEXP groupsSEXP, SEXP check_structureSEXP, SEXP max_iterSEXP, SEXP threshold_checkSEXP, SEXP threshold_convSEXP, SEXP seedSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Eigen::Matrix<unsigned int, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> const& >::type G(GSEXP);
    Rcpp::traits::input_parameter< double const& >::type b(bSEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXd& >::type D(DSEXP);
    Rcpp::traits::input_parameter< Rcpp::String >::type norm(normSEXP);
    Rcpp::traits::input_parameter< Rcpp::String >::type form(formSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<Rcpp::List> >::type groups(groupsSEXP);
    Rcpp::traits::input_parameter< bool >::type check_structure(check_structureSEXP);
    Rcpp::traits::input_parameter< unsigned int const& >::type max_iter(max_iterSEXP);
    Rcpp::traits::input_parameter< long double const& >::type threshold_check(threshold_checkSEXP);
    Rcpp::traits::input_parameter< long double const& >::type threshold_conv(threshold_convSEXP);
    Rcpp::traits::input_parameter< int >::type seed(seedSEXP);
    rcpp_result_gen = Rcpp::wrap(rGwish(G, b, D, norm, form, groups, check_structure, max_iter, threshold_check, threshold_conv, seed));
    return rcpp_result_gen;
END_RCPP
}
// log_Gconstant
long double log_Gconstant(Eigen::Matrix<unsigned int, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> const& G, double const& b, Eigen::MatrixXd const& D, unsigned int const& MCiteration, Rcpp::Nullable<Rcpp::List> groups, int seed);
RcppExport SEXP _BGSL_log_Gconstant(SEXP GSEXP, SEXP bSEXP, SEXP DSEXP, SEXP MCiterationSEXP, SEXP groupsSEXP, SEXP seedSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Eigen::Matrix<unsigned int, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> const& >::type G(GSEXP);
    Rcpp::traits::input_parameter< double const& >::type b(bSEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXd const& >::type D(DSEXP);
    Rcpp::traits::input_parameter< unsigned int const& >::type MCiteration(MCiterationSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<Rcpp::List> >::type groups(groupsSEXP);
    Rcpp::traits::input_parameter< int >::type seed(seedSEXP);
    rcpp_result_gen = Rcpp::wrap(log_Gconstant(G, b, D, MCiteration, groups, seed));
    return rcpp_result_gen;
END_RCPP
}
// Create_RandomGraph
Rcpp::List Create_RandomGraph(int const& p, int const& n_groups, Rcpp::String form, Rcpp::Nullable<Rcpp::List> groups, double sparsity, int seed);
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
// rmvnormal
Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> rmvnormal(Eigen::VectorXd mean, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> Mat, bool isPrec, bool isChol, bool isUpper, int seed);
RcppExport SEXP _BGSL_rmvnormal(SEXP meanSEXP, SEXP MatSEXP, SEXP isPrecSEXP, SEXP isCholSEXP, SEXP isUpperSEXP, SEXP seedSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Eigen::VectorXd >::type mean(meanSEXP);
    Rcpp::traits::input_parameter< Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> >::type Mat(MatSEXP);
    Rcpp::traits::input_parameter< bool >::type isPrec(isPrecSEXP);
    Rcpp::traits::input_parameter< bool >::type isChol(isCholSEXP);
    Rcpp::traits::input_parameter< bool >::type isUpper(isUpperSEXP);
    Rcpp::traits::input_parameter< int >::type seed(seedSEXP);
    rcpp_result_gen = Rcpp::wrap(rmvnormal(mean, Mat, isPrec, isChol, isUpper, seed));
    return rcpp_result_gen;
END_RCPP
}
// rwishart
Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> rwishart(double const& b, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> D, bool isChol, bool isUpper, int seed);
RcppExport SEXP _BGSL_rwishart(SEXP bSEXP, SEXP DSEXP, SEXP isCholSEXP, SEXP isUpperSEXP, SEXP seedSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< double const& >::type b(bSEXP);
    Rcpp::traits::input_parameter< Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> >::type D(DSEXP);
    Rcpp::traits::input_parameter< bool >::type isChol(isCholSEXP);
    Rcpp::traits::input_parameter< bool >::type isUpper(isUpperSEXP);
    Rcpp::traits::input_parameter< int >::type seed(seedSEXP);
    rcpp_result_gen = Rcpp::wrap(rwishart(b, D, isChol, isUpper, seed));
    return rcpp_result_gen;
END_RCPP
}
// rnormal
double rnormal(double const& mean, double const& sd, int seed);
RcppExport SEXP _BGSL_rnormal(SEXP meanSEXP, SEXP sdSEXP, SEXP seedSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< double const& >::type mean(meanSEXP);
    Rcpp::traits::input_parameter< double const& >::type sd(sdSEXP);
    Rcpp::traits::input_parameter< int >::type seed(seedSEXP);
    rcpp_result_gen = Rcpp::wrap(rnormal(mean, sd, seed));
    return rcpp_result_gen;
END_RCPP
}
// Generate_Basis
Rcpp::List Generate_Basis(int const& n_basis, Rcpp::NumericVector range, int n_points, Rcpp::NumericVector grid_points, int order);
RcppExport SEXP _BGSL_Generate_Basis(SEXP n_basisSEXP, SEXP rangeSEXP, SEXP n_pointsSEXP, SEXP grid_pointsSEXP, SEXP orderSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int const& >::type n_basis(n_basisSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type range(rangeSEXP);
    Rcpp::traits::input_parameter< int >::type n_points(n_pointsSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type grid_points(grid_pointsSEXP);
    Rcpp::traits::input_parameter< int >::type order(orderSEXP);
    rcpp_result_gen = Rcpp::wrap(Generate_Basis(n_basis, range, n_points, grid_points, order));
    return rcpp_result_gen;
END_RCPP
}
// Generate_Basis_derivatives
Rcpp::List Generate_Basis_derivatives(int const& n_basis, int const& nderiv, Rcpp::NumericVector range, int n_points, Rcpp::NumericVector grid_points, int order);
RcppExport SEXP _BGSL_Generate_Basis_derivatives(SEXP n_basisSEXP, SEXP nderivSEXP, SEXP rangeSEXP, SEXP n_pointsSEXP, SEXP grid_pointsSEXP, SEXP orderSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int const& >::type n_basis(n_basisSEXP);
    Rcpp::traits::input_parameter< int const& >::type nderiv(nderivSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type range(rangeSEXP);
    Rcpp::traits::input_parameter< int >::type n_points(n_pointsSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type grid_points(grid_pointsSEXP);
    Rcpp::traits::input_parameter< int >::type order(orderSEXP);
    rcpp_result_gen = Rcpp::wrap(Generate_Basis_derivatives(n_basis, nderiv, range, n_points, grid_points, order));
    return rcpp_result_gen;
END_RCPP
}
// Read_InfoFile
Rcpp::List Read_InfoFile(Rcpp::String const& file_name);
RcppExport SEXP _BGSL_Read_InfoFile(SEXP file_nameSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::String const& >::type file_name(file_nameSEXP);
    rcpp_result_gen = Rcpp::wrap(Read_InfoFile(file_name));
    return rcpp_result_gen;
END_RCPP
}
// Compute_Quantiles
Rcpp::List Compute_Quantiles(Rcpp::String const& file_name, unsigned int const& p, unsigned int const& n, unsigned int const& stored_iterG, unsigned int const& stored_iter, bool Beta, bool Mu, bool TauEps, bool Precision, unsigned int const& prec_elem, double const& lower_qtl, double const& upper_qtl);
RcppExport SEXP _BGSL_Compute_Quantiles(SEXP file_nameSEXP, SEXP pSEXP, SEXP nSEXP, SEXP stored_iterGSEXP, SEXP stored_iterSEXP, SEXP BetaSEXP, SEXP MuSEXP, SEXP TauEpsSEXP, SEXP PrecisionSEXP, SEXP prec_elemSEXP, SEXP lower_qtlSEXP, SEXP upper_qtlSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::String const& >::type file_name(file_nameSEXP);
    Rcpp::traits::input_parameter< unsigned int const& >::type p(pSEXP);
    Rcpp::traits::input_parameter< unsigned int const& >::type n(nSEXP);
    Rcpp::traits::input_parameter< unsigned int const& >::type stored_iterG(stored_iterGSEXP);
    Rcpp::traits::input_parameter< unsigned int const& >::type stored_iter(stored_iterSEXP);
    Rcpp::traits::input_parameter< bool >::type Beta(BetaSEXP);
    Rcpp::traits::input_parameter< bool >::type Mu(MuSEXP);
    Rcpp::traits::input_parameter< bool >::type TauEps(TauEpsSEXP);
    Rcpp::traits::input_parameter< bool >::type Precision(PrecisionSEXP);
    Rcpp::traits::input_parameter< unsigned int const& >::type prec_elem(prec_elemSEXP);
    Rcpp::traits::input_parameter< double const& >::type lower_qtl(lower_qtlSEXP);
    Rcpp::traits::input_parameter< double const& >::type upper_qtl(upper_qtlSEXP);
    rcpp_result_gen = Rcpp::wrap(Compute_Quantiles(file_name, p, n, stored_iterG, stored_iter, Beta, Mu, TauEps, Precision, prec_elem, lower_qtl, upper_qtl));
    return rcpp_result_gen;
END_RCPP
}
// Compute_PosteriorMeans
Rcpp::List Compute_PosteriorMeans(Rcpp::String const& file_name, unsigned int const& p, unsigned int const& n, unsigned int const& stored_iterG, unsigned int const& stored_iter, bool Beta, bool Mu, bool TauEps, bool Precision, unsigned int const& prec_elem);
RcppExport SEXP _BGSL_Compute_PosteriorMeans(SEXP file_nameSEXP, SEXP pSEXP, SEXP nSEXP, SEXP stored_iterGSEXP, SEXP stored_iterSEXP, SEXP BetaSEXP, SEXP MuSEXP, SEXP TauEpsSEXP, SEXP PrecisionSEXP, SEXP prec_elemSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::String const& >::type file_name(file_nameSEXP);
    Rcpp::traits::input_parameter< unsigned int const& >::type p(pSEXP);
    Rcpp::traits::input_parameter< unsigned int const& >::type n(nSEXP);
    Rcpp::traits::input_parameter< unsigned int const& >::type stored_iterG(stored_iterGSEXP);
    Rcpp::traits::input_parameter< unsigned int const& >::type stored_iter(stored_iterSEXP);
    Rcpp::traits::input_parameter< bool >::type Beta(BetaSEXP);
    Rcpp::traits::input_parameter< bool >::type Mu(MuSEXP);
    Rcpp::traits::input_parameter< bool >::type TauEps(TauEpsSEXP);
    Rcpp::traits::input_parameter< bool >::type Precision(PrecisionSEXP);
    Rcpp::traits::input_parameter< unsigned int const& >::type prec_elem(prec_elemSEXP);
    rcpp_result_gen = Rcpp::wrap(Compute_PosteriorMeans(file_name, p, n, stored_iterG, stored_iter, Beta, Mu, TauEps, Precision, prec_elem));
    return rcpp_result_gen;
END_RCPP
}
// Extract_Chain
Eigen::VectorXd Extract_Chain(Rcpp::String const& file_name, Rcpp::String const& variable, unsigned int const& stored_iter, unsigned int const& p, unsigned int const& n, unsigned int index1, unsigned int index2, unsigned int const& prec_elem);
RcppExport SEXP _BGSL_Extract_Chain(SEXP file_nameSEXP, SEXP variableSEXP, SEXP stored_iterSEXP, SEXP pSEXP, SEXP nSEXP, SEXP index1SEXP, SEXP index2SEXP, SEXP prec_elemSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::String const& >::type file_name(file_nameSEXP);
    Rcpp::traits::input_parameter< Rcpp::String const& >::type variable(variableSEXP);
    Rcpp::traits::input_parameter< unsigned int const& >::type stored_iter(stored_iterSEXP);
    Rcpp::traits::input_parameter< unsigned int const& >::type p(pSEXP);
    Rcpp::traits::input_parameter< unsigned int const& >::type n(nSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type index1(index1SEXP);
    Rcpp::traits::input_parameter< unsigned int >::type index2(index2SEXP);
    Rcpp::traits::input_parameter< unsigned int const& >::type prec_elem(prec_elemSEXP);
    rcpp_result_gen = Rcpp::wrap(Extract_Chain(file_name, variable, stored_iter, p, n, index1, index2, prec_elem));
    return rcpp_result_gen;
END_RCPP
}
// Summary_Graph
Rcpp::List Summary_Graph(Rcpp::String const& file_name, unsigned int const& stored_iterG, unsigned int const& p, Rcpp::Nullable<Rcpp::List> groups);
RcppExport SEXP _BGSL_Summary_Graph(SEXP file_nameSEXP, SEXP stored_iterGSEXP, SEXP pSEXP, SEXP groupsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::String const& >::type file_name(file_nameSEXP);
    Rcpp::traits::input_parameter< unsigned int const& >::type stored_iterG(stored_iterGSEXP);
    Rcpp::traits::input_parameter< unsigned int const& >::type p(pSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<Rcpp::List> >::type groups(groupsSEXP);
    rcpp_result_gen = Rcpp::wrap(Summary_Graph(file_name, stored_iterG, p, groups));
    return rcpp_result_gen;
END_RCPP
}
// SimulateData_GGM_c
Rcpp::List SimulateData_GGM_c(unsigned int const& p, unsigned int const& n, unsigned int const& n_groups, Rcpp::String const& form, Rcpp::String const& graph, Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> const& adj_mat, unsigned int seed, bool mean_null, double const& sparsity, Rcpp::Nullable<Rcpp::List> groups);
RcppExport SEXP _BGSL_SimulateData_GGM_c(SEXP pSEXP, SEXP nSEXP, SEXP n_groupsSEXP, SEXP formSEXP, SEXP graphSEXP, SEXP adj_matSEXP, SEXP seedSEXP, SEXP mean_nullSEXP, SEXP sparsitySEXP, SEXP groupsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< unsigned int const& >::type p(pSEXP);
    Rcpp::traits::input_parameter< unsigned int const& >::type n(nSEXP);
    Rcpp::traits::input_parameter< unsigned int const& >::type n_groups(n_groupsSEXP);
    Rcpp::traits::input_parameter< Rcpp::String const& >::type form(formSEXP);
    Rcpp::traits::input_parameter< Rcpp::String const& >::type graph(graphSEXP);
    Rcpp::traits::input_parameter< Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> const& >::type adj_mat(adj_matSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type seed(seedSEXP);
    Rcpp::traits::input_parameter< bool >::type mean_null(mean_nullSEXP);
    Rcpp::traits::input_parameter< double const& >::type sparsity(sparsitySEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<Rcpp::List> >::type groups(groupsSEXP);
    rcpp_result_gen = Rcpp::wrap(SimulateData_GGM_c(p, n, n_groups, form, graph, adj_mat, seed, mean_null, sparsity, groups));
    return rcpp_result_gen;
END_RCPP
}
// CreateGroups
Rcpp::List CreateGroups(unsigned int const& p, unsigned int const& n_groups);
RcppExport SEXP _BGSL_CreateGroups(SEXP pSEXP, SEXP n_groupsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< unsigned int const& >::type p(pSEXP);
    Rcpp::traits::input_parameter< unsigned int const& >::type n_groups(n_groupsSEXP);
    rcpp_result_gen = Rcpp::wrap(CreateGroups(p, n_groups));
    return rcpp_result_gen;
END_RCPP
}
// GGM_sampling_c
Rcpp::List GGM_sampling_c(Eigen::MatrixXd const& data, int const& p, int const& n, int const& niter, int const& burnin, double const& thin, Rcpp::String file_name, Eigen::MatrixXd D, double const& b, Eigen::Matrix<unsigned int, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> const& G0, Eigen::MatrixXd const& K0, int const& MCprior, int const& MCpost, double const& threshold, Rcpp::String form, Rcpp::String prior, Rcpp::String algo, Rcpp::Nullable<Rcpp::List> groups, int seed, double const& Gprior, double const& sigmaG, double const& paddrm, bool print_info);
RcppExport SEXP _BGSL_GGM_sampling_c(SEXP dataSEXP, SEXP pSEXP, SEXP nSEXP, SEXP niterSEXP, SEXP burninSEXP, SEXP thinSEXP, SEXP file_nameSEXP, SEXP DSEXP, SEXP bSEXP, SEXP G0SEXP, SEXP K0SEXP, SEXP MCpriorSEXP, SEXP MCpostSEXP, SEXP thresholdSEXP, SEXP formSEXP, SEXP priorSEXP, SEXP algoSEXP, SEXP groupsSEXP, SEXP seedSEXP, SEXP GpriorSEXP, SEXP sigmaGSEXP, SEXP paddrmSEXP, SEXP print_infoSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Eigen::MatrixXd const& >::type data(dataSEXP);
    Rcpp::traits::input_parameter< int const& >::type p(pSEXP);
    Rcpp::traits::input_parameter< int const& >::type n(nSEXP);
    Rcpp::traits::input_parameter< int const& >::type niter(niterSEXP);
    Rcpp::traits::input_parameter< int const& >::type burnin(burninSEXP);
    Rcpp::traits::input_parameter< double const& >::type thin(thinSEXP);
    Rcpp::traits::input_parameter< Rcpp::String >::type file_name(file_nameSEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXd >::type D(DSEXP);
    Rcpp::traits::input_parameter< double const& >::type b(bSEXP);
    Rcpp::traits::input_parameter< Eigen::Matrix<unsigned int, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> const& >::type G0(G0SEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXd const& >::type K0(K0SEXP);
    Rcpp::traits::input_parameter< int const& >::type MCprior(MCpriorSEXP);
    Rcpp::traits::input_parameter< int const& >::type MCpost(MCpostSEXP);
    Rcpp::traits::input_parameter< double const& >::type threshold(thresholdSEXP);
    Rcpp::traits::input_parameter< Rcpp::String >::type form(formSEXP);
    Rcpp::traits::input_parameter< Rcpp::String >::type prior(priorSEXP);
    Rcpp::traits::input_parameter< Rcpp::String >::type algo(algoSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<Rcpp::List> >::type groups(groupsSEXP);
    Rcpp::traits::input_parameter< int >::type seed(seedSEXP);
    Rcpp::traits::input_parameter< double const& >::type Gprior(GpriorSEXP);
    Rcpp::traits::input_parameter< double const& >::type sigmaG(sigmaGSEXP);
    Rcpp::traits::input_parameter< double const& >::type paddrm(paddrmSEXP);
    Rcpp::traits::input_parameter< bool >::type print_info(print_infoSEXP);
    rcpp_result_gen = Rcpp::wrap(GGM_sampling_c(data, p, n, niter, burnin, thin, file_name, D, b, G0, K0, MCprior, MCpost, threshold, form, prior, algo, groups, seed, Gprior, sigmaG, paddrm, print_info));
    return rcpp_result_gen;
END_RCPP
}
// FLM_sampling_c
Rcpp::List FLM_sampling_c(Eigen::MatrixXd const& data, int const& niter, int const& burnin, double const& thin, Eigen::MatrixXd const& BaseMat, Eigen::Matrix<unsigned int, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> G, Eigen::MatrixXd const& Beta0, Eigen::VectorXd const& mu0, double const& tau_eps0, Eigen::VectorXd const& tauK0, Eigen::MatrixXd const& K0, double const& a_tau_eps, double const& b_tau_eps, double const& sigmamu, double const& aTauK, double const& bTauK, double const& bK, Eigen::MatrixXd const& DK, Rcpp::String file_name, bool diagonal_graph, double const& threshold_GWish, int seed, bool print_info);
RcppExport SEXP _BGSL_FLM_sampling_c(SEXP dataSEXP, SEXP niterSEXP, SEXP burninSEXP, SEXP thinSEXP, SEXP BaseMatSEXP, SEXP GSEXP, SEXP Beta0SEXP, SEXP mu0SEXP, SEXP tau_eps0SEXP, SEXP tauK0SEXP, SEXP K0SEXP, SEXP a_tau_epsSEXP, SEXP b_tau_epsSEXP, SEXP sigmamuSEXP, SEXP aTauKSEXP, SEXP bTauKSEXP, SEXP bKSEXP, SEXP DKSEXP, SEXP file_nameSEXP, SEXP diagonal_graphSEXP, SEXP threshold_GWishSEXP, SEXP seedSEXP, SEXP print_infoSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Eigen::MatrixXd const& >::type data(dataSEXP);
    Rcpp::traits::input_parameter< int const& >::type niter(niterSEXP);
    Rcpp::traits::input_parameter< int const& >::type burnin(burninSEXP);
    Rcpp::traits::input_parameter< double const& >::type thin(thinSEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXd const& >::type BaseMat(BaseMatSEXP);
    Rcpp::traits::input_parameter< Eigen::Matrix<unsigned int, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> >::type G(GSEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXd const& >::type Beta0(Beta0SEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXd const& >::type mu0(mu0SEXP);
    Rcpp::traits::input_parameter< double const& >::type tau_eps0(tau_eps0SEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXd const& >::type tauK0(tauK0SEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXd const& >::type K0(K0SEXP);
    Rcpp::traits::input_parameter< double const& >::type a_tau_eps(a_tau_epsSEXP);
    Rcpp::traits::input_parameter< double const& >::type b_tau_eps(b_tau_epsSEXP);
    Rcpp::traits::input_parameter< double const& >::type sigmamu(sigmamuSEXP);
    Rcpp::traits::input_parameter< double const& >::type aTauK(aTauKSEXP);
    Rcpp::traits::input_parameter< double const& >::type bTauK(bTauKSEXP);
    Rcpp::traits::input_parameter< double const& >::type bK(bKSEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXd const& >::type DK(DKSEXP);
    Rcpp::traits::input_parameter< Rcpp::String >::type file_name(file_nameSEXP);
    Rcpp::traits::input_parameter< bool >::type diagonal_graph(diagonal_graphSEXP);
    Rcpp::traits::input_parameter< double const& >::type threshold_GWish(threshold_GWishSEXP);
    Rcpp::traits::input_parameter< int >::type seed(seedSEXP);
    Rcpp::traits::input_parameter< bool >::type print_info(print_infoSEXP);
    rcpp_result_gen = Rcpp::wrap(FLM_sampling_c(data, niter, burnin, thin, BaseMat, G, Beta0, mu0, tau_eps0, tauK0, K0, a_tau_eps, b_tau_eps, sigmamu, aTauK, bTauK, bK, DK, file_name, diagonal_graph, threshold_GWish, seed, print_info));
    return rcpp_result_gen;
END_RCPP
}
// FGM_sampling_c
Rcpp::List FGM_sampling_c(Eigen::MatrixXd const& data, int const& niter, int const& burnin, double const& thin, double const& thinG, Eigen::MatrixXd const& BaseMat, Rcpp::String const& file_name, Eigen::MatrixXd const& Beta0, Eigen::VectorXd const& mu0, double const& tau_eps0, Eigen::Matrix<unsigned int, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> const& G0, Eigen::MatrixXd const& K0, double const& a_tau_eps, double const& b_tau_eps, double const& sigmamu, double const& bK, Eigen::MatrixXd const& DK, double const& sigmaG, double const& paddrm, double const& Gprior, int const& MCprior, int const& MCpost, double const& threshold, Rcpp::String form, Rcpp::String prior, Rcpp::String algo, Rcpp::Nullable<Rcpp::List> groups, int seed, bool print_info);
RcppExport SEXP _BGSL_FGM_sampling_c(SEXP dataSEXP, SEXP niterSEXP, SEXP burninSEXP, SEXP thinSEXP, SEXP thinGSEXP, SEXP BaseMatSEXP, SEXP file_nameSEXP, SEXP Beta0SEXP, SEXP mu0SEXP, SEXP tau_eps0SEXP, SEXP G0SEXP, SEXP K0SEXP, SEXP a_tau_epsSEXP, SEXP b_tau_epsSEXP, SEXP sigmamuSEXP, SEXP bKSEXP, SEXP DKSEXP, SEXP sigmaGSEXP, SEXP paddrmSEXP, SEXP GpriorSEXP, SEXP MCpriorSEXP, SEXP MCpostSEXP, SEXP thresholdSEXP, SEXP formSEXP, SEXP priorSEXP, SEXP algoSEXP, SEXP groupsSEXP, SEXP seedSEXP, SEXP print_infoSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Eigen::MatrixXd const& >::type data(dataSEXP);
    Rcpp::traits::input_parameter< int const& >::type niter(niterSEXP);
    Rcpp::traits::input_parameter< int const& >::type burnin(burninSEXP);
    Rcpp::traits::input_parameter< double const& >::type thin(thinSEXP);
    Rcpp::traits::input_parameter< double const& >::type thinG(thinGSEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXd const& >::type BaseMat(BaseMatSEXP);
    Rcpp::traits::input_parameter< Rcpp::String const& >::type file_name(file_nameSEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXd const& >::type Beta0(Beta0SEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXd const& >::type mu0(mu0SEXP);
    Rcpp::traits::input_parameter< double const& >::type tau_eps0(tau_eps0SEXP);
    Rcpp::traits::input_parameter< Eigen::Matrix<unsigned int, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> const& >::type G0(G0SEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXd const& >::type K0(K0SEXP);
    Rcpp::traits::input_parameter< double const& >::type a_tau_eps(a_tau_epsSEXP);
    Rcpp::traits::input_parameter< double const& >::type b_tau_eps(b_tau_epsSEXP);
    Rcpp::traits::input_parameter< double const& >::type sigmamu(sigmamuSEXP);
    Rcpp::traits::input_parameter< double const& >::type bK(bKSEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXd const& >::type DK(DKSEXP);
    Rcpp::traits::input_parameter< double const& >::type sigmaG(sigmaGSEXP);
    Rcpp::traits::input_parameter< double const& >::type paddrm(paddrmSEXP);
    Rcpp::traits::input_parameter< double const& >::type Gprior(GpriorSEXP);
    Rcpp::traits::input_parameter< int const& >::type MCprior(MCpriorSEXP);
    Rcpp::traits::input_parameter< int const& >::type MCpost(MCpostSEXP);
    Rcpp::traits::input_parameter< double const& >::type threshold(thresholdSEXP);
    Rcpp::traits::input_parameter< Rcpp::String >::type form(formSEXP);
    Rcpp::traits::input_parameter< Rcpp::String >::type prior(priorSEXP);
    Rcpp::traits::input_parameter< Rcpp::String >::type algo(algoSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<Rcpp::List> >::type groups(groupsSEXP);
    Rcpp::traits::input_parameter< int >::type seed(seedSEXP);
    Rcpp::traits::input_parameter< bool >::type print_info(print_infoSEXP);
    rcpp_result_gen = Rcpp::wrap(FGM_sampling_c(data, niter, burnin, thin, thinG, BaseMat, file_name, Beta0, mu0, tau_eps0, G0, K0, a_tau_eps, b_tau_eps, sigmamu, bK, DK, sigmaG, paddrm, Gprior, MCprior, MCpost, threshold, form, prior, algo, groups, seed, print_info));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_BGSL_rGwish", (DL_FUNC) &_BGSL_rGwish, 11},
    {"_BGSL_log_Gconstant", (DL_FUNC) &_BGSL_log_Gconstant, 6},
    {"_BGSL_Create_RandomGraph", (DL_FUNC) &_BGSL_Create_RandomGraph, 6},
    {"_BGSL_rmvnormal", (DL_FUNC) &_BGSL_rmvnormal, 6},
    {"_BGSL_rwishart", (DL_FUNC) &_BGSL_rwishart, 5},
    {"_BGSL_rnormal", (DL_FUNC) &_BGSL_rnormal, 3},
    {"_BGSL_Generate_Basis", (DL_FUNC) &_BGSL_Generate_Basis, 5},
    {"_BGSL_Generate_Basis_derivatives", (DL_FUNC) &_BGSL_Generate_Basis_derivatives, 6},
    {"_BGSL_Read_InfoFile", (DL_FUNC) &_BGSL_Read_InfoFile, 1},
    {"_BGSL_Compute_Quantiles", (DL_FUNC) &_BGSL_Compute_Quantiles, 12},
    {"_BGSL_Compute_PosteriorMeans", (DL_FUNC) &_BGSL_Compute_PosteriorMeans, 10},
    {"_BGSL_Extract_Chain", (DL_FUNC) &_BGSL_Extract_Chain, 8},
    {"_BGSL_Summary_Graph", (DL_FUNC) &_BGSL_Summary_Graph, 4},
    {"_BGSL_SimulateData_GGM_c", (DL_FUNC) &_BGSL_SimulateData_GGM_c, 10},
    {"_BGSL_CreateGroups", (DL_FUNC) &_BGSL_CreateGroups, 2},
    {"_BGSL_GGM_sampling_c", (DL_FUNC) &_BGSL_GGM_sampling_c, 23},
    {"_BGSL_FLM_sampling_c", (DL_FUNC) &_BGSL_FLM_sampling_c, 23},
    {"_BGSL_FGM_sampling_c", (DL_FUNC) &_BGSL_FGM_sampling_c, 29},
    {NULL, NULL, 0}
};

RcppExport void R_init_BGSL(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
